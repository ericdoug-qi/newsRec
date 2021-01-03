# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: multi_recall.py
   Description : 
   Author : ericdoug
   date：2021/1/2
-------------------------------------------------
   Change Activity:
         2021/1/2: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
import pickle
import faiss
import random

# third packages
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.keras.models import Model
import tensorflow as tf
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss


# my packages
from datasets import NewDataSet
from config import DATA_ROOT
from config import MODEL_DATA_ROOT
from utils import max_min_scaler


class MultiRecall(object):

    def __init__(self, data_dir, samples=None):
        self._new_datasets = NewDataSet(data_dir=data_dir, samples=samples)

        self._new_datasets.all_click['click_timestamp'] = self._new_datasets.all_click[['click_timestamp']].apply(
            max_min_scaler)

        self._item_emb_dict = self._new_datasets.get_article_embedding(data_dir)

        # 获取文章的属性信息，保存成字典的形式方便查询
        article_info_df = self._new_datasets.get_article_info(data_dir)
        self._item_type_dict, self._item_words_dict, self._item_created_time_dict = self._new_datasets.get_article_info_dict(
            article_info_df)

        # 定义多路召回的字典，将各路召回的结果都保存在这个字典中
        self._multi_recall_dict = {
            "itemcf_sim_itemcf_recall": {},
            "embedding_sim_item_recall": {},
            "youtubednn_recall": {},
            "youtubednn_usercf_recall": {},
            "cold_start_recall": {}
        }

        # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
        self._train_hist_click_df, self._train_last_click_df = self._new_datasets.get_hist_and_last_click()

    def metrics_recall(self, user_recall_items_dict, train_last_click_df, topk=5):
        """依次评估召回的前10, 20, 30, 40, 50个文章中的击中率

        :param user_recall_items_dict: 
        :param train_last_click_df:
        :param topk: 
        :return: 
        """
        last_click_item_dict = dict(zip(train_last_click_df['user_id'], train_last_click_df['click_article_id']))
        user_num = len(user_recall_items_dict)

        for k in range(10, topk + 1, 10):
            hit_num = 0
            for user, item_list in user_recall_items_dict.items():
                # 获取前k个召回的结果
                tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
                if last_click_item_dict[user] in set(tmp_recall_items):
                    hit_num += 1

            hit_rate = round(hit_num * 1.0 / user_num, 5)
            print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)

    def itemcf_sim(self, item_df, item_created_time_dict):
        """计算文章与文章之间的相似性矩阵
            1. 用户点击的时间权重
            2. 用户点击的顺序权重
            3. 文章创建的时间权重

        :param item_df:
        :param item_created_time_dict:
        :return:
        """
        user_item_time_dict = self._new_datasets.get_user_item_time(item_df)

        # 计算物品相似度
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user, item_time_list in tqdm(user_item_time_dict.items()):
            # 在基于商品的协同过滤优化的时候可以考虑时间因素
            for loc1, (i, i_click_time) in enumerate(item_time_list):
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})

                for loc2, (j, j_click_time) in enumerate(item_time_list):
                    if i == j:
                        continue

                    # 考虑文章的正向顺序点击和反向顺序点击
                    loc_alpha = 1.0 if loc2 > loc1 else 0.7
                    # 位置信息权重，其中的参数可以调节
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                    # 点击时间权重，其中的参数可以调节
                    click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                    # 两篇文章创建时间的权重，其中的参数可以调节
                    created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                    i2i_sim[i].setdefault(j, 0)
                    # 考虑多种因素的权重计算最终的文章之间的相似度
                    i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)

        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

        # 将得到的相似性矩阵保存到本地
        itemcf_i2i_sim_file = os.path.join(MODEL_DATA_ROOT, 'itemcf_i2i_sim.pkl')
        if not os.path.exists(itemcf_i2i_sim_file):
            pickle.dump(i2i_sim_, open(itemcf_i2i_sim_file, 'wb'))

        return i2i_sim_

    def get_user_activate_degree_dict(self, all_click_df):

        all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

        # 用户活跃度归一化
        mm = MinMaxScaler()
        all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
        user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

        return user_activate_degree_dict

    def usercf_sim(self, all_click_df, user_activate_degree_dict):
        """
            用户相似性矩阵计算
            :param all_click_df: 数据表
            :param user_activate_degree_dict: 用户活跃度的字典
            return 用户相似性矩阵

            思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
        """
        item_user_time_dict = self._new_datasets.get_article_click_by_user()

        u2u_sim = {}
        user_cnt = defaultdict(int)
        for item, user_time_list in tqdm(item_user_time_dict.items()):
            for u, click_time in user_time_list:
                user_cnt[u] += 1
                u2u_sim.setdefault(u, {})
                for v, click_time in user_time_list:
                    u2u_sim[u].setdefault(v, 0)
                    if u == v:
                        continue
                    # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                    activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                    u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

        u2u_sim_ = u2u_sim.copy()
        for u, related_users in u2u_sim.items():
            for v, wij in related_users.items():
                u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

        # 将得到的相似性矩阵保存到本地
        usercf_u2u_sim_file = os.path.join(MODEL_DATA_ROOT, 'usercf_u2u_sim.pkl')
        if not os.path.exists(usercf_u2u_sim_file):
            pickle.dump(u2u_sim_, open(usercf_u2u_sim_file, 'wb'))

        return u2u_sim_

    def embdding_sim(click_df, item_emb_df, save_path, topk):
        """向量检索相似度计算

            基于内容的文章embedding相似性矩阵计算
            :param click_df: 数据表
            :param item_emb_df: 文章的embedding
            :param save_path: 保存路径
            :patam topk: 找最相似的topk篇, 每个item, faiss搜索后返回最相似的topk个item
            return 文章相似性矩阵

            思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
        """

        # 文章索引与文章id的字典映射
        item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

        item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
        item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
        # 向量进行单位化
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

        # 建立faiss索引
        item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
        item_index.add(item_emb_np)
        # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
        sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

        # 将向量检索的结果保存成原始id的对应关系
        item_sim_dict = defaultdict(dict)
        for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
            target_raw_id = item_idx_2_rawid_dict[target_idx]
            # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                     0) + sim_value

        # 保存i2i相似度矩阵
        emb_i2i_sim_file = os.path.join(MODEL_DATA_ROOT, 'emb_i2i_sim.pkl')
        if not os.path.exists(emb_i2i_sim_file):
            pickle.dump(item_sim_dict, open(emb_i2i_sim_file, 'wb'))

        return item_sim_dict


    def gen_data_set(self, data, negsample=0):
        """获取双塔召回时的训练验证数据

        :param data:
        :param negsample: 通过滑窗构建样本的时候，负样本的数量
        :return:
        """
        data.sort_values("click_timestamp", inplace=True)
        item_ids = data['click_article_id'].unique()

        train_set = []
        test_set = []
        for reviewerID, hist in tqdm(data.groupby('user_id')):
            pos_list = hist['click_article_id'].tolist()

            if negsample > 0:
                candidate_set = list(set(item_ids) - set(pos_list))  # 用户没看过的文章里面选择负样本
                neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample,
                                            replace=True)  # 对于每个正样本，选择n个负样本

            # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
            if len(pos_list) == 1:
                train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
                test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))

            # 滑窗构造正负样本
            for i in range(1, len(pos_list)):
                hist = pos_list[:i]

                if i != len(pos_list) - 1:
                    train_set.append((reviewerID, hist[::-1], pos_list[i], 1,
                                      len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                    for negi in range(negsample):
                        train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0,
                                          len(hist[::-1])))  # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
                else:
                    # 将最长的那一个序列长度作为测试数据
                    test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))

        random.shuffle(train_set)
        random.shuffle(test_set)

        return train_set, test_set


    def gen_model_input(self, train_set, user_profile, seq_max_len):
        """将输入的数据进行padding，使得序列特征的长度都一致

        :param train_set:
        :param user_profile:
        :param seq_max_len:
        :return:
        """

        train_uid = np.array([line[0] for line in train_set])
        train_seq = [line[1] for line in train_set]
        train_iid = np.array([line[2] for line in train_set])
        train_label = np.array([line[3] for line in train_set])
        train_hist_len = np.array([line[4] for line in train_set])

        train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
        train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                             "hist_len": train_hist_len}

        return train_model_input, train_label


    def youtubednn_u2i_dict(self, data, topk=20):

        sparse_features = ["click_article_id", "user_id"]
        SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

        user_profile_ = data[["user_id"]].drop_duplicates('user_id')
        item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

        # 类别编码
        features = ["click_article_id", "user_id"]
        feature_max_idx = {}

        for feature in features:
            lbe = LabelEncoder()
            data[feature] = lbe.fit_transform(data[feature])
            feature_max_idx[feature] = data[feature].max() + 1

        # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
        user_profile = data[["user_id"]].drop_duplicates('user_id')
        item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

        user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
        item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

        # 划分训练和测试集
        # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
        train_set, test_set = self.gen_data_set(data, 0)
        # 整理输入数据，具体的操作可以看上面的函数
        train_model_input, train_label = self.gen_model_input(train_set, user_profile, SEQ_LEN)
        test_model_input, test_label = self.gen_model_input(test_set, user_profile, SEQ_LEN)

        # 确定Embedding的维度
        embedding_dim = 16

        # 将数据整理成模型可以直接输入的形式
        user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                                VarLenSparseFeat(
                                    SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                               embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
        item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]

        # 模型的定义
        # num_sampled: 负采样时的样本数量
        model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
                           user_dnn_hidden_units=(64, embedding_dim))
        # 模型编译
        model.compile(optimizer="adam", loss=sampledsoftmaxloss)

        # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练
        history = model.fit(train_model_input, train_label, batch_size=256, epochs=1, verbose=1, validation_split=0.0)

        # 训练完模型之后,提取训练的Embedding，包括user端和item端
        test_user_model_input = test_model_input
        all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}

        user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
        item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

        # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
        user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
        item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

        # embedding保存之前归一化一下
        user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
        item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

        # 将Embedding转换成字典的形式方便查询
        raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                    v for k, v in zip(user_profile['user_id'], user_embs)}
        raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                    v for k, v in zip(item_profile['click_article_id'], item_embs)}
        # 将Embedding保存到本地
        user_youtube_emb_file = os.path.join(MODEL_DATA_ROOT, 'user_youtube_emb.pkl')
        if not os.path.exists(user_youtube_emb_file):
            pickle.dump(raw_user_id_emb_dict, open(user_youtube_emb_file, 'wb'))
        item_youtube_emb_file = os.path.join(MODEL_DATA_ROOT, 'item_youtube_emb.pkl')
        if not os.path.exists(item_youtube_emb_file):
            pickle.dump(raw_item_id_emb_dict, open(item_youtube_emb_file, 'wb'))

        # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
        index = faiss.IndexFlatIP(embedding_dim)
        # 上面已经进行了归一化，这里可以不进行归一化了
        #     faiss.normalize_L2(user_embs)
        #     faiss.normalize_L2(item_embs)
        index.add(item_embs)  # 将item向量构建索引
        sim, idx = index.search(np.ascontiguousarray(user_embs), topk)  # 通过user去查询最相似的topk个item

        user_recall_items_dict = defaultdict(dict)
        for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
            target_raw_id = user_index_2_rawid[target_idx]
            # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_index_2_rawid[rele_idx]
                user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                         .get(rele_raw_id, 0) + sim_value

        user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                                  user_recall_items_dict.items()}
        # 将召回的结果进行排序

        # 保存召回的结果
        # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
        # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
        youtube_u2i_dict_file = os.path.join(MODEL_DATA_ROOT, 'youtube_u2i_dict.pkl')
        if not os.path.join(youtube_u2i_dict_file):
            pickle.dump(user_recall_items_dict, open(youtube_u2i_dict_file, 'wb'))
        return user_recall_items_dict




if __name__ == '__main__':
    multi_recall = MultiRecall(data_dir=DATA_ROOT, samples=1000)
