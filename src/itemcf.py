# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: itemcf.py
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
import pickle

# third packages


# my packages
from datasets import NewDataSet
from config import MODEL_DATA_ROOT


def itemcf_sim(click_data_file):

    news_datasets = NewDataSet(click_data_file)
    click_df = news_datasets.all_click
    # 获取用户的点击文章字典
    user_item_click_dict = news_datasets.get_user_item_time(click_df)

    # 计算item相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_click_dict.items()):
        for item_i, itemi_click_time in item_time_list:
            item_cnt[item_i] += 1
            i2i_sim.setdefault(item_i, {})
            for item_j, itemj_click_time in item_time_list:
                if item_i == item_j:
                    continue
                i2i_sim[item_i].setdefault(item_j, 0)
                i2i_sim[item_i][item_j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])


    i2i_sim_file = os.path.join(MODEL_DATA_ROOT, 'itemcf_i2i_sim.pkl')
    if not os.path.exists(i2i_sim_file):
        pickle.dump(i2i_sim_, open(i2i_sim_file, 'wb'))

    return i2i_sim_

def item_based_recomm(user_id, user_item_time_dict, i22_sim, sim_item_topk, recall_item_num, item_topk_click):
    """基于文章协同过滤召回

    :param user_id: 用户id
    :param user_item_time_dict: 根据点击时间获取用户的点击文章序列
    :param i22_sim:  文章相似度矩阵
    :param sim_item_topk:  选择与当前文章最相似的前k篇文章
    :param recall_item_num:  最后召回文章数量
    :param item_opk_click:  点击次数最多的文章列表， 用户召回补全
    :return:  召回文章列表  {article_id: score}
    """

    # 获取用户历史交互的文章
    user_history_items = user_item_time_dict[user_id]
    user_history_items_ = {user_id for user_id, _ in user_history_items}

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_history_items):
        for j, wij in sorted(i22_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_history_items_:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():
                continue
            item_rank[item] = -i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

if __name__ == '__main__':
    click_data_file = os.path.join(DATA_ROOT, 'train_click_log.csv')
    itemcf_sim(click_data_file)





