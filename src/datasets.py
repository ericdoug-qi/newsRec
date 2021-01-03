# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: datasets.py
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
import pickle

# third packages
import pandas as pd
import numpy as np

# my packages
from config import MODEL_DATA_ROOT


class NewDataSet(object):

    def __init__(self, data_dir, samples=None, offline=True):
        if samples is not None:
            self._all_click = self.get_click_sample(data_dir, samples=samples)
        else:
            self._all_click = self.get_all_clicks(data_dir, offline=offline)

        self._user_click_items = self.get_user_item_time(self._all_click)

    @property
    def all_click(self):
        return self._all_click

    @all_click.setter
    def all_click(self, all_click):
        self._all_click = all_click

    @property
    def user_click_items(self):
        return self._user_click_items

    def get_article_info(self, article_dir):
        """获取文章基本属性
        
        :param article_dir: 
        :return: 
        """
        article_info = pd.read_csv(os.path.join(article_dir, 'articles.csv'))
        article_info = article_info.rename(columns={'article_id': 'click_article_id'})
        return article_info

    def get_article_embedding(self, article_emb_dir):
        """获取文章embedding数据

        :param article_emb_dir:
        :return:
        """
        article_emb_df = pd.read_csv(os.path.join(article_emb_dir, 'articles_emb.csv'))

        article_emb_cols = [x for x in article_emb_df.columns if 'emb' in x]
        article_emb_np = np.ascontiguousarray(article_emb_df[article_emb_cols])

        # 进行归一化
        article_emb_np = article_emb_np / np.linalg.norm(article_emb_np, axis=1, keepdims=True)

        article_emb_dict = dict(zip(article_emb_df['article_id'], article_emb_np))

        article_emb_file = os.path.join(MODEL_DATA_ROOT, 'article_content_emb.pkl')
        if not os.path.exists(article_emb_file):
            pickle.dump(article_emb_dict, open(article_emb_file, 'wb'))

        return article_emb_dict

    def get_article_click_by_user(self):

        click_df = self._all_click.copy().sort_values('click_timestamp')

        article_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(
            lambda x: self.make_user_time_pair(x)).reset_index().rename(columns={0: 'user_time_list'})

        article_user_time_dict = dict(
            zip(article_user_time_df['click_article_id'], article_user_time_df['user_time_list']))
        return article_user_time_dict

    def get_all_clicks(self, click_dir, offline=True):
        """获取所有点击数据
        
        :param click_dir: 
        :param offline: 
        :return: 
        """
        if offline:
            all_click = pd.read_csv(os.path.join(click_dir, 'train_click_log.csv'))
        else:
            train_click = pd.read_csv(os.path.join(click_dir, 'train_click_log.csv'))
            testa_click = pd.read_csv(os.path.join(click_dir, 'testA_click_log.csv'))

            all_click = train_click.append(testa_click)

        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
        return all_click

    def get_click_sample(self, click_dir, samples=10000):
        """获取点击数据

        :param click_data_file:
        :param samples:
        :return:
        """
        all_click = pd.read_csv(os.path.join(click_dir, 'train_click_log.csv'))
        all_user_ids = all_click.user_id.unique()

        if samples:
            sample_user_ids = np.random.choice(all_user_ids, size=samples, replace=False)
            all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

        return all_click

    def make_item_time_pair(self, df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    def make_user_time_pair(self, df):
        return list(zip(df['user_id'], df['click_timestamp']))

    def get_user_item_time(self):
        click_df = self._all_click.copy().sort_values('click_timestamp')
        user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
            lambda x: self.make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
        user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

        return user_item_time_dict

    def hist_func(self, user_df):
        """

        :param user_df:
        :return:
        """
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    def get_hist_and_last_click(self):
        """获取当前数据的历史点击和最后一次点击

        :return:
        """
        all_click = self._all_click.copy().sort_values(by=['user_id', 'click_timestamp'])
        click_last_df = all_click.groupby('user_id').tail(1)

        click_hist_df = all_click.groupby('user_id').apply(self.hist_func).reset_index(drop=True)

        return click_hist_df, click_last_df

    def get_article_info_dict(self, article_info_df):
        """获取文章id对应的基本属性，保存成字典的形式，方便召回阶段，冷启动阶段直接使用

        :param article_info_df:
        :return:
        """
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        article_info_df['created_at_ts'] = article_info_df[['created_at_ts']].apply(max_min_scaler)

        article_type_dict = dict(zip(article_info_df['click_article_id'], article_info_df['category_id']))
        article_words_dict = dict(zip(article_info_df['click_article_id'], article_info_df['words_count']))
        article_created_time_dict = dict(zip(article_info_df['click_article_id'], article_info_df['created_at_ts']))

        return article_type_dict, article_words_dict, article_created_time_dict

    def get_user_hist_article_info_dict(self):

        all_click = self._all_click.copy()

        # 获取user_id对应的用户历史点击文章类型的集合字典
        user_hist_item_types = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
        user_hist_item_types_dict = dict(zip(user_hist_item_types['user_id'], user_hist_item_types['category_id']))

        # 获取user_id对应的用户点击文章的集合
        user_hist_article_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
        user_hist_article_ids_dict = dict(
            zip(user_hist_article_ids_dict['user_id'], user_hist_article_ids_dict['click_article_id']))

        # 获取user_id对应的用户历史点击的文章的平均字数字典
        user_hist_article_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
        user_hist_item_words_dict = dict(zip(user_hist_article_words['user_id'], user_hist_article_words['words_count']))

        # 获取user_id对应的用户最后一次点击的文章的创建时间
        all_click_ = all_click.sort_values('click_timestamp')
        user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(
            lambda x: x.iloc[-1]).reset_index()

        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(
            max_min_scaler)

        user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                    user_last_item_created_time['created_at_ts']))

        return user_hist_item_types_dict, user_hist_article_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


    def get_item_topk_click(self, k):
        """获取近期点击最多的文章

        :param k:
        :return:
        """
        topk_click = self._all_click.copy()['click_article_id'].value_counts().index[:k]

        return topk_click


    def get_item_topk_click(self, click_df, k=5):
        """获取点击最多的top k文章

        :param click_df:
        :param k:
        :return:
        """
        topk_click = click_df['click_article_id'].value_counts().index[:k]

        return topk_click
