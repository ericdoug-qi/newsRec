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

# third packages
import pandas as pd
import numpy as np


# my packages

class NewDataSet(object):

    def __init__(self, click_data_file):
        self._all_click = self.get_click_data(click_data_file)

    @property
    def all_click(self):
        return self._all_click

    @all_click.setter
    def all_click(self, all_click):
        self._all_click = all_click

    def get_click_data(self, click_data_file, samples=10000):
        """获取点击数据

        :param click_data_file:
        :param samples:
        :return:
        """
        all_click = pd.read_csv(click_data_file)
        all_user_ids = all_click.user_id.unique()

        if samples:
            sample_user_ids = np.random.choice(all_user_ids, size=samples, replace=False)
            all_click = all_click[all_click['user_id']].isin(sample_user_ids)

        all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

        return all_click

    def make_item_time_pair(self, df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    def get_user_item_time(self, click_df):
        click_df = click_df.sort_values('click_timestamp')
        user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
            lambda x: self.make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
        user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

        return user_item_time_dict

    def get_item_topk_click(self, click_df, k=5):
        """获取点击最多的top k文章

        :param click_df:
        :param k:
        :return:
        """
        topk_click = click_df['click_article_id'].value_counts().index[:k]

        return topk_click