# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: utils.py
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
from datetime import datetime

# third packages

# my packages
from config import SUBMIT_ROOT

def submit(recall_df, topk=5, model_name=None):
    """生成提交文件

    :param recall_df:
    :param topk:
    :param model_name:
    :return:
    """
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})
    if model_name:
        save_name = os.path.join(SUBMIT_ROOT,  model_name + '_' + datetime.today().strftime('%m-%d') + '.csv')
    else:
        save_name = os.path.join(SUBMIT_ROOT,   'submit_' + datetime.today().strftime('%m-%d') + '.csv')
    submit.to_csv(save_name, index=False, header=True)