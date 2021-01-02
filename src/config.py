# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: config.py.py
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


# my packages
BASE_ROOT = os.path.dirname(__file__)

DATA_ROOT = '/Users/ericdoug/Documents/competitions/tianchi/news/datas'

MODEL_DATA_ROOT = os.path.join(BASE_ROOT, "..", "datas")
SUBMIT_ROOT = os.path.join(BASE_ROOT, "..", "submits")
