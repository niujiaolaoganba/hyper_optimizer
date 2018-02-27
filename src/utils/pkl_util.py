# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief: pkl 存储和读写

@todo:

"""

import pickle

def _save(fname, data, protocol = 3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)