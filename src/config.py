# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: 配置文件路径，系统设置，参数等
@todo:

"""

import os
# ------------------------ PATH ------------------------
ROOT_DIR = '../'
OUTPUT_DIR = "%s/data/output"%ROOT_DIR
LOG_DIR = "%s/log"%ROOT_DIR
MODEL_COMPARE = "%s/data/models_compare.csv"%ROOT_DIR

THREAD = 8

if not os.path.exists("%s/data" % ROOT_DIR):
    os.mkdir("%s/data" % ROOT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

