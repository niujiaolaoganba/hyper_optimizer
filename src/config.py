# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: 配置文件路径，系统设置，参数等
@todo:


"""

import numpy as np
from  utils import os_utils

# ---------------------- Overall -----------------------
TASK = "all"
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = "../.."

DATA_DIR = "%s/Data"%ROOT_DIR
CLEAN_DATA_DIR = "%s/Clean"%DATA_DIR

FEAT_DIR = "%s/Feat"%ROOT_DIR
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = "./conf"

OUTPUT_DIR = "%s/Output"%ROOT_DIR
SUBM_DIR = "%s/Subm"%OUTPUT_DIR

LOG_DIR = "%s/Log"%ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR
TMP_DIR = "%s/Tmp"%ROOT_DIR


