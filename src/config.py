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
THIRDPARTY_DIR = "%s/Thirdparty"%ROOT_DIR

# index split
SPLIT_DIR = "%s/split"%DATA_DIR

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = "%s/train.csv"%DATA_DIR
TEST_DATA = "%s/test.csv"%DATA_DIR
ATTR_DATA = "%s/attributes.csv"%DATA_DIR
DESC_DATA = "%s/product_descriptions.csv"%DATA_DIR
SAMPLE_DATA = "%s/sample_submission.csv"%DATA_DIR

ALL_DATA_RAW = "%s/all.raw.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED = "%s/all.lemmatized.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED_STEMMED = "%s/all.lemmatized.stemmed.csv.pkl"%CLEAN_DATA_DIR
INFO_DATA = "%s/info.csv.pkl"%CLEAN_DATA_DIR

# size
TRAIN_SIZE = 74067
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 166693
VALID_SIZE_MAX = 60000 # 0.7 * TRAIN_SIZE

TRAIN_MEAN = 2.381634
TRAIN_VAR = 0.285135


# ------------------------ PARAM ------------------------
# attribute name and value SEPARATOR
ATTR_SEPARATOR = " | "

# cv
N_RUNS = 5
N_FOLDS = 1

# xgboost
# mean of relevance in training set
BASE_SCORE = TRAIN_MEAN

# missing value
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

# ------------------------ OTHER ------------------------
RANDOM_SEED = 2016
NUM_CORES = 14

DATA_PROCESSOR_N_JOBS = 6

## rgf
RGF_CALL_EXE = "%s/rgf1.2/test/call_exe.pl"%THIRDPARTY_DIR
RGF_EXE = "%s/rgf1.2/bin/rgf%s"%(THIRDPARTY_DIR, RGF_EXTENSION)
