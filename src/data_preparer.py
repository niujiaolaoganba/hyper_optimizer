# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: 将原始数据生成data frame
@todo:

"""

import gc
import numpy as np
import pandas as pd
import config
from utils import pkl_utils

def main():
    # load provided data
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
    dfAttr = pd.read_csv(config.ATTR_DATA)
    dfDesc = pd.read_csv(config.DESC_DATA)

    # concat train and test
    dfAll = pd.concat((dfTrain, dfTest), ignore_index=True)
    del dfTrain
    del dfTest
    gc.collect()

    dfAll = pd.merge(dfAll, dfDesc, on="product_uid", how="left")
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfDesc
    gc.collect()

    # merge product brand
    dfBrand = dfAttr[dfAttr.name == "MFG Brand Name"][["product_uid", "value"]].rename(
        columns={"value": "product_brand"})
    dfAll = pd.merge(dfAll, dfBrand, on="product_uid", how="left")
    dfBrand["product_brand"] = dfBrand["product_brand"].values.astype(str)
    dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
    del dfBrand
    gc.collect()

    # save data
    if config.TASK == "sample":
        dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy()
    pkl_utils._save(config.ALL_DATA_RAW, dfAll)

    # info
    dfInfo = dfAll[["id", "relevance"]].copy()
    pkl_utils._save(config.INFO_DATA, dfInfo)


if __name__ == "__main__":
    main()