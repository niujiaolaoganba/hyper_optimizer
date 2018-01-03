# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: 构造特征 基本特征
{docid, docidecho(product_uid), doconehot, doclen, docfreq, docentropy,
digitcount, digitratio,}
@todo:
nlp_utils,
"""
import re
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelBinarizer

import config
from utils import ngram_utils, nlp_utils, np_utils
from utils import time_utils, logging_utils, pkl_utils
from feature_base ijmport BaseEstimator, StandaloneFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens

class DocId(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        obs_set = set(obs_corpus)
        self.encoder = dict(zip(obs_set), range(len(obs_set)))

    def __name__(self):
        return "DocId"

    def transform_one(self, obs, target, id):
        return self.encoder[obs]

class DocIdEcho(BaseEstimator):
    """ for product_uid"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocIdEcho"

    def transform_one(self, obs, target, id):
        return obs

class DocIdOneHot(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocIdOneHot"

    def transform(self):
        lb = LabelBinarizer(sparse_output=True)
        return lb.fit_transform(self.obs_corpus)

class DocLen(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocLen"

    def transform_one(self, obs, target, id):
        obs_tokens =nlp_utils._tokenize(obs, token_pattern)
        return len(obs_tokens)

class DocFreq(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.counter = Counter(obs_corpus)

    def __name__(self):
        return "DocFreq"

    def transform_one(self, obs, target, id):
        return self.counter[obs]

class DocEntropy(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocEntropy"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        counter = Counter(obs_tokens)
        count = np.asarray(list(counter.values()))
        proba = count/np.sum(count)
        return np_utils._entropy(proba)

class DigitCount(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitCount"

    def transform_one(self, obs, target, id):
        return len(re.findall(r"\d", obs))

class DigitRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitRatio"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        return np_utils._try_divide(len(re.findall(r"\d", obs)), len(obs_tokens))

#---------------- Main ---------------------------
def main():
    logname = "generate_feature_basic_%s.log" % time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    ## basic
    generators = [DocId, DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
    obs_fields = ["search_term", "product_title", "product_description", "product_attribute", "product_brand", "product_color"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator,dfAll,obs_fields,param_list,config.FEAT_DIR, logger)
        sf.go()

    ## for product_uid
    generators = [DocIdEcho, DocFreq]
    obs_fields = ["product_uid"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields,param_list,config.FEAT_DIR, logger)
        sf.go()


if __name__ == "__main__":
    main()