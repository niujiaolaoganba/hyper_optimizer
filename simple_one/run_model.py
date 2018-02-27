# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief: 构造 learner类，feature类，
        task类，optimizer类，
        及ensemble, stacking相关类

@usage demo:
完整版 python task.py -m single -f basic_nonlinear_%s -l reg_xgb_tree -e 100
简版 python run_model.py

@todo:
ModelParamSpace
Config(FEAT_DIR, OUTPUT_DIR, SUBM_DIR, FIG_DIR)
pkl_utils(load_data)
logger
dist_utiles(rmse)
为什么predict_proba需要用learner.learner
学习hyperopt包，重点fmin, tpe, hp, STATUS_OK, Tirals, space_eval
"""

import os
import time
# from optparse import OptionParser

import numpy as np
from sklearn.linear_model import Lasso, Ridge, BayesianRidge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx


# import config
from model_param_space import ModelParamSpace
# from utils.skl_utils import SVR, LinearSVR, KNNRegressor, AdaBoostRegressor, RandomRidge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        if self.learner_name == 'clf_skl_lr':
            return LogisticRegression(**self.param_dict)
        if self.learner_name == 'clf_skl_knn':
            return KNeighborsClassifier(**self.param_dict)
        if self.learner_name == 'clf_skl_etr':
            return ExtraTreesClassifier(**self.param_dict)
        if self.learner_name == 'clf_skl_rf':
            return RandomForestClassifier(**self.param_dict)
        if self.learner_name == 'clf_skl_gbm':
            return GradientBoostingClassifier(**self.param_dict)
        if self.learner_name == 'clf_skl_adaboost':
            return AdaBoostClassifier(**self.param_dict)
        if self.learner_name == 'clf_skl_svc':
            return SVC(**self.param_dict)

        if self.learner_name == 'reg_skl_lasso':
            return Lasso(**self.param_dict)
        if self.learner_name == 'reg_skl_ridge':
            return Ridge(**self.param_dict)
        if self.learner_name == 'reg_skl_bayesian_ridge':
            return BayesianRidge(**self.param_dict)
        if self.learner_name == 'reg_skl_svr':
            return SVR(**self.param_dict)
        if self.learner_name == 'reg_skl_lsvr':
            return LinearSVR(**self.param_dict)
        if self.learner_name == 'reg_skl_knn':
            return KNNRegressor(**self.param_dict)
        if self.learner_name == 'reg_skl_etr':
            return ExtraTreesRegressor(**self.param_dict)
        if self.learner_name == 'reg_skl_rf':
            return RandomForestRegressor(**self.param_dict)
        if self.learner_name == 'reg_skl_gbm':
            return GradientBoostingRegressor(**self.param_dict)
        if self.learner_name == 'reg_skl_adaboost':
            return AdaBoostRegressor(**self.param_dict)
        return None

    def fit(self, X, y, feature_names = None):
        if feature_names is not None:
            self.learner.fit(X, y, feature_names)
        else:
            self.learner.fit(X, y)

        return self

    def predict(self, X, feature_names = None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, feature_names)
        else:
            y_pred = self.learner.predict(X)

        return y_pred


class TaskOptimizer:
    def __init__(self, leaner_name, X, y, max_evals = 100):
        self.leaner_name = leaner_name
        self.X = X
        self.y = y
        self.max_evals = max_evals
        self.model_param_space = ModelParamSpace(self.leaner_name)

    def _obj(self, param_dict):
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.leaner_name, param_dict)
        X_train, X_valid, y_train, y_valid = train_test_split(self.X,self.y,stratify=self.y,
                                                      test_size = 0.2,
                                                     random_state = 2018)

        learner.fit(X_train, y_train)
        y_pred = learner.predict(X_valid)
        loss = - roc_auc_score(y_valid, y_pred)

        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=2018)
        # auc = []
        # for train, test in cv.split(X, y):
        #     learner.fit(X.iloc[train, :], y[train])
        #     y_pred = learner.predict(X.iloc[test, :])
        #     auc.append(roc_auc_score(y[test], y_pred))
        # loss = - np.mean(auc)

        ret = {
            "loss": loss,
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        # trials = Trials()
        trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
        best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        trial_loss = np.asarray(trials.losses(), dtype = float)
        best_loss = trial_loss[np.argmin(trial_loss)]
        return trials


if __name__ == "__main__":
    learner_name = 'clf_skl_gbm'
    data = pd.read_csv('./data/model_data.csv')
    X = data.tags
    y = data.is_reg
    tag_vector = CountVectorizer(tokenizer=lambda x: x.split(','))
    features = tag_vector.fit(X).transform(X).toarray()
    X = pd.DataFrame(features, columns=tag_vector.get_feature_names())
    optimizer = TaskOptimizer(learner_name, X, y, max_evals = 500)
    trials = optimizer.run()
    trial_loss = np.asarray(trials.losses(), dtype=float)
    print(trials.trials[np.argmin(trial_loss)]['misc']['vals'])
    print(trials.trials[np.argmin(trial_loss)]['result'])

