# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief: 构造 learner类，ensemblelearner类，
        task类，optimizer类
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, rand, Trials
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


import config
from model_param_space import ModelParamSpace
from utils import logging_utils, time_utils
from utils.ensemble_learner import EnsembleLearner

learner_space = {
    "single": ["reg_xgb_tree","reg_skl_lasso","reg_skl_gbm","reg_ensemble"],
    "stacking": ["ensemble",],
}

learner_name_space = {
"clf_skl_lr": LogisticRegression,
"clf_xgb_tree": XGBClassifier,
"clf_skl_rf": RandomForestClassifier,
"ensemble": EnsembleLearner,
}

class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = learner_name_space[learner_name](**param_dict)

    def __str__(self):
        return self.learner_name

    def fit(self, X, y):
        return self.learner.fit(X, y)

    def predict(self, X):
        return self.learner.predict(X)

    def predict_proba(self, X):
        return self.learner.predict_proba(X)[:,1]


class Task:
    def __init__(self, learner, X_train, y_train, X_test, y_test, n_iter,
                             suffix, logger, verbose):
        self.learner = learner
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_iter = n_iter
        self.suffix = suffix
        self.verbose = verbose
        self.test_auc = 0
        self.train_auc = 0
        self.train_logloss = 0

    def __str__(self):
        return "[Learner@%s]%s" % (str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="     ", incr_prefix = "    "):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info('%s%s:' % (prefix, k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix, k, v))

    def cv(self):
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("     %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(self.learner.param_dict)
            self.learner.info("Result")
            self.logger.info("     Run     AUC     Shape")

        auc_cv = np.zeros(self.n_iter)
        stacking_feature_train = np.zeros(self.X_train.shape[0])
        stacking_feature_test = np.zeros(self.X_test.shape[0])
        shuffle = KFold(n_splits = self.n_iter, random_state=42)
        i = 0
        for train_index, valid_index in shuffle.split(self.X_train, self.y_train):
            i += 1
            X_train_cv, y_train_cv, X_valid_cv, y_valid_cv = self.X_train.iloc[train_index,:], self.y_train[train_index], self.X_train.iloc[valid_index,:], self.y_train[valid_index]
            self.learner.fit(X_train_cv, y_train_cv)
            y_pred = self.learner.predict_proba(X_valid_cv)
            y_pred_test = self.learner.predict_proba(self.X_test)
            auc_cv[i] = roc_auc_score(y_valid_cv, y_pred)
            stacking_feature_train[valid_index] = y_pred
            stacking_feature_test += y_pred_test

            # log
            self.logger.info("     {:>3}     {:>8}     {} x {}".format(
                i+1, np.round(auc_cv[i],6), X_train_cv.shape[0], X_train_cv.shape[1]
            ))

        stacking_feature_test /= self.n_iter
        auc_cv_mean = np.mean(auc_cv)
        auc_cv_test = roc_auc_score(stacking_feature_test, self.y_test)
        end = time.time()
        time_cost = time_utils.time_diff(start, end)

        # save
        train_fname = "%s/stacking_train_%s.csv" % (config.OUTPUT_DIR, self.__str__())
        test_fname = "%s/stacking_test_%s.csv" % (config.OUTPUT_DIR, self.__str__())
        pd.DataFrame({"%s"%self.__str__(): stacking_feature_train}).to_csv(train_fname, index=False)
        pd.DataFrame({"%s"%self.__str__(): stacking_feature_test}).to_csv(test_fname, index=False)

        if self.verbose:
            self.logger.info("AUC")
            self.logger.info("     cv_mean: %.6f" % auc_cv_mean)
            self.logger.info("     cv_test: %.6f" % auc_cv_test)
            self.logger.info("Time")
            self.logger.info("     %s" % time_cost)
            self.logger.info("-"*50)

        return self

    def refit(self):
        start = time.time()
        self.learner.fit(self.X_train, self.y_train)
        y_pred_train = self.learner.predict_proba(self.X_train)
        y_pred_test = self.learner.predict_proba(self.X_test)
        self.train_auc = roc_auc_score(y_pred_train, self.y_train)
        self.test_auc = roc_auc_score(y_pred_test, self.y_test)
        end = time.time()
        self.refit_time = time_utils.time_diff(start, end)

        fname = "%s/refit_test_%s_[auc%.6f].csv" % (config.OUTPUT_DIR, self.__str__(), self.test_auc)
        pd.DataFrame({"prediction": y_pred_test}).to_csv(fname, index = False)

        return self

    def go(self):
        self.cv()
        self.refit()
        return self

class TaskOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, cv = 5, max_evals = 2, verbose=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_iter = cv
        self.max_evals = max_evals
        self.verbose = verbose
        self.trial_counter = 0

    def _obj(self, param_dict, task_mode):
        self.trial_counter += 1
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.leaner_name, param_dict)
        suffix = "_[Id@%s]" % str(self.trial_counter)
        if task_mode == 'single':
            self.task = Task(learner, self.X_train, self.y_train, self.X_test, self.y_test, self.n_iter,
                             suffix, self.logger, self.verbose)
        elif task_mode == "stacking":
            fnames = os.listdir(" %s/stacking*.csv"%config.OUTPUT_DIR)
            dfs_train = []
            dfs_test = []
            for f in fnames:
                if 'train' in f:
                    dfs_train.append(pd.read_csv(f))
                else :
                    dfs_test.append(pd.read_csv(f))

            stacking_level1_train = pd.concat(dfs_train, ignore_index=True)
            stacking_level1_test = pd.concat(dfs_test, ignore_index=True)
            self.task = Task(learner, stacking_level1_train, self.y_train, stacking_level1_test, self.y_test, self.n_iter,
                             suffix, self.logger, self.verbose)
        self.task.go()
        result = {
            "loss": -self.task.test_auc,
            "attachments": {
                "train_auc": self.task.train_auc,
                "train_logloss": self.task.train_logloss,
                "refit_time": self.task.refit_time,
            },
            "status": STATUS_OK,
        }
        return result

    def run(self):
        line_index = 1
        for task_mode in ('single', 'stacking'):
            print('start %s model task') % task_mode
            for learner in learner_space[task_mode]:
                print('optimizing %s') % learner
                self.leaner_name = learner
                self.model_param_space = ModelParamSpace(self.leaner_name)
                start = time.time()
                trials = Trials()
                logname = "%s_%s_%s.log" % (task_mode,learner,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
                self.logger = logging_utils._get_logger(config.LOG_DIR, logname)
                best = fmin(lambda param: self._obj(param, task_mode), self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)

                end = time.time()
                time_cost = time_utils.time_diff(start, end)
                self.logger.info("Hyperopt_Time")
                self.logger.info("     %s" % time_cost)
                self.logger.info("-" * 50)

                best_params = space_eval(self.model_param_space._build_space(), best)
                best_params = self.model_param_space._convert_int_param(best_params)
                trial_loss = np.asarray(trials.loss(), dtype=float)
                best_ind = np.argmin(trial_loss)
                test_auc = - trial_loss[best_ind]
                train_auc = trials.trial_attachments(trials.trials[best_ind])["train_auc"]

                with open(config.MODEL_COMPARE, 'w+') as f:
                    if line_index:
                        line_index = 0
                        f.writelines("task_mode   learner   test_auc   train_auc   best_time   best_params \n")
                    f.writelines("%s   %s   %.4f   %.4f   %s   %s \n" % (task_mode, learner, test_auc, train_auc, best_time, best_params))




