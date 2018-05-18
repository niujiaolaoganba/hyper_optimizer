# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief: 构造 learner类，ensemblelearner类，
        task类，optimizer类
"""

import os
import time
import datetime
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, rand, Trials, space_eval
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss

import config
from model_param_space import ModelParamSpace
from utils import logging_utils, time_utils
from utils.ensemble_learner import EnsembleLearner

learner_space = {
    "single": ["clf_skl_lr", "clf_xgb_tree", "clf_skl_rf","clf_lgb_tree", "clf_cbst_tree", "ensemble",],
    "stacking": ["ensemble", ],
}

learner_name_space = {'clf_skl_lr': LogisticRegression,
                      'clf_xgb_tree': XGBClassifier,
                      'clf_skl_rf': RandomForestClassifier,
                      'ensemble': EnsembleLearner,
                      'clf_lgb_tree': LGBMClassifier,
                      'clf_cbst_tree': CatBoostClassifier}

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
        return self.learner.predict_proba(X)[:, 1]


class Task:
    def __init__(self, learner, X_train, y_train, X_test, y_test, n_iter,
                 prefix, suffix, logger, verbose):
        self.learner = learner
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_iter = n_iter
        self.prefix = prefix
        self.suffix = suffix
        self.logger = logger
        self.verbose = verbose
        self.test_auc = 0
        self.train_auc = 0
        self.train_logloss = 0
        self.auc_cv_mean = 0

    def __str__(self):
        return "%s_Learner@%s%s" % (str(self.prefix), str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="     ", incr_prefix="    "):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info('%s%s:' % (prefix, k))
                self._print_param_dict(v, prefix + incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix, k, v))

    def cv(self):
        start = time.time()
        if self.verbose:
            self.logger.info("=" * 50)
            self.logger.info("Task")
            self.logger.info("     %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(self.learner.param_dict)
            self.logger.info("Result")
            self.logger.info("     Run     AUC     Shape")

        auc_cv = np.zeros(self.n_iter)
        stacking_feature_train = np.zeros(self.X_train.shape[0])
        stacking_feature_test = np.zeros(self.X_test.shape[0])
        shuffle = KFold(n_splits=self.n_iter, random_state=42)
        i = 0
        for train_index, valid_index in shuffle.split(self.X_train, self.y_train):
            i += 1
            X_train_cv, y_train_cv, X_valid_cv, y_valid_cv = self.X_train.iloc[train_index, :], self.y_train[
                train_index], self.X_train.iloc[valid_index, :], self.y_train[valid_index]
            self.learner.fit(X_train_cv, y_train_cv)
            y_pred = self.learner.predict_proba(X_valid_cv)
            y_pred_test = self.learner.predict_proba(self.X_test)
            auc_cv[i - 1] = roc_auc_score(y_valid_cv, y_pred)
            stacking_feature_train[valid_index] = y_pred
            stacking_feature_test += y_pred_test

            # log
            self.logger.info("     {:>3}     {:>8}     {} x {}".format(
                i, np.round(auc_cv[i - 1], 6), X_train_cv.shape[0], X_train_cv.shape[1]
            ))

        stacking_feature_test /= self.n_iter
        self.auc_cv_mean = np.mean(auc_cv)
        if self.y_test is not None:
            auc_cv_test = roc_auc_score(self.y_test, stacking_feature_test)
        else:
            auc_cv_test = 0
        end = time.time()
        time_cost = time_utils.time_diff(start, end)

        # save
        train_fname = "%s/train_%s_%s.csv" % (
        config.OUTPUT_DIR, self.__str__(), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        test_fname = "%s/test_%s_%s.csv" % (
        config.OUTPUT_DIR, self.__str__(), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        pd.DataFrame({"%s" % self.__str__(): stacking_feature_train}).to_csv(train_fname, index=False)
        pd.DataFrame({"%s" % self.__str__(): stacking_feature_test}).to_csv(test_fname, index=False)

        if self.verbose:
            self.logger.info("AUC")
            self.logger.info("     cv_mean: %.6f" % self.auc_cv_mean)
            self.logger.info("     cv_test: %.6f" % auc_cv_test)
            self.logger.info("Time")
            self.logger.info("     %s" % time_cost)
            self.logger.info("-" * 50)

        return self

    def refit(self):
        start = time.time()
        self.learner.fit(self.X_train, self.y_train)
        y_pred_train = self.learner.predict_proba(self.X_train)
        y_pred_test = self.learner.predict_proba(self.X_test)
        self.train_auc = roc_auc_score(self.y_train, y_pred_train)
        if self.y_test is not None:
            self.test_auc = roc_auc_score(self.y_test, y_pred_test)
        else:
            self.test_auc = 0
        self.train_logloss = log_loss(self.y_train, y_pred_train)
        end = time.time()
        self.refit_time = time_utils.time_diff(start, end)

        fname = "%s/refit_test_%s_%s_[auc%.6f].csv" % (
        config.OUTPUT_DIR, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), self.__str__(), self.test_auc)
        pd.DataFrame({"prediction": y_pred_test}).to_csv(fname, index=False)

        return self

    def go(self):
        self.cv()
        self.refit()
        return self


class TaskOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, cv=5, max_evals=2, verbose=True):
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
        param_dict = self.param_space._convert_int_param(param_dict)
        if self.leaner_name == 'ensemble':
            learner = EnsembleLearner(param_dict)
        else :
            learner = Learner(self.leaner_name, param_dict)
        suffix = "_Id@%s" % str(self.trial_counter)
        prefix = "%s" % task_mode
        if task_mode == 'single':
            self.task = Task(learner, self.X_train, self.y_train, self.X_test, self.y_test, self.n_iter,
                             prefix, suffix, self.logger, self.verbose)
        elif task_mode == "stacking":
            train_fnames = glob.iglob("%s/train_single*.csv" % config.OUTPUT_DIR)
            test_fnames = glob.iglob("%s/test_single*.csv" % config.OUTPUT_DIR)
            stacking_level1_train = pd.concat([pd.read_csv(f) for f in train_fnames], axis=1)
            stacking_level1_test = pd.concat([pd.read_csv(f) for f in test_fnames], axis = 1)
            stacking_level1_test = stacking_level1_test[stacking_level1_train.columns]
            self.task = Task(learner, stacking_level1_train, self.y_train, stacking_level1_test, self.y_test,
                             self.n_iter,prefix, suffix, self.logger, self.verbose)
        self.task.go()
        result = {
            "loss": -self.task.auc_cv_mean,
            "attachments": {
                "train_auc": self.task.train_auc,
                "test_auc": self.task.test_auc,
                "refit_time": self.task.refit_time,
            },
            "status": STATUS_OK,
        }
        return result

    def run(self):
        line_index = 1
        self.param_space = ModelParamSpace()
        for task_mode in ("single","stacking",):
            if task_mode not in learner_space:
                print('%s model missed' % task_mode)
                continue
            print('start %s model task' % task_mode)
            for learner in learner_space[task_mode]:
                print('optimizing %s' % learner)
                self.leaner_name = learner
                start = time.time()
                trials = Trials()
                logname = "%s_%s_%s.log" % (task_mode, learner, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
                self.logger = logging_utils._get_logger(config.LOG_DIR, logname)
                best = fmin(lambda param: self._obj(param, task_mode), self.param_space._build_space(learner),
                            tpe.suggest, self.max_evals, trials)

                end = time.time()
                time_cost = time_utils.time_diff(start, end)
                self.logger.info("Hyperopt_Time")
                self.logger.info("     %s" % time_cost)
                self.logger.info("-" * 50)
                print("   Finished %d hyper train with %d-fold cv, took %s" % (self.max_evals, self.n_iter, time_cost))

                best_params = space_eval(self.param_space._build_space(learner), best)
                best_params = self.param_space._convert_int_param(best_params)
                trial_loss = np.asarray(trials.losses(), dtype=float)
                best_ind = np.argmin(trial_loss)
                auc_cv_mean = - trial_loss[best_ind]
                test_auc = trials.trial_attachments(trials.trials[best_ind])["test_auc"]
                refit_time = trials.trial_attachments(trials.trials[best_ind])["refit_time"]

                with open(config.MODEL_COMPARE, 'a+') as f:
                    if line_index:
                        line_index = 0
                        f.writelines("task_mode   learner   auc_cv_mean   test_auc   refit_time   best_params \n")
                    f.writelines("%s   %s   %.4f   %.4f   %s   %s \n" % (
                    task_mode, learner, auc_cv_mean, test_auc, refit_time, best_params))
                f.close()
