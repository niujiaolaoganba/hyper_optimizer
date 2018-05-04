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
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTeesRegressor, RandomForestRegressor, GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Tirals, space_eval


import config
from model_param_space import ModelParamSpace
from utils.skl_utils import SVR, LinearSVR, KNNRegressor, AdaBoostRegressor, RandomRidge
from utils import logging_utils

learner_space = {
    "single": ["reg_xgb_tree","reg_skl_lasso","reg_skl_gbm","reg_ensemble"],
    "stacking": ["ensemble",],
}


class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
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
            return ExtraTeesRegressor(**self.param_dict)
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

class EnsembleLearner:
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            p = self.learner_dict[learner_name]["param"]
            l = Learner(learner_name, p)._get_learner()
            if l is not None:
                self.learner_dict[learner_name]["learner"] = l.fit(X, y)
            else:
                self.learner_dict[learner_name]["learner"] = None
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            if l is not None:
                w = self.learner_dict[learner_name]["weight"]
                y_pred += w * l.predict(X)
                w_sum += w
        y_pred /= w_sum
        return y_pred


class Task:
    def __init__(self, X_train, y_train, kfold, X_test, y_test, learner, suffix, logger, verbose=True, plot_importance=False):
        self.learner = learner
        self.suffix = suffix
        self.logger = logger
        self.verbose = verbose
        self.plot_importance = plot_importance
        self.n_iter = self.feature.n_iter
        self.rmse_cv_mean = 0
        self.rmse_cv_std = 0

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
            self.logger.info("     Run     RMSE     Shape")

        rmse_cv = np.zeros(self.n_iter)
        for i in range(self.n_iter):
            X_train, y_train, X_valid, y_valid = self.feature._get_train_valid_data(i)
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_valid)
            rmse_cv[i] = dist_utils._rmse(y_valid, y_pred)
            # log
            self.logger.info("     {:>3}     {:>8}     {} x {}".format(
                i+1, np.round(rmse_cv[i],6), X_train.shape[0], X_train.shape[1]
            ))
            # save
            fname = "%s/Run%d/valid.pred.%s.csv"%(config.OUTPUT_DIR, i+1, self.__str__())
            df = pd.DataFrame({"target": y_valid,
                               "prediction": y_pred})
            df.to_csv(fname, index = False, columns = ["target", "prediction"])
            if hasattr(self.learner.learner, "predict_proba"):
                y_proba = self.learner.learner.predict_proba(X_valid)
                fname = "%s/Run%d/valid.proba.%s.csv" % (config.OUTPUT_DIR, i+1, self.__str__())
                columns = ["proba%d"%i for i in range(y_proba.shape[1])]
                df = pd.DataFrame(y_proba, columns = columns)
                df["target"] = y_valid
                dt.to_csv(fname, index = False)

        self.rmse_cv_mean = np.mean(rmse_cv)
        self.rmse_cv_std = np.std(rmse_cv)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("RMSE")
            self.logger.info("     Mean: %.6f" % self.rmse_cv_mean)
            self.logger.info("     Std: %.6f" % self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("     %d mins" % _min)
            else:
                self.logger.info("     %d secs" % _sec)
            self.logger.info("-"*50)

        return self

    def refit(self):
        X_train, y_train, X_test = self.feature._get_train_test_data()
        if self.plot_importance:
            feature_names = self.feature._get_feature_names()
            self.learner.fit(X_train, y_train, feature_names)
            y_pred = self.learner.predict(X_test, feature_names)
        else:
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_test)

        id_test = self.feature.data_dict["id_test"].astype(int)

        fname = "%s/%s/test.pred.%s.csv" % (config.OUTPUT_DIR, "All", self.__str__())
        pd.DataFrame({"id":id_test,
                      "prediction": y_pred}).to_csv(fname, index = False)
        if hasattr(self.learner.learner, "predict_proba"):
            if self.plot_importance:
                feature_names = self.feature._get_feature_names()
                y_proba = self.learner.learner.predict_proba(X_test, feature_names)
            else:
                y_proba = self.learner.learner.predict_proba(X_test)
            fname = "%s/%s/test.proba.%s.csv" % (config.OUTPUT_DIR, "All", self.__str__())
            columns = ["proba%d" % i for i in range(y_proba.shape[1])]
            df = pd.DataFrame(y_proba, columns)
            df["id"] = id_test
            df.to_csv(fname, index = False)

        fname = "%s/test.pred.%s.[Mean%.6f]_[Std%.6f].csv" % (
            config.SUBM_DIR, self__str__(), self.rmse_cv_mean, self.rmse_cv_std
        )
        pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(fname, index = False)

        if self.plot_importance:
            ax = self.learner.plot_importance()
            ax.figure.savefig("%s/%s.pdf" % config.FIG_DIR, self.__str__())

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
        self.n_iter = cv,
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
            stacking_level1_train =
            stacking_level1_test =
            self.task = Task(learner, stacking_level1_train, self.y_train, stacking_level1_test, self.y_test, self.n_iter,
                             suffix, self.logger, self.verbose)
        self.task.go()
        result = {
            "loss": -self.task.test_auc,
            "attachments": {
                "train_auc": self.task.train_auc,
                "train_logloss": self.task.train_logloss,
                "train_time": self.task.train_time,
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




