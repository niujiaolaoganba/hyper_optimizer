# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief:
定义模型超参空间，模型name和空间的字典
定义集成学习bagging的模型、参数空间、权重
把参数转成整数的类
"""

import numpy as np
from hyperopt import hp
from utils.ensemble_learner import param_space_ensemble


param_space_clf_xgb_tree = {
    # "booster": "gbtree",
    "objective": "binary:logistic",
    "base_score": 0.5,
    "n_estimators" : hp.quniform("n_estimators", 100, 1000, 10),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "grow_policy": hp.choice("grow_policy", ['depthwise', 'lossguide']),
    "gamma": hp.loguniform("gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-10), np.log(1e2)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1, 0.05),
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": 16,
    "seed": 42,
}

param_space_clf_skl_lr = {
    "C": hp.loguniform("C", np.log(1e-10), np.log(1e10)),
    "penalty": hp.choice("penalty", ['l1', 'l2']),
    "random_state": 42,
}

param_space_clf_skl_knn = {
    "normalize": hp.choice("normalize", [True, False]),
    "n_neighbors": hp.quniform("n_neighbors", 1, 20, 1),
    "weights": hp.choice("weights", ["uniform", "distance"]),
    "leaf_size": hp.quniform("leaf_size", 10, 100, 10),
    "metric": hp.choice("metric", ["cosine", "minkowski"][1:]),
    "random_state": 42,
}

param_space_clf_skl_etr = {
    "n_estimators": hp.quniform("n_estimators", 10, 1000, 10),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("min_samples_split", 2, 15, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "random_state": 42,
    "n_jobs": 8,
    "verbose": 0,
}

param_space_clf_skl_rf = {
    "n_estimators": hp.quniform("n_estimators", 10, 1000, 10),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "random_state": 42,
    "n_jobs": 8,
    "verbose": 0,
}

param_space_clf_skl_gbm = {
    "n_estimators": hp.quniform("n_estimators", 10, 1000, 10),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 15, 1),
    "random_state": 42,
    "verbose": 0,
}

param_space_clf_skl_adaboost = {
    "base_estimator": hp.choice("base_estimator", ["dtr", "etr"]),
    "n_estimators": hp.quniform("n_estimators", 10, 1000, 10),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    "random_state": 42,
}

param_space_dict = {
    "clf_skl_lr": param_space_clf_skl_lr,
    "clf_xgb_tree": param_space_clf_xgb_tree,
    "clf_skl_knn": param_space_clf_skl_knn,
    # "clf_skl_etr": param_space_clf_skl_etr,
    # "clf_skl_rf": param_space_clf_skl_rf,
    "clf_skl_gbm": param_space_clf_skl_gbm,
    "clf_skl_adaboost": param_space_clf_skl_adaboost,
    "ensemble":param_space_ensemble
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)



class ModelParamSpace:
    def __init__(self):
        pass

    def _build_space(self, learner_name):
        return param_space_dict[learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict

