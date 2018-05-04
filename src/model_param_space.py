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

import config

#sklearn
skl_random_seed = config.RANDOM_SEED
skl_n_jobs = config.NUM_CORES
skl_n_estimators_min = 100
skl_n_estimators_max = 1000
skl_n_estimators_step = 10


## lasso
param_space_reg_skl_lasso = {
    "alpa": hp.loguniform("alpha", np.log(0.00001), np.log(0.1)),
    "normalize": hp.choice("normalize", [True, False]),
    "random_state": skl_random_seed,
}

## ridge_regression
param_space_reg_skl_ridge = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
    "normalize": hp.choice("normalize", [True, False]),
    "random_state": skl_random_seed
}

## Baysian Ridge regression
param_space_reg_skl_bayesian_ridge = {
    "alpha_1": hp.loguniform("alpha_1", np.log(1e-10), np.log(1e2)),
    "alpha_2": hp.loguniform("alpha_2", np.log(1e-10), np.log(1e2)),
    "lambda_1": hp.loguniform("alpha_1", np.log(1e-10), np.log(1e2)),
    "lambda_2": hp.loguniform("alpha_2", np.log(1e-10), np.log(1e2)),
    "normalize": hp.choice("normalize", [True, False]),
}

## random ridge regression
param_space_reg_skl_random_ridge ={
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20))
    "normalize": hp.choice("normalize", [True, False]),
    "poly": hp.choice("poly", [False]),
    "n_estimators": hp.quniform("n_estimators", 2, 50, 2),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "random_state": skl_random_seed
}

## linear support vector regression
param_space_reg_skl_lsvr = {
    "normalize": hp.choice("normalize", [True, False]),
    "C": hp.loguniform("C", np.log(1), np.log(100)),
    "epsion": hp.loguniform("epsion", np.log(0.001), np.log(0.1)),
    "loss": hp.choice("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
    "random_state": skl_random_seed,
}

## suport vector regression
param_space_reg_skl_svr = {
    "normalize": hp.choice("normalize", [True, False]),
    "C": hp.loguniform("C", np.log(1), np.log(100)),
    "gamma": hp.logunifrom("gamma", np.log(0.001), np.log(0.1)),
    "degree": hp.quniform("degree", 1, 3, 1),
    "epsilon": hp.loguniform("epsion", np.log(0.001), np.log(0.1)),
    "kernel": hp.choice("kernel", ["rbf", "poly"])
}

## K Nearest Neighbors Regression
param_space_reg_skl_knn = {
    "normalize": hp.choice("normalize", [True, False]),
    "n_neighbors": hp.quniform("n_neighbors", 1, 20, 1),
    "weights": hp.choice("weights", ["uniform", "distance"]),
    "leaf_size": hp.quniform("leaf_size", 10, 100, 10),
    "metric": hp.choice("metric", ["cosine", "minkowski"][1:])
}

## extra trees regressor
param_space_reg_skl_etr = {
    "n_estimators": hp.quniform("skl_etr__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "max_features": hp.quniform("skl_etr__max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("skl_etr__min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("skl_etr__min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("skl_etr__max_depth", 1, 10, 1),
    "random_state": skl_random_seed,
    "n_jobs": skl_n_jobs,
    "verbose": 0,
}

## random forest regressor
param_space_reg_skl_rf = {
    "n_estimators": hp.quniform("skl_rf__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "max_features": hp.quniform("skl_rf__max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("skl_rf__min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("skl_rf__min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("skl_rf__max_depth", 1, 10, 1),
    "random_state": skl_random_seed,
    "n_jobs": skl_n_jobs,
    "verbose": 0,
}

## gradient boosting regressor
param_space_reg_skl_gbm = {
    "n_estimators": hp.quniform("skl_gbm__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("skl__gbm_learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("skl_gbm__max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("skl_gbm__max_depth", 1, 10, 1),
    "min_samples_leaf": hp.quniform("skl_gbm__min_samples_leaf", 1, 15, 1),
    "random_state": skl_random_seed,
    "verbose": 0,
}

## adaboost regressor
param_space_reg_skl_adaboost = {
    "base_estimator": hp.choice("base_estimator", ["dtr", "etr"]),
    "n_estimators": hp.quniform("n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    "random_state": skl_random_seed,
}




# -------------------------------------- Ensemble ---------------------------------------------
# 1. The following learners are chosen to build ensemble for their fast learning speed.
# 2. In our final submission, we used fix weights.
#    However, you can also try to optimize the ensemble weights in the meantime.
param_space_reg_ensemble = {
    # 1. fix weights (used in final submission)
    "learner_dict": {
        "reg_skl_ridge": {
            "param": param_space_reg_skl_ridge,
            "weight": 4.0,
        },
        "reg_keras_dnn": {
            "param": param_space_reg_keras_dnn,
            "weight": 1.0,
        },
        "reg_xgb_tree": {
            "param": param_space_reg_xgb_tree,
            "weight": 1.0,
        },
        "reg_skl_etr": {
            "param": param_space_reg_skl_etr,
            "weight": 1.0,
        },
        "reg_skl_rf": {
            "param": param_space_reg_skl_rf,
            "weight": 1.0,
        },
    },
    # # 2. optimizing weights
    # "learner_dict": {
    #     "reg_skl_ridge": {
    #         "param": param_space_reg_skl_ridge,
    #         "weight": hp.quniform("reg_skl_ridge__weight", 1.0, 1.0, 0.1), # fix this one
    #     },
    #     "reg_keras_dnn": {
    #         "param": param_space_reg_keras_dnn,
    #         "weight": hp.quniform("reg_keras_dnn__weight", 0.0, 1.0, 0.1),
    #     },
    #     "reg_xgb_tree": {
    #         "param": param_space_reg_xgb_tree,
    #         "weight": hp.quniform("reg_xgb_tree__weight", 0.0, 1.0, 0.1),
    #     },
    #     "reg_skl_etr": {
    #         "param": param_space_reg_skl_etr,
    #         "weight": hp.quniform("reg_skl_etr__weight", 0.0, 1.0, 0.1),
    #     },
    #     "reg_skl_rf": {
    #         "param": param_space_reg_skl_rf,
    #         "weight": hp.quniform("reg_skl_rf__weight", 0.0, 1.0, 0.1),
    #     },
    # },
}

# -------------------------------------- All ---------------------------------------------
param_space_dict = {
    # xgboost
    "reg_xgb_tree": param_space_reg_xgb_tree,
    "reg_xgb_tree_best_single_model": param_space_reg_xgb_tree_best_single_model,
    "reg_xgb_linear": param_space_reg_xgb_linear,
    "clf_xgb_tree": param_space_clf_xgb_tree,
    # sklearn
    "reg_skl_lasso": param_space_reg_skl_lasso,
    "reg_skl_ridge": param_space_reg_skl_ridge,
    "reg_skl_bayesian_ridge": param_space_reg_skl_bayesian_ridge,
    "reg_skl_random_ridge": param_space_reg_skl_random_ridge,
    "reg_skl_lsvr": param_space_reg_skl_lsvr,
    "reg_skl_svr": param_space_reg_skl_svr,
    "reg_skl_knn": param_space_reg_skl_knn,
    "reg_skl_etr": param_space_reg_skl_etr,
    "reg_skl_rf": param_space_reg_skl_rf,
    "reg_skl_gbm": param_space_reg_skl_gbm,
    "reg_skl_adaboost": param_space_reg_skl_adaboost,
    # keras
    "reg_keras_dnn": param_space_reg_keras_dnn,
    # rgf
    "reg_rgf": param_space_reg_rgf,
    # ensemble
    "reg_ensemble": param_space_reg_ensemble,
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
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            "see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

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

