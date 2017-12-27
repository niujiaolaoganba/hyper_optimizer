# -*- coding: utf-8 -*-
"""
@author: Charles Liu <guobinliulgb@gmail.com>
@brief:

@todo:
**skl_utils**
config, RANDOM_SEED, NUM_CORES
学习hyperopt，hp
学习回归模型skl_lasso，skl_ridge，skl_bayesian_ridge
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




