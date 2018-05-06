import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from hyperopt import hp

param_space_clf_skl_lr = {
    "C": hp.loguniform("C", np.log(1e-10), np.log(1e10)),
    "penalty": hp.choice("penalty", ['l1', 'l2']),
    "random_state": 42,
}

param_space_clf_skl_rf = {
    "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "random_state": 42,
    "n_jobs": 8,
    "verbose": 0,
}

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
# -------------------------------------- Ensemble ---------------------------------------------
# 1. The following learners are chosen to build ensemble for their fast learning speed.
# 2. In our final submission, we used fix weights.
#    However, you can also try to optimize the ensemble weights in the meantime.
param_space_ensemble = {
    # 1. fix weights (used in final submission)

        "clf_skl_lr": {
            "learner":LogisticRegression,
            "param": param_space_clf_skl_lr,
            "weight": 4.0,
        },
        "clf_xgb_tree": {
            "param": param_space_clf_xgb_tree,
            "learner":XGBClassifier,
            "weight": 1.0,
        },
        "clf_skl_rf": {
            "param": param_space_clf_skl_rf,
            "learner":RandomForestClassifier,
            "weight": 1.0,
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

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)

def convert_int_param(param_dict):
    if isinstance(param_dict, dict):
        for k, v in param_dict.items():
            if k in int_params:
                param_dict[k] = int(v)
            elif isinstance(v, list) or isinstance(v, tuple):
                for i in range(len(v)):
                    convert_int_param(v[i])
            elif isinstance(v, dict):
                convert_int_param(v)
    return param_dict


class EnsembleLearner:
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            p = convert_int_param(self.learner_dict[learner_name]["param"])
            l = self.learner_dict[learner_name]["learner"](**p)
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

    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            if l is not None:
                w = self.learner_dict[learner_name]["weight"]
                y_pred += w * l.predict_proba(X)[:,1]
                w_sum += w
        y_pred /= w_sum
        return y_pred
