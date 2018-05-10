import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from hyperopt import hp

param_space_clf_skl_lr = {
    "C": hp.loguniform("C", np.log(1e-7), np.log(1e2)),
    "penalty": hp.choice("penalty", ['l1', 'l2']),
    "random_state": 42,
}

param_space_clf_skl_rf = {
    "n_estimators": hp.quniform("skl_rf__n_estimators", 10, 1000, 10),
    "max_features": hp.quniform("skl_rf__max_features", 0.3, 1, 0.05),
    "min_samples_split": hp.quniform("skl_rf__min_samples_split", 5, 15, 1),
    "min_samples_leaf": hp.quniform("skl_rf__min_samples_leaf", 5, 15, 1),
    "max_depth": hp.quniform("skl_rf__max_depth", 2, 10, 1),
    "random_state": 42,
    "n_jobs": 8,
    "verbose": 0,
}

param_space_clf_xgb_tree = {
    'max_depth': hp.quniform('xgb_tree__max_depth', 2, 10, 1),
    'subsample': hp.uniform('xgb_tree__subsample', 0.5, 1),
    "n_estimators": hp.quniform("xgb_tree__n_estimators", 100, 1000, 10),
    "learning_rate": hp.qloguniform("xgb_tree__learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "gamma": hp.loguniform("xgb_tree__gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha": hp.loguniform("xgb_tree__reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda": hp.loguniform("xgb_tree__reg_lambda", np.log(1e-10), np.log(1e1)),
    'min_child_weight': hp.loguniform('xgb_tree__min_child_weight', -16, 5),
    'colsample_bytree': hp.uniform('xgb_tree__colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('xgb_tree__colsample_bylevel', 0.5, 1),
    "nthread": 8,
    "seed": 42,
}
# -------------------------------------- Ensemble ---------------------------------------------
# 1. The following learners are chosen to build ensemble for their fast learning speed.
# 2. In our final submission, we used fix weights.
#    However, you can also try to optimize the ensemble weights in the meantime.
param_space_ensemble = {
    # 1. fix weights (used in final submission)

    # "clf_skl_lr": {
    #     "learner":LogisticRegression(),
    #     "param": param_space_clf_skl_lr,
    #     "weight": 4.0,
    # },
    # "clf_xgb_tree": {
    #     "param": param_space_clf_xgb_tree,
    #     "learner":XGBClassifier(),
    #     "weight": 1.0,
    # },
    # "clf_skl_rf": {
    #     "param": param_space_clf_skl_rf,
    #     "learner":RandomForestClassifier(),
    #     "weight": 1.0,
    # },

    # # 2. optimizing weights
        "clf_skl_lr": {
            "param": param_space_clf_skl_lr,
            "weight": hp.quniform("clf_skl_lr__weight", 1.0, 1.0, 0.1),  # fix this one
        },
        "clf_xgb_tree": {
            "param": param_space_clf_xgb_tree,
            "weight": hp.quniform("reg_xgb_tree__weight", 0.0, 1.0, 0.1),
        },
        "clf_skl_rf": {
            "param": param_space_clf_skl_rf,
            "weight": hp.quniform("reg_skl_rf__weight", 0.0, 1.0, 0.1),
        },
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)

learner_name_space = {
    "clf_skl_lr": LogisticRegression,
    "clf_xgb_tree": XGBClassifier,
    "clf_skl_rf": RandomForestClassifier,
}

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
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.param_dict.keys():
            p = convert_int_param(self.param_dict[learner_name]["param"])
            l = learner_name_space[learner_name](**p)
            if l is not None:
                self.param_dict[learner_name]["learner"] = l.fit(X, y)
            else:
                self.param_dict[learner_name]["learner"] = None
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.param_dict.keys():
            l = self.param_dict[learner_name]["learner"]
            if l is not None:
                w = self.param_dict[learner_name]["weight"]
                y_pred += w * l.predict(X)
                w_sum += w
        y_pred /= w_sum
        return y_pred

    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.param_dict.keys():
            l = self.param_dict[learner_name]["learner"]
            if l is not None:
                w = self.param_dict[learner_name]["weight"]
                y_pred += w * l.predict_proba(X)[:, 1]
                w_sum += w
        y_pred /= w_sum
        return y_pred
