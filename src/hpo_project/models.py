from inspect import signature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from .preprocessing import make_preprocessor


def make_model(name, task, seed, n_jobs):
    regression = task == "regression"
    if name == "elastic_net":
        if regression:
            model = ElasticNet(max_iter=50000, random_state=seed)
        else:
            options = dict(solver="saga", l1_ratio=0.5, max_iter=10000, random_state=seed)
            # Before sklearn 1.8, the penalty must be explicit. Newer versions
            # infer it from l1_ratio and C; n_jobs is unnecessary for SAGA.
            penalty = signature(LogisticRegression).parameters.get("penalty")
            if penalty is not None and penalty.default == "l2":
                options["penalty"] = "elasticnet"
            model = LogisticRegression(**options)
    elif name == "svm":
        model = SVR() if regression else SVC(probability=True, random_state=seed)
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor, LGBMClassifier
        model = (LGBMRegressor if regression else LGBMClassifier)(random_state=seed, n_jobs=n_jobs,
            verbosity=-1, subsample_freq=1)
    else:
        raise ValueError(name)
    # Pilot uses one shared fold-fitted encoder; native LightGBM categoricals are pending.
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])
