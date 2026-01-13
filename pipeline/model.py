from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np


def _scale_pos_weight(y):
    """
    Compute imbalance ratio = (#normal / #anomaly).
    This tells XGBoost to care more about the positive class (anomaly).
    """
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos


def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Improved model training function with:
    - separate validation split for early stopping
    - tuned XGBoost hyperparameters
    - scale_pos_weight for class imbalance
    - fit inside a Pipeline(preproc → model)
    """

    # ------------------------------------------------------
    # 1. Create validation set for early stopping
    # ------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train
    )

    # imbalance management
    spw = _scale_pos_weight(y_train)

    # ------------------------------------------------------
    # 2. Improved XGBoost configuration
    # ------------------------------------------------------
    clf = XGBClassifier(
        n_estimators=500,             # more estimators for early stopping
        learning_rate=0.05,           # smaller LR → more stable
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0
    )

    # ------------------------------------------------------
    # 3. Build pipeline = preprocessing + XGBoost
    # ------------------------------------------------------
    pipe = Pipeline([
        ("preproc", preproc),
        ("model", clf)
    ])

    # ------------------------------------------------------
    # 4. Fit with early stopping *inside the pipeline*
    # ------------------------------------------------------
    # NOTE: XGBoost *only* receives transformed data inside its .fit().
    pipe.named_steps["model"].fit(
        preproc.fit_transform(X_tr),  # fit preproc on training only
        y_tr,
        eval_set=[(preproc.transform(X_val), y_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    return pipe
