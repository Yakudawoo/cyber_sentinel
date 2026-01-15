from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np


def _scale_pos_weight(y):
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos


def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Train XGBoost with proper preprocessing and early stopping.
    Preprocessing is applied BEFORE passing data to XGBoost (no strings).
    """

    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train
    )
    print(X_train.dtypes) # Toujours vérifier dtypes avant d’entraîner un modèle
    # Si vous voyez du string → c’est un problème pouyr xgboost


    # Fit preprocessing ONLY on training set
    preproc.fit(X_tr) # étape 1 : on fit uniquement dans preprocessing sur le train

    # Transform train + validation on transorme X_tr et X_val
    X_tr_trans = preproc.transform(X_tr)
    X_val_trans = preproc.transform(X_val) # point clé en utilisant early_stopping

    # Class imbalance
    spw = _scale_pos_weight(y_train)

    # XGBoost config
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
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

    # Fit XGBoost on transformed data On passe ces données transformées au modèle
    clf.fit(
        X_tr_trans,
        y_tr,
        eval_set=[(X_val_trans, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    # Build final pipeline: preprocessing + trained model
    # On encapsule tout dans un pipeline final pour les prédictions
    pipe = Pipeline([
        ("preproc", preproc),
        ("model", clf)
    ])

    return pipe
