import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.data import load_data
from pipeline.preprocess import preprocess
from pipeline.model import train_model
from pipeline.evaluate import evaluate
from pipeline.visualize import make_plots


OUTPUT_DIR = "test_outputs"


def setup_module(module):
    """Create output directory for plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def teardown_module(module):
    """Clean generated files after tests."""
    for f in Path(OUTPUT_DIR).glob("*"):
        f.unlink()
    Path(OUTPUT_DIR).rmdir()


# ---------------------------------------------------
# 1. TEST load_data()
# ---------------------------------------------------
def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "class" in df.columns
    assert set(df["class"].unique()).issubset({0, 1})


# ---------------------------------------------------
# 2. TEST preprocess()
# ---------------------------------------------------
def test_preprocess():
    df = load_data()
    X_train, X_test, y_train, y_test, preproc = preprocess(df)

    # Splits return non-empty arrays
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # y must be binary
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

    # Preprocessor must implement fit/transform
    assert hasattr(preproc, "fit")
    assert hasattr(preproc, "transform")


# ---------------------------------------------------
# 3. TEST train_model()
# ---------------------------------------------------
def test_train_model():
    df = load_data()
    X_train, X_test, y_train, y_test, preproc = preprocess(df)

    model = train_model(X_train, y_train, preproc)

    # Must be a sklearn Pipeline
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)
    assert "model" in model.named_steps
    assert "preproc" in model.named_steps

    # Should be able to predict
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


# ---------------------------------------------------
# 4. TEST evaluate()
# ---------------------------------------------------
def test_evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test, preproc = preprocess(df)
    model = train_model(X_train, y_train, preproc)
    metrics = evaluate(model, X_test, y_test)

    # Basic structure checks
    required = [
        "accuracy", "precision", "recall", "f1", 
        "pr_auc", "positives_test", "negatives_test"
    ]
    for key in required:
        assert key in metrics

    assert metrics["accuracy"] >= 0
    assert metrics["recall"] >= 0
    assert metrics["precision"] >= 0


# ---------------------------------------------------
# 5. TEST visualize()
# ---------------------------------------------------
def test_visualize_creates_pngs():
    df = load_data()
    X_train, X_test, y_train, y_test, preproc = preprocess(df)
    model = train_model(X_train, y_train, preproc)
    metrics = evaluate(model, X_test, y_test)

    make_plots(model, X_test, y_test, df, OUTPUT_DIR)

    # Check expected PNG files
    expected_files = [
        "roc_curve.png",
        "pr_curve.png",
        "confusion_matrix.png",
        "feature_importance.png",
    ]

    for fname in expected_files:
        path = Path(OUTPUT_DIR) / fname
        assert path.exists(), f"{fname} was not generated"
