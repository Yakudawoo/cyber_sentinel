import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix
)


def make_plots(model, X_test, y_test, metrics: dict, output_dir: str = "outputs"):
    """
    Generate professional visualizations:
    - ROC curve
    - Precision-Recall curve
    - Confusion matrix heatmap
    - Feature importance (XGBoost)
    Saves PNGs into /outputs.
    """

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Probabilities for curves
    # --------------------------------------------------------
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: approximate scores
        if hasattr(model.named_steps["model"], "decision_function"):
            raw = model.decision_function(X_test)
            y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            y_proba = model.predict(X_test).astype(float)

    # --------------------------------------------------------
    # 2. ROC Curve
    # --------------------------------------------------------
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve (Anomaly Detection)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=120)
    plt.close()

    # --------------------------------------------------------
    # 3. Precision–Recall Curve
    # --------------------------------------------------------
    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title("Precision-Recall Curve (Anomaly = Positive Class)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=120)
    plt.close()

    # --------------------------------------------------------
    # 4. Confusion Matrix
    # --------------------------------------------------------
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["normal", "anomaly"],
                yticklabels=["normal", "anomaly"])
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=120)
    plt.close()

    # --------------------------------------------------------
    # 5. Feature Importance (XGBoost)
    # --------------------------------------------------------
    model_xgb = model.named_steps["model"]

    if hasattr(model_xgb, "feature_importances_"):
        # Get feature names after preprocessing (OHE expands categories)
        try:
            feature_names = model.named_steps["preproc"].get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(len(model_xgb.feature_importances_))]

        importances = model_xgb.feature_importances_
        idx = np.argsort(importances)[-20:]  # show top 20

        plt.figure(figsize=(10, 7))
        plt.barh(np.array(feature_names)[idx], importances[idx], color="navy")
        plt.title("Top 20 Feature Importances (XGBoost)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=120)
        plt.close()

    # --------------------------------------------------------
    # 6. Save metrics.json as pretty JSON
    # --------------------------------------------------------
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
