import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, balanced_accuracy_score
)


def _optimal_threshold_f1(y_true, y_proba):
    """
    Find threshold that maximizes F1 score.
    """
    best_thr = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(0.01, 0.99, 200)

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred_thr, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def _threshold_for_recall(y_true, y_proba, target_recall=0.90):
    """
    Return the lowest threshold that achieves at least target recall.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        recall_val = recall_score(y_true, y_pred_thr, zero_division=0)
        if recall_val >= target_recall:
            return thr, recall_val
    return None, None


def evaluate(model, X_test, y_test) -> dict:
    """
    Advanced evaluation including:
    - base metrics : accuracy, precision, recall, F1
    - ROC-AUC, PR-AUC
    - confusion matrix (TP, FP, FN, TN)
    - balanced accuracy
    - threshold optimization (max F1, target recall)
    - calibration (Brier score)
    """

    # ------------------------------------------------------------
    # 1. Predictions & probabilities
    # ------------------------------------------------------------
    y_pred = model.predict(X_test)

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model.named_steps["model"], "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    # ------------------------------------------------------------
    # 2. Base metrics
    # ------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None
    pr_auc = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # ------------------------------------------------------------
    # 3. Confusion matrix details
    # ------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp + 1e-9)  # avoids division by zero

    # ------------------------------------------------------------
    # 4. Threshold optimization
    # ------------------------------------------------------------
    best_thr_f1, best_f1 = _optimal_threshold_f1(y_test, y_proba)
    thr_recall90, achieved_recall90 = _threshold_for_recall(y_test, y_proba, target_recall=0.90)

    # ------------------------------------------------------------
    # 5. Return structured dictionary
    # ------------------------------------------------------------
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),

        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(balanced_acc),
        "brier_score": float(brier),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(specificity),

        # thresholds
        "best_threshold_f1": float(best_thr_f1),
        "best_f1_score": float(best_f1),

        "threshold_recall_90": float(thr_recall90) if thr_recall90 is not None else None,
        "achieved_recall_90": float(achieved_recall90) if achieved_recall90 is not None else None,

        # dataset info
        "positives_test": int(np.sum(y_test == 1)),
        "negatives_test": int(np.sum(y_test == 0)),
    }

    return metrics
