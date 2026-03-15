"""
metrics.py – Tổng hợp metrics cho classification, regression, clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """Tính toàn bộ classification metrics."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
            metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Tính MAE, RMSE, R², MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (tránh chia 0)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Tính Silhouette, DBI, CHI cho clustering."""
    unique = set(labels)
    unique.discard(-1)

    if len(unique) < 2:
        return {"silhouette": -1, "davies_bouldin": 999, "calinski_harabasz": 0}

    mask = labels != -1
    return {
        "silhouette": round(silhouette_score(X[mask], labels[mask]), 4),
        "davies_bouldin": round(davies_bouldin_score(X[mask], labels[mask]), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X[mask], labels[mask]), 2),
    }


def get_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
) -> pd.DataFrame:
    """Tạo confusion matrix dưới dạng DataFrame."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [f"Class_{i}" for i in range(cm.shape[0])]
    return pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels],
                        columns=[f"Pred_{l}" for l in labels])


def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
) -> pd.DataFrame:
    """Classification report dưới dạng DataFrame."""
    if target_names is None:
        target_names = ["Normal", "Failure"]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).T


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """So sánh nhiều mô hình theo metrics."""
    return pd.DataFrame(results)


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame = None,
) -> Dict:
    """
    Phân tích lỗi: false positive, false negative, pattern.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    analysis = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": round(fp / (fp + tn + 1e-10), 4),
        "false_negative_rate": round(fn / (fn + tp + 1e-10), 4),
        "false_alarm_rate": round(fp / (fp + tn + 1e-10), 4),
        "miss_rate": round(fn / (fn + tp + 1e-10), 4),
    }

    if df is not None:
        # Indices of errors
        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
        analysis["false_positive_indices"] = fp_idx.tolist()
        analysis["false_negative_indices"] = fn_idx.tolist()

    return analysis
