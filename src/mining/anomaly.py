"""
anomaly.py – Phát hiện bất thường / outlier cho Predictive Maintenance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class AnomalyDetector:
    """
    Phát hiện anomaly bằng Isolation Forest, LOF, One-Class SVM.
    So sánh kết quả anomaly detection với nhãn failure thực tế.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.models = {}
        self.predictions = {}

    def fit_isolation_forest(
        self, X: np.ndarray, contamination: float = 0.034
    ) -> np.ndarray:
        """Isolation Forest – contamination ≈ tỷ lệ failure."""
        model = IsolationForest(
            contamination=contamination,
            random_state=self.params["seed"],
            n_estimators=200,
        )
        preds = model.fit_predict(X)  # 1 = normal, -1 = anomaly
        anomaly_labels = (preds == -1).astype(int)
        
        self.models["isolation_forest"] = model
        self.predictions["isolation_forest"] = anomaly_labels
        print(f"[anomaly] Isolation Forest: {anomaly_labels.sum()} anomalies detected")
        return anomaly_labels

    def fit_lof(self, X: np.ndarray, contamination: float = 0.034) -> np.ndarray:
        """Local Outlier Factor."""
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            novelty=False,
        )
        preds = model.fit_predict(X)
        anomaly_labels = (preds == -1).astype(int)
        
        self.models["lof"] = model
        self.predictions["lof"] = anomaly_labels
        print(f"[anomaly] LOF: {anomaly_labels.sum()} anomalies detected")
        return anomaly_labels

    def fit_ocsvm(self, X: np.ndarray, nu: float = 0.034) -> np.ndarray:
        """One-Class SVM (dùng trên sample nhỏ nếu data lớn)."""
        model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        preds = model.fit_predict(X)
        anomaly_labels = (preds == -1).astype(int)
        
        self.models["ocsvm"] = model
        self.predictions["ocsvm"] = anomaly_labels
        print(f"[anomaly] One-Class SVM: {anomaly_labels.sum()} anomalies detected")
        return anomaly_labels

    def compare_with_actual(
        self, y_true: np.ndarray
    ) -> pd.DataFrame:
        """
        So sánh anomaly detection với nhãn lỗi thực.
        Tính precision/recall/f1 coi anomaly = failure.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        rows = []
        for name, preds in self.predictions.items():
            n_detected = preds.sum()
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            acc = accuracy_score(y_true, preds)

            rows.append({
                "Method": name,
                "Anomalies Detected": n_detected,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "Accuracy": round(acc, 4),
            })

        result = pd.DataFrame(rows)
        print("[anomaly] Comparison with actual failures:")
        print(result.to_string(index=False))
        return result
