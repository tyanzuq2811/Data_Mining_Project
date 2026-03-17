"""
semi_supervised.py – Bán giám sát: Self-Training & Label Spreading.

Kịch bản: Giữ lại p% nhãn (5/10/20%), phần còn lại coi là unlabeled.
So sánh Supervised-only (ít nhãn) vs Semi-supervised.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class SemiSupervisedTrainer:
    """
    Thử nghiệm semi-supervised learning cho bài toán thiếu nhãn.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.results = []
        self.learning_curves = {}
        self.pseudo_label_risk_records = []

    def _decision_threshold(self) -> float:
        return float(self.params.get("semi_supervised", {}).get("decision_threshold", 0.35))

    def create_partially_labeled(
        self,
        y: np.ndarray,
        label_pct: float,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Tạo nhãn bán giám sát: giữ label_pct% nhãn, còn lại = -1.
        Stratified sampling để giữ tỷ lệ lớp.
        """
        rng = np.random.RandomState(random_state)
        y_semi = np.full_like(y, -1)

        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_keep = max(1, int(len(cls_idx) * label_pct))
            keep_idx = rng.choice(cls_idx, size=n_keep, replace=False)
            y_semi[keep_idx] = y[keep_idx]

        n_labeled = (y_semi != -1).sum()
        n_unlabeled = (y_semi == -1).sum()
        print(f"[semi] {label_pct*100:.0f}% labels: {n_labeled} labeled, {n_unlabeled} unlabeled")
        return y_semi

    def train_supervised_only(
        self,
        X_train: np.ndarray,
        y_train_semi: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_pct: float,
    ) -> Dict:
        """
        Baseline: Supervised-only với chỉ labeled samples.
        """
        labeled_mask = y_train_semi != -1
        X_labeled = X_train[labeled_mask]
        y_labeled = y_train_semi[labeled_mask]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced",
            random_state=self.params["seed"],
            n_jobs=-1,
        )
        model.fit(X_labeled, y_labeled)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if y_prob is not None:
            y_pred = (y_prob >= self._decision_threshold()).astype(int)

        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        result = {
            "method": "supervised_only",
            "label_pct": label_pct,
            "n_labeled": int(labeled_mask.sum()),
            **metrics,
        }
        self.results.append(result)
        return result

    def train_self_training(
        self,
        X_train: np.ndarray,
        y_train_true: np.ndarray,
        y_train_semi: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_pct: float,
    ) -> Dict:
        """
        Self-Training: Base classifier = RandomForest, threshold cao.
        """
        st_params = self.params.get("semi_supervised", {}).get("self_training", {})
        threshold = st_params.get("threshold", 0.95)
        max_iter = st_params.get("max_iter", 30)
        criterion = st_params.get("criterion", "threshold")
        k_best = int(st_params.get("k_best", 300))

        base_clf = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced",
            random_state=self.params["seed"],
            n_jobs=-1,
        )

        st_kwargs = {
            "estimator": base_clf,
            "max_iter": max_iter,
            "verbose": False,
            "criterion": criterion,
        }
        if criterion == "k_best":
            st_kwargs["k_best"] = k_best
        else:
            st_kwargs["threshold"] = threshold

        model = SelfTrainingClassifier(**st_kwargs)
        model.fit(X_train, y_train_semi)

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        if y_prob is not None:
            y_pred = (y_prob >= self._decision_threshold()).astype(int)

        # Pseudo-label analysis
        pseudo_labels = model.transduction_
        original_unlabeled = y_train_semi == -1
        n_pseudo = original_unlabeled.sum()

        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        result = {
            "method": "self_training",
            "label_pct": label_pct,
            "n_labeled": int((y_train_semi != -1).sum()),
            "n_pseudo_labeled": int(n_pseudo),
            "threshold": threshold,
            "criterion": criterion,
            "k_best": k_best if criterion == "k_best" else 0,
            **metrics,
        }
        self.results.append(result)

        # Phân tích pseudo-label
        risk = self._analyze_pseudo_labels(
            y_true_full=y_train_true,
            y_semi_original=y_train_semi,
            y_transduction=pseudo_labels,
            unlabeled_mask=original_unlabeled,
            label_pct=label_pct,
        )
        result.update(risk)
        return result

    def train_label_spreading(
        self,
        X_train: np.ndarray,
        y_train_semi: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_pct: float,
    ) -> Dict:
        """Label Spreading (graph-based semi-supervised)."""
        ls_params = self.params.get("semi_supervised", {}).get("label_spreading", {})
        kernel = ls_params.get("kernel", "rbf")
        alpha = ls_params.get("alpha", 0.2)
        max_iter_ls = ls_params.get("max_iter", 100)

        # Label Spreading cần -1 làm unlabeled
        model = LabelSpreading(
            kernel=kernel,
            alpha=alpha,
            max_iter=max_iter_ls,
        )

        # Subsample nếu dữ liệu quá lớn (Label Spreading O(n²))
        n_max = 5000
        if len(X_train) > n_max:
            rng = np.random.RandomState(self.params["seed"])
            idx = rng.choice(len(X_train), n_max, replace=False)
            X_fit = X_train[idx]
            y_fit = y_train_semi[idx]
        else:
            X_fit = X_train
            y_fit = y_train_semi

        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if y_prob is not None:
            y_pred = (y_prob >= self._decision_threshold()).astype(int)

        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        result = {
            "method": "label_spreading",
            "label_pct": label_pct,
            "n_labeled": int((y_train_semi != -1).sum()),
            **metrics,
        }
        self.results.append(result)
        return result

    def run_all_experiments(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Chạy toàn bộ thực nghiệm bán giám sát cho các % nhãn.
        """
        label_pcts = self.params.get("semi_supervised", {}).get("label_percentages", [0.05, 0.10, 0.20])
        self.results = []
        self.pseudo_label_risk_records = []

        for pct in label_pcts:
            print(f"\n{'='*60}")
            print(f"  Label percentage: {pct*100:.0f}%")
            print(f"{'='*60}")

            y_semi = self.create_partially_labeled(y_train, pct, self.params["seed"])

            # Supervised-only baseline
            self.train_supervised_only(X_train, y_semi, X_test, y_test, pct)

            # Self-Training
            self.train_self_training(X_train, y_train, y_semi, X_test, y_test, pct)

            # Label Spreading
            self.train_label_spreading(X_train, y_semi, X_test, y_test, pct)

        results_df = pd.DataFrame(self.results)
        return results_df

    def get_learning_curve_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pct_range: List[float] = None,
    ) -> pd.DataFrame:
        """
        Tạo dữ liệu learning curve: F1 theo % nhãn cho các phương pháp.
        """
        if pct_range is None:
            pct_range = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

        curve_data = []
        for pct in pct_range:
            print(f"\n[semi] Learning curve – {pct*100:.0f}%")

            if pct >= 1.0:
                # Full supervised
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    class_weight="balanced",
                    random_state=self.params["seed"], n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                curve_data.append({"pct": pct, "method": "supervised_only", "f1": f1})
                curve_data.append({"pct": pct, "method": "self_training", "f1": f1})
                curve_data.append({"pct": pct, "method": "label_spreading", "f1": f1})
            else:
                y_semi = self.create_partially_labeled(y_train, pct, self.params["seed"])

                # Supervised-only
                r1 = self.train_supervised_only(X_train, y_semi, X_test, y_test, pct)
                curve_data.append({"pct": pct, "method": "supervised_only", "f1": r1["f1"]})

                # Self-Training
                r2 = self.train_self_training(X_train, y_train, y_semi, X_test, y_test, pct)
                curve_data.append({"pct": pct, "method": "self_training", "f1": r2["f1"]})

                # Label Spreading
                r3 = self.train_label_spreading(X_train, y_semi, X_test, y_test, pct)
                curve_data.append({"pct": pct, "method": "label_spreading", "f1": r3["f1"]})

        return pd.DataFrame(curve_data)

    def _compute_metrics(self, y_true, y_pred, y_prob=None) -> Dict:
        """Tính các metrics."""
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        metrics = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
                metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
            except Exception:
                pass

        return metrics

    def _analyze_pseudo_labels(
        self,
        y_true_full: np.ndarray,
        y_semi_original: np.ndarray,
        y_transduction: np.ndarray,
        unlabeled_mask: np.ndarray,
        label_pct: float,
    ):
        """Phân tích rủi ro pseudo-label."""
        # So sánh pseudo-label vs label gốc (nếu biết)
        # Trong thực tế ta KHÔNG biết nhãn thật của unlabeled, nhưng vì đây là thử nghiệm...
        n_pseudo = int(unlabeled_mask.sum())
        y_true_u = y_true_full[unlabeled_mask]
        y_pred_u = y_transduction[unlabeled_mask]

        pseudo_pos = int((y_pred_u == 1).sum())
        pseudo_neg = int((y_pred_u == 0).sum())
        pseudo_tp = int(((y_pred_u == 1) & (y_true_u == 1)).sum())
        pseudo_fp = int(((y_pred_u == 1) & (y_true_u == 0)).sum())
        pseudo_fn = int(((y_pred_u == 0) & (y_true_u == 1)).sum())
        true_failures_unlabeled = int((y_true_u == 1).sum())

        pseudo_precision = pseudo_tp / (pseudo_pos + 1e-10)
        false_alarm_rate = pseudo_fp / (pseudo_pos + 1e-10)
        miss_rate_unlabeled_failures = pseudo_fn / (true_failures_unlabeled + 1e-10)

        print(f"  [pseudo-label analysis] {label_pct*100:.0f}%:")
        print(f"    Total pseudo-labeled: {n_pseudo}")
        print(f"    Pseudo positive (failure): {pseudo_pos}")
        print(f"    Pseudo negative (normal): {pseudo_neg}")
        print(f"    Pseudo failure rate: {pseudo_pos/(n_pseudo+1e-10)*100:.2f}%")
        print(f"    False alarms in pseudo positives: {pseudo_fp} ({false_alarm_rate*100:.2f}%)")

        risk = {
            "label_pct": round(float(label_pct), 4),
            "n_unlabeled": n_pseudo,
            "n_true_failures_unlabeled": true_failures_unlabeled,
            "n_pseudo_positive": pseudo_pos,
            "n_pseudo_negative": pseudo_neg,
            "pseudo_tp": pseudo_tp,
            "pseudo_fp_false_alarm": pseudo_fp,
            "pseudo_fn_missed_failure": pseudo_fn,
            "pseudo_precision": round(float(pseudo_precision), 4),
            "pseudo_false_alarm_rate": round(float(false_alarm_rate), 4),
            "pseudo_miss_rate_on_unlabeled_failures": round(float(miss_rate_unlabeled_failures), 4),
        }
        self.pseudo_label_risk_records.append(risk)
        return risk

    def get_pseudo_label_risk_table(self) -> pd.DataFrame:
        """Return pseudo-label false alarm/miss analysis by labeled percentage."""
        if len(self.pseudo_label_risk_records) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.pseudo_label_risk_records).sort_values("label_pct")
