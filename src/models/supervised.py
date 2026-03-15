"""
supervised.py – Train/Predict cho Classification & Regression.

Classification: Phân lớp lỗi/không lỗi (hoặc loại lỗi) – Imbalance handling.
Regression: Dự đoán Tool wear [min].
Time Series: ARIMA/lag-features với UDI ≈ time index.
"""

import time
import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, make_scorer,
    mean_absolute_error, mean_squared_error, r2_score,
)
import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class SupervisedTrainer:
    """
    Huấn luyện và đánh giá các mô hình supervised learning.
    Hỗ trợ classification (binary) và regression.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.training_times = {}

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================
    def get_classifiers(self) -> Dict[str, Any]:
        """Tạo các classifier với hyperparameters mặc định hợp lý."""
        seed = self.params["seed"]
        classifiers = {
            "logistic_regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=seed,
            ),
        }

        if HAS_XGBOOST:
            classifiers["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=28.5,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            )

        if HAS_LIGHTGBM:
            classifiers["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                is_unbalance=True,
                random_state=seed,
                verbose=-1,
            )

        return classifiers

    def train_classifiers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """
        Huấn luyện tất cả classifier, đánh giá trên test set.
        
        Returns DataFrame so sánh các mô hình.
        """
        classifiers = self.get_classifiers()
        results = []

        for name, model in classifiers.items():
            print(f"\n[supervised] Training {name}...")
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
            pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else 0
            cm = confusion_matrix(y_test, y_pred)

            res = {
                "model": name,
                "f1": round(f1, 4),
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr_auc, 4),
                "precision": round(cm[1, 1] / (cm[0, 1] + cm[1, 1] + 1e-10), 4),
                "recall": round(cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-10), 4),
                "train_time_s": round(train_time, 2),
            }
            results.append(res)
            self.models[name] = model
            self.results[name] = {**res, "confusion_matrix": cm, "y_pred": y_pred, "y_prob": y_prob}
            self.training_times[name] = train_time
            print(f"  → F1={f1:.4f}, ROC-AUC={roc:.4f}, PR-AUC={pr_auc:.4f}, Time={train_time:.2f}s")

        results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
        
        # Best model
        self.best_model_name = results_df.iloc[0]["model"]
        self.best_model = self.models[self.best_model_name]
        print(f"\n[supervised] Best classifier: {self.best_model_name}")

        return results_df

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> pd.DataFrame:
        """Cross-validation cho tất cả classifier."""
        classifiers = self.get_classifiers()
        cv_results = []

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.params["seed"])

        for name, model in classifiers.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
            cv_results.append({
                "model": name,
                "cv_f1_mean": round(scores.mean(), 4),
                "cv_f1_std": round(scores.std(), 4),
            })
            print(f"[supervised] {name} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

        return pd.DataFrame(cv_results)

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Trích xuất feature importance từ mô hình."""
        if model_name is None:
            model_name = self.best_model_name
        model = self.models.get(model_name)
        if model is None:
            return pd.DataFrame()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return pd.DataFrame()

        fi_df = pd.DataFrame({
            "feature": range(len(importances)),
            "importance": importances,
        }).sort_values("importance", ascending=False)

        return fi_df

    def save_model(self, model_name: str = None, path: str = None):
        """Lưu mô hình."""
        if model_name is None:
            model_name = self.best_model_name
        if path is None:
            path = os.path.join(self.params["paths"]["models"], f"{model_name}.pkl")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.models[model_name], f)
        print(f"[supervised] Saved {model_name} to {path}")

    # ==================================================================
    # REGRESSION
    # ==================================================================
    def get_regressors(self) -> Dict[str, Any]:
        """Tạo các regressor cho dự đoán Tool wear."""
        seed = self.params["seed"]
        regressors = {
            "linear_regression": LinearRegression(),
            "random_forest_reg": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1,
            ),
            "gradient_boosting_reg": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=seed,
            ),
        }
        if HAS_XGBOOST:
            regressors["xgboost_reg"] = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=seed, verbosity=0,
            )
        return regressors

    def train_regressors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Huấn luyện tất cả regressor, đánh giá MAE/RMSE/R²."""
        regressors = self.get_regressors()
        results = []

        for name, model in regressors.items():
            print(f"\n[supervised] Training regressor: {name}...")
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            res = {
                "model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "train_time_s": round(train_time, 2),
            }
            results.append(res)
            self.models[name] = model
            self.results[name] = {**res, "y_pred": y_pred}
            print(f"  → MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        return pd.DataFrame(results).sort_values("RMSE")
