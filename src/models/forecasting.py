"""
forecasting.py – Time series forecasting cho Tool wear / failure.

Giả định: UDI ≈ time index (dataset không có timestamp).
Chia train/test theo thứ tự quan sát (không shuffle).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


class TimeSeriesForecaster:
    """
    Forecasting Tool wear bằng ARIMA + lag-features regression.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.models = {}
        self.results = {}

    def temporal_train_test_split(
        self, df: pd.DataFrame, target_col: str = "Tool wear [min]", train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia train/test theo thứ tự quan sát (giữ temporal order).
        """
        df = df.sort_values("UDI").reset_index(drop=True)
        n_train = int(len(df) * train_ratio)
        train = df.iloc[:n_train].copy()
        test = df.iloc[n_train:].copy()
        print(f"[forecasting] Temporal split: train={len(train)}, test={len(test)}")
        return train, test

    def fit_arima(
        self,
        train_series: pd.Series,
        test_series: pd.Series,
        order: Tuple = (2, 1, 2),
    ) -> Dict:
        """
        Fit ARIMA model cho chuỗi thời gian.
        """
        if not HAS_ARIMA:
            print("[forecasting] statsmodels not installed, skipping ARIMA")
            return {}

        print(f"[forecasting] Fitting ARIMA{order}...")
        model = ARIMA(train_series.values, order=order)
        fitted = model.fit()

        # Forecast
        n_test = len(test_series)
        forecast = fitted.forecast(steps=n_test)

        mae = mean_absolute_error(test_series.values, forecast)
        rmse = np.sqrt(mean_squared_error(test_series.values, forecast))

        self.models["arima"] = fitted
        result = {
            "model": f"ARIMA{order}",
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "AIC": round(fitted.aic, 2),
            "BIC": round(fitted.bic, 2),
        }
        self.results["arima"] = {**result, "forecast": forecast, "actual": test_series.values}

        print(f"[forecasting] ARIMA MAE={mae:.4f}, RMSE={rmse:.4f}")
        return result

    def fit_lag_regression(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "Tool wear [min]",
        feature_cols: list = None,
    ) -> Dict:
        """
        Regression dùng lag features (đã tạo trong FeatureBuilder).
        Chia theo thứ tự quan sát.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if feature_cols is None:
            feature_cols = [c for c in train_df.columns
                           if c not in ["UDI", "Product ID", target_col, "Machine failure"]
                           and not c.startswith(("TWF", "HDF", "PWF", "OSF", "RNF"))]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=self.params["seed"],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        self.models["lag_regression"] = model
        result = {
            "model": "GBR_lag_features",
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }
        self.results["lag_regression"] = {**result, "y_pred": y_pred, "y_test": y_test}

        print(f"[forecasting] Lag Regression MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        return result

    def get_results_table(self) -> pd.DataFrame:
        """Bảng so sánh kết quả forecasting."""
        rows = []
        for name, res in self.results.items():
            rows.append({k: v for k, v in res.items() if k not in ["forecast", "actual", "y_pred", "y_test"]})
        return pd.DataFrame(rows)
