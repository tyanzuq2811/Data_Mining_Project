"""
cleaner.py – Xử lý thiếu, outlier, encoding cơ bản cho dataset AI4I 2020.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataCleaner:
    """
    Pipeline làm sạch dữ liệu:
      1. Xử lý missing values
      2. Xử lý outliers
      3. Xử lý duplicates
      4. Encoding biến phân loại
      5. Chuẩn hoá (scaling)
    """

    def __init__(self, params: Dict):
        self.params = params
        self.scaler = None
        self.encoder = None
        self._scaler_fitted = False
        self._encoder_fitted = False
        self.stats_before = {}
        self.stats_after = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit và transform toàn bộ pipeline trên training data."""
        df = df.copy()
        self._collect_stats(df, stage="before")

        df = self.handle_missing(df)
        df = self.handle_duplicates(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df, fit=True)
        df = self.scale_features(df, fit=True)

        self._collect_stats(df, stage="after")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu mới dùng scaler/encoder đã fit."""
        df = df.copy()
        df = self.handle_missing(df)
        df = self.handle_duplicates(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df

    # ------------------------------------------------------------------
    # 1. Missing values
    # ------------------------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý giá trị thiếu – dataset AI4I thường không có missing."""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
            print(f"[cleaner] Filled {missing_count} missing values")
        else:
            print("[cleaner] No missing values found ✓")
        return df

    # ------------------------------------------------------------------
    # 2. Duplicates
    # ------------------------------------------------------------------
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ bản ghi trùng lặp (ngoại trừ cột UDI/Product ID)."""
        id_cols = self.params["data"].get("id_columns", [])
        check_cols = [c for c in df.columns if c not in id_cols]
        n_dup = df.duplicated(subset=check_cols).sum()
        if n_dup > 0:
            df = df.drop_duplicates(subset=check_cols, keep="first").reset_index(drop=True)
            print(f"[cleaner] Removed {n_dup} duplicate rows")
        else:
            print("[cleaner] No duplicates found ✓")
        return df

    # ------------------------------------------------------------------
    # 3. Outliers
    # ------------------------------------------------------------------
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phát hiện và clip outlier bằng IQR hoặc z-score."""
        if not self.params["preprocessing"].get("handle_outliers", False):
            return df

        method = self.params["preprocessing"].get("outlier_method", "iqr")
        threshold = self.params["preprocessing"].get("outlier_threshold", 1.5)
        numeric_cols = self.params["data"]["numeric_features"]
        total_clipped = 0

        for col in numeric_cols:
            if col not in df.columns:
                continue
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
            else:
                continue

            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                df[col] = df[col].clip(lower, upper)
                total_clipped += n_outliers

        print(f"[cleaner] Clipped {total_clipped} outlier values ({method})")
        return df

    # ------------------------------------------------------------------
    # 4. Encoding
    # ------------------------------------------------------------------
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Mã hoá biến phân loại (One-Hot hoặc Label Encoding)."""
        cat_cols = self.params["data"].get("categorical_features", [])
        encoding = self.params["preprocessing"].get("encoding", "onehot")
        cat_cols = [c for c in cat_cols if c in df.columns]

        if not cat_cols:
            return df

        if encoding == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
            print(f"[cleaner] One-Hot encoded: {cat_cols}")
        elif encoding == "label":
            if fit:
                self.encoder = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoder[col] = le
                self._encoder_fitted = True
            else:
                for col in cat_cols:
                    df[col] = self.encoder[col].transform(df[col].astype(str))
            print(f"[cleaner] Label encoded: {cat_cols}")
        elif encoding == "ordinal":
            mapping = {"L": 0, "M": 1, "H": 2}
            for col in cat_cols:
                if col == "Type":
                    df[col] = df[col].map(mapping)
            print(f"[cleaner] Ordinal encoded: {cat_cols}")

        return df

    # ------------------------------------------------------------------
    # 5. Scaling
    # ------------------------------------------------------------------
    def scale_features(
        self, df: pd.DataFrame, fit: bool = True,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Chuẩn hoá các đặc trưng số."""
        scaler_type = self.params["preprocessing"].get("scaler", "standard")
        numeric_cols = self.params["data"]["numeric_features"]
        # Chỉ scale các cột còn tồn tại (sau encoding có thể thay đổi)
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        if exclude_cols:
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if not numeric_cols:
            return df

        if fit:
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()

            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self._scaler_fitted = True
            print(f"[cleaner] Fitted & applied {scaler_type} scaler on {len(numeric_cols)} cols")
        else:
            if self._scaler_fitted:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
                print(f"[cleaner] Applied {scaler_type} scaler (transform only)")
            else:
                print("[cleaner] Warning: scaler not fitted, skipping scaling")

        return df

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _collect_stats(self, df: pd.DataFrame, stage: str = "before"):
        """Thu thập thống kê trước/sau xử lý."""
        stats = {
            "shape": df.shape,
            "missing_total": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.value_counts().to_dict(),
        }
        if stage == "before":
            self.stats_before = stats
        else:
            self.stats_after = stats

    def get_comparison(self) -> pd.DataFrame:
        """So sánh thống kê trước và sau xử lý."""
        rows = []
        for key in ["shape", "missing_total", "duplicates"]:
            rows.append({
                "Metric": key,
                "Before": str(self.stats_before.get(key, "N/A")),
                "After": str(self.stats_after.get(key, "N/A")),
            })
        return pd.DataFrame(rows)
