"""
builder.py – Feature Engineering cho dataset AI4I 2020 Predictive Maintenance.

Các nhóm đặc trưng:
  1. Đặc trưng gốc đã chuẩn hoá
  2. Rời rạc hoá trạng thái máy (bin sensor/setting)
  3. Đặc trưng dẫn xuất: temp_diff, power, torque/speed ratio
  4. Lag features (coi UDI ~ thời gian)
  5. Rolling statistics
  6. Đặc trưng tương tác
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureBuilder:
    """
    Tạo đặc trưng cho bài toán predictive maintenance.
    Gọi build() để tạo toàn bộ đặc trưng.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.feature_names: List[str] = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline tạo toàn bộ đặc trưng."""
        df = df.copy()
        
        # Sắp xếp theo UDI (coi như time index)
        if "UDI" in df.columns:
            df = df.sort_values("UDI").reset_index(drop=True)

        fe_params = self.params.get("feature_engineering", {})

        # 1. Đặc trưng dẫn xuất
        df = self._create_derived_features(df, fe_params)

        # 2. Rời rạc hoá tool wear
        if fe_params.get("bin_tool_wear", False):
            df = self._bin_tool_wear(df, fe_params)

        # 3. Lag features
        if fe_params.get("create_lag_features", False):
            df = self._create_lag_features(df, fe_params)

        # 4. Rolling features
        if fe_params.get("create_rolling_features", False):
            df = self._create_rolling_features(df, fe_params)

        # 5. Interaction features
        df = self._create_interaction_features(df)

        # Ghi lại danh sách feature
        exclude = (
            self.params["data"]["id_columns"]
            + [self.params["data"]["target"]]
            + self.params["data"]["failure_types"]
        )
        self.feature_names = [c for c in df.columns if c not in exclude]

        print(f"[builder] Created {len(self.feature_names)} features total")
        return df

    # ------------------------------------------------------------------
    def _create_derived_features(self, df: pd.DataFrame, fe_params: Dict) -> pd.DataFrame:
        """Tạo đặc trưng dẫn xuất từ sensor/setting."""

        # Temperature difference
        if fe_params.get("create_temp_diff", False):
            if "Process temperature [K]" in df.columns and "Air temperature [K]" in df.columns:
                df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
                print("[builder] Created: temp_diff")

        # Power = Torque × Rotational speed (× 2π/60 cho Watt, nhưng giữ scale đơn giản)
        if fe_params.get("create_power", False):
            if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
                df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
                print("[builder] Created: power (Watt)")

        # Torque per speed ratio
        if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
            df["torque_speed_ratio"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1e-8)
            print("[builder] Created: torque_speed_ratio")

        # Wear × Torque (liên quan OSF)
        if "Tool wear [min]" in df.columns and "Torque [Nm]" in df.columns:
            df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
            print("[builder] Created: wear_torque")

        return df

    def _bin_tool_wear(self, df: pd.DataFrame, fe_params: Dict) -> pd.DataFrame:
        """Rời rạc hoá Tool wear thành bins."""
        bins = fe_params.get("tool_wear_bins", [0, 50, 100, 150, 200, 300])
        labels = fe_params.get("tool_wear_labels", ["very_low", "low", "medium", "high", "very_high"])

        if "Tool wear [min]" in df.columns:
            df["tool_wear_bin"] = pd.cut(
                df["Tool wear [min]"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            # One-hot encode bins
            bin_dummies = pd.get_dummies(df["tool_wear_bin"], prefix="tw_bin", dtype=int)
            df = pd.concat([df, bin_dummies], axis=1)
            df = df.drop(columns=["tool_wear_bin"])
            print(f"[builder] Binned Tool wear into {len(labels)} categories")

        return df

    def _create_lag_features(self, df: pd.DataFrame, fe_params: Dict) -> pd.DataFrame:
        """Tạo lag features (UDI đã sort ≈ time index)."""
        windows = fe_params.get("lag_windows", [1, 3, 5])
        sensor_cols = self.params["data"]["numeric_features"]
        sensor_cols = [c for c in sensor_cols if c in df.columns]

        created = 0
        for col in sensor_cols:
            for lag in windows:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
                created += 1

        # Fill NaN từ lag bằng giá trị đầu tiên
        df = df.fillna(method="bfill")
        print(f"[builder] Created {created} lag features (windows: {windows})")
        return df

    def _create_rolling_features(self, df: pd.DataFrame, fe_params: Dict) -> pd.DataFrame:
        """Tạo rolling mean/std features."""
        windows = fe_params.get("rolling_windows", [5, 10, 20])
        sensor_cols = self.params["data"]["numeric_features"]
        sensor_cols = [c for c in sensor_cols if c in df.columns]

        created = 0
        for col in sensor_cols:
            for w in windows:
                df[f"{col}_rmean{w}"] = df[col].rolling(window=w, min_periods=1).mean()
                df[f"{col}_rstd{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
                created += 2

        print(f"[builder] Created {created} rolling features (windows: {windows})")
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng tương tác giữa các sensor."""
        interactions = [
            ("Air temperature [K]", "Rotational speed [rpm]", "air_temp_x_speed"),
            ("Process temperature [K]", "Torque [Nm]", "proc_temp_x_torque"),
        ]

        for col1, col2, name in interactions:
            if col1 in df.columns and col2 in df.columns:
                df[name] = df[col1] * df[col2]

        return df

    def get_feature_names(self) -> List[str]:
        """Trả về danh sách tên feature đã tạo."""
        return self.feature_names

    def get_apriori_features(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Tạo DataFrame nhị phân cho Apriori:
        Rời rạc hoá tất cả sensor thành high/low/normal rồi one-hot encode.
        """
        apriori_df = pd.DataFrame()

        # Type
        if "Type" in df.columns:
            for t in ["L", "M", "H"]:
                apriori_df[f"Type_{t}"] = (df["Type"] == t).astype(int)
        # Dùng onehot encoded nếu có
        for col in df.columns:
            if col.startswith("Type_"):
                apriori_df[col] = df[col]

        # Numeric features → chia thành Low/Normal/High
        numeric_cols = params["data"]["numeric_features"]
        for col in numeric_cols:
            if col not in df.columns:
                continue
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            apriori_df[f"{col}_low"] = (df[col] < q25).astype(int)
            apriori_df[f"{col}_normal"] = ((df[col] >= q25) & (df[col] <= q75)).astype(int)
            apriori_df[f"{col}_high"] = (df[col] > q75).astype(int)

        # Failure columns
        failure_types = params["data"]["failure_types"]
        for col in failure_types:
            if col in df.columns:
                apriori_df[col] = df[col]

        # Machine failure
        target = params["data"]["target"]
        if target in df.columns:
            apriori_df[target] = df[target]

        apriori_df = apriori_df.astype(bool)
        print(f"[builder] Created Apriori binary DataFrame: {apriori_df.shape}")
        return apriori_df
