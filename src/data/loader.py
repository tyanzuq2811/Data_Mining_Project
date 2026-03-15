"""
loader.py – Đọc dữ liệu, kiểm tra schema, và cung cấp dữ liệu cho pipeline.
Dataset: UCI AI4I 2020 Predictive Maintenance
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def load_params(config_path: str = "configs/params.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def load_raw_data(params: Optional[Dict] = None, path: Optional[str] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu gốc từ CSV.
    
    Parameters
    ----------
    params : dict, optional
        Tham số từ params.yaml
    path : str, optional
        Đường dẫn trực tiếp đến file CSV
    
    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        if params is None:
            params = load_params()
        path = params["paths"]["raw_data"]
    
    df = pd.read_csv(path)
    print(f"[loader] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame, params: Optional[Dict] = None) -> bool:
    """
    Kiểm tra schema dữ liệu: tên cột, kiểu dữ liệu, giá trị target hợp lệ.
    
    Returns True nếu hợp lệ, raise ValueError nếu không.
    """
    if params is None:
        params = load_params()
    
    expected_cols = (
        params["data"]["id_columns"]
        + params["data"]["categorical_features"]
        + params["data"]["numeric_features"]
        + [params["data"]["target"]]
        + params["data"]["failure_types"]
    )
    
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"[loader] Missing columns: {missing}")
    
    # Kiểm tra target values
    target = params["data"]["target"]
    unique_vals = df[target].unique()
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"[loader] Target '{target}' has unexpected values: {unique_vals}")
    
    # Kiểm tra kiểu dữ liệu numeric
    for col in params["data"]["numeric_features"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"[loader] Column '{col}' is not numeric: {df[col].dtype}")
    
    print("[loader] Schema validation passed ✓")
    return True


def load_processed_data(params: Optional[Dict] = None) -> pd.DataFrame:
    """Đọc dữ liệu đã tiền xử lý từ parquet (hoặc CSV fallback)."""
    if params is None:
        params = load_params()
    
    parquet_path = params["paths"]["processed_data"]
    csv_path = params["paths"]["processed_csv"]
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"[loader] Loaded processed data (parquet): {df.shape}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[loader] Loaded processed data (CSV): {df.shape}")
    else:
        raise FileNotFoundError(
            f"[loader] Processed data not found at {parquet_path} or {csv_path}. "
            "Run preprocessing first."
        )
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Tạo summary thống kê của DataFrame."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_stats": df.describe().to_dict(),
    }
    return summary


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo Data Dictionary cho dataset AI4I 2020.
    
    Returns
    -------
    pd.DataFrame với các cột: Column, Type, Description, Range/Values, Missing
    """
    descriptions = {
        "UDI": "Unique identifier (1–10000), có thể coi là time index",
        "Product ID": "Mã sản phẩm gồm chữ Type (L/M/H) + số serial",
        "Type": "Chất lượng sản phẩm: L (Low, 60%), M (Medium, 30%), H (High, 10%)",
        "Air temperature [K]": "Nhiệt độ không khí (Kelvin), sinh từ N(300, 2) + random walk",
        "Process temperature [K]": "Nhiệt độ gia công (Kelvin), = Air temp + 10 + noise",
        "Rotational speed [rpm]": "Tốc độ quay (vòng/phút), sinh từ power law ~2860 rpm",
        "Torque [Nm]": "Mô-men xoắn (Newton-metre), ~40 Nm ± 10",
        "Tool wear [min]": "Thời gian mài mòn dụng cụ (phút): H +5, M +3, L +2 min/obs",
        "Machine failure": "Nhãn mục tiêu: 1 = lỗi, 0 = bình thường (3.39% failure)",
        "TWF": "Tool Wear Failure – lỗi do mài mòn dụng cụ (200–240 min)",
        "HDF": "Heat Dissipation Failure – lỗi tản nhiệt (ΔT < 8.6K & speed < 1380)",
        "PWF": "Power Failure – lỗi công suất (power ngoài [3500, 9000]W)",
        "OSF": "Overstrain Failure – lỗi quá tải (tool_wear × torque > threshold per Type)",
        "RNF": "Random Failure – lỗi ngẫu nhiên (0.1% xác suất)",
    }
    
    records = []
    for col in df.columns:
        desc = descriptions.get(col, "N/A")
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            range_val = f"[{df[col].min()}, {df[col].max()}]"
        else:
            range_val = str(df[col].unique()[:10].tolist())
        
        records.append({
            "Column": col,
            "Type": dtype,
            "Description": desc,
            "Range/Values": range_val,
            "Missing": missing,
        })
    
    return pd.DataFrame(records)
