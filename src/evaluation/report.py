"""
report.py – Tổng hợp bảng/biểu đồ kết quả cho báo cáo cuối cùng.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional


class ReportGenerator:
    """
    Tổng hợp kết quả từ các module, xuất bảng/hình cho báo cáo.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.tables = {}
        self.insights = []

    def add_table(self, name: str, df: pd.DataFrame):
        """Thêm bảng kết quả."""
        self.tables[name] = df

    def add_insight(self, insight: str):
        """Thêm insight/khuyến nghị."""
        self.insights.append(insight)

    def save_tables(self, output_dir: str = None):
        """Lưu tất cả bảng ra CSV."""
        if output_dir is None:
            output_dir = self.params["paths"]["tables"]
        os.makedirs(output_dir, exist_ok=True)

        for name, df in self.tables.items():
            path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(path, index=True)
            print(f"[report] Saved table: {path}")

    def generate_insights_summary(self) -> str:
        """Tổng hợp insights thành text."""
        if not self.insights:
            return "Không có insight nào được ghi nhận."

        lines = ["=" * 60, "INSIGHTS & KHUYẾN NGHỊ HÀNH ĐỘNG", "=" * 60]
        for i, insight in enumerate(self.insights, 1):
            lines.append(f"\n{i}. {insight}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def save_insights(self, output_dir: str = None):
        """Lưu insights ra file text."""
        if output_dir is None:
            output_dir = self.params["paths"]["reports"]
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, "insights.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_insights_summary())
        print(f"[report] Saved insights: {path}")

    def create_model_comparison_table(
        self, classification_results: pd.DataFrame, regression_results: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Tạo bảng so sánh tổng hợp."""
        self.add_table("classification_comparison", classification_results)
        if regression_results is not None:
            self.add_table("regression_comparison", regression_results)
        return classification_results
