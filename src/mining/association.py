"""
association.py – Luật kết hợp & Pattern Mining (Apriori) cho Predictive Maintenance.

Mục tiêu: tìm combo điều kiện máy liên quan lỗi.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules


class AssociationMiner:
    """
    Apriori mining trên dữ liệu đã rời rạc hoá.
    Tìm frequent itemsets và luật kết hợp liên quan failure modes.
    """

    def __init__(self, params: Dict):
        self.params = params if isinstance(params, dict) else {}
        # Accept either full params (with mining.apriori) or apriori-only config.
        if isinstance(self.params, dict) and "mining" in self.params:
            mining_cfg = self.params.get("mining", {}).get("apriori", {})
            self.data_cfg = self.params.get("data", {})
        else:
            mining_cfg = self.params
            self.data_cfg = {}
        self.min_support = mining_cfg.get("min_support", 0.01)
        self.min_confidence = mining_cfg.get("min_confidence", 0.5)
        self.min_lift = mining_cfg.get("min_lift", 1.5)
        self.max_len = mining_cfg.get("max_len", 4)
        self.policy = mining_cfg.get("policy", {})
        self.frequent_itemsets = None
        self.rules = None

    def mine(self, binary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chạy Apriori và sinh luật kết hợp.

        Parameters
        ----------
        binary_df : pd.DataFrame
            DataFrame nhị phân (0/1) đã rời rạc hoá từ FeatureBuilder.get_apriori_features()

        Returns
        -------
        (frequent_itemsets, rules) : tuple of DataFrames
        """
        print(f"[association] Running Apriori (min_support={self.min_support}, "
              f"min_confidence={self.min_confidence}, min_lift={self.min_lift})")

        # Chạy Apriori
        self.frequent_itemsets = apriori(
            binary_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )
        print(f"[association] Found {len(self.frequent_itemsets)} frequent itemsets")

        if len(self.frequent_itemsets) == 0:
            print("[association] No frequent itemsets found. Try lowering min_support.")
            self.rules = pd.DataFrame()
            return self.frequent_itemsets, self.rules

        # Sinh luật kết hợp
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
            num_itemsets=len(self.frequent_itemsets),
        )

        # Lọc theo lift
        self.rules = self.rules[self.rules["lift"] >= self.min_lift]

        # Enforce rule direction and anti-leakage policy for actionable maintenance rules.
        self.rules = self.filter_actionable_rules(self.rules)
        self.rules = self.rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

        print(f"[association] Found {len(self.rules)} rules (lift >= {self.min_lift})")
        return self.frequent_itemsets, self.rules

    def filter_actionable_rules(self, rules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply policy to keep operationally useful rules:
        - many -> one (antecedent length >= 2, consequent length == 1)
        - consequents limited to target/failure outcomes
        - remove leakage-like antecedents containing failure labels
        """
        if rules_df is None or len(rules_df) == 0:
            return pd.DataFrame(columns=getattr(rules_df, "columns", []))

        policy = self.policy if isinstance(self.policy, dict) else {}
        rule_mode = policy.get("rule_mode", "many_to_one")
        min_ant_len = int(policy.get("min_antecedent_len", 2))
        max_con_len = int(policy.get("max_consequent_len", 1))

        target = self.data_cfg.get("target", "Machine failure")
        failure_types = self.data_cfg.get("failure_types", [])
        allowed_consequents = policy.get("allowed_consequents", [target] + list(failure_types))
        allowed_consequents = set(map(str, allowed_consequents))

        exclude_failure_in_ant = bool(policy.get("exclude_failure_in_antecedents", True))
        forbidden_ant = set([target] + list(failure_types)) if exclude_failure_in_ant else set()

        out = rules_df.copy()

        if rule_mode == "many_to_one":
            out = out[out["antecedents"].apply(lambda s: len(s) >= min_ant_len)]
            out = out[out["consequents"].apply(lambda s: len(s) <= max_con_len)]

        out = out[
            out["consequents"].apply(
                lambda s: len(set(map(str, s)) - allowed_consequents) == 0 and len(s) > 0
            )
        ]

        if forbidden_ant:
            out = out[
                out["antecedents"].apply(
                    lambda s: len(set(map(str, s)) & forbidden_ant) == 0
                )
            ]

        return out.reset_index(drop=True)

    def get_failure_rules(self, failure_col: str = "Machine failure") -> pd.DataFrame:
        """Lọc luật có consequent chứa failure."""
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()

        mask = self.rules["consequents"].apply(lambda x: failure_col in x)
        failure_rules = self.rules[mask].copy()
        print(f"[association] {len(failure_rules)} rules with '{failure_col}' as consequent")
        return failure_rules

    def get_failure_type_rules(self, failure_types: List[str]) -> Dict[str, pd.DataFrame]:
        """Lọc luật cho từng loại lỗi."""
        result = {}
        if self.rules is None or len(self.rules) == 0:
            return result

        for ft in failure_types:
            mask = self.rules["consequents"].apply(lambda x: ft in x)
            ft_rules = self.rules[mask].copy()
            if len(ft_rules) > 0:
                result[ft] = ft_rules
                print(f"[association] {len(ft_rules)} rules for {ft}")

        return result

    def get_top_rules(self, n: int = 20, sort_by: str = "lift") -> pd.DataFrame:
        """Top N luật theo metric chỉ định."""
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        return self.rules.nlargest(n, sort_by)

    def rules_to_text(self, rules_df: pd.DataFrame = None, top_n: int = 10) -> List[str]:
        """Chuyển luật thành dạng text dễ đọc."""
        if rules_df is None:
            rules_df = self.rules
        if rules_df is None or len(rules_df) == 0:
            return []

        texts = []
        for _, row in rules_df.head(top_n).iterrows():
            ant = ", ".join(sorted(row["antecedents"]))
            con = ", ".join(sorted(row["consequents"]))
            texts.append(
                f"IF ({ant}) → THEN ({con})  "
                f"[support={row['support']:.4f}, "
                f"confidence={row['confidence']:.3f}, "
                f"lift={row['lift']:.2f}]"
            )
        return texts
