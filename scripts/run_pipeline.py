#!/usr/bin/env python
"""
run_pipeline.py  –  Chạy toàn bộ pipeline từ dòng lệnh (không cần notebook).

Usage
-----
    python scripts/run_pipeline.py            # chạy full
    python scripts/run_pipeline.py --step eda # chạy 1 bước
"""

import sys, os, argparse, warnings
from pathlib import Path

# Đảm bảo import được src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


def get_clustering_feature_columns(df_feat: pd.DataFrame, params: dict) -> list:
    """Use a compact operational feature set for clustering to improve cluster separability."""
    cfg_cols = params.get("mining", {}).get("clustering", {}).get("feature_subset", [])
    cols = [c for c in cfg_cols if c in df_feat.columns]
    if cols:
        return cols

    fallback = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temp_diff",
        "power",
        "torque_speed_ratio",
        "wear_torque",
        "air_temp_x_speed",
        "proc_temp_x_torque",
    ]
    return [c for c in fallback if c in df_feat.columns]

# ──────────────────────────────────────────────────────
# 1. EDA
# ──────────────────────────────────────────────────────
def step_eda():
    from src.data.loader import load_raw_data, get_data_summary, create_data_dictionary
    print("\n" + "=" * 60)
    print("STEP 1 ▸ Exploratory Data Analysis")
    print("=" * 60)
    df = load_raw_data()
    summary = get_data_summary(df)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    dd = create_data_dictionary(df)
    print("\nData Dictionary:")
    print(dd.to_string())
    print(f"\nTarget distribution:\n{df['Machine failure'].value_counts()}")
    print(f"Failure rate: {df['Machine failure'].mean():.2%}")
    return df


# ──────────────────────────────────────────────────────
# 2. Preprocessing & Feature Engineering
# ──────────────────────────────────────────────────────
def step_preprocess(df=None):
    from src.data.loader import load_raw_data, load_params
    from src.data.cleaner import DataCleaner
    from src.features.builder import FeatureBuilder

    print("\n" + "=" * 60)
    print("STEP 2 ▸ Preprocessing & Feature Engineering")
    print("=" * 60)

    if df is None:
        df = load_raw_data()

    params = load_params()

    # Clean
    cleaner = DataCleaner(params)
    df_clean = cleaner.fit_transform(df)
    stats_df = cleaner.get_comparison()
    print("  Cleaning summary:")
    print(stats_df.to_string(index=False))

    # Features
    builder = FeatureBuilder(params)
    df_feat = builder.build(df_clean)
    print(f"  Features engineered : {df_feat.shape[1]} columns")

    # Save
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_dir / "ai4i2020_processed.parquet", index=False)
    df_feat.to_csv(out_dir / "ai4i2020_processed.csv", index=False)
    print(f"  Saved to {out_dir}")
    return df_feat


# ──────────────────────────────────────────────────────
# 3. Mining / Clustering / Anomaly
# ──────────────────────────────────────────────────────
def step_mining(df_feat=None):
    from src.data.loader import load_processed_data, load_params
    from src.features.builder import FeatureBuilder
    from src.mining.association import AssociationMiner
    from src.mining.clustering import ClusterAnalyzer
    from src.mining.anomaly import AnomalyDetector

    print("\n" + "=" * 60)
    print("STEP 3 ▸ Association Rules · Clustering · Anomaly Detection")
    print("=" * 60)

    if df_feat is None:
        df_feat = load_processed_data()

    params = load_params()

    # ── Association Rules ──
    print("\n── Apriori Association Rules ──")
    builder = FeatureBuilder(params.get("feature_engineering", {}))
    df_binary = builder.get_apriori_features(df_feat, params)
    miner = AssociationMiner(params)
    freq, rules = miner.mine(df_binary)
    print(f"  Frequent itemsets : {len(freq)}")
    print(f"  Rules found       : {len(rules)}")
    fail_rules = miner.get_failure_rules()
    print(f"  Failure‑related   : {len(fail_rules)}")

    # Save association outputs for dashboard/API
    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if rules is not None and len(rules) > 0:
        rules_out = rules.copy()
        rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: ", ".join(sorted(map(str, list(x)))))
        rules_out["consequents"] = rules_out["consequents"].apply(lambda x: ", ".join(sorted(map(str, list(x)))))
        rules_out["rule"] = rules_out["antecedents"] + " -> " + rules_out["consequents"]
        rules_out["antecedent_len"] = rules_out["antecedents"].apply(lambda x: len([t for t in x.split(",") if t.strip()]))
        rules_out["consequent_len"] = rules_out["consequents"].apply(lambda x: len([t for t in x.split(",") if t.strip()]))
        rules_out = rules_out[["rule", "antecedents", "consequents", "support", "confidence", "lift"]]
        rules_out = rules_out.sort_values(["lift", "confidence", "support"], ascending=False)
        rules_out.to_csv(tables_dir / "association_rules.csv", index=False)

    # ── Clustering ──
    print("\n── Clustering ──")
    feat_cols = get_clustering_feature_columns(df_feat, params)
    print(f"  Clustering feature subset: {len(feat_cols)} columns")
    from sklearn.preprocessing import StandardScaler
    X_clust = StandardScaler().fit_transform(df_feat[feat_cols])

    ca = ClusterAnalyzer(params)
    ca.fit_kmeans(X_clust)
    ca.fit_dbscan(X_clust)
    ca.fit_hierarchical(X_clust)
    scores = ca.get_scores_table()
    print(scores.to_string())

    # Save clustering outputs for dashboard/API
    scores.to_csv(tables_dir / "clustering_comparison.csv", index=False)

    best_name, best_labels = ca.get_best_model(metric="silhouette")
    print(f"  Selected best clustering config: {best_name}")

    df_cluster = df_feat.copy()
    df_cluster["cluster"] = best_labels
    df_cluster_valid = df_cluster[df_cluster["cluster"] != -1]
    failure_by_cluster = df_cluster_valid.groupby("cluster").agg(
        count=("Machine failure", "count"),
        n_failures=("Machine failure", "sum"),
        failure_rate=("Machine failure", "mean"),
        avg_tool_wear=("Tool wear [min]", "mean"),
        avg_torque=("Torque [Nm]", "mean"),
        avg_speed=("Rotational speed [rpm]", "mean"),
    ).round(4)
    failure_by_cluster.to_csv(tables_dir / "cluster_failure_profiles.csv")
    print(f"  Saved clustering tables to {tables_dir}")

    # ── Anomaly Detection ──
    print("\n── Anomaly Detection ──")
    anomaly_cfg = {
        **params.get("mining", {}).get("anomaly", {}),
        "seed": params.get("seed", 42),
    }
    ad = AnomalyDetector(anomaly_cfg)
    contamination = float(df_feat["Machine failure"].mean())
    ad.fit_isolation_forest(X_clust, contamination=contamination)
    ad.fit_lof(X_clust, contamination=contamination)
    results = ad.compare_with_actual(df_feat["Machine failure"].values)
    results.to_csv(tables_dir / "anomaly_comparison.csv", index=False)
    print(results.to_string())

    return rules, scores


# ──────────────────────────────────────────────────────
# 4. Supervised Modelling
# ──────────────────────────────────────────────────────
def step_model(df_feat=None):
    from src.data.loader import load_processed_data, load_params
    from src.models.supervised import SupervisedTrainer
    from src.models.forecasting import TimeSeriesForecaster
    from src.mining.clustering import ClusterAnalyzer
    from src.evaluation.metrics import classification_metrics, compare_models
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import json

    print("\n" + "=" * 60)
    print("STEP 4 ▸ Supervised Modelling")
    print("=" * 60)

    if df_feat is None:
        df_feat = load_processed_data()

    params = load_params()
    info_path = ROOT / "data" / "processed" / "feature_info.json"
    if info_path.exists():
        with open(info_path) as f:
            feat_info = json.load(f)
        feature_cols = feat_info.get("feature_columns", feat_info.get("feature_cols", []))
    else:
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[feature_cols]
    y = df_feat["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    X_train_arr = X_train_s.values
    X_test_arr = X_test_s.values

    # ── Classification ──
    print("\n── Classification ──")
    trainer = SupervisedTrainer(params)
    clf_df = trainer.train_classifiers(X_train_arr, y_train, X_test_arr, y_test)
    cv_df = trainer.cross_validate(X_train_arr, y_train.values, cv=params.get("modeling", {}).get("classification", {}).get("cv_folds", 5))
    print(clf_df.to_string(index=False))

    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    clf_df.to_csv(tables_dir / "classification_results.csv", index=False)
    cv_df.to_csv(tables_dir / "cv_results.csv", index=False)

    # ── Error Analysis by Cluster & Product Type ──
    best_name = trainer.best_model_name
    y_pred_test = trainer.results.get(best_name, {}).get("y_pred")
    if y_pred_test is not None:
        eval_df = pd.DataFrame({
            "row_index": X_test.index,
            "y_true": y_test.values.astype(int),
            "y_pred": np.array(y_pred_test).astype(int),
        })

        # Decode product type from one-hot columns.
        type_h = df_feat["Type_H"] if "Type_H" in df_feat.columns else pd.Series(0, index=df_feat.index)
        type_m = df_feat["Type_M"] if "Type_M" in df_feat.columns else pd.Series(0, index=df_feat.index)
        type_l = df_feat["Type_L"] if "Type_L" in df_feat.columns else pd.Series(0, index=df_feat.index)
        type_series = np.where(type_h.values == 1, "H", np.where(type_m.values == 1, "M", np.where(type_l.values == 1, "L", "Unknown")))
        eval_df["product_type"] = pd.Series(type_series, index=df_feat.index).loc[eval_df["row_index"]].values

        # Attach cluster labels using the same clustering selection logic.
        feat_cols_cluster = get_clustering_feature_columns(df_feat, params)
        X_clust = StandardScaler().fit_transform(df_feat[feat_cols_cluster])
        ca = ClusterAnalyzer(params)
        ca.fit_kmeans(X_clust)
        ca.fit_dbscan(X_clust)
        ca.fit_hierarchical(X_clust)
        _, labels_all = ca.get_best_model(metric="silhouette")
        cluster_series = pd.Series(labels_all, index=df_feat.index)
        eval_df["cluster"] = cluster_series.loc[eval_df["row_index"]].values

        def grouped_error_table(df_in: pd.DataFrame, group_cols) -> pd.DataFrame:
            if isinstance(group_cols, str):
                group_cols = [group_cols]
            g = df_in.groupby(group_cols)
            out = g.apply(
                lambda s: pd.Series({
                    "n_samples": int(len(s)),
                    "actual_failures": int((s["y_true"] == 1).sum()),
                    "predicted_failures": int((s["y_pred"] == 1).sum()),
                    "tp": int(((s["y_true"] == 1) & (s["y_pred"] == 1)).sum()),
                    "fp": int(((s["y_true"] == 0) & (s["y_pred"] == 1)).sum()),
                    "fn": int(((s["y_true"] == 1) & (s["y_pred"] == 0)).sum()),
                    "tn": int(((s["y_true"] == 0) & (s["y_pred"] == 0)).sum()),
                })
            ).reset_index()

            out["fp_rate_in_group"] = (out["fp"] / (out["n_samples"] + 1e-10)).round(4)
            out["fn_rate_in_group"] = (out["fn"] / (out["n_samples"] + 1e-10)).round(4)
            out["fn_rate_on_actual_failures"] = (out["fn"] / (out["actual_failures"] + 1e-10)).round(4)
            return out.sort_values(["fn", "fp", "n_samples"], ascending=[False, False, False]).reset_index(drop=True)

        err_cluster = grouped_error_table(eval_df, "cluster")
        err_type = grouped_error_table(eval_df, "product_type")
        err_cluster_type = grouped_error_table(eval_df, ["cluster", "product_type"])
        err_cluster.to_csv(tables_dir / "error_by_cluster.csv", index=False)
        err_type.to_csv(tables_dir / "error_by_type.csv", index=False)
        err_cluster_type.to_csv(tables_dir / "error_by_cluster_type.csv", index=False)

    # ── Regression ──
    print("\n── Regression (target = Tool wear [min]) ──")
    if "Tool wear [min]" in df_feat.columns:
        y_reg = df_feat["Tool wear [min]"]
        leakage_markers = [
            "Tool wear [min]",
            "tw_bin_",
            "wear_torque",
            "_lag",
            "_rmean",
            "_rstd",
        ]
        reg_feats = [
            c for c in feature_cols
            if c != "Tool wear [min]" and not any(m in c for m in leakage_markers)
        ]
        reg_feats = [c for c in reg_feats if c in df_feat.columns]
        print(f"  Regression features after anti-leakage filter: {len(reg_feats)}")
        X_reg = df_feat[reg_feats]
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        reg_df = trainer.train_regressors(Xr_tr.values, yr_tr.values, Xr_te.values, yr_te.values)
        print(reg_df.to_string(index=False))
        reg_df.to_csv(tables_dir / "regression_results.csv", index=False)

    # ── Time Series ──
    print("\n── Time‑Series Forecasting ──")
    forecaster = TimeSeriesForecaster(params)
    if "Tool wear [min]" in df_feat.columns and "UDI" in df_feat.columns:
        ts_feature_cols = [c for c in df_feat.columns if "_lag" in c]
        ts_feature_cols = [c for c in ts_feature_cols if c in feature_cols]
        print(f"  Time-series lag features: {len(ts_feature_cols)}")
        train_ts_df, test_ts_df = forecaster.temporal_train_test_split(
            df_feat,
            target_col="Tool wear [min]",
            train_ratio=params.get("modeling", {}).get("time_series", {}).get("train_ratio", 0.8),
        )
        try:
            forecaster.fit_arima(train_ts_df["Tool wear [min]"], test_ts_df["Tool wear [min]"], order=tuple(params.get("modeling", {}).get("time_series", {}).get("arima_order", [2, 1, 2])))
        except Exception as e:
            print(f"  ARIMA skipped: {e}")
        forecaster.fit_lag_regression(train_ts_df, test_ts_df, target_col="Tool wear [min]", feature_cols=ts_feature_cols)
        ts_table = forecaster.get_results_table()
        print(ts_table.to_string(index=False))
        ts_table.to_csv(tables_dir / "timeseries_results.csv", index=False)

    return clf_df


# ──────────────────────────────────────────────────────
# 4b. Semi‑supervised
# ──────────────────────────────────────────────────────
def step_semi(df_feat=None):
    from src.data.loader import load_processed_data, load_params
    from src.models.semi_supervised import SemiSupervisedTrainer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import json

    print("\n" + "=" * 60)
    print("STEP 4b ▸ Semi‑supervised Learning")
    print("=" * 60)

    if df_feat is None:
        df_feat = load_processed_data()

    params = load_params()
    info_path = ROOT / "data" / "processed" / "feature_info.json"
    if info_path.exists():
        with open(info_path) as f:
            feat_info = json.load(f)
        feature_cols = feat_info.get("feature_columns", feat_info.get("feature_cols", []))
    else:
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[feature_cols].values
    y = df_feat["Machine failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=params.get("seed", 42), stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trainer = SemiSupervisedTrainer(params)
    results = trainer.run_all_experiments(X_train_scaled, y_train, X_test_scaled, y_test)
    print(results.to_string())

    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables_dir / "semi_supervised_results.csv", index=False)

    pseudo_risk_df = trainer.get_pseudo_label_risk_table()
    if not pseudo_risk_df.empty:
        pseudo_risk_df.to_csv(tables_dir / "pseudo_label_risk.csv", index=False)

    curve_df = trainer.get_learning_curve_data(X_train_scaled, y_train, X_test_scaled, y_test)
    curve_df.to_csv(tables_dir / "learning_curve.csv", index=False)
    return results


# ──────────────────────────────────────────────────────
# 5. Evaluation / Report
# ──────────────────────────────────────────────────────
def step_report():
    from src.evaluation.report import ReportGenerator

    print("\n" + "=" * 60)
    print("STEP 5 ▸ Final Report & Artifacts")
    print("=" * 60)

    rg = ReportGenerator(str(ROOT / "outputs"))
    rg.add_insight("Pipeline completed via run_pipeline.py – see notebooks for detailed visuals.")
    rg.save_all()
    print("  Artifacts saved to outputs/")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────
STEPS = {
    "eda": step_eda,
    "preprocess": step_preprocess,
    "mining": step_mining,
    "model": step_model,
    "semi": step_semi,
    "report": step_report,
}

def main():
    parser = argparse.ArgumentParser(description="Run the Data‑Mining pipeline.")
    parser.add_argument("--step", choices=list(STEPS.keys()),
                        help="Run a single step instead of the full pipeline.")
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        df = step_eda()
        df_feat = step_preprocess(df)
        step_mining(df_feat)
        step_model(df_feat)
        step_semi(df_feat)
        step_report()

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
