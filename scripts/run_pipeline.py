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
    cleaner = DataCleaner(params.get("preprocessing", {}))
    df_clean = cleaner.fit_transform(df)
    stats = cleaner.get_stats()
    print(f"  Rows after cleaning : {stats['after']['n_rows']}")

    # Features
    builder = FeatureBuilder(params.get("feature_engineering", {}))
    df_feat = builder.fit_transform(df_clean)
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
    df_binary = builder.get_apriori_features(df_feat)
    miner = AssociationMiner(params.get("mining", {}).get("apriori", {}))
    freq, rules = miner.mine(df_binary)
    print(f"  Frequent itemsets : {len(freq)}")
    print(f"  Rules found       : {len(rules)}")
    fail_rules = miner.get_failure_rules()
    print(f"  Failure‑related   : {len(fail_rules)}")

    # ── Clustering ──
    print("\n── Clustering ──")
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    feat_cols = [c for c in numeric_cols if c not in exclude]
    from sklearn.preprocessing import StandardScaler
    X_clust = StandardScaler().fit_transform(df_feat[feat_cols])

    ca = ClusterAnalyzer(params.get("mining", {}).get("clustering", {}))
    ca.fit_kmeans(X_clust)
    ca.fit_dbscan(X_clust)
    ca.fit_hierarchical(X_clust)
    scores = ca.get_scores_table()
    print(scores.to_string())

    # ── Anomaly Detection ──
    print("\n── Anomaly Detection ──")
    ad = AnomalyDetector(params.get("mining", {}).get("anomaly", {}))
    results = ad.compare_with_actual(X_clust, df_feat["Machine failure"].values)
    print(results.to_string())

    return rules, scores


# ──────────────────────────────────────────────────────
# 4. Supervised Modelling
# ──────────────────────────────────────────────────────
def step_model(df_feat=None):
    from src.data.loader import load_processed_data, load_params
    from src.models.supervised import SupervisedTrainer
    from src.models.forecasting import TimeSeriesForecaster
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
        feature_cols = feat_info.get("feature_columns", [])
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

    # ── Classification ──
    print("\n── Classification ──")
    trainer = SupervisedTrainer(params.get("classification", {}))
    clf_results = trainer.train_classifiers(X_train_s, y_train, X_test_s, y_test)
    rows = []
    for name, info in clf_results.items():
        m = classification_metrics(y_test, info["y_pred"], info.get("y_proba"))
        rows.append({"model": name, **m})
    clf_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print(clf_df.to_string(index=False))

    # ── Regression ──
    print("\n── Regression (target = Tool wear [min]) ──")
    if "Tool wear [min]" in df_feat.columns:
        y_reg = df_feat["Tool wear [min]"]
        reg_feats = [c for c in feature_cols if c != "Tool wear [min]"]
        reg_feats = [c for c in reg_feats if c in df_feat.columns]
        X_reg = df_feat[reg_feats]
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        reg_results = trainer.train_regressors(Xr_tr, yr_tr, Xr_te, yr_te)
        for name, info in reg_results.items():
            print(f"  {name}  MAE={info['mae']:.3f}  RMSE={info['rmse']:.3f}  R2={info['r2']:.3f}")

    # ── Time Series ──
    print("\n── Time‑Series Forecasting ──")
    ts_cfg = params.get("time_series", {})
    forecaster = TimeSeriesForecaster(ts_cfg)
    ts_target = df_feat.get("rolling_failure_rate_20",
                             df_feat.get("Machine failure"))
    if ts_target is not None:
        train_ts, test_ts = forecaster.temporal_train_test_split(ts_target)
        try:
            forecaster.fit_arima(train_ts, test_ts)
        except Exception as e:
            print(f"  ARIMA skipped: {e}")
        forecaster.fit_lag_regression(ts_target)
        ts_table = forecaster.get_results_table()
        print(ts_table.to_string())

    return clf_df


# ──────────────────────────────────────────────────────
# 4b. Semi‑supervised
# ──────────────────────────────────────────────────────
def step_semi(df_feat=None):
    from src.data.loader import load_processed_data, load_params
    from src.models.semi_supervised import SemiSupervisedTrainer
    from sklearn.preprocessing import StandardScaler
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
        feature_cols = feat_info.get("feature_columns", [])
    else:
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[feature_cols].values
    y = df_feat["Machine failure"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    semi_cfg = params.get("semi_supervised", {})
    trainer = SemiSupervisedTrainer(semi_cfg)
    results = trainer.run_all_experiments(X_scaled, y)
    print(results.to_string())
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
