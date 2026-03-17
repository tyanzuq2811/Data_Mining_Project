"""
FastAPI Backend — Predictive Maintenance API
Serves ML model predictions + data for the Dash dashboard.
"""

import os, sys, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List

# ── paths ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH   = ROOT / "outputs" / "models" / "gradient_boosting.pkl"
DATA_PATH    = ROOT / "data" / "processed" / "ai4i2020_processed.parquet"
FEAT_PATH    = ROOT / "data" / "processed" / "feature_info.json"
TABLES_DIR   = ROOT / "outputs" / "tables"
FIGURES_DIR  = ROOT / "outputs" / "figures"
REPORTS_DIR  = ROOT / "outputs" / "reports"
STATIC_DIR   = ROOT / "app" / "static"
TEMPLATES_DIR = ROOT / "app" / "templates"

# ── load artefacts ────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEAT_PATH, "r") as f:
    feat_info = json.load(f)

df_processed = pd.read_parquet(DATA_PATH)

# Fit scaler on training data for API predictions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

feature_cols = feat_info["feature_columns"] if "feature_columns" in feat_info else feat_info.get("feature_cols", [])
feature_cols = [c for c in feature_cols if c in df_processed.columns]

X_all = df_processed[feature_cols]
y_all = df_processed["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
scaler = StandardScaler()
scaler.fit(X_train)

_association_cache = None


def _split_items(value: str) -> list:
    return [s.strip() for s in str(value).split(",") if s and str(s).strip()]


def _apply_association_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only many->one actionable rules to avoid leakage and improve operational relevance."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["rule", "antecedents", "consequents", "support", "confidence", "lift"])

    out = df.copy()

    if "antecedents" in out.columns:
        out = out[out["antecedents"].apply(lambda x: len(_split_items(x)) >= 2)]

    if "consequents" in out.columns:
        out = out[out["consequents"].astype(str).str.strip() == "Machine failure"]

    forbidden = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
    if "antecedents" in out.columns:
        out = out[out["antecedents"].apply(lambda x: len(set(_split_items(x)) & forbidden) == 0)]

    for c in ["support", "confidence", "lift"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=[c for c in ["support", "confidence", "lift"] if c in out.columns])
    out = out.sort_values(["lift", "confidence", "support"], ascending=False)
    return out.reset_index(drop=True)


def _compute_association_rules() -> pd.DataFrame:
    """Compute association rules from processed data and return a web-friendly table."""
    from src.data.loader import load_params
    from src.features.builder import FeatureBuilder
    from src.mining.association import AssociationMiner

    params = load_params(str(ROOT / "configs" / "params.yaml"))
    builder = FeatureBuilder(params)
    binary_df = builder.get_apriori_features(df_processed, params)

    miner = AssociationMiner(params)
    _, rules = miner.mine(binary_df)
    if rules is None or rules.empty:
        return pd.DataFrame(columns=["rule", "antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(map(str, list(x)))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(map(str, list(x)))))
    rules["rule"] = rules["antecedents"] + " -> " + rules["consequents"]

    cols = ["rule", "antecedents", "consequents", "support", "confidence", "lift"]
    out = rules[cols].sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return _apply_association_policy(out)

# ── FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="API dự đoán lỗi máy móc sử dụng AI4I 2020 dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


# ── Pydantic schemas ─────────────────────────────────
class SensorInput(BaseModel):
    air_temperature: float = Field(..., description="Air temperature [K]", ge=290, le=310)
    process_temperature: float = Field(..., description="Process temperature [K]", ge=300, le=320)
    rotational_speed: float = Field(..., description="Rotational speed [rpm]", ge=1000, le=3000)
    torque: float = Field(..., description="Torque [Nm]", ge=0, le=80)
    tool_wear: float = Field(..., description="Tool wear [min]", ge=0, le=260)
    product_type: str = Field("M", description="Product type: L / M / H")


class PredictionResult(BaseModel):
    prediction: int
    failure_probability: float
    risk_level: str
    risk_factors: List[str]
    recommendation: str


# ── helper: build full feature vector from 5 sensor values ──
def build_feature_vector(inp: SensorInput) -> np.ndarray:
    """Create the full feature vector (with derived + lag/rolling = median fill)."""
    d = {}
    d["Air temperature [K]"] = inp.air_temperature
    d["Process temperature [K]"] = inp.process_temperature
    d["Rotational speed [rpm]"] = inp.rotational_speed
    d["Torque [Nm]"] = inp.torque
    d["Tool wear [min]"] = inp.tool_wear

    # one-hot Type
    d["Type_H"] = 1 if inp.product_type == "H" else 0
    d["Type_L"] = 1 if inp.product_type == "L" else 0
    d["Type_M"] = 1 if inp.product_type == "M" else 0

    # derived
    d["temp_diff"] = inp.process_temperature - inp.air_temperature
    d["power"] = inp.torque * inp.rotational_speed * 2 * np.pi / 60
    d["torque_speed_ratio"] = inp.torque / max(inp.rotational_speed, 1)
    d["wear_torque"] = inp.tool_wear * inp.torque

    # tool wear bins
    tw = inp.tool_wear
    d["tw_bin_very_low"]  = 1 if tw < 50 else 0
    d["tw_bin_low"]       = 1 if 50 <= tw < 100 else 0
    d["tw_bin_medium"]    = 1 if 100 <= tw < 150 else 0
    d["tw_bin_high"]      = 1 if 150 <= tw < 200 else 0
    d["tw_bin_very_high"] = 1 if tw >= 200 else 0

    # For lag/rolling features: use the current value as proxy (no history)
    for base_col in ["Air temperature [K]", "Process temperature [K]",
                      "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
        val = d[base_col]
        for lag in [1, 3, 5, 10]:
            d[f"{base_col}_lag{lag}"] = val
        for w in [5, 10, 20]:
            d[f"{base_col}_rmean{w}"] = val
            d[f"{base_col}_rstd{w}"] = 0.0  # no variation for single point

    # interaction
    d["air_temp_x_speed"] = inp.air_temperature * inp.rotational_speed
    d["proc_temp_x_torque"] = inp.process_temperature * inp.torque

    # build vector in correct order
    vec = []
    for col in feature_cols:
        vec.append(d.get(col, 0.0))

    return np.array(vec).reshape(1, -1)


def assess_risk(inp: SensorInput, prob: float):
    """Đánh giá mức độ rủi ro và các yếu tố ảnh hưởng."""
    factors = []
    if inp.tool_wear > 200:
        factors.append("Độ mài dao cụ cực cao (>200 phút)")
    elif inp.tool_wear > 150:
        factors.append("Độ mài dao cụ cao (>150 phút)")

    temp_diff = inp.process_temperature - inp.air_temperature
    if temp_diff < 8.6 and inp.rotational_speed < 1380:
        factors.append("Nguy cơ tản nhiệt kém (chênh lệch nhiệt < 8.6K, tốc độ thấp)")

    power = inp.torque * inp.rotational_speed * 2 * np.pi / 60
    if power < 3500 or power > 9000:
        factors.append(f"Công suất ngoài phạm vi an toàn ({power:.0f}W)")

    if inp.torque > 60:
        factors.append("Mô-men xoắn rất cao (>60 Nm)")

    if inp.tool_wear * inp.torque > 10000:
        factors.append("Tương tác Mài×Mô-men cao — nguy cơ quá tải")

    if prob >= 0.7:
        level = "CRITICAL"
        rec = "DỪNG MÁY NGAY — Kiểm tra và bảo trì khẩn cấp!"
    elif prob >= 0.4:
        level = "HIGH"
        rec = "Lên kế hoạch bảo trì trong 24h. Giảm tải máy nếu có thể."
    elif prob >= 0.15:
        level = "MEDIUM"
        rec = "Theo dõi chặt chẽ. Kiểm tra dao cụ và thông số nhiệt."
    else:
        level = "LOW"
        rec = "Máy hoạt động bình thường. Tiếp tục theo dõi định kỳ."

    if not factors:
        factors.append("Không phát hiện yếu tố rủi ro cụ thể")

    return level, factors, rec


# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main frontend page."""
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/predict", response_model=PredictionResult)
async def predict(sensor: SensorInput):
    """Predict machine failure from sensor readings."""
    X = build_feature_vector(sensor)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])
    level, factors, rec = assess_risk(sensor, prob)

    return PredictionResult(
        prediction=pred,
        failure_probability=round(prob, 4),
        risk_level=level,
        risk_factors=factors,
        recommendation=rec,
    )


@app.get("/api/data/summary")
async def data_summary():
    """Return dataset summary stats."""
    total = len(df_processed)
    failures = int(df_processed["Machine failure"].sum())
    return {
        "total_records": total,
        "failures": failures,
        "normal": total - failures,
        "failure_rate": round(failures / total * 100, 2),
        "features_count": len(feature_cols),
    }


@app.get("/api/data/distribution")
async def data_distribution():
    """Return sensor value distributions."""
    sensors = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "proc_temp",
        "Rotational speed [rpm]": "rot_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
    }
    result = {}
    for col, key in sensors.items():
        if col in df_processed.columns:
            vals = df_processed[col]
            result[key] = {
                "mean": round(float(vals.mean()), 2),
                "std": round(float(vals.std()), 2),
                "min": round(float(vals.min()), 2),
                "max": round(float(vals.max()), 2),
                "q25": round(float(vals.quantile(0.25)), 2),
                "q75": round(float(vals.quantile(0.75)), 2),
            }
    return result


@app.get("/api/results/classification")
async def classification_results():
    """Return classification comparison table."""
    path = TABLES_DIR / "classification_results.csv"
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/api/results/association")
async def association_results(limit: int = 20):
    """Return top association rules for dashboard."""
    global _association_cache
    path = TABLES_DIR / "association_rules.csv"

    if _association_cache is None:
        if path.exists():
            df = pd.read_csv(path).fillna(0)
        else:
            df = _compute_association_rules()
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        _association_cache = _apply_association_policy(df)

    df = _association_cache.copy()

    limit = max(5, min(int(limit), 100))
    return df.head(limit).to_dict(orient="records")


@app.get("/api/results/clustering")
async def clustering_results():
    """Return top clustering results."""
    path = TABLES_DIR / "clustering_comparison.csv"
    raw = pd.read_csv(path).fillna(0)

    # Normalize possible column naming variants (e.g. MODEL/model, DAVIES_BOULDIN/davies_bouldin).
    col_map = {c.strip().lower(): c for c in raw.columns}

    def pick(*cands):
        for c in cands:
            if c in col_map:
                return col_map[c]
        return None

    c_model = pick("model")
    c_sil = pick("silhouette", "silhouette_score")
    c_db = pick("davies_bouldin", "davies-bouldin", "dbi")
    c_ch = pick("calinski_harabasz", "calinski-harabasz", "chi")

    if not all([c_model, c_sil, c_db, c_ch]):
        return []

    df = pd.DataFrame({
        "model": raw[c_model].astype(str),
        "silhouette": pd.to_numeric(raw[c_sil], errors="coerce").fillna(0.0),
        "davies_bouldin": pd.to_numeric(raw[c_db], errors="coerce").fillna(0.0),
        "calinski_harabasz": pd.to_numeric(raw[c_ch], errors="coerce").fillna(0.0),
    })

    df = df.sort_values("silhouette", ascending=False).head(10).reset_index(drop=True)
    return df.to_dict(orient="records")


@app.get("/api/results/cluster_ranking")
@app.get("/api/results/cluster-ranking")
async def cluster_ranking():
    """Return cluster risk-priority ranking to support maintenance actions."""
    path = TABLES_DIR / "cluster_failure_profiles.csv"
    raw = pd.read_csv(path).fillna(0)

    col_map = {c.strip().lower(): c for c in raw.columns}

    def pick(*cands):
        for c in cands:
            if c in col_map:
                return col_map[c]
        return None

    c_cluster = pick("cluster")
    c_count = pick("count", "n_samples")
    c_fail_rate = pick("failure_rate", "fail_rate")
    c_fail_n = pick("n_failures", "failures")
    c_wear = pick("avg_tool_wear", "tool_wear_mean")
    c_torque = pick("avg_torque", "torque_mean")

    if not all([c_cluster, c_count, c_fail_rate, c_fail_n, c_wear, c_torque]):
        return []

    df = pd.DataFrame({
        "cluster": pd.to_numeric(raw[c_cluster], errors="coerce").fillna(-1).astype(int),
        "count": pd.to_numeric(raw[c_count], errors="coerce").fillna(0),
        "n_failures": pd.to_numeric(raw[c_fail_n], errors="coerce").fillna(0),
        "failure_rate": pd.to_numeric(raw[c_fail_rate], errors="coerce").fillna(0.0),
        "avg_tool_wear": pd.to_numeric(raw[c_wear], errors="coerce").fillna(0.0),
        "avg_torque": pd.to_numeric(raw[c_torque], errors="coerce").fillna(0.0),
    })

    def norm(s):
        mn, mx = float(s.min()), float(s.max())
        if mx <= mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    # Priority score (0-100): high failure rate + severe wear + high torque + enough support.
    score = (
        0.60 * norm(df["failure_rate"]) +
        0.20 * norm(df["avg_tool_wear"]) +
        0.15 * norm(df["avg_torque"]) +
        0.05 * norm(df["count"])
    ) * 100
    df["priority_score"] = score.round(1)

    def to_level(v):
        if v >= 70:
            return "Rất cao"
        if v >= 45:
            return "Cao"
        if v >= 25:
            return "Trung bình"
        return "Thấp"

    df["priority_level"] = df["priority_score"].map(to_level)
    df = df.sort_values(["priority_score", "failure_rate", "avg_tool_wear"], ascending=[False, False, False])
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.to_dict(orient="records")


@app.get("/api/results/semi_supervised")
async def semi_supervised_results():
    """Return semi-supervised results."""
    path = TABLES_DIR / "semi_supervised_results.csv"
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/pseudo_label_risk")
async def pseudo_label_risk_results():
    """Return pseudo-label false alarm and miss analysis by labeled ratio."""
    path = TABLES_DIR / "pseudo_label_risk.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/error_by_cluster")
async def error_by_cluster_results():
    """Return FP/FN analysis grouped by cluster on test split."""
    path = TABLES_DIR / "error_by_cluster.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/error_by_type")
async def error_by_type_results():
    """Return FP/FN analysis grouped by product type on test split."""
    path = TABLES_DIR / "error_by_type.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/error_by_cluster_type")
async def error_by_cluster_type_results():
    """Return FP/FN analysis grouped by cluster x product type on test split."""
    path = TABLES_DIR / "error_by_cluster_type.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/regression")
async def regression_results():
    """Return regression results."""
    path = TABLES_DIR / "regression_results.csv"
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/anomaly")
async def anomaly_results():
    """Return anomaly detection results."""
    path = TABLES_DIR / "anomaly_comparison.csv"
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/timeseries")
async def timeseries_results():
    """Return timeseries results."""
    path = TABLES_DIR / "timeseries_results.csv"
    df = pd.read_csv(path).fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/results/insights")
async def insights():
    """Return actionable insights."""
    path = REPORTS_DIR / "insights.txt"
    text = path.read_text(encoding="utf-8")
    # Parse numbered insights
    lines = text.strip().split("\n")
    insight_list = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            insight_list.append(line)
    return {"insights": insight_list}


@app.get("/api/figures")
async def list_figures():
    """List available figures."""
    figs = [f.name for f in FIGURES_DIR.glob("*.png")]
    return {"figures": sorted(figs)}


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_probe():
    """Silence harmless Chrome DevTools probe to reduce noisy 404 logs."""
    return {}


@app.get("/api/data/scatter/{feature_x}/{feature_y}")
async def scatter_data(feature_x: str, feature_y: str):
    """Return scatter plot data for two features."""
    col_map = {
        "air_temp": "Air temperature [K]",
        "proc_temp": "Process temperature [K]",
        "rot_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]",
    }
    x_col = col_map.get(feature_x, feature_x)
    y_col = col_map.get(feature_y, feature_y)

    if x_col not in df_processed.columns or y_col not in df_processed.columns:
        return {"error": "Invalid column name"}

    # Sample 2000 points for performance
    sample = df_processed.sample(min(2000, len(df_processed)), random_state=42)
    return {
        "x": sample[x_col].tolist(),
        "y": sample[y_col].tolist(),
        "failure": sample["Machine failure"].tolist(),
        "x_label": x_col,
        "y_label": y_col,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
