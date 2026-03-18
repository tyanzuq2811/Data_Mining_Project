"""
Dash Dashboard - Bảng Điều Khiển Bảo Trì Dự Đoán
Cổng 8050 - Hỗ trợ chế độ Tối / Sáng
Static layout — no dynamic re-rendering of page structure.
"""

from pathlib import Path
import math
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, callback, no_update

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_params, load_raw_data, create_data_dictionary

# ── Dữ liệu ──────────────────────────────────────
ROOT   = PROJECT_ROOT
TABLES = ROOT / "outputs" / "tables"
DATA   = ROOT / "data" / "processed"

clf   = pd.read_csv(TABLES / "classification_results.csv")
cv    = pd.read_csv(TABLES / "cv_results.csv")
clust = pd.read_csv(TABLES / "clustering_comparison.csv")
anom  = pd.read_csv(TABLES / "anomaly_comparison.csv")
reg   = pd.read_csv(TABLES / "regression_results.csv")
semi  = pd.read_csv(TABLES / "semi_supervised_results.csv")
df    = pd.read_parquet(DATA / "ai4i2020_processed.parquet")

params = load_params(str(ROOT / "configs" / "params.yaml"))
try:
    df_raw = load_raw_data(params=params)
except Exception as exc:
    raw_csv = ROOT / "data" / "raw" / "ai4i2020.csv"
    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv)
    else:
        raise RuntimeError("Khong the tai du lieu raw cho EDA.") from exc

EDA_DF = df_raw

SENSOR_COLS = [c for c in [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
] if c in EDA_DF.columns]

FEATURE_LABELS = {
    "Air temperature [K]": "Nhiệt độ không khí [K]",
    "Process temperature [K]": "Nhiệt độ quy trình [K]",
    "Rotational speed [rpm]": "Tốc độ quay [rpm]",
    "Torque [Nm]": "Mô-men xoắn [Nm]",
    "Tool wear [min]": "Độ mòn dụng cụ [phút]",
}

MODEL_LABELS = {
    "gradient_boosting": "Gradient Boosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "random_forest_reg": "Random Forest Regressor",
    "linear_regression": "Linear Regression",
    "gradient_boosting_reg": "Gradient Boosting Regressor",
    "xgboost_reg": "XGBoost Regressor",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor (LOF)",
    "supervised_only": "Chỉ học có nhãn",
    "self_training": "Tự huấn luyện (Self-Training)",
    "label_spreading": "Lan truyền nhãn (Label Spreading)",
}


def feature_label(col):
    return FEATURE_LABELS.get(col, col)


def model_label(name):
    s = str(name)
    return MODEL_LABELS.get(s, s.replace("_", " ").title())


def model_family(name):
    s = str(name).lower()
    if s.startswith("kmeans"):
        return "K-Means"
    if s.startswith("dbscan"):
        return "DBSCAN"
    if s.startswith("hac"):
        return "HAC"
    return "Khác"

N_FEATURES = len([c for c in df.columns if c != "Machine failure" and
                   c not in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]])

EDA_NUMERIC_COLS = SENSOR_COLS if SENSOR_COLS else [
    c for c in EDA_DF.select_dtypes(include="number").columns if c != "Machine failure"
]

EDA_SUMMARY = {
    "n_rows": int(len(EDA_DF)),
    "n_cols": int(EDA_DF.shape[1]),
    "missing_values": int(EDA_DF.isna().sum().sum()),
    "duplicate_rows": int(EDA_DF.duplicated().sum()),
    "n_variables": int(max(EDA_DF.shape[1] - 1, 0)),
    "failure_rate": float(EDA_DF["Machine failure"].mean() * 100) if "Machine failure" in EDA_DF.columns else float("nan"),
}


_dict_src = create_data_dictionary(EDA_DF)
DATA_DICTIONARY = pd.DataFrame({
    "TÊN BIẾN": _dict_src["Column"],
    "TÊN TIẾNG VIỆT": _dict_src["Column"].map(feature_label),
    "KIỂU DỮ LIỆU": _dict_src["Type"],
    "MÔ TẢ": _dict_src["Description"],
})


def dictionary_table(theme="dark"):
    t = T(theme)
    head = html.Tr([
        html.Th("TÊN BIẾN", style={"textAlign": "left", "padding": "10px", "color": t["font_sub"], "fontSize": "12px"}),
        html.Th("TÊN TIẾNG VIỆT", style={"textAlign": "left", "padding": "10px", "color": t["font_sub"], "fontSize": "12px"}),
        html.Th("KIỂU DỮ LIỆU", style={"textAlign": "left", "padding": "10px", "color": t["font_sub"], "fontSize": "12px"}),
        html.Th("MÔ TẢ", style={"textAlign": "left", "padding": "10px", "color": t["font_sub"], "fontSize": "12px"}),
    ])

    rows = []
    for i, row in DATA_DICTIONARY.iterrows():
        row_bg = "rgba(255,255,255,0.02)" if i % 2 else "rgba(255,255,255,0)"
        rows.append(html.Tr([
            html.Td(html.Span(str(row["TÊN BIẾN"]), style={
                "display": "inline-block",
                "padding": "2px 8px",
                "borderRadius": "6px",
                "backgroundColor": t["tab_sel_bg"],
                "color": t["accent"],
                "fontFamily": "monospace",
                "fontSize": "12px",
            }), style={"padding": "10px", "borderTop": f"1px solid {t['card_bdr']}"}),
            html.Td(str(row["TÊN TIẾNG VIỆT"]), style={"padding": "10px", "borderTop": f"1px solid {t['card_bdr']}"}),
            html.Td(html.Span(str(row["KIỂU DỮ LIỆU"]), style={
                "display": "inline-block",
                "padding": "2px 8px",
                "borderRadius": "6px",
                "backgroundColor": "rgba(255,153,0,0.15)",
                "color": "#ff9900",
                "fontFamily": "monospace",
                "fontSize": "12px",
            }), style={"padding": "10px", "borderTop": f"1px solid {t['card_bdr']}"}),
            html.Td(str(row["MÔ TẢ"]), style={"padding": "10px", "borderTop": f"1px solid {t['card_bdr']}"}),
        ], style={"backgroundColor": row_bg}))

    return html.Div([
        html.H4("Mô tả biến (Data Dictionary)", style={"margin": "0 0 12px", "color": t["accent"]}),
        html.Div(style={"overflowX": "auto"}, children=[
            html.Table([
                html.Thead(head),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "15px"}),
        ]),
    ])

# ==================================================
# THEME
# ==================================================
DARK = dict(
    bg="#060d17", font="#b3ecff", font_sub="#b3ecff55", accent="#00bfff",
    card="linear-gradient(145deg,rgba(10,22,40,.93),rgba(13,31,53,.93))",
    card_bdr="rgba(0,191,255,.2)",
    grid="rgba(0,191,255,.07)", zero="rgba(0,191,255,.13)",
    hover_bg="#0a1628", hover_fc="#e0f7ff",
    title_shadow="0 0 10px rgba(0,191,255,.53)",
    tab_bg="#060d17", tab_c="#b3ecff66", tab_bdr="#00bfff22",
    tab_sel_bg="#00bfff22", tab_sel_bdr="#00bfff",
    kpi=["#00bfff","#ff6b6b","#00ff88","#ffd60a"],
    bar=[("rgba(0,191,255,.6)","#00bfff"),("rgba(0,255,136,.6)","#00ff88"),
         ("rgba(255,214,10,.53)","#ffd60a"),("rgba(255,107,107,.53)","#ff6b6b")],
    seq=["#00bfff","#00e5ff","#00ff88","#ffd60a","#ff6b6b","#a855f7","#f97316","#06b6d4"],
    semi_c={"supervised_only":"#00bfff","self_training":"#00ff88","label_spreading":"#ffd60a"},
    pos=("rgba(0,191,255,.53)","#00bfff"), neg=("rgba(255,107,107,.53)","#ff6b6b"),
    hist_n=("rgba(0,191,255,.27)","#00bfff"), hist_f=("rgba(255,45,85,.27)","#ff2d55"),
    scat_n="rgba(0,191,255,.27)", scat_f="#ff2d55",
    dd_bg="#0a1628", dd_c="#b3ecff", dd_bdr="#00bfff33",
    toggle_label="☀️ Sáng",
)
LIGHT = dict(
    bg="#f0f4f8", font="#1a202c", font_sub="#71809688", accent="#0077b6",
    card="linear-gradient(145deg,#fff,#f8f9fa)",
    card_bdr="rgba(0,0,0,.10)",
    grid="rgba(0,0,0,.06)", zero="rgba(0,0,0,.10)",
    hover_bg="#fff", hover_fc="#1a202c",
    title_shadow="none",
    tab_bg="#f0f4f8", tab_c="#718096", tab_bdr="rgba(0,0,0,.10)",
    tab_sel_bg="rgba(0,119,182,.13)", tab_sel_bdr="#0077b6",
    kpi=["#0077b6","#e53e3e","#38a169","#d69e2e"],
    bar=[("rgba(0,119,182,.7)","#0077b6"),("rgba(56,161,105,.7)","#38a169"),
         ("rgba(214,158,46,.65)","#d69e2e"),("rgba(229,62,62,.65)","#e53e3e")],
    seq=["#0077b6","#0096c7","#38a169","#d69e2e","#e53e3e","#805ad5","#dd6b20","#0891b2"],
    semi_c={"supervised_only":"#0077b6","self_training":"#38a169","label_spreading":"#d69e2e"},
    pos=("rgba(0,119,182,.65)","#0077b6"), neg=("rgba(229,62,62,.65)","#e53e3e"),
    hist_n=("rgba(0,119,182,.30)","#0077b6"), hist_f=("rgba(229,62,62,.30)","#e53e3e"),
    scat_n="rgba(0,119,182,.35)", scat_f="#e53e3e",
    dd_bg="#fff", dd_c="#1a202c", dd_bdr="rgba(0,0,0,.15)",
    toggle_label="🌙 Tối",
)

def T(theme): return LIGHT if theme == "light" else DARK

def _lo(t, title=None, **kw):
    """Build plotly layout dict; title passed as dict to avoid duplicate-kwarg."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["font"], family="Inter", size=14),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(gridcolor=t["grid"], zerolinecolor=t["zero"], tickfont=dict(size=12)),
        yaxis=dict(gridcolor=t["grid"], zerolinecolor=t["zero"], tickfont=dict(size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13)),
        hoverlabel=dict(bgcolor=t["hover_bg"], bordercolor=t["accent"],
                        font_color=t["hover_fc"], font_size=13),
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=18, color=t["accent"]))
    d.update(kw)
    return d


# ==================================================
# CHART BUILDERS (all accept theme string)
# ==================================================

def fig_clf(theme="dark"):
    t = T(theme)
    model_names = clf["model"].map(model_label)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=clf["f1"], name="F1",
        marker_color=t["bar"][0][0], marker_line=dict(color=t["bar"][0][1], width=1.5),
        text=clf["f1"].round(3), textposition="auto", textfont=dict(size=13),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: F1 Score<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Cân bằng giữa Precision và Recall, càng cao càng tốt."
            "<extra></extra>"
        )))
    fig.add_trace(go.Bar(x=model_names, y=clf["pr_auc"], name="PR-AUC",
        marker_color=t["bar"][1][0], marker_line=dict(color=t["bar"][1][1], width=1.5),
        text=clf["pr_auc"].round(3), textposition="auto", textfont=dict(size=13),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: PR-AUC<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Chất lượng bắt lỗi trên dữ liệu mất cân bằng, càng cao càng tốt."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "So sánh phân loại: F1 & PR-AUC",
        barmode="group",
        xaxis_title="Mô hình phân loại",
        yaxis_title="Điểm đánh giá (0-1)",
    ))
    return fig

def fig_cv(theme="dark"):
    t = T(theme)
    model_names = cv["model"].map(model_label)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=cv["cv_f1_mean"], name="F1 trung bình CV",
        error_y=dict(type="data", array=cv["cv_f1_std"], visible=True, color=t["bar"][0][1]),
        marker_color=t["bar"][0][0], marker_line=dict(color=t["bar"][0][1], width=1.5),
        text=cv["cv_f1_mean"].round(3), textposition="auto", textfont=dict(size=13),
        customdata=cv[["cv_f1_std"]].to_numpy(),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: F1 Cross-Validation (5-fold)<br>"
            "Giá trị trung bình: %{y:.4f}<br>"
            "Độ lệch chuẩn: %{customdata[0]:.4f}<br>"
            "Diễn giải ngắn: Điểm cao và độ lệch chuẩn thấp nghĩa là mô hình ổn định."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "Cross-Validation 5-Fold (F1 ± Độ lệch chuẩn)",
        xaxis_title="Mô hình phân loại",
        yaxis_title="F1 trung bình qua 5 fold",
    ))
    return fig

def fig_pr(theme="dark"):
    t = T(theme)
    model_names = clf["model"].map(model_label)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=clf["precision"], name="Precision",
        marker_color=t["bar"][2][0], marker_line=dict(color=t["bar"][2][1], width=1),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: Precision<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Tỷ lệ cảnh báo đúng trong tất cả cảnh báo đã đưa ra."
            "<extra></extra>"
        )))
    fig.add_trace(go.Bar(x=model_names, y=clf["recall"], name="Recall",
        marker_color=t["bar"][3][0], marker_line=dict(color=t["bar"][3][1], width=1),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: Recall<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Khả năng bắt được lỗi thật, càng cao càng ít bỏ sót."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "Precision vs Recall",
        barmode="group",
        xaxis_title="Mô hình phân loại",
        yaxis_title="Điểm đánh giá (0-1)",
    ))
    return fig

def fig_time(theme="dark"):
    t = T(theme)
    model_names = clf["model"].map(model_label)
    fig = go.Figure(go.Bar(y=model_names, x=clf["train_time_s"], orientation="h",
        marker_color=t["seq"][:len(clf)],
        text=clf["train_time_s"].round(2), textposition="auto",
        hovertemplate=(
            "Mô hình: %{y}<br>"
            "Tên chỉ số: Thời gian huấn luyện<br>"
            "Giá trị: %{x:.2f} giây<br>"
            "Diễn giải ngắn: Thời gian thấp hơn giúp triển khai và lặp thử nghiệm nhanh hơn."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "Thời gian huấn luyện (giây)",
        margin=dict(l=160, r=30, t=50, b=50),
        xaxis_title="Thời gian huấn luyện (giây)",
        yaxis_title="Mô hình phân loại",
    ))
    return fig

def fig_cluster(theme="dark"):
    t = T(theme)
    top = clust.sort_values("silhouette", ascending=False).head(10)
    model_names = top["model"].map(model_label)
    c = [t["pos"][0] if s > 0 else t["neg"][0] for s in top["silhouette"]]
    b = [t["pos"][1] if s > 0 else t["neg"][1] for s in top["silhouette"]]
    fig = go.Figure(go.Bar(y=model_names, x=top["silhouette"], orientation="h",
        marker_color=c, marker_line=dict(color=b, width=1.5),
        text=top["silhouette"].round(3), textposition="auto",
        hovertemplate=(
            "Cấu hình: %{y}<br>"
            "Tên chỉ số: Silhouette Score<br>"
            "Giá trị: %{x:.4f}<br>"
            "Diễn giải ngắn: Cụm càng tách biệt và cô đặc thì điểm càng cao."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "Silhouette Score — Phân cụm (Top 10)",
        margin=dict(l=170, r=30, t=50, b=50),
        yaxis_autorange="reversed",
        xaxis_title="Silhouette score (cao hơn là tốt hơn)",
        yaxis_title="Cấu hình mô hình phân cụm",
    ))
    return fig


def fig_cluster_tradeoff(theme="dark"):
    t = T(theme)
    d = clust.copy()
    d["model_name"] = d["model"].map(model_label)
    d["family"] = d["model"].map(model_family)

    # Size marker by Calinski-Harabasz on log scale so outliers don't dominate.
    ch_log = d["calinski_harabasz"].clip(lower=0).map(lambda x: math.log1p(x))
    ch_min, ch_max = ch_log.min(), ch_log.max()
    if ch_max > ch_min:
        sizes = 10 + (ch_log - ch_min) * 16 / (ch_max - ch_min)
    else:
        sizes = pd.Series(14, index=d.index)

    color_map = {
        "K-Means": t["seq"][0],
        "DBSCAN": t["seq"][2],
        "HAC": t["seq"][4],
        "Khác": t["seq"][6],
    }

    fig = go.Figure()
    for fam in ["DBSCAN", "K-Means", "HAC", "Khác"]:
        s = d[d["family"] == fam]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["davies_bouldin"],
            y=s["silhouette"],
            mode="markers",
            name=fam,
            marker=dict(
                size=sizes.loc[s.index],
                color=color_map[fam],
                line=dict(color=t["accent"], width=0.8),
                opacity=0.85,
            ),
            customdata=s[["model_name", "calinski_harabasz"]].to_numpy(),
            hovertemplate=(
                "Cấu hình: %{customdata[0]}<br>"
                "Tên chỉ số: Davies-Bouldin<br>"
                "Giá trị: %{x:.4f} (thấp hơn là tốt hơn)<br>"
                "Tên chỉ số: Silhouette<br>"
                "Giá trị: %{y:.4f} (cao hơn là tốt hơn)<br>"
                "Tên chỉ số: Calinski-Harabasz<br>"
                "Giá trị: %{customdata[1]:.2f} (cao hơn thường tốt hơn)"
                "<extra></extra>"
            ),
        ))

    best = d.loc[d["silhouette"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["davies_bouldin"]],
        y=[best["silhouette"]],
        mode="markers+text",
        name="Điểm Silhouette cao nhất",
        text=["Best"],
        textposition="top center",
        marker=dict(symbol="star", size=18, color=t["accent"], line=dict(color=t["font"], width=1.2)),
        hovertemplate=(
            "Cấu hình tốt nhất theo Silhouette: " + model_label(best["model"]) + "<br>"
            f"Silhouette: {best['silhouette']:.4f}<br>"
            f"Davies-Bouldin: {best['davies_bouldin']:.4f}<br>"
            f"Calinski-Harabasz: {best['calinski_harabasz']:.2f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(**_lo(
        t,
        "Bản đồ chất lượng cụm: Silhouette vs Davies-Bouldin",
        xaxis_title="Davies-Bouldin (thấp hơn là tốt hơn)",
        yaxis_title="Silhouette (cao hơn là tốt hơn)",
    ))
    fig.add_hline(y=d["silhouette"].median(), line_dash="dot", line_color=t["grid"])
    fig.add_vline(x=d["davies_bouldin"].median(), line_dash="dot", line_color=t["grid"])
    return fig

def fig_anom(theme="dark"):
    t = T(theme)
    method_names = anom["Method"].map(model_label)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Số lượng phát hiện", "F1 Score"])
    fig.add_trace(go.Bar(x=method_names, y=anom["Anomalies Detected"],
        marker_color=[t["bar"][0][0], t["bar"][1][0]],
        marker_line=dict(color=[t["bar"][0][1], t["bar"][1][1]], width=1.5),
        showlegend=False,
        hovertemplate=(
            "Phương pháp: %{x}<br>"
            "Tên chỉ số: Số lượng bất thường phát hiện<br>"
            "Giá trị: %{y:.0f} mẫu<br>"
            "Diễn giải ngắn: Càng cao nghĩa là phương pháp đánh dấu nhiều điểm nghi ngờ hơn."
            "<extra></extra>"
        )), 1, 1)
    fig.add_trace(go.Bar(x=method_names, y=anom["F1"],
        marker_color=[t["bar"][2][0], t["bar"][3][0]],
        marker_line=dict(color=[t["bar"][2][1], t["bar"][3][1]], width=1.5),
        text=anom["F1"].round(4), textposition="auto", showlegend=False,
        hovertemplate=(
            "Phương pháp: %{x}<br>"
            "Tên chỉ số: F1 Score (Anomaly)<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Cân bằng giữa bắt đúng và tránh cảnh báo nhầm."
            "<extra></extra>"
        )), 1, 2)
    fig.update_layout(**_lo(t, "So sánh phát hiện bất thường"))
    fig.update_xaxes(title_text="Phương pháp", row=1, col=1)
    fig.update_yaxes(title_text="Số mẫu bị phát hiện là bất thường", row=1, col=1)
    fig.update_xaxes(title_text="Phương pháp", row=1, col=2)
    fig.update_yaxes(title_text="F1 score (0-1)", row=1, col=2)
    return fig

def fig_reg(theme="dark"):
    t = T(theme)
    d = reg.sort_values(["MAE", "RMSE"], ascending=[True, True]).copy()
    model_names = d["model"].map(model_label)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=d["MAE"], name="MAE",
        marker_color=t["pos"][0], marker_line=dict(color=t["pos"][1], width=1.5),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: MAE<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Sai số tuyệt đối trung bình, càng thấp càng tốt."
            "<extra></extra>"
        )))
    fig.add_trace(go.Bar(x=model_names, y=d["RMSE"], name="RMSE",
        marker_color=t["neg"][0], marker_line=dict(color=t["neg"][1], width=1.5),
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: RMSE<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: Phạt mạnh lỗi lớn, càng thấp càng an toàn cho dự báo."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        "Hồi quy: MAE & RMSE",
        barmode="group",
        xaxis_title="Mô hình hồi quy",
        yaxis_title="Sai số (càng thấp càng tốt)",
    ))
    return fig


def fig_reg_r2_time(theme="dark"):
    t = T(theme)
    d = reg.sort_values("train_time_s", ascending=True).copy()
    model_names = d["model"].map(model_label)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["R2", "Thời gian huấn luyện (giây)"])

    fig.add_trace(go.Bar(
        x=model_names,
        y=d["R2"],
        marker_color=[t["neg"][0] if v < 0 else t["pos"][0] for v in d["R2"]],
        marker_line=dict(color=t["accent"], width=1.2),
        text=d["R2"].round(3),
        textposition="auto",
        showlegend=False,
        hovertemplate=(
            "Mô hình: %{x}<br>"
            "Tên chỉ số: R2<br>"
            "Giá trị: %{y:.4f}<br>"
            "Diễn giải ngắn: R2 âm nghĩa là kém hơn cả dự báo trung bình."
            "<extra></extra>"
        ),
    ), 1, 1)

    fig.add_trace(go.Bar(
        y=model_names,
        x=d["train_time_s"],
        orientation="h",
        marker_color=t["bar"][0][0],
        marker_line=dict(color=t["bar"][0][1], width=1.2),
        text=d["train_time_s"].round(2),
        textposition="auto",
        showlegend=False,
        hovertemplate=(
            "Mô hình: %{y}<br>"
            "Tên chỉ số: Thời gian huấn luyện<br>"
            "Giá trị: %{x:.2f} giây<br>"
            "Diễn giải ngắn: Thời gian thấp giúp lặp thử nghiệm nhanh hơn."
            "<extra></extra>"
        ),
    ), 1, 2)

    fig.update_layout(**_lo(t, "Hồi quy: độ khớp (R2) và chi phí huấn luyện"))
    fig.update_xaxes(title_text="Mô hình hồi quy", row=1, col=1)
    fig.update_yaxes(title_text="R2 (cao hơn là tốt hơn)", row=1, col=1)
    fig.update_xaxes(title_text="Thời gian (giây)", row=1, col=2)
    fig.update_yaxes(title_text="Mô hình hồi quy", row=1, col=2)
    return fig

def fig_semi(theme="dark"):
    t = T(theme)
    fig = go.Figure()
    for m in semi["method"].unique():
        s = semi[semi["method"] == m].sort_values("label_pct")
        fig.add_trace(go.Scatter(x=s["label_pct"]*100, y=s["f1"],
            mode="lines+markers", name=model_label(m),
            line=dict(color=t["semi_c"].get(m, t["accent"]), width=2), marker=dict(size=8),
            hovertemplate=(
                "Phương pháp: %{fullData.name}<br>"
                "Tên chỉ số: F1 theo tỷ lệ nhãn<br>"
                "Tỷ lệ nhãn: %{x:.1f}%<br>"
                "Giá trị F1: %{y:.4f}<br>"
                "Diễn giải ngắn: Cùng tỷ lệ nhãn, đường cao hơn là phương pháp tốt hơn."
                "<extra></extra>"
            )))
    fig.update_layout(**_lo(t, "Học bán giám sát: F1 theo tỷ lệ nhãn",
                            xaxis_title="Tỷ lệ nhãn (%)", yaxis_title="F1 Score"))
    return fig


def fig_semi_gain(theme="dark"):
    t = T(theme)
    base = semi[semi["method"] == "supervised_only"][["label_pct", "f1"]].rename(columns={"f1": "f1_base"})
    other = semi[semi["method"].isin(["self_training", "label_spreading"])].copy()
    d = other.merge(base, on="label_pct", how="left")
    d["f1_gain"] = d["f1"] - d["f1_base"]
    d["label_pct_plot"] = d["label_pct"] * 100

    fig = go.Figure()
    for m in ["self_training", "label_spreading"]:
        s = d[d["method"] == m].sort_values("label_pct")
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s["label_pct_plot"],
            y=s["f1_gain"],
            name=model_label(m),
            marker_color=t["semi_c"].get(m, t["accent"]),
            marker_line=dict(color=t["accent"], width=0.8),
            hovertemplate=(
                "Phương pháp: %{fullData.name}<br>"
                "Tỷ lệ nhãn: %{x:.1f}%<br>"
                "Tên chỉ số: F1 tăng thêm so với chỉ học có nhãn<br>"
                "Giá trị: %{y:.4f}<br>"
                "Diễn giải ngắn: Dương là tốt hơn baseline, âm là kém hơn baseline."
                "<extra></extra>"
            ),
        ))

    fig.update_layout(**_lo(
        t,
        "Mức cải thiện F1 so với baseline có nhãn",
        barmode="group",
        xaxis_title="Tỷ lệ nhãn (%)",
        yaxis_title="Delta F1 so với supervised_only",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=t["accent"])
    return fig


def fig_semi_pseudo(theme="dark"):
    t = T(theme)
    s = semi[semi["method"] == "self_training"].copy()
    if "n_pseudo_labeled" not in s.columns or s["n_pseudo_labeled"].dropna().empty:
        return go.Figure(layout=_lo(t, "Self-Training: số nhãn giả đã thêm"))

    s = s.sort_values("label_pct")
    fig = go.Figure(go.Bar(
        x=s["label_pct"] * 100,
        y=s["n_pseudo_labeled"],
        marker_color=t["bar"][1][0],
        marker_line=dict(color=t["bar"][1][1], width=1.2),
        text=s["n_pseudo_labeled"].fillna(0).astype(int),
        textposition="auto",
        hovertemplate=(
            "Phương pháp: Self-Training<br>"
            "Tỷ lệ nhãn ban đầu: %{x:.1f}%<br>"
            "Tên chỉ số: Số nhãn giả được thêm<br>"
            "Giá trị: %{y:.0f}<br>"
            "Diễn giải ngắn: Mô hình tự mở rộng tập huấn luyện bằng mẫu tự tin cao (threshold 0.95)."
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_lo(
        t,
        "Self-Training: số nhãn giả được bổ sung",
        xaxis_title="Tỷ lệ nhãn ban đầu (%)",
        yaxis_title="Số mẫu pseudo-labeled",
    ))
    return fig


def fig_outlier_detection(theme="dark", threshold=1.5):
    t = T(theme)
    if not EDA_NUMERIC_COLS:
        return go.Figure(layout=_lo(t, "Phát hiện outlier theo IQR"))

    rows = []
    for c in EDA_NUMERIC_COLS:
        q1 = EDA_DF[c].quantile(0.25)
        q3 = EDA_DF[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - threshold * iqr
        hi = q3 + threshold * iqr
        mask = (EDA_DF[c] < lo) | (EDA_DF[c] > hi)
        rows.append({
            "column": c,
            "count": int(mask.sum()),
            "pct": float(mask.mean() * 100),
            "lo": float(lo),
            "hi": float(hi),
        })

    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    out["label"] = out["column"].map(feature_label)

    fig = go.Figure(go.Bar(
        x=out["label"],
        y=out["count"],
        marker_color=t["bar"][3][0],
        marker_line=dict(color=t["bar"][3][1], width=1),
        text=out["pct"].map(lambda v: f"{v:.2f}%"),
        textposition="auto",
        customdata=out[["lo", "hi", "pct"]].to_numpy(),
        hovertemplate=(
            "Biến: %{x}<br>"
            "Outlier (IQR): %{y}<br>"
            "Tỷ lệ: %{customdata[2]:.3f}%<br>"
            "Ngưỡng dưới: %{customdata[0]:.4f}<br>"
            "Ngưỡng trên: %{customdata[1]:.4f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_lo(
        t,
        "Phát hiện outlier theo IQR (dữ liệu thô)",
        xaxis_title="Biến số",
        yaxis_title="Số điểm outlier",
    ))
    fig.update_xaxes(tickangle=-20)
    if out["count"].sum() == 0:
        fig.add_annotation(
            text="Không phát hiện outlier theo ngưỡng IQR hiện tại.",
            xref="paper", yref="paper", x=0.5, y=1.08,
            showarrow=False, font=dict(color=t["font_sub"], size=12),
        )
        fig.update_yaxes(range=[-0.1, 1])
    return fig


def fig_boxplot_detail(theme="dark"):
    t = T(theme)
    if not EDA_NUMERIC_COLS:
        fig = go.Figure()
        fig.update_layout(**_lo(t, "Boxplot chi tiết từng biến"))
        return fig

    n_cols = 2
    n_rows = math.ceil(len(EDA_NUMERIC_COLS) / n_cols)
    specs = [[{"type": "box"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[feature_label(c) for c in EDA_NUMERIC_COLS],
        specs=specs,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, c in enumerate(EDA_NUMERIC_COLS):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(go.Box(
            y=EDA_DF[c],
            name="",
            boxpoints="outliers",
            marker=dict(color=t["seq"][i % len(t["seq"])]),
            line=dict(color=t["seq"][i % len(t["seq"])]),
            hovertemplate=(
                "Biến: " + feature_label(c) + "<br>"
                "Giá trị: %{y:.4f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(title_text="Giá trị", row=row, col=col)

    fig.update_layout(**_lo(
        t,
        "Boxplot chi tiết từng biến",
        height=max(420, 320 * n_rows),
    ))
    return fig

def fig_hist(col, theme="dark"):
    t = T(theme)
    n = df[df["Machine failure"] == 0][col]
    f = df[df["Machine failure"] == 1][col]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=n, name="Bình thường",
        marker_color=t["hist_n"][0], marker_line=dict(color=t["hist_n"][1], width=.5), opacity=.7,
        hovertemplate=(
            "Nhóm: Bình thường<br>"
            "Đặc trưng: " + feature_label(col) + "<br>"
            "Giá trị vùng: %{x}<br>"
            "Tên chỉ số: Số lượng mẫu<br>"
            "Giá trị: %{y}<br>"
            "Diễn giải ngắn: Số mẫu không lỗi nằm trong khoảng giá trị này."
            "<extra></extra>"
        )))
    fig.add_trace(go.Histogram(x=f, name="Lỗi",
        marker_color=t["hist_f"][0], marker_line=dict(color=t["hist_f"][1], width=.5), opacity=.7,
        hovertemplate=(
            "Nhóm: Lỗi<br>"
            "Đặc trưng: " + feature_label(col) + "<br>"
            "Giá trị vùng: %{x}<br>"
            "Tên chỉ số: Số lượng mẫu<br>"
            "Giá trị: %{y}<br>"
            "Diễn giải ngắn: Số mẫu lỗi nằm trong khoảng giá trị này."
            "<extra></extra>"
        )))
    fig.update_layout(**_lo(
        t,
        f"Phân phối: {feature_label(col)}",
        barmode="overlay",
        xaxis_title=f"Giá trị của đặc trưng: {feature_label(col)}",
        yaxis_title="Số lượng mẫu",
    ))
    return fig

def fig_scat(xc, yc, theme="dark"):
    t = T(theme)
    s = df.sample(min(2000, len(df)), random_state=42)
    fig = go.Figure()
    for lb, nm, co, sy, sz in [(0,"Bình thường",t["scat_n"],"circle",5),
                                (1,"Lỗi",t["scat_f"],"x",10)]:
        d = s[s["Machine failure"] == lb]
        fig.add_trace(go.Scattergl(x=d[xc], y=d[yc], mode="markers", name=nm,
            marker=dict(color=co, size=sz, symbol=sy),
            hovertemplate=(
                "Nhóm: %{fullData.name}<br>"
                "" + feature_label(xc) + ": %{x:.2f}<br>"
                "" + feature_label(yc) + ": %{y:.2f}<br>"
                "Diễn giải ngắn: Mỗi điểm là 1 bản ghi vận hành của máy."
                "<extra></extra>"
            )))
    fig.update_layout(**_lo(
        t,
        f"{feature_label(xc)} vs {feature_label(yc)}",
        xaxis_title=feature_label(xc),
        yaxis_title=feature_label(yc),
    ))
    return fig


# ==================================================
# DASH APP — FULLY STATIC LAYOUT
# ==================================================

app = Dash(
    __name__,
    title="Bảng Điều Khiển Bảo Trì Dự Đoán",
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)

def _card(children, **extra):
    """Card with default dark style; styles overridden by callback."""
    return html.Div(children, className="dash-card", style={
        "background": DARK["card"], "border": f"1px solid {DARK['card_bdr']}",
        "borderRadius": "12px", "padding": "20px", "marginBottom": "16px",
        **extra,
    })

def _kpi(label, value, color):
    return _card([
        html.P(label, style={"fontSize":"11px","letterSpacing":"2px","margin":"0 0 4px",
                              "color": DARK["font_sub"]}, className="kpi-label"),
        html.H2(value, style={"fontFamily":"Orbitron","fontSize":"32px","margin":"0",
                               "color": color}, className="kpi-value"),
    ])

def _tab_style(t):
    return dict(backgroundColor=t["tab_bg"], color=t["tab_c"],
                border=f"1px solid {t['tab_bdr']}", padding="8px 16px", fontFamily="Inter")
def _tab_sel(t):
    return dict(backgroundColor=t["tab_sel_bg"], color=t["accent"],
                border=f"1px solid {t['tab_sel_bdr']}", padding="8px 16px",
                fontFamily="Inter", fontWeight="600")

# Build static layout
app.layout = html.Div(id="root", style={
    "backgroundColor": DARK["bg"], "color": DARK["font"],
    "fontFamily": "Inter, sans-serif", "minHeight": "100vh", "padding": "20px 30px",
    "transition": "background-color .4s, color .4s",
}, children=[
    dcc.Store(id="theme-store", data="dark", storage_type="local"),

    # ── Header row ──
    html.Div(id="header-row", style={
        "marginBottom":"30px", "paddingBottom":"16px",
        "borderBottom": f"1px solid {DARK['accent']}22",
        "display":"flex", "alignItems":"center", "justifyContent":"space-between",
    }, children=[
        html.Div([
            html.H1("BẢNG ĐIỀU KHIỂN BẢO TRÌ DỰ ĐOÁN", id="hdr-title",
                     style={"fontFamily":"Orbitron,sans-serif","color":DARK["accent"],
                            "fontSize":"28px","letterSpacing":"3px",
                            "textShadow":DARK["title_shadow"],"margin":"0"}),
            html.P("AI4I 2020 - Khai phá Dữ liệu - Trực quan hóa Tương tác",
                   id="hdr-sub", style={"color":DARK["font_sub"],
                   "fontSize":"13px","letterSpacing":"2px"}),
        ]),
        html.Button("☀️ Sáng", id="theme-btn", n_clicks=0, style={
            "background": DARK["card"], "border": f"1px solid {DARK['card_bdr']}",
            "borderRadius":"8px","padding":"8px 18px","color":DARK["accent"],
            "cursor":"pointer","fontFamily":"Inter","fontSize":"15px",
        }),
    ]),

    # ── KPI row ──
    html.Div(id="kpi-row", style={
        "display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"16px","marginBottom":"30px",
    }, children=[
        _kpi("SỐ DÒNG", f"{EDA_SUMMARY['n_rows']:,}", DARK["kpi"][0]),
        _kpi("SỐ CỘT", str(EDA_SUMMARY["n_cols"]), DARK["kpi"][1]),
        _kpi("GIÁ TRỊ THIẾU", f"{EDA_SUMMARY['missing_values']:,}", DARK["kpi"][2]),
        _kpi("DÒNG TRÙNG LẶP", f"{EDA_SUMMARY['duplicate_rows']:,}", DARK["kpi"][3]),
    ]),

    # ── Tabs (STATIC — always in DOM) ──
    dcc.Tabs(id="main-tabs", value="eda",
        colors={"border":DARK["tab_bdr"],"primary":DARK["accent"],"background":DARK["tab_bg"]},
        style={"marginBottom":"20px"}, children=[
            dcc.Tab(label="Khám phá EDA",   value="eda",     id="tab-eda",     style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
        ],
    ),

    # ── Tab content (filled by callback) ──
    html.Div(id="tab-content"),
])


# ==================================================
# CALLBACKS
# ==================================================

# ---- Toggle theme ----
@callback(Output("theme-store","data"),
          Input("theme-btn","n_clicks"),
          State("theme-store","data"),
          prevent_initial_call=True)
def toggle(n, cur):
    return "light" if cur == "dark" else "dark"


# ---- Apply theme to shell + KPIs + tabs ----
@callback(
    Output("root","style"),
    Output("header-row","style"),
    Output("hdr-title","style"),
    Output("hdr-sub","style"),
    Output("theme-btn","children"),
    Output("theme-btn","style"),
    Output("main-tabs","colors"),
    Output("kpi-row","children"),
    Output("tab-eda","style"), Output("tab-eda","selected_style"),
    Input("theme-store","data"),
)
def apply_theme(theme):
    t = T(theme)
    root = {"backgroundColor":t["bg"],"color":t["font"],
            "fontFamily":"Inter,sans-serif","minHeight":"100vh","padding":"20px 30px",
            "transition":"background-color .4s, color .4s"}
    hdr = {"marginBottom":"30px","paddingBottom":"16px",
           "borderBottom":f"1px solid {t['accent']}22",
           "display":"flex","alignItems":"center","justifyContent":"space-between"}
    title = {"fontFamily":"Orbitron,sans-serif","color":t["accent"],"fontSize":"28px",
             "letterSpacing":"3px","textShadow":t["title_shadow"],"margin":"0"}
    sub = {"color":t["font_sub"],"fontSize":"13px","letterSpacing":"2px"}
    btn = {"background":t["card"],"border":f"1px solid {t['card_bdr']}",
           "borderRadius":"8px","padding":"8px 18px","color":t["accent"],
           "cursor":"pointer","fontFamily":"Inter","fontSize":"15px"}
    colors = {"border":t["tab_bdr"],"primary":t["accent"],"background":t["tab_bg"]}

    # Rebuild KPI cards with correct theme colors
    cs = lambda children, **kw: html.Div(children, style={
        "background":t["card"],"border":f"1px solid {t['card_bdr']}",
        "borderRadius":"12px","padding":"20px","marginBottom":"16px", **kw})
    kpis = [
        cs([
             html.P("SỐ DÒNG", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
             html.H2(f"{EDA_SUMMARY['n_rows']:,}", style={"fontFamily":"Orbitron","fontSize":"32px",
                     "margin":"0","color":t["kpi"][0]}),
        ]),
        cs([
             html.P("SỐ CỘT", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
             html.H2(str(EDA_SUMMARY["n_cols"]),
                     style={"fontFamily":"Orbitron","fontSize":"32px",
                            "margin":"0","color":t["kpi"][1]}),
        ]),
        cs([
             html.P("GIÁ TRỊ THIẾU", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
             html.H2(f"{EDA_SUMMARY['missing_values']:,}", style={"fontFamily":"Orbitron","fontSize":"32px",
                "margin":"0","color":t["kpi"][2]}),
         ]),
         cs([
             html.P("DÒNG TRÙNG LẶP", style={"fontSize":"11px","letterSpacing":"2px",
                 "margin":"0 0 4px","color":t["font_sub"]}),
             html.H2(f"{EDA_SUMMARY['duplicate_rows']:,}", style={"fontFamily":"Orbitron","fontSize":"32px",
                     "margin":"0","color":t["kpi"][3]}),
        ]),
    ]
    tab_s = _tab_style(t)
    tab_ss = _tab_sel(t)
    return (root, hdr, title, sub, t["toggle_label"], btn, colors, kpis, tab_s, tab_ss)


# ---- Render tab content (reacts to BOTH tab switch AND theme change) ----
@callback(
    Output("tab-content","children"),
    Input("main-tabs","value"),
    Input("theme-store","data"),
)
def render_tab(tab, theme):
    t = T(theme)
    GHF = {"height":"480px"}

    def crd(children, **kw):
        return html.Div(children, style={
            "background":t["card"],"border":f"1px solid {t['card_bdr']}",
            "borderRadius":"12px","padding":"20px","marginBottom":"16px", **kw})

    summary_items = [
        ("Số dòng", f"{EDA_SUMMARY['n_rows']:,}"),
        ("Số cột", str(EDA_SUMMARY["n_cols"])),
        ("Giá trị thiếu", f"{EDA_SUMMARY['missing_values']:,}"),
        ("Dòng trùng lặp", f"{EDA_SUMMARY['duplicate_rows']:,}"),
        ("Biến số", str(EDA_SUMMARY["n_variables"])),
        ("Tỷ lệ lỗi", f"{EDA_SUMMARY['failure_rate']:.2f}%"),
    ]

    return html.Div([
        crd([
            html.H4("Bắt đầu từ đây", style={"margin":"0 0 10px", "color":t["accent"]}),
            html.P(
                "Dash này chỉ tập trung EDA và đã bỏ các tab mô hình vì đã có trên dashboard API. "
                "Outlier và boxplot được tính trên dữ liệu thô để phản ánh đúng phân phối ban đầu trước tiền xử lý.",
                style={"margin":"0", "lineHeight":"1.7"},
            ),
        ]),
        html.Div(style={"display":"grid", "gridTemplateColumns":"repeat(3, minmax(180px, 1fr))", "gap":"12px"}, children=[
            crd([
                html.P(lbl, style={"fontSize":"12px", "letterSpacing":"1px", "margin":"0 0 6px", "color":t["font_sub"]}),
                html.H3(val, style={"margin":"0", "color":t["accent"], "fontFamily":"Orbitron", "fontSize":"24px"}),
            ], padding="14px")
            for lbl, val in summary_items
        ]),
        crd([dictionary_table(theme)]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            crd([dcc.Graph(figure=fig_outlier_detection(theme), config={"displayModeBar":False}, style=GHF)]),
            crd([dcc.Graph(figure=fig_boxplot_detail(theme), config={"displayModeBar":False}, style={"height":"640px"})]),
        ]),
    ])


# ==================================================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
