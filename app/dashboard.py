"""
Dash Dashboard - Bảng Điều Khiển Bảo Trì Dự Đoán
Cổng 8050 - Hỗ trợ chế độ Tối / Sáng
Static layout — no dynamic re-rendering of page structure.
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, callback, no_update

# ── Dữ liệu ──────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
DATA   = ROOT / "data" / "processed"

clf   = pd.read_csv(TABLES / "classification_results.csv")
cv    = pd.read_csv(TABLES / "cv_results.csv")
clust = pd.read_csv(TABLES / "clustering_comparison.csv")
anom  = pd.read_csv(TABLES / "anomaly_comparison.csv")
reg   = pd.read_csv(TABLES / "regression_results.csv")
semi  = pd.read_csv(TABLES / "semi_supervised_results.csv")
df    = pd.read_parquet(DATA / "ai4i2020_processed.parquet")

SENSOR_COLS = [c for c in [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
] if c in df.columns]

N_FEATURES = len([c for c in df.columns if c != "Machine failure" and
                   c not in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]])

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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=clf["model"], y=clf["f1"], name="F1",
        marker_color=t["bar"][0][0], marker_line=dict(color=t["bar"][0][1], width=1.5),
        text=clf["f1"].round(3), textposition="auto", textfont=dict(size=13)))
    fig.add_trace(go.Bar(x=clf["model"], y=clf["pr_auc"], name="PR-AUC",
        marker_color=t["bar"][1][0], marker_line=dict(color=t["bar"][1][1], width=1.5),
        text=clf["pr_auc"].round(3), textposition="auto", textfont=dict(size=13)))
    fig.update_layout(**_lo(t, "So sánh phân loại: F1 & PR-AUC", barmode="group"))
    return fig

def fig_cv(theme="dark"):
    t = T(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cv["model"], y=cv["cv_f1_mean"], name="F1 trung bình CV",
        error_y=dict(type="data", array=cv["cv_f1_std"], visible=True, color=t["bar"][0][1]),
        marker_color=t["bar"][0][0], marker_line=dict(color=t["bar"][0][1], width=1.5),
        text=cv["cv_f1_mean"].round(3), textposition="auto", textfont=dict(size=13)))
    fig.update_layout(**_lo(t, "Cross-Validation 5-Fold (F1 ± Độ lệch chuẩn)"))
    return fig

def fig_pr(theme="dark"):
    t = T(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=clf["model"], y=clf["precision"], name="Precision",
        marker_color=t["bar"][2][0], marker_line=dict(color=t["bar"][2][1], width=1)))
    fig.add_trace(go.Bar(x=clf["model"], y=clf["recall"], name="Recall",
        marker_color=t["bar"][3][0], marker_line=dict(color=t["bar"][3][1], width=1)))
    fig.update_layout(**_lo(t, "Precision vs Recall", barmode="group"))
    return fig

def fig_time(theme="dark"):
    t = T(theme)
    fig = go.Figure(go.Bar(y=clf["model"], x=clf["train_time_s"], orientation="h",
        marker_color=t["seq"][:len(clf)],
        text=clf["train_time_s"].round(2), textposition="auto"))
    fig.update_layout(**_lo(t, "Thời gian huấn luyện (giây)",
                            margin=dict(l=160, r=30, t=50, b=50)))
    return fig

def fig_cluster(theme="dark"):
    t = T(theme)
    top = clust.head(10)
    c = [t["pos"][0] if s > 0 else t["neg"][0] for s in top["silhouette"]]
    b = [t["pos"][1] if s > 0 else t["neg"][1] for s in top["silhouette"]]
    fig = go.Figure(go.Bar(y=top["model"], x=top["silhouette"], orientation="h",
        marker_color=c, marker_line=dict(color=b, width=1.5),
        text=top["silhouette"].round(3), textposition="auto"))
    fig.update_layout(**_lo(t, "Silhouette Score — Phân cụm (Top 10)",
                            margin=dict(l=170, r=30, t=50, b=50),
                            yaxis_autorange="reversed"))
    return fig

def fig_anom(theme="dark"):
    t = T(theme)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Số lượng phát hiện", "F1 Score"])
    fig.add_trace(go.Bar(x=anom["Method"], y=anom["Anomalies Detected"],
        marker_color=[t["bar"][0][0], t["bar"][1][0]],
        marker_line=dict(color=[t["bar"][0][1], t["bar"][1][1]], width=1.5),
        showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=anom["Method"], y=anom["F1"],
        marker_color=[t["bar"][2][0], t["bar"][3][0]],
        marker_line=dict(color=[t["bar"][2][1], t["bar"][3][1]], width=1.5),
        text=anom["F1"].round(4), textposition="auto", showlegend=False), 1, 2)
    fig.update_layout(**_lo(t, "So sánh phát hiện bất thường"))
    return fig

def fig_reg(theme="dark"):
    t = T(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=reg["model"], y=reg["MAE"], name="MAE",
        marker_color=t["pos"][0], marker_line=dict(color=t["pos"][1], width=1.5)))
    fig.add_trace(go.Bar(x=reg["model"], y=reg["RMSE"], name="RMSE",
        marker_color=t["neg"][0], marker_line=dict(color=t["neg"][1], width=1.5)))
    fig.update_layout(**_lo(t, "Hồi quy: MAE & RMSE", barmode="group"))
    return fig

def fig_semi(theme="dark"):
    t = T(theme)
    fig = go.Figure()
    for m in semi["method"].unique():
        s = semi[semi["method"] == m]
        fig.add_trace(go.Scatter(x=s["label_pct"]*100, y=s["f1"],
            mode="lines+markers", name=m.replace("_"," ").title(),
            line=dict(color=t["semi_c"].get(m, t["accent"]), width=2), marker=dict(size=8)))
    fig.update_layout(**_lo(t, "Học bán giám sát: F1 theo tỷ lệ nhãn",
                            xaxis_title="Tỷ lệ nhãn (%)", yaxis_title="F1 Score"))
    return fig

def fig_hist(col, theme="dark"):
    t = T(theme)
    n = df[df["Machine failure"] == 0][col]
    f = df[df["Machine failure"] == 1][col]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=n, name="Bình thường",
        marker_color=t["hist_n"][0], marker_line=dict(color=t["hist_n"][1], width=.5), opacity=.7))
    fig.add_trace(go.Histogram(x=f, name="Lỗi",
        marker_color=t["hist_f"][0], marker_line=dict(color=t["hist_f"][1], width=.5), opacity=.7))
    fig.update_layout(**_lo(t, f"Phân phối: {col}", barmode="overlay", xaxis_title=col))
    return fig

def fig_scat(xc, yc, theme="dark"):
    t = T(theme)
    s = df.sample(min(2000, len(df)), random_state=42)
    fig = go.Figure()
    for lb, nm, co, sy, sz in [(0,"Bình thường",t["scat_n"],"circle",5),
                                (1,"Lỗi",t["scat_f"],"x",10)]:
        d = s[s["Machine failure"] == lb]
        fig.add_trace(go.Scattergl(x=d[xc], y=d[yc], mode="markers", name=nm,
            marker=dict(color=co, size=sz, symbol=sy)))
    fig.update_layout(**_lo(t, f"{xc} vs {yc}", xaxis_title=xc, yaxis_title=yc))
    return fig


# ==================================================
# DASH APP — FULLY STATIC LAYOUT
# ==================================================

app = Dash(__name__, title="Bảng Điều Khiển Bảo Trì Dự Đoán",
           suppress_callback_exceptions=True)

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
        _kpi("TỔNG BẢN GHI", f"{len(df):,}", DARK["kpi"][0]),
        _kpi("TỶ LỆ LỖI", f"{df['Machine failure'].mean()*100:.2f}%", DARK["kpi"][1]),
        html.Div([
            _kpi("ĐIỂM F1 TỐT NHẤT", f"{clf['f1'].max():.4f}", DARK["kpi"][2]),
            html.P(clf.loc[clf["f1"].idxmax(),"model"].upper(),
                   style={"fontSize":"11px","color":DARK["kpi"][2],"margin":"-12px 0 0 20px",
                          "opacity":".6"}, className="kpi-model"),
        ]),
        _kpi("ĐẶC TRƯNG", str(N_FEATURES), DARK["kpi"][3]),
    ]),

    # ── Tabs (STATIC — always in DOM) ──
    dcc.Tabs(id="main-tabs", value="clf",
        colors={"border":DARK["tab_bdr"],"primary":DARK["accent"],"background":DARK["tab_bg"]},
        style={"marginBottom":"20px"}, children=[
            dcc.Tab(label="Phân loại",      value="clf",     id="tab-clf",     style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
            dcc.Tab(label="Phân cụm",       value="cluster", id="tab-cluster", style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
            dcc.Tab(label="Bất thường",     value="anomaly", id="tab-anomaly", style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
            dcc.Tab(label="Hồi quy",        value="reg",     id="tab-reg",     style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
            dcc.Tab(label="Bán giám sát",   value="semi",    id="tab-semi",    style=_tab_style(DARK), selected_style=_tab_sel(DARK)),
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
    Output("tab-clf","style"), Output("tab-clf","selected_style"),
    Output("tab-cluster","style"), Output("tab-cluster","selected_style"),
    Output("tab-anomaly","style"), Output("tab-anomaly","selected_style"),
    Output("tab-reg","style"), Output("tab-reg","selected_style"),
    Output("tab-semi","style"), Output("tab-semi","selected_style"),
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
            html.P("TỔNG BẢN GHI", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
            html.H2(f"{len(df):,}", style={"fontFamily":"Orbitron","fontSize":"32px",
                     "margin":"0","color":t["kpi"][0]}),
        ]),
        cs([
            html.P("TỶ LỆ LỖI", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
            html.H2(f"{df['Machine failure'].mean()*100:.2f}%",
                     style={"fontFamily":"Orbitron","fontSize":"32px",
                            "margin":"0","color":t["kpi"][1]}),
        ]),
        html.Div([
            cs([
                html.P("ĐIỂM F1 TỐT NHẤT", style={"fontSize":"11px","letterSpacing":"2px",
                       "margin":"0 0 4px","color":t["font_sub"]}),
                html.H2(f"{clf['f1'].max():.4f}", style={"fontFamily":"Orbitron","fontSize":"32px",
                         "margin":"0","color":t["kpi"][2]}),
            ]),
            html.P(clf.loc[clf["f1"].idxmax(),"model"].upper(),
                   style={"fontSize":"11px","color":t["kpi"][2],
                          "margin":"-12px 0 0 20px","opacity":".6"}),
        ]),
        cs([
            html.P("ĐẶC TRƯNG", style={"fontSize":"11px","letterSpacing":"2px",
                   "margin":"0 0 4px","color":t["font_sub"]}),
            html.H2(str(N_FEATURES), style={"fontFamily":"Orbitron","fontSize":"32px",
                     "margin":"0","color":t["kpi"][3]}),
        ]),
    ]
    tab_s = _tab_style(t)
    tab_ss = _tab_sel(t)
    return (root, hdr, title, sub, t["toggle_label"], btn, colors, kpis,
            tab_s, tab_ss, tab_s, tab_ss, tab_s, tab_ss,
            tab_s, tab_ss, tab_s, tab_ss, tab_s, tab_ss)


# ---- Render tab content (reacts to BOTH tab switch AND theme change) ----
@callback(
    Output("tab-content","children"),
    Input("main-tabs","value"),
    Input("theme-store","data"),
)
def render_tab(tab, theme):
    t = T(theme)
    GH  = {"height":"420px"}
    GHF = {"height":"480px"}

    def crd(children, **kw):
        return html.Div(children, style={
            "background":t["card"],"border":f"1px solid {t['card_bdr']}",
            "borderRadius":"12px","padding":"20px","marginBottom":"16px", **kw})

    if tab == "clf":
        return html.Div([
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                crd([dcc.Graph(figure=fig_clf(theme),  config={"displayModeBar":False}, style=GH)]),
                crd([dcc.Graph(figure=fig_cv(theme),   config={"displayModeBar":False}, style=GH)]),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                crd([dcc.Graph(figure=fig_pr(theme),   config={"displayModeBar":False}, style=GH)]),
                crd([dcc.Graph(figure=fig_time(theme),  config={"displayModeBar":False}, style=GH)]),
            ]),
        ])
    if tab == "cluster":
        return crd([dcc.Graph(figure=fig_cluster(theme), config={"displayModeBar":False},
                              style={"height":"500px"})])
    if tab == "anomaly":
        return crd([dcc.Graph(figure=fig_anom(theme), config={"displayModeBar":False}, style=GHF)])
    if tab == "reg":
        return crd([dcc.Graph(figure=fig_reg(theme), config={"displayModeBar":False}, style=GHF)])
    if tab == "semi":
        return crd([dcc.Graph(figure=fig_semi(theme), config={"displayModeBar":False}, style=GHF)])
    if tab == "eda":
        dd = {"backgroundColor":t["dd_bg"],"color":t["dd_c"],"border":f"1px solid {t['dd_bdr']}"}
        return html.Div([
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                crd([
                    html.Label("Chọn đặc trưng:", style={"fontSize":"13px","color":t["font_sub"]}),
                    dcc.Dropdown(id="hist-feature",
                        options=[{"label":c,"value":c} for c in SENSOR_COLS],
                        value=SENSOR_COLS[0] if SENSOR_COLS else None, style=dd),
                    dcc.Graph(id="hist-chart", config={"displayModeBar":False}, style=GH),
                ]),
                crd([
                    html.Div(style={"display":"flex","gap":"12px","marginBottom":"8px"}, children=[
                        html.Div([
                            html.Label("Trục X:", style={"fontSize":"13px","color":t["font_sub"]}),
                            dcc.Dropdown(id="scatter-x-dash",
                                options=[{"label":c,"value":c} for c in SENSOR_COLS],
                                value=SENSOR_COLS[2] if len(SENSOR_COLS)>2 else SENSOR_COLS[0] if SENSOR_COLS else None,
                                style={**dd,"width":"220px"}),
                        ]),
                        html.Div([
                            html.Label("Trục Y:", style={"fontSize":"13px","color":t["font_sub"]}),
                            dcc.Dropdown(id="scatter-y-dash",
                                options=[{"label":c,"value":c} for c in SENSOR_COLS],
                                value=SENSOR_COLS[3] if len(SENSOR_COLS)>3 else SENSOR_COLS[0] if SENSOR_COLS else None,
                                style={**dd,"width":"220px"}),
                        ]),
                    ]),
                    dcc.Graph(id="scatter-chart-dash", config={"displayModeBar":False}, style=GH),
                ]),
            ]),
        ])
    return html.Div()


@callback(Output("hist-chart","figure"),
          Input("hist-feature","value"),
          State("theme-store","data"))
def update_hist(col, theme):
    if not col: return go.Figure()
    return fig_hist(col, theme)


@callback(Output("scatter-chart-dash","figure"),
          Input("scatter-x-dash","value"),
          Input("scatter-y-dash","value"),
          State("theme-store","data"))
def update_scatter(xc, yc, theme):
    if not xc or not yc: return go.Figure()
    return fig_scat(xc, yc, theme)


# ==================================================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
