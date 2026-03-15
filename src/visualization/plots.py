"""
plots.py – Hàm vẽ biểu đồ dùng chung cho toàn bộ project.

Sử dụng matplotlib + seaborn. Tất cả hình có thể lưu ra outputs/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# Default style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def save_fig(fig, name: str, output_dir: str = "outputs/figures"):
    """Lưu figure ra file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[plots] Saved: {path}")


# ==================================================================
# EDA PLOTS
# ==================================================================
def plot_target_distribution(y: pd.Series, title: str = "Machine Failure Distribution"):
    """Biểu đồ phân bố target (imbalanced)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = y.value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title(title)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=["Normal (0)", "Failure (1)"],
                autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
                startangle=90, explode=[0, 0.1])
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    return fig


def plot_failure_types(df: pd.DataFrame, failure_cols: List[str]):
    """Biểu đồ phân bố các loại lỗi."""
    counts = df[failure_cols].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Reds_r", len(counts))
    counts.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Failure Type Distribution")
    ax.set_xlabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str]):
    """Histogram + boxplot cho từng biến số."""
    n = len(numeric_cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(numeric_cols):
        # Histogram
        axes[i, 0].hist(df[col], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        axes[i, 0].set_title(f"{col} – Distribution")
        axes[i, 0].set_xlabel(col)

        # Boxplot
        sns.boxplot(x=df[col], ax=axes[i, 1], color="#3498db")
        axes[i, 1].set_title(f"{col} – Boxplot")

    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: List[str] = None):
    """Heatmap ma trận tương quan."""
    if cols:
        corr = df[cols].corr()
    else:
        corr = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_feature_vs_target(df: pd.DataFrame, feature: str, target: str = "Machine failure"):
    """Boxplot / violin plot feature theo target class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(x=target, y=feature, data=df, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title(f"{feature} by {target}")

    sns.violinplot(x=target, y=feature, data=df, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title(f"{feature} by {target} (Violin)")

    plt.tight_layout()
    return fig


# ==================================================================
# MODEL EVALUATION PLOTS
# ==================================================================
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ["Normal", "Failure"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_roc_curves(results: Dict[str, Dict], title="ROC Curves"):
    """ROC curves cho nhiều mô hình."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, res in results.items():
        y_true = res.get("y_true")
        y_prob = res.get("y_prob")
        if y_true is None or y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_precision_recall_curves(results: Dict[str, Dict], title="Precision-Recall Curves"):
    """PR curves cho nhiều mô hình."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, res in results.items():
        y_true = res.get("y_true")
        y_prob = res.get("y_prob")
        if y_true is None or y_prob is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: pd.DataFrame, top_n: int = 15, title="Feature Importance"):
    """Bar chart feature importance."""
    df_plot = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df_plot)), df_plot["importance"].values, color="#3498db")
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"].values)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric_cols: List[str] = None):
    """Bar chart so sánh nhiều mô hình."""
    if metric_cols is None:
        metric_cols = ["f1", "roc_auc", "pr_auc"]
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.8 / len(metric_cols)

    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, results_df[col], width, label=col)

    ax.set_xticks(x + width * len(metric_cols) / 2)
    ax.set_xticklabels(results_df["model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


# ==================================================================
# CLUSTERING PLOTS
# ==================================================================
def plot_elbow(inertias: Dict[int, float]):
    """Elbow method plot cho KMeans."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]
    ax.plot(ks, vals, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    plt.tight_layout()
    return fig


def plot_silhouette_scores(scores: Dict[int, float]):
    """Silhouette score theo k."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    ax.plot(ks, vals, "go-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    plt.tight_layout()
    return fig


def plot_cluster_profiles(profile_df: pd.DataFrame, features: List[str]):
    """Radar/bar chart profile cho từng cluster."""
    fig, ax = plt.subplots(figsize=(12, 6))
    profile_df[features].T.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Profiles (Mean Values)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Value (Scaled)")
    ax.legend(title="Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ==================================================================
# SEMI-SUPERVISED PLOTS
# ==================================================================
def plot_learning_curve(curve_df: pd.DataFrame, title="Learning Curve: F1 vs % Labels"):
    """Learning curve cho semi-supervised experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in curve_df["method"].unique():
        subset = curve_df[curve_df["method"] == method]
        ax.plot(subset["pct"] * 100, subset["f1"], "o-", label=method, linewidth=2, markersize=8)

    ax.set_xlabel("Labeled Data (%)")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Residual plot cho regression."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual scatter
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(title)

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="#3498db", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    return fig
