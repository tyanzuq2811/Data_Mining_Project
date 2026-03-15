"""
clustering.py – Phân cụm máy/chu kỳ theo hành vi + Profiling cụm.

Thuật toán: KMeans, DBSCAN, Hierarchical (HAC).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class ClusterAnalyzer:
    """
    Phân cụm và profiling cho dataset predictive maintenance.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.models = {}
        self.labels = {}
        self.scores = {}
        self.best_model_name = None
        self.best_labels = None

    def fit_kmeans(
        self, X: np.ndarray, feature_names: List[str] = None
    ) -> Dict[int, Dict]:
        """
        Chạy KMeans với nhiều giá trị k, trả về evaluation cho mỗi k.
        """
        k_range = self.params["mining"]["clustering"]["kmeans"]["n_clusters_range"]
        results = {}

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.params["seed"], n_init=10)
            labels = model.fit_predict(X)
            scores = self._evaluate_clustering(X, labels)
            results[k] = {
                "model": model,
                "labels": labels,
                "inertia": model.inertia_,
                **scores,
            }
            self.models[f"kmeans_k{k}"] = model
            self.labels[f"kmeans_k{k}"] = labels
            self.scores[f"kmeans_k{k}"] = scores

        print(f"[clustering] KMeans fitted for k = {k_range}")
        return results

    def fit_dbscan(self, X: np.ndarray) -> Dict[str, Dict]:
        """Chạy DBSCAN với nhiều (eps, min_samples)."""
        eps_range = self.params["mining"]["clustering"]["dbscan"]["eps_range"]
        min_samples_range = self.params["mining"]["clustering"]["dbscan"]["min_samples_range"]
        results = {}

        for eps in eps_range:
            for ms in min_samples_range:
                model = DBSCAN(eps=eps, min_samples=ms)
                labels = model.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()

                key = f"dbscan_eps{eps}_ms{ms}"
                if n_clusters >= 2:
                    scores = self._evaluate_clustering(X, labels)
                else:
                    scores = {"silhouette": -1, "davies_bouldin": 999, "calinski_harabasz": 0}

                results[key] = {
                    "model": model,
                    "labels": labels,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    **scores,
                }
                self.models[key] = model
                self.labels[key] = labels
                self.scores[key] = scores

        print(f"[clustering] DBSCAN fitted for {len(results)} configurations")
        return results

    def fit_hierarchical(self, X: np.ndarray) -> Dict[int, Dict]:
        """Chạy Hierarchical Clustering (HAC)."""
        k_range = self.params["mining"]["clustering"]["hierarchical"]["n_clusters_range"]
        linkage = self.params["mining"]["clustering"]["hierarchical"]["linkage"]
        results = {}

        for k in k_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X)
            scores = self._evaluate_clustering(X, labels)

            key = f"hac_k{k}"
            results[k] = {
                "model": model,
                "labels": labels,
                **scores,
            }
            self.models[key] = model
            self.labels[key] = labels
            self.scores[key] = scores

        print(f"[clustering] Hierarchical fitted for k = {k_range}")
        return results

    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Đánh giá clustering bằng silhouette, DBI, CHI."""
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) < 2:
            return {"silhouette": -1, "davies_bouldin": 999, "calinski_harabasz": 0}

        # Bỏ noise points cho DBSCAN
        mask = labels != -1
        if mask.sum() < 2:
            return {"silhouette": -1, "davies_bouldin": 999, "calinski_harabasz": 0}

        sil = silhouette_score(X[mask], labels[mask])
        dbi = davies_bouldin_score(X[mask], labels[mask])
        chi = calinski_harabasz_score(X[mask], labels[mask])

        return {"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}

    def get_best_model(self, metric: str = "silhouette") -> Tuple[str, np.ndarray]:
        """Chọn mô hình clustering tốt nhất theo metric."""
        best_name = None
        best_score = -np.inf if metric != "davies_bouldin" else np.inf

        for name, scores in self.scores.items():
            s = scores.get(metric, None)
            if s is None:
                continue
            if metric == "davies_bouldin":
                if s < best_score:
                    best_score = s
                    best_name = name
            else:
                if s > best_score:
                    best_score = s
                    best_name = name

        self.best_model_name = best_name
        self.best_labels = self.labels.get(best_name)
        print(f"[clustering] Best model: {best_name} ({metric}={best_score:.4f})")
        return best_name, self.best_labels

    def profile_clusters(
        self, df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Profiling cụm: thống kê mean, std, failure rate cho mỗi cluster.
        """
        df_prof = df[feature_cols].copy()
        df_prof["cluster"] = labels

        # Thêm failure info nếu có
        target = self.params["data"]["target"]
        if target in df.columns:
            df_prof[target] = df[target].values

        profile = df_prof.groupby("cluster").agg(["mean", "std", "count"])
        
        # Failure rate per cluster
        if target in df_prof.columns:
            failure_rate = df_prof.groupby("cluster")[target].mean()
            failure_count = df_prof.groupby("cluster")[target].sum()
            print("\n[clustering] Cluster Failure Profile:")
            for c in sorted(df_prof["cluster"].unique()):
                if c == -1:
                    continue
                n = (df_prof["cluster"] == c).sum()
                fr = failure_rate.get(c, 0)
                fc = failure_count.get(c, 0)
                print(f"  Cluster {c}: n={n}, failures={int(fc)}, failure_rate={fr:.4f}")

        return profile

    def get_scores_table(self) -> pd.DataFrame:
        """Bảng so sánh tất cả model clustering."""
        rows = []
        for name, scores in self.scores.items():
            rows.append({"model": name, **scores})
        return pd.DataFrame(rows).sort_values("silhouette", ascending=False)
