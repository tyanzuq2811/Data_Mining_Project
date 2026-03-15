# ✅ Rà soát Yêu cầu — Đối chiếu với Rubric gốc

> Tài liệu này đối chiếu **từng yêu cầu trong đề bài gốc** với **file/code đã triển khai**.
> ✅ = Đã đáp ứng đầy đủ | ⚠️ = Đáp ứng một phần | ❌ = Chưa đáp ứng

---

## Rubric (10 điểm)

### A. Chọn đề tài & dữ liệu phù hợp (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Chọn 1 trong các đề tài cho sẵn | ✅ | Đề tài 16: Phân tích lỗi sản xuất & Dự đoán lỗi |
| Bộ dữ liệu phù hợp với đề tài | ✅ | AI4I 2020 Predictive Maintenance (UCI) |
| Dữ liệu đủ lớn (≥ vài nghìn dòng) | ✅ | 10 000 dòng × 14 cột |
| Nêu rõ nguồn dữ liệu | ✅ | `README.md`, `docs/README_DATASET.md`, `01_eda.ipynb` cell 1–2 |

---

### B. EDA / Tiền xử lý (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Mô tả dữ liệu (data dictionary) | ✅ | `01_eda.ipynb` → `create_data_dictionary()` |
| Thống kê mô tả (mean, std, min, max) | ✅ | `01_eda.ipynb` → `df.describe()`, `get_data_summary()` |
| Kiểm tra missing values | ✅ | `01_eda.ipynb` + `02_preprocess_feature.ipynb` |
| Kiểm tra duplicates | ✅ | `01_eda.ipynb` + `02_preprocess_feature.ipynb` |
| Hiển thị phân bố biến | ✅ | `01_eda.ipynb` → histograms, boxplots, bar charts |
| Ma trận tương quan | ✅ | `01_eda.ipynb` → `plot_correlation_matrix()` |
| Phân bố biến mục tiêu | ✅ | `01_eda.ipynb` → `plot_target_distribution()` (bar + pie) |
| Xử lý outliers | ✅ | `02_preprocess_feature.ipynb` → `cleaner.handle_outliers()` (IQR) |
| Mã hóa biến categorical | ✅ | `02_preprocess_feature.ipynb` → One-Hot encoding cho `Type` |
| Chuẩn hóa dữ liệu | ✅ | `02_preprocess_feature.ipynb` + `04_modeling.ipynb` (StandardScaler) |

---

### C. Feature Engineering (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Tạo biến phái sinh có ý nghĩa | ✅ | `src/features/builder.py` → temp_diff, power, torque_speed_ratio, wear_torque |
| Phân bin / rời rạc hóa | ✅ | `builder._bin_tool_wear()` → 5 bin |
| Lag features (thời gian) | ✅ | `builder._create_lag_features()` → lag 1, 3, 5, 10 |
| Rolling features | ✅ | `builder._create_rolling_features()` → mean/std window 5, 10, 20 |
| Interaction features | ✅ | `builder._create_interaction_features()` → air_temp × speed, proc_temp × torque |
| Giải thích ý nghĩa feature | ✅ | `docs/README_DATASET.md` mục 5, `02_preprocess_feature.ipynb` markdown cells |
| Loại bỏ feature gây leakage | ✅ | `02_preprocess_feature.ipynb` → loại TWF, HDF, PWF, OSF, RNF khỏi feature set |

---

### D. Khai phá (Apriori / Clustering) (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| **Apriori — Luật kết hợp** | ✅ | `src/mining/association.py` + `03_mining_or_clustering.ipynb` cells 4–10 |
| Tìm tập phổ biến (frequent itemsets) | ✅ | `miner.mine()` → `apriori()` từ mlxtend |
| Sinh luật kết hợp | ✅ | `association_rules()` với min_confidence, min_lift |
| Luật liên quan đến failure | ✅ | `miner.get_failure_rules()`, `get_failure_type_rules()` |
| In luật dạng text | ✅ | `miner.rules_to_text()` |
| **KMeans** | ✅ | `analyzer.fit_kmeans()` → k=2–8, elbow + silhouette |
| **DBSCAN** | ✅ | `analyzer.fit_dbscan()` → grid search eps × min_samples |
| **HAC** | ✅ | `analyzer.fit_hierarchical()` → Ward linkage |
| So sánh clustering algorithms | ✅ | `analyzer.get_scores_table()` → Silhouette, DBI, CHI |
| Profile clusters | ✅ | `analyzer.profile_clusters()` → mean features + failure rate per cluster |
| Anomaly Detection | ✅ | `src/mining/anomaly.py` → Isolation Forest + LOF |
| So sánh anomaly vs actual | ✅ | `detector.compare_with_actual()` → precision, recall, F1 |

---

### E. Mô hình (Phân loại / Hồi quy) (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| **≥ 3 thuật toán phân loại** | ✅ | 5 thuật toán: LogReg, RF, GBR, XGBoost, LightGBM |
| Xử lý imbalanced data | ✅ | class_weight="balanced", scale_pos_weight=28.5, is_unbalance=True |
| Cross-validation | ✅ | `trainer.cross_validate()` → StratifiedKFold 5-fold |
| Feature importance | ✅ | `trainer.get_feature_importance()` → top 20 |
| **Hồi quy** | ✅ | 4 thuật toán: Linear, RF, GBR, XGBoost → target = Tool wear |
| Save model | ✅ | `trainer.save_model()` → pickle in outputs/models/ |

---

### F. Đánh giá (metrics, so sánh) (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Classification metrics (F1, Precision, Recall, ROC-AUC, PR-AUC) | ✅ | `src/evaluation/metrics.py` → `classification_metrics()` |
| Confusion matrix | ✅ | `metrics.get_confusion_matrix_df()` + `plots.plot_confusion_matrix()` |
| ROC curve | ✅ | `plots.plot_roc_curves()` |
| PR curve | ✅ | `plots.plot_precision_recall_curves()` |
| Regression metrics (MAE, RMSE, R²) | ✅ | `metrics.regression_metrics()` |
| Clustering metrics (Silhouette, DBI) | ✅ | `metrics.clustering_metrics()` |
| So sánh nhiều model | ✅ | `metrics.compare_models()` + `plots.plot_model_comparison()` |
| Error analysis (FP/FN) | ✅ | `metrics.error_analysis()` → FP, FN, miss rate |
| Bảng kết quả so sánh | ✅ | `05_evaluation_report.ipynb` → tổng hợp tất cả |

---

### G. Trình bày / Cấu trúc repo (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Notebooks đánh số 01–05 | ✅ | 01_eda, 02_preprocess, 03_mining, 04_modeling, 04b_semi, 05_evaluation |
| Notebook gọi hàm từ src/ (không code trực tiếp) | ✅ | Tất cả logic trong src/, notebooks chỉ import + gọi |
| src/ tổ chức theo chức năng | ✅ | data/, features/, mining/, models/, evaluation/, visualization/ |
| configs/ chứa params | ✅ | params.yaml |
| scripts/ chạy tự động | ✅ | run_pipeline.py, run_papermill.py |
| outputs/ lưu kết quả | ✅ | figures/, tables/, models/, reports/ |
| requirements.txt | ✅ | Có đầy đủ |
| README.md | ✅ | Có + 4 file docs/ chi tiết |
| .gitignore | ✅ | Có |
| Reproducible | ✅ | Seed=42, params.yaml tập trung, parquet intermediary |

---

### H. Đề xuất & phản biện (1 điểm) ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Insights / đề xuất hành động | ✅ | `05_evaluation_report.ipynb` → 7 actionable insights |
| Phân tích ưu/nhược từng phương pháp | ✅ | `05_evaluation_report.ipynb` → bảng pros/cons 5 classifiers |
| Thách thức gặp phải & giải pháp | ✅ | `05_evaluation_report.ipynb` → 5 challenges + solutions |
| Hướng phát triển tương lai | ✅ | `05_evaluation_report.ipynb` → 6 future directions |

---

### +1 điểm: Chuỗi thời gian ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| ARIMA | ✅ | `src/models/forecasting.py` → `fit_arima()`, ARIMA(2,1,2) |
| Chia train/test theo thời gian | ✅ | `temporal_train_test_split()` — 80% đầu train, 20% cuối test |
| Lag regression | ✅ | `fit_lag_regression()` → GBR với lag features |
| Đánh giá MAE/RMSE | ✅ | `get_results_table()` |
| Biểu đồ forecast | ✅ | `04_modeling.ipynb` cell 22 |

---

### +1 điểm: Semi-supervised Learning ✅

| Yêu cầu | Trạng thái | Triển khai tại |
|----------|-----------|---------------|
| Self-Training | ✅ | `src/models/semi_supervised.py` → `train_self_training()` |
| Label Spreading | ✅ | `train_label_spreading()` → RBF kernel, alpha=0.2, subsample ≤ 5000 |
| Thử nhiều % nhãn (5, 10, 20%) | ✅ | `run_all_experiments()` → 3 methods × 3 percentages |
| So sánh với supervised baseline | ✅ | `train_supervised_only()` → RF trên phần có nhãn |
| Learning curve | ✅ | `get_learning_curve_data()` → 9 points (2%–100%) |
| Phân tích pseudo-label | ✅ | `04b_semi_supervised.ipynb` → accuracy, FP rate per % |

---

## Tổng kết

| Mục | Điểm | Trạng thái |
|-----|------|-----------|
| A. Đề tài & dữ liệu | 1/1 | ✅ Đầy đủ |
| B. EDA / Tiền xử lý | 1/1 | ✅ Đầy đủ |
| C. Feature Engineering | 1/1 | ✅ Đầy đủ |
| D. Khai phá (Apriori/Clustering) | 1/1 | ✅ Đầy đủ |
| E. Mô hình (Classification/Regression) | 1/1 | ✅ Đầy đủ |
| F. Đánh giá | 1/1 | ✅ Đầy đủ |
| G. Trình bày / Cấu trúc | 1/1 | ✅ Đầy đủ |
| H. Đề xuất & phản biện | 1/1 | ✅ Đầy đủ |
| +1 Chuỗi thời gian | 1/1 | ✅ Đầy đủ |
| +1 Semi-supervised | 1/1 | ✅ Đầy đủ |
| **TỔNG** | **10/10** | **✅ Đáp ứng đầy đủ** |

---

## Ghi chú bổ sung

### Cấu trúc repo đúng theo yêu cầu
```
✅ notebooks/ gọi hàm từ src/ (không viết logic trực tiếp)
✅ src/ chia module theo chức năng (data, features, mining, models, evaluation, visualization)
✅ configs/ chứa params.yaml tập trung
✅ scripts/ có 2 script chạy tự động
✅ outputs/ tổ chức thành figures/tables/models/reports
✅ Mỗi module trong src/ độc lập — không import lẫn nhau
✅ Dữ liệu truyền qua filesystem (parquet, CSV, JSON)
```

### Tổng số file đã tạo
| Loại | Số lượng | Files |
|------|---------|-------|
| Config | 1 | params.yaml |
| Source code (.py) | 12 | loader, cleaner, builder, association, clustering, anomaly, supervised, semi_supervised, forecasting, metrics, report, plots |
| Package markers | 7 | __init__.py |
| Notebooks | 6 | 01_eda → 05_evaluation_report |
| Scripts | 2 | run_pipeline.py, run_papermill.py |
| Docs | 4 | README_DATASET, README_PROJECT, README_STRUCTURE, README_PIPELINE |
| Root files | 3 | README.md, requirements.txt, .gitignore |
| **Tổng** | **35 files** | |
