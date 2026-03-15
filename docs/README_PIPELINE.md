# 🔄 Luồng hoạt động của chương trình (Pipeline Flow)

> Tài liệu này mô tả **trình tự chạy**, **dữ liệu đi qua đâu**, và **mỗi bước làm gì**.

---

## 1. Tổng quan luồng

```
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║                         LUỒNG CHẠY CHƯƠNG TRÌNH                        ║
 ╠══════════════════════════════════════════════════════════════════════════╣
 ║                                                                         ║
 ║  data/raw/ai4i2020.csv                                                  ║
 ║        │                                                                ║
 ║        ▼                                                                ║
 ║  ┌─────────────────┐     Notebook 01                                    ║
 ║  │   01. EDA        │     Khám phá, vẽ biểu đồ, phát hiện rủi ro       ║
 ║  └────────┬────────┘                                                    ║
 ║           │ (không thay đổi data)                                       ║
 ║           ▼                                                             ║
 ║  ┌─────────────────┐     Notebook 02                                    ║
 ║  │ 02. Tiền xử lý  │     Clean → Feature Engineering → Save            ║
 ║  │ & Feature Eng.   │                                                   ║
 ║  └────────┬────────┘                                                    ║
 ║           │                                                             ║
 ║           ▼                                                             ║
 ║  data/processed/ai4i2020_processed.parquet  ← DỮ LIỆU ĐÃ XỬ LÝ       ║
 ║  data/processed/feature_info.json           ← THÔNG TIN FEATURE        ║
 ║           │                                                             ║
 ║     ┌─────┼──────────────┬──────────────┐                               ║
 ║     ▼     ▼              ▼              ▼                               ║
 ║  ┌──────┐ ┌────────┐ ┌────────┐ ┌──────────┐                           ║
 ║  │  03  │ │   04   │ │  04b   │ │    05    │                            ║
 ║  │Mining│ │Modeling│ │Semi-sup│ │ Report   │                            ║
 ║  └──┬───┘ └───┬────┘ └───┬────┘ └────┬─────┘                           ║
 ║     │         │          │           │                                   ║
 ║     ▼         ▼          ▼           ▼                                   ║
 ║  outputs/tables/     outputs/figures/     outputs/reports/               ║
 ║  outputs/models/                                                        ║
 ║                                                                         ║
 ╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Chi tiết từng bước

### Bước 1: EDA (`01_eda.ipynb`)

```
INPUT:  data/raw/ai4i2020.csv
OUTPUT: Hiểu biết về dữ liệu (không tạo file mới)
```

| Thứ tự | Hành động | Hàm gọi từ src/ |
|--------|----------|-----------------|
| 1.1 | Đọc CSV gốc | `loader.load_raw_data()` |
| 1.2 | Kiểm tra cấu trúc | `loader.validate_schema()` |
| 1.3 | Tạo data dictionary | `loader.create_data_dictionary()` |
| 1.4 | Thống kê mô tả | `loader.get_data_summary()` |
| 1.5 | Vẽ phân bố target | `plots.plot_target_distribution()` |
| 1.6 | Vẽ failure types | `plots.plot_failure_types()` |
| 1.7 | Phân bố biến số | `plots.plot_numeric_distributions()` |
| 1.8 | Ma trận tương quan | `plots.plot_correlation_matrix()` |
| 1.9 | Feature vs Target | `plots.plot_feature_vs_target()` |
| 1.10 | Phân tích rủi ro | Markdown: imbalance, leakage, temporal |

**Kết luận bước 1:** Dữ liệu mất cân bằng 28.5:1, 5 cột failure type gây leakage, không có missing/duplicates.

---

### Bước 2: Tiền xử lý & Feature Engineering (`02_preprocess_feature.ipynb`)

```
INPUT:  data/raw/ai4i2020.csv
OUTPUT: data/processed/ai4i2020_processed.parquet
        data/processed/ai4i2020_processed.csv
        data/processed/feature_info.json
```

| Thứ tự | Hành động | Hàm gọi | Chi tiết |
|--------|----------|---------|---------|
| 2.1 | Đọc dữ liệu gốc | `loader.load_raw_data()` | 10 000 × 14 |
| 2.2 | Xử lý missing | `cleaner.handle_missing()` | Median cho số, mode cho categorical |
| 2.3 | Xử lý duplicates | `cleaner.handle_duplicates()` | Loại dòng trùng |
| 2.4 | Xử lý outliers | `cleaner.handle_outliers()` | Clip bằng IQR (1.5×) |
| 2.5 | Mã hóa Type | `cleaner.encode_categorical()` | One-Hot → Type_H, Type_L, Type_M |
| 2.6 | Tạo feature phái sinh | `builder.build()` | temp_diff, power, ratio, wear_torque |
| 2.7 | Phân bin Tool wear | `builder._bin_tool_wear()` | 5 bin → one-hot |
| 2.8 | Tạo lag features | `builder._create_lag_features()` | Lag 1, 3, 5, 10 cho mỗi biến số |
| 2.9 | Tạo rolling features | `builder._create_rolling_features()` | Mean & Std cửa sổ 5, 10, 20 |
| 2.10 | Xác định feature set | Loại UDI, Product ID, 5 failure types | Tránh leakage |
| 2.11 | Lưu kết quả | `to_parquet()`, `to_csv()`, `json.dump()` | 3 file output |

**Kết quả bước 2:** DataFrame ~10 000 dòng × ~80+ cột (gốc + feature mới), sẵn sàng cho mọi bước sau.

---

### Bước 3: Khai phá — Apriori + Clustering + Anomaly (`03_mining_or_clustering.ipynb`)

```
INPUT:  data/raw/ai4i2020.csv (cho Apriori — cần dữ liệu gốc để binary hóa)
        data/processed/ai4i2020_processed.parquet (cho Clustering & Anomaly)
OUTPUT: Bảng luật, bảng so sánh clustering, bảng anomaly → outputs/tables/
```

#### 3A. Apriori Association Rules

| Thứ tự | Hành động | Hàm gọi |
|--------|----------|---------|
| 3A.1 | Chuyển dữ liệu → nhị phân | `builder.get_apriori_features()` |
| 3A.2 | Tìm tập phổ biến | `miner.mine()` → `apriori()` từ mlxtend |
| 3A.3 | Sinh luật kết hợp | `association_rules()` → lọc bởi min_confidence, min_lift |
| 3A.4 | Lọc luật liên quan hỏng | `miner.get_failure_rules()` |
| 3A.5 | Luật theo từng loại lỗi | `miner.get_failure_type_rules()` |
| 3A.6 | In dạng text | `miner.rules_to_text()` |

**Ví dụ output:**
```
IF (torque_high, tool_wear_high) → THEN (Machine failure)
  support=0.012, confidence=0.78, lift=23.0
```

#### 3B. Clustering

| Thứ tự | Hành động | Hàm gọi |
|--------|----------|---------|
| 3B.1 | Chọn feature cảm biến + scale | `StandardScaler()` |
| 3B.2 | KMeans k=2→8 | `analyzer.fit_kmeans()` → elbow plot, silhouette plot |
| 3B.3 | DBSCAN grid | `analyzer.fit_dbscan()` |
| 3B.4 | HAC | `analyzer.fit_hierarchical()` |
| 3B.5 | So sánh tất cả | `analyzer.get_scores_table()` |
| 3B.6 | Profile cụm tốt nhất | `analyzer.profile_clusters()` → tỷ lệ hỏng mỗi cụm |
| 3B.7 | PCA 2D visualization | Scatter plot: cluster label vs actual failure |

#### 3C. Anomaly Detection

| Thứ tự | Hành động | Hàm gọi |
|--------|----------|---------|
| 3C.1 | Isolation Forest | `detector.fit_isolation_forest()` |
| 3C.2 | LOF | `detector.fit_lof()` |
| 3C.3 | So sánh với nhãn thực | `detector.compare_with_actual()` → precision, recall, F1 |

---

### Bước 4: Mô hình hóa (`04_modeling.ipynb`)

```
INPUT:  data/processed/ai4i2020_processed.parquet
        data/processed/feature_info.json
OUTPUT: outputs/tables/classification_results.csv
        outputs/tables/cv_results.csv
        outputs/tables/regression_results.csv
        outputs/tables/time_series_results.csv
        outputs/models/best_classifier.pkl
        outputs/figures/ (ROC, PR, confusion matrix, ...)
```

#### 4A. Classification (Phân loại — Bài toán chính)

```
Dữ liệu processed
    ↓
Stratified train/test split (80/20, giữ tỷ lệ hỏng)
    ↓
StandardScaler (fit trên train, transform cả test)
    ↓
Train 5 classifiers cùng lúc:
    ├── Logistic Regression (class_weight=balanced)
    ├── Random Forest (class_weight=balanced)
    ├── Gradient Boosting
    ├── XGBoost (scale_pos_weight=28.5)
    └── LightGBM (is_unbalance=True)
    ↓
Đánh giá: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
    ↓
Cross-validation 5-fold (Stratified)
    ↓
Feature Importance (top 20)
    ↓
Error Analysis (FP/FN, phân tích theo failure type)
    ↓
Lưu best model → pickle
```

#### 4B. Regression (Hồi quy — Tool wear)

```
Dữ liệu processed (target = Tool wear [min])
    ↓
Train/test split (80/20, temporal)
    ↓
Train 4 regressors:
    ├── Linear Regression
    ├── Random Forest
    ├── Gradient Boosting
    └── XGBoost
    ↓
Đánh giá: MAE, RMSE, R²
```

#### 4C. Time Series (Chuỗi thời gian)

```
Sắp xếp theo UDI (pseudo-time)
    ↓
Split 80% đầu = train, 20% cuối = test
    ↓
├── ARIMA(2,1,2) → forecast
└── Lag Regression (GBR + lag features)
    ↓
Đánh giá: MAE, RMSE
```

---

### Bước 4b: Học bán giám sát (`04b_semi_supervised.ipynb`)

```
INPUT:  data/processed/ai4i2020_processed.parquet
OUTPUT: outputs/tables/semi_supervised_results.csv
        outputs/figures/ (learning curve, pseudo-label analysis)
```

```
Dữ liệu processed → Stratified split → StandardScaler
    ↓
Với mỗi label_pct ∈ {5%, 10%, 20%}:
    ↓
    Che nhãn (stratified) → giữ label_pct, phần còn lại = −1
    ↓
    ├── Supervised-only baseline (RF trên phần có nhãn)
    ├── Self-Training (RF base, threshold=0.95)
    └── Label Spreading (RBF kernel, alpha=0.2, subsample≤5000)
    ↓
    Đánh giá F1 trên test set (nhãn thực)
    ↓
So sánh: Self-Training thường > Supervised-only > Label Spreading
    ↓
Learning curve: F1 vs % nhãn (từ 2% → 100%)
    ↓
Phân tích pseudo-label: accuracy, false positive rate
```

---

### Bước 5: Đánh giá tổng hợp (`05_evaluation_report.ipynb`)

```
INPUT:  Tất cả CSV từ outputs/tables/
OUTPUT: outputs/reports/insights.txt
        outputs/tables/ (comparison tables)
        outputs/figures/ (dashboard 4-panel)
```

| Thứ tự | Hành động |
|--------|----------|
| 5.1 | Đọc tất cả bảng kết quả từ bước 3, 4, 4b |
| 5.2 | Vẽ dashboard tổng hợp 4 panel (classification, training time, regression, semi-supervised) |
| 5.3 | Bảng so sánh pros/cons của 5 classifier |
| 5.4 | So sánh supervised vs semi-supervised |
| 5.5 | **7 đề xuất hành động** (actionable insights) cho nhà máy |
| 5.6 | Bảng thách thức & giải pháp |
| 5.7 | Tổng kết & hướng phát triển tương lai |
| 5.8 | Lưu toàn bộ artifacts |

---

## 3. Dữ liệu chảy qua đâu? (Data Flow)

```
data/raw/ai4i2020.csv
        │
        │  [Notebook 01: chỉ ĐỌC, không ghi]
        │
        ▼
data/raw/ai4i2020.csv ──────► [Notebook 02: ĐỌC → XỬ LÝ → GHI]
                                      │
                                      ▼
                        data/processed/ai4i2020_processed.parquet
                        data/processed/feature_info.json
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            [Notebook 03]     [Notebook 04]      [Notebook 04b]
            Mining/Cluster    Classification      Semi-supervised
                    │         Regression
                    │         Time Series
                    │                 │                  │
                    ▼                 ▼                  ▼
            outputs/tables/   outputs/tables/    outputs/tables/
            outputs/figures/  outputs/figures/   outputs/figures/
                              outputs/models/
                    │                 │                  │
                    └─────────────────┼──────────────────┘
                                      ▼
                              [Notebook 05]
                              Tổng hợp & Báo cáo
                                      │
                                      ▼
                              outputs/reports/insights.txt
```

---

## 4. Hai cách chạy

### Cách 1: Chạy notebook thủ công (khuyến nghị)
```
Mở Jupyter → chọn kernel "Python (Predictive Maintenance)"
Chạy lần lượt: 01 → 02 → 03 → 04 → 04b → 05
```
**Ưu điểm**: Thấy biểu đồ, đọc kết quả, tương tác được

### Cách 2: Chạy tự động qua script
```bash
# Qua CLI (không có biểu đồ)
python scripts/run_pipeline.py

# Qua papermill (chạy notebook, có biểu đồ)
python scripts/run_papermill.py
```
**Ưu điểm**: Nhanh, tự động hóa, tái tạo được
