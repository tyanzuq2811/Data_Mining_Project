# 📂 Giải thích Cấu trúc Dự án — Từng Folder, Từng File

> Mỗi file trong dự án đều có chức năng rõ ràng. Tài liệu này giải thích **tất cả**.

---

## Tổng quan cây thư mục

```
DATA_MINING_PROJECT/
│
├── 📄 README.md                  ← Tổng quan dự án (giới thiệu ngắn)
├── 📄 requirements.txt           ← Danh sách thư viện Python cần cài
├── 📄 .gitignore                 ← File nào Git KHÔNG theo dõi
├── 📄 DATA_MINING_PROJECT.csv    ← File CSV gốc ban đầu
│
├── 📁 configs/                   ← CẤU HÌNH — Tham số cho toàn bộ dự án
│   └── params.yaml
│
├── 📁 data/                      ← DỮ LIỆU
│   ├── raw/                      ← Dữ liệu gốc (chưa xử lý)
│   │   └── ai4i2020.csv
│   └── processed/                ← Dữ liệu đã xử lý (tạo khi chạy notebook 02)
│       ├── ai4i2020_processed.parquet
│       ├── ai4i2020_processed.csv
│       └── feature_info.json
│
├── 📁 docs/                      ← TÀI LIỆU giải thích chi tiết
│   ├── README_DATASET.md
│   ├── README_PROJECT.md
│   ├── README_STRUCTURE.md       ← (file này)
│   └── README_PIPELINE.md
│
├── 📁 src/                       ← MÃ NGUỒN — Tất cả logic xử lý
│   ├── data/                     ← Module đọc & làm sạch dữ liệu
│   ├── features/                 ← Module tạo feature mới
│   ├── mining/                   ← Module khai phá (Apriori, Clustering, Anomaly)
│   ├── models/                   ← Module mô hình (Classification, Regression, Semi-supervised, Time Series)
│   ├── evaluation/               ← Module đánh giá & báo cáo
│   └── visualization/            ← Module vẽ biểu đồ
│
├── 📁 notebooks/                 ← NOTEBOOK — Nơi chạy & trình bày kết quả
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
│
├── 📁 scripts/                   ← SCRIPT — Chạy tự động không cần notebook
│   ├── run_pipeline.py
│   └── run_papermill.py
│
└── 📁 outputs/                   ← KẾT QUẢ — Tạo khi chạy pipeline
    ├── figures/                  ← Biểu đồ PNG
    ├── tables/                   ← Bảng kết quả CSV
    ├── models/                   ← Model đã train (pickle)
    └── reports/                  ← Báo cáo văn bản
```

---

## Chi tiết từng file

### 📁 `configs/` — Cấu hình

| File | Chức năng |
|------|----------|
| **`params.yaml`** | **Trung tâm điều khiển** của toàn bộ dự án. Chứa MỌI tham số: đường dẫn file, tên cột, siêu tham số mô hình, ngưỡng Apriori, cấu hình clustering, ... Tất cả notebook & module đều đọc file này. Khi muốn thay đổi tham số → chỉ cần sửa 1 file duy nhất |

---

### 📁 `data/` — Dữ liệu

| File | Chức năng |
|------|----------|
| **`raw/ai4i2020.csv`** | Bản sao của CSV gốc. Không bao giờ bị sửa đổi — luôn giữ nguyên trạng thái ban đầu |
| **`processed/ai4i2020_processed.parquet`** | Dữ liệu sau khi xử lý ngoại lai + mã hóa + tạo feature. Format Parquet (nhanh, nhỏ) |
| **`processed/ai4i2020_processed.csv`** | Bản CSV tương đương (để dễ mở bằng Excel nếu cần) |
| **`processed/feature_info.json`** | Danh sách: feature nào dùng để train, target là gì, cột nào bị loại vì leakage |

---

### 📁 `src/data/` — Module đọc & làm sạch dữ liệu

| File | Class/Function | Chức năng |
|------|---------------|----------|
| **`loader.py`** | `load_params()` | Đọc file `params.yaml` thành dictionary |
| | `load_raw_data()` | Đọc CSV gốc từ `data/raw/` |
| | `validate_schema()` | Kiểm tra dữ liệu có đúng cấu trúc (đủ cột, đúng kiểu) |
| | `load_processed_data()` | Đọc dữ liệu đã xử lý từ `data/processed/` |
| | `get_data_summary()` | Thống kê: shape, missing, duplicates, ... |
| | `create_data_dictionary()` | Tạo bảng mô tả từng cột (tên, kiểu, ý nghĩa) |
| **`cleaner.py`** | `DataCleaner` class | Pipeline làm sạch tuần tự: xử lý missing → loại duplicate → xử lý outlier (IQR) → mã hóa categorical (One-Hot) → chuẩn hóa (StandardScaler). Lưu thống kê before/after |

---

### 📁 `src/features/` — Module tạo feature

| File | Class/Function | Chức năng |
|------|---------------|----------|
| **`builder.py`** | `FeatureBuilder` class | Tạo toàn bộ feature mới: `temp_diff`, `power`, `torque_speed_ratio`, `wear_torque`, phân bin Tool wear, lag features (lùi 1/3/5/10 bước), rolling mean/std (cửa sổ 5/10/20), interaction features |
| | `get_apriori_features()` | Chuyển dữ liệu số → nhị phân (low/normal/high) để dùng cho Apriori |

---

### 📁 `src/mining/` — Module khai phá

| File | Class | Chức năng |
|------|-------|----------|
| **`association.py`** | `AssociationMiner` | Chạy **thuật toán Apriori** (mlxtend): tìm tập phổ biến (frequent itemsets), sinh luật kết hợp (association rules), lọc luật liên quan đến Machine failure, luật theo từng loại lỗi |
| **`clustering.py`** | `ClusterAnalyzer` | Chạy **3 thuật toán phân cụm**: KMeans (thử k=2–8), DBSCAN (grid search eps × min_samples), HAC. Đánh giá bằng Silhouette, Davies-Bouldin, Calinski-Harabasz. Profile từng cụm (trung bình feature, tỷ lệ hỏng) |
| **`anomaly.py`** | `AnomalyDetector` | Chạy **Isolation Forest** và **LOF** để phát hiện bất thường. So sánh kết quả với nhãn hỏng thực tế (precision, recall, F1) |

---

### 📁 `src/models/` — Module mô hình

| File | Class | Chức năng |
|------|-------|----------|
| **`supervised.py`** | `SupervisedTrainer` | **Phân loại**: Train 5 model (LogReg, RF, GBR, XGBoost, LightGBM) với xử lý imbalanced. Cross-validate. Feature importance. **Hồi quy**: Train 4 model (Linear, RF, GBR, XGBoost) để dự đoán Tool wear |
| **`semi_supervised.py`** | `SemiSupervisedTrainer` | Thí nghiệm **học bán giám sát**: che nhãn (giữ 5/10/20%), train bằng Self-Training & Label Spreading, so sánh với baseline supervised. Phân tích pseudo-label. Learning curve |
| **`forecasting.py`** | `TimeSeriesForecaster` | **Chuỗi thời gian**: Chia train/test theo thời gian. Fit ARIMA(2,1,2). Fit Lag Regression (GBR với lag features). So sánh MAE/RMSE |

---

### 📁 `src/evaluation/` — Module đánh giá

| File | Class/Function | Chức năng |
|------|---------------|----------|
| **`metrics.py`** | `classification_metrics()` | Tính accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| | `regression_metrics()` | Tính MAE, RMSE, R², MAPE |
| | `clustering_metrics()` | Tính Silhouette, Davies-Bouldin, Calinski-Harabasz |
| | `error_analysis()` | Phân tích FP/FN, tỷ lệ miss, index của các mẫu bị dự đoán sai |
| **`report.py`** | `ReportGenerator` class | Gom tất cả bảng kết quả + insights → lưu thành CSV và TXT vào `outputs/` |

---

### 📁 `src/visualization/` — Module vẽ biểu đồ

| File | Các hàm (20+) | Chức năng |
|------|--------------|----------|
| **`plots.py`** | `plot_target_distribution()` | Biểu đồ phân bố target (bar + pie) |
| | `plot_failure_types()` | Biểu đồ số lượng từng loại lỗi |
| | `plot_numeric_distributions()` | Grid histogram + boxplot cho mỗi biến số |
| | `plot_correlation_matrix()` | Ma trận tương quan (heatmap tam giác dưới) |
| | `plot_confusion_matrix()` | Confusion matrix (heatmap) |
| | `plot_roc_curves()` | Đường cong ROC cho nhiều model |
| | `plot_precision_recall_curves()` | Đường cong PR cho nhiều model |
| | `plot_feature_importance()` | Biểu đồ thanh ngang (top features) |
| | `plot_elbow()` | Elbow method cho KMeans |
| | `plot_silhouette_scores()` | Silhouette score vs k |
| | `plot_learning_curve()` | F1 vs % nhãn (semi-supervised) |
| | `plot_residuals()` | Biểu đồ phần dư (regression) |
| | ... | Và nhiều hàm khác |

---

### 📁 `notebooks/` — Các Notebook (nơi chạy & trình bày)

| Notebook | Số cell | Chức năng |
|----------|---------|----------|
| **`01_eda.ipynb`** | 23 | Khám phá dữ liệu: phân bố, tương quan, missing, duplicates, phân tích rủi ro (imbalance, leakage) |
| **`02_preprocess_feature.ipynb`** | 17 | Làm sạch + tạo feature → lưu dữ liệu processed |
| **`03_mining_or_clustering.ipynb`** | 24 | Apriori (luật kết hợp) + KMeans/DBSCAN/HAC + Anomaly Detection |
| **`04_modeling.ipynb`** | 23 | Classification (5 model) + Regression (4 model) + Time Series |
| **`04b_semi_supervised.ipynb`** | 13 | Semi-supervised: Self-Training & Label Spreading ở 5/10/20% |
| **`05_evaluation_report.ipynb`** | 19 | Tổng hợp kết quả, so sánh, 7 insights, lưu artifacts |

> **Quan trọng**: Notebook KHÔNG chứa logic xử lý — chúng chỉ gọi hàm từ `src/`.
> Toàn bộ logic nằm trong `src/` modules. Notebook là nơi **trình bày kết quả**.

---

### 📁 `scripts/` — Script tự động

| File | Chức năng |
|------|----------|
| **`run_pipeline.py`** | Chạy toàn bộ pipeline qua dòng lệnh (không cần mở Jupyter). Hỗ trợ `--step` để chạy từng bước riêng |
| **`run_papermill.py`** | Chạy tuần tự 6 notebook bằng thư viện `papermill`. Kết quả lưu vào `outputs/executed_notebooks/` |

---

### 📁 `outputs/` — Kết quả (tạo khi chạy)

| Thư mục | Nội dung |
|---------|----------|
| **`figures/`** | Tất cả biểu đồ PNG (EDA, ROC, PR, confusion matrix, elbow, ...) |
| **`tables/`** | Bảng CSV (so sánh model, cross-validation, clustering scores, ...) |
| **`models/`** | File `.pkl` chứa model đã train (best classifier) |
| **`reports/`** | `insights.txt` — 7 đề xuất hành động |

---

### 📄 File gốc (root)

| File | Chức năng |
|------|----------|
| **`README.md`** | Giới thiệu tổng quan dự án (cho người đọc lần đầu) |
| **`requirements.txt`** | Danh sách thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn, mlxtend, xgboost, lightgbm, statsmodels, pyarrow, pyyaml, papermill, jupyter, ipykernel |
| **`.gitignore`** | Bỏ qua: `data/raw/`, `data/processed/`, `outputs/`, `__pycache__/`, `.ipynb_checkpoints/`, IDE files |
| **`DATA_MINING_PROJECT.csv`** | File CSV gốc ban đầu (bản gốc, giữ nguyên ở root) |

---

## Triết lý thiết kế

```
┌─────────────────────────────────────────────────────┐
│  configs/params.yaml   ← Tham số tập trung          │
│         ↓                                            │
│  src/ modules          ← Logic (KHÔNG trình bày)     │
│         ↓                                            │
│  notebooks/            ← Gọi src/, hiển thị kết quả │
│         ↓                                            │
│  outputs/              ← Lưu artifacts cuối cùng     │
└─────────────────────────────────────────────────────┘
```

- **Tách biệt hoàn toàn** giữa logic (`src/`) và trình bày (`notebooks/`)
- **Không module nào import module khác** trong `src/` — tất cả độc lập
- **Dữ liệu truyền qua filesystem**: raw → processed (parquet) → outputs (CSV, PNG, pkl)
- **Một file cấu hình duy nhất** (`params.yaml`) kiểm soát mọi thứ
