# 📊 Topic 16 – Phân tích lỗi sản xuất & Dự đoán lỗi

> **Production Error Analysis & Failure Prediction**
> Môn: Data Mining · Đề tài 16

---

## 1. Giới thiệu

Dự án sử dụng bộ dữ liệu **AI4I 2020 Predictive Maintenance** (UCI Machine Learning Repository) gồm 10 000 bản ghi cảm biến từ dây chuyền sản xuất, nhằm:

| Mục tiêu | Phương pháp |
|-----------|-------------|
| Khám phá & tiền xử lý dữ liệu | EDA, xử lý ngoại lai, mã hoá, chuẩn hoá |
| Khai phá luật kết hợp | Apriori (mlxtend) |
| Phân cụm & phát hiện bất thường | K‑Means, DBSCAN, HAC, Isolation Forest, LOF |
| Phân loại lỗi máy | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| Hồi quy (Tool wear) | Linear, RF, GBR, XGBoost |
| Dự báo chuỗi thời gian | ARIMA, Lag Regression |
| Học bán giám sát | Self‑Training, Label Spreading (5 / 10 / 20 % nhãn) |

---

## 2. Nguồn dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| Tên | AI4I 2020 Predictive Maintenance Dataset |
| Nguồn | <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| Kích thước | 10 000 dòng × 14 cột |
| Biến mục tiêu | `Machine failure` (0 / 1, tỷ lệ lỗi ≈ 3.39 %) |

Tải CSV và đặt vào `data/raw/ai4i2020.csv`.

---

## 3. Cấu trúc thư mục

```
DATA_MINING_PROJECT/
├── configs/
│   └── params.yaml              # Siêu tham số
├── data/
│   ├── raw/                     # Dữ liệu gốc (git‑ignored)
│   └── processed/               # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/          loader.py · cleaner.py
│   ├── features/      builder.py
│   ├── mining/        association.py · clustering.py · anomaly.py
│   ├── models/        supervised.py · semi_supervised.py · forecasting.py
│   ├── evaluation/    metrics.py · report.py
│   └── visualization/ plots.py
├── scripts/
│   ├── run_pipeline.py          # Chạy pipeline CLI
│   └── run_papermill.py         # Chạy notebook tự động
├── outputs/                     # Bảng kết quả, hình ảnh (git‑ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Cài đặt & Chạy

### 4.1 Tạo môi trường

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4.2 Chạy notebook thủ công

Mở Jupyter và chạy lần lượt:

```
01_eda  →  02_preprocess_feature  →  03_mining_or_clustering  →  04_modeling  →  04b_semi_supervised  →  05_evaluation_report
```

### 4.3 Chạy toàn bộ bằng script

```bash
# Qua dòng lệnh (không notebook)
python scripts/run_pipeline.py

# Hoặc chạy notebook tự động
python scripts/run_papermill.py
```

---

## 5. Kết quả chính

> *(Sẽ được cập nhật sau khi chạy pipeline.)*

- **Classification**: PR‑AUC, F1, Recall (xử lý imbalanced ~ 28.5:1)
- **Clustering**: Silhouette, Davies‑Bouldin
- **Semi‑supervised**: So sánh Self‑Training & Label Spreading ở 5 / 10 / 20 % nhãn
- **Association Rules**: Các luật kết hợp liên quan đến Machine failure

---

## 6. Rubric

| Mục | Nội dung | Điểm |
|-----|----------|------|
| A | Chọn đề tài & dữ liệu phù hợp | 1 |
| B | EDA / Tiền xử lý | 1 |
| C | Feature Engineering | 1 |
| D | Khai phá (Apriori / Clustering) | 1 |
| E | Mô hình (Phân loại / Hồi quy) | 1 |
| F | Đánh giá (metrics, so sánh) | 1 |
| G | Trình bày / Cấu trúc repo | 1 |
| H | Đề xuất & phản biện | 1 |
| +1 | Chuỗi thời gian | 1 |
| +1 | Semi‑supervised | 1 |
| **Tổng** | | **10** |

---

## 7. Giấy phép

Dự án phục vụ mục đích học tập. Dữ liệu thuộc bản quyền UCI ML Repository (CC BY 4.0).
