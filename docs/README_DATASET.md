# 📋 Giải thích Bộ dữ liệu — AI4I 2020 Predictive Maintenance

> **Nguồn**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
> **File gốc**: `data/raw/ai4i2020.csv` (10 000 dòng × 14 cột)

---

## 1. Bối cảnh

Bộ dữ liệu mô phỏng dữ liệu cảm biến thu thập từ **dây chuyền sản xuất công nghiệp**.
Mỗi dòng tương ứng với **một lần quan sát máy** (1 chu kỳ vận hành), bao gồm các phép đo
nhiệt độ, tốc độ, lực xoắn, độ mòn dao cụ, cùng với nhãn xác định **máy có hỏng hay không**.

---

## 2. Danh sách thuộc tính (14 cột)

### 2.1 Cột định danh

| # | Tên cột | Kiểu | Mô tả | Ví dụ |
|---|---------|------|-------|-------|
| 1 | `UDI` | int | **Unique Data Identifier** — Số thứ tự quan sát (1 → 10 000). Được dùng như chỉ số thời gian giả (pseudo-time index) vì bộ dữ liệu không có cột thời gian thực. | 1, 2, …, 10000 |
| 2 | `Product ID` | string | Mã sản phẩm, gồm ký tự loại (`L`/`M`/`H`) + số serial. Ví dụ `L50096`. | L50096, M14860, H9537 |

### 2.2 Cột phân loại

| # | Tên cột | Kiểu | Mô tả | Phân bố |
|---|---------|------|-------|---------|
| 3 | `Type` | string | **Chất lượng sản phẩm**: `L` = Low (thấp), `M` = Medium (trung bình), `H` = High (cao). Ảnh hưởng đến khả năng hỏng. | L: 6 000 (60%), M: 2 997 (30%), H: 1 003 (10%) |

### 2.3 Cột cảm biến (số liệu liên tục)

| # | Tên cột | Đơn vị | Mô tả chi tiết | Khoảng giá trị |
|---|---------|--------|----------------|----------------|
| 4 | `Air temperature [K]` | Kelvin | Nhiệt độ không khí xung quanh máy. Nhiệt độ cao hơn → nguy cơ quá nhiệt tăng. | 295.3 – 304.5 K (~22–31 °C) |
| 5 | `Process temperature [K]` | Kelvin | Nhiệt độ quá trình gia công bên trong máy. Luôn cao hơn Air temp khoảng 10 K. | 305.7 – 313.8 K (~33–41 °C) |
| 6 | `Rotational speed [rpm]` | vòng/phút | Tốc độ quay trục chính. Tốc độ quá cao hoặc quá thấp đều có thể gây lỗi. | 1 168 – 2 886 rpm |
| 7 | `Torque [Nm]` | Newton·mét | Mô-men xoắn đặt lên dao cắt. Torque cao quá → quá tải; thấp quá → cắt không đủ. | 3.8 – 76.6 Nm |
| 8 | `Tool wear [min]` | phút | Thời gian sử dụng tích lũy của dao cụ. Càng lớn → dao càng mòn → nguy cơ hỏng tăng. | 0 – 253 phút |

### 2.4 Biến mục tiêu (Target)

| # | Tên cột | Kiểu | Mô tả | Phân bố |
|---|---------|------|-------|---------|
| 9 | **`Machine failure`** | binary (0/1) | **Nhãn chính** — Máy có bị hỏng hay không. `1` = hỏng, `0` = bình thường. | 0: 9 661 (96.61%), 1: 339 (3.39%) |

### 2.5 Các cột loại lỗi (Failure Types)

> ⚠️ **CẢNH BÁO RÒ RỈ DỮ LIỆU (Data Leakage)**:
> 5 cột dưới đây là **thành phần cấu thành** biến `Machine failure`.
> Nếu bất kỳ cột nào = 1 → `Machine failure` = 1.
> **KHÔNG ĐƯỢC** dùng làm feature khi dự đoán `Machine failure`, nếu không mô hình sẽ đạt ~100% accuracy giả tạo.

| # | Tên cột | Viết tắt | Mô tả | Số lượng |
|---|---------|----------|-------|---------|
| 10 | `TWF` | Tool Wear Failure | Lỗi do **mòn dao cụ** — dao cắt bị mài mòn vượt ngưỡng an toàn. | 46 (0.46%) |
| 11 | `HDF` | Heat Dissipation Failure | Lỗi do **tản nhiệt kém** — chênh lệch nhiệt độ (Process − Air) quá thấp + tốc độ quay thấp. | 115 (1.15%) |
| 12 | `PWF` | Power Failure | Lỗi do **công suất** — tích Torque × Speed nằm ngoài khoảng vận hành an toàn. | 95 (0.95%) |
| 13 | `OSF` | Overstrain Failure | Lỗi do **quá tải** — tổ hợp Tool wear × Torque vượt ngưỡng cho phép (phụ thuộc Type). | 98 (0.98%) |
| 14 | `RNF` | Random Failure | Lỗi **ngẫu nhiên** — máy hỏng không xác định rõ nguyên nhân (xác suất 0.1%). | 19 (0.19%) |

---

## 3. Quan hệ giữa Failure Types và Machine failure

```
Machine failure = max(TWF, HDF, PWF, OSF, RNF)
```

- Nếu **bất kỳ** cột failure type = 1 → `Machine failure` = 1
- Có **24 trường hợp** xảy ra **đồng thời nhiều loại lỗi** (ví dụ HDF + OSF cùng lúc)
- Có **4 trường hợp** `Machine failure` = 1 nhưng tất cả 5 failure types đều = 0 (lỗi không phân loại được)

---

## 4. Đặc điểm quan trọng của dữ liệu

### 4.1 Mất cân bằng nghiêm trọng (Class Imbalance)
```
Bình thường (0) : 9 661  (96.61%)
Hỏng       (1) :   339  ( 3.39%)
Tỷ lệ            :  28.5 : 1
```
→ Phải dùng kỹ thuật xử lý imbalanced: `class_weight="balanced"`, `scale_pos_weight`, PR-AUC, F1-score

### 4.2 Không có dữ liệu thiếu (No Missing Values)
- Tất cả 14 cột đều có đầy đủ 10 000 giá trị.

### 4.3 Không có dòng trùng lặp (No Duplicates)

### 4.4 Không có cột thời gian thực
- `UDI` được dùng như proxy cho thứ tự thời gian.
- Khi train/test split theo thời gian: lấy 80% đầu làm train, 20% cuối làm test.

### 4.5 Tương quan giữa các biến
| Cặp biến | Tương quan | Ý nghĩa |
|----------|-----------|---------|
| Air temp ↔ Process temp | +0.88 | Rất cao (cùng tăng cùng giảm) — nhiệt độ xử lý phụ thuộc môi trường |
| Torque ↔ Rotational speed | −0.88 | Nghịch tương quan mạnh — tốc độ tăng thì lực xoắn giảm (vật lý) |
| Tool wear ↔ Machine failure | +0.11 | Yếu — mòn dao tăng nhẹ nguy cơ hỏng |

---

## 5. Các feature được tạo thêm (Feature Engineering)

Ngoài 14 cột gốc, pipeline tạo thêm các feature phái sinh:

| Feature mới | Công thức | Ý nghĩa |
|-------------|----------|---------|
| `temp_diff` | Process temp − Air temp | Hiệu suất tản nhiệt |
| `power` | Torque × Speed × 2π/60 | Công suất cơ học (Watt) |
| `torque_speed_ratio` | Torque / Speed | Tỷ lệ lực vs tốc độ |
| `wear_torque` | Tool wear × Torque | Tương tác mòn–lực |
| `tw_bin_*` | Chia Tool wear thành 5 bin | Phân nhóm mức mòn |
| `*_lag_N` | Giá trị cách N bước | Xu hướng thời gian |
| `*_rolling_mean_W` | Trung bình trượt W bước | Xu hướng trung bình |
| `*_rolling_std_W` | Độ lệch chuẩn trượt W bước | Biến động gần đây |

---

## 6. Cách sử dụng trong dự án

```python
from src.data.loader import load_raw_data, create_data_dictionary

df = load_raw_data()              # → DataFrame 10000 × 14
dd = create_data_dictionary(df)   # → Bảng mô tả từng cột
```
