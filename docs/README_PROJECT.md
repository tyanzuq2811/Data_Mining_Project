# 🎯 Giải thích Bài toán — Phân tích lỗi sản xuất & Dự đoán lỗi

> **Đề tài 16**: Production Error Analysis & Failure Prediction
> **Bộ dữ liệu**: AI4I 2020 Predictive Maintenance (UCI ML Repository)

---

## 1. Bài toán đang giải quyết cái gì?

### Bối cảnh thực tế
Trong nhà máy sản xuất, máy móc hoạt động liên tục và có thể hỏng bất kỳ lúc nào.
Mỗi lần máy hỏng → **dừng dây chuyền** → thiệt hại lớn về thời gian và chi phí.

**Câu hỏi cần trả lời:**
1. **Tại sao máy hỏng?** → Phân tích nguyên nhân gốc (root cause analysis)
2. **Khi nào máy sẽ hỏng?** → Dự đoán trước khi hỏng (predictive maintenance)
3. **Có quy luật nào dẫn đến hỏng?** → Tìm luật kết hợp (association rules)
4. **Máy hoạt động bình thường vs bất thường?** → Phát hiện bất thường (anomaly detection)

### Tóm tắt một câu
> Dùng dữ liệu cảm biến (nhiệt độ, tốc độ, lực, độ mòn) để **phân tích** tại sao máy hỏng
> và **dự đoán** liệu máy có hỏng ở chu kỳ vận hành tiếp theo hay không.

---

## 2. Các bài toán con trong dự án

Dự án không chỉ giải 1 bài toán mà bao gồm **7 bài toán con**, phản ánh toàn bộ quy trình Data Mining:

### 2.1 Khai phá dữ liệu (EDA)
- **Mục tiêu**: Hiểu dữ liệu trước khi làm bất cứ điều gì
- **Câu hỏi**: Dữ liệu có bao nhiêu dòng? Có missing không? Phân bố ra sao? Có tương quan gì?
- **Kết quả**: Phát hiện mất cân bằng 28.5:1, cảnh báo data leakage từ failure type columns

### 2.2 Tiền xử lý & Feature Engineering
- **Mục tiêu**: Biến dữ liệu thô thành dạng sẵn sàng cho mô hình
- **Làm gì**: Xử lý ngoại lai (IQR), mã hóa (One-Hot), chuẩn hóa (StandardScaler), tạo feature mới (temp_diff, power, lag, rolling)

### 2.3 Khai phá luật kết hợp (Association Rules — Apriori)
- **Mục tiêu**: Tìm quy luật dạng "NẾU ... THÌ máy hỏng"
- **Ví dụ kết quả**: "NẾU torque_high VÀ tool_wear_high THÌ Machine failure" (confidence=0.8, lift=4.2)
- **Ý nghĩa**: Giúp kỹ sư biết tổ hợp điều kiện nào gây hỏng

### 2.4 Phân cụm (Clustering)
- **Mục tiêu**: Nhóm các chu kỳ vận hành thành các cụm tương tự nhau
- **Phương pháp**: KMeans (k=2–8), DBSCAN, HAC (Hierarchical Agglomerative Clustering)
- **Ý nghĩa**: Xem cụm nào có tỷ lệ hỏng cao → xác định "chế độ vận hành nguy hiểm"

### 2.5 Phát hiện bất thường (Anomaly Detection)
- **Mục tiêu**: So sánh — liệu phát hiện bất thường (không cần nhãn) có trùng với lỗi thực không?
- **Phương pháp**: Isolation Forest, LOF (Local Outlier Factor)
- **Ý nghĩa**: Kiểm chứng xem các lần hỏng có thực sự là "bất thường" trong dữ liệu cảm biến

### 2.6 Phân loại & Hồi quy (Supervised Learning)
| Bài toán | Target | Phương pháp | Mục tiêu |
|----------|--------|-------------|----------|
| **Phân loại** (Classification) | `Machine failure` (0/1) | LogReg, RF, GBR, XGBoost, LightGBM | Dự đoán máy có hỏng không |
| **Hồi quy** (Regression) | `Tool wear [min]` | Linear, RF, GBR, XGBoost | Dự đoán độ mòn dao cụ |
| **Chuỗi thời gian** (Time Series) | Tỷ lệ hỏng trung bình trượt | ARIMA, Lag Regression | Dự đoán xu hướng hỏng theo thời gian |

### 2.7 Học bán giám sát (Semi-supervised Learning)
- **Mục tiêu**: Nếu chỉ có 5–20% dữ liệu được gán nhãn, mô hình còn hoạt động tốt không?
- **Phương pháp**: Self-Training, Label Spreading
- **Ý nghĩa thực tế**: Trong thực tế, việc gán nhãn (lỗi/không lỗi) rất tốn kém và chậm

---

## 3. Thách thức chính

| Thách thức | Mô tả | Giải pháp trong dự án |
|-----------|-------|----------------------|
| **Mất cân bằng 28.5:1** | Chỉ 3.39% mẫu hỏng — model dễ dự đoán toàn "không hỏng" và vẫn đạt 96.6% accuracy | `class_weight="balanced"`, `scale_pos_weight=28.5`, dùng PR-AUC, F1 thay vì Accuracy |
| **Data Leakage** | 5 cột TWF/HDF/PWF/OSF/RNF là thành phần của target → dùng làm feature = gian lận | Loại bỏ hoàn toàn 5 cột này khỏi feature set |
| **Không có timestamp** | Bộ dữ liệu không có thời gian thật → khó làm time series chính xác | Dùng UDI làm proxy thời gian, chia train/test theo thứ tự |
| **Ít failure types** | TWF chỉ có 46 mẫu, RNF chỉ 19 → khó phân loại từng loại lỗi | Tập trung vào binary (hỏng/không hỏng) thay vì multi-class |

---

## 4. Tiêu chí đánh giá thành công

| Bài toán | Metric chính | Tại sao? |
|----------|-------------|---------|
| Classification | **PR-AUC**, F1, Recall | Với dữ liệu imbalanced, Accuracy vô nghĩa. PR-AUC đánh giá khả năng tìm đúng lỗi |
| Regression | MAE, RMSE, R² | Metrics chuẩn cho hồi quy |
| Clustering | Silhouette, Davies-Bouldin | Đo chất lượng phân cụm (không cần nhãn) |
| Time Series | MAE, RMSE | Đo sai số dự báo |
| Semi-supervised | F1 so với supervised baseline | Xem mất bao nhiêu hiệu suất khi ít nhãn |

---

## 5. Kết quả mong đợi

Sau khi chạy xong toàn bộ pipeline, bạn sẽ có:

1. **Báo cáo EDA** — Hiểu rõ dữ liệu, phân bố, tương quan, rủi ro
2. **Dữ liệu đã xử lý** — Sạch, có feature mới, sẵn sàng modeling
3. **Luật kết hợp** — Các quy luật dạng "NẾU ... THÌ hỏng" với confidence & lift
4. **Kết quả phân cụm** — Bảng so sánh KMeans/DBSCAN/HAC, profile từng cụm
5. **Bảng so sánh 5 classifier** — F1, PR-AUC, ROC-AUC, training time
6. **Biểu đồ ROC & PR curves** — So sánh trực quan
7. **Feature importance** — Biến nào quan trọng nhất cho dự đoán
8. **Kết quả semi-supervised** — F1 ở 5/10/20% nhãn vs 100% nhãn
9. **7 đề xuất hành động** — Insights thực tiễn cho nhà máy
