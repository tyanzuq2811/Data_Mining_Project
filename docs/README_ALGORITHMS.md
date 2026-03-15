# Sổ Tay Thuật Toán Và Cấu Hình Dự Án

Tài liệu này là phiên bản học lại từ đầu, dành cho trường hợp bạn quên kiến thức.
Mỗi thuật toán đều có cùng khung giải thích:
1. Trực giác dễ hiểu
2. Định nghĩa chính xác
3. Quy trình chạy từng bước
4. Tham số quan trọng và ý nghĩa
5. Cấu hình thật đang dùng trong đồ án
6. Cách đọc kết quả và ví dụ đời thực

Nguồn đối chiếu chính trong code:
- configs/params.yaml
- src/data/cleaner.py
- src/features/builder.py
- src/mining/association.py
- src/mining/clustering.py
- src/mining/anomaly.py
- src/models/supervised.py
- src/models/forecasting.py
- src/models/semi_supervised.py

---

## 1) Bức tranh tổng thể

### 1.1 Bài toán lớn
Bạn đang giải bài toán Predictive Maintenance: dùng dữ liệu cảm biến để dự đoán máy có sắp hỏng hay không, đồng thời phân tích nguyên nhân và trạng thái vận hành.

### 1.2 Các bài toán con trong đồ án
1. Classification: dự đoán hỏng/không hỏng.
2. Association Rules: tìm luật kiểu “nếu A và B thì thường hỏng”.
3. Clustering: nhóm chế độ vận hành.
4. Anomaly Detection: dò điểm bất thường.
5. Regression: dự đoán Tool wear.
6. Time Series: dự báo theo thứ tự thời gian giả (UDI).
7. Semi-supervised: học khi thiếu nhãn.

### 1.3 Thách thức đặc thù của dữ liệu AI4I
1. Mất cân bằng rất nặng: tỷ lệ hỏng chỉ khoảng 3.39%.
2. Data leakage: 5 cột TWF/HDF/PWF/OSF/RNF liên quan trực tiếp target.
3. Không có timestamp thật: dùng UDI làm chỉ số thời gian gần đúng.

---

## 2) Preprocessing và vì sao phải làm

### 2.1 Missing values
Trực giác:
- Mô hình không hiểu ô trống. Nếu để trống, nhiều thuật toán sẽ lỗi hoặc học sai.

Trong đồ án này:
- Cột số: điền median (trung vị).
- Cột phân loại: điền mode (giá trị xuất hiện nhiều nhất).

Vì sao dùng median cho số:
- Median ít bị kéo lệch bởi outlier hơn mean.

Ví dụ:
- Torque có vài giá trị rỗng, điền median giúp dữ liệu ổn định hơn so với mean khi phân bố lệch.

Code:
- cleaner.py -> handle_missing()

### 2.2 Duplicate handling
Trực giác:
- Dữ liệu trùng làm mô hình “tin” một mẫu quá nhiều, gây lệch học.

Trong đồ án này:
- Kiểm tra duplicate trên nội dung kỹ thuật.
- Bỏ qua cột ID (UDI, Product ID) khi dò trùng.

Lý do bỏ qua ID:
- Hai dòng có bản chất giống nhau nhưng ID khác nhau vẫn là trùng thông tin vận hành.

Code:
- cleaner.py -> handle_duplicates()

### 2.3 Outlier handling bằng IQR
Trực giác:
- Outlier giống “điểm cực đoan”, có thể làm mô hình bị kéo lệch.

Công thức:
- Q1 = phân vị 25%
- Q3 = phân vị 75%
- IQR = Q3 - Q1
- Lower = Q1 - k*IQR
- Upper = Q3 + k*IQR

Trong đồ án này:
- method = iqr
- threshold k = 1.5
- xử lý bằng clip (cắt về ngưỡng), không xóa dòng

Vì sao clip thay vì xóa:
- Dataset không quá lớn, xóa nhiều sẽ mất thông tin.
- Clip giữ bản ghi nhưng giảm ảnh hưởng cực đoan.

Code:
- cleaner.py -> handle_outliers()

### 2.4 Encoding categorical
Trực giác:
- Mô hình ML cần số, không hiểu chữ L/M/H trực tiếp.

Trong đồ án này:
- one-hot cho Type
- tạo Type_L, Type_M, Type_H

Vì sao one-hot:
- Không áp đặt thứ bậc giả giữa L/M/H.

Code:
- cleaner.py -> encode_categorical()

### 2.5 Scaling
Trực giác:
- Nếu một cột có đơn vị lớn hơn hẳn, mô hình dễ ưu tiên sai cột đó.

Trong đồ án này:
- StandardScaler
- đưa dữ liệu về mean ~ 0 và std ~ 1

Code:
- cleaner.py -> scale_features()

---

## 3) Feature Engineering: phần cực quan trọng

Feature engineering là bước “biến dữ liệu thô thành tín hiệu tốt cho mô hình”.

### 3.1 Nhóm feature dẫn xuất
1. temp_diff = Process temperature - Air temperature
2. power = Torque * Rotational speed * 2*pi/60
3. torque_speed_ratio = Torque / Rotational speed
4. wear_torque = Tool wear * Torque

Trực giác:
- temp_diff phản ánh khả năng tản nhiệt.
- power phản ánh mức tải thực tế.
- wear_torque là tương tác quan trọng với lỗi quá tải.

### 3.2 Bin Tool wear
Trong đồ án:
- bins: [0, 50, 100, 150, 200, 300]
- labels: very_low, low, medium, high, very_high

Lợi ích:
- Biến liên tục thành mức rủi ro dễ diễn giải.
- Hữu ích cho Apriori và dashboard giải thích.

### 3.3 Lag features
Trong đồ án:
- window [1, 3, 5, 10] cho các biến sensor.

Trực giác:
- Hành vi máy thường có quán tính, giá trị trước đó giúp dự đoán hiện tại.

Ví dụ:
- Torque tăng dần 3 bước liên tiếp có thể báo hiệu quá tải dù thời điểm hiện tại chưa vượt ngưỡng.

### 3.4 Rolling features
Trong đồ án:
- rolling mean/std với window [5, 10, 20]

Trực giác:
- mean cho xu hướng ngắn hạn.
- std cho độ dao động bất thường.

### 3.5 Interaction features
Trong đồ án:
- air_temp_x_speed
- proc_temp_x_torque

Trực giác:
- Nhiều lỗi không do một biến đơn lẻ mà do tổ hợp nhiều điều kiện.

Code tổng:
- builder.py -> build()

---

## 4) Association Rules (Apriori)

### 4.1 Trực giác
Hãy tưởng tượng bạn muốn tìm mẫu hành vi lặp lại trong lịch sử:
- “khi mô-men cao và mòn dao cao thì hay hỏng”.
Apriori chính là công cụ phát hiện các mẫu kiểu đó.

### 4.2 Định nghĩa ngắn
Luật kết hợp có dạng:
- IF (antecedents) THEN (consequents)

### 4.3 3 độ đo bắt buộc phải hiểu
1. support:
- Tần suất xuất hiện của cả vế trái và vế phải.

2. confidence:
- Xác suất vế phải đúng khi vế trái đúng.

3. lift:
- Mức tăng xác suất so với ngẫu nhiên.
- lift > 1: có liên hệ dương.
- lift = 1: gần như độc lập.
- lift < 1: liên hệ âm.

### 4.4 Cấu hình đang dùng trong đồ án
Trong params.yaml:
- min_support = 0.01
- min_confidence = 0.5
- min_lift = 1.5
- max_len = 4

Ý nghĩa chọn ngưỡng:
- support 1% để bỏ luật quá hiếm.
- confidence 50% để luật có mức tin cậy tối thiểu.
- lift 1.5 để giữ luật có giá trị thực tế.

### 4.5 Quy trình chạy trong code
1. Rời rạc hóa dữ liệu thành nhị phân (low/normal/high).
2. Chạy Apriori tìm frequent itemsets.
3. Sinh association rules.
4. Lọc theo lift.
5. Lọc riêng luật có hậu quả Machine failure.

Code:
- association.py -> mine()
- association.py -> get_failure_rules()

### 4.6 Ví dụ đọc kết quả
Luật:
- IF (Torque_high, ToolWear_high) THEN (Machine failure)
- support=0.012, confidence=0.78, lift=23

Diễn giải:
1. 1.2% toàn bộ dữ liệu có mẫu này.
2. Khi mẫu này xuất hiện, xác suất hỏng là 78%.
3. Rủi ro cao gấp 23 lần mức nền.

---

## 5) Clustering (Phân cụm)

Clustering giúp trả lời: “máy đang vận hành theo bao nhiêu chế độ khác nhau?”

### 5.1 KMeans
Trực giác:
- Chọn k tâm cụm, kéo mỗi điểm về tâm gần nhất, cập nhật tâm, lặp cho đến ổn định.

Tham số quan trọng:
1. n_clusters (k): số cụm.
2. n_init: số lần khởi tạo lại để tránh rơi vào nghiệm xấu.
3. random_state: tái lập kết quả.

Cấu hình đồ án:
- thử k = 2..8
- n_init=10

Khi nào KMeans phù hợp:
- cụm dạng “tròn” tương đối.
- dữ liệu đã scale tốt.

Code:
- clustering.py -> fit_kmeans()

### 5.2 DBSCAN
Trực giác:
- Cụm là vùng có mật độ cao.
- Điểm lẻ không đủ mật độ sẽ thành noise (-1).

Tham số quan trọng:
1. eps: bán kính lân cận.
2. min_samples: số điểm tối thiểu để là vùng dày.

Cấu hình đồ án:
- eps [0.3, 0.5, 0.7, 1.0]
- min_samples [3, 5, 10]

Điểm mạnh:
- phát hiện cụm hình dạng bất kỳ.
- tự phát hiện noise.

Code:
- clustering.py -> fit_dbscan()

### 5.3 Hierarchical (HAC)
Trực giác:
- Mỗi điểm là một cụm, sau đó gộp dần cụm gần nhau.

Tham số:
1. n_clusters
2. linkage

Trong đồ án:
- linkage = ward
- n_clusters = 2..5

Code:
- clustering.py -> fit_hierarchical()

### 5.4 Cách đánh giá cụm
1. Silhouette: cao hơn là cụm tách tốt hơn.
2. Davies-Bouldin: thấp hơn là cụm gọn, xa nhau.
3. Calinski-Harabasz: cao hơn thường tốt.

Code:
- clustering.py -> _evaluate_clustering()
- clustering.py -> get_best_model()

### 5.5 Ví dụ diễn giải business
Nếu Cluster 2 có failure_rate cao nhất:
1. Xem profile trung bình của cluster.
2. Tìm đặc trưng nổi bật (ví dụ torque cao, temp_diff thấp).
3. Gắn nhãn “chế độ vận hành rủi ro” cho cluster đó.

---

## 6) Anomaly Detection

Mục tiêu: tìm các điểm “lạ” dù không cần nhãn.

### 6.1 Isolation Forest
Trực giác:
- Điểm bất thường thường dễ bị “cô lập” bằng ít lần chia cây hơn.

Tham số chính:
1. contamination: tỷ lệ bất thường kỳ vọng.
2. n_estimators: số cây.

Trong đồ án:
- contamination ~ 0.034 (bám tỷ lệ failure)
- n_estimators = 200

Code:
- anomaly.py -> fit_isolation_forest()

### 6.2 LOF
Trực giác:
- So mật độ cục bộ của một điểm với hàng xóm.
- Nếu mật độ thấp hơn nhiều => điểm bất thường.

Tham số chính:
1. n_neighbors
2. contamination

Trong đồ án:
- n_neighbors = 20
- contamination ~ 0.034

Code:
- anomaly.py -> fit_lof()

### 6.3 One-Class SVM
Trực giác:
- Học biên bao quanh dữ liệu bình thường.
- Điểm nằm ngoài biên là bất thường.

Tham số chính:
1. kernel
2. nu: ước lượng tỷ lệ outlier.

Trong đồ án:
- kernel = rbf
- nu ~ 0.034

Code:
- anomaly.py -> fit_ocsvm()

### 6.4 Đánh giá với nhãn thật
Trong đồ án, anomaly được so với Machine failure để tính:
- Precision, Recall, F1, Accuracy.

Code:
- anomaly.py -> compare_with_actual()

Lưu ý quan trọng:
- Anomaly không phải lúc nào cũng trùng failure thật.
- Vì “lạ” không đồng nghĩa “hỏng”, và ngược lại.

---

## 7) Classification (Bài toán chính)

### 7.1 Vì sao dùng nhiều mô hình
Mỗi mô hình có cách học khác nhau:
1. Logistic Regression: baseline tuyến tính dễ giải thích.
2. Random Forest: mạnh với dữ liệu phi tuyến, ít cần tuning phức tạp.
3. Gradient Boosting: học tuần tự, tối ưu sai số tốt.
4. XGBoost: boosting tối ưu hóa cao, mạnh trên tabular data.
5. LightGBM: nhanh, hiệu quả khi dữ liệu lớn.

### 7.2 Cấu hình thực tế trong code (đang chạy)
Logistic Regression:
- C = 1.0
- max_iter = 1000
- class_weight = balanced

Random Forest:
- n_estimators = 200
- max_depth = 10
- min_samples_split = 5
- class_weight = balanced

Gradient Boosting:
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05

XGBoost (nếu có):
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05
- scale_pos_weight = 28.5

LightGBM (nếu có):
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05
- is_unbalance = true

Lưu ý quan trọng:
- params.yaml có grid tham số cho nghiên cứu.
- Nhưng supervised.py hiện tại khởi tạo bộ tham số cố định ở mức hợp lý.

### 7.3 Tham số nào ảnh hưởng gì
1. n_estimators:
- tăng thì mô hình ổn định hơn nhưng chậm hơn.

2. max_depth:
- sâu quá dễ overfit, nông quá dễ underfit.

3. learning_rate (boosting):
- thấp hơn thường cần nhiều cây hơn.

4. class_weight / scale_pos_weight:
- tăng trọng số lớp hỏng để mô hình không bỏ qua lớp hiếm.

### 7.4 Metrics bắt buộc hiểu
1. Precision: dự đoán hỏng thì đúng bao nhiêu.
2. Recall: hỏng thật bắt được bao nhiêu.
3. F1: cân bằng precision-recall.
4. ROC-AUC: phân biệt tổng quát.
5. PR-AUC: quan trọng khi lớp dương hiếm.

Trong bài này ưu tiên:
- F1 và PR-AUC (vì dữ liệu mất cân bằng).

Code:
- supervised.py -> train_classifiers()
- supervised.py -> cross_validate()

### 7.5 Ví dụ vì sao accuracy dễ gây hiểu lầm
Nếu 10,000 mẫu có 339 hỏng:
- model đoán tất cả là không hỏng vẫn đạt ~96.61% accuracy.
- nhưng recall lớp hỏng ~0.
=> nên dùng F1/PR-AUC.

---

## 8) Regression (Dự đoán Tool wear)

### 8.1 Trực giác
Bạn muốn dự đoán số phút mòn dao cụ liên tục, không phải nhãn 0/1.

### 8.2 Mô hình dùng trong đồ án
1. Linear Regression
2. RandomForestRegressor
3. GradientBoostingRegressor
4. XGBRegressor (nếu cài)

Cấu hình đang chạy:
- RF: n_estimators=200, max_depth=10
- GBR: n_estimators=200, max_depth=5, learning_rate=0.05
- XGB: n_estimators=200, max_depth=5, learning_rate=0.05

### 8.3 Cách đọc metrics
1. MAE: sai số tuyệt đối trung bình (đơn vị phút).
2. RMSE: phạt nặng lỗi lớn.
3. R2: mức giải thích phương sai (gần 1 là tốt).

Ví dụ:
- MAE = 4.8 nghĩa là lệch trung bình khoảng 4.8 phút.

Code:
- supervised.py -> train_regressors()

---

## 9) Time Series Forecasting

### 9.1 Bản chất trong đồ án này
Do không có timestamp thật, bạn giả định:
- UDI tăng dần tương đương thời gian.

Nên train/test phải chia theo thứ tự, không shuffle.

Code:
- forecasting.py -> temporal_train_test_split()

### 9.2 ARIMA
Trực giác:
- Dùng quá khứ của chính chuỗi để dự báo tương lai.

Ý nghĩa order (p,d,q):
1. p: số bậc tự hồi quy.
2. d: số lần sai phân để chuỗi dừng.
3. q: số bậc trung bình trượt của nhiễu.

Trong đồ án:
- order = (2,1,2)
- báo cáo thêm AIC/BIC

Code:
- forecasting.py -> fit_arima()

### 9.3 Lag-feature Regression
Trực giác:
- Biến chuỗi thời gian thành supervised bằng feature trễ.

Trong đồ án:
- dùng GradientBoostingRegressor trên tập feature đã có lag.

Code:
- forecasting.py -> fit_lag_regression()

Khi nào cách này mạnh:
- khi có nhiều biến ngoại sinh (sensor) hỗ trợ dự báo.

---

## 10) Semi-supervised Learning

Mục tiêu:
- Mô phỏng thực tế thiếu nhãn do gán nhãn tốn chi phí.

Trong đồ án:
- tỷ lệ nhãn thử: 5%, 10%, 20%
- so sánh 3 cách: supervised_only, self_training, label_spreading

### 10.1 Supervised-only (baseline)
Trực giác:
- Chỉ học trên phần có nhãn, bỏ phần chưa nhãn.

Ý nghĩa:
- làm mốc để đo lợi ích của semi-supervised.

Code:
- semi_supervised.py -> train_supervised_only()

### 10.2 Self-Training
Trực giác:
- Model tự gán pseudo-label cho mẫu chưa nhãn nếu rất tự tin.
- Sau đó học lại với dữ liệu đã bổ sung pseudo-label.

Tham số quan trọng:
1. threshold: ngưỡng tự tin.
2. max_iter: số vòng lặp tự gán nhãn.

Trong đồ án:
- base model: RandomForestClassifier
- threshold = 0.95
- max_iter = 30

Trade-off:
- threshold cao: ít pseudo-label nhưng chất lượng cao.
- threshold thấp: nhiều pseudo-label nhưng dễ nhiễu.

Code:
- semi_supervised.py -> train_self_training()

### 10.3 Label Spreading
Trực giác:
- Xây đồ thị điểm dữ liệu, điểm gần nhau sẽ “truyền” nhãn cho nhau.

Tham số quan trọng:
1. kernel: cách đo tương đồng (rbf).
2. alpha: mức giữ nhãn cũ so với nhãn lan truyền.
3. max_iter: số vòng lan truyền.

Trong đồ án:
- kernel = rbf
- alpha = 0.2
- max_iter = 100
- giới hạn n_max = 5000 để tránh O(n^2) quá nặng.

Code:
- semi_supervised.py -> train_label_spreading()

### 10.4 Cách đọc kết quả semi-supervised
1. So F1 giữa 3 phương pháp ở từng mức nhãn.
2. Xem đường learning curve khi tăng tỷ lệ nhãn.
3. Phân tích pseudo-label rate để xem độ an toàn.

---

## 11) Mapping tham số: params.yaml đang điều khiển phần nào

### 11.1 Được dùng trực tiếp rõ ràng
1. seed -> hầu hết module mô hình.
2. preprocessing.* -> cleaner.py.
3. feature_engineering.* -> builder.py.
4. mining.apriori.* -> association.py.
5. mining.clustering.* -> clustering.py.
6. semi_supervised.* -> semi_supervised.py.

### 11.2 Khai báo nhưng code đang dùng preset cứng
1. modeling.classification.hyperparams
2. modeling.regression.hyperparams

Lưu ý:
- mục này có thể dùng cho GridSearch sau này, nhưng phiên bản hiện tại train bằng bộ tham số cố định trong supervised.py.

---

## 12) Quy trình học lại đề xuất

Nếu bạn muốn học chắc, đi theo thứ tự này:
1. Đọc mục 2 và 3 (nền dữ liệu + feature).
2. Đọc mục 7 (classification chính).
3. Đọc mục 4, 5, 6 (mining mở rộng).
4. Đọc mục 8, 9 (regression/time-series).
5. Đọc mục 10 (semi-supervised).
6. Cuối cùng đối chiếu mục 11 với params.yaml và code thật.

---

## 13) Bảng nhớ nhanh tham số (cheat sheet)

### 13.1 Apriori
1. min_support tăng -> ít luật hơn.
2. min_confidence tăng -> luật chắc hơn.
3. min_lift tăng -> luật “có giá trị” hơn.

### 13.2 KMeans
1. k nhỏ -> cụm thô.
2. k lớn -> cụm mảnh.

### 13.3 DBSCAN
1. eps nhỏ -> nhiều noise.
2. eps lớn -> dễ gộp cụm.
3. min_samples lớn -> khó tạo cụm hơn.

### 13.4 Tree/Boosting models
1. n_estimators tăng -> tốt hơn nhưng chậm hơn.
2. max_depth tăng -> dễ overfit.
3. learning_rate giảm -> cần nhiều cây hơn.

### 13.5 Imbalance
1. class_weight/scale_pos_weight rất quan trọng.
2. Theo dõi F1, PR-AUC thay vì chỉ accuracy.

### 13.6 Semi-supervised
1. threshold quá cao -> ít pseudo-label.
2. threshold quá thấp -> nhiễu pseudo-label.

---

## 14) Glossary dễ nhớ

1. Support: tỷ lệ mẫu có cùng pattern.
2. Confidence: xác suất hậu quả khi điều kiện xảy ra.
3. Lift: mức mạnh của mối liên hệ so với ngẫu nhiên.
4. Silhouette: điểm chất lượng phân cụm (cao tốt).
5. PR-AUC: thước đo tốt cho dữ liệu mất cân bằng.
6. Recall: bắt được bao nhiêu lỗi thật.
7. Precision: cảnh báo lỗi có chính xác không.

---

## 15) Kết luận

Bạn có thể dùng tài liệu này theo 2 cách:
1. Ôn thi/thuyết trình: đọc mục 1 -> 3 -> 7 -> 10.
2. Đi debug code: đọc mục 11 rồi mở đúng module tương ứng.

Nếu cần mở rộng, bước hợp lý tiếp theo là:
1. GridSearch đúng nghĩa với hyperparams trong params.yaml.
2. Thêm calibration cho xác suất lỗi.
3. Thêm SHAP để giải thích prediction theo từng mẫu.

---

## 16) Phiên bản siêu dễ hiểu (dành cho học sinh trung học)

Mục này giải thích bằng ngôn ngữ đời thường. Bạn có thể dùng để dạy lại cho bạn bè chưa học Machine Learning.

### 16.1 Dữ liệu trong bài này là gì

Mỗi dòng dữ liệu là “một lần kiểm tra máy” với các thông số:
1. Nhiệt độ không khí (`Air temperature [K]`)
2. Nhiệt độ quá trình (`Process temperature [K]`)
3. Tốc độ quay (`Rotational speed [rpm]`)
4. Mô-men xoắn (`Torque [Nm]`)
5. Độ mòn dao (`Tool wear [min]`)
6. Kết quả máy có hỏng không (`Machine failure`: 0 hoặc 1)

Hiểu đơn giản:
- Đây giống như “phiếu sức khỏe” của máy tại từng thời điểm.

### 16.2 Luật kết hợp (Apriori) - như tìm thói quen

Ý tưởng:
- Tìm các mẫu lặp lại kiểu: “Nếu A và B cùng xảy ra thì C thường xảy ra”.

Ví dụ đời thường:
- Nếu trời mưa + tan học giờ cao điểm thì đường thường kẹt xe.

Ví dụ từ bộ dữ liệu AI4I:
- Nếu `Torque` cao và `Tool wear` cao thì `Machine failure` thường tăng.

Tại sao hữu ích:
1. Dễ giải thích cho người vận hành.
2. Dùng để tạo cảnh báo dạng luật.

### 16.3 Phân cụm (Clustering) - như chia nhóm học sinh

Ý tưởng:
- Không cần biết nhãn trước, chỉ dựa vào độ giống nhau để chia nhóm.

Ví dụ đời thường:
- Chia lớp thành nhóm học tương tự nhau dựa vào điểm Toán, Văn, Anh.

Ví dụ từ AI4I:
- Nhóm 1: tốc độ cao, mô-men vừa, mòn thấp.
- Nhóm 2: mô-men cao, mòn cao, nhiệt bất lợi.
- Có thể thấy nhóm 2 có tỷ lệ hỏng cao hơn.

Tại sao hữu ích:
1. Biết máy đang ở “chế độ vận hành” nào.
2. Tập trung theo dõi nhóm rủi ro.

### 16.4 Phân lớp (Classification) - như bác sĩ chẩn đoán

Ý tưởng:
- Dựa vào dữ liệu đầu vào để dự đoán thuộc lớp nào.

Ví dụ đời thường:
- Bác sĩ nhìn triệu chứng để kết luận: “cảm cúm” hay “không cảm cúm”.

Ví dụ từ AI4I:
- Đầu vào: nhiệt độ, tốc độ quay, mô-men, độ mòn.
- Đầu ra: `Machine failure = 1` (có hỏng) hoặc `0` (không hỏng).

Điểm quan trọng trong bài này:
- Máy hỏng rất ít (khoảng 3.39%), nên không thể chỉ nhìn Accuracy.
- Phải quan tâm Recall, F1, PR-AUC để không bỏ sót lỗi.

### 16.5 Bán giám sát (Semi-supervised) - như giáo viên chỉ chấm một phần bài

Ý tưởng:
- Có rất ít dữ liệu có nhãn, rất nhiều dữ liệu chưa nhãn.
- Mô hình học từ phần có nhãn, rồi tự đoán nhãn cho phần còn lại.

Ví dụ đời thường:
- Giáo viên chấm kỹ 10 bài, sau đó dùng mẫu đó để phân loại thêm 90 bài còn lại.

Ví dụ từ AI4I:
- Giữ nhãn ở mức 5%, 10%, 20%.
- So sánh 3 cách: chỉ học nhãn ít, Self-Training, Label Spreading.

Tại sao hữu ích:
1. Tiết kiệm công gán nhãn.
2. Gần với thực tế nhà máy hơn.

### 16.6 Hồi quy (Regression) - như dự đoán một con số

Ý tưởng:
- Dự đoán giá trị liên tục, không phải đúng/sai.

Ví dụ đời thường:
- Dự đoán nhiệt độ ngày mai là bao nhiêu độ C.

Ví dụ từ AI4I:
- Dự đoán `Tool wear [min]` là bao nhiêu phút.

Tại sao hữu ích:
- Biết “mòn bao nhiêu” để chủ động thay dao trước khi quá muộn.

### 16.7 Chuỗi thời gian (Time Series) - như nhìn biểu đồ theo ngày

Ý tưởng:
- Dùng dữ liệu quá khứ theo thứ tự thời gian để dự đoán tương lai.

Ví dụ đời thường:
- Nhìn lượng điện tiêu thụ các ngày trước để đoán ngày mai.

Ví dụ từ AI4I:
- Dữ liệu không có ngày giờ thật, nên dùng `UDI` như trục thời gian giả.
- Dùng ARIMA và mô hình có lag feature để dự báo xu hướng.

### 16.8 Một ví dụ “đọc máy” hoàn chỉnh

Giả sử một bản ghi có:
1. `Tool wear` rất cao
2. `Torque` cao
3. `Process temperature` cao hơn `Air temperature` không nhiều

Diễn giải trực giác:
1. Dao đã mòn nhiều.
2. Máy đang chịu tải lớn.
3. Tản nhiệt có thể không tốt.

Khi đó:
1. Luật kết hợp có thể coi đây là mẫu rủi ro.
2. Mô hình phân lớp có thể tăng xác suất dự đoán lỗi.
3. Mô hình hồi quy có thể cho thấy độ mòn đang ở mức cao.
4. Hệ thống có thể khuyến nghị bảo trì sớm.

### 16.9 Câu chốt dễ nhớ

1. Luật kết hợp: tìm “thói quen cùng xuất hiện”.
2. Phân cụm: chia “nhóm giống nhau”.
3. Phân lớp: dự đoán “thuộc loại nào”.
4. Bán giám sát: học khi “thiếu nhãn”.
5. Hồi quy: dự đoán “một con số”.
6. Chuỗi thời gian: dự đoán “bước tiếp theo theo thời gian”.
