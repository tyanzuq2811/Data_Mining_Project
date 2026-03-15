# Sổ Tay Thuật Toán Và Cấu Hình Dự Án

Tài liệu này giúp bạn học lại nhanh toàn bộ thuật toán trong đồ án:
- Thuộc nhóm nào
- Dùng để giải bài toán gì
- Tham số quan trọng là gì
- Trong dự án này đã cấu hình như thế nào
- Cách đọc kết quả và ví dụ dễ hiểu

Nguồn đối chiếu chính:
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

## 1) Bài toán tổng thể đang giải

Mục tiêu của bài toán:
1. Dự đoán máy có hỏng hay không (classification, bài toán chính)
2. Tìm quy luật điều kiện dẫn đến hỏng (association rules)
3. Nhóm trạng thái vận hành thành cụm (clustering)
4. Phát hiện điểm bất thường (anomaly detection)
5. Dự đoán độ mòn dao cụ (regression)
6. Dự báo theo thứ tự vận hành (time series)
7. Học bán giám sát khi thiếu nhãn (semi-supervised)

Đặc thù dữ liệu AI4I:
- Tỷ lệ hỏng thấp (khoảng 3.39%) nên mất cân bằng lớp nặng
- Có 5 cột TWF/HDF/PWF/OSF/RNF dễ gây leakage nếu dùng sai
- Không có timestamp thật, dùng UDI làm chỉ số thời gian giả

---

## 2) Preprocessing: làm sạch và chuẩn hóa dữ liệu

### 2.1 Missing values
Định nghĩa:
- Giá trị thiếu trong bảng dữ liệu.

Trong bài này:
- Thường không có missing.
- Nếu có: cột số dùng median, cột phân loại dùng mode.

Thực thi:
- cleaner.py -> handle_missing()

Ví dụ dễ hiểu:
- Cột Torque có vài dòng bị rỗng, thay bằng trung vị của Torque để hạn chế ảnh hưởng outlier.

### 2.2 Duplicate handling
Định nghĩa:
- Dòng dữ liệu bị trùng nội dung.

Trong bài này:
- Kiểm tra duplicate bỏ qua cột định danh UDI, Product ID.
- Nếu trùng thì giữ dòng đầu.

Thực thi:
- cleaner.py -> handle_duplicates()

### 2.3 Outlier handling (IQR)
Định nghĩa:
- Giá trị quá lớn hoặc quá nhỏ so với phân bố chung.

Công thức IQR:
- Q1 = phân vị 25%
- Q3 = phân vị 75%
- IQR = Q3 - Q1
- Ngưỡng dưới = Q1 - k*IQR
- Ngưỡng trên = Q3 + k*IQR

Trong bài này:
- params.yaml: outlier_method = iqr, outlier_threshold = 1.5
- Xử lý bằng clip (cắt về ngưỡng), không xóa dòng.

Thực thi:
- cleaner.py -> handle_outliers()

### 2.4 Encoding categorical
Định nghĩa:
- Biến chữ (Type = L/M/H) cần đổi sang số để mô hình học.

Trong bài này:
- encoding = onehot
- Type -> Type_L, Type_M, Type_H

Thực thi:
- cleaner.py -> encode_categorical()

### 2.5 Feature scaling
Định nghĩa:
- Đưa các cột số về cùng thang đo để mô hình ổn định hơn.

Trong bài này:
- scaler = standard (StandardScaler)

Thực thi:
- cleaner.py -> scale_features()

---

## 3) Feature Engineering: tạo biến mới

### 3.1 Các feature được tạo
Trong bài này (builder.py):
1. temp_diff = Process temperature - Air temperature
2. power = Torque * Rotational speed * 2*pi/60
3. torque_speed_ratio = Torque / Rotational speed
4. wear_torque = Tool wear * Torque
5. Bin Tool wear theo các mức:
   - [0,50), [50,100), [100,150), [150,200), [200,300]
6. Lag features cho 5 biến sensor với window [1,3,5,10]
7. Rolling mean/std với window [5,10,20]
8. Interaction:
   - air_temp_x_speed
   - proc_temp_x_torque

Thực thi:
- builder.py -> build() và các hàm con

Ví dụ dễ hiểu:
- Hai máy cùng Torque = 50 Nm, nhưng máy A speed 1200, máy B speed 2500.
- power của B lớn hơn, nguy cơ tải hệ thống có thể khác.

---

## 4) Association Rules (Apriori)

### 4.1 Định nghĩa ngắn gọn
Apriori tìm các mẫu dạng:
- NẾU (điều kiện A, B, C) THÌ (Machine failure)

### 4.2 Các độ đo quan trọng
1. support:
- Tỷ lệ xuất hiện của tập mục trong toàn bộ dữ liệu.

2. confidence:
- Xác suất vế phải xảy ra khi vế trái xảy ra.

3. lift:
- Mức tăng liên quan so với ngẫu nhiên.
- lift > 1 là liên quan dương.

### 4.3 Cấu hình trong bài này
Từ params.yaml:
- min_support: 0.01
- min_confidence: 0.5
- min_lift: 1.5
- max_len: 4

Thực thi:
- association.py -> AssociationMiner

### 4.4 Ví dụ dễ hiểu
Ví dụ luật:
- IF (Torque_high, ToolWear_high) THEN (Machine failure)
- support=0.012, confidence=0.78, lift=23

Hiểu nhanh:
- 1.2% bản ghi có combo đó
- Trong các bản ghi có combo đó, 78% bị hỏng
- Khả năng hỏng cao gấp 23 lần mức trung bình

---

## 5) Clustering (Phân cụm)

### 5.1 KMeans
Định nghĩa:
- Chia điểm dữ liệu thành k cụm để giảm tổng khoảng cách tới tâm cụm.

Tham số quan trọng:
- n_clusters (k)
- n_init
- random_state

Cấu hình bài này:
- k range: [2..8]
- n_init = 10

Thực thi:
- clustering.py -> fit_kmeans()

### 5.2 DBSCAN
Định nghĩa:
- Tìm cụm dựa trên mật độ, có thể tìm noise/outlier.

Tham số quan trọng:
- eps: bán kính lân cận
- min_samples: số điểm tối thiểu tạo vùng dày đặc

Cấu hình bài này:
- eps_range: [0.3, 0.5, 0.7, 1.0]
- min_samples_range: [3, 5, 10]

Thực thi:
- clustering.py -> fit_dbscan()

### 5.3 Hierarchical Agglomerative Clustering (HAC)
Định nghĩa:
- Ban đầu mỗi điểm là 1 cụm, rồi gộp dần.

Tham số:
- n_clusters
- linkage (ward/complete/average)

Cấu hình bài này:
- n_clusters_range: [2, 3, 4, 5]
- linkage: ward

Thực thi:
- clustering.py -> fit_hierarchical()

### 5.4 Metric đánh giá clustering
1. Silhouette (cao hơn tốt hơn)
2. Davies-Bouldin (thấp hơn tốt hơn)
3. Calinski-Harabasz (cao hơn tốt hơn)

Thực thi:
- clustering.py -> _evaluate_clustering()

---

## 6) Anomaly Detection

### 6.1 Isolation Forest
Định nghĩa:
- Điểm bất thường dễ bị tách bởi cây ngẫu nhiên hơn.

Tham số trong bài này:
- contamination ~ 0.034 (gần tỷ lệ failure)
- n_estimators = 200

Thực thi:
- anomaly.py -> fit_isolation_forest()

### 6.2 LOF (Local Outlier Factor)
Định nghĩa:
- So sánh mật độ cục bộ của điểm với lân cận.

Tham số trong bài này:
- contamination ~ 0.034
- n_neighbors = 20

Thực thi:
- anomaly.py -> fit_lof()

### 6.3 One-Class SVM
Định nghĩa:
- Học biên của lớp "bình thường", điểm nằm ngoài là anomaly.

Tham số trong bài này:
- kernel = rbf
- nu ~ 0.034

Thực thi:
- anomaly.py -> fit_ocsvm()

### 6.4 Đánh giá anomaly
So sánh với nhãn thật, xem anomaly như failure:
- Precision, Recall, F1, Accuracy

Thực thi:
- anomaly.py -> compare_with_actual()

---

## 7) Classification (Dự đoán có hỏng hay không)

### 7.1 Vì sao dùng nhiều model
Mỗi model có ưu/nhược điểm khác nhau.
Dự án train và so sánh để chọn model tốt nhất theo F1.

### 7.2 Các model và cấu hình thực tế
Trong supervised.py (giá trị mặc định đang chạy):

1. Logistic Regression
- C = 1.0
- max_iter = 1000
- class_weight = balanced

2. Random Forest
- n_estimators = 200
- max_depth = 10
- min_samples_split = 5
- class_weight = balanced

3. Gradient Boosting
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05

4. XGBoost (nếu cài đặt)
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05
- scale_pos_weight = 28.5

5. LightGBM (nếu cài đặt)
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05
- is_unbalance = true

Lưu ý:
- params.yaml có danh sách hyperparameter dạng grid,
  nhưng module hiện tại đang dùng bộ tham số cố định như trên để train.

### 7.3 Metrics được báo cáo
- F1, ROC-AUC, PR-AUC, Precision, Recall, train_time

Thực thi:
- supervised.py -> train_classifiers()
- supervised.py -> cross_validate() với StratifiedKFold cv=5

### 7.4 Ví dụ dễ hiểu về imbalance
Nếu 1000 mẫu, chỉ có 34 mẫu hỏng:
- Model đoán tất cả là "không hỏng" vẫn có accuracy 96.6%
- Nhưng Recall/F1 cho lớp hỏng sẽ rất tệ
=> Vì vậy bài này ưu tiên F1 và PR-AUC hơn Accuracy.

---

## 8) Regression (Dự đoán Tool wear)

### 8.1 Model sử dụng
1. Linear Regression
2. RandomForestRegressor
3. GradientBoostingRegressor
4. XGBRegressor (nếu có)

Cấu hình thực tế (supervised.py):
- RF reg: n_estimators=200, max_depth=10
- GBR reg: n_estimators=200, max_depth=5, learning_rate=0.05
- XGB reg: n_estimators=200, max_depth=5, learning_rate=0.05

### 8.2 Metrics
- MAE
- RMSE
- R2

Thực thi:
- supervised.py -> train_regressors()

---

## 9) Time Series Forecasting

### 9.1 Nguyên tắc trong bài này
- Dataset không có timestamp thật
- Dùng UDI làm thứ tự thời gian giả
- Chia train/test theo thứ tự (không shuffle)

Thực thi:
- forecasting.py -> temporal_train_test_split()

### 9.2 ARIMA
Định nghĩa:
- Mô hình chuỗi thời gian với thành phần tự hồi quy + sai phân + trung bình trượt.

Cấu hình trong bài này:
- order = (2,1,2)
- Báo cáo thêm AIC, BIC

Thực thi:
- forecasting.py -> fit_arima()

### 9.3 Lag-feature Regression
Định nghĩa:
- Biến bài toán chuỗi thời gian thành supervised:
  dự đoán y_t từ x_t và các giá trị trễ.

Trong bài này:
- Dùng GradientBoostingRegressor với lag features đã tạo.

Thực thi:
- forecasting.py -> fit_lag_regression()

---

## 10) Semi-supervised Learning

Mục tiêu:
- Mô phỏng tình huống dữ liệu ít nhãn.

Trong bài này:
- Tỷ lệ nhãn thử: 5%, 10%, 20%
- Các phương pháp:
  1) supervised_only (baseline)
  2) self_training
  3) label_spreading

### 10.1 Self-Training
Định nghĩa:
- Model tự gán pseudo-label cho mẫu chưa nhãn,
  chỉ giữ dự đoán có độ tin cậy cao.

Cấu hình bài này:
- base estimator: RandomForestClassifier
- threshold = 0.95
- max_iter = 30

Thực thi:
- semi_supervised.py -> train_self_training()

### 10.2 Label Spreading
Định nghĩa:
- Truyền nhãn trên đồ thị độ tương đồng mẫu.

Cấu hình bài này:
- kernel = rbf
- alpha = 0.2
- max_iter = 100
- Nếu train quá lớn: subsample tối đa 5000 mẫu

Thực thi:
- semi_supervised.py -> train_label_spreading()

### 10.3 Baseline supervised_only
- Chỉ train trên phần có nhãn
- Model baseline: RandomForestClassifier

Thực thi:
- semi_supervised.py -> train_supervised_only()

---

## 11) Mapping tham số: params.yaml -> module nào dùng

1. seed
- Dùng trong supervised.py, clustering.py, anomaly.py, forecasting.py, semi_supervised.py

2. preprocessing.*
- Dùng trong cleaner.py

3. feature_engineering.*
- Dùng trong builder.py

4. mining.apriori.*
- Dùng trong association.py

5. mining.clustering.*
- Dùng trong clustering.py

6. modeling.classification.*, modeling.regression.*
- Có khai báo trong params.yaml, nhưng code hiện tại trong supervised.py đang khởi tạo bộ tham số cố định.

7. modeling.time_series.*
- forecasting.py dùng cho ARIMA order và train ratio theo tham số gọi.

8. semi_supervised.*
- Dùng trong semi_supervised.py

---

## 12) Cách sử dụng trong bài này (workflow để nhớ)

Thứ tự notebook:
1. 01_eda.ipynb
2. 02_preprocess_feature.ipynb
3. 03_mining_or_clustering.ipynb
4. 04_modeling.ipynb
5. 04b_semi_supervised.ipynb
6. 05_evaluation_report.ipynb

Hoặc chạy script:
- python scripts/run_pipeline.py
- python scripts/run_papermill.py

---

## 13) Mẹo nhớ nhanh tham số quan trọng

1. Apriori
- Giảm min_support => thêm nhiều luật hơn, có thể nhiễu hơn
- Tăng min_lift => luật chất lượng hơn nhưng ít hơn

2. KMeans
- k quá nhỏ: gộp nhiều hành vi khác nhau vào cùng cụm
- k quá lớn: cụm mảnh, khó diễn giải

3. DBSCAN
- eps nhỏ quá: nhiều noise
- eps lớn quá: dễ gộp thành 1 cụm lớn

4. Classification imbalanced
- Nên ưu tiên F1, PR-AUC
- class_weight/scale_pos_weight rất quan trọng

5. Semi-supervised
- threshold self-training quá cao => ít pseudo-label
- threshold quá thấp => pseudo-label nhiều nhưng dễ sai

---

## 14) Glossary ngắn gọn

1. Support
- Tỷ lệ xuất hiện của mẫu.

2. Confidence
- Xác suất vế phải đúng khi vế trái đúng.

3. Lift
- Mức liên quan vượt ngẫu nhiên.

4. Silhouette
- Độ tách/cụm chặt (cao tốt).

5. PR-AUC
- Hữu ích khi lớp dương ít (imbalance).

6. Recall
- Bắt được bao nhiêu mẫu hỏng thật.

7. Precision
- Trong các mẫu dự đoán hỏng, có bao nhiêu mẫu hỏng thật.

---

## 15) Kết luận

Nếu bạn quên kiến thức, hãy học lại theo thứ tự:
1. Mục 2 + 3 để nhớ preprocessing và feature engineering
2. Mục 4 + 5 + 6 để nhớ data mining
3. Mục 7 + 8 + 9 + 10 để nhớ machine learning
4. Mục 11 để biết tham số nào nằm ở đâu

Sau đó mở notebook 04 và 04b để xem kết quả thực tế của từng thuật toán.
