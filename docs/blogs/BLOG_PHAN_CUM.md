# Blog: Phân Cụm (Clustering - KMeans, DBSCAN, HAC)

## 1. Giới Thiệu

Mục tiêu phân cụm trong đồ án là:
- Tìm các "chế độ vận hành" khác nhau từ dữ liệu cảm biến mà không cần nhãn.

Khác với phân lớp:
1. Phân lớp cần nhãn hỏng/không hỏng.
2. Phân cụm tự tìm nhóm tự nhiên trong dữ liệu.

Giá trị thực tiễn:
- Tạo bản đồ trạng thái vận hành.
- Nhận diện cụm có tỷ lệ hỏng cao.
- Hỗ trợ chiến lược vận hành theo chế độ máy.

## 2. Thiết Lập Thí Nghiệm

### 2.1 Dữ liệu và đặc trưng

- Dùng tập processed data.
- Chọn feature số và chuẩn hóa trước khi phân cụm.
- Loại cột ID và các cột failure leakage khỏi không gian phân cụm.

### 2.2 Thuật toán được thử

1. KMeans: cụm theo khoảng cách tới centroid.
2. DBSCAN: cụm theo mật độ, có nhãn noise.
3. HAC (Hierarchical): gom cụm phân cấp.

### 2.3 Cấu hình trong dự án

Theo `configs/params.yaml`:

KMeans:
- `n_clusters_range = [2,3,4,5,6,7,8]`

DBSCAN:
- `eps_range = [0.3,0.5,0.7,1.0]`
- `min_samples_range = [3,5,10]`

HAC:
- `n_clusters_range = [2,3,4,5]`
- `linkage = ward`

### 2.4 Chỉ số đánh giá

1. Silhouette Score: càng cao càng tốt.
2. Davies-Bouldin Index: càng thấp càng tốt.
3. Calinski-Harabasz Index: càng cao càng tốt.

Ngoài ra đồ án còn profile từng cụm theo failure rate để kiểm tra ý nghĩa vận hành.

## 3. Kết Quả Tổng Quan

### 3.1 Nhận xét theo thuật toán

1. KMeans:
- Ổn định, dễ diễn giải centroid.
- Nhạy với k và giả định cụm tương đối cầu.

2. DBSCAN:
- Mạnh trong phát hiện điểm nhiễu.
- Nhạy với `eps`, có thể sinh nhiều noise khi tham số chặt.

3. HAC:
- Tốt để quan sát cấu trúc phân cấp.
- Dễ hiểu khi so cụm ở các mức khác nhau.

### 3.2 Ý nghĩa nghiệp vụ

Thông qua profile cụm có thể thấy:
1. Cụm rủi ro cao thường đi với tải cao và mòn dao lớn.
2. Cụm ổn định có thông số nhiệt-tải cân bằng hơn.

Kết quả này giúp thiết kế chính sách vận hành theo cụm:
- Cụm rủi ro cao: kiểm tra dày hơn, giảm tải.
- Cụm ổn định: theo dõi định kỳ.

### 3.3 Giới hạn

1. Cụm không đồng nghĩa nguyên nhân gốc.
2. Chất lượng phụ thuộc mạnh vào tiền xử lý và scale.
3. DBSCAN/HAC có thể khó tối ưu nếu số chiều tăng cao.

## 4. Kết Luận và Khuyến Nghị

### 4.1 Kết luận

- Phân cụm giúp nhìn dữ liệu theo góc "trạng thái vận hành" thay vì chỉ nhãn lỗi.
- Đây là lớp phân tích bổ trợ rất tốt cho mô hình phân lớp.

### 4.2 Khuyến nghị

1. Dùng Silhouette làm tiêu chí chọn cấu hình ban đầu.
2. Luôn kiểm tra failure rate theo cụm trước khi kết luận nghiệp vụ.
3. Với DBSCAN, tune `eps` theo biểu đồ khoảng cách lân cận.
4. Kết hợp PCA/UMAP để trực quan cụm dễ hơn khi báo cáo.