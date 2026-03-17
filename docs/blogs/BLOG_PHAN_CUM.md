# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 2: Phân Cụm (Clustering)

## 1. Giới Thiệu: Bài Toán Học Phi Giám Sát (Unsupervised Learning)
Trong khai phá dữ liệu, **Phân cụm (Clustering)** là cơ chế hạt nhân thuộc nhóm **Học Phi Giám Sát (Unsupervised Learning)**. Trái ngược hoàn toàn với Phân lớp, thuật toán khởi đầu trên một mỏ dữ liệu mù (Unlabeled Data) - nghĩa là hoàn toàn không được tiêm trước thông tin rằng "Hàng thu thập số 314 là máy đang Bình thường hay Hỏng hóc".
Nhiệm vụ của các bộ xử lý phân cụm là tự động khám phá ra các **Biểu kiến cấu trúc tàng hình (Hidden Patterns)**, cấu trúc dồn nhóm các đối tượng sở hữu thước đo chuẩn vector hành vi tương đồng mật thiết vào các Cụm. Đối với vận hành công nghiệp, kỹ thuật này giúp "tự động hóa cô lập các cá thể mang hành vi dị thường" mà không đòi hỏi chuyên gia cắm mắt dán nhãn thủ công (Domain Labeling) ngay bước ban đầu.

## 2. Giải Thích Điển Hình Hai Lõi Thuật Toán Phân Cụm

### 2.1. K-Means Clustering (Phân Cụm Định Hướng Trọng Tâm)
- **Thuật ngữ chuyên môn:** Đây là một thuật toán dồn cụm dựa trên Tâm (Centroid-based Algorithm). K-Means tối ưu hóa việc phân nhánh dữ liệu vào **K** cụm cố định bằng cách lặp lại việc giảm thiểu Tổng bình phương khoảng cách (WCSS / Inertia) đo từ điểm dữ liệu đơn lẻ rút về Tâm Cụm của chúng. 
- **Diễn giải trực quan:** Tưởng tượng một quản đốc xưởng chia xưởng ra `K = 3` nhóm tổ (Tâm nhóm). Máy móc (Điểm dữ liệu) trên băng đo đạc sự giống nhau về độ chênh nhiệt hay lực rung, và tự động dồn về đăng ký gia nhập ở cái "Lều trung tâm" có tính chất đồng điệu gần nó nhất. Tâm tổ thợ sau đó "chuyển dời" xê dịch đến điểm trung bình bù trừ cân bằng của số hội viên đó. Vòng lặp (`max_iter`) lặp lại liên tục cho đến khi chẳng ai thèm dời lều nữa.

### 2.2. DBSCAN (Phân Cụm Dựa Trên Mật Độ Cấu Cấu Mạng Không Gian)
- **Thuật ngữ chuyên môn:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) mở rộng cụm từ các Điểm Lõi (Core points). Sự ưu việt tuyệt đối so với K-Means là DBSCAN định hình ranh giới cụm dựa trên dải phân phối tỷ trọng không gian khổng lồ (kể cả biểu kiến hình dạng cụm có lõm khuyết, méo mó). Đồng thời, máy tự động trục xuất triệt để các dữ liệu cô lập vô quy tắc thành Nhãn Nhiễu (Outliers / Noise).
- **Diễn giải trực quan:** Quẳng K-Means đi, DBSCAN giống hệt việc phái kỹ sư soi đèn pin đo mật độ đám đông. Nó không cần khai báo máy sẽ tạo thành mấy tổ (`K`). Với vòng không gian soi vòng rọi của chóa đèn (`eps` - bán kính Euclidean), nếu xoay một phát mà ôm trúng vào vòng nén đủ lượng mặt thiết bị vây quanh tề tựu (`min_samples` - ngưỡng biên độ thành viên), kỹ sư lập ngay chúng thành 1 "Băng Nhóm Ổn Định". Cá thể nào văng hoang lạc lõng quanh viền, cách rời "Băng" sẽ bị định danh gông cổ làm "Nhiễu".

## 3. Yếu Tổ Tinh Chỉnh Tham Số Đầu Vào Và Kết Quả Đo Phân Cụm
- **Xử Lý Co Hẹp Tính Năng (Feature Scaling):** Cực kỳ sinh tử trong chuẩn khoảng cách không gian. Dải biến thiên Vòng tua máy RPM (Vọt kịch biên tới 3000 vòng/phút) sẽ thao túng tuyệt đối khoảng đo của đại lượng Lực Quay Nhỏ Bé (dao động bèo bọt \~70 Nm). Chặn đứng việc đè ép này, MinMax Scaler kích quy đồng trượt về khuôn tỷ lệ `[0,1]`.
- **Thước Đo Đánh Giá Cụm Silhouette Score:** Là bộ chỉ số nội tại định lượng tính cự ly mật thiết kết tụ của thành viên đối khu nội cụm với đo kiểm khoảng độ giãn xa so với ranh mép biên cụm khác. Tỷ số lướt từ `[-1, 1]`. Tiệm cận biên độ `1` tuyên thị độ "lý tưởng càn quét".
  - **Sân trượt của K-Means K=3:** Mô hình lao dốc thảm thiết đánh vớt mốc Silhouette nhạt nhòa **0.262**. Bản chất lõi lỗi trục thiết bị công nghiệp bung tỏa phân tán khắp nơi rải rác, không gò bo thành cầu tròn viên mãn như thuật toán kỳ vọng.
  - **Sức Quét Cô Lập Từ DBSCAN:** Sử dụng bộ dồn nghẹt `(Eps = 0.3, Min_samples = 10)`, DBSCAN vãn ngược Silhoutte bật phóng lên **0.473**. Tại giới hạn hẹp 0.3, DBSCAN đóng rắn toàn bộ tập máy vận hành Bình thường thành Một Bệ Băng Nhóm Khổng Lồ Duy Nhất. Dồn bóp tát phăng văng toàn bộ hệ thống máy chực chờ hỏng hóc lủng lẳng ra mép vực đài gắn thẻ **Nhiễu (Noise)**.

## 4. Kết Luận Và Định Hướng Ứng Dụng Hiện Trường
Toàn bộ dải dữ liệu bị DBSCAN chối bỏ (Nhiễu) chính là những cỗ xe hàm súc 100% rủi ro hỏng hóc của xưởng đo AI4I!
**Kiến trúc Cập Lập Dashboard Đo Cảnh Báo:** Tích ngầm khối kiến trúc DBSCAN đằng sau Cổng Dữ liệu trực tuyến Data Stream (Pipeline trực tiếp). Bất kỳ máy móc phần cứng nào bung biểu đồ đo lệch chuẩn thông số, không xâu chuỗi dung hợp được với Đám Đông Cụm Cốt Lõi, lập tức mồi văng gá định danh "Outliers", bật kích hoạt đội quản lý xếp lịch thay cuộn linh kiện cắt phay bảo hành ép định thời chu kỳ sớm!
