# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 3: Phân Lớp (Classification)

## 1. Giới Thiệu: Bài Toán Phân Lớp Mất Cân Bằng (Imbalanced Classification)
Trong Học máy (Machine Learning), **Phân lớp (Classification)** thuộc nhóm **Học có giám sát (Supervised Learning)**. Thuật toán được huấn luyện trên một tập hợp dữ liệu mẫu dĩ vãng đã được dán nhãn (Labeled Data) rõ ràng – ví dụ: 0 là "Thiết bị hoạt động Bình thường", 1 là "Thiết bị Gặp sự cố". Mục tiêu là hệ thống học được một ranh giới quyết định (Decision Boundary) hoặc đồ thị thuộc tính (Feature mapping) để phán bệnh cho những cỗ máy chạy ở ngày mai.
Tuy nhiên, trạm bơm AI4I đối diện với bức tường thách thức kỹ thuật lớn nhất: **Dữ liệu vô cùng mất cân bằng (Sự cố rất hiếm - Extreme Label Imbalance)**. Bạn cày quét hồ sơ 10.000 ca chạy máy nhưng mới moi bắt được vạch đúng 339 lệnh báo hỏng hóc thực (Tỷ lệ lõi 3.39%). Nếu học sinh AI lười biếng bỏ thi bằng cách học vẹt nhắm mắt phán "100% máy tôi duyệt không hề hỏng", điểm cúp mạ Accuracy (Độ chính xác) dễ dàng vút đỉnh 96.6%. Trong khuôn thiết lập vận hành nhà máy, điều đó đồng nghĩa sự cố tai nạn được dung túng bỏ lọt chết người (False Negative), gây lụn sụp cả cơ nghiệp sản xuất.

## 2. Giải Thích Kiến Trúc Thuật Toán Tập Hợp (Ensemble Methods)
Để triệt tiêu lỗi rập khuôn đơn tuyến, giới nghiên cứu Data Science huy động năng lực khổng lồ từ các nhóm Kiến trúc thuật toán **Học Tập Hợp (Ensemble Learning)**, phân tạc biểu tượng từ mô phỏng rừng và cây quyết định (Decision Trees):

### 2.1. Random Forest (Rừng Quyết Định Ngẫu Nhiên)
- **Nền tảng học thuật:** Thuật toán hoạt động dựa trên triết lý chập **Bagging (Bootstrap Aggregating)**. Một ma trận đồ sộ các cây phân nhánh (Decision Trees) độc lập được gieo cấy. Tại đích đến cuối cùng, một cơ chế biểu quyết đa số (Majority Voting) được trưng cầu để vọt chốt nhãn dự đoán sau cùng, giúp hệ thống cân khử nhiễu sai số (Variance Reduction).
- **Diễn giải trực quan:** Quẳng bỏ việc chỉ phó thác quyết định sống còn vận hành cho duy nhất một tay thợ máy, ta trưng cầu một "Cuộc họp hội đồng bao phủ cả trăm tay thợ". Mỗi cá nhân kiểm điện ngẫu nhiên các mảnh thuộc đo (Người dí mắt chĩa riêng kim vòng nhào Tourque, người kiểm nhiệt độ Temp). Cuối cùng hội đồng lấy trát ấn bầu cử phán quyết. Đặc quyền này giúp cả khối rừng khó bị chệch bánh trượt ray.
- **Tham số lõi vận mạch:**
  - `n_estimators` (Lượng quy điểm cây): Rừng nhét bao nhiêu thợ. Tỉ lệ tăng đảm bảo bệ phòng thủ biểu quyết khép kín vững chắc hơn nhưng ăn tươi nuốt sống RAM xử lý phần cứng.
  - `max_depth` (Soi xét thâm sâu): Lệnh cấp quyền thợ truy gốc rễ chuỗi logic dài hạn tới đâu. Kéo quá sâu sẽ dẫn tới thói tựu hỏng **Học vẹt (Overfitting)** – chỉ biết xử máy theo máy mẫu ngày dĩ vãng, nhưng trầy vạch với máy lạ.

### 2.2. Gradient Boosting / LightGBM (Cấu Trúc Cây Học Suy Suyển)
- **Nền tảng học thuật:** Xây đắp trên đế vương **Boosting**, mô hình cấu trúc các cây học tuần tự tiếp nối (Sequential). Cây đời sau ra đời chỉ nhằm thực thi một sứ mệnh: tối thiểu giảm bớt sai số (Minimize Loss Function / Residuals) do gốc cây đằng trước đùn đẩy để lại. Bản thân đỉnh **LightGBM** nâng tầm tối cường hóa bằng cơ chế phân lá (Leaf-wise growth) và kết hợp đóng rập điểm biểu lượng nhóm (Histogram-based), giúp nó bứt vọt tốc độ xử lý siêu hạng.
- **Diễn giải trực quan:** Dựng xếp các thợ máy dàn hàng dọc thành một đường ray băng thông sửa khuyết (Error-correcting line). Thợ tiền tuyến chẩn ban đầu tiên đoán – sẽ đà xuất hiện điểm nứt báo sai. Thợ thứ 2 không nhìn đại trà nữa mà vắt óc cày cuốc trừng trị đè đúng lên tệp tài liệu dính bắt chệch nhịp đó. Kiến trúc cứ nối đuôi gọt dũa bù lấp vòng khuyết. LightGBM y hệt việc lũ thợ chia kho điểm rác vào nhiều thùng rỗng đếm khối (Histogram) để phím lệnh chạy vèo thần sầu mà không tốn công mòn ngón rà soát từng dòng đơn điệu.
- **Tham số lõi vận mạch:**
  - `learning_rate` (Tỷ lệ học tập bám hụt): Mức độ "nhả trớn" cẩn trọng. Kéo dải tốc học quá gắt (cỡ 0.5) – mô hình lao đao phạt mạnh tay uốn sai số cũ nhưng chuốc họa học lố giới hạn Overfitting. Đệm ép trớn nhả nhỏ (0.01) – AI đi những bước chững chạc từ tốn, năng lực tổng quát hóa (Generalization) trùm biên vượt ngục cực xịn, nhưng tốn tiền đợi đốt điện nhang.

## 3. Quản Trị Đánh Giá & Masking Thí Nghiệm Rèn Quân
- **Loại bỏ Mất Cân Bằng Bằng Nội Suy Nghịch Đảo (SMOTE - Synthetic Minority Over-sampling Technique):** 
  - Kỹ thuật tự sinh tạo ảo nội suy. AI thuật toán tự vác cơ chế lấp đầy: giả lập nhào nặn bù chèn hơn 9.000 bản sao đồ thị lỗi mẻ dao (từ cục nhân bé tẹo 3% ỏi hẹp) thành một dải băng hỏng hóc lèn xẹp ngang ngửa tệp số máy khỏe mơn mởn. Điều này ép buột AI kinh qua môi trường ma trận "trải nghiệm tơi bời" trước hố họa hỏng hóc, bứng đi thói mù lọt.
- **Triệt Tiêu Accuracy Ảo - Thay Bằng Hệ Chỉ Hệ Số Tuyệt Vời PR-AUC & F1-Score:** 
  - *Precision (Chuẩn Dự Báo Chính Xác):* Phím độ lọc sàng lọc uy tín. Trong 100 lần AI kích nổ loa trần "Bơm hỏng", mấy lần nổ máy hiện thân là thật? Tránh hệ lụy báo cáo hoang loạn (False Positives) làm đứt chuỗi cung ứng.
  - *Recall (Độ Thu Hồi Nhạy Cảm):* Tỷ xuất phủ quét dọn sạch sành sanh bến đỗ thảm họa. Trong toàn tuyến thực hư có 339 mũi vòng kẹt vỡ trục, AI lôi cổ móc trọn được mấy ca gãy nát? Chặn đứng hiểm nguy lọt hố trần đen mịt (False Negatives). Thuật PR-AUC thọc kéo dung hòa chẻ cân đối trọn gói vòng lặp đôi này lập quỹ đánh giá quyền uy!

## 4. Trích Xuất File Bảng Danh Vọng (Model Performance)
- **Gradient Boosting (GBM):** Cho sức bọc bảo hộ đáng tín nhiệm nền vững (Điểm kéo dải F1 = 0.841, PR-AUC = 0.846). Khuyết điểm duy nhất vũng lầy lề mề xử lý nhúng hụp ngâm thân tới hơn ~24.9 giây để hoàn tất duyệt nhãn bãi luyện xưởng.
- **Tột Đỉnh Chói Phá Từ LightGBM:** Phát quang quyền năng mẻ lưới cày nhanh từ Histogram-based, đại gia LightGBM châm lửa quạt trọn cực đỉnh hệ cân đối thu đạt mốc hiếm có **PR-AUC 0.8896**. Ôm phất cờ chiến thắng chóa xua Recall (Bám chặt chẽ túm 86.7% điểm xước nứt). Lẽ dĩ nhiên, vòng nạp đào tạo siêu tốc (Training time) giật điện ẵm đĩnh chênh phẩy kinh hồn... nhỉnh gọn **3.2 giây** vèo qua là lách băng qua biên cờ đỏ.

## 5. Kết Luận Và Định Vị Sản Phẩm Thực Địa Truyền Tải
- Mô hình cấu trúc Cây Học Nhóm (Ensemble Learning) như LightGBM nên được cắm nẹp định gắn (Deployment) làm "Bộ Não Gác Sân" gác băng tải chuyền hệ thống vĩnh cửu. 
- Ưu tiên tùy chình căn biên vạch điều hướng xả ngưỡng lệnh chặn xác suất (Threshold Probability). Sấp ngửa cho phép hạ trượt hẹp ngưỡng báo xả cảnh báo xuôi xuống – nôm na là đánh thét lố "Thà phát loa sai lầm cho thợ xách cờ lê kiểm vòng, còn ngàn tỉ lần nhịn nuốt lọt thảm cảnh vỡ văng hỏng buồng lò nung kim loại!".
