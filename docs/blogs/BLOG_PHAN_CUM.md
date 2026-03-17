# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 2: Phân Cụm (Clustering)

## 1. Giới Thiệu
### Ý tưởng cực dễ hiểu (Cô lập kẻ dị thường)
Phân cụm (Clustering) là trò chơi tìm ra những nhóm "cùng sở thích" trong một cộng đồng mà hoàn toàn KHÔNG CẦN BIẾT NHÃN (không cần ai mách bảo trước máy nào hư, máy nào khoẻ). Chúng ta cho thuật toán quét qua 10,000 dòng dữ liệu nhiệt độ, lực vặn. Thế mà, một cách vi diệu, những chiếc máy đang âm ỉ hỏng hóc lại có hành vi khác hẳn các máy khoẻ mạnh, và tự giác văng ra tạo thành một Cụm "Dị Thường" (Anomaly).

## 2. Giải Thích Thuật Toán & Các Tham Số (Kèm Ví Dụ AI4I)
Có hai thuật toán nổi tiếng bậc nhất để chơi trò này là K-Means và DBSCAN:
- **K-Means (Gộp vòng tròn tâm):**
  - K-Means cố ép các điểm dữ liệu vào các quả bóng tròn xung quanh tâm cụm. 
  - *Tham số `K` (Số cụm):* Trong dự án AI4I, nếu ta khai báo `K=3` (Kỳ vọng 1 cụm Khoẻ, 1 cụm Nóng, 1 cụm Hỏng), K-Means sẽ mù quáng cắt cái bánh sinh nhật dữ liệu thành đúng 3 phần, dù máy hư thực chất phân bổ móp méo như tổ ong.
- **DBSCAN (Kết nối đàn kiến):** DBSCAN tìm cụm dựa trên mật độ. Những ai đứng gần nhau (máy ở trạng thái khoẻ giống nhau y hệt) sẽ nối thành cụm khủng lồ. Ai bay ra ngoài ranh giới đó (nhiệt độ vọt lên, mòn quá đáng) bị cho là Noise (Nhiễu).
  - *Tham số `eps` (Bán kính tìm kiếm):* Giới hạn khoảng cách. Nếu `eps = 0.3`, máy A có lực quay 40Nm, máy B 42Nm thì gọi chung là 1 cụm. Máy C vọt lên 60Nm thì cho cút!
  - *Tham số `min_samples` (Kích thước tổ đội):* Số lượng máy tối thiểu để lập 1 cụm chính danh.

## 3. Thiết Lập Thí Nghiệm Trong Dự Án (AI4I 2020)
- **Tiền xử lý:** Bắt buộc phải phóng to/thu nhỏ (MinMaxScaler) tất cả các cột như vòng tua (tới 3000 rpm) và lực vặn (chỉ tới 70 Nm) về chung kích cỡ `[0,1]` để tránh cột RPM "đè bẹp" tiếng nói của cột Torque.
- **Kết quả đo bằng Silhouette (Độ ngoan của cụm):** Silhouette (chạy từ -1 đến 1) càng gần 1 chứng tỏ cụm càng đặc và rời rạc nhau. 
  - *K-Means K=3:* Thu về Silhouette cực thấp **~0.262**. Quá tệ! Vì lỗi Hỏng (Failure) làm gì có hình tròn, nó rải rác tứ tung.
  - *DBSCAN (Eps = 0.3, Min_Samples = 10):* Thu về Silhouette lên tới **0.473**. Eps 0.3 siêu nhỏ đã ép toàn bộ máy khoẻ mạnh vào một "quả cầu đặc" duy nhất và sút thẳng toàn bộ các dữ liệu mấp mé hỏng hóc ra rìa vực thẳm làm "Nhiễu (Noise)".

## 4. Kết Quả Tổng Quan 
Trích xuất từ file đầu ra (`cluster_failure_profiles.csv`), nếu ta mang nhóm "Nhiễu" bị DBSCAN vứt bỏ đi nội soi xem có gì trong đó:
- **Tập trung rủi ro:** Nhóm "Nhiễu" này ôm hầu hết khối lượng lỗi hỏng hóc thực sự của Toàn bộ Nhà máy! Dù ban đầu DBSCAN chưa hề được học xem "Chữ Failure là gì".
- **Chân dung Dị thường:** Các điểm nằm trong cụm dị thường nổi bật luôn luôn có độ Mài Mòn (Tool Wear) vọt qua ngưỡng giới hạn và tỷ lệ Lực vặn (Torque) chót vót. 

## 5. Kết Luận và Khuyến Nghị
Đừng bao giờ cố dùng K-Means để truy vết hư hỏng công nghiệp.
- **Khuyến Nghị Hiện Trường:** Tích hợp DBSCAN chạy ngầm. Bất kỳ khi nào thông số thời gian thực của một cái máy bị thuật toán "đá" văng ra khỏi Cụm-Máy-Khoẻ, tự động biến thành luồng khí Nhiễu (Noise), kỹ sư phải xếp lịch đại tu nó trong vòng **15 ngày** (giảm 1 nửa so với chu kỳ 30 ngày an toàn).
- Cùng đón đợi ở phần sau, khi chúng ta cấp đáp án trắc nghiệm chỉ điểm rõ ai hư ai hỏng vào bảng dữ liệu trong **Phân Lớp (Classification)**.
