# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 1: Luật Kết Hợp

## 1. Giới Thiệu
### Ý tưởng cực dễ hiểu (Liên kết các hiện tượng)
Khai phá luật kết hợp (Association Rules) giống như việc chủ siêu thị phân tích thói quen mua sắm: "Hễ ai mua sách vở và bút bi thì 80% sẽ mua thêm tẩy". Trong hệ thống máy móc (Predictive Maintenance), chúng ta tìm những "kết hợp" chết người: "Nếu **Nhiệt độ máy cao** VÀ **Tốc độ quay thấp**, thì liệu máy có bị **hỏng** không?". 

## 2. Giải Thích Thuật Toán & Các Tham Số (Kèm Ví Dụ Trực Tiếp Từ Bộ AI4I)
Thuật toán làm việc này tên là **Apriori** hoặc **FP-Growth**. Chúng quét 10,000 dòng lịch sử máy móc trong bộ AI4I để tìm ra các "cặp sự cố" luôn đi chung với nhau. Để lọc bỏ sự trùng hợp ngẫu nhiên, ta dùng 3 lăng kính kiểm duyệt:

- **Support (Độ phổ biến):** Tỷ lệ một sự kiện xuất hiện trên tổng số 10,000 lần máy chạy. 
  *Ví dụ từ AI4I:* Trong xưởng có hiện tượng "Máy bị quá tải" (OSF) xuất hiện. Nhưng chỉ có mòn dao (Tool Wear) > 200 phút và Lực vặn (Torque) > 50 Nm xuất hiện đồng thời trong 98 dòng (98/10000). Vậy Support của quy luật này là `~0.98%`.
- **Confidence (Độ tự tin/Độ chắc chắn):** Tính tỷ lệ phần trăm chắc chắn điều kiện B sẽ theo sau điều kiện A.
  *Ví dụ từ AI4I:* Nếu BIẾT CHẮC CỤ THỂ một cái máy đang có mũi khoan mòn > 200 phút và Torque > 50 Nm, hỏi xác suất cái máy đó văng miểng do OSF là bao nhiêu? Trong dữ liệu trả về, xác suất này (Confidence) lên tới `85%`.
- **Lift (Mức độ nhân quả/ Cường độ hút):** Cho biết quy luật này là do nguyên nhân - kết quả thật sự, hay là ngẫu nhiên vớ vẩn.
  *Ví dụ từ AI4I:* Có phải Việc mũi khoan bị mòn thực sự GÂY RA máy hỏng nặng, hay vì trong xưởng ngày hôm nay có tỷ lệ xui xẻo máy hỏng quá nhiều? Thuật toán tính ra vạch Lift của luật này > 50. (Bất kỳ chỉ số Lift > 1 nào nghĩa là chúng cộng hưởng sinh mệnh với nhau).

## 3. Thiết Lập Thí Nghiệm & Thông Số Trong Dự Án (AI4I 2020)
Dự án của chúng ta gặp một thách thức thực tế lớn: Lỗi máy móc rất hiếm khi xảy ra. Tổng số lỗi chỉ chiếm **3.39%** (339 dòng trên 10,000 dòng). Nếu lỗi quá ít, hệ thống rất dễ bỏ qua.
- **Chiến thuật (Tiền xử lý):** Phân nhóm (Binning) để gom tốc độ số cực lẻ thành "Thấp, Trung bình, Cao".
- **Tham số `min_support` = 0.01 (1%):** Cực kỳ quan trọng! Ta bắt buộc ép tỷ lệ phổ biến cực thấp xuống độ ngưỡng. Lý do: nếu ta đợi một hiện tượng lỗi đạt 5% sự cố mới thèm chú ý, thì vĩnh viễn AI sẽ câm như hến vì "Toàn bộ máy hỏng của nguyên một nhà máy gom lại mới được 3.39% chứ đâu ra 5%!". Do đó phải nhích cái ngưỡng này bé xíu xuống `0.01` (1%).
- **Tham số `min_confidence` = 0.5 (50%):** Nghĩa là khi một tổ hợp hoàn cảnh (ví dụ: Nhiệt độ Process cao) xuất hiện, ta yêu cầu phải báo cáo cho kỹ sư là máy MỚI chắc chắn > 50% sắp nổ.
- **Tham số `min_lift` > 1:** Ép mọi quy luật phải có sức nặng thực sự.

## 4. Kết Quả Tổng Quan
Nhờ ép các tham số siêu nhỏ trên, thuật toán Apriori đã lôi cổ 2 hiện tượng nguy hiểm nhất từ bảng số liệu:
- **Nguy cơ OSF (Overstrain Failure) cực độ:** Khi `Tool wear > 200` phút CỘNG VỚI lực vặn dao `Torque` ở mức Cao (> 50 Nm), bộ luật tính toán ngay lập tức đẩy *Lift* lên trên 10. Đây là dấu hiệu chắc nịch của sự nhồi nhét quá tải vỡ thiết bị.
- **Nguy cơ HDF (Heat Dissipation Failure) nổ máy:** Tưởng là chênh lệch nhiệt thấp là máy mát, nhưng khi nhìn vào bảng `insights`, `Temp_diff` < 8.6K KẾT HỢP VỚI `Tốc độ quay` rất chậm (< 1380 rpm) lại chỉ ra luồng không khí chẳng thèm lấy đi hơi nóng chết chóc. Độ tự tin (Confidence) rằng máy chuẩn bị bốc hơi lên tới gần 100%.

## 5. Kết Luận và Khuyến Nghị
Công nhân bảo trì không cần phải nhìn chằm chằm kiểm tra từng linh kiện cơ học một cách cô đơn nữa.
- **Khuyến nghị hành động:** Hệ thống giám sát (Monitoring) cần lập ngưỡng kép (Double-Threshold Trigger) dựa trên điều kiện KẾT HỢP. Ví dụ, nếu Tool wear chạm 180 phút nhưng Torque đang > 50 Nm, cần bấm còi dừng dây chuyền ngay lập tức.
- **Tiếp theo:** Liệu các cỗ máy bệnh tật với trạng thái lỗi này có xu hướng kết tụ lại một cách quái đản không? Đón xem phần 2: **Phân Cụm (Clustering)** để cô lập chúng khỏi các cỗ máy khoẻ mạnh.
