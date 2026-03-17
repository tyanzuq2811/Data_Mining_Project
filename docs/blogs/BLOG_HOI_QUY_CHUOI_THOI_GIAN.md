# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 5: Hồi Quy & Chuỗi Thời Gian (Regression/TimeSeries)

## 1. Giới Thiệu
### Ý tưởng cực dễ hiểu (Đếm giờ quả bom nổ chậm)
Các kỹ thuật trước chúng ta làm một việc là Dán Nhãn Lệnh bài "CÓ" (Hỏng) hoặc "KHÔNG" (Khỏe) vào máy móc. Nhưng trong nghiệp vụ nhà xưởng, sếp giám đốc thèm khát con số thực tế hơn: "Dụng cụ chuẩn bị bị mài mòn đi CHÍNH XÁC LÀ MẤY PHÚT sau ngày hôm nay?", "Bao nhiêu giờ nữa thì mũi khoan phất cờ khởi nghĩa gãy làm đôi (RUL)?".
Để trả lời trọn vẹn bằng **MỘT CON SỐ CỤ THỂ**, chúng ta gọi sức mạnh của thuật toán **Hồi quy (Regression)**. Ngoài ra, thay vì đọc độc lập từng dòng dữ liệu hỗn độn, nếu ép bộ dữ liệu di chuyển theo "trình tự tíc tắc của thời gian tuần tự", và bắt AI nhìn về đoạn thẳng tương lai, chúng ta gọi nó là **Chuỗi Thời Gian (Time-Series)**.

## 2. Giải Thích Thuật Toán & Tham Số (Giải Phẫu AI4I)
- **Hồi quy (Linear Regression / Random Forest Regressor):** Cố gắng kẻ 1 vệt phấn (Đường thẳng / Đường hầm cong) băng xuyên qua toàn bộ bản đồ các dấu chấm dữ liệu sao cho nó nằm chính giữa nhất.
- **Chuỗi Thời Gian (ARIMA):** Khéo léo lấy dĩ vãng nuôi tương lai: Nhiệt độ của "Ngày Hôm Nay = (Ngày hôm qua x 0.8) + (Ngày hôm kìa x 0.2)". Trọng số trí nhớ ngắn hạn sẽ lớn hơn đoạn dài hạn.
- **Tham số `MSE` (Mean Squared Error) / `RMSE`:** Đây là cây gậy trừng phạt. Ví dụ độ mòn thật của máy là mòn mất "10 phút". AI đoán là mòn "8 phút". Suy ra nó sai 2 phút. Lấy 2 đem bình phương lên thành "Phạt 4 độ". Hàm tính ép AI giảm thiểu điểm Phạt này Kịch Sàn (Càng nhỏ càng cực chuẩn).
- **Tham số `R-squared (R2)`:** Độ tinh khiết của lời giải thích. Nếu phương trình báo R2 = 0.82, có nghĩa là "Sự thay đổi của lực vặn dao đã giải thích được một cách Logic tới 82% lý do vì sao cái mũi dao bị gãy".

## 3. Thiết Lập Thí Nghiệm Trong Dự Án (AI4I 2020)
- **Nạp Đạn Hồi Quy RUL (Remaining Useful Life):** Hướng súng bắn thẳng vào biến số `Tool Wear [min] (Độ đo mòn theo số phút)`. Loại bỏ các nhãn có hỏng hay không, thay vào đó ta đẩy vòng tua (RPM) và lực vặn (Torque) vào xem nó có quy ra được số phút mòn không.
- **Đường ray Time-Series:** Từ bản AI4I trộn lộn xộn, chúng ta tái tạo giả lập một đường nối tiếp 100 giờ chạy máy không ngừng nghỉ để quan sát biểu đồ sóng nhiệt.

## 4. Kết Quả Tổng Quan
Từ các file đo lặp `regression_results.csv` và hệ thống báo cáo `insights.txt`:
- **Phát Kiến Vĩ Đại Từ Công Suất:** Các biến số gốc không nói lên nhiều điều, nhưng khi kỹ sư ghép Toán Học: `Tính Toán (Torque × Rotational Speed)` sẽ ra Chỉ số Công suất sinh điện (Power). Mô hình Random Forest Regressor lập tức bắn tín hiệu phát giác mãnh liệt: Khi Power (công suất) rơi ra khỏi khoảng an toàn `[3500, 9000] W` —> Tool Wear (Độ Lão Hóa Của Dao) sẽ tăng dốc đứng cắm thẳng vào trạng thái Vỡ Mũi (OSF). Chỉ số giải thích R2 rất đáng tin. 
- **Theo Dõi Vệt Sóng Của Chuỗi Thời Gian:** Sóng nội suy chỉ ra rằng: Nhiệt độ không thay đổi kiểu "Nhảy Đột Ngột". Nó bò theo con dốc tuyến tính. Mô hình Time-Series tự động nối đường chân nhiệt bóp nghẹt biến số `Temp_diff` xuống mức tiệm cận của lòng chảo 8.6K. Hệ thống AI đoạt quyền dự báo điểm vỡ nhiệt (HDF).

## 5. Kết Luận và Khuyến Nghị
Đưa định lượng tuyệt đối vào thế giới vận hành.
- **Khuyến Nghị Lập Dashboard (Trạm Điều Khiển):** Các kĩ sư lập tức trích xuất phương trình Hồi quy lên màn hình LED của nhà máy để làm mồi "Đếm lùi RUL". Lúc biểu đồ chuỗi thời gian của "Temp_diff" chúc đầu xuống gần vạch tử thần 8.6K, hệ thống không chỉ rít còi mà nó còn nảy báo giá: *"Hãy thay mới đoạn dao cắt ngay lập tức, Quý Ngài chỉ còn đúng **20 phút tủy sống** trước khi ngưỡng 200 min sẽ chặt đứt đôi tất cả"*.
- **Lời Cáo Chung:** Khép lại cuộn biên niên sử của 5 Series Blog, từ Dự đoán Cụm trôi nổi đến Phương trình Tiên Tri Tương Lai. Phân Tích Dữ Liệu Predictive Maintenance đã chứng minh nó không phải là một bộ môn hàn lâm cho mọt sách nằm kho, mà là Hệ Hô Hấp Cứu Tinh Giữ Chặt Hàng Triệu Đô-la Cho Dây Chuyền Cơ Khí Thực Chiến! Cảm ơn Các Bạn Cơ Khí Tương Lai!
