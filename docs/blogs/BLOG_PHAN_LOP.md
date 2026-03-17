# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 3: Phân Lớp (Classification)

## 1. Giới Thiệu
### Ý tưởng cực dễ hiểu (Hội đồng Trọng tài bắt gian lận)
Phân lớp (Classification) có giám sát giống hệt như trường học phát cho bạn một chồng 10,000 bài làm và đưa kèm LUÔN ĐÁP ÁN (Nhãn - Máy hư = Lỗi, Máy sống = Không lỗi). Mô hình học sinh AI sẽ ráng "thuộc lòng" và rút kinh nghiệm từ bộ trắc nghiệm này để chấm thi những cái máy móc của ngày mai.
Nhưng tại trạm bơm AI4I, cạm bẫy lớn nhất là: **Lỗi quá HIẾM!**. Bạn mang 10,000 cái máy chạy mới có đúng 339 cái văng miểng (Nhãn hỏng là Số 1). Nếu học sinh AI siêu lười biếng phán bừa: "100% tất cả máy móc tôi nhận đều là Sống (Số 0)", AI đó nghiễm nhiên lãnh 96.6% điểm Accuracy (Độ chính xác) nhưng THỰC CHẤT là một cỗ máy thảm hoạ vì mù lòa khi đâm trực diện 339 vụ tai nạn chết chóc (False Negative - Bỏ lọt Hư hỏng). 

## 2. Giải Thích Thuật Toán & Tham Số (Kèm phân tích bộ AI4I)
Người đại diện chống mù này gọi là **Họ Boosting/Cây Quyết Định (Ví dụ: LightGBM, Gradient Boosting)**. 
- *Cách nó làm việc:* Thay vì giao cho 1 đứa học sinh giải toán, nó thuê 100 đứa học sinh yếu nhưng đứa học sinh thứ 2 sẽ CỐ TÌNH MỞ TO MẮT sửa lỗi sai của đứa học sinh thứ nhất. Đứa số 3 nhìn vào cái sai của đứa số 2 rồi vá dần.
- **Tham số `n_estimators`:** Trong bài toán này là lượng cây, kiểu "gọi bao nhiêu học sinh vào sửa bài chéo". Quá ít thì bắt hụt lỗi, gọi 500 cây thì máy nhà xưởng lag đơ.
- **Tham số `learning_rate`:** Mức độ rụt rè. Tốc độ sửa bài quá nhanh thì dễ đi lố (nhận diện nhầm), quá chậm (0.01) thì AI học siêu lâu.

## 3. Thiết Lập Thí Nghiệm Trong Dự Án (AI4I 2020)
- **Tuyệt chiêu SMOTE (Chống học vẹt):** Chỉ có vỏn vẹn 3% nhãn lỗi nên ta dùng Oversampling. Hiểu đơn giản là kỹ sư sẽ tạo ra 9,000 cái Nhãn Lỗi Gỉả từ 339 cái thật đó và nhồi vào giáo án, bắt AI đọc lại. "Đọc đến khi nào quen mặt dấu hiệu vỡ dao thì thôi!".
- **Từ Chối Accuracy, Dùng PR-AUC & F1-Score:** 
  - *Precision:* Trong 100 lần AI túm còi báo "Máy đang nổ", có bao nhiêu lần là nổ thật? (Tránh báo động giả gây tốn tiền nhân viên đi kiểm tra). 
  - *Recall:* Trong CẢ NHÀ MÁY có thật 339 cái vỡ, AI bắt được mấy cái? Ghi điểm PR-AUC là thước đo chung dung hòa được 2 kẻ thù này.

## 4. Kết Quả Tổng Quan
Cập nhật bảng phong thần từ file đào tạo `classification_results.csv`:
- **Gradient Boosting (GBM):** Đủ tốt (F1 = 0.841, PR-AUC = 0.846) nhưng ngâm mình rất lâu (Tốn tới ~24.9 giây để chấm bài).
- **Ông Hoàng LightGBM:** Nhờ chia khung Histogram xé nhỏ bảng dữ liệu mà LightGBM bọc lọt điểm **PR-AUC Lớn Nhất (0.8896)** (Rất ít khi báo động nhầm), mang về chức vô địch Recall (Tóm được 86.7% tất cả các lỗi có thật chôn dấu trong máy). Chưa kể điểm mạnh nhất là nó chỉ mất đúng **3.2 giây** vèo qua là học xong.
- **Random Forest:** Dù đứng kế bảng, nhưng do cây học quá độc lập nên chỉ đạt PR-AUC 0.79, không sửa được những dòng lỗi khó xơi như LightGBM.

## 5. Kết Luận và Khuyến Nghị
Đè đầu máy đọc số liệu và tránh xa bảng điểm Accuracy ảo ma.
- **Khuyến nghị Lắp Đặt:** Gắn thẳng bộ não `LightGBM` vào chuỗi truyền tải dây chuyền. Tiếp tục căn chỉnh ngưỡng Threshold xuống một tí để tăng tối đa điểm `Recall` (Thà phán báo động nhầm còn hơn bỏ lọt tên tội phạm).
- Ở bài sau, nếu người thu thập dữ liệu lại quá bận rộn để rải nhãn 0 với 1 này nữa thì ta làm gì? Mời đọc tiếp: Phép thuật Cầu Cảng của **Học Bán Giám Sát**.
