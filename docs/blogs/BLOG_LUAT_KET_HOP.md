# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 1: Luật Kết Hợp (Association Rules)

## 1. Giới Thiệu: Khai Phá Quy Luật Tương Quan
Khai phá **Luật kết hợp (Association Rules Learning)** là một kỹ thuật phân tích dựa trên luật (Rule-based Machine Learning) để khám phá ra các quy luật tương quan ẩn sâu giữa các biến số trong một tập dữ liệu lớn. Kỹ thuật này nổi tiếng nhất với bài toán "Phân tích Giỏ hàng" (Market Basket Analysis) trong bán lẻ - "Khách mua bỉm thường mua thêm bia".
Tuy nhiên, trong bối cảnh công nghiệp và **Bảo trì dự đoán (Predictive Maintenance)**, các kỹ sư ứng dụng kỹ thuật này để tìm ra hiện tượng "Domino của sự cố hỏng hóc". Hiệu ứng lây lan này có thể là: "Nếu máy đang bị lỗi Quá tải (OSF), thì xác suất rất cao sẽ kéo theo lỗi Tản nhiệt (HDF) làm phá hủy hệ thống".

## 2. Giải Thích Thuật Toán Đặc Tiêu: Apriori & FP-Growth
Việc phải quét qua hàng vạn dòng lịch sử để tìm các tổ hợp trạng thái bệnh lý là một bài toán tốn cực kỳ nhiều chi phí tính toán (Computational complexity). Hai thuật toán tối ưu tiêu biểu được sử dụng:
- **Thuật toán Apriori:** Hoạt động dựa trên nguyên lý "Cắt tỉa nhánh tĩnh" (Pruning). *Diễn giải:* Giống như một anh thợ đi kiểm tra lỗi, nếu anh ta thấy việc "Mòn dao (TWF)" đứng một mình vốn dĩ đã hiếm khi xảy ra, anh ta sẽ lập tức loại bỏ việc xét xem "Mòn dao kết hợp với hỏng cầu chì thì sao" để tiết kiệm thời gian.
- **Thuật toán FP-Growth (Frequent Pattern Growth):** Tinh vi hơn, nó nén cơ sở dữ liệu lại thành một cây tần suất cấu trúc bộ nhớ (FP-Tree). *Diễn giải:* Các lỗi hay đi sát với nhau tạo chung một nhánh dẫn, thuật toán chỉ cần duyệt qua các nhánh cây gia phả này mà không cần càn quét lặp lại rà từng dòng dữ liệu vòng quanh (Database Scanning).

## 3. Các Chỉ Số Cốt Lõi (Metrics) Đánh Giá Độ Tin Cậy
Một bộ luật tự động sinh ra như: `{Nhiệt độ Process cao, Tốc độ chập chờn} -> {Lỗi Tản Nhiệt HDF}` luôn phải được thẩm định qua 3 lăng kính Thống Kê:

### 3.1. Support (Độ Phổ Biến / Độ Hỗ Trợ)
- **Thuật ngữ:** Tỷ lệ phần trăm các điểm dữ liệu (Transactions/Records) xuất hiện cả hai hiện tượng X và Y trên tổng toàn bộ dữ liệu. `Support(X, Y) = Freq(X, Y) / N`
- **Diễn giải:** Trong xưởng 10.000 mẻ máy chạy, có 100 lần diễn ra sự trạng thái máy "Vừa nhiệt cao, vừa nhảy tốc độ, vừa hỏng quạt". Support = `100 / 10000 = 1%`. Tham số `min_support` thường được cài cực kỳ thấp vì các tổ hợp gây lỗi bản chất vốn rất hiếm so với thời gian chạy bình thường của nhà máy.

### 3.2. Confidence (Độ Tin Cậy)
- **Thuật ngữ:** Thước đo xác suất có điều kiện. Trong số các trường hợp ĐÃ xảy ra hiện tượng X, có bao nhiêu khả năng kéo theo hiện tượng Y. `Confidence(X -> Y) = Freq(X, Y) / Freq(X)`.
- **Diễn giải:** Thước đo sự uy tín! "Confidence = 80%" có nghĩa là nếu thợ bảo trì ĐÃ biết chắc chiếc máy này {Vừa Nóng Vừa Giật}, thì tỷ lệ lên đến 80% là quạt tản nhiệt của nó sẽ nhận giấy tử tử.

### 3.3. Lift (Mức Độ Nâng Cấp/Hệ Số Tương Quan)
- **Thuật ngữ:** Tỉ lệ giữa xác suất xuất hiện đồng thời sự kiện X và Y so với xác suất xuất hiện nếu như chúng phân phối hoàn toàn độc lập với nhau. `Lift = Confidence(X -> Y) / Support(Y)`.
- **Diễn giải:** Đây là sức mạnh cốt lõi. Giả sử việc bộ phận "Quạt hỏng độc lập" (Support Y) vốn dĩ diễn ra nhan nhản không rõ nguyên nhân. Lift bóc trần sự thật định lượng: Nếu `Lift = 3`, tức một cỗ máy khi có biểu hiện {Vừa nóng vừa giật} sẽ làm bộ quạt Tăng Khả Năng Rách Nát Hủy Hoại **Lên Gấp 3 Lần** so với chạy bình thường. Nếu `Lift = 1`, hai diễn biến kia xuất hiện cạnh nhau chỉ do trùng hợp ngẫu nhiên.

## 4. Insights Chuyên Sâu Từ Dự Án Hệ AI4I
Cài đặt ép giảm độ võng tham số Support đã trục vớt được những quy luật ngầm chấn động:
- **Tuyến Lỗi Vỡ Thiết Bị (Overstrain Failure - OSF):** Chỉ số bóc tách phơi bày rằng, trạng thái cấu trúc `Tool wear (Mòn) > 200` KẾT HỢP với biến số `Torque (Lực vặn) cường độ > 50 Nm`, đẩy giá trị `Lift` hệ thống vọt ngưỡng > 10. Đây là bằng chứng định lượng đắt giá cho dấu hiệu thiết bị chịu lực ép quá tải dẫn đến gãy trục.
- **Tuyến Hủy Hoại Luồng Tĩnh Khí (HDF):** Sự chênh lệch nhiệt `Temp_diff` ngập vùng biên dưới cực hẹp < 8.6K KẾT TỤ đồng thời với Tốc Quay Lồng Rô-tơ rề chậm (< 1380 rpm) đã kéo mạn Confidence của một vụ đun cháy vỡ thiết bị lên mức trần xấp xỉ 100%. Kiến thức này lập tức trở thành ngưỡng cảnh báo (Threshold-trigger) cho hệ thống!
