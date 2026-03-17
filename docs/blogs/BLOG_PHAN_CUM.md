# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 2: Phân Cụm (Clustering)

## 1. Mục tiêu của tab này
Tab Phân cụm dùng để chia máy thành các nhóm hành vi tương đồng khi chưa có nhãn lỗi cho từng mẫu. Mục tiêu chính:
- Tìm nhóm máy có rủi ro cao để ưu tiên bảo trì.
- Phát hiện cụm bất thường/outlier để kiểm tra sớm.

## 2. Cách đọc chỉ số
- Silhouette: càng cao càng tách cụm tốt.
- Davies-Bouldin: càng thấp càng tốt.
- Calinski-Harabasz: càng cao càng tốt (so sánh trong cùng dữ liệu).

## 3. Kết quả định lượng chính
Từ bảng so sánh hiện tại:

1. Mô hình nổi bật về chất lượng cụm là `dbscan_eps0.3_ms5`
  - Silhouette = 0.806
  - Davies-Bouldin = 0.245
  - n_clusters = 3
  - n_noise = 9985

2. Các cấu hình KMeans có silhouette thấp hơn rõ rệt
  - `kmeans_k2`: silhouette = 0.282
  - `kmeans_k3`: silhouette = 0.197

3. Hồ sơ rủi ro theo cụm:
  - Cụm 0: 9863 mẫu, failure_rate = 2.69%
  - Cụm 1: 6 mẫu, failure_rate = 33.33%

Nhận xét nhanh:
- Cụm 1 là cụm nhỏ nhưng rủi ro rất cao, cần đưa vào diện theo dõi trọng điểm.

## 4. Ý nghĩa vận hành
- Phân cụm không thay thế phân lớp, nhưng giúp trả lời câu hỏi "nhóm máy nào cần ưu tiên trước".
- Cụm có failure_rate cao phù hợp để lập danh sách kiểm tra chuyên sâu.
- Outlier từ DBSCAN phù hợp làm tín hiệu cảnh báo kỹ thuật ban đầu.

## 5. Khuyến nghị hành động cụ thể
1. Đưa cụm có failure_rate > 10% vào danh sách "critical watchlist".
2. Rút chu kỳ kiểm tra cụm rủi ro cao (ví dụ 30 ngày xuống 15 ngày).
3. Dùng outlier của DBSCAN làm trigger kiểm tra bổ sung, không dùng làm quyết định dừng máy độc lập.
4. Theo dõi lại phân bố cụm theo tuần/tháng để phát hiện drift vận hành.

## 6. Câu nói chuẩn để dùng trong báo cáo
"Tab Phân cụm giúp khoanh vùng nhóm máy rủi ro cao và nhóm bất thường, từ đó tối ưu thứ tự ưu tiên bảo trì thay vì phân bổ nguồn lực đồng đều cho toàn bộ thiết bị."
