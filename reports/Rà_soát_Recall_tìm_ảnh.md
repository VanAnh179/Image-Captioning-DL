# Rà soát Recall tìm ảnh A và B

**Đã xác nhận số liệu bàn giao từ hai CSV và tái lập Recall của B.** Còn thiếu file để tái lập chỉ mục A. [Báo cáo Recall](Báo_cáo_Recall_tìm_ảnh_A_và_B.md)

- Hai CSV khớp manifest ở từng ID ảnh đúng, số thứ tự và nội dung câu tham chiếu. Mỗi file có 1.875 truy vấn thuộc 375 ảnh. Tính lại số lượt trúng R@1/5/10 được A **88 / 309 / 456** và B **82 / 268 / 401**. Trung vị, biên theo hòa điểm và khoảng tin cậy lấy mẫu theo ảnh đều khớp kết quả lưu. Cả ba khoảng tin cậy của chênh lệch A − B đều chứa 0.
- Dựng lại chỉ mục B từ file dự đoán có SHA-256 khớp biên bản. Đối chiếu 1.875 hạng mới với `CaptionSearchIndex.search` cho kết quả trùng từng truy vấn. So với CSV gốc, OpenBLAS một luồng cho 3 hạng lệch một bậc và 4 dòng chỉ khác biên hoặc số ảnh hòa điểm. Với số luồng mặc định, tổng cộng 19 dòng khác, trong đó 9 dòng đổi hạng. Các khác biệt nằm ở điểm cosine sát nhau. Ba mức Recall của B khớp ở cả hai lượt. CSV gốc được giữ để đối chiếu với số bàn giao.
- **Đã sửa** cách xử lý truy vấn toàn từ ngoài từ vựng. CSV lưu hạng trống và Recall tính truy vấn đó là một lượt không trúng. Chuyển thư mục mặc định của chỉ mục đánh giá để tránh ghi đè chỉ mục A dùng cho demo. Cả 70 kiểm thử đạt. Lệnh CLI trên dữ liệu nhỏ tạo đủ hai chỉ mục và xuất đúng dòng truy vấn không có hạng.
- **Còn thiếu** file `runs/train30/A/val_predictions.json` có SHA-256 `36f3268037362d4afe41d6115c46b5fe402d4f459f8c144d52340fdb7273615e`, hoặc chỉ mục tạo từ đúng file đó. Khi có, có thể đối chiếu trực tiếp 1.875 hạng A với bản bàn giao.
