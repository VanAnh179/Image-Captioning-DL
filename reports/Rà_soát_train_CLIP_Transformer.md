# Rà soát train CLIP + Transformer

**Duyệt task train C và so sánh B/C trên val.** Nhánh `clip-transformer-training` có bảng điểm, số đo chi phí, log chọn epoch và ví dụ cho cả hai chiều chênh lệch. C đạt CIDEr 79,12 so với B 70,47 trên RTX 2050 ở seed 42. Trung bình chênh lệch theo cặp seed là +9,38 trên RTX và +7,28 trên T4. [Báo cáo B/C](Báo_cáo_train_CLIP_Transformer_C.md)

- **Đã sửa** phần tổng hợp để báo chênh lệch theo cặp cùng seed và bỏ phép kiểm định gộp hai nhóm. Báo cáo giới hạn kết luận ở hai môi trường đã chạy trên cùng 375 ảnh val, ghi rõ chi phí đo từ cache và bổ sung bước export checkpoint tốt nhất.
- **Đã kiểm tra** toàn bộ BLEU và CIDEr của hai cặp seed 42, 6 ví dụ mỗi cặp, khoảng tin cậy theo ảnh và bảng 5 seed mỗi môi trường. Log C RTX khớp epoch tốt nhất 5/9, 331 giây train và đỉnh cấp phát 403,3 MiB. Số tham số B/C khớp khi dựng decoder từ cấu hình. Bộ 63 kiểm thử đạt và lệnh train, resume, export chạy qua trên mẫu nhỏ dùng đặc trưng CLIP thật.
- **Giới hạn** số các seed còn lại và một phần chi phí dựa trên metric bàn giao. Cache CLIP val tại máy duyệt khớp manifest nhưng khác chữ ký lưu trong lượt train. Chưa có cache train và checkpoint C gốc tại đây. Điểm seed 42 trên T4 của B được đối chiếu bằng file dự đoán của lượt Kaggle trước.
