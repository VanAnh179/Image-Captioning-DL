# Nhận xét duyệt Transformer và milestone

**Ngày rà soát** 17/11/2026 · **Nhánh** `review-transformer-milestone` · **Mã nguồn đối chiếu** `main` tại `dfb6b52`.

## Kết luận theo từng task

| Task | Kết quả rà soát | Phần cần bổ sung để duyệt trọn task |
|---|---|---|
| B học đúng 16 cặp của Vanh | Code main lấy cùng 16 ảnh theo seed 42, caption đầu mỗi ảnh. Pilot có kiểm tra loss, resume và PAD. | Log B sinh câu, số mẫu có EOS, kiểm tra sinh lại sau nạp checkpoint và danh sách 16 ID. |
| Milestone tối đa 4 trang | Đã đọc bản Markdown và hai ảnh trong thư mục export. Có baseline, tiến độ, khó khăn, kế hoạch và sáu ID minh họa. | PDF để kiểm tra giới hạn 4 trang, link log với quyền truy cập. Đã sửa các kết luận thống kê trong bản Markdown. |
| Train B và bàn giao đầu ra | File `D:/Downloads/val_predictions.json` hợp lệ và chấm lại khớp toàn bộ điểm B RTX 2050, seed 42. | Checkpoint/history/metadata, log các seed và file caption B Kaggle 72,42 dùng trong phần so sánh cùng máy. |

**Có thể duyệt phần định dạng và điểm số của file caption B RTX seed 42. Các minh chứng còn mở cần được bàn giao để duyệt đầy đủ ba task.**

## Đã kiểm chứng trực tiếp

File B có **375 ID duy nhất**, đúng thứ tự `val_ids.json`, đủ **1.875 tham chiếu**, mọi caption đều có nội dung. Có **299 câu khác nhau**, trung bình **12,083 âm tiết/câu**; câu lặp nhiều nhất xuất hiện **9 lần**, khớp báo cáo.

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|---:|---:|---:|---:|---:|
| 61,51 | 47,29 | 36,14 | 28,30 | 70,47 |

Lệnh đối chiếu từ gốc repo dùng `python -m scripts.score_captions --predictions D:/Downloads/val_predictions.json --split val`. Điểm theo thang nhân 100. SHA-256 file B là `81ad0993fb8dbb9fe1283b438f38c86dfd153801c21ad50e32c7538347b99fa8`.

Bảng năm seed được tính lại từ số tác giả cung cấp. CIDEr trung bình **71,444**, độ lệch chuẩn mẫu **0,9251**, khoảng tin cậy t 95% có nửa độ rộng **1,1487**. Log gốc từng seed còn cần để xác nhận nguồn số.

Sáu caption và điểm CIDEr từng ảnh của A trong bảng minh họa đều khớp file A Kaggle 512/1120/512. Các câu tham chiếu đầu tiên cũng khớp manifest. File B được gửi là bản RTX 70,47, còn bảng minh họa ghi dùng B Kaggle 72,42; 5/6 câu B trong bảng khác file này, phù hợp với việc chúng thuộc hai lượt khác nhau. Cần đúng file Kaggle để xác minh toàn bộ ví dụ, số ảnh khác câu và số ảnh thắng/thua.

16 ID chuẩn của lượt Vanh là `9504, 1413, 290, 10664, 3463, 3129, 2908, 1809, 10622, 1261, 9825, 10655, 6109, 1063, 7353, 4923`. Cần đối chiếu cả caption đầu của từng ID khi nhận log B.

## Các sửa chữa đã thực hiện

- `main` vẫn tái hiện lỗi đọc checkpoint chứa metric NumPy bằng `weights_only=True`. Đã dùng lại bản sửa từ nhánh LSTM: metric mới lưu float Python, bộ đọc cho phép đúng các kiểu NumPy cần thiết của checkpoint cũ. Sau sửa, **41/41 kiểm thử** qua và đọc được checkpoint thật từng gây lỗi.
- Trong bản Markdown export, thay kết luận quy phần lớn chênh lệch cho máy bằng mô tả sự khác nhau giữa các lượt, kèm yêu cầu nhiều seed trên mỗi môi trường. Một seed mỗi ô chưa đủ để định lượng tác động của phần cứng.
- Thay diễn giải “điểm val cao hơn thực tế khoảng 4 điểm” bằng chênh **4,09 điểm giữa hai cách chọn checkpoint** trên val. Mức độ lạc quan so với khả năng tổng quát hóa cần dữ liệu đánh giá độc lập.
- Thay cách gọi checkpoint đỉnh CIDEr là “lỗi/đỉnh nhiễu” bằng mô tả checkpoint được chọn theo quy tắc đã chốt. Với tập tham chiếu val cố định, IDF cố định giữa các epoch; câu dự đoán thay đổi làm điểm thay đổi.
- Ghi rõ A dự kiến tối đa **20 epoch**, B được báo cáo tối đa **30**. Các lượt B báo cáo dừng ở 10–12, nên chưa chạm cả hai giới hạn. Cần chốt giới hạn chung cho những lượt so sánh tiếp theo.

Bản Markdown được sửa tại thư mục export người dùng cung cấp, giữ liên kết hai ảnh. Bản gốc được lưu trước khi sửa tại `runs/transformer_review/source_comparison_original.md`. Kết quả kiểm tra chi tiết lưu tại `runs/transformer_review/audit.json` và `scores_downloaded_predictions.json`.

`main` đã pull và đang ở `dfb6b52`. Log, checkpoint và PDF được bàn giao qua đường dẫn hoặc link riêng theo quy ước repo. Nhánh rà soát chứa bản sửa code và nhận xét để Vanh duyệt trước khi quyết định merge.
