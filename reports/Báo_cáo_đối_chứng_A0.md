# Báo cáo đối chứng A0

**Phụ trách** Vanh · **Ngày** 17/11/2026 · **Nhánh** `lstm-attention`.

## Đối chứng A0

A0 dùng `configs/lstm_attention_uniform.yaml`: vùng và embedding 512 chiều, hidden 1120, attention 512, dropout 0,1. Ở mỗi bước, 49 vùng đều nhận trọng số **1/49**; ngữ cảnh là trung bình các vùng đã chiếu. Các lớp giữ nguyên cấu trúc và khởi tạo như A. Nhánh tính điểm attention được giữ trong checkpoint để đối chiếu khởi tạo, còn đầu ra A0 dùng trọng số cố định.

Lượt A0 được gửi tới [Kaggle T4](https://www.kaggle.com/code/wann179/ktvic-a0-uniform-20261117), cùng cache, split, từ vựng, seed 42, batch 32, AdamW LR 0,0003, weight decay 0,01, tối đa 20 epoch, patience 4 theo CIDEr val, greedy tối đa 40 token. A0 được huấn luyện lại từ đầu. Mốc đối chiếu là A `512/1120/512`, checkpoint epoch 5, CIDEr val 76,96.

`scripts/compare_caption_models.py` kiểm tra cấu hình, thiết lập và chữ ký dữ liệu trước khi tính **A − A0** trên đủ 375 ảnh. Ví dụ được chọn từ các ảnh đổi caption, ưu tiên ba mức chênh CIDEr lớn nhất mỗi phía; đây là lựa chọn minh họa. `scripts/plot_caption_comparison.py` ghép ảnh gốc cùng hai caption và một câu tham chiếu.

Kết quả A0 đang chờ GPU Kaggle. Cập nhật tiếp sau khi đã có kết quả chạy.

Đã kiểm tra mã nguồn: trọng số A0 bằng 1/49, khởi tạo chung trùng A, gradient truyền tới LSTM/phép chiếu và phép đối chiếu A−A0. Điểm A0, chênh lệch với A và ảnh so sánh caption còn chờ output huấn luyện.

Phần tìm ảnh đã hoàn thành và được tách thành [báo cáo TF-IDF riêng](Báo_cáo_tìm_ảnh_TF_IDF.md).
