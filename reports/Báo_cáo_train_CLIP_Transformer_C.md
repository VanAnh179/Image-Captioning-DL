# Báo cáo train CLIP + Transformer (mô hình C) và so sánh với B

Nhánh `clip-transformer-training`. C là mô hình B với đặc trưng vùng lấy từ CLIP ViT-B/32 đóng băng thay cho ResNet-50. Phần nào khác B xem `docs/clip_encoder_thay_so_voi_B.md` và `reports/Báo_cáo_ghép_CLIP_với_Transformer.md`; ở lần train này chỉ có `feature_dim` 2048 → 768 (cấu hình `configs/clip_transformer.yaml`).

## 1. Thiết lập

- **Ngân sách (giống B):** tối đa 20 epoch, batch 32, learning rate 0,0003, weight decay 0,01, seed 42, dừng sớm sau 4 epoch không tăng CIDEr val, chọn checkpoint theo CIDEr val cao nhất, sinh greedy tối đa 40 token.
- **Máy:** RTX 2050 4 GB. B so sánh là lần train cùng máy, cùng seed và cùng cài đặt (`runs/train30/B`, CIDEr val 70,47; giới hạn 30 epoch nhưng dừng sớm ở epoch 11 nên giống hệt ngân sách 20 epoch).
- **Lệnh:** `python -m scripts.train_caption_model --model B --config configs/clip_transformer.yaml --feature-prefix clip_vit_b32_7x7 --output runs/clip_transformer/c_seed42 --epochs 20 --seed 42`. Cần thêm tùy chọn `--feature-prefix` (đã sửa ở `scripts/train_caption_model.py` và `src/data/cached_caption_dataset.py`, mặc định vẫn là cache ResNet-50 nên B không đổi).
- **Lần chạy bị ngắt sau epoch 1** do môi trường (không có lỗi trong log). Tôi chạy tiếp từ `last.pt` bằng `--resume`; trọng số, optimizer và trạng thái ngẫu nhiên được khôi phục từ checkpoint, nên đây là một lần train liên tục về mặt thuật toán. Log đủ 9 epoch ở `reports/clip_transformer/log_c_seed42.txt`.
- **Chỉ dùng val.** Không dùng test để chọn epoch hay sửa mô hình.

## 2. Bảng B/C (val, 375 ảnh, 1.875 câu tham chiếu)

| | B (ResNet-50) | C (CLIP) | C − B |
|---|---:|---:|---:|
| BLEU-1 | 61,51 | 65,61 | +4,10 |
| BLEU-2 | 47,29 | 51,62 | +4,33 |
| BLEU-3 | 36,14 | 39,94 | +3,80 |
| BLEU-4 | 28,30 | 31,13 | +2,83 |
| **CIDEr** | **70,47** | **79,12** | **+8,65** |
| Epoch tốt nhất / đã chạy | 7 / 11 | 5 / 9 | |
| Val loss thấp nhất | 1,898 | 1,886 | −0,012 |

Điểm được tính lại bằng `scripts.score_captions` từ file dự đoán và khớp số lưu trong checkpoint.

**Chi phí** (đo trên RTX 2050):

| | B | C |
|---|---:|---:|
| Tham số train được | 15.447.220 | 14.791.860 (−4,2%, hết ở lớp chiếu vùng: 512 × 1.280) |
| Thời gian train mỗi epoch | 37,4 giây | 36,8 giây |
| Tổng thời gian train | 412 giây (11 epoch) | 331 giây (9 epoch) |
| VRAM đỉnh khi train | 423 MiB | 403 MiB |
| Checkpoint tốt nhất | 177 MiB | 170 MiB |
| Cache đặc trưng 3 tập | 826 MiB | 310 MiB |
| Lập cache tập val (375 ảnh) | 9 giây | 10 giây |

Encoder CLIP đóng băng nên không tốn thêm tham số train hay VRAM khi train; chi phí duy nhất là chạy encoder một lần để lập cache (cùng cỡ với ResNet-50) và khi suy luận trên ảnh mới. Cache nhỏ hơn 2,7 lần vì 768 chiều thay cho 2.048.

## 3. Ví dụ

Hình [reports/assets/b_vs_c_val.jpg](assets/b_vs_c_val.jpg) cho sáu ảnh val: ba ảnh C có điểm CIDEr cao hơn B nhiều nhất và ba ảnh B hơn C nhiều nhất, trong số 370 ảnh hai mô hình sinh câu khác nhau. Đây là ví dụ minh họa chọn theo chênh lệch lớn, không đại diện cho cả tập. Chi tiết trong `reports/clip_transformer/so_sanh_B_C.json`.

| ID | B | C | Một câu tham chiếu |
|---|---|---|---|
| 4770 | có một chiếc xe hơi đang đỗ ở bên lề đường | có nhiều phương tiện giao thông đang di chuyển ở trên đường | có nhiều phương tiện giao thông đang di chuyển trên đường |
| 2278 | có một người đàn ông đang ngồi ở bên một quầy hàng | có nhiều người đang ngồi ăn ở bên trong quán ăn | có nhiều người đang xuất hiện trong quán ăn |
| 11435 | có nhiều người đang đứng xung quanh các quầy hàng | có nhiều người đang xuất hiện ở trong khu chợ | có nhiều người đang xuất hiện ở bên trong khu chợ |
| 10461 | có nhiều mặt hàng đồ trang trí xuất hiện ở trong cửa hàng | có nhiều mặt hàng được bày bán ở trong quầy hàng | có nhiều người xuất hiện ở trong cửa hàng bán đồ trang trí |
| 5799 | có một người đàn ông đang cầm trên tay một quyển sách | có một người đàn ông đang cầm trên tay một chiếc hộp quà | có một quyển sách ở trên tay của người trong bức ảnh |
| 10921 | có một vài chiếc đèn lồng được treo ở trên trần nhà | có một vài người đang xuất hiện ở trong bức ảnh | có một người đàn ông đang ngồi trước một ngôi nhà cổ |

Ở ba ảnh đầu C nhận đúng cảnh (đường nhiều xe, quán ăn đông người, chợ). Ở ID 5799 C nhận ra vật cầm trên tay nhưng gọi nhầm quyển sách thành hộp quà (lỗi vật thể); B đúng. Ở ID 10461 và 10921 C bỏ chi tiết mà câu tham chiếu nhắc.

## 4. Nhận xét về tác động khi đổi encoder

1. **C cao hơn B ở cả năm chỉ số:** BLEU tăng 2,8 đến 4,3 điểm và CIDEr tăng 8,65 điểm (70,47 → 79,12). C hơn B ở 221 ảnh, B hơn C ở 149 ảnh, 5 ảnh bằng nhau.
2. **Khoảng cách lớn so với dao động giữa các seed của B.** Năm seed của B cho CIDEr trung bình 71,44, độ lệch chuẩn 0,93, cao nhất 72,84. C seed 42 (79,12) cao hơn mức cao nhất đó 6,3 điểm. CIDEr của C ở mọi epoch từ 4 đến 9 (73,3 đến 79,1) đều cao hơn CIDEr cao nhất của B ở bất kỳ epoch nào (70,47).
3. **Nhưng chưa phải bằng chứng chắc chắn.** (a) Mới có một lần train của C. (b) Khi lấy mẫu lại theo ảnh, chênh lệch CIDEr trung bình từng ảnh là +8,66 với khoảng tin cậy 95% từ −1,02 đến +18,34, chứa 0, vì CIDEr từng ảnh rất lệch (một vài ảnh tới ±400 đến ±500). (c) Val loss chỉ thấp hơn 0,012 (0,6%), nên phần lớn mức tăng hiện ở BLEU/CIDEr chứ không ở độ khớp từng từ.
4. **Cách C khác B:** C hay chọn câu mô tả cảnh chung phổ biến trong tham chiếu (nhiều người, quán ăn, khu chợ) và đúng hơn ở các ảnh đông người hoặc có nhiều xe; số câu khác nhau (304 so với 299 trên 375 ảnh) và độ dài (12,10 so với 12,08 từ) gần như không đổi. Đây là nhận xét từ vài ví dụ, chưa kiểm chứng thống kê.
5. **Chi phí gần như không đổi:** thời gian mỗi epoch, VRAM và kích thước checkpoint tương đương; cache nhỏ hơn 2,7 lần. C chọn epoch 5 (B chọn epoch 7 theo CIDEr); val loss của cả hai bắt đầu tăng sau epoch 4 đến 5, tức là quá khớp từ khoảng epoch đó.

## 5. Giới hạn

- Một lần train của C (seed 42), một lần train của B cùng seed, một máy (RTX 2050). Kết quả Kaggle của B (72,42) khác bản RTX (70,47) và chưa có bản C trên Kaggle.
- Chỉ đo trên val bằng BLEU/CIDEr; chưa chấm lỗi bằng người và chưa chạy test cho C.
- Đặc trưng CLIP chỉ dùng 49 vùng 7 × 7 ở lớp cuối, giống số vùng của ResNet-50; kết quả không nói được về cách dùng CLIP khác (cả CLS, fine-tune).
- Muốn khẳng định mức tăng cần thêm vài seed cho C (mỗi seed khoảng 6 phút train).

## 6. File

- `reports/clip_transformer/so_sanh_B_C.json`: bảng điểm, chi phí, đếm ảnh, ví dụ.
- `reports/clip_transformer/c_seed42_metrics.json`, `c_seed42_history.json`, `c_seed42_val_predictions.json`, `log_c_seed42.txt`: kết quả train của C.
- `reports/clip_transformer/b_seed42_metrics.json`, `b_seed42_val_predictions.json`: B dùng để so sánh.
- Checkpoint tốt nhất của C ở `runs/clip_transformer/c_seed42/best.pt` (170 MiB, ngoài Git).
- Mã: `scripts/compare_b_c.py` (bảng và hình), `scripts/train_caption_model.py`.
