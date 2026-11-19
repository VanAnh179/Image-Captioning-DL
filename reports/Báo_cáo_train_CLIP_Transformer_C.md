# Báo cáo train CLIP + Transformer (mô hình C) và so sánh với B

Nhánh `clip-transformer-training`. C là mô hình B với đặc trưng vùng lấy từ CLIP ViT-B/32 đóng băng thay cho ResNet-50.

## 1. Thiết lập

- **Ngân sách (giống B):** tối đa 20 epoch, batch 32, learning rate 0,0003, weight decay 0,01, seed 42, dừng sớm sau 4 epoch không tăng CIDEr val, chọn checkpoint theo CIDEr val cao nhất, sinh greedy tối đa 40 token.
- **Hai nơi train:** RTX 2050 4 GB (máy này) và Kaggle Tesla T4 (kernel `pengunins/ktvic-c-b-kaggle-seeds-20261119`, torch 2.11.0+cu130, cùng image với lần chạy B Kaggle trước). Mục 2 là RTX; phần "Kaggle T4" ở cuối mục 2 là Kaggle.
- **RTX:** B so sánh là lần train cùng máy, cùng seed và cùng cài đặt (`runs/train30/B`, CIDEr val 70,47; giới hạn 30 epoch nhưng dừng sớm ở epoch 11 nên giống hệt ngân sách 20 epoch).
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

### Năm seed

Sau lần train seed 42, tôi train thêm C với seed 1, 2, 3, 4, cùng ngân sách và đúng các seed đã dùng cho B (mỗi seed một lần train, cùng máy), để so sánh với dao động giữa các seed.

| Seed | B CIDEr | C CIDEr | C − B | Epoch tốt nhất B / C |
|---:|---:|---:|---:|---:|
| 1 | 72,84 | 80,87 | +8,03 | 6 / 5 |
| 2 | 71,64 | 82,87 | +11,23 | 7 / 5 |
| 3 | 71,52 | 79,84 | +8,32 | 6 / 6 |
| 4 | 70,75 | 81,44 | +10,69 | 8 / 8 |
| 42 | 70,47 | 79,12 | +8,65 | 7 / 5 |
| **Trung bình ± độ lệch chuẩn** | **71,44 ± 0,93** | **80,83 ± 1,45** | **+9,38** | |

Trung bình ± độ lệch chuẩn của BLEU-1/2/3/4: B 62,46±0,91 / 47,99±0,77 / 36,65±0,57 / 28,57±0,46; C 65,41±1,07 / 51,49±1,36 / 40,16±1,28 / 31,60±1,11. Chi tiết ở `reports/clip_transformer/nam_seed_B_C.json`.

### Kaggle T4

Để so sánh công bằng với B ở cả hai nơi, tôi train C và B trong cùng một kernel Kaggle, cùng môi trường, cùng ngân sách và cùng 5 seed (42, 1, 2, 3, 4; mỗi mô hình mỗi seed một lần train). Dữ liệu: cache CLIP của tôi và cache ResNet-50 trong dataset Kaggle sẵn có của B. Mã kernel: `scripts/kaggle_train_c_b.py`.

| Seed | B CIDEr | C CIDEr | C − B | Epoch tốt nhất B / C |
|---:|---:|---:|---:|---:|
| 1 | 75,98 | 82,75 | +6,77 | 5 / 7 |
| 2 | 72,22 | 76,70 | +4,48 | 4 / 7 |
| 3 | 69,99 | 79,44 | +9,45 | 7 / 9 |
| 4 | 70,47 | 79,87 | +9,40 | 8 / 6 |
| 42 | 72,42 | 78,73 | +6,31 | 6 / 4 |
| **Trung bình ± độ lệch chuẩn** | **72,22 ± 2,36** | **79,50 ± 2,19** | **+7,28** | |

- **B seed 42 tái lập đúng:** file dự đoán val của B seed 42 trong kernel này giống từng byte lần chạy Kaggle trước (CIDEr 72,42).
- **BLEU-1/2/3/4 trung bình:** B 63,54 / 48,97 / 37,22 / 28,78; C 65,22 / 51,30 / 40,10 / 31,47 (C − B: +1,68 / +2,33 / +2,88 / +2,69).
- **Tách hoàn toàn:** CIDEr thấp nhất của C (76,70) cao hơn CIDEr cao nhất của B (75,98), nhưng chỉ hơn 0,72 điểm. BLEU-3 và BLEU-4 cũng tách hoàn toàn; BLEU-1 và BLEU-2 thì các seed chồng lấn (C thấp nhất 63,02 và 49,25, B cao nhất 64,76 và 49,83). Xác suất tách hoàn toàn CIDEr nếu không có khác biệt là 1/252 (0,4%), thống kê t kiểu Welch 5,1.
- **Seed 42 chi tiết** (`reports/clip_transformer/so_sanh_B_C_kaggle.json`, hình `reports/assets/b_vs_c_val_kaggle.jpg`): B 72,42 → C 78,73 (+6,31); BLEU-1/2/3/4 +2,14 / +2,59 / +3,26 / +3,35. 365 trong 375 caption khác nhau; C hơn B ở 206 ảnh, B hơn C ở 159, bằng nhau 10. Khoảng tin cậy 95% khi lấy mẫu lại theo ảnh của chênh lệch CIDEr từng ảnh: −1,07 đến +14,06 (chứa 0).
- **Chi phí (T4):** thời gian train trung bình một lần B 212,7 giây, C 204,9 giây; seed 42: 21,3 so với 19,2 giây mỗi epoch, VRAM đỉnh 471 so với 452 MiB.
- **Dao động giữa seed trên Kaggle lớn hơn trên RTX** (độ lệch chuẩn CIDEr 2,2 đến 2,4 so với 0,9 đến 1,5), nên mức tăng ở từng seed trải rộng hơn (+4,5 đến +9,5).

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
2. **Mức tăng lặp lại ở cả 5 seed.** C cao hơn B ở mọi seed, từ +8,0 đến +11,2 CIDEr (trung bình +9,38; thống kê t kiểu Welch khoảng 12). CIDEr thấp nhất của C (79,12) vẫn cao hơn CIDEr cao nhất của B (72,84); BLEU-1 đến BLEU-4 cũng tách hoàn toàn như vậy. Với 5 lần train mỗi bên, xác suất tách hoàn toàn như thế nếu hai mô hình không khác nhau là 1/252, khoảng 0,4% (kiểm định hoán vị chính xác, một phía). Trong lần seed 42, CIDEr của C ở mọi epoch từ 4 đến 9 (73,3 đến 79,1) cũng cao hơn CIDEr cao nhất của B ở bất kỳ epoch nào (70,47).
3. **Kết quả lặp lại trên Kaggle T4.** Trong kernel Kaggle chung cho C và B, C cũng cao hơn B ở cả 5 seed (+4,5 đến +9,5 CIDEr, trung bình +7,28 so với +9,38 trên RTX) và CIDEr thấp nhất của C vẫn cao hơn cao nhất của B, nhưng biên chỉ 0,72 điểm do B Kaggle dao động mạnh hơn (độ lệch chuẩn 2,36). Gộp hai nơi: 10 cặp C−B cùng seed đều dương, nhỏ nhất +4,48. Mức tăng trên Kaggle nhỏ hơn trên RTX nhưng cùng hướng, nên không phụ thuộc vào máy train.
4. **Điều các seed chưa trả lời được.** (a) Năm seed chỉ cho biết dao động do khởi tạo và thứ tự dữ liệu; tất cả đều đo trên cùng 375 ảnh val. (b) Ở cấp từng ảnh nhiễu rất lớn: khi lấy mẫu lại theo ảnh (seed 42), chênh lệch CIDEr trung bình từng ảnh là +8,66 với khoảng tin cậy 95% từ −1,02 đến +18,34, chứa 0, vì CIDEr từng ảnh rất lệch (một vài ảnh tới ±400 đến ±500). (c) Val loss thấp nhất của seed 42 chỉ hơn 0,012 (0,6%), nên mức tăng hiện ở BLEU/CIDEr nhiều hơn ở độ khớp từng từ.
5. **Cách C khác B:** C hay chọn câu mô tả cảnh chung phổ biến trong tham chiếu (nhiều người, quán ăn, khu chợ) và đúng hơn ở các ảnh đông người hoặc có nhiều xe; số câu khác nhau (304 so với 299 trên 375 ảnh) và độ dài (12,10 so với 12,08 từ) gần như không đổi. Đây là nhận xét từ vài ví dụ, chưa kiểm chứng thống kê.
6. **Chi phí gần như không đổi (cả hai nơi):** thời gian mỗi epoch, VRAM và kích thước checkpoint tương đương; cache nhỏ hơn 2,7 lần. C chọn epoch 5 (B chọn epoch 7 theo CIDEr); val loss của cả hai bắt đầu tăng sau epoch 4 đến 5, tức là quá khớp từ khoảng epoch đó.

## 5. Giới hạn

- Mỗi seed một lần train cho mỗi mô hình, năm seed (1, 2, 3, 4, 42), ở hai nơi (RTX 2050 và Kaggle T4). B Kaggle seed 42 (72,42) khác B RTX seed 42 (70,47) nên số tuyệt đối thay đổi theo máy; chỉ so B với C trong cùng một nơi.
- Tất cả đo trên cùng một tập val 375 ảnh; chưa có số trên test.
- Chỉ đo trên val bằng BLEU/CIDEr; chưa chấm lỗi bằng người và chưa chạy test cho C.
- Đặc trưng CLIP chỉ dùng 49 vùng 7 × 7 ở lớp cuối, giống số vùng của ResNet-50; kết quả không nói được về cách dùng CLIP khác (cả CLS, fine-tune).
- Thời gian train mỗi lần chạy phụ thuộc số epoch dừng sớm (C 9 đến 12 epoch, B 10 đến 12 epoch); không so tổng thời gian giữa hai mô hình theo từng seed.
