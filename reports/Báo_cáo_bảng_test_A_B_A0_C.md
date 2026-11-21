# Báo cáo bảng điểm test của A, B, A0 và C

Nhánh `test-score-table`. Mỗi mô hình đã khóa sinh một caption cho mỗi ID test; BLEU và CIDEr được tính lại từ file dự đoán đã lưu. Checkpoint đã chốt trên val trước khi chấm test, không chọn lại sau khi có điểm test.

## 1. Bảng

Tập test: **558 ảnh, 2.790 câu tham chiếu** (5 câu mỗi ảnh). Điểm nhân 100 (BLEU-1..4 và CIDEr-D, `pycocoevalcap` 1.2). Cả bốn mô hình lấy từ lần chạy Kaggle, seed 42, chọn epoch có CIDEr val cao nhất (C cũng dùng bản Kaggle để cùng môi trường với A, B và A0).

| Mô hình | Encoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | CIDEr val | Tham số học được | Epoch chọn | Train tới epoch chọn (giây) | Sinh 558 caption test (giây) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | ResNet-50 | 64,52 | 50,03 | 37,89 | 29,08 | **78,67** | 76,96 | 15.477.236 | 5 | 184,3 | 1,0 |
| B | ResNet-50 | 63,64 | 48,58 | 35,95 | 26,84 | **68,66** | 72,42 | 15.447.220 | 6 | 127,9 | 4,3 |
| A0 | ResNet-50 | 64,00 | 49,56 | 37,75 | 29,13 | **78,48** | 77,95 | 15.477.236 | 6 | 174,8 | 0,8 |
| C | CLIP ViT-B/32 | 66,55 | 51,74 | 39,60 | 30,51 | **75,98** | 78,73 | 14.791.860 | 4 | 77,0 | 3,9 |

- **Số ảnh:** mỗi mô hình có đúng 558 caption, đúng thứ tự `test_manifest.jsonl`, không caption rỗng, không caption nào quá 40 token. Có hai caption suy biến do lặp: C ở ảnh `5240` lặp từ "hàng" tới đúng giới hạn 40 token (không kết thúc bằng EOS), B ở ảnh `709` lặp cụm "của một cái hòm thư". Hai câu này vẫn được tính vào điểm.
- **Tham số học được:** tổng tham số train được của phần decoder (gồm lớp chiếu vùng ảnh). Encoder ResNet-50 và CLIP đóng băng, đặc trưng được lập cache nên không tính. A0 trùng A vì chỉ đổi cách tính trọng số attention thành 1/49. C ít hơn B 655.360 tham số vì lớp chiếu nhận 768 chiều thay cho 2.048.
- **Thời gian train:** cộng thời gian train các epoch từ đầu đến epoch được chọn, đọc từ lịch sử trong chính checkpoint (Kaggle, torch 2.11.0+cu130). Tổng thời gian cả lần chạy gồm các epoch dừng sớm lớn hơn và có ghi trong `metrics.json` (B 213,2 giây / 10 epoch, C 153,5 giây / 8 epoch, A0 291,3 giây / 10 epoch; chưa có `metrics.json` của A trên máy này).
- **Thời gian sinh:** đo trên máy này (RTX 2050), một lượt sau một lượt khởi động, chỉ gồm giải mã 558 ảnh (không gồm nạp checkpoint và cache). B và C chậm hơn A vì Transformer tính lại toàn bộ tiền tố ở mỗi bước. Mỗi số chỉ đo một lần nên dao động vài phần mười giây giữa các lần chạy.

## 2. Tạo lại bảng

Chỉ cần các file đã lưu (không cần checkpoint hay GPU):

```sh
python -m scripts.build_test_score_table
```

Lệnh đọc `reports/test_results/{A,B,A0,C}_test_predictions.json` và `{A,B,A0,C}_info.json`, kiểm tra đủ ID đúng thứ tự, chấm lại BLEU/CIDEr trên test và ghi `test_table.json`, `test_table.md`.

Sinh lại caption từ checkpoint rồi lập bảng (cần cache đặc trưng test của ResNet-50 và CLIP):

```sh
python -m scripts.build_test_score_table --generate --checkpoint A=<best.pt của A> --checkpoint B=<best.pt của B> --checkpoint A0=<best.pt của A0> --checkpoint C=<best.pt của C>
```

| Mô hình | SHA-256 checkpoint |
|---|---|
| A | `a80ea2a8b5bb3687952f3a638e036614d67ac9dd28ad447d7c31c478a38dcc63` |
| B | `90598b48a2ef6b04913eb366d47e9d17ef8635c2977002645c29ad345ce174fd` |
| A0 | `c0a10c18c4ada5a674ea8d3919315c4ed2431eb63283d97408e053068fae0ab7` |
| C (Kaggle) | `5a0fd307fffe504cacbbeec39455ec97a101a9d7e43565c415644f0353a7f9f3` |

Sinh greedy từ BOS, bỏ PAD/BOS khỏi lựa chọn, dừng ở EOS hoặc tối đa 40 token, batch 32, theo cài đặt lưu trong checkpoint. `test_manifest.jsonl` SHA-256 `d9c2e3b1…`. Lệnh ghi cả mã commit đã dùng vào `test_table.json`.

## 3. Đã kiểm tra

- Chấm lại từng file dự đoán bằng lệnh có sẵn `python -m scripts.score_captions --split test` cho đúng các số trong bảng.
- Sinh hai lượt liên tiếp ra cùng caption cho cả bốn mô hình (`generation_repeat_identical = true`); chạy lại cả quy trình hai lần cho cùng điểm.
- Với 25 ảnh trong phiếu chấm lỗi ẩn danh của Linh Khánh, ba caption trong phiếu khớp ba caption A/B/A0 của lần sinh này (so sánh theo tập, không đối chiếu mô hình), cho thấy phiếu và bảng dùng cùng caption.
- 79 kiểm thử của repo đạt, gồm kiểm thử mới cho script lập bảng.

## 4. Nhận xét

- **B thấp nhất ở mọi chỉ số** (CIDEr 68,66), thấp hơn A khoảng 10 điểm và thấp hơn C 7,3 điểm. C cao hơn B cũng đúng chiều với kết quả val (+6,3 ở seed 42 trên Kaggle).
- **A, A0 và C gần nhau về CIDEr** (78,67; 78,48; 75,98), còn C cao nhất ở cả bốn chỉ số BLEU. Trên val C cao hơn A và A0, trên test thấp hơn 2,5 đến 2,7 điểm CIDEr.
- Chênh lệch giữa A, A0 và C (tối đa 2,7 CIDEr) nhỏ hơn dao động giữa các seed đã đo trên val ở B và C trên Kaggle (độ lệch chuẩn 2,2 đến 2,4), nên không kết luận thứ hạng giữa ba mô hình này. A chưa được đo nhiều seed. Riêng khoảng cách của B với ba mô hình còn lại lớn hơn dao động đó.
- Theo quy ước đã chốt, không dùng điểm test để chọn lại checkpoint; kết quả C thấp hơn A trên test được giữ nguyên để phân tích.

## 5. Giới hạn

- Mỗi mô hình một checkpoint, một seed; chưa có khoảng tin cậy cho điểm test.
- Điểm BLEU/CIDEr không đo lỗi sai nội dung; bảng lỗi đã chấm tay là phần riêng trên 40 ảnh.
- Thời gian sinh đo trên RTX 2050 của máy này, không phải trên GPU train; thời gian train chỉ tính tới epoch được chọn.
- Điểm test của bản C RTX (79,12 trên val) không được tính vì đã chốt dùng bản Kaggle.

## 6. File

- `reports/test_results/{A,B,A0,C}_test_predictions.json`: 558 caption mỗi mô hình.
- `reports/test_results/{A,B,A0,C}_info.json`: hash checkpoint, tham số, epoch, thời gian train và sinh.
- `reports/test_results/test_table.json`, `test_table.md`: bảng điểm đầy đủ và bản markdown.
- Mã: `scripts/build_test_score_table.py`, `scripts/generate_caption_predictions.py`.
