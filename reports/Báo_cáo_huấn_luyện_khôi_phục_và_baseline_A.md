# Báo cáo vòng huấn luyện, khôi phục checkpoint và baseline A

**Phụ trách** Vanh · **Ngày** 13/11/2026 · Codex hỗ trợ viết mã, kiểm thử và gửi job Kaggle.

## Vòng train dùng chung A và B

`scripts/train_caption_model.py` dùng `src/training/caption_trainer.py` cho cả hai decoder. Dữ liệu được đọc bằng `src/data/cached_caption_dataset.py`, đối chiếu ID và hash manifest với cache. Train đầy đủ dùng mọi cặp ảnh–caption của train, validation dùng đủ năm câu tham chiếu mỗi ảnh.

Loss là tổng cross-entropy trên nhãn khác PAD chia số token hợp lệ, kể cả EOS. Khi tổng hợp theo epoch, loss được cân theo số token của từng batch. Validation chạy ở chế độ eval và tắt gradient. Điểm chọn checkpoint mặc định là CIDEr val sau khi sinh greedy toàn bộ validation. Caption rỗng do sinh EOS ngay được tính điểm trong vòng train, các kiểm tra ID vẫn áp dụng.

| File trong thư mục run | Nội dung |
|---|---|
| `last.pt` | Trọng số, optimizer, epoch, số bước, cấu hình, từ vựng, hash dữ liệu, trạng thái ngẫu nhiên và bản tốt nhất |
| `best.pt` | Checkpoint có CIDEr val cao nhất, giữ bản sớm hơn khi hòa điểm |
| `history.json` | Loss train/val, điểm số, thời gian và bộ nhớ từng epoch |
| `val_predictions.json` | Mỗi ID val một caption, sau export lấy từ checkpoint tốt nhất |
| `metrics.json`, `loss_curve.png` | Điểm của checkpoint tốt nhất, thời gian train/validation và đường loss |

Checkpoint được ghi vào file tạm rồi thay thế nguyên tử. Khôi phục từ epoch đã hoàn tất gần nhất. Nếu ngắt giữa epoch, lần resume chạy lại epoch dở. `last.pt` giữ bản tốt nhất bên trong để sửa lại `best.pt` khi có sự cố ghi file. Resume từ chối cấu hình hoặc dữ liệu khác bản đã lưu.

```sh
python -m scripts.train_caption_model --model A --output runs/a_baseline --epochs 20 --device cuda
python -m scripts.train_caption_model --model A --output runs/a_baseline --epochs 20 --resume runs/a_baseline/last.pt --device cuda
python -m scripts.export_caption_results --run-dir runs/a_baseline
```

Đổi `--model A` thành `--model B` và dùng thư mục output khác để train Transformer. Cấu hình lấy từ `configs/lstm_attention.yaml` hoặc `configs/transformer_decoder.yaml`. `--epochs` là tổng số epoch mục tiêu, bao gồm phần đã chạy. Khi resume, giữ nguyên các tùy chọn huấn luyện, đặc biệt batch, learning rate, seed và cách chọn checkpoint.

## Kiểm chứng đã hoàn tất

Trên máy phát triển, **40/40 kiểm thử đạt**. A và B đều được thử chạy liền 3 epoch so với chạy 1 epoch rồi khôi phục đến epoch 3. Trọng số, optimizer, trạng thái ngẫu nhiên, số bước và loss từng epoch khớp hoàn toàn trên CPU. Thêm 5 PAD giữ nguyên loss và số token hợp lệ. Các kiểm thử còn xác nhận chọn đúng checkpoint theo CIDEr, dừng sớm, xuất đúng caption của bản tốt nhất và giữ checkpoint cũ khi ghi file thất bại.

## Pilot và baseline đã gửi lên Kaggle

[Job riêng tư KTVIC A Baseline 20261113](https://www.kaggle.com/code/wann179/ktvic-a-baseline-20261113), kernel ID **144738063**, phiên bản **1**, đã được Kaggle nhận. Yêu cầu GPU T4, bật Internet để cài bộ chấm, giới hạn phiên 3 giờ. Đầu vào riêng tư chứa cache train/val và mã nguồn, tách khỏi token xác thực.

Job thực hiện tuần tự các bước sau.

1. Chạy bộ kiểm thử đã đóng gói, thử khôi phục A/B trên GPU và kiểm tra PAD.
2. Đo 20 batch train thật với batch 32 sau một batch làm nóng, ghi thời gian và VRAM allocated/reserved của A/B. Ước lượng thời gian train một epoch bằng thời gian mỗi batch nhân số batch train. Chi phí sinh caption val và lưu checkpoint được tính riêng.
3. Cho A học 300 lượt trên 16 cặp train, dropout 0, kiểm tra câu tự sinh, EOS và tổng attention gần 1. Mỗi lượt gồm một cập nhật trên đủ 16 cặp. Log ghi thời gian từng lượt và bộ nhớ cực đại của phần học.
4. Train A đầy đủ, chọn checkpoint theo CIDEr trên đủ 375 ảnh val. Sau cùng xuất điểm và caption từ bản tốt nhất.

| Cấu hình baseline A | Giá trị |
|---|---|
| Kích thước vùng, embedding, hidden, attention | 256 |
| Dropout | 0,1 |
| Optimizer, learning rate, weight decay | AdamW, 0,0003, 0,01 |
| Batch, seed | 32, 42 |
| Giới hạn | Tối đa 20 epoch, dừng sau 4 epoch không tăng CIDEr val |
| Giải mã | Greedy, tối đa 40 token, dừng EOS |

Ngân sách khởi đầu cho A và B là tối đa 20 epoch với cùng batch 32 và lịch đánh giá mỗi epoch. B hiện dùng cấu hình 512 chiều, 3 lớp, 8 đầu, có nhiều tham số hơn A. Pilot sẽ cung cấp số đo thực tế để chốt thời gian GPU cho B. Số giờ cần được tính từ số đo trên Kaggle, gồm cả validation và checkpoint. Thông số dừng sớm 4 epoch lấy từ phương án trong bản thiết kế.

## Vị trí kết quả khi job hoàn tất

Trong output Kaggle, thư mục `project/runs/` chứa `training_pilot/report.json`, `lstm_attention_check/report.json` và `a_baseline/`. Gói `a_baseline_results.zip` gom các run để tải về. Trong `a_baseline/`, `metrics.json` có BLEU-1…4 và CIDEr của checkpoint tốt nhất, `runtime.json` có thời gian thực chạy và GPU, `loss_curve.png` có đường loss, `val_predictions.json` có bản ghi dạng `{"image_id": ..., "caption": ...}` cho toàn bộ val, dùng trực tiếp cho bước lập chỉ mục tìm ảnh.

Theo yêu cầu chạy ngầm, lượt này dừng ở xác nhận gửi job. Điểm baseline, số đo GPU và kết quả pilot Kaggle chờ lần kiểm tra do Vanh yêu cầu. Các số kiểm thử CPU ở trên đã có thực, còn số thực nghiệm Kaggle sẽ bổ sung sau khi lấy output.
