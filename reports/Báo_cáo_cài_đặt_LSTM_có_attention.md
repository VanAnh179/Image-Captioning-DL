# Báo cáo cài đặt LSTM có attention

**Người phụ trách** Vanh · **Ngày** 13/11/2026 · **Trạng thái** Đã cài đặt và kiểm tra.

## Phần Linh Khánh đã bàn giao

Nhánh `transformer-decoder` có Transformer decoder tự viết và phép thử mặt nạ nhân quả. Đã sửa trường hợp toàn bộ attention bị che gây NaN, hỗ trợ số chiều vị trí lẻ, kiểm tra cấu hình và chặn phép thử trên tập rỗng. Kiểm tra trên 375 ảnh val cho 2.527 vị trí tiền tố có sai lệch logits bằng 0 khi thay hậu tố. Đã merge vào `main` tại `01ee78d` và giữ nhánh gốc.

## Phần LSTM đã hoàn thành

Ở mỗi bước, additive attention dùng trạng thái LSTM trước đó để tính trọng số cho 49 vùng. Softmax chuẩn hóa trọng số, tổng có trọng số tạo vector ngữ cảnh. LSTM nhận vector này cùng embedding token trước để dự đoán token tiếp theo.

- Đầu vào là đặc trưng ResNet `[B,49,2048]`. Lớp chiếu gồm Linear, LayerNorm và embedding vị trí, dùng chung cách cài với Transformer. Mỗi mô hình học trọng số riêng.
- `forward(regions, decoder_input)` trả `logits [B,T,1716]` và `attention [B,T,49]`. Khi học, dùng cross-entropy bỏ qua PAD.
- `generate(regions, max_len=40)` sinh greedy từ BOS, dừng riêng từng ảnh ở EOS hoặc giới hạn độ dài. Đầu ra có token ID, logits, attention, độ dài và cờ đã kết thúc. Token PAD/BOS được chặn khi chọn từ mới.
- Attention ở bước `t` ứng với token vừa sinh ở bước `t`. Đổi `[49]` thành `[7,7]` để vẽ bản đồ, vùng `i = hàng × 7 + cột`. Các bước đệm sau EOS có trọng số bằng 0.

| File | Vai trò |
|---|---|
| `src/models/lstm_attention_decoder.py` | Additive attention, LSTMCell, teacher forcing và sinh token |
| `src/models/region_projection.py` | Phép chiếu vùng dùng lại cho A và B, giữ tương thích checkpoint B |
| `configs/lstm_attention.yaml` | Cấu hình mặc định 256 chiều, dropout 0,1 |
| `scripts/check_lstm_attention.py` | Học thử 16 cặp train, kiểm tra attention và lưu/nạp checkpoint |
| `scripts/generate_lstm_captions.py` | Nạp checkpoint, xuất câu, logits và attention gắn với ID ảnh |
| `tests/test_lstm_attention_decoder.py` | Kiểm tra gradient, PAD, nhân quả, EOS, giới hạn độ dài và checkpoint |

## Kết quả kiểm tra

Chạy trên RTX 3060, PyTorch 2.11.0+cu128, seed 42. Phép thử chọn 16 ảnh train, lấy caption đầu của mỗi ảnh, tắt dropout và chạy AdamW 300 bước với learning rate 0,003. Mô hình có 2.469.556 tham số học được.

| Kiểm tra | Kết quả |
|---|---|
| Toàn bộ kiểm thử của repo | 34/34 đạt |
| Loss trên 16 cặp | 7,44856 → 0,00213 |
| Câu tự sinh khớp câu mẫu | 16/16, tất cả dừng ở EOS |
| Sai số lớn nhất của tổng attention so với 1 | 2,38 × 10⁻⁷ |
| Nạp lại checkpoint | Token, logits và attention giống hoàn toàn |
| Logits / attention trong lượt thử | `[16,18,1716]` / `[16,18,49]` |
| Lệnh sinh câu trên val | Đủ 375 ID, logits hữu hạn, attention khớp token, 375/375 có EOS |
| Tương thích B sau tách lớp chiếu | Nạp đúng toàn bộ trọng số, logits giữ nguyên |

Ví dụ ảnh train ID **1413** sinh “hai bé gái mặc áo dài màu trắng”, khớp câu mẫu. ID **9504** sinh “có bốn cô gái đang cầm trên tay những chiếc túi”, cũng khớp câu mẫu.

Các số trên xác nhận khả năng học tập nhỏ và tính đúng của đầu ra. Checkpoint hiện tại học trên 16 cặp train. Đánh giá chất lượng caption trên ảnh mới cần huấn luyện đầy đủ và chọn checkpoint bằng validation. Cấu hình B hiện mặc định 512 chiều, 3 lớp, 8 đầu, nên cần thống nhất ngân sách/cấu hình trước thí nghiệm so sánh A–B.

## Chạy lại và lấy kết quả

Chạy từ gốc repo sau khi đã có dữ liệu và cache theo README.

```sh
python -m unittest discover -s tests -v
python -m scripts.check_decoder_causality
python -m scripts.check_lstm_attention --device cuda
python -m scripts.generate_lstm_captions --checkpoint runs/lstm_attention_check/checkpoint.pt --split val --output runs/lstm_attention_check/val_generation --device cuda
```

Có thể đổi `cuda` thành `cpu`. File `runs/lstm_attention_check/report.json` chứa loss, cấu hình, ID và câu sinh. `checkpoint.pt` lưu mô hình thử, `outputs.pt` lưu logits và attention của 16 mẫu. Thư mục `val_generation/` chứa `val_predictions.json` và `val_attention.pt`. Mỗi bản ghi attention có ID, đường dẫn ảnh, token, logits `[T,1716]`, trọng số `[T,49]` và trạng thái EOS. Cache, checkpoint và kết quả chạy được lưu cục bộ theo quy ước repo.
