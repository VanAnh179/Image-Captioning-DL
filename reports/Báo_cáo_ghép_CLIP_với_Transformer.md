# Báo cáo ghép CLIP với Transformer decoder

**Triển khai** Linh Khánh · **Rà soát** 18/11/2026 · **Nhánh nguồn** `clip-encoder`.

## Thay đổi so với B

Mô hình C lấy 49 vùng từ CLIP ViT-B/32 đóng băng và đưa chúng vào Transformer decoder của B. Bộ xử lý tải cùng phiên bản trọng số [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32/raw/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/preprocessor_config.json).

| Thành phần | B | C |
|---|---|---|
| Chuẩn bị ảnh | Co thẳng 224×224 bằng bilinear, chuẩn hóa ImageNet | Co cạnh ngắn về 224 bằng bicubic, cắt giữa 224×224, chuẩn hóa CLIP |
| Đặc trưng ảnh | ResNet-50 `layer4`, `[49, 2048]` | CLIP ViT-B/32, 49 token patch của lớp cuối sau khi bỏ CLS, `[49, 768]` |
| Chiếu vùng | `Linear(2048, 512)` | `Linear(768, 512)` |
| Số tham số học của decoder và lớp chiếu | 15.447.220 | 14.791.860 |

C dùng phép chiếu chung của repo gồm Linear, LayerNorm và embedding vị trí 49 vùng. Decoder dùng cùng 3 lớp, 8 đầu, `d_model=512`, FFN 2.048, từ vựng 1.716 token và quy tắc sinh greedy của B. Checkpoint đầy đủ của B có ma trận chiếu khác kích thước. Cần checkpoint riêng khi huấn luyện C.

## Kết quả Linh Khánh bàn giao

| Log | Kết quả |
|---|---|
| [Pipeline 8 ảnh val](clip_encoder/clip_pipeline_check.json) | `[8, 3, 224, 224]` → `[8, 49, 768]` → logits `[8, 20, 1716]`. Hash và truy ngược ID khớp 8/8. CLIP đóng băng, lớp chiếu có gradient. Sau 300 bước học trên chính 8 ảnh, decoder sinh đúng 8 câu được dùng để học, loss 0,00019. |
| [Kiểm tra cache CLIP](clip_encoder/clip_features_check.json) | Tính lại 3.379 train, 375 val và 558 test. Sai lệch tuyệt đối tối đa so với cache float16 là 0,00391. Truy ngược đúng 52/52 mẫu mỗi tập theo nội dung ảnh. |
| [Ảnh hưởng TF32 lên B](clip_encoder/tf32_effect_on_B.json) | Đối với checkpoint B trong log, tắt TF32 lúc trích lại đặc trưng đổi 1/375 caption và CIDEr val từ 70,47 xuống 70,30. Kết quả mô tả riêng checkpoint và cấu hình đã thử. |

Đã đối chiếu số ảnh của ba tập và từng ID, đường dẫn, câu tham chiếu đầu tiên của tám ảnh val với manifest trong repo. Ba JSON giữ nguyên nội dung được cung cấp. Kết quả học tám ảnh kiểm tra đường truyền dữ liệu. C còn chờ huấn luyện đầy đủ và chấm BLEU/CIDEr.

## Kết quả chạy lại sau khi ghép `main`

- RTX 3060, PyTorch 2.11.0+cu128 và Transformers 5.8.1. Nạp đúng phiên bản trọng số CLIP nêu trên.
- Pipeline qua đầy đủ tám ảnh val. Truy ngược đúng 8/8 ID. Sau 300 bước học thử, sinh đúng 8/8 câu đã học, loss 0,00019.
- Tính lại đủ 375 ảnh val. Truy ngược theo nội dung ảnh đạt 52/52. Sai lệch tối đa với cache float16 là 0,00391. Tất cả 56 kiểm thử mã nguồn đạt.
- Số liệu cache train/test và TF32 lấy từ log Linh Khánh bàn giao. Lượt chạy lại tập trung vào pipeline và cache val.

## Sửa khi tích hợp và cách chạy

- Bộ đọc cache nhận kích thước vùng của từng encoder và kiểm tra shape, ID, đường dẫn cùng hash. Script xác minh CLIP kiểm tra thêm phiên bản trọng số và hash manifest. Cache ResNet và CLIP có tên tệp riêng.
- Transformer dùng lớp chiếu chung trên `main`. Script pipeline dùng cùng hàm greedy với vòng train và báo mã lỗi khi các kiểm tra chính thất bại. Khai báo `transformers==5.8.1` theo môi trường chạy lại. Báo cáo gốc dùng 5.8.0.
- Kết quả lệnh chạy mới nằm trong `runs/clip_encoder/`. Cache nằm trong `artifacts/features/ktvic/`.

```sh
python -m scripts.extract_clip_features --splits val --device cuda
python -m scripts.check_clip_pipeline --device cuda
python -m scripts.verify_clip_features --splits val --device cuda
```

Lệnh đầu tải trọng số nếu máy chưa có bản cache Hugging Face. Bỏ `--splits val` ở lệnh trích và xác minh để xử lý cả ba tập. Dùng `--overfit-steps 0` để kiểm tra nhanh đường đi, bỏ lượt học trên tám ảnh.
