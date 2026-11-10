# KTVIC · Dữ liệu, đặc trưng ảnh và bộ chấm

## Dữ liệu cho hai mô hình

Với seed 42, bước chuẩn bị loại 15 ID ảnh train trùng hoặc gần trùng ảnh test, giữ các bản của cùng một ảnh trong cùng tập và giữ nguyên test gốc. Mỗi ảnh có năm caption. [Danh sách nhóm ảnh đã đối chiếu](../configs/ktvic_duplicate_image_groups.yaml)

| Tập | Ảnh | Caption |
|---|---:|---:|
| Train | 3.379 | 16.895 |
| Validation | 375 | 1.875 |
| Test | 558 | 2.790 |

Trong `data/processed/ktvic/`, mỗi dòng `{split}_manifest.jsonl` ghi ID, đường dẫn ảnh, hash pixel RGB và năm caption. File `{split}_ids.json` xác định thứ tự ảnh. Từ vựng gồm 1.716 token xây từ train sau khi chuẩn hóa Unicode NFC, viết thường, bỏ dấu câu và tách theo âm tiết. Decoder nhận `BOS + câu`, nhãn là `câu + EOS`. Mask loại PAD khi tính loss. [Mã tạo batch](../src/data/vietnamese_caption_batch.py)

## Đọc đặc trưng theo ID

ResNet-50 dùng trọng số `IMAGENET1K_V1`, đóng băng đến hết layer4 và chạy ở chế độ `eval()`. Ảnh RGB được chỉnh hướng theo EXIF, resize bilinear về 224×224 và chuẩn hóa theo ImageNet. Mỗi ảnh cho 49 vector 2.048 chiều trước phép chiếu của decoder. Chỉ số vùng là `hàng × 7 + cột`, đếm từ 0.

```python
from src.data.region_features import RegionFeatureStore

store = RegionFeatureStore.load("artifacts/features/ktvic", "val")
ids = store.image_ids[:2].tolist()
features = store.get_batch(ids)  # [2, 49, 2048], đúng thứ tự ID
origin = store.trace(ids[0])     # ID, đường dẫn ảnh và hash pixel
```

Mỗi cache `resnet50_7x7_{split}.pt` lưu tensor float16 cùng ID, đường dẫn, hash pixel và hash manifest. Báo cáo `runs/region_features_check/report.json` ghi số ID khớp và kết quả tính lại tensor từ ảnh gốc. Với các ID có pixel giống hệt nhau, phép truy ngược đối chiếu theo hash pixel.

## Chấm caption

Chuẩn bị một câu dự đoán cho mỗi ID của tập cần chấm. File JSON nhận danh sách các bản ghi như `[{"image_id": 2, "caption": "một người đi xe đạp"}]` hoặc ánh xạ từ ID sang câu. JSONL nhận một bản ghi trên mỗi dòng. Truyền đường dẫn file dự đoán qua `--predictions`. Ví dụ sau dùng file do lệnh chạy thử trong README tạo ra.

```sh
python -m scripts.score_captions --split val --predictions runs/scorer_trial/val_predictions_dung_ID.json --output runs/scorer_trial/val_scores.json
```

Bộ chấm so mỗi dự đoán với đủ năm caption của ảnh sau cùng bước chuẩn hóa văn bản. Điểm được nhân 100. BLEU-1…4 nằm trong khoảng 0–100, CIDEr-D trong khoảng 0–1.000. Khi so hai mô hình, dùng cùng một tập vì CIDEr tính IDF từ tham chiếu của tập đó.

Thiếu dự đoán hoặc thiếu file ảnh, lệnh chấm trả mã 2 và liệt kê ID liên quan. Ví dụ `Thiếu dự đoán cho 5/375 ảnh`. Khi chấm trên máy chỉ có manifest và dự đoán, dùng `--no-image-check` để bỏ bước kiểm tra file ảnh.

## Kết quả kiểm tra ngày 11/11/2026

Trên RTX 3060 với PyTorch 2.11.0+cu128 và torchvision 0.26.0+cu128, cache khớp đủ 4.312 ID. Cả 81 mẫu truy ngược đúng ảnh gốc. Bộ chấm dùng pycocoevalcap 1.2 và NumPy 2.4.4. [Phiên bản các gói](../requirements.txt)

Phép thử lấy caption đầu tiên của mỗi ảnh làm dự đoán, chấm một lượt đúng ID rồi tráo ID để kiểm tra bộ chấm.

| Tập | BLEU-4 đúng ID | BLEU-4 tráo ID | CIDEr đúng ID | CIDEr tráo ID |
|---|---:|---:|---:|---:|
| Validation | 100,00 | 8,18 | 348,87 | 10,75 |
| Test | 100,00 | 8,59 | 353,21 | 12,02 |

Chạy lại phép thử bằng `python -m scripts.run_scorer_trial --split val` hoặc thay `val` bằng `test`. Kết quả nằm trong `runs/scorer_trial/`. Cả 17 kiểm thử đều đạt với `python -m unittest discover -s tests -v`.
