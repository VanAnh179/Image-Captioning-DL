# Chú thích ảnh tiếng Việt

Đồ án Học sâu của Vanh và Linh Khánh so sánh LSTM có attention (A, A0) với Transformer decoder (B, C) trên bộ ảnh KTVIC, rồi dùng caption tự sinh để tìm ảnh bằng câu tiếng Việt. Hướng dẫn dưới đây đi từ máy trống đến khi mở được demo. Chạy mọi lệnh từ gốc repo. Mỗi bước nói rõ file nào được tạo ra và kết quả cần thấy.

## 1. Môi trường

Cần Python 3.13. Chọn một trong hai cách:

```sh
# conda
conda env create -f environment.yml
conda activate ktvic

# hoặc venv
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

`environment.yml` và `requirements.txt` ghim cùng phiên bản các gói. Máy có GPU NVIDIA trên Windows cần cài torch bản CUDA trước, vì PyPI chỉ có bản CPU cho Windows:

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Kiểm tra máy và thư mục bằng một lệnh. Lệnh báo gói nào thiếu hoặc khác phiên bản, có CUDA hay không, có trọng số ResNet-50 hay chưa và thư mục nào còn thiếu. Mã thoát 1 nghĩa là thiếu gói.

```sh
python -m scripts.check_setup
```

Hai bộ trọng số có sẵn được tải về lần đầu khi có mạng, sau đó máy chạy offline được:
- ResNet-50: lệnh trích đặc trưng tự tải, hoặc tải trước bằng `python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"`. File `resnet50-0676ba61.pth` nằm trong thư mục `checkpoints` của cache Torch Hub. Demo offline đọc file này, nên máy không có mạng cần chép nó vào đúng thư mục đó.
- CLIP ViT-B/32 (chỉ cho mô hình C): `extract_clip_features` tự tải `openai/clip-vit-base-patch32` vào cache Hugging Face ở lần chạy đầu.

Không có GPU thì thêm `--device cpu` vào các lệnh dưới đây.

## 2. Thư mục

Mỗi loại tệp có một chỗ riêng. Các thư mục này không vào Git. Mọi lệnh đều có tham số để đổi đường dẫn khi cần. Các đường dẫn mặc định khai báo ở [src/data/paths.py](src/data/paths.py).

| Loại | Đường dẫn | Do lệnh nào tạo |
|---|---|---|
| Dữ liệu thô | `data/raw/ktvic/` | Bạn tải về (mục 3) |
| Dữ liệu đã xử lý | `data/processed/ktvic/` | `scripts.prepare_ktvic_dataset` |
| Đặc trưng ảnh | `artifacts/features/ktvic/` | `scripts.extract_region_features`, `scripts.extract_clip_features` |
| Checkpoint | `artifacts/checkpoints/ktvic/<tên>/best.pt` | `scripts.train_caption_model` hoặc file bàn giao |
| Chỉ mục tìm ảnh | `artifacts/retrieval/` | `scripts.build_image_search_index`, `scripts.evaluate_caption_retrieval` |
| Log | `logs/` | Huấn luyện, trích đặc trưng và sinh caption tự ghi vào đây |
| Dự đoán, bảng điểm | `reports/`, kết quả Recall ở `runs/` | `scripts.build_test_score_table` và các lệnh ở mục 6 |

Một lần huấn luyện tên `A` ghi `best.pt`, `last.pt`, `history.json` và `val_predictions.json` vào `artifacts/checkpoints/ktvic/A/`, còn log ở `logs/train_A.log`.

## 3. Dữ liệu và đặc trưng ảnh

Đặt `train_data.json` và `test_data.json` vào `data/raw/ktvic/`, rồi giải nén hai gói ảnh vào đó để có `train-images/` và `public-test-images/`. [Tải annotation KTVIC](https://github.com/pacman-ctm/ktvic/tree/0412501549d57987e8b1db8b39a97a9040872822) · [Tải hai gói ảnh](https://drive.google.com/drive/folders/16e8cd3AKusPS1H-h55JModiINM0og-ke)

```sh
python -m scripts.prepare_ktvic_dataset --seed 42
python -m scripts.extract_region_features --device cuda --batch-size 32 --workers 4
python -m scripts.verify_region_features --device cuda --samples 25
```

- Lệnh đầu chia dữ liệu thành train 3.379 ảnh, val 375 ảnh và test 558 ảnh, ghi vào `data/processed/ktvic/`. SHA-256 của `train_manifest.jsonl` phải bắt đầu bằng `cf9835ca`, `val_manifest.jsonl` bằng `ace6e16f`, `test_manifest.jsonl` bằng `d9c2e3b1`.
- Lệnh thứ hai ghi ba file `resnet50_7x7_{train,val,test}.pt` vào `artifacts/features/ktvic/`. Trên RTX 2050 mất khoảng 1,5 phút.
- Lệnh thứ ba kết thúc bằng dòng `OK: số ID khớp từng tập…`.

Mô hình C cần thêm đặc trưng CLIP. Bảng điểm test (mục 6) chỉ cần tập test, còn huấn luyện C cần cả ba tập. Trên RTX 2050, tập train mất khoảng 3 phút.

```sh
python -m scripts.extract_clip_features --device cuda                  # cả ba tập
python -m scripts.extract_clip_features --device cuda --splits test    # chỉ tập test
```

Chi tiết dữ liệu và cách chấm: [hướng dẫn KTVIC](docs/ktvic.md).

## 4. Chạy một batch huấn luyện

Lệnh sau lấy 32 cặp ảnh–câu, đúng một batch, huấn luyện một epoch và chấm trên chính 32 cặp đó. Dùng để kiểm tra cả đường đi từ dữ liệu đến checkpoint.

```sh
python -m scripts.train_caption_model --model A --overfit-pairs 32 --batch-size 32 --epochs 1 --output artifacts/checkpoints/ktvic/A_batch
```

Kết quả cần thấy là một dòng `epoch 1/1 train=7.4… val=7.3… cider=… best_epoch=1`. Loss khoảng 7,4 là đúng, vì từ vựng có 1.716 token và mô hình chưa học gì sau một bước. Các file được tạo:
- `artifacts/checkpoints/ktvic/A_batch/best.pt`, `last.pt`, `history.json`, `val_predictions.json`
- `logs/train_A_batch.log`, bản sao toàn bộ dòng in ra màn hình

Chạy lại với cùng `--output` báo lỗi vì đã có `last.pt`. Đổi tên `--output`, hoặc huấn luyện tiếp bằng cách thêm `--resume artifacts/checkpoints/ktvic/A_batch/last.pt` và tăng `--epochs` (ví dụ `--epochs 2`). Log được ghi nối vào cùng file.

Huấn luyện đầy đủ (tối đa 20 epoch, dừng sớm sau 4 epoch không cải thiện CIDEr trên val):

```sh
python -m scripts.train_caption_model --model A  --output artifacts/checkpoints/ktvic/A
python -m scripts.train_caption_model --model A0 --output artifacts/checkpoints/ktvic/A0
python -m scripts.train_caption_model --model B  --output artifacts/checkpoints/ktvic/B
python -m scripts.train_caption_model --model B  --config configs/clip_transformer.yaml --feature-prefix clip_vit_b32_7x7 --output artifacts/checkpoints/ktvic/C
```

Dòng cuối là mô hình C (cần đặc trưng CLIP của cả ba tập). Huấn luyện bằng CPU rất chậm, nên dùng GPU hoặc Kaggle.

## 5. Nạp checkpoint

Bốn checkpoint đã khóa để làm bảng test được bàn giao qua Media board. Đặt vào `artifacts/checkpoints/ktvic/{A,A0,B,C}/best.pt` và kiểm tra SHA-256 (`Get-FileHash <file>` trên PowerShell, `sha256sum <file>` trên Linux):

| Mô hình | SHA-256 bắt đầu bằng |
|---|---|
| A | `a80ea2a8` |
| A0 | `c0a10c18` |
| B | `90598b48` |
| C | `5a0fd307` |

Nạp checkpoint A, sinh caption cho tập val và chấm:

```sh
python -m scripts.generate_caption_predictions --checkpoint artifacts/checkpoints/ktvic/A/best.pt --split val --output reports/val_predictions_A.json
python -m scripts.score_captions --split val --predictions reports/val_predictions_A.json
```

Điểm cần thấy cho A trên val: BLEU-4 30,17 và CIDEr-D 76,96. Log của lệnh sinh caption nằm ở `logs/generate_val_predictions_A.log`.

## 6. Tái tạo hai bảng điểm

**Bảng captioning** sinh lại caption test từ bốn checkpoint, rồi chấm BLEU-1…4 và CIDEr-D. Cần đặc trưng ResNet-50 và CLIP của tập test.

```sh
python -m scripts.build_test_score_table --generate --checkpoint A=artifacts/checkpoints/ktvic/A/best.pt --checkpoint B=artifacts/checkpoints/ktvic/B/best.pt --checkpoint A0=artifacts/checkpoints/ktvic/A0/best.pt --checkpoint C=artifacts/checkpoints/ktvic/C/best.pt
```

Kết quả ghi vào `reports/test_results/` (`test_table.md`, `test_table.json` và caption từng mô hình). CIDEr-D phải là A 78,67, B 68,66, A0 78,48, C 75,98.

**Bảng tìm ảnh** lập chỉ mục TF-IDF từ caption tự sinh của từng mô hình, dùng 2.790 câu tham chiếu test làm truy vấn, rồi ghép với bảng captioning. Đặt `OPENBLAS_NUM_THREADS=1` để số truy vấn hòa điểm khớp lần đã ghi.

```sh
# PowerShell: $env:OPENBLAS_NUM_THREADS = '1'      bash: export OPENBLAS_NUM_THREADS=1
python -m scripts.evaluate_caption_retrieval --split test --predictions-a reports/test_results/A_test_predictions.json --predictions-b reports/test_results/B_test_predictions.json
python -m scripts.evaluate_caption_retrieval --split test --predictions-a reports/test_results/A0_test_predictions.json --predictions-b reports/test_results/C_test_predictions.json --output-dir runs/retrieval_test_eval_a0_c --index-dir artifacts/retrieval/recall_eval_a0_c
python -m scripts.build_caption_retrieval_table --recall A,B=runs/retrieval_test_eval --recall A0,C=runs/retrieval_test_eval_a0_c
```

Bảng ghép ở `reports/caption_retrieval_table/bang_captioning_va_tim_anh.md`. Số truy vấn trúng ở top 1, 5 và 10 phải là A 107, 360, 541; B 108, 286, 466; A0 128, 369, 544; C 89, 308, 470. Lệnh ghép chỉ ghi bảng khi mọi kiểm tra chéo giữa các file đều đạt.

## 7. Demo offline

Demo cần checkpoint A và chỉ mục lập từ chính caption val của checkpoint đó. Lập chỉ mục từ file dự đoán ở mục 5, rồi mở demo:

```sh
python -m scripts.build_image_search_index --predictions reports/val_predictions_A.json
python -m scripts.run_web_demo
```

Chỉ mục được ghi ở `artifacts/retrieval/ktvic_a_val.npz`. Demo tự tìm checkpoint ở `artifacts/checkpoints/ktvic/A/best.pt` và chỉ mục ở đường dẫn trên. Nó từ chối chạy nếu chỉ mục không lập từ caption của checkpoint này. Trang web mở ở `http://127.0.0.1:8000`: chọn ảnh để sinh caption, hoặc gõ câu tiếng Việt để xem tối đa 5 ảnh khớp nhất kèm caption và các từ khớp. Nhấn Ctrl+C trong terminal để dừng.

Chạy được offline vì chỉ đọc file local: checkpoint, chỉ mục, ảnh dưới `data/raw/ktvic/` và `resnet50-0676ba61.pth` trong cache Torch Hub.

Tùy chọn: `--checkpoint`, `--index` và `--image-root` để chỉ đường dẫn khác, `--device cpu` để chạy bằng CPU, `--no-browser` để không tự mở trình duyệt, `--port` để đổi cổng, `--window` để mở cửa sổ riêng bằng Edge hay Chrome. Bản cửa sổ Tkinter chạy bằng `python -m scripts.run_demo`.

## 8. Kiểm thử

```sh
python -m unittest discover -s tests
```

Cả bộ kiểm thử không cần dữ liệu hay GPU và chạy trong khoảng 15 giây.

## Lỗi thường gặp

- `Không thấy train_manifest.jsonl`: chưa chạy `prepare_ktvic_dataset` hoặc thiếu dữ liệu thô ở `data/raw/ktvic/`.
- `Không có CUDA` trong `check_setup` trên máy có GPU NVIDIA: torch đang là bản CPU. Cài lại theo mục 1.
- `OMP: Error #15` trên Windows: môi trường có hai bản OpenMP. Tạo môi trường mới theo mục 1 thay vì dùng chung với gói khác.
- `Chỉ mục … không được lập từ caption của checkpoint`: lập lại chỉ mục bằng `build_image_search_index` từ file dự đoán của đúng checkpoint đang dùng.
- `Run already has last.pt`: thư mục `--output` đã có lần huấn luyện cũ. Đổi tên hoặc dùng `--resume`.

Báo cáo nội bộ, log huấn luyện và kết quả đánh giá chi tiết được bàn giao riêng qua Media board.
