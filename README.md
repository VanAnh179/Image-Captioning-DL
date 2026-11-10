# Chú thích ảnh tiếng Việt

Bài tập lớn Học sâu, đề tài 05 của Vanh và Linh Khánh. Hệ thống nhận một ảnh RGB và sinh một câu mô tả tiếng Việt. Hai thí nghiệm chính dùng chung encoder CNN để so sánh LSTM có attention với Transformer decoder bằng BLEU và CIDEr.

Phạm vi mở rộng gồm hai sản phẩm. Linh Khánh thử encoder ảnh CLIP với cùng Transformer để đo tác động của việc thay encoder. Vanh xây chức năng tìm ảnh bằng câu tiếng Việt qua các caption mô hình tự sinh. Linh Khánh tích hợp giao diện và chấm Recall@1/5/10. Khi bàn giao, người dùng có thể chọn ảnh để sinh câu hoặc nhập câu để tìm 5 ảnh phù hợp.

## Bắt đầu làm việc

Bộ chuẩn bị dữ liệu KTVIC đã chạy được từ gốc repo theo hướng dẫn bên dưới. Bản thiết kế nằm ở `docs/DRAFT.md` trên máy khởi tạo. Vanh chia sẻ bản này qua thư mục riêng của nhóm để Linh Khánh đọc được sau khi clone. Hai bạn dùng Notion cho đầu việc và dùng cùng một nơi lưu riêng cho tài liệu, log và checkpoint. Ghi đường dẫn chia sẻ vào Notion và kiểm tra quyền truy cập từ tài khoản của bạn còn lại. Mời giảng viên vào repo ngay giai đoạn đầu và cấp quyền xem log theo đề môn.

| Thư mục | Đặt gì ở đây | Lưu trên Git |
|---|---|---|
| `src/data/` | Đọc annotation, chia tập, tokenizer và tạo batch | Mã Python |
| `src/models/` | Encoder ảnh, LSTM-attention, Transformer và biến thể ablation | Mã Python |
| `src/retrieval/` | Lập chỉ mục caption tự sinh, biến câu truy vấn thành vector và xếp hạng ảnh | Mã Python |
| `src/evaluation/` | Chuẩn hóa caption, tính metric, tổng hợp bảng | Mã Python |
| `scripts/` | Điểm chạy chuẩn bị dữ liệu, trích đặc trưng, train, sinh caption, đánh giá | Mã Python |
| `configs/` | Cấu hình thí nghiệm dùng chung, seed, phiên bản encoder và tham số train | YAML dùng chung |
| `tests/`, `demo/` | Kiểm tra dữ liệu/mask/metric và mã demo suy luận | Mã Python |
| `data/raw/`, `data/processed/` | Ảnh gốc, annotation, manifest, split và từ vựng | Chỉ lưu riêng |
| `artifacts/features/`, `artifacts/checkpoints/` | Đặc trưng ảnh và trọng số, kể cả checkpoint phục hồi | Chỉ lưu riêng |
| `runs/` | Mỗi lần chạy một thư mục chứa log, cấu hình thực dùng, predictions, chỉ mục tìm ảnh, bảng thứ hạng, metrics và hình phân tích | Chỉ lưu riêng |
| `docs/`, `reports/`, `slides/` | Đề bài, draft, tài liệu đọc, nguồn LaTeX, PDF, slide và tư liệu demo | Chỉ lưu riêng |

Git chỉ giữ file nên các thư mục mã trống có `.gitkeep`. Sau khi clone, tạo các thư mục riêng bằng PowerShell từ gốc repo.

```powershell
$dirs = 'data/raw','data/processed','artifacts/features','artifacts/checkpoints','runs','docs','reports','slides'
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
```

## Quy ước Git và bàn giao

- Chỉ đưa lên Git README, quy tắc Git, file môi trường và mã/cấu hình dùng chung theo bảng. Quy tắc mặc định bỏ qua file mới ngoài danh sách cho phép, kể cả ảnh, dataset, annotation, notebook, tài liệu, slide, báo cáo Markdown/PDF, file nén, log và checkpoint. Giữ cấu hình riêng ở `*.local.yaml`. Dùng biến môi trường hoặc Kaggle Secrets cho thông tin xác thực.
- Trước mỗi commit, chạy `git status --short`, thêm đúng file theo tên, rồi đọc `git diff --cached`. Kiểm tra một đường dẫn bằng `git check-ignore -v -- <đường-dẫn>`. Tránh `git add -f`. Quy tắc ignore áp dụng cho file chưa được theo dõi. Mỗi commit nên tương ứng một đầu việc có thể chạy hoặc kiểm tra độc lập.
- Khi bàn giao thí nghiệm, ghi ở task Notion mã commit, cấu hình, seed, phiên bản/split dữ liệu, phiên bản cache, lệnh chạy, phần cứng, thời gian và link riêng tới log/checkpoint. Hai bạn thử mở link bằng tài khoản bên nhận. Ghi người viết, người review và việc đã dùng AI vào nhật ký để tổng hợp phần đóng góp và khai báo AI trong báo cáo.

## Chuẩn bị KTVIC và tạo batch dùng chung (03, 05, 08)

Vanh phụ trách kiểm tra dữ liệu, chia tập và chuẩn bị đầu vào cho hai decoder. Linh Khánh dùng cùng manifest, từ vựng và hàm tạo batch. Lưu hai annotation và hai gói ảnh vào `data/raw/ktvic/`. Giải nén ngay tại đó để có cấu trúc sau. [Annotation ở phiên bản đã kiểm tra](https://github.com/pacman-ctm/ktvic/tree/0412501549d57987e8b1db8b39a97a9040872822) · [Hai gói ZIP ảnh do tác giả chia sẻ](https://drive.google.com/drive/folders/16e8cd3AKusPS1H-h55JModiINM0og-ke)

```text
data/raw/ktvic/
  train_data.json        test_data.json
  train-images.zip       public-test-images.zip
  train-images/          public-test-images/
```

Môi trường đã chạy trên CPU dùng Python 3.13.7, Pillow 12.2.0, PyYAML 6.0.3 và PyTorch 2.11.0. Từ gốc repo, chạy

```sh
python -m scripts.prepare_ktvic_dataset --seed 42
python -m unittest discover -s tests -v
```

Chương trình giải mã từng ảnh, kiểm tra tên file, ID và năm caption, rồi giữ nguyên test gốc. Trong 3.769 ảnh train gốc, 15 ID (75 caption) trùng hoặc gần trùng ảnh test. Sau khi loại những ID này, chương trình lấy khoảng 10% ảnh còn lại cho validation. Các bản của cùng một ảnh được chia cùng nhau theo SHA-256 của pixel RGB và danh sách nhóm đã đối chiếu bằng mắt. Mỗi dòng manifest chứa một ID ảnh, đường dẫn tương đối, hash pixel và đủ năm caption gốc. [Nhóm ảnh đã đối chiếu](configs/ktvic_duplicate_image_groups.yaml) · [Số liệu công bố, mục 3.2](https://arxiv.org/pdf/2401.08100)

| Tập | Ảnh | Caption |
|---|---:|---:|
| Train | 3.379 | 16.895 |
| Validation | 375 | 1.875 |
| Test gốc | 558 | 2.790 |

Các file tái tạo trong `data/processed/ktvic/` gồm `{train,val,test}_ids.json`, `{train,val,test}_manifest.jsonl`, `caption_vocabulary.json`, `dataset_audit.json`, `sample_caption_batch.pt` và bản đọc được `sample_caption_batch.json`. Audit chứa ID loại, lý do, số token lạ, hash nguồn và hash manifest. Bản kiểm tra ảnh trực quan và báo cáo bàn giao trên máy nằm trong `reports/ktvic/`.

Văn bản được đưa về Unicode NFC, chữ thường, thay dấu câu bằng khoảng trắng, gộp khoảng trắng rồi tách theo âm tiết. Từ vựng gồm 1.716 token, giữ mọi âm tiết xuất hiện ít nhất một lần trong train, với PAD=0, BOS=1, EOS=2, UNK=3. Hàm `decode` bỏ BOS/PAD và dừng ở EOS. Ví dụ dùng lại batch cho cả hai mô hình

```python
import json
from pathlib import Path
from src.data.vietnamese_caption_batch import CaptionVocabulary, make_caption_batch

outputs = Path("data/processed/ktvic")
vocab = CaptionVocabulary.load(outputs / "caption_vocabulary.json")
with (outputs / "train_manifest.jsonl").open(encoding="utf-8") as manifest:
    row = json.loads(next(manifest))
samples = [(row["image_id"], row["image_path"], text) for text in row["captions"]]
batch = make_caption_batch(samples, vocab, "data/raw/ktvic")
sentence = vocab.decode(batch["labels"][0])
```

Batch mẫu lưu trên máy gồm bốn ảnh RGB float32 trong [0, 1], kích thước `[4, 3, 224, 224]`. Đầu vào decoder, nhãn int64 và mask bool cùng kích thước `[4, 14]`. Mask bằng True ở token cần tính loss, kể cả EOS. Bước chuẩn hóa ảnh theo trọng số encoder do bộ trích đặc trưng thực hiện.

## Đặc trưng ResNet-50 và bộ chấm dùng chung (Linh Khánh)

Cài môi trường bằng `python -m pip install -r requirements.txt`. Bản kiểm tra dùng Python 3.13.7, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Pillow 12.2.0, PyYAML 6.0.3, NumPy 2.4.4 và pycocoevalcap 1.2. `requirements.txt` chốt phiên bản gói Python. Bản CUDA của torch/torchvision phụ thuộc nguồn wheel và phần cứng. Bộ chấm BLEU/CIDEr chạy trên CPU, không gọi Java hoặc tải tokenizer.

Từ gốc repo, sau khi đã chuẩn bị dữ liệu, chạy

```sh
python -m scripts.extract_region_features --device cuda --batch-size 32 --workers 4
python -m scripts.verify_region_features --device cuda --samples 25
python -m scripts.run_scorer_trial --split val
python -m scripts.run_scorer_trial --split test
python -m unittest discover -s tests -v
```

Máy chỉ có CPU dùng `--device cpu` ở hai lệnh đầu. Lần trích đầu tải trọng số torchvision `ResNet50_Weights.IMAGENET1K_V1`. Encoder luôn ở `eval()`, mọi trọng số đóng băng. Ảnh được EXIF transpose, chuyển RGB, resize bilinear về 224×224, rồi chuẩn hóa bằng mean/std ImageNet. Đầu ra conv1…layer4 được xếp từ `[2048, 7, 7]` thành `[49, 2048]`. Chỉ số vùng là `hàng * 7 + cột`, hàng/cột bắt đầu từ 0. Cache lưu đặc trưng trước phép chiếu và embedding của decoder.

Ba file `artifacts/features/ktvic/resnet50_7x7_{train,val,test}.pt` chứa lần lượt 3.379 / 375 / 558 tensor float16. Mỗi file gồm `features [N,49,2048]`, `image_ids [N]`, `image_paths`, `pixel_sha256` và `meta` ghi phiên bản thư viện, trọng số, chuẩn hóa và SHA-256 manifest. File hoàn chỉnh thay thế cache cũ sau khi ghi thành công. Cả hai decoder tra theo ID qua cùng API

```python
from src.data.region_features import RegionFeatureStore

store = RegionFeatureStore.load("artifacts/features/ktvic", "val")
image_ids = store.image_ids[:2].tolist()
features = store.get_batch(image_ids)  # float32 [2, 49, 2048], đúng thứ tự yêu cầu
origin = store.trace(image_ids[0])     # ID, đường dẫn ảnh và hash pixel
found_id, distance = store.nearest_id(features[0])
```

Lệnh verify đối chiếu số lượng, toàn bộ ID/thứ tự, đường dẫn, hash và metadata với split/manifest, kiểm tra NaN/Inf và ID trùng giữa các tập. Với mỗi tập, lệnh tính lại tensor của 25 ID ngẫu nhiên cộng ID đầu/cuối, tìm tensor gần nhất trong cache và đối chiếu ảnh gốc. Các ID có cùng pixel có thể cho cùng tensor, nên truy ngược được chấp nhận khi hash pixel trùng. Báo cáo lưu tại `runs/region_features_check/report.json`, exit 1 nếu có sai lệch. Cache và báo cáo được lưu riêng theo quy ước Git.

File dự đoán nhận JSON dạng `[{"image_id": 2, "caption": "một câu mô tả"}, ...]`, object `{"2": "một câu mô tả", ...}`, hoặc JSONL mỗi dòng một object có `image_id` và `caption`. Mỗi ID trong tập được chấm phải có đúng một câu dự đoán. Chạy chấm và lưu kết quả bằng

```sh
python -m scripts.score_captions --predictions runs/scorer_trial/val_predictions_dung_ID.json --split val --output runs/scorer_trial/val_scores.json
```

Bộ chấm lấy toàn bộ năm caption của từng ID trong manifest. Dự đoán và tham chiếu dùng cùng `normalize_caption` của pipeline dữ liệu, tách token theo âm tiết. BLEU-1…4 là điểm corpus của pycocoevalcap. Mục `CIDEr` dùng implementation có clipping và Gaussian length penalty (CIDEr-D) của gói này. Mọi điểm đều nhân 100, nên BLEU có thang 0–100 và CIDEr-D có thang 0–1.000. CIDEr có thể vượt 100. IDF được tính từ tham chiếu của chính tập đang chấm, vì vậy so sánh các run trên cùng tập và cùng cấu hình chấm. JSON kết quả ghi số ảnh, số câu tham chiếu, thang điểm và phiên bản thư viện.

Lệnh trial dùng caption tham chiếu đầu tiên làm dự đoán, rồi tráo caption theo chu trình để mọi ảnh nhận caption của ID khác. Hai trường hợp đều được chấm trên đủ ảnh và đủ năm tham chiếu. Đây là phép kiểm tra bộ chấm bằng caption con người, không đo chất lượng mô hình. Lệnh kiểm tra điểm đúng ID cao hơn điểm tráo ID ở cả năm metric, kiểm tra lỗi thiếu dự đoán/thiếu file ảnh, ghi `runs/scorer_trial/{split}_trial_report.json` và exit 1 nếu điều kiện thất bại.

Lượt xác minh ngày 11/11/2026 trên RTX 3060 khớp đủ 4.312 ID ở ba tập và truy ngược đúng 81/81 mẫu tensor. Cả 17 kiểm thử đều qua, gồm kiểm thử manifest bị đổi ID/hash và lỗi đọc dự đoán. Kết quả trial trên dữ liệu đầy đủ, theo thang nhân 100 ở trên

| Tập | BLEU-4 đúng ID | BLEU-4 tráo ID | CIDEr đúng ID | CIDEr tráo ID |
|---|---:|---:|---:|---:|
| Validation (375 ảnh, 1.875 tham chiếu) | 100,00 | 8,18 | 348,87 | 10,75 |
| Test (558 ảnh, 2.790 tham chiếu) | 100,00 | 8,59 | 353,21 | 12,02 |

Khi đầu vào thiếu hoặc sai, lệnh chấm exit 2 và chỉ ra vấn đề, ví dụ `Thiếu dự đoán cho 5/375 ảnh ... (ID: ...)` hoặc `Thiếu file ảnh ... cho 3 ID (ID: ...)`. ID lạ, nhiều dự đoán cho một ID, caption rỗng và ID trùng trong manifest cũng bị từ chối. Mặc định kiểm tra file ảnh tồn tại. Chấm ở máy chỉ có manifest/predictions có thể truyền `--no-image-check`. Các lệnh nhận `--processed-dir`, `--image-root` để dùng dữ liệu ngoài thư mục mặc định, còn extract/verify nhận đường dẫn cache qua `--output`/`--features`.

## Môi trường huấn luyện dự kiến

Dùng PyTorch trên Kaggle. Mỗi run nhắm một T4. Phiên bản cho pipeline dữ liệu, trích đặc trưng và chấm điểm nằm trong `requirements.txt`. Người phụ trách huấn luyện kiểm tra lại môi trường trên Kaggle khi bổ sung decoder. Trên Kaggle, ánh xạ đường dẫn input và thư mục ghi kết quả qua cấu hình. Trước khi kết thúc phiên, lưu checkpoint và log vào nơi nhóm có quyền truy cập.

## Điều kiện bàn giao phiên bản chạy được

| Khi hoàn thành | Nội dung người phụ trách phải bổ sung vào README này |
|---|---|
| Pipeline dữ liệu | Phiên bản dữ liệu, cây thư mục sau giải nén, lệnh tạo split/vocab/cache và số ảnh kỳ vọng mỗi tập |
| Huấn luyện | Lệnh cài từ môi trường sạch, lệnh train từng cấu hình và cách resume trên Kaggle |
| Đánh giá | Lệnh sinh một caption cho từng ID, tính BLEU/CIDEr với toàn bộ câu tham chiếu và tái tạo bảng kết quả từ predictions |
| Tìm ảnh | Lệnh lập chỉ mục từ caption tự sinh của từng ảnh, chạy truy vấn tiếng Việt và tính Recall@1/5/10 trên kho ảnh đã chốt |
| Demo và nộp | Lệnh mở giao diện có cả sinh caption lẫn tìm ảnh top-5, link tải riêng checkpoint/chỉ mục, link log cho giảng viên và mã commit tái tạo bản nộp |

Chỉ công bố lệnh đã chạy thành công. Người còn lại chạy theo README từ một checkout sạch và đối chiếu số ảnh, tên metric cùng số trong bảng trước khi chốt báo cáo.
