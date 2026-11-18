# Báo cáo lập chỉ mục và tìm ảnh bằng TF-IDF

**Phụ trách:** Vanh · **Ngày:** 18/11/2026 · **Nhánh:** `lstm-attention` · **Trạng thái:** hoàn thành phần hàm tìm ảnh và gói bàn giao.

## Mục tiêu và cách làm

Nhận một câu tiếng Việt, xếp hạng và trả **5 ảnh val** trên CPU dựa vào caption **do mô hình A sinh**. Luồng xử lý:

**Caption train → học TF-IDF → vector hóa caption A trên val → so độ giống với câu truy vấn → top 5 ID ảnh.**

- Học từ vựng và IDF từ **16.895 caption train của 3.379 ảnh**; mỗi caption là một văn bản. Sau chuẩn hóa có **1.712 âm tiết** trong từ vựng TF-IDF.
- Chỉ mục gồm **375 caption A trên val**, mỗi caption gắn với ID và đường dẫn ảnh từ manifest. Nguồn caption: A `d_model=512`, `hidden_size=1120`, `attention_size=512`, checkpoint tốt nhất epoch 5, greedy tối đa 40 token.
- TF là số lần xuất hiện; IDF có làm trơn: `log((1 + số caption train)/(1 + số caption train chứa từ)) + 1`. Vector chuẩn hóa L2; xếp theo cosine giảm dần, hòa điểm thì ID nhỏ hơn đứng trước.
- Truy vấn và caption dùng cùng bước chuẩn hóa tiếng Việt. Từ vựng/IDF chỉ học từ train; tham chiếu val dùng để đối chiếu ID, không dùng để học hay làm caption ứng viên.

## Kết quả đã kiểm tra

Câu truy vấn: **“người phụ nữ đang đạp xe đạp”**.

| Hạng | ID ảnh | Caption do A sinh | Cosine |
|---|---|---|---|
| 1 | 3474 | có một người phụ nữ đang đạp xe đạp ở trên đường | 0,929425 |
| 2 | 5944 | có một người phụ nữ đang đạp xe đạp ở trên đường | 0,929425 |
| 3 | 4364 | có một chiếc xe đạp chở hoa đang di chuyển ở trên đường | 0,451311 |
| 4 | 7355 | có một cô gái đang tạo dáng chụp ảnh ở bên một chiếc xe đạp | 0,422568 |
| 5 | 5738 | có một người phụ nữ đang đứng ở bên một chiếc xe máy | 0,312492 |

Ngày 18/11 đã chạy lại truy vấn: thứ tự, caption và score khớp log; đủ năm file ảnh tồn tại. Hash SHA-256 của train manifest, val manifest và file dự đoán khớp metadata trong chỉ mục.

Chỉ mục nặng **74.010 byte, khoảng 74 KB**. Log đo ngày 17/11 ghi trung vị **0,26 ms/truy vấn** qua 100 lần sau khi nạp chỉ mục; chưa tính khởi động Python, nạp chỉ mục hay mở ảnh. Đây là thời gian trên máy phát triển, không phải cam kết cho mọi CPU.

**4/4 kiểm thử tìm ảnh đạt ngày 18/11:** chỉ fit train; xếp hạng và xử lý hòa điểm; báo lỗi đầu vào/ID không hợp lệ; lưu/nạp giữ nguyên kết quả, không cần pickle.

## Cách dùng và bàn giao cho Linh Khánh

Hàm chính: `src/retrieval/caption_search.py`; CLI: `scripts/search_images.py`. Mỗi kết quả trả `image_id`, `image_path`, `caption` và `score`.

```python
from src.retrieval.caption_search import CaptionSearchIndex

index = CaptionSearchIndex.load("artifacts/retrieval/ktvic_a_val.npz")
results = index.search("người phụ nữ đang đạp xe đạp", k=5,
                       image_root="data/raw/ktvic")
```

Chạy từ gốc repo sau khi cài `requirements.txt` và có chỉ mục:

```sh
python -m scripts.search_images --query "người phụ nữ đang đạp xe đạp"
python -m unittest tests.test_caption_search -v
```

- **Chỉ mục:** `artifacts/retrieval/ktvic_a_val.npz`.
- **Gói bàn giao:** `artifacts/handoff/tfidf_image_search_a.zip`, gồm chỉ mục, hàm, CLI, module dùng chung, `requirements.txt`, ví dụ và báo cáo này. Giải nén vào thư mục riêng, cài môi trường rồi chạy lệnh mẫu. Đặt `--image-root` hoặc `image_root` trỏ đến thư mục ảnh KTVIC trên máy Linh Khánh; gói không chứa ảnh gốc.
- **Log ví dụ và số đo:** `runs/image_search/handoff_example.json`. Môi trường kiểm tra: Python 3.13.7, NumPy 2.4.4, PyTorch 2.11.0, Pillow 12.2.0; phần tìm kiếm chạy trên CPU.
- Mã nguồn và báo cáo được push trên nhánh `lstm-attention`; chỉ mục và ZIP là artifact bàn giao riêng, được `.gitignore` loại khỏi Git.

Khi cần dựng lại chỉ mục từ đúng file caption A:

```sh
python -m scripts.build_image_search_index --predictions runs/kaggle_matched_inspection/project/runs/a_baseline_matched/val_predictions.json
```

## Giới hạn

TF-IDF tìm theo từ xuất hiện trong caption, chưa hiểu đầy đủ từ đồng nghĩa hay ý nghĩa câu. Kết quả phụ thuộc độ đúng của caption A: ví dụ hạng 5 nói đến xe máy dù truy vấn hỏi xe đạp. Cosine đo độ giống văn bản, không phải xác suất ảnh đúng. Chưa đo Recall@5 trên bộ truy vấn có nhãn; ví dụ trên xác nhận chức năng chạy được.

Hàm báo `ValueError` khi câu rỗng, toàn từ ngoài từ vựng hoặc `k` không hợp lệ.
