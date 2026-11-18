# Báo cáo lập chỉ mục và tìm ảnh bằng TF-IDF

**Vanh · Cập nhật 18/11/2026 · Kiểm tra truy vấn trên nhánh `retrieval-query-check`.**

## Mục tiêu và cách làm

Nhận một câu tiếng Việt, xếp hạng và trả **tối đa 5 ảnh val có điểm dương** trên CPU dựa vào caption **do mô hình A sinh**.

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

**5/5 kiểm thử tìm ảnh đạt.** Đã thử thêm các trường hợp dưới đây bằng hàm Python và CLI với chỉ mục thật của 375 ảnh.

## Truy vấn mẫu cho giao diện

| Truy vấn | Kết quả đã kiểm tra |
|---|---|
| `một cái cổng thành` | ID **2607** đứng đầu (0,775150). Ảnh gốc có cổng thành. Dùng để thử hiển thị kết quả. |
| `người phụ nữ đang đạp xe đạp` | Top 5 lần lượt là **3474, 5944, 4364, 7355, 5738**. Viết hoa, thêm khoảng trắng, thêm dấu câu hoặc dùng Unicode dạng tách dấu vẫn ra cùng kết quả. |
| Truy vấn trên với `k=1` | Hai ID **3474** và **5944** bằng điểm (0,929425). ID **3474** đứng trước và được giữ khi chỉ lấy một ảnh. |
| Chuỗi rỗng, chỉ khoảng trắng hoặc `, .!!` | `empty_query`. Hiện “Hãy nhập một câu mô tả có nội dung.” |
| `zzqv xxyw nnnq` | `unknown_terms`. Hiện “Chưa nhận ra từ nào trong câu. Hãy thử mô tả lại.” |
| `zzqv xe đạp` | Từ lạ bị bỏ qua, kết quả giống `xe đạp`. |
| `bơi` | Từ có trong từ vựng train nhưng vắng trong caption đã lập chỉ mục. Hàm trả `[]`. Hiện “Chưa tìm thấy ảnh phù hợp.” |
| `cô gái đang đạp xe đạp` | ID **3474** đứng đầu (0,785992). Caption A nói “đang đạp xe”. |
| `cô gái đang tạo dáng bên xe đạp` | ID **7355** đứng đầu (0,837710). Caption A nói “tạo dáng chụp ảnh”. |

Hai câu cuối cùng đều nhắc xe đạp, nhưng đổi hành động làm thay đổi thứ tự. Khi mở ảnh gốc, ID 3474 cho thấy người đi bộ và xe trên phố trước chùa. ID 7355 cho thấy hai người ngồi xem máy ảnh. Các caption A đã tả sai chi tiết trong hai ảnh này. Câu “người phụ nữ đang đứng bên xe đạp” cũng đưa ID 3474 lên đầu trong lần thử, nên dùng các câu về xe đạp để kiểm tra hành vi xếp hạng.

## Cách dùng và bàn giao cho Linh Khánh

Hàm nằm ở `src/retrieval/caption_search.py`, CLI ở `scripts/search_images.py`. Mỗi kết quả trả `image_id`, `image_path`, `caption` và `score`.

```python
from src.retrieval.caption_search import CaptionSearchIndex

index = CaptionSearchIndex.load("artifacts/retrieval/ktvic_a_val.npz")
results = index.search("người phụ nữ đang đạp xe đạp", k=5,
                       image_root="data/raw/ktvic")
```

Giao diện bắt `SearchInputError` (lớp con của `ValueError`) và dùng `error.code` để chọn thông báo trong bảng. Kiểu truy vấn sai dùng `invalid_query`, tham số số lượng sai dùng `invalid_top_k`. Khi truy vấn lỗi hoặc trả `[]`, xóa kết quả cũ rồi hiện thông báo tương ứng. Khi có kết quả, giữ thứ tự hàm trả về. Hàm xếp theo cosine đầy đủ trước khi làm tròn để hiển thị. CLI báo lỗi nhập bằng tiếng Việt và trả mã 2. Truy vấn hợp lệ có thể trả `[]` với mã 0.

Chạy từ gốc repo sau khi cài `requirements.txt` và có chỉ mục:

```sh
python -m scripts.search_images --query "người phụ nữ đang đạp xe đạp"
python -m unittest tests.test_caption_search -v
```

- **Chỉ mục:** `artifacts/retrieval/ktvic_a_val.npz`.
- **Gói bàn giao** `artifacts/handoff/tfidf_image_search_a.zip` gồm chỉ mục, hàm, CLI, kiểm thử, môi trường, ví dụ và báo cáo này. Giải nén, cài môi trường và chạy lệnh mẫu. Đặt `--image-root` hoặc `image_root` trỏ đến thư mục ảnh KTVIC của Linh Khánh.
- **Log ví dụ và số đo:** `runs/image_search/handoff_example.json`. Môi trường kiểm tra: Python 3.13.7, NumPy 2.4.4, PyTorch 2.11.0, Pillow 12.2.0; phần tìm kiếm chạy trên CPU.
- Mã nguồn và báo cáo cập nhật trên nhánh `retrieval-query-check`. Chỉ mục và ZIP là file bàn giao riêng ngoài Git.

Khi cần dựng lại chỉ mục từ đúng file caption A:

```sh
python -m scripts.build_image_search_index --predictions runs/kaggle_matched_inspection/project/runs/a_baseline_matched/val_predictions.json
```

## Giới hạn

TF-IDF tính độ giống giữa truy vấn và caption A theo từ trùng, nên ảnh trả về phụ thuộc vào chất lượng caption. Hàm giữ dấu tiếng Việt và không tự phục hồi dấu cho câu gõ không dấu. Cosine là độ giống văn bản. Các truy vấn mẫu kiểm tra cách hàm xử lý đầu vào và xếp hạng. Recall@5 cần bộ truy vấn có nhãn để đo.
