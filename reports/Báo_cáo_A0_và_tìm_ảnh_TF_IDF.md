# Báo cáo A0 và tìm ảnh TF-IDF

**Phụ trách** Vanh · **Ngày** 17/11/2026 · **Nhánh** `lstm-attention`.

## Đối chứng A0

A0 dùng `configs/lstm_attention_uniform.yaml`: vùng và embedding 512 chiều, hidden 1120, attention 512, dropout 0,1. Ở mỗi bước, 49 vùng đều nhận trọng số **1/49**; ngữ cảnh là trung bình các vùng đã chiếu. Các lớp giữ nguyên cấu trúc và khởi tạo như A. Nhánh tính điểm attention được giữ trong checkpoint để đối chiếu khởi tạo, còn đầu ra A0 dùng trọng số cố định.

Lượt A0 được gửi tới [Kaggle T4](https://www.kaggle.com/code/wann179/ktvic-a0-uniform-20261117), cùng cache, split, từ vựng, seed 42, batch 32, AdamW LR 0,0003, weight decay 0,01, tối đa 20 epoch, patience 4 theo CIDEr val, greedy tối đa 40 token. A0 được huấn luyện lại từ đầu. Mốc đối chiếu là A `512/1120/512`, checkpoint epoch 5, CIDEr val 76,96.

`scripts/compare_caption_models.py` kiểm tra cấu hình, thiết lập và chữ ký dữ liệu trước khi tính **A − A0** trên đủ 375 ảnh. Ví dụ được chọn từ các ảnh đổi caption, ưu tiên ba mức chênh CIDEr lớn nhất mỗi phía; đây là lựa chọn minh họa. `scripts/plot_caption_comparison.py` ghép ảnh gốc cùng hai caption và một câu tham chiếu.

Kết quả A0 đang chờ GPU Kaggle. Bảng chênh lệch và ảnh minh họa sẽ được điền từ output thực tế.

## Tìm ảnh trên CPU đã chạy được

Hàm nằm tại `src/retrieval/caption_search.py`. TF-IDF học từ **16.895 caption train của 3.379 ảnh**, mỗi caption là một document, từ vựng gồm **1.712 âm tiết** sau cùng bước chuẩn hóa văn bản của repo. IDF có làm trơn, TF dùng số lần xuất hiện, vector chuẩn hóa L2. Cả câu truy vấn và caption ứng viên dùng cùng bộ biến đổi.

Chỉ mục chứa đúng **375 caption do A mới sinh trên val**, gắn với ID và đường dẫn từ manifest. Từ vựng/IDF được giữ cố định sau khi fit train. Caption tham chiếu val chỉ phục vụ đối chiếu ID. Cùng score thì ID nhỏ hơn đứng trước.

```python
from src.retrieval.caption_search import CaptionSearchIndex

index = CaptionSearchIndex.load("artifacts/retrieval/ktvic_a_val.npz")
results = index.search("người phụ nữ đang đạp xe đạp", k=5,
                       image_root="data/raw/ktvic")
```

Mỗi kết quả có `image_id`, `image_path`, `caption` do A sinh và `score` cosine. API báo `ValueError` khi câu rỗng, toàn từ ngoài từ vựng train hoặc k không hợp lệ, để giao diện yêu cầu nhập lại.

```sh
python -m scripts.build_image_search_index --predictions runs/kaggle_matched_inspection/project/runs/a_baseline_matched/val_predictions.json
python -m scripts.search_images --query "người phụ nữ đang đạp xe đạp"
```

Ví dụ trên trả thứ tự ID **3474, 5944, 4364, 7355, 5738**. Hai ảnh đầu cùng caption “có một người phụ nữ đang đạp xe đạp ở trên đường”, score **0,929425**. Đã xác nhận đủ năm file ảnh tồn tại. Tìm ảnh dựa trên từ vựng của caption nên kết quả còn phụ thuộc việc A mô tả đúng ảnh; ví dụ top 5 có một caption nói đến xe máy.

## Bàn giao cho Linh Khánh

- Hàm và CLI nằm trong nhánh `lstm-attention`, các file code dùng tên tiếng Anh.
- Chỉ mục `artifacts/retrieval/ktvic_a_val.npz` khoảng **74 KB**, lưu cùng metadata và SHA-256 của file train/val/predictions.
- Gói `artifacts/handoff/tfidf_image_search_a.zip` chứa chỉ mục, hàm tìm kiếm, CLI, các module dùng chung, file môi trường, ví dụ và báo cáo. Giải nén ở thư mục làm việc riêng, dùng cùng môi trường Python của repo. Khi gọi API/CLI, đặt `image_root` trỏ đến thư mục ảnh KTVIC trên máy Linh Khánh.
- Ví dụ và số đo tại `runs/image_search/handoff_example.json`. Trung vị 100 truy vấn sau khi nạp chỉ mục là khoảng **0,26 ms** trên máy phát triển, chưa gồm thời gian nạp Python, chỉ mục hoặc mở ảnh.

**50/50 kiểm thử mã nguồn đạt**, gồm trọng số A0 bằng 1/49, khởi tạo chung trùng A, truyền gradient tới LSTM/phép chiếu, phép đối chiếu A−A0, TF-IDF chỉ fit train, ID thiếu/trùng, truy vấn rỗng/OOV, thứ tự khi hòa score và lưu/nạp chỉ mục. Chỉ mục dùng định dạng NumPy không yêu cầu pickle.
