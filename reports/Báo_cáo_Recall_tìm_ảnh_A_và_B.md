# Báo cáo Recall tìm ảnh bằng caption tự sinh của A và B

Nhánh `retrieval-recall-eval`. Đo trên val bằng chỉ mục TF-IDF có sẵn (`CaptionSearchIndex`), không đổi cách xếp hạng.

## 1. Thiết lập

- **Kho ứng viên:** toàn bộ 375 ảnh val.
- **Truy vấn:** từng câu tham chiếu của mỗi ảnh val, 375 × 5 = **1.875 truy vấn**. Đáp án là ID của ảnh gốc câu đó mô tả.
- **Hai chỉ mục:** mỗi chỉ mục lập từ caption tự sinh của một mô hình trên 375 ảnh val. IDF học từ caption train (16.895 câu, từ vựng 1.712 âm tiết), vector L2-normalised, xếp hạng theo cosine giảm dần, hòa điểm thì ID ảnh tăng dần, đúng như hàm `search`.
- **Nguồn caption:** A = `runs/train30/A` (CIDEr val 81,39, epoch 5) và B = `runs/train30/B` (CIDEr val 70,47, epoch 7). Cùng máy RTX 2050, seed 42, cùng cache ResNet-50 và cùng cách sinh greedy tối đa 40 token.
- **Chạy lại:** `python -m scripts.evaluate_caption_retrieval --predictions-a runs/train30/A/val_predictions.json --predictions-b runs/train30/B/val_predictions.json`. Có thể chỉ định `--output-dir` để lưu lượt chạy mới ở thư mục riêng.

## 2. Kết quả

| | R@1 | R@5 | R@10 | Hạng trung vị |
|---|---:|---:|---:|---:|
| A | 4,69% (88/1875) | 16,48% (309) | 24,32% (456) | 48 |
| B | 4,37% (82/1875) | 14,29% (268) | 21,39% (401) | 64 |
| Ngẫu nhiên (k/375) | 0,27% | 1,33% | 2,67% | |

Cả hai cao hơn ngẫu nhiên khoảng 16 đến 18 lần ở R@1, 11 đến 12 lần ở R@5 và 8 đến 9 lần ở R@10.

**Hòa điểm.** Caption tự sinh hay trùng nhau (A có 285, B có 299 câu khác nhau trên 375 ảnh), nên có 702 truy vấn của A và 605 truy vấn của B có ảnh khác cùng điểm với ảnh đúng. Kết quả chính ở trên dùng quy tắc hòa điểm của `search` (ID nhỏ đứng trước). Nếu ảnh đúng được xếp trước hoặc sau mọi ảnh cùng điểm:

| Khoảng (tốt nhất / xấu nhất) | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| A | 4,21% – 6,13% | 15,63% – 17,55% | 23,25% – 24,91% |
| B | 3,89% – 5,60% | 13,17% – 15,68% | 20,16% – 22,45% |

**A so với B.** Chênh lệch A trừ B và khoảng tin cậy 95% khi lấy mẫu lại ảnh (2.000 lần, 5 truy vấn của cùng một ảnh giữ cùng nhau):

| | Chênh lệch | KTC 95% |
|---|---:|---|
| R@1 | +0,32 điểm | −1,33 đến +2,03 |
| R@5 | +2,19 điểm | −0,96 đến +5,55 |
| R@10 | +2,93 điểm | −0,75 đến +6,88 |

Trên 1.875 truy vấn, A có hạng tốt hơn B ở 970 truy vấn, B tốt hơn ở 865, bằng nhau ở 40.

## 3. Nhận xét

A nhỉnh hơn B ở cả ba ngưỡng, nhưng khoảng tin cậy đều chứa 0, nên với một lần train mỗi mô hình chưa đủ căn cứ nói A tìm ảnh tốt hơn B. Mức tuyệt đối thấp (khoảng 1/4 truy vấn có ảnh đúng trong 10 kết quả đầu): truy vấn là câu tham chiếu nhiều chi tiết, còn caption tự sinh ngắn, chung chung và trùng nhau, nên nhiều ảnh gần như không phân biệt được bằng từ.

## 4. File kết quả và kiểm tra

- `reports/retrieval_recall/ranks_A.csv`, `ranks_B.csv`: mỗi dòng một truy vấn (1.875 dòng) gồm ID ảnh đúng, số thứ tự câu tham chiếu, nội dung truy vấn, hạng ảnh đúng trong toàn bộ kho, hạng tốt nhất và xấu nhất khi hòa điểm, số ảnh cùng điểm, điểm của ảnh đúng, ID ảnh xếp hạng 1. Bảng đánh giá vẫn xếp các ảnh điểm 0. Demo chỉ hiển thị ảnh có điểm dương.
- `reports/retrieval_recall/recall_summary.json`: toàn bộ số ở bảng trên, hash file dự đoán (A `36f32680…`, B `81ad0993…`).
- Lệnh tạo hai chỉ mục `ktvic_a_val.npz` và `ktvic_b_val.npz` trong `artifacts/retrieval/recall_eval/` theo mặc định. Truy vấn có vector rỗng xuất hạng trống, được tính là không tìm thấy ảnh đúng và vẫn nằm trong tổng số truy vấn.
- **Lượt gốc.** Tác giả đối chiếu 3.750 hạng với `CaptionSearchIndex.search(k=375)`. Cả hai file CSV ghi 0 truy vấn có vector rỗng.
- **Lượt rà soát.** Đã dựng lại cả hai chỉ mục từ file dự đoán đúng SHA-256. Recall@1/5/10 của A và B đều khớp số công bố. Một số hạng hoặc biên hòa điểm thay đổi ở các điểm cosine sát nhau. [Chi tiết rà soát](Rà_soát_Recall_tìm_ảnh.md)

## 5. Giới hạn

- Kết quả áp dụng cho hai checkpoint RTX 2050, seed 42 đã nêu ở mục thiết lập.
- Mỗi mô hình một lần train, một seed. Chưa đo dao động giữa các seed của kết quả tìm ảnh, nên chưa biết chênh lệch A và B có lớn hơn dao động đó không.
- Chỉ mục chỉ có một caption cho mỗi ảnh. Truy vấn lấy từ chính tập tham chiếu dùng để chấm BLEU/CIDEr nên không phải truy vấn người dùng thật.
