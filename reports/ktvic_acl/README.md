# Bài báo ACL về chú thích và tìm ảnh trên KTVIC

Mở `paper.tex` làm file chính. Trên Overleaf, chọn XeLaTeX hoặc LuaLaTeX. Nếu biên dịch bằng TeX Live, chạy trong thư mục này với TeX Gyre TermesX đã cài.

```sh
latexmk -xelatex paper.tex
```

`acl.sty` và `acl_natbib.bst` lấy nguyên trạng từ [template ACL tại `d5adc82`](https://github.com/acl-org/acl-style-files/tree/d5adc823ff0f80f98c80405ca0ab66c68e684409). Các hình đã có sẵn để biên dịch. `authors.local.tex` trong gói bàn giao điền MSSV hai tác giả. Bản Git dùng tên tác giả khi file riêng này vắng mặt.

## Đọc chéo trước khi nộp

- Linh Khánh đọc chéo bản bài báo, tập trung vào công thức và mô tả B/C, cách tính thời gian và kết luận ở phần test. Ghi góp ý theo số mục hoặc hình để chỉnh đúng chỗ.
- Mục Nghiên cứu liên quan và `references.bib` là bản dự thảo. Hai thành viên đối chiếu với các tài liệu thực tế đã đọc và xác nhận khai báo AI theo phạm vi đã dùng của nhóm.
- Điền mã nhóm vào tên file nộp `NhomXX_DeTai05.pdf` theo mã của lớp.

## Đối chiếu số liệu và hình

| Nội dung | Tệp nguồn trong workspace |
| --- | --- |
| Điểm test, tham số, thời gian | `reports/test_results/test_table.json` (bốn checkpoint Kaggle T4, seed 42) |
| Recall và hạng ảnh | `runs/retrieval_test_eval/`, `runs/retrieval_test_eval_a0_c/` và bảng đối chiếu `reports/caption_retrieval_table/bang_captioning_va_tim_anh.json` |
| Bảng chấm lỗi | `reports/manual_caption_review/summary_all40.json` |
| Biểu đồ B/C | `reports/clip_transformer/nam_seed_B_C_kaggle.json` |
| Ảnh lỗi và trọng số attention | Manifest test KTVIC và `artifacts/handoff/Phieu_25_anh_va_attention_KTVIC/reports/caption_test_attention/ban_do_theo_tu.json` |

Khi có đủ tệp bàn giao, chạy `python reports/ktvic_acl/draw_figures.py` từ gốc repo để vẽ lại hình dữ liệu. Script dùng Matplotlib, NumPy và Pillow. Sơ đồ kiến trúc nằm trong `figures/architecture.tex`.
