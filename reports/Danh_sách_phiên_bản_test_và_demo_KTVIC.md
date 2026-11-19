# Danh sách phiên bản chạy test và demo KTVIC

Vanh và Linh Khánh dùng danh sách này cho bảng A/B/A0/C và demo. Mã chạy được ghim tại commit `a6575d5864fae9274fbfb6aacf9c53eaf112cb56`. Cả bốn checkpoint đã được xác minh SHA-256, nạp lại và sinh khớp 375/375 caption val mỗi mô hình.

## Checkpoint chọn trên val

Tất cả dùng seed 42 và chọn epoch có CIDEr val cao nhất trong lượt tương ứng. A, B và A0 dùng bộ checkpoint Kaggle T4 đã bàn giao cho phiếu chấm lỗi. Chọn C RTX 2050 vì đã có file checkpoint để xác minh và sinh lại toàn bộ val. Bảng test phản ánh các checkpoint đã ghim. Khi phân tích tác động đổi encoder, dùng kết quả B/C ghép theo seed trên cùng môi trường ở val. [Đối chiếu phiếu chấm lỗi](https://github.com/VanAnh179/Image-Captioning-DL/blob/63ea9c6/reports/%C4%90%E1%BB%81_xu%E1%BA%A5t_phi%E1%BA%BFu_ch%E1%BA%A5m_l%E1%BB%97i_caption.md) · [Thực nghiệm B/C](Báo_cáo_train_CLIP_Transformer_C.md)

| Mô hình | Epoch | CIDEr val | Đường dẫn `best.pt` tính từ gốc repo |
|---|---:|---:|---|
| A · ResNet-50 + LSTM attention, 512/1120/512 | 5 | 76,96 | `runs/kaggle_matched_inspection/project/runs/a_baseline_matched/best.pt` |
| B · ResNet-50 + Transformer | 6 | 72,42 | `artifacts/checkpoints/ktvic/B_kaggle_seed42/best.pt` |
| A0 · A với attention đều 1/49 | 6 | 77,95 | `runs/a0_kaggle/project/runs/a0_uniform/best.pt` |
| C · CLIP ViT-B/32 + Transformer, RTX 2050 | 5 | 79,12 | `artifacts/checkpoints/ktvic/C_rtx2050_seed42/best.pt` |

SHA-256 của checkpoint đã kiểm bằng `Get-FileHash`.

```text
A   a80ea2a8b5bb3687952f3a638e036614d67ac9dd28ad447d7c31c478a38dcc63
B   90598b48a2ef6b04913eb366d47e9d17ef8635c2977002645c29ad345ce174fd
A0  c0a10c18c4ada5a674ea8d3919315c4ed2431eb63283d97408e053068fae0ab7
C   a26189adea07f5dab7a4163f4ef0867e2db326612f9a2473527f1a9abc227f3a
```

File val sinh lại của C có SHA-256 `1965c30d2d9a3e5ac7b495c3d2fc1e8afcfa427356aa1cf45c35157ec3116e4e`, khớp file bàn giao. [Dự đoán C trên val](clip_transformer/c_seed42_val_predictions.json) · [Metric C](clip_transformer/c_seed42_metrics.json)

## Split, sinh câu và chấm

Dùng gói `artifacts/handoff/ktvic_processed_d796bc1.zip` hoặc chuẩn bị lại bằng `python -m scripts.prepare_ktvic_dataset --seed 42`. Giữ quy tắc loại 15 ID train trùng hoặc gần trùng test và giữ nguyên test gốc. Dùng các file dưới `data/processed/ktvic/`. Manifest ghi cả thứ tự ID, ảnh và năm câu tham chiếu. [Quy trình dữ liệu](../docs/ktvic.md)

| File | Ảnh / câu | SHA-256 |
|---|---:|---|
| `train_manifest.jsonl` | 3.379 / 16.895 | `cf9835ca90eb573c7e472e3155ee1c0a4955f029b59fd2c1d1f66c7740f0576a` |
| `val_manifest.jsonl` | 375 / 1.875 | `ace6e16f54c5bf81b9a947d0c1968143ec86d1876a58f9e9e4e3f7d724730420` |
| `test_manifest.jsonl` | 558 / 2.790 | `d9c2e3b1af114155eb65eb6336a1f79bce8efe080c75bc57c70c7068c547e2d3` |
| `test_ids.json` | 558 ID | `514e6930ec10bee1a223444593670f07ab6f2374419b160b38e044c465ec9513` |
| `caption_vocabulary.json` | 1.716 token | `f548268127394cbc174a5c775965c593df824db2d65b97a27e42a9f555b7f799` |

- **Đặc trưng.** A/B/A0 dùng cùng `artifacts/features/ktvic/resnet50_7x7_test.pt` (49 × 2.048, float16), SHA-256 `cae5c0f6215cff5bfa2440907c83d81fb2f2a223af58aa108f5b815bd968cf9b`. C dùng `clip_vit_b32_7x7_test.pt` (49 × 768, float16). Khi cần tạo cache C, chạy `python -m scripts.extract_clip_features --splits test --batch-size 32 --device cuda`. Revision encoder CLIP đã ghim trong mã.
- **Sinh.** Giải mã greedy, batch 32, bắt đầu BOS, bỏ PAD/BOS khỏi lựa chọn, dừng ở EOS hoặc tối đa 40 token. Lệnh đọc cấu hình, trọng số và từ vựng của checkpoint, xuất một caption mỗi ID theo thứ tự manifest. Dùng cùng bản cache và môi trường cho cả hai người. [Phiên bản thư viện](../requirements.txt)
- **Chấm.** Chấm BLEU-1…4 và CIDEr-D nhân 100 với đủ năm câu tham chiếu của mỗi ID. Dùng `pycocoevalcap==1.2`, `numpy==2.4.4`. Chấm trên cả 558 ID test, cùng manifest cho cả bốn mô hình.

Chạy hai lệnh sau cho mỗi hàng checkpoint, thay tên mô hình và đường dẫn tương ứng. Với C thêm `--feature-prefix clip_vit_b32_7x7` vào lệnh sinh.

```sh
python -m scripts.generate_caption_predictions --split test --checkpoint "<best.pt>" --device cuda --output "runs/experiment_version_lock/<model>_test_predictions.json"
python -m scripts.score_captions --split test --predictions "runs/experiment_version_lock/<model>_test_predictions.json" --output "runs/experiment_version_lock/<model>_test_scores.json"
```

## Chỉ mục đã chọn cho demo

Chọn **A Kaggle epoch 5** vì demo trả tối đa 5 ảnh. Trên cùng 375 ảnh val và 1.875 truy vấn tham chiếu, dùng đúng bộ xếp hạng demo, A đạt **312/1.875 (16,64%)** ở R@5, B đạt **290/1.875 (15,47%)**. R@1/R@10 lần lượt là 5,23%/23,20% cho A và 4,37%/23,89% cho B. Khoảng tin cậy 95% của chênh lệch R@5 chứa 0 (−1,76 đến +4,27 điểm phần trăm).

- Dùng `artifacts/retrieval/ktvic_a_val.npz` từ `artifacts/handoff/tfidf_image_search_a.zip`. Hash của chỉ mục đã có và bản dựng lại đều là `0ea7de4c6ff7b366439ea1df0fe8756e8b331cc4a40fe18a16447d0734d02ad8`. Nguồn caption là `val_predictions.json` trong thư mục checkpoint A ở trên, SHA-256 `ea3591eee423f053bc7fd5f6b5b6475b26e5d239eb9dd35d1f2f2e838c6eae5a`.
- TF-IDF học IDF từ 16.895 câu train, kho tìm kiếm chứa 375 caption tự sinh trên val. Chuẩn hóa L2, xếp theo cosine giảm dần rồi ID tăng dần khi hòa điểm. Demo chỉ trả kết quả có điểm dương.

```sh
python -m scripts.build_image_search_index --predictions runs/kaggle_matched_inspection/project/runs/a_baseline_matched/val_predictions.json
python -m scripts.search_images --query "có nhiều xe máy đang dựng bên lề đường" --top-k 5
```

## Kiểm tra bàn giao

- Dùng Python 3.13.7, PyTorch 2.11.0+cu128 trên RTX 3060 để sinh lại A/B/A0/C. Điểm BLEU và CIDEr val của cả bốn khớp kết quả lưu. Cả 72 kiểm thử đạt. Lượt rà soát này dùng val để đo chất lượng.
- Lượt Recall RTX 2050 ban đầu đã được chạy lại đủ A và B với file A đúng SHA-256. Sửa bước đối chiếu của lệnh Recall để chấp nhận ảnh điểm 0 nằm sau các kết quả demo. Hai CSV hạng và JSON tổng hợp khớp từng byte với lượt rà soát trước. Chỉ mục demo vẫn dùng A Kaggle.
