# Chú thích ảnh tiếng Việt

Đồ án Học sâu của Vanh và Linh Khánh so sánh LSTM có attention với Transformer decoder trên bộ ảnh KTVIC. Hai mô hình dùng chung đặc trưng từ ResNet-50 đóng băng. Repo hiện có các lệnh chuẩn bị dữ liệu, trích đặc trưng và chấm caption bằng BLEU và CIDEr.

## Chạy từ gốc repo

Đặt `train_data.json` và `test_data.json` vào `data/raw/ktvic/`, rồi giải nén hai gói ảnh vào đó để có `train-images/` và `public-test-images/`. [Tải annotation KTVIC](https://github.com/pacman-ctm/ktvic/tree/0412501549d57987e8b1db8b39a97a9040872822) · [Tải hai gói ảnh](https://drive.google.com/drive/folders/16e8cd3AKusPS1H-h55JModiINM0og-ke)

```sh
python -m pip install -r requirements.txt
python -m scripts.prepare_ktvic_dataset --seed 42
python -m scripts.extract_region_features --device cuda --batch-size 32 --workers 4
python -m scripts.verify_region_features --device cuda --samples 25
python -m scripts.run_scorer_trial --split val
```

Đã chạy với Python 3.13.7. Nếu dùng CPU, đổi `--device cuda` thành `--device cpu`. Cache của ba tập nằm trong `artifacts/features/ktvic/`.

Để chấm caption của mô hình, xem định dạng file dự đoán và lệnh chấm trong [hướng dẫn KTVIC](docs/ktvic.md#chấm-caption).

## Demo offline

Chọn ảnh để sinh caption, hoặc gõ câu tiếng Việt để xem tối đa 5 ảnh khớp nhất và caption dưới từng ảnh. Demo web chạy tại `http://127.0.0.1:8000` và dùng môi trường đã cài `requirements.txt`.

```sh
python -m scripts.run_web_demo
```

Đặt checkpoint A Kaggle epoch 5 ở `runs/kaggle_matched_inspection/project/runs/a_baseline_matched/best.pt` hoặc `best.pt` tại gốc repo. Đặt chỉ mục tương ứng ở `artifacts/retrieval/ktvic_a_val.npz` và ảnh KTVIC dưới `data/raw/ktvic/`. Máy cần có sẵn file `resnet50-0676ba61.pth` trong thư mục `checkpoints` của cache Torch Hub. Demo nạp trọng số từ file local.

Có thể chỉ định vị trí khác bằng `--checkpoint`, `--index` và `--image-root`. Dùng `--device cpu` để chạy trên CPU. Dùng `--no-browser` để tự mở địa chỉ được in trong terminal, `--port` để đổi cổng hoặc `--window` để mở cửa sổ riêng bằng Edge hay Chrome. Bản Tkinter chạy bằng `python -m scripts.run_demo`.

### Tìm theo đặc trưng ảnh

Trên bản có chế độ này, chuẩn bị cache ResNet-50 của train và val. MiniLM phải có sẵn trong cache Hugging Face trên máy chạy demo. Để chuẩn bị cache một lần trên máy có mạng, dùng revision đã ghim.

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42', allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])"
```

Hai lệnh sau chạy offline. Bước tạo chỉ mục học phép chiếu từ ảnh và câu tham chiếu train, rồi lưu vector ảnh val vào `artifacts/retrieval/visual/ktvic_val.pt`.

```sh
python -m scripts.build_visual_search_index
python -m scripts.run_web_demo --visual-index artifacts/retrieval/visual/ktvic_val.pt
```

Chế độ này xếp ảnh theo độ tương đồng giữa câu truy vấn và đặc trưng ảnh. Chỉ mục caption A cung cấp caption và đường dẫn cho từng thẻ. Chạy lệnh demo cơ bản để dùng xếp hạng TF-IDF.

Báo cáo nội bộ, log huấn luyện và kết quả đánh giá chi tiết được bàn giao riêng qua Media board.
