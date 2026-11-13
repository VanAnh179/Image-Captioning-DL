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

LSTM có attention đã có đầu ra logits, câu tự sinh và trọng số 49 vùng. [Báo cáo triển khai và lệnh kiểm tra](reports/Báo_cáo_cài_đặt_LSTM_có_attention.md)
