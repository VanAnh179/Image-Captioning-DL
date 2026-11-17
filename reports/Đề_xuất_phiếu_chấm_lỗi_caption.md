# Đề xuất phiếu chấm lỗi caption

**Vanh · 17/11/2026 · Trạng thái: chờ Vanh và Linh Khánh nhận xét/chốt.**

Mục tiêu: so sánh lỗi của A, B và A0 trên cùng ảnh. Dùng **val để tập chấm và sửa tiêu chí**; sau khi thống nhất mới áp dụng lên 40 ảnh test. Đây là đề xuất quy trình, chưa phải kết quả chấm của hai người.

## 1. Các quyết định cần chốt

Mỗi người điền “Đồng ý” hoặc ghi phương án thay thế vào hai cột cuối. Các mục dưới đây đã có mặc định để có thể chốt trực tiếp.

| Nội dung | Phương án đề xuất | Vanh | Linh Khánh |
|---|---|---|---|
| Phạm vi | A, B, A0; mỗi mô hình một caption/ảnh, cùng 40 ảnh test | Chờ chốt | Chờ chốt |
| Tiêu chí | Sáu cờ lỗi ở mục 2; một caption có thể có nhiều loại lỗi | Chờ chốt | Chờ chốt |
| Tập chấm thử | 6 ảnh val: `2, 2789, 4770, 4788, 6000, 9826` | Chờ chốt | Chờ chốt |
| Chọn và chia ảnh test | Seed 42 chọn 40 ID; seed 43 chia việc, mỗi người 25 ảnh, trùng 10 | Chờ chốt | Chờ chốt |
| Ẩn mô hình | Trộn riêng từng ảnh; chỉ mở bảng đối chiếu sau khi khóa phiếu | Chờ chốt | Chờ chốt |
| Chốt nhãn | Chấm độc lập → đối chiếu phần trùng → thảo luận bất đồng | Chờ chốt | Chờ chốt |

Chỉ mở đợt chấm test khi đã đủ output và khóa checkpoint/cách sinh câu của cả ba mô hình trên val. Nếu bỏ A0 hoặc thêm C, sửa phạm vi ở đây trước khi chấm test; giữ nguyên mẫu 40 ảnh.

## 2. Sáu loại lỗi và ví dụ

**Căn cứ chính là ảnh gốc.** Năm câu tham chiếu giúp đối chiếu, không phải danh sách bắt buộc phải kể đủ. Câu ngắn, dùng từ đồng nghĩa hoặc bỏ bớt chi tiết đúng không tự động là lỗi. Không suy đoán tuổi, nghề nghiệp, ý định hoặc hành động khó thấy từ một ảnh tĩnh.

![Ba ảnh val dùng minh họa; ID từ trái sang phải: 2, 2789, 4788](assets/caption_error_val_examples.jpg)

Các câu sai dưới đây **được tự đặt để minh họa**, không phải caption hay lỗi đã đo của mô hình. Ảnh `2` có ba thuyền trên sông; `2789` có người đàn ông đội mũ lưỡi trai đứng giữa ảnh, mặc áo sọc xám; `4788` có hai lá cờ chính phía trên vòm cổng.

| Mã / loại lỗi | Đánh dấu khi | Ví dụ caption sai → căn cứ sửa |
|---|---|---|
| **VT — Vật thể** | Gọi sai hoặc thêm người, con vật, đồ vật không có trong ảnh | Val `2`: “Trong ảnh có ba chiếc **ô tô**” → ba chiếc **thuyền** |
| **HD — Hành động** | Nói sai việc hoặc tư thế của chủ thể đã xác định | Val `2789`: “Người đàn ông đội mũ lưỡi trai ở giữa ảnh đang **ngồi**” → đang **đứng** |
| **QH — Quan hệ** | Sai vị trí tương đối, chủ thể–đối tượng hoặc quan hệ sở hữu có thể thấy rõ | Val `4788`: “Hai lá cờ chính ở **dưới** vòm cổng” → ở **phía trên** |
| **SL — Số lượng** | Khẳng định số đếm sai của một nhóm xác định | Val `2`: “Trên sông có **tổng cộng hai** chiếc thuyền” → thấy **ba** chiếc |
| **BD — Chi tiết bịa** | Thêm thuộc tính/bối cảnh sai hoặc không có căn cứ nhìn thấy, ngoài bốn loại trên | Val `2789`: “Người đàn ông đội mũ lưỡi trai ở giữa ảnh mặc **áo đỏ**” → áo **sọc xám** |
| **LT — Lặp từ** | Lặp từ/cụm/mệnh đề thừa, không bổ sung ý | Val `2`: “Có ba chiếc thuyền, **có ba chiếc thuyền** ở trên sông” → bỏ cụm lặp |

**Tránh tính trùng:** cùng một ý sai chỉ gán loại cụ thể nhất: ô tô thay thuyền → VT; hai thay ba → SL; áo đỏ → BD. Không cộng thêm BD cho chính lỗi VT/HD/QH/SL đó. Nếu có hai ý sai khác nhau, có thể đánh dấu hai loại. Mỗi loại chỉ tính một lần/caption dù xuất hiện nhiều chỗ.

“Một người” có thể chỉ một người trong nhóm; chỉ chấm SL khi câu xác định số lượng của nhóm, như “tổng cộng/chỉ có…”. Lặp từ có vai trò ngữ pháp bình thường, như “một người cầm một chiếc ô”, không phải LT. Trường hợp bị che khuất hoặc không đủ căn cứ dùng `?`, kèm lý do.

## 3. Phiếu chấm dùng trực tiếp

**Một dòng = một caption của một ảnh, do một người chấm.** Phiếu hiện ảnh đủ lớn và nguyên văn caption; mặc định chỉ xem ảnh + caption, có thể mở toàn bộ năm tham chiếu khi cần. Không sửa caption trước khi chấm.

| Người chấm | image_id | item_id ẩn danh | Caption | VT | HD | QH | SL | BD | LT | Cụm sai → bằng chứng/sửa đề xuất |
|---|---|---|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … | … | … | … |

Điền `0` = không có lỗi loại đó; `1` = có; `?` = chưa rõ. Không để trống khi nộp. Nếu có `1` hoặc `?`, bắt buộc ghi cụm liên quan và căn cứ trong ảnh. Sáu số `0` nghĩa là **không phát hiện sáu lỗi này**, không có nghĩa caption mô tả đầy đủ hay hay nhất.

**Buổi chấm thử val:** hai người cùng dùng sáu ID ở mục 1 và caption của các mô hình hiện có, nhưng điền phiếu độc lập. Đối chiếu từng ô khác nhau, ghi một dòng “trường hợp → quy tắc chốt”, rồi sửa mục 2 và ghi phiên bản phiếu. Nếu còn bất đồng về quy tắc, thử thêm ảnh val trước khi chốt. Không cộng lượt thử này vào kết quả test.

## 4. Chọn 40 ảnh test bằng seed

Chọn theo **ID ảnh**, không chọn theo caption hay điểm mô hình; không thay ảnh vì khó, giống nhau hoặc mô hình sinh câu kém. Dùng danh sách test hiện có, chuyển ID về số nguyên, sắp tăng dần rồi lấy mẫu không hoàn lại. Đoạn mã sau chạy từ gốc repo sau khi hai người chốt:

```python
import hashlib, json, platform, random
from pathlib import Path

source = Path("data/processed/ktvic/test_ids.json")
raw = source.read_bytes()
ids = sorted(map(int, json.loads(raw)))
assert len(ids) == len(set(ids)) and len(ids) >= 40
selected = sorted(random.Random(42).sample(ids, 40))
order = random.Random(43).sample(selected, 40)
overlap = order[:10]
record = {
    "selection_seed": 42, "assignment_seed": 43,
    "python_version": platform.python_version(),
    "test_ids_sha256": hashlib.sha256(raw).hexdigest(),
    "selected_ids": selected, "overlap_ids": overlap,
    "vanh_ids": overlap + order[10:25],
    "linh_khanh_ids": overlap + order[25:40],
}
out = Path("artifacts/manual_caption_review/selection.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
```

Lưu file này ngay khi chốt mẫu và trước khi xem caption test; dùng lại chính danh sách ID đã lưu thay vì bốc lại trên máy khác. Mỗi người chấm đủ ba caption/ảnh: **25 ảnh × 3 = 75 dòng/người**. Có 10 ảnh trùng, tổng cộng 40 ảnh riêng biệt, không phải 50. Không dùng kết quả test để chọn lại checkpoint hay sửa mô hình.

## 5. Ẩn tên mô hình và khóa kết quả

1. Chuẩn bị gói chấm từ cùng 40 ID và ba file prediction đã khóa. Kiểm tra mỗi file có đúng một caption cho từng ID; thiếu, trùng hoặc rỗng thì sửa gói trước khi chấm. Caption giống hệt nhau vẫn giữ thành các mục riêng.
2. Với **từng ảnh**, dùng bộ sinh ngẫu nhiên hệ thống để trộn ba caption độc lập và cấp `item_id` vô nghĩa. Không dùng một nhãn cố định như X = A cho mọi ảnh. Hai người nhận cùng `item_id` và thứ tự caption trên 10 ảnh trùng.
3. Phiếu chỉ chứa ảnh, `image_id`, `item_id`, caption và cột chấm. Bỏ tên mô hình, tên file/run, checkpoint, attention map và điểm BLEU/CIDEr. Không có cột bí mật hoặc cột bị ẩn chứa tên mô hình trong cùng phiếu.
4. Bảng riêng `blind_key.json` giữ `item_id → model, image_id`, hash các file prediction, hash checkpoint và phiên bản phiếu. Để ngoài gói chấm, ngoài Git và nơi chia sẻ phiếu; cả hai thống nhất không mở bảng này khi chấm. Đây là cách giảm thiên kiến; người từng đọc caption có thể vẫn đoán được nguồn.
5. Mỗi người nộp bản phiếu độc lập; lưu nguyên bản trước khi trao đổi. Đối chiếu và chốt bất đồng khi vẫn đang ẩn tên. Sau đó khóa bản kết quả, mới mở bảng đối chiếu để tổng hợp theo A/B/A0.

## 6. Cách tổng hợp và bàn giao

- **Theo mô hình:** báo từng loại lỗi dưới dạng `số caption lỗi / số caption đã chốt` và %. Báo thêm tỷ lệ caption có ít nhất một lỗi. Nhiều loại có thể cùng xuất hiện nên tổng tỷ lệ không cần bằng 100%. Một mô hình có tối đa 40 caption riêng biệt; phần chấm trùng chỉ tính một lần sau thống nhất.
- **Mức thống nhất ban đầu:** trên 10 ảnh trùng × 3 mô hình × 6 loại = **180 ô**; báo số ô hai người điền giống nhau trên số ô cả hai đã điền `0/1`, kèm số ô có `?` và các loại hay bất đồng. Tính từ hai phiếu gốc, trước khi thảo luận.
- Nếu sau thảo luận vẫn còn `?`, giữ nguyên và báo riêng số chưa rõ; không đổi thành `0`. Mỗi tỷ lệ ghi rõ mẫu số đã chốt của loại đó. Tỷ lệ “ít nhất một lỗi” dùng caption có ít nhất một `1` hoặc cả sáu ô đều `0`.
- **Bốn ví dụ cuối:** sau khi chấm xong, lấy ID nhỏ nhất chưa dùng trong từng nhóm: A có lỗi/B không; B có lỗi/A không; A0 có lỗi/A không; cả A và B có lỗi. Chỉ dùng nhãn đã rõ. Nếu nhóm không còn ID phù hợp, ghi rõ rồi bổ sung ID nhỏ nhất có lỗi chưa dùng; nếu vẫn thiếu thì dùng ID nhỏ nhất còn lại. Mỗi ví dụ có ảnh, ID, caption, cả năm tham chiếu và lời giải thích. Đây là ví dụ trong mẫu 40, không đại diện toàn tập test.
- **Gói bàn giao sau duyệt:** phiên bản phiếu đã chốt, `selection.json`, hai phiếu gốc, phiếu sau thống nhất, bảng tổng hợp và bốn ví dụ; giữ khóa ẩn danh riêng cho đến lúc khóa nhãn. Đây là phân tích mô tả trên 40 ảnh, không đủ để kết luận mô hình vượt trội trên mọi ảnh.

**Đã kiểm tra khi soạn:** val có 375 ID duy nhất, test có 558 ID duy nhất; sáu ID tập chấm thử đều thuộc val. Đã mở ba ảnh minh họa và đối chiếu annotation. Chạy thử đoạn mã trong thư mục tạm: lặp lại ra cùng danh sách, đủ 40 ID, chia 25/25 và trùng đúng 10. Chưa có nhãn chấm thử của Vanh/Linh Khánh; chưa chốt mẫu test, chưa tạo gói ẩn danh và chưa có tỷ lệ lỗi thực nghiệm.
