"""Manual caption-error review on the 40 test images: parse sheets, compare two graders, tabulate by model.

Check the Linh Khanh side alone using the submitted Markdown sheet and saved predictions:
  python -m scripts.summarize_manual_caption_review --linh-sheet <sheet.md>
For all 40 images, use the original Vanh labels beside their items.json and the locked final labels:
  python -m scripts.summarize_manual_caption_review --linh-sheet <sheet.md> --vanh-labels <vanh.json> \
      --locked-labels <labels_test_locked.json> --example-ids 3091 297 756 66
Flat label lists and a separate --resolution file ({"item_id": {"QH": 1, "BD": 0}}) are also supported.
After labels are locked, --blind-key can provide exact item-to-model identities for identical captions.
Flags are 0 (no error found), 1 (error) or "?" (unclear, never turned into 0).
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data.region_features import read_manifest

FLAGS = ("VT", "HD", "QH", "SL", "BD", "LT")
GENERATORS = ("A", "B", "A0")  # model C was not part of the blind pack
ROW = re.compile(r"\| (it-[0-9a-f]+) \| (.*?) \| ([01?]) \| ([01?]) \| ([01?]) \| ([01?]) \| ([01?]) \| ([01?]) \| (.*) \|$")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vi(value, digits=2):
    """Vietnamese number style: comma for decimals, dot for thousands."""
    return f"{value:,.{digits}f}".translate({ord(","): ".", ord("."): ","})


def parse_sheet(path):
    """Rows of a submitted Markdown sheet, in the same shape as the JSON label files."""
    rows, image_id = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        heading = re.match(r"## \**Ảnh (\d+)\**$", line)
        if heading:
            image_id = int(heading.group(1))
            continue
        match = ROW.match(line)
        if match:
            flags = {name: (value if value == "?" else int(value)) for name, value in zip(FLAGS, match.groups()[2:8])}
            rows.append({"image_id": image_id, "item_id": match.group(1), "caption": match.group(2), **flags})
    return rows


def load_label_rows(path):
    """Read flat rows or locked annotations, joining captions from the original items when needed."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        return document
    rows = document["annotations"]
    captions = {}
    if any("caption" not in row for row in rows):
        items = json.loads(path.with_name("items.json").read_text(encoding="utf-8"))
        if items["batch"] != document["batch"]:
            raise ValueError("Locked labels and items.json belong to different batches")
        for image in items["images"]:
            for item in image["items"]:
                if item["item_id"] in captions:
                    raise ValueError(f"Repeated item in items.json: {item['item_id']}")
                captions[item["item_id"]] = (image["image_id"], item["caption"])
    result = []
    for row in rows:
        image_id, item_id = row["image_id"], row["item_id"]
        caption = row.get("caption")
        if caption is None:
            if item_id not in captions or captions[item_id][0] != image_id:
                raise ValueError(f"Item {item_id} does not match image {image_id} in items.json")
            caption = captions[item_id][1]
        flags = {name: int(value) if value in ("0", "1") else value for name, value in row["labels"].items()}
        result.append({"image_id": image_id, "item_id": item_id, "caption": caption, **flags})
    return result


def check_rows(rows, image_ids, name):
    """Exactly three captions for each listed image, unique item IDs, every flag 0, 1 or "?"."""
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError(f"{name}: item_id repeated")
    if Counter(row["image_id"] for row in rows) != Counter({image_id: 3 for image_id in image_ids}):
        raise ValueError(f"{name}: need exactly three captions for each of {len(image_ids)} images")
    for row in rows:
        if any(row[flag] not in (0, 1, "?") for flag in FLAGS):
            raise ValueError(f"{name}: {row['item_id']} has a flag that is not 0, 1 or ?")


def attribute_models(rows, predictions, blind_key=None):
    """Resolve models from the key or match captions when identical captions have identical flags."""
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    owner = {}
    for image_id, image_rows in sorted(by_image.items()):
        pending = {name: predictions[name][image_id] for name in GENERATORS}
        signature_by_caption = {}
        for row in image_rows:
            item_id, caption = row["item_id"], row["caption"]
            if blind_key is None:
                signature = tuple(row[f] for f in FLAGS)
                if signature_by_caption.setdefault(caption, signature) != signature:
                    raise ValueError(f"Image {image_id}: identical captions have different flags; pass --blind-key")
                name = next((n for n, text in pending.items() if text == caption), None)
            else:
                item = blind_key.get(item_id, {})
                name = item.get("model") if item.get("image_id") == image_id else None
            if name is None or pending.get(name) != caption:
                raise ValueError(f"Image {image_id}: item {item_id} does not match a saved prediction")
            owner[item_id] = name
            del pending[name]
    return owner


def agreement(first, second):
    """Cell agreement before discussion: same value / different value among cells that are 0 or 1 on both sides."""
    second = {row["item_id"]: row for row in second}
    result = {"cells": 0, "both_clear": 0, "same": 0, "different": 0, "both_unclear": 0, "one_unclear": 0,
              "by_flag": {flag: 0 for flag in FLAGS}, "different_cells": []}
    for row in first:
        for flag in FLAGS:
            a, b = row[flag], second[row["item_id"]][flag]
            result["cells"] += 1
            if a == "?" and b == "?":
                result["both_unclear"] += 1
            elif "?" in (a, b):
                result["one_unclear"] += 1
            else:
                result["both_clear"] += 1
                if a == b:
                    result["same"] += 1
                else:
                    result["different"] += 1
                    result["by_flag"][flag] += 1
                    result["different_cells"].append({"item_id": row["item_id"], "flag": flag, "first": a, "second": b})
    return result


def merge_final(linh_rows, vanh_rows, shared_ids, resolution):
    """One row per caption: own labels on own images; shared images use Linh's labels with the agreed cells."""
    final = [dict(row) for row in vanh_rows if row["image_id"] not in shared_ids]
    for row in linh_rows:
        final.append({**row, **resolution.get(row["item_id"], {})} if row["image_id"] in shared_ids else dict(row))
    return sorted(final, key=lambda row: (row["image_id"], row["item_id"]))


def locked_resolution(linh_rows, vanh_rows, shared_ids, locked_rows):
    """Check the locked table against both originals and read decisions for disputed cells only."""
    baseline = merge_final(linh_rows, vanh_rows, shared_ids, {})
    check_rows(locked_rows, sorted({row["image_id"] for row in baseline}), "Locked table")
    by_item = {row["item_id"]: row for row in locked_rows}
    vanh = {row["item_id"]: row for row in vanh_rows}
    resolution = {}
    for row in baseline:
        item_id, image_id = row["item_id"], row["image_id"]
        locked = by_item.get(item_id)
        if locked is None or (locked["image_id"], locked["caption"]) != (image_id, row["caption"]):
            raise ValueError(f"Locked table changes the identity or caption of {item_id}")
        for flag in FLAGS:
            if image_id in shared_ids and row[flag] != vanh[item_id][flag]:
                resolution.setdefault(item_id, {})[flag] = locked[flag]
            elif row[flag] != locked[flag]:
                raise ValueError(f"Locked table changes an undisputed label at {item_id} {flag}")
    return resolution


def error_table(rows, owner):
    """Per model and flag: captions with the error, captions with a clear 0/1 label, unclear captions."""
    table = {}
    for name in GENERATORS:
        mine = [row for row in rows if owner[row["item_id"]] == name]
        entry = {"captions": len(mine)}
        for flag in FLAGS:
            values = [row[flag] for row in mine]
            entry[flag] = {"errors": values.count(1), "clear": len(values) - values.count("?"), "unclear": values.count("?")}
        errors = sum(any(row[f] == 1 for f in FLAGS) for row in mine)
        zero = sum(all(row[f] == 0 for f in FLAGS) for row in mine)
        entry["any"] = {"errors": errors, "clear": errors + zero, "unclear": len(mine) - errors - zero}
        table[name] = entry
    return table


def table_markdown(table):
    lines = ["| Loại lỗi | " + " | ".join(f"{n} ({table[n]['captions']} caption)" for n in table) + " |",
             "|---|" + "---|" * len(table)]
    for flag in (*FLAGS, "any"):
        cells = []
        for name in table:
            cell = table[name][flag]
            rate = vi(100 * cell["errors"] / cell["clear"]) if cell["clear"] else "–"
            cells.append(f"{cell['errors']}/{cell['clear']} ({rate}%) · {cell['unclear']} ?")
        lines.append(f"| {'**Có ít nhất một lỗi**' if flag == 'any' else flag} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def trace_examples(image_ids, rows, owner, predictions, manifest_rows):
    """Image file and pixel hash, each model's saved caption with its item ID and flags, and the five references."""
    by_id = {row["image_id"]: row for row in manifest_rows}
    traced = []
    for image_id in image_ids:
        entry = {"image_id": image_id, "image_path": by_id[image_id]["image_path"],
                 "pixel_sha256": by_id[image_id]["pixel_sha256"], "references": by_id[image_id]["captions"], "models": {}}
        for name in GENERATORS:
            caption = predictions[name][image_id]
            item = next((r for r in rows if r["image_id"] == image_id and owner[r["item_id"]] == name), None)
            entry["models"][name] = {"caption": caption, "item_id": item["item_id"] if item else None,
                                     "flags": {f: item[f] for f in FLAGS} if item else None}
        traced.append(entry)
    return traced


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--linh-sheet", type=Path, required=True)
    parser.add_argument("--vanh-labels", type=Path)
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--resolution", type=Path)
    decisions.add_argument("--locked-labels", type=Path)
    parser.add_argument("--blind-key", type=Path, help="Use the private item-to-model key after locking labels")
    parser.add_argument("--selection", type=Path, default=Path("artifacts/manual_caption_review/selection.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("reports/test_results"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/ktvic/test_manifest.jsonl"))
    parser.add_argument("--example-ids", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/manual_caption_review"))
    args = parser.parse_args()
    if (args.resolution or args.locked_labels) and not args.vanh_labels:
        parser.error("Adjudication requires --vanh-labels")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    shared = set(selection["overlap_ids"])
    predictions = {name: {row["image_id"]: row["caption"] for row in json.loads(
        (args.predictions_dir / f"{name}_test_predictions.json").read_text(encoding="utf-8"))} for name in GENERATORS}
    linh = parse_sheet(args.linh_sheet)
    check_rows(linh, selection["linh_khanh_ids"], "Linh Khánh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"linh_sheet_sha256": sha256_of(args.linh_sheet), "selection_sha256": sha256_of(args.selection),
               "linh_rows": len(linh),
               "linh_rows_with_error": sum(any(r[f] == 1 for f in FLAGS) for r in linh),
               "linh_rows_only_unclear": sum(all(r[f] != 1 for f in FLAGS) and any(r[f] == "?" for f in FLAGS) for r in linh),
               "linh_rows_all_zero": sum(all(r[f] == 0 for f in FLAGS) for r in linh)}
    rows = linh
    if args.vanh_labels:
        vanh = load_label_rows(args.vanh_labels)
        check_rows(vanh, selection["vanh_ids"], "Vanh")
        shared_linh = [r for r in linh if r["image_id"] in shared]
        by_vanh = {r["item_id"]: r for r in vanh}
        for row in shared_linh:  # the two graders must have seen the same blind item
            matching = by_vanh.get(row["item_id"])
            if matching is None or (matching["image_id"], matching["caption"]) != (row["image_id"], row["caption"]):
                raise ValueError(f"Shared item {row['item_id']} differs between the two sheets")
        summary["agreement_shared"] = agreement(shared_linh, [by_vanh[r["item_id"]] for r in shared_linh])
        if args.locked_labels:
            resolution = locked_resolution(linh, vanh, shared, load_label_rows(args.locked_labels))
        elif args.resolution:
            resolution = json.loads(args.resolution.read_text(encoding="utf-8"))
        else:
            shared_agreement = summary["agreement_shared"]
            if shared_agreement["different"] or shared_agreement["one_unclear"]:
                raise ValueError("The shared labels disagree; supply --resolution or --locked-labels")
            resolution = {}
        rows = merge_final(linh, vanh, shared, resolution)
        check_rows(rows, selection["selected_ids"], "Final table")
        summary["resolution_cells"] = sum(len(v) for v in resolution.values())
    blind_key = json.loads(args.blind_key.read_text(encoding="utf-8"))["items"] if args.blind_key else None
    owner = attribute_models(rows, predictions, blind_key)
    summary["table"] = error_table(rows, owner)
    summary["table_images"] = len({r["image_id"] for r in rows})
    if args.example_ids:
        summary["examples"] = trace_examples(args.example_ids, rows, owner, predictions, read_manifest(args.manifest))
    scope = "all40" if args.vanh_labels else "linh25"  # never label a one-grader table as the 40-image result
    (args.output_dir / f"labels_{scope}.json").write_text(json.dumps(
        [{**r, "model": owner[r["item_id"]]} for r in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"summary_{scope}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = table_markdown(summary["table"])
    (args.output_dir / f"table_{scope}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print({k: v for k, v in summary.items() if k not in ("table", "examples", "agreement_shared")})
    if "agreement_shared" in summary:
        print({k: v for k, v in summary["agreement_shared"].items() if k != "different_cells"})


if __name__ == "__main__":
    main()
