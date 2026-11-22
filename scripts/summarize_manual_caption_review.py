"""Manual caption-error review on the 40 test images: parse sheets, compare two graders, tabulate by model.

Check the Linh Khanh side alone (needs only her submitted Markdown sheet and the saved predictions):
  python -m scripts.summarize_manual_caption_review --linh-sheet <sheet.md>
Add the second grader's locked labels and the agreed resolution to get the agreement on the 10 shared images and
the table over all 40 images. Both JSON files are lists of rows
  {"image_id": 245, "item_id": "it-...", "caption": "...", "VT": 0, "HD": 0, "QH": 0, "SL": 0, "BD": 0, "LT": 0}
for the labels and {"item_id": {"QH": 1, "BD": 0}} for the resolution:
  python -m scripts.summarize_manual_caption_review --linh-sheet <sheet.md> --vanh-labels <vanh.json> \
      --resolution <resolution.json> --example-ids 3091 297 756 66
Flags are 0 (no error found), 1 (error) or "?" (unclear, never turned into 0).
"""

import argparse
import hashlib
import json
import re
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
        heading = re.match(r"## Ảnh (\d+)$", line)
        if heading:
            image_id = int(heading.group(1))
            continue
        match = ROW.match(line)
        if match:
            flags = {name: (value if value == "?" else int(value)) for name, value in zip(FLAGS, match.groups()[2:8])}
            rows.append({"image_id": image_id, "item_id": match.group(1), "caption": match.group(2), **flags})
    return rows


def check_rows(rows, image_ids, name):
    """Exactly three captions for each listed image, unique item IDs, every flag 0, 1 or "?"."""
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError(f"{name}: item_id repeated")
    if sorted({row["image_id"] for row in rows}) != sorted(image_ids) or len(rows) != 3 * len(image_ids):
        raise ValueError(f"{name}: need exactly three captions for each of {len(image_ids)} images")
    for row in rows:
        if any(row[flag] not in (0, 1, "?") for flag in FLAGS):
            raise ValueError(f"{name}: {row['item_id']} has a flag that is not 0, 1 or ?")


def attribute_models(rows, predictions):
    """Give each blind item the model whose saved prediction for that image has the same caption.

    Models that wrote the same caption cannot be told apart and carry the same flags, so one is used per item.
    """
    owner = {}
    for image_id in sorted({row["image_id"] for row in rows}):
        pending = {name: predictions[name][image_id] for name in GENERATORS}
        for row in (r for r in rows if r["image_id"] == image_id):
            name = next((n for n, caption in pending.items() if caption == row["caption"]), None)
            if name is None:
                raise ValueError(f"Image {image_id}: caption of {row['item_id']} is not a saved prediction")
            owner[row["item_id"]] = name
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
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--selection", type=Path, default=Path("artifacts/manual_caption_review/selection.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("reports/test_results"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/ktvic/test_manifest.jsonl"))
    parser.add_argument("--example-ids", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/manual_caption_review"))
    args = parser.parse_args()

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
        vanh = json.loads(args.vanh_labels.read_text(encoding="utf-8"))
        check_rows(vanh, selection["vanh_ids"], "Vanh")
        resolution = json.loads(args.resolution.read_text(encoding="utf-8")) if args.resolution else {}
        shared_linh = [r for r in linh if r["image_id"] in shared]
        by_vanh = {r["item_id"]: r for r in vanh}
        for row in shared_linh:  # the two graders must have seen the same blind item
            if row["item_id"] not in by_vanh or by_vanh[row["item_id"]]["caption"] != row["caption"]:
                raise ValueError(f"Shared item {row['item_id']} differs between the two sheets")
        summary["agreement_shared"] = agreement(shared_linh, [by_vanh[r["item_id"]] for r in shared_linh])
        rows = merge_final(linh, vanh, shared, resolution)
        summary["resolution_cells"] = sum(len(v) for v in resolution.values())
    owner = attribute_models(rows, predictions)
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
