"""BLEU-1..4 and CIDEr for one predicted caption per image against all of that image's references."""

import json
from importlib import metadata
from pathlib import Path

from src.data.vietnamese_caption_batch import normalize_caption

# All scores are multiplied by 100. This pycocoevalcap implementation uses CIDEr-D
# (clipping and a length penalty), whose native range is 0..10. Its IDF weights
# come from the reference set being scored, so comparisons require the same split.
SCORE_SCALE = 100.0
METRICS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr")


class ScorerInputError(ValueError):
    """Raised with a readable Vietnamese message when predictions and references do not line up."""


def library_versions():
    import numpy

    try:
        scorer = metadata.version("pycocoevalcap")
    except metadata.PackageNotFoundError:
        scorer = "chưa cài"
    return {"pycocoevalcap": scorer, "numpy": numpy.__version__}


def load_references(manifest_path):
    """{image_id: [all reference captions]} from a *_manifest.jsonl (five captions per image)."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ScorerInputError(f"Không thấy file tham chiếu: {path}")
    references = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    image_id = int(row["image_id"])
                    captions = row["captions"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    raise ScorerInputError(f"Manifest {path}, dòng {line_number}: image_id/captions không hợp lệ.") from None
                if image_id in references:
                    raise ScorerInputError(f"Manifest có ID trùng: {image_id} (dòng {line_number}).")
                if not isinstance(captions, list) or not captions or any(
                    not isinstance(c, str) or not normalize_caption(c) for c in captions
                ):
                    raise ScorerInputError(f"Ảnh ID {image_id}: captions phải là danh sách câu tham chiếu hợp lệ.")
                references[image_id] = captions
    return references


def load_image_paths(manifest_path):
    with Path(manifest_path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {int(row["image_id"]): row["image_path"] for row in rows}


def load_predictions(path):
    """Read predictions as a list of (image_id, caption).

    Accepts a JSON list of {"image_id", "caption"}, a JSON object {"<id>": "<caption>"}, or JSONL of the first form.
    """
    path = Path(path)
    if not path.is_file():
        raise ScorerInputError(f"Không thấy file dự đoán: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ScorerInputError(f"File dự đoán không phải JSON/JSONL hợp lệ: {error}") from None
    try:
        if isinstance(data, dict):
            if "image_id" in data and "caption" in data:
                return [(int(data["image_id"]), data["caption"])]
            return [(int(key), value) for key, value in data.items()]
        return [(int(item["image_id"]), item["caption"]) for item in data]
    except (KeyError, TypeError, ValueError):
        raise ScorerInputError(
            'Mỗi dự đoán cần có "image_id" (số nguyên) và "caption" (chuỗi), '
            'hoặc dùng dạng {"<image_id>": "<caption>"}.') from None


def _preview(ids, limit=10):
    shown = ", ".join(str(i) for i in sorted(ids)[:limit])
    return shown + (f", ... (+{len(ids) - limit})" if len(ids) > limit else "")


def validate(predictions, references, image_paths=None, image_root=None, allow_empty=False):
    """Return {image_id: caption} for the IDs to score, or raise ScorerInputError listing every problem."""
    problems, by_id, repeated = [], {}, set()
    for image_id, caption in predictions:
        if image_id in by_id:
            repeated.add(image_id)
        by_id[image_id] = caption
    if repeated:
        problems.append(f"{len(repeated)} ảnh có nhiều hơn một dự đoán (ID: {_preview(repeated)}). Mỗi ảnh chỉ một câu.")
    unknown = set(by_id) - set(references)
    if unknown:
        problems.append(f"{len(unknown)} ID dự đoán không thuộc tập tham chiếu (ID: {_preview(unknown)}). "
                        "Kiểm tra đã chấm đúng tập (val/test) chưa.")
    missing = set(references) - set(by_id)
    if missing:
        problems.append(f"Thiếu dự đoán cho {len(missing)}/{len(references)} ảnh của tập tham chiếu "
                        f"(ID: {_preview(missing)}). Sinh đủ caption cho mọi ảnh của tập.")
    bad = [i for i, c in by_id.items() if i in references
           and (not isinstance(c, str) or (not allow_empty and not normalize_caption(c)))]
    if bad:
        problems.append(f"{len(bad)} ảnh có caption rỗng hoặc không phải chuỗi (ID: {_preview(bad)}).")
    short = [i for i in by_id if i in references and not references[i]]
    if short:
        problems.append(f"{len(short)} ảnh không có câu tham chiếu (ID: {_preview(short)}).")
    if image_paths is not None and image_root is not None:
        absent = [i for i in by_id if i in image_paths and not (Path(image_root) / image_paths[i]).is_file()]
        if absent:
            problems.append(f"Thiếu file ảnh trong {image_root} cho {len(absent)} ID (ID: {_preview(absent)}). "
                            "Giải nén lại train-images/ và public-test-images/.")
    if not by_id:
        problems.append("File dự đoán không có câu nào.")
    if problems:
        raise ScorerInputError("\n".join(f"- {p}" for p in problems))
    return {i: by_id[i] for i in sorted(by_id)}


def score_captions(predictions, references, **validate_options):
    """Score (image_id, caption) pairs against all references; tokens follow normalize_caption."""
    chosen = validate(predictions, references, **validate_options)
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    res = {i: [normalize_caption(c)] for i, c in chosen.items()}
    gts = {i: [normalize_caption(r) for r in references[i]] for i in chosen}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    values = [*bleu, cider]
    return {
        "num_images": len(chosen),
        "num_references": sum(len(v) for v in gts.values()),
        "scale": f"x{SCORE_SCALE:g}; BLEU 0-100, CIDEr-D 0-1000",
        "scores": {name: round(float(value) * SCORE_SCALE, 2) for name, value in zip(METRICS, values)},
        "libraries": library_versions(),
    }
