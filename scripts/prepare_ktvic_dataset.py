"""Audit unzipped author images and generate grouped KTVIC manifests and a sample batch."""

import argparse
import hashlib
import json
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.data.vietnamese_caption_batch import (
    UNK, CaptionVocabulary, make_caption_batch, normalize_caption,
)


IMAGE_FOLDERS = {"train": "train-images", "test": "public-test-images"}


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_source(root, split):
    folder = root / IMAGE_FOLDERS[split]
    source = json.loads((root / f"{split}_data.json").read_text(encoding="utf-8"))
    captions = defaultdict(list)
    seen_caption_ids = set()
    repeated, spacing, non_nfc = [], [], []
    for item in source["annotations"]:
        aid, image_id, text = item["id"], item["image_id"], item["caption"]
        if aid in seen_caption_ids or not normalize_caption(text) or "\ufffd" in text:
            raise ValueError(f"Invalid caption {split}/{aid} for image {image_id}")
        seen_caption_ids.add(aid)
        reference = {"image_id": image_id, "caption_id": aid}
        if text in captions[image_id]:
            repeated.append(reference)
        if text != " ".join(text.split()):
            spacing.append(reference)
        if not unicodedata.is_normalized("NFC", text):
            non_nfc.append(reference)
        captions[image_id].append(text)

    rows, names = {}, set()
    for item in source["images"]:
        image_id, name = item["id"], item["filename"]
        if image_id in rows or Path(name).name != name or int(Path(name).stem) != image_id:
            raise ValueError(f"Duplicate ID or invalid filename in {split}: {item}")
        path = folder / name
        with Image.open(path) as source_image:
            rgb = source_image.convert("RGB")
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode("ascii"))
            digest.update(rgb.tobytes())
        rows[image_id] = {
            "image_id": image_id, "image_path": path.relative_to(root).as_posix(),
            "pixel_sha256": digest.hexdigest(), "captions": captions[image_id],
        }
        names.add(name)
    if set(rows) != set(captions) or any(len(group) != 5 for group in captions.values()):
        raise ValueError(f"Image IDs and groups of five references disagree in {split}")
    files = {p.name for p in folder.iterdir() if p.is_file()}
    if files != names:
        raise ValueError(f"Image inventory differs in {split}: {sorted(files ^ names)}")
    return rows, {
        "images": len(rows), "captions": len(seen_caption_ids),
        "repeated_references": repeated, "spacing": spacing, "non_nfc": non_nfc,
    }


def split_images(train, test, seed, reviewed_groups=()):
    if train.keys() & test.keys():
        raise ValueError("Official train and test share image IDs")
    all_rows = train | test
    by_hash = defaultdict(set)
    for i, row in all_rows.items():
        by_hash[row["pixel_sha256"]].add(i)
    family = {i: group for group in by_hash.values() for i in group}
    for related_ids in reviewed_groups:
        merged = set().union(*(family[i] for i in related_ids))
        for i in merged:
            family[i] = merged
    groups = {}
    excluded = []
    for i in sorted(train):
        test_ids = sorted(j for j in family[i] if j in test)
        if test_ids:
            identical = any(test[j]["pixel_sha256"] == train[i]["pixel_sha256"] for j in test_ids)
            excluded.append({"train_id": i, "test_ids": test_ids, "captions": len(train[i]["captions"]),
                             "reason": "identical RGB pixels" if identical else "visually verified duplicate"})
        else:
            groups.setdefault(min(family[i]), []).append(i)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    target = round(sum(map(len, group_list)) * 0.1)
    val_ids = set()
    for group in group_list:
        if len(val_ids) >= target:
            break
        val_ids.update(group)
    splits = {"train": [], "val": [], "test": [test[i] for i in sorted(test)]}
    for group in group_list:
        name = "val" if group[0] in val_ids else "train"
        splits[name].extend(train[i] for i in group)
    for rows in splits.values():
        rows.sort(key=lambda row: row["image_id"])
    for field in ("image_id", "pixel_sha256"):
        seen = set()
        for rows in splits.values():
            keys = {row[field] for row in rows}
            if seen & keys:
                raise ValueError(f"Cross-split overlap in {field}")
            seen.update(keys)
    seen_families = set()
    for rows in splits.values():
        keys = {min(family[row["image_id"]]) for row in rows}
        if seen_families & keys:
            raise ValueError("A reviewed photo group appears in multiple splits")
        seen_families.update(keys)
    return splits, excluded, [group for group in groups.values() if len(group) > 1]


def prepare(raw, output, seed):
    train, train_audit = read_source(raw, "train")
    test, test_audit = read_source(raw, "test")
    config = Path(__file__).resolve().parents[1] / "configs/ktvic_duplicate_image_groups.yaml"
    reviewed_groups = yaml.safe_load(config.read_text(encoding="utf-8"))
    splits, excluded, duplicates = split_images(train, test, seed, reviewed_groups)
    vocab = CaptionVocabulary.from_training_captions(
        cap for row in splits["train"] for cap in row["captions"]
    )
    summary = {}
    hashes = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        manifest = output / f"{name}_manifest.jsonl"
        with manifest.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[manifest.name] = sha256_file(manifest)
        write_json(output / f"{name}_ids.json", [row["image_id"] for row in rows])
        lengths, token_counts = [], Counter()
        for row in rows:
            for cap in row["captions"]:
                tokens = vocab.encode(cap)
                lengths.append(len(tokens))
                token_counts.update(tokens)
        summary[name] = {"images": len(rows), "captions": len(lengths),
                         "token_lengths_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)],
                         "tokens": token_counts.total(), "unk": token_counts[UNK]}

    samples = [(row["image_id"], row["image_path"], row["captions"][index])
               for index, row in enumerate(splits["train"][:4])]
    batch = make_caption_batch(samples, vocab, raw)
    torch.save(batch, output / "sample_caption_batch.pt")
    write_json(output / "sample_caption_batch.json", {
        "shapes": {key: list(value.shape) for key, value in batch.items()},
        "samples": [dict(image_id=i, caption=cap, decoded=vocab.decode(labels))
                    for (i, _, cap), labels in zip(samples, batch["labels"])],
        **{key: value.tolist() for key, value in batch.items() if key != "images"},
    })
    write_json(output / "caption_vocabulary.json", vocab.tokens)
    audit = {"seed": seed, "source": {"train": train_audit, "test": test_audit},
             "source_sha256": {p.name: sha256_file(p) for p in sorted(raw.iterdir())
                               if p.name in ("train_data.json", "test_data.json",
                                             "train-images.zip", "public-test-images.zip")},
             "excluded": excluded, "train_duplicate_groups_kept_together": duplicates,
             "cross_split_id_overlap": 0, "cross_split_pixel_hash_overlap": 0,
             "cross_split_reviewed_group_overlap": 0,
             "reviewed_group_config_sha256": sha256_file(config),
             "splits": summary, "vocabulary_size": len(vocab.tokens), "manifest_sha256": hashes}
    write_json(output / "dataset_audit.json", audit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.seed)
