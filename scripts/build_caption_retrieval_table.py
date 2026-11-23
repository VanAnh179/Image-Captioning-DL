"""Test-split caption scores and image-search Recall side by side, checked against the two scripts' own outputs.

  python -m scripts.build_caption_retrieval_table --recall A,B=runs/retrieval_test_eval
Add `--recall A0,C=<dir>` when Recall was also run for A0 and C. Each directory holds recall_summary.json and
ranks_A.csv / ranks_B.csv written by `scripts.evaluate_caption_retrieval --split test`; the two names say which
models that run's "A" and "B" slots were fed with. Caption scores come from reports/test_results/test_table.json
(`scripts.build_test_score_table`).

The table is only written after these checks pass:
  - every caption file is the one the scores were computed from, and re-scoring it gives the same BLEU/CIDEr;
  - every Recall run used the same caption file (up to line endings), the full test pool and five queries per image,
    and its hits and median rank recomputed from the rank CSV equal the summary.
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from scripts.build_test_score_table import ENCODERS, METRICS, vi
from src.data.region_features import read_manifest
from src.evaluation.caption_scorer import load_predictions, score_captions

CUTOFFS = (1, 5, 10)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def check_caption_scores(entry, prediction_file, references):
    """Scores in test_table.json are the scores of this exact file, recomputed."""
    if sha256_bytes(prediction_file.read_bytes()) != entry["predictions_sha256"]:
        raise ValueError(f"{entry['model']}: {prediction_file.name} is not the file the table was built from")
    again = score_captions(load_predictions(prediction_file), references)["scores"]
    if again != entry["test_scores"]:
        raise ValueError(f"{entry['model']}: re-scoring gives {again}, table has {entry['test_scores']}")


def read_ranks(path, expected_queries=None):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if expected_queries is not None:
        actual = [(int(row["query_number"]), int(row["image_id"]), int(row["reference_number"]), row["query"])
                  for row in rows]
        if actual != expected_queries:
            raise ValueError(f"{path}: query IDs, references or order differ from the test manifest")
        if any(row["zero_query"] not in ("0", "1") or (not row["rank"]) != (row["zero_query"] == "1")
               for row in rows):
            raise ValueError(f"{path}: a missing rank must correspond to an empty query vector")
    return [int(row["rank"]) if row["rank"] else None for row in rows]


def check_recall_slot(summary, slot, ranks, prediction_file, images, queries):
    """Recall numbers recomputed from the per-query ranks, and the caption file the index was built from."""
    block = summary["models"][slot]
    if (block["candidate_images"], block["queries"], len(ranks)) != (images, queries, queries):
        raise ValueError(f"slot {slot}: expected {images} images and {queries} queries, "
                         f"got {block['candidate_images']}, {block['queries']}, {len(ranks)} ranks")
    ranked = [rank for rank in ranks if rank is not None]
    if any(rank < 1 or rank > images for rank in ranked):
        raise ValueError(f"slot {slot}: rank outside the candidate pool")
    for k in CUTOFFS:
        hits = sum(rank <= k for rank in ranked)
        entry = block["recall"][f"R@{k}"]
        if hits != entry["hits_search_rule"] or hits / queries != entry["search_rule"]:
            raise ValueError(f"slot {slot}: R@{k} recomputed {hits}/{queries} disagrees with the summary")
    median = statistics.median(ranked) if ranked else None
    if median != block["median_rank"]:
        raise ValueError(f"slot {slot}: median rank recomputed {median}, summary {block['median_rank']}")
    data = prediction_file.read_bytes()
    if block["predictions_sha256"] == sha256_bytes(data):
        return "identical"
    lf = data.replace(b"\r\n", b"\n")
    if block["predictions_sha256"] in (sha256_bytes(lf), sha256_bytes(lf.replace(b"\n", b"\r\n"))):
        return "same bytes except line endings"
    raise ValueError(f"slot {slot}: the index was built from a different caption file than {prediction_file.name}")


def recall_cells(directory, names, results_dir, images, expected_queries):
    summary = json.loads((directory / "recall_summary.json").read_text(encoding="utf-8"))
    queries = len(expected_queries)
    cells, notes = {}, {}
    for slot, name in zip(("A", "B"), names):
        ranks = read_ranks(directory / f"ranks_{slot}.csv", expected_queries)
        notes[name] = check_recall_slot(summary, slot, ranks, results_dir / f"{name}_test_predictions.json", images, queries)
        block = summary["models"][slot]
        cells[name] = {"queries": queries, "median_rank": block["median_rank"],
                       **{f"R@{k}": block["recall"][f"R@{k}"]["hits_search_rule"] for k in CUTOFFS}}
    return cells, notes


def leaders(rows, columns, models):
    """Per column, the models with the best value (highest, or lowest for rank columns); ties are all listed."""
    result = {}
    for column, higher_is_better in columns:
        values = {row["model"]: row[column] for row in rows if row["model"] in models and row.get(column) is not None}
        if values:
            best = max(values.values()) if higher_is_better else min(values.values())
            result[column] = sorted(name for name, value in values.items() if value == best)
    return result


def table_markdown(rows, lead_all, lead_first, first_models, queries):
    head = ["Mô hình", "Encoder", *METRICS, "R@1", "R@5", "R@10", "Hạng trung vị"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    for row in rows:
        recall = [f"{vi(100 * row[f'R@{k}'] / queries)}% ({vi(row[f'R@{k}'], 0)})" if f"R@{k}" in row else "–" for k in CUTOFFS]
        rank = row.get("median_rank")
        median = vi(rank, 1 if rank % 1 else 0) if rank is not None else "–"
        lines.append(f"| {row['model']} | {ENCODERS[row['model']]} | "
                     + " | ".join(vi(row[m]) for m in METRICS) + " | " + " | ".join(recall) + f" | {median} |")
    columns = [*METRICS, "R@1", "R@5", "R@10", "median_rank"]
    for label, lead in (("**Đứng đầu (mọi mô hình có số đo)**", lead_all),
                        (f"**Đứng đầu, chỉ xét {' và '.join(first_models)}**", lead_first)):
        lines.append(f"| {label} | | " + " | ".join(", ".join(lead.get(c, ["–"])) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recall", action="append", required=True, metavar="NAME_A,NAME_B=DIR")
    parser.add_argument("--results-dir", type=Path, default=Path("reports/test_results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/caption_retrieval_table"))
    args = parser.parse_args()

    manifest_path = args.processed_dir / "test_manifest.jsonl"
    manifest = read_manifest(manifest_path)
    references = {row["image_id"]: row["captions"] for row in manifest}
    images = len(manifest)
    if not references or len(references) != images or any(len(captions) != 5 for captions in references.values()):
        raise ValueError("Expected unique test IDs and five references per image")
    query_rows = ((row["image_id"], number, caption)
                  for row in manifest for number, caption in enumerate(row["captions"], 1))
    expected_queries = [(order, *query) for order, query in enumerate(query_rows, 1)]
    queries = len(expected_queries)
    table = json.loads((args.results_dir / "test_table.json").read_text(encoding="utf-8"))
    if (table["split"], table["images"], table["references"]) != ("test", images, queries) or (
            table["test_manifest_sha256"] != sha256_bytes(manifest_path.read_bytes())):
        raise ValueError("The caption table was built from a different test manifest")
    rows = {entry["model"]: {"model": entry["model"], **entry["test_scores"]} for entry in table["models"]}
    if set(rows) != set(ENCODERS) or len(table["models"]) != len(ENCODERS):
        raise ValueError("Expected one caption-table row for each of A, B, A0 and C")
    for entry in table["models"]:
        check_caption_scores(entry, args.results_dir / f"{entry['model']}_test_predictions.json", references)

    sources, first_models = {}, None
    for spec in args.recall:
        names, _, directory = spec.partition("=")
        names = names.split(",")
        if len(names) != 2 or names[0] == names[1] or not directory or not set(names) <= set(rows):
            parser.error(f"--recall needs NAME_A,NAME_B=DIR with names from {sorted(rows)}: {spec!r}")
        cells, notes = recall_cells(Path(directory), names, args.results_dir, images, expected_queries)
        first_models = first_models or names
        for name in names:
            if "R@1" in rows[name]:
                parser.error(f"{name} appears in more than one --recall")
            rows[name].update(cells[name])
            sources[name] = {"directory": directory, "caption_file_vs_retrieval_index": notes[name]}

    ordered = [rows[entry["model"]] for entry in table["models"]]
    columns = [*((m, True) for m in METRICS), *((f"R@{k}", True) for k in CUTOFFS), ("median_rank", False)]
    lead_all = leaders(ordered, columns, {row["model"] for row in ordered})
    lead_first = leaders(ordered, columns, set(first_models))
    markdown = table_markdown(ordered, lead_all, lead_first, first_models, queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bang_captioning_va_tim_anh.md").write_text(markdown, encoding="utf-8")
    result = {"images": images, "queries": queries, "test_table_git_commit": table["git_commit"], "rows": ordered,
              "leaders_all": lead_all, "leaders_first_run": {"models": first_models, **lead_first},
              "recall_sources": sources, "checks": "caption scores re-scored; Recall recomputed from rank CSVs"}
    (args.output_dir / "bang_captioning_va_tim_anh.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    print(json.dumps({"leaders_all": lead_all, "sources": sources}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
