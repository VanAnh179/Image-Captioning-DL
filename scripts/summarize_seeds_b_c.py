"""Summarize B and C validation scores and their differences for matched seeds."""

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr")


def load(directory):
    saved = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
    return {"seed": saved["settings"]["seed"], "best_epoch": saved["best_epoch"], "last_epoch": saved["last_epoch"],
            "train_seconds": round(saved["train_seconds"], 1), **saved["scores"]["scores"]}


def summarize(runs):
    return {name: {"mean": round(statistics.mean(r[name] for r in runs), 2),
                   "sd": round(statistics.stdev(r[name] for r in runs), 2),
                   "min": min(r[name] for r in runs), "max": max(r[name] for r in runs)} for name in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=Path, nargs="+", required=True, help="run directories of B")
    parser.add_argument("--c", type=Path, nargs="+", required=True, help="run directories of C")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b_runs, c_runs = [load(d) for d in args.b], [load(d) for d in args.c]
    b_seeds = {r["seed"] for r in b_runs}
    c_seeds = {r["seed"] for r in c_runs}
    if b_seeds != c_seeds or len(b_seeds) != len(b_runs) or len(c_seeds) != len(c_runs):
        raise ValueError("B and C must each contain one run for every matched seed")
    if len(b_seeds) < 2:
        raise ValueError("At least two matched seeds are required to summarize variability")
    b_runs.sort(key=lambda r: r["seed"])
    c_runs.sort(key=lambda r: r["seed"])
    b_cider, c_cider = [r["CIDEr"] for r in b_runs], [r["CIDEr"] for r in c_runs]
    report = {"seeds": [r["seed"] for r in b_runs], "B": {"runs": b_runs, "summary": summarize(b_runs)},
              "C": {"runs": c_runs, "summary": summarize(c_runs)},
              "cider_c_minus_b_same_seed": [round(c - b, 2) for b, c in zip(b_cider, c_cider)],
              "cider_mean_difference": round(statistics.mean(c_cider) - statistics.mean(b_cider), 2),
              "every_c_seed_above_every_b_seed": min(c_cider) > max(b_cider)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("B", "C")}, indent=2))
    print(json.dumps({"B": report["B"]["summary"], "C": report["C"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
