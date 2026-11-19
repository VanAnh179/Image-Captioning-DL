"""Summarize B and C validation scores over the same seeds and test whether the two groups separate."""

import argparse
import itertools
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


def separation_p_value(low_group, high_group):
    """One-sided exact permutation probability that every value of high_group outranks every value of low_group."""
    pooled = list(low_group) + list(high_group)
    size = len(high_group)
    hits = sum(min(pooled[i] for i in chosen) > max(pooled[i] for i in range(len(pooled)) if i not in chosen)
               for chosen in itertools.combinations(range(len(pooled)), size))
    return hits / sum(1 for _ in itertools.combinations(range(len(pooled)), size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=Path, nargs="+", required=True, help="run directories of B")
    parser.add_argument("--c", type=Path, nargs="+", required=True, help="run directories of C")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b_runs, c_runs = [load(d) for d in args.b], [load(d) for d in args.c]
    if sorted(r["seed"] for r in b_runs) != sorted(r["seed"] for r in c_runs):
        raise ValueError("B and C must use the same seeds")
    b_runs.sort(key=lambda r: r["seed"])
    c_runs.sort(key=lambda r: r["seed"])
    b_cider, c_cider = [r["CIDEr"] for r in b_runs], [r["CIDEr"] for r in c_runs]
    se = (statistics.variance(b_cider) / len(b_cider) + statistics.variance(c_cider) / len(c_cider)) ** 0.5
    report = {"seeds": [r["seed"] for r in b_runs], "B": {"runs": b_runs, "summary": summarize(b_runs)},
              "C": {"runs": c_runs, "summary": summarize(c_runs)},
              "cider_c_minus_b_same_seed": [round(c - b, 2) for b, c in zip(b_cider, c_cider)],
              "cider_mean_difference": round(statistics.mean(c_cider) - statistics.mean(b_cider), 2),
              "welch_t_cider": round((statistics.mean(c_cider) - statistics.mean(b_cider)) / se, 1),
              "every_c_seed_above_every_b_seed": min(c_cider) > max(b_cider),
              "exact_permutation_p_one_sided": separation_p_value(b_cider, c_cider)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("B", "C")}, indent=2))
    print(json.dumps({"B": report["B"]["summary"], "C": report["C"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
