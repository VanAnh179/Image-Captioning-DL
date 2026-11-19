"""Kaggle entry: train models C (CLIP regions) and B (ResNet-50 regions) with the agreed budget on the same T4.

Same settings as the local runs: batch 32, lr 3e-4, weight decay 0.01, CIDEr selection, patience 4, max 20 epochs.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUNS = [("C", 42), ("B", 42), ("C", 1), ("B", 1), ("C", 2), ("B", 2), ("C", 3), ("B", 3), ("C", 4), ("B", 4)]
SETTINGS = {"B": ("configs/transformer_decoder.yaml", []),
            "C": ("configs/clip_transformer.yaml", ["--feature-prefix", "clip_vit_b32_7x7"])}
KEEP_CHECKPOINT_SEED = 42


def run(command):
    subprocess.run([sys.executable, *command], check=True)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONUTF8"] = "1"
input_root = Path("/kaggle/input")
source_root = next(input_root.rglob("clip_transformer.yaml")).parents[1]
work = Path("/kaggle/working/project")
shutil.copytree(source_root, work, dirs_exist_ok=True)
features = work / "artifacts/features/ktvic"
features.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    source = next(input_root.rglob(f"resnet50_7x7_{split}.pt"))
    (features / source.name).symlink_to(source)
os.chdir(work)
sys.path.insert(0, str(work))
run(["-m", "pip", "install", "--quiet", "pycocoevalcap==1.2", "PyYAML==6.0.3"])
import torch

if not torch.cuda.is_available():
    raise RuntimeError("This run requires a Kaggle GPU")
gpu = torch.cuda.get_device_name(0)
print("gpu", gpu, "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
output = Path("/kaggle/working/results")
output.mkdir(parents=True, exist_ok=True)
summary = []
for model, seed in RUNS:
    name = f"{model.lower()}_kaggle_seed{seed}"
    config, extra = SETTINGS[model]
    run_dir = Path("runs") / name
    started = time.perf_counter()
    try:
        run(["-u", "-m", "scripts.train_caption_model", "--model", "B", "--config", config, "--output", str(run_dir),
             "--epochs", "20", "--batch-size", "32", "--learning-rate", "0.0003", "--weight-decay", "0.01",
             "--selection-metric", "cider", "--patience", "4", "--seed", str(seed), "--max-len", "40",
             "--device", "cuda", *extra])
        run(["-m", "scripts.export_caption_results", "--run-dir", str(run_dir)])
        info = {"model": model, "seed": seed, "gpu": gpu, "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                "wall_seconds": time.perf_counter() - started, "config": config, "status": "ok"}
    except Exception:
        traceback.print_exc()
        info = {"model": model, "seed": seed, "status": "failed", "wall_seconds": time.perf_counter() - started}
    summary.append(info)
    target = output / name
    if run_dir.exists():
        target.mkdir(parents=True, exist_ok=True)
        (run_dir / "runtime.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        keep = ["metrics.json", "history.json", "val_predictions.json", "loss_curve.png", "runtime.json"]
        if seed == KEEP_CHECKPOINT_SEED:
            keep.append("best.pt")
        for file in keep:
            if (run_dir / file).exists():
                shutil.copy2(run_dir / file, target / file)
        shutil.rmtree(run_dir, ignore_errors=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
shutil.rmtree(work, ignore_errors=True)
print(json.dumps(summary), flush=True)
if any(row["status"] != "ok" for row in summary):
    raise SystemExit("at least one run failed")
