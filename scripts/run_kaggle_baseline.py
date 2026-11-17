"""Kaggle entry point: validate, run pilots, train A and export its selected validation baseline."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(command):
    subprocess.run([sys.executable, *command], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_attention.yaml"))
    parser.add_argument("--run-name", default="a_baseline")
    args = parser.parse_args(argv)
    if args.config.parent != Path("configs") or args.config.suffix != ".yaml":
        parser.error("config must be a YAML file directly inside configs/")
    if not args.run_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in args.run_name):
        parser.error("run-name must use lowercase English letters, digits or underscores")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    input_root = Path("/kaggle/input")
    source_root = next(input_root.rglob(args.config.name)).parents[1]
    work = Path("/kaggle/working/project")
    shutil.copytree(source_root, work, dirs_exist_ok=True)
    features = work / "artifacts/features/ktvic"
    features.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        source = next(input_root.rglob(f"resnet50_7x7_{split}.pt"))
        (features / source.name).symlink_to(source)
    os.chdir(work)
    sys.path.insert(0, str(work))
    run(["-m", "pip", "install", "--quiet", "pycocoevalcap==1.2", "PyYAML==6.0.3", "Pillow==12.2.0"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires a Kaggle GPU")
    torch.set_num_threads(2)
    started = time.perf_counter()
    run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    run(["-u", "-m", "scripts.check_training_pilot", "--a-config", str(args.config)])
    run(["-u", "-m", "scripts.check_lstm_attention", "--config", str(args.config), "--device", "cuda"])
    run_dir = Path("runs") / args.run_name
    baseline_started = time.perf_counter()
    run(["-u", "-m", "scripts.train_caption_model", "--model", "A", "--config", str(args.config), "--output", str(run_dir),
         "--epochs", "20", "--batch-size", "32", "--learning-rate", "0.0003", "--weight-decay", "0.01",
         "--selection-metric", "cider", "--patience", "4", "--seed", "42", "--max-len", "40", "--device", "cuda"])
    run(["-m", "scripts.export_caption_results", "--run-dir", str(run_dir)])
    summary = {"wall_seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(0),
               "baseline_wall_seconds": time.perf_counter() - baseline_started,
               "torch": str(torch.__version__), "config_path": str(args.config),
               "purpose": "A baseline selected by validation CIDEr"}
    (run_dir / "runtime.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.make_archive(f"/kaggle/working/{args.run_name}_results", "zip", "runs")


if __name__ == "__main__":
    main()
