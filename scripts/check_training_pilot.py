"""Kaggle pilot for shared training recovery, PAD invariance and A/B resource budgets."""

import contextlib
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from src.data.cached_caption_dataset import CachedCaptionDataset
from src.data.vietnamese_caption_batch import PAD, CaptionVocabulary
from src.training.caption_trainer import build_model, fit, logits_from, make_loader, run_epoch, seed_everything, token_loss


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def main():
    output = Path("runs/training_pilot")
    output.mkdir(parents=True, exist_ok=True)
    vocabulary = CaptionVocabulary.load("data/processed/ktvic/caption_vocabulary.json")
    data = CachedCaptionDataset.load("data/processed/ktvic", "artifacts/features/ktvic", "train", vocabulary)
    rows = random.Random(42).sample(data.rows, 16)
    small = CachedCaptionDataset(rows, vocabulary, data.store, first_caption_only=True)
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "models": {}}
    for name, filename in (("A", "lstm_attention.yaml"), ("B", "transformer_decoder.yaml")):
        config = yaml.safe_load((Path("configs") / filename).read_text(encoding="utf-8"))
        settings = {"batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.01,
                    "selection_metric": "val_loss", "max_len": 40, "seed": 42, "purpose": "resume_pilot"}
        fit(name, config, small, small, output / name / "continuous", settings, 3, "cuda")
        fit(name, config, small, small, output / name / "resumed", settings, 1, "cuda")
        fit(name, config, small, small, output / name / "resumed", settings, 3, "cuda", output / name / "resumed/last.pt")
        continuous = torch.load(output / name / "continuous/last.pt", weights_only=True, map_location="cpu")
        resumed = torch.load(output / name / "resumed/last.pt", weights_only=True, map_location="cpu")
        resume_equal = all(equal_state(continuous[key], resumed[key]) for key in ("model_state", "optimizer_state", "rng_state", "epoch", "global_step"))
        del continuous, resumed
        seed_everything(42)
        model = build_model(name, config).cuda().eval()
        batch = small.collate([small[i] for i in range(16)])
        regions, inputs, labels = (batch[key].cuda() for key in ("regions", "decoder_input", "labels"))
        with torch.no_grad():
            loss, _ = token_loss(logits_from(model(regions, inputs)), labels)
            padded_loss, _ = token_loss(logits_from(model(regions, F.pad(inputs, (0, 5), value=PAD))), F.pad(labels, (0, 5), value=PAD))
        pad_difference = abs(loss.item() - padded_loss.item())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        run_epoch(model, itertools.islice(make_loader(data, 32, 1, 42, True), 1), "cuda", optimizer)
        stats = run_epoch(model, itertools.islice(make_loader(data, 32, 2, 42, True), 20), "cuda", optimizer)
        seconds_per_batch = stats["seconds"] / stats["batches"]
        report["models"][name] = {"config": config, "resume_equal": resume_equal, "padding_loss_difference": pad_difference,
                                    "benchmark": stats, "estimated_train_epoch_seconds": seconds_per_batch * math.ceil(len(data) / 32),
                                    "parameters": sum(p.numel() for p in model.parameters())}
        del model, optimizer, regions, inputs, labels
        torch.cuda.empty_cache()
    report["passed"] = all(row["resume_equal"] and row["padding_loss_difference"] < 1e-5 for row in report["models"].values())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("Training pilot failed")


if __name__ == "__main__":
    main()
