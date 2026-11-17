"""Overfit 16 training image-caption pairs and verify LSTM generation, attention and reload."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.data.paths import resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import BOS, EOS, PAD, CaptionVocabulary, normalize_caption
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_attention.yaml"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--output", type=Path, default=Path("runs/lstm_attention_check"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        parser.error("steps and learning-rate must be positive")

    torch.manual_seed(args.seed)
    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    rows = random.Random(args.seed).sample(read_manifest(processed / "train_manifest.jsonl"), 16)
    store = RegionFeatureStore.load(args.features, "train")
    image_ids = [row["image_id"] for row in rows]
    regions = store.get_batch(image_ids).to(args.device)
    encoded = [vocabulary.encode(row["captions"][0]) for row in rows]
    inputs = pad_sequence([torch.tensor([BOS, *tokens]) for tokens in encoded], batch_first=True, padding_value=PAD).to(args.device)
    labels = pad_sequence([torch.tensor([*tokens, EOS]) for tokens in encoded], batch_first=True, padding_value=PAD).to(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["vocab_size"] != len(vocabulary.tokens):
        raise ValueError("Configured vocab_size differs from the saved vocabulary")
    config["dropout"] = 0.0  # Disable regularization for this memorization check.
    model = LSTMAttentionCaptionDecoder(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    history = []
    started = time.perf_counter()
    epoch_seconds = []
    if regions.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(regions, inputs)
        loss = F.cross_entropy(output["logits"].flatten(0, 1), labels.flatten(), ignore_index=PAD)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        history.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if regions.is_cuda:
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - step_started)
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step {step + 1}/{args.steps}, loss={loss.item():.6f}", flush=True)

    train_peak_allocated = torch.cuda.max_memory_allocated() / 2**20 if regions.is_cuda else None
    train_peak_reserved = torch.cuda.max_memory_reserved() / 2**20 if regions.is_cuda else None
    model.eval()
    with torch.no_grad():
        teacher = model(regions, inputs)
        final_loss = F.cross_entropy(teacher["logits"].flatten(0, 1), labels.flatten(), ignore_index=PAD).item()
    generated = model.generate(regions, max_len=labels.shape[1] + 5)
    predictions = [vocabulary.decode(tokens) for tokens in generated["token_ids"].tolist()]
    references = [normalize_caption(row["captions"][0]) for row in rows]
    exact_matches = sum(a == b for a, b in zip(predictions, references))
    active = torch.arange(generated["token_ids"].shape[1], device=regions.device)[None] < generated["lengths"][:, None]
    attention_error = (generated["attention"].sum(-1)[active] - 1).abs().max().item()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": config, "vocabulary": vocabulary.tokens,
                "seed": args.seed, "steps": args.steps, "purpose": "overfit_16_train_pairs",
                "image_ids": image_ids, "feature_meta": store.meta}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    restored = LSTMAttentionCaptionDecoder(**checkpoint["config"]).to(args.device).eval()
    restored.load_state_dict(checkpoint["model_state"])
    reloaded = restored.generate(regions, max_len=labels.shape[1] + 5)
    reload_equal = all(torch.equal(generated[key], reloaded[key]) for key in generated)
    torch.save({"image_ids": torch.tensor(image_ids), "teacher_logits": teacher["logits"].cpu(),
                **{key: value.cpu() for key, value in generated.items()}}, args.output / "outputs.pt")
    report = {
        "purpose": "Implementation check on 16 training pairs", "seed": args.seed,
        "config": config, "steps": args.steps, "learning_rate": args.learning_rate,
        "parameters": sum(p.numel() for p in model.parameters()), "device": args.device,
        "torch": str(torch.__version__), "seconds": round(time.perf_counter() - started, 2),
        "epoch_seconds": epoch_seconds,
        "epoch_definition": "One update over all 16 pairs, no checkpoint IO in the timer",
        "peak_allocated_mib": train_peak_allocated,
        "peak_reserved_mib": train_peak_reserved,
        "initial_loss": history[0], "final_loss": final_loss,
        "exact_captions": exact_matches, "images": len(rows),
        "eos_count": generated["finished"].sum().item(), "attention_sum_max_error": attention_error,
        "checkpoint_reload_equal": reload_equal,
        "teacher_logits_shape": list(teacher["logits"].shape),
        "generated_attention_shape": list(generated["attention"].shape),
        "manifest_sha256": hashlib.sha256((processed / "train_manifest.jsonl").read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path), "loss_history": history,
        "examples": [{"image_id": row["image_id"], "image_path": row["image_path"],
                      "reference": reference, "caption": caption}
                     for row, reference, caption in zip(rows, references, predictions)],
    }
    report["passed"] = (final_loss < history[0] * 0.1 and exact_matches == len(rows)
                        and generated["finished"].all().item() and attention_error < 1e-5 and reload_equal)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("loss_history", "epoch_seconds", "examples")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
