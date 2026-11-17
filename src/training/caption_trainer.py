"""Shared token-masked training, validation and epoch-boundary checkpoint recovery."""

import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.vietnamese_caption_batch import BOS, EOS, PAD
from src.evaluation.caption_scorer import score_captions
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder
from src.models.transformer_decoder import TransformerCaptionDecoder


def build_model(model_name, config):
    models = {"A": LSTMAttentionCaptionDecoder, "B": TransformerCaptionDecoder}
    return models[model_name](**config)


def logits_from(output):
    return output["logits"] if isinstance(output, dict) else output


def token_loss(logits, labels):
    """Sum CE only over non-PAD targets and divide by their count, including EOS."""
    count = labels.ne(PAD).sum()
    if count == 0:
        raise ValueError("A training/evaluation batch must contain a non-PAD target")
    total = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=PAD, reduction="sum")
    return total / count, count


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def cpu_snapshot(value):
    """Detach checkpoint tensors from live parameters and optimizer buffers."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, map_location="cpu"):
    """Read checkpoints, including older metric fields stored as NumPy scalars."""
    numpy_scalar = type(np.float64(0))
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype("float64")), numpy_scalar]):
        return torch.load(path, map_location=map_location, weights_only=True)


def make_loader(dataset, batch_size, epoch, seed, shuffle):
    # Epoch-derived ordering permits replay from a completed epoch without
    # depending on the lifetime of a DataLoader or its internal iterator.
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, collate_fn=dataset.collate)


def run_epoch(model, loader, device, optimizer=None, clip_norm=5.0):
    device = torch.device(device)
    model.train(optimizer is not None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_loss, total_tokens, batches = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            regions = batch["regions"].to(device)
            inputs, labels = batch["decoder_input"].to(device), batch["labels"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = logits_from(model(regions, inputs))
            loss, count = token_loss(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite token loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
                optimizer.step()
            tokens = count.item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            batches += 1
    if total_tokens == 0:
        raise ValueError("Epoch contains no valid target tokens")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"loss": total_loss / total_tokens, "tokens": total_tokens, "batches": batches,
            "seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}


@torch.no_grad()
def generate_tokens(model, regions, max_len):
    model.eval()
    if isinstance(model, LSTMAttentionCaptionDecoder):
        return model.generate(regions, max_len)["token_ids"]
    prefix = torch.full((regions.shape[0], 1), BOS, dtype=torch.long, device=regions.device)
    finished = torch.zeros(regions.shape[0], dtype=torch.bool, device=regions.device)
    for _ in range(max_len):
        scores = logits_from(model(regions, prefix))[:, -1].clone()
        scores[:, [PAD, BOS]] = float("-inf")
        token = scores.argmax(-1).masked_fill(finished, PAD)
        prefix = torch.cat((prefix, token[:, None]), dim=1)
        finished |= token.eq(EOS)
        if finished.all():
            break
    return prefix[:, 1:]


@torch.no_grad()
def evaluate_captions(model, dataset, batch_size, device, max_len):
    predictions = []
    for start in range(0, len(dataset.rows), batch_size):
        ids = [row["image_id"] for row in dataset.rows[start:start + batch_size]]
        regions = dataset.store.get_batch(ids).to(device)
        tokens = generate_tokens(model, regions, max_len)
        predictions.extend((image_id, dataset.vocabulary.decode(sentence)) for image_id, sentence in zip(ids, tokens.tolist()))
    references = {row["image_id"]: row["captions"] for row in dataset.rows}
    result = score_captions(predictions, references, allow_empty=True)
    return result, [{"image_id": image_id, "caption": caption} for image_id, caption in predictions]


def fit(model_name, config, train_data, val_data, output_dir, settings, epochs, device="cpu", resume=None):
    """Train to a total epoch count. Resume replays any interrupted, uncommitted epoch.

    last.pt is the authoritative transaction and embeds the best snapshot.
    best.pt/history.json are derived files, repaired from last.pt on resume.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if epochs <= 0 or settings["batch_size"] <= 0 or settings["learning_rate"] <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if settings["weight_decay"] < 0 or settings.get("patience", 0) < 0:
        raise ValueError("weight_decay and patience must be nonnegative")
    if settings["selection_metric"] not in ("val_loss", "cider") or settings["max_len"] <= 0:
        raise ValueError("Invalid selection_metric or max_len")
    if model_name == "B":
        longest = max(len(tokens) + 1 for _, tokens in train_data.samples + val_data.samples)
        if max(longest, settings["max_len"]) > config["max_len"]:
            raise ValueError("Transformer max_len is shorter than the captions or generation limit")
    seed_everything(settings["seed"])
    model = build_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    data_signature = {"train": train_data.signature(), "val": val_data.signature()}
    history, best, start_epoch, global_step = [], None, 1, 0
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        expected = {"model_name": model_name, "config": config, "settings": settings, "data_signature": data_signature}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Resume {key} differs from the saved checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history, global_step = checkpoint["history"], checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint.get("best_checkpoint", checkpoint)
        atomic_save(best, output_dir / "best.pt")
        restore_rng(checkpoint["rng_state"])
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    elif (output_dir / "last.pt").exists():
        raise ValueError("Run already has last.pt; use --resume or a new output directory")

    for epoch in range(start_epoch, epochs + 1):
        train_stats = run_epoch(model, make_loader(train_data, settings["batch_size"], epoch, settings["seed"], True), device, optimizer)
        val_stats = run_epoch(model, make_loader(val_data, settings["batch_size"], epoch, settings["seed"], False), device)
        metric = -val_stats["loss"]
        scores, predictions = None, None
        caption_started = time.perf_counter()
        if settings["selection_metric"] == "cider":
            scores, predictions = evaluate_captions(model, val_data, settings["batch_size"], device, settings["max_len"])
            metric = scores["scores"]["CIDEr"]
        record = {"epoch": epoch, "train": train_stats, "validation": val_stats,
                  "caption_seconds": time.perf_counter() - caption_started,
                  "scores": scores, "selection_value": metric}
        history.append(record)
        global_step += train_stats["batches"]
        snapshot = {"format_version": 1, "model_name": model_name, "config": config, "settings": settings,
                    "vocabulary": train_data.vocabulary.tokens, "data_signature": data_signature,
                    "model_state": cpu_snapshot(model.state_dict()), "optimizer_state": cpu_snapshot(optimizer.state_dict()),
                    "epoch": epoch, "global_step": global_step, "selection_value": metric,
                    "rng_state": rng_state(), "history": list(history), "purpose": settings["purpose"],
                    "validation_scores": scores, "validation_predictions": predictions,
                    "torch": str(torch.__version__), "device": str(device)}
        improved = best is None or metric > best["selection_value"]
        if improved:
            best = snapshot
        atomic_save({**snapshot, "best_checkpoint": best}, output_dir / "last.pt")
        if improved:
            atomic_save(best, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if predictions is not None:
            (output_dir / "val_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"epoch {epoch}/{epochs} train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
              f"{settings['selection_metric']}={metric:.6f} best_epoch={best['epoch']}", flush=True)
        if settings.get("patience", 0) and epoch - best["epoch"] >= settings["patience"]:
            print(f"Early stopping after {settings['patience']} epochs without improvement", flush=True)
            break
    return model, history
