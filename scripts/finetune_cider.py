"""Self-critical (CIDEr-D) fine-tuning of an A checkpoint on cached ResNet regions.

Each step samples K captions per image, rewards each with CIDEr-D against the train references (document frequencies
from train), and uses the mean reward of the other K-1 samples of the same image as the baseline. Every epoch the model
is scored on val with beam search; the best epoch by val CIDEr is saved as best.pt. Test is never read.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from src.data.paths import resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import BOS, EOS, PAD, CaptionVocabulary, normalize_caption
from src.evaluation.caption_scorer import load_references, score_captions
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder
from src.training.caption_trainer import atomic_save, cpu_snapshot, load_checkpoint
from src.training.cider_reward import CiderD


def sample_captions(model, regions, samples, max_len):
    """Sample `samples` captions per image with gradients: token ids [B*K,T], log-probs [B*K,T], mask [B*K,T]."""
    regions = regions.repeat_interleave(samples, dim=0)
    memory, keys, state = model._prepare(regions)
    token = torch.full((regions.shape[0],), BOS, dtype=torch.long, device=regions.device)
    finished = torch.zeros_like(token, dtype=torch.bool)
    tokens, log_probs, masks = [], [], []
    for _ in range(max_len):
        logits, _, state = model._step(token, memory, keys, state)
        logits = logits.clone()
        logits[:, [PAD, BOS]] = float("-inf")
        distribution = Categorical(logits=logits)
        token = distribution.sample().masked_fill(finished, PAD)
        log_probs.append(distribution.log_prob(token))
        masks.append(~finished)
        tokens.append(token)
        finished = finished | token.eq(EOS)
        if finished.all():
            break
    return torch.stack(tokens, 1), torch.stack(log_probs, 1), torch.stack(masks, 1)


def decode(vocabulary, token_ids, mask):
    return [vocabulary.decode(row[:int(length)].cpu()) for row, length in zip(token_ids, mask.sum(dim=1))]


@torch.no_grad()
def beam_predictions(model, store, ids, vocabulary, device, beam_size, length_penalty, max_len, batch_size=64):
    predictions = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        output = model.generate_beam(store.get_batch(batch).to(device), beam_size, max_len, length_penalty)
        for row, image_id in enumerate(batch):
            length = int(output["lengths"][row])
            predictions.append((image_id, vocabulary.decode(output["token_ids"][row, :length].cpu())))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A checkpoint to start from (best.pt)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="images per step")
    parser.add_argument("--samples", type=int, default=5, help="captions sampled per image")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--length-penalty", type=float, default=0.7)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2 (the baseline is the mean of the other samples)")
    if (args.output / "best.pt").exists():
        parser.error(f"{args.output} already has best.pt; use a new output directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    base = load_checkpoint(args.checkpoint)
    if base["model_name"] != "A" or base["vocabulary"] != vocabulary.tokens:
        raise ValueError("Start from an A checkpoint with the processed vocabulary")
    device = torch.device(args.device)
    model = LSTMAttentionCaptionDecoder(**base["config"]).to(device)
    model.load_state_dict(base["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_rows = read_manifest(processed / "train_manifest.jsonl")
    train_store = RegionFeatureStore.load(args.features, "train")
    val_store = RegionFeatureStore.load(args.features, "val")
    val_ids = [row["image_id"] for row in read_manifest(processed / "val_manifest.jsonl")]
    val_refs = load_references(processed / "val_manifest.jsonl")
    train_refs = {row["image_id"]: [normalize_caption(c) for c in row["captions"]] for row in train_rows}
    reward = CiderD(train_refs)
    prepared = {image_id: reward.prepare(captions) for image_id, captions in train_refs.items()}
    train_ids = [row["image_id"] for row in train_rows]

    def evaluate():
        predictions = beam_predictions(model, val_store, val_ids, vocabulary, device, args.beam_size,
                                       args.length_penalty, args.max_len)
        return score_captions(predictions, val_refs), predictions

    best_scores, best_predictions = evaluate()
    best_value, best_epoch = best_scores["scores"]["CIDEr"], 0
    history = [{"epoch": 0, "val_cider": best_value}]
    print(f"epoch 0 (start) val CIDEr (beam {args.beam_size}) = {best_value:.2f}", flush=True)
    best_state = cpu_snapshot(model.state_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = random.sample(train_ids, len(train_ids))
        reward_sum, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            regions = train_store.get_batch(batch).to(device)
            tokens, log_probs, mask = sample_captions(model, regions, args.samples, args.max_len)
            captions = decode(vocabulary, tokens, mask)
            rewards = torch.tensor([reward.score(normalize_caption(caption), prepared[batch[i // args.samples]])
                                    for i, caption in enumerate(captions)], device=device)
            rewards = rewards.reshape(len(batch), args.samples)
            baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (args.samples - 1)
            advantage = (rewards - baseline).reshape(-1, 1)
            loss = -(advantage * log_probs * mask).sum() / mask.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            reward_sum += float(rewards.sum())
            count += rewards.numel()
        scores, predictions = evaluate()
        value = scores["scores"]["CIDEr"]
        history.append({"epoch": epoch, "train_mean_reward": reward_sum / count, "val_cider": value})
        if value > best_value:
            best_value, best_epoch, best_predictions, best_scores = value, epoch, predictions, scores
            best_state = cpu_snapshot(model.state_dict())
        print(f"epoch {epoch}/{args.epochs} train reward={reward_sum / count:.3f} val CIDEr={value:.2f} "
              f"best={best_value:.2f}@{best_epoch} elapsed={time.perf_counter() - started:.0f}s", flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement", flush=True)
            break
    settings = {**base["settings"], "scst": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                              "processed_dir": str(processed)}}
    snapshot = {**{key: base[key] for key in ("format_version", "model_name", "config", "vocabulary", "data_signature")},
                "settings": settings, "model_state": best_state, "epoch": best_epoch, "selection_value": best_value,
                "history": history, "purpose": "scst_cider_finetune_from_" + str(base.get("purpose", "A")),
                "validation_scores": best_scores,
                "validation_predictions": [{"image_id": i, "caption": c} for i, c in best_predictions],
                "decode": {"beam_size": args.beam_size, "length_penalty": args.length_penalty},
                "start_checkpoint_epoch": base["epoch"], "torch": str(torch.__version__), "device": str(device)}
    atomic_save(snapshot, args.output / "best.pt")
    print(f"Saved {args.output / 'best.pt'}: val CIDEr {best_value:.2f} at SCST epoch {best_epoch} "
          f"(start {history[0]['val_cider']:.2f})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
