"""Run one batch of KTVIC images end to end: image file -> CLIP preprocessing -> frozen CLIP -> Transformer -> Vietnamese words.

The decoder is not trained, so its first captions are meaningless on purpose. The checks prove the plumbing:
shapes, preprocessing, ID traceability, batch independence, frozen encoder, gradients. A final tiny overfit on the same
images shows the CLIP regions really reach the decoder (it can then reproduce a reference sentence per image).
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

from src.data.clip_features import CLIP_CACHE_PREFIX, CLIP_FEATURE_DIM, FrozenCLIPVisionGrid, load_clip_image, load_clip_processor
from src.data.paths import resolve_image_root, resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import BOS, EOS, PAD, CaptionVocabulary
from src.models.transformer_decoder import TransformerCaptionDecoder
from src.training.caption_trainer import generate_tokens


def manual_clip_preprocess(path, processor):
    """Independent re-implementation of the CLIP recipe: PIL bicubic resize of the short side to 224 (long side
    truncated with int(), as in OpenAI CLIP), centre crop with floor offsets, rescale, mean/std."""
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    size = processor.size["shortest_edge"]
    width, height = rgb.size
    new_width, new_height = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    rgb = rgb.resize((new_width, new_height), Image.Resampling.BICUBIC)
    left, top = (new_width - size) // 2, (new_height - size) // 2
    rgb = rgb.crop((left, top, left + size, top + size))
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(size, size, 3).permute(2, 0, 1).float() / 255
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    return (pixels - mean) / std


def teacher_forcing_batch(captions, vocabulary, device):
    tokens = [vocabulary.encode(text) for text in captions]
    inputs = pad_sequence([torch.tensor([BOS, *t]) for t in tokens], batch_first=True, padding_value=PAD)
    labels = pad_sequence([torch.tensor([*t, EOS]) for t in tokens], batch_first=True, padding_value=PAD)
    return inputs.to(device), labels.to(device)


def validate_pipeline_report(report):
    """Return failed checks so a diagnostic cannot silently succeed with broken features or gradients."""
    checks, count = report["checks"], report["num_images"]
    errors = []
    for name in ("pixel_hash_matches_manifest", "regions_finite", "encoder_in_eval_mode_after_train_call",
                 "nearest_cached_id_equals_input_id", "trace_pixel_hash_equals_manifest", "encoder_gradients_none"):
        if checks.get(name) is not True:
            errors.append(f"{name}: expected True, got {checks.get(name)!r}")
    for name, expected in (("pixel_shape", [count, 3, 224, 224]),
                           ("regions_shape", [count, 49, CLIP_FEATURE_DIM]),
                           ("encoder_trainable_parameters", 0)):
        if checks.get(name) != expected:
            errors.append(f"{name}: expected {expected}, got {checks.get(name)!r}")
    for name, tolerance in (("preprocess_max_abs_diff_vs_manual", 0.05),
                            ("batch_vs_single_image_max_abs_diff", 1e-3),
                            ("live_vs_cached_max_abs_diff", 0.01),
                            ("logits_batch_vs_single_max_abs_diff", 1e-4)):
        value = checks.get(name)
        if value is None or not math.isfinite(value) or not 0 <= value <= tolerance:
            errors.append(f"{name}: expected finite value in [0, {tolerance}], got {value!r}")
    for name in ("initial_loss", "projection_gradient_norm", "untrained_logits_depend_on_image_max_diff"):
        value = checks.get(name)
        if value is None or not math.isfinite(value) or value <= 0:
            errors.append(f"{name}: expected finite positive value, got {value!r}")
    shape = checks.get("logits_shape", [])
    if len(shape) != 3 or shape[0] != count or shape[1] <= 0 or shape[2] != report["config"]["vocab_size"]:
        errors.append(f"logits_shape: incompatible with batch/vocabulary: {shape}")
    if checks.get("overfit_steps", 0) > 0:
        value = checks.get("overfit_final_loss")
        if value is None or not math.isfinite(value) or value > 0.1:
            errors.append(f"overfit_final_loss: expected <= 0.1, got {value!r}")
        if checks.get("overfit_exact_matches") != count:
            errors.append(f"overfit_exact_matches: expected {count}, got {checks.get('overfit_exact_matches')!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--config", type=Path, default=Path("configs/clip_transformer.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/clip_encoder/clip_pipeline_check.json"))
    parser.add_argument("--overfit-steps", type=int, default=300, help="Tiny memorisation check; 0 skips training")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    checks = {}

    processed, image_root = resolve_processed_dir(), resolve_image_root()
    rows = read_manifest(processed / f"{args.split}_manifest.jsonl")
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["vocab_size"] != len(vocabulary.tokens):
        raise ValueError(f"vocab_size {config['vocab_size']} khác từ vựng {len(vocabulary.tokens)}")
    if not 2 <= args.num_images <= len(rows):
        parser.error("num-images must be between 2 and the number of images in the split")
    if args.overfit_steps < 0 or not 1 <= args.max_len <= config["max_len"]:
        parser.error("overfit-steps must be >= 0 and max-len must fit the decoder configuration")
    step = len(rows) // args.num_images  # evenly spaced images so sizes and scenes differ
    picked = [rows[i * step] for i in range(args.num_images)]
    ids = [row["image_id"] for row in picked]

    # 1. image file -> CLIP preprocessing, with the manifest pixel hash and an independent re-implementation.
    processor = load_clip_processor()
    loaded = [load_clip_image(image_root / row["image_path"], processor) for row in picked]
    pixels = torch.stack([tensor for tensor, _ in loaded])
    checks["pixel_hash_matches_manifest"] = all(d == row["pixel_sha256"] for (_, d), row in zip(loaded, picked))
    manual = torch.stack([manual_clip_preprocess(image_root / row["image_path"], processor) for row in picked])
    checks["preprocess_max_abs_diff_vs_manual"] = (pixels - manual).abs().max().item()  # one 8-bit step is about 0.015
    checks["preprocess_mean_abs_diff_vs_manual"] = (pixels - manual).abs().mean().item()
    checks["pixel_shape"] = list(pixels.shape)
    sizes = []
    for row in picked:
        with Image.open(image_root / row["image_path"]) as image:
            sizes.append(list(image.size))

    # 2. frozen CLIP -> 49 regions of 768; the live batch must equal the cached features and each image alone.
    encoder = FrozenCLIPVisionGrid().to(device)
    regions = encoder(pixels.to(device)).float().cpu()
    checks["regions_shape"] = list(regions.shape)
    checks["regions_finite"] = bool(torch.isfinite(regions).all())
    checks["encoder_trainable_parameters"] = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    checks["encoder_in_eval_mode_after_train_call"] = not encoder.train().training
    single = torch.cat([encoder(pixels[i:i + 1].to(device)).float().cpu() for i in range(len(picked))])
    checks["regions_abs_max"] = regions.abs().max().item()
    checks["batch_vs_single_image_max_abs_diff"] = (regions - single).abs().max().item()
    checks["batch_vs_single_image_mean_abs_diff"] = (regions - single).abs().mean().item()
    checks["regions_mean_abs"] = regions.abs().mean().item()
    flat = regions.flatten(1)
    distance = torch.cdist(flat, flat)
    checks["min_distance_between_different_images"] = distance[~torch.eye(len(ids), dtype=torch.bool)].min().item()
    store = RegionFeatureStore.load(args.features, args.split, CLIP_CACHE_PREFIX, feature_dim=CLIP_FEATURE_DIM)
    cached = store.get_batch(ids)
    checks["live_vs_cached_max_abs_diff"] = (regions - cached).abs().max().item()
    traced = [store.nearest_id(region.to(store.features.dtype)) for region in regions]
    checks["nearest_cached_id_equals_input_id"] = [t[0] for t in traced] == ids
    checks["trace_pixel_hash_equals_manifest"] = all(
        store.trace(i)["pixel_sha256"] == row["pixel_sha256"] for i, row in zip(ids, picked))

    # 3. Transformer (random weights, same architecture as B except feature_dim) -> words.
    torch.manual_seed(args.seed)
    decoder = TransformerCaptionDecoder(**config).to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    inputs, labels = teacher_forcing_batch([row["captions"][0] for row in picked], vocabulary, device)
    decoder.eval()
    with torch.no_grad():
        logits = decoder(regions.to(device), inputs).cpu()
        singles = torch.cat([decoder(regions[i:i + 1].to(device), inputs[i:i + 1]).cpu() for i in range(len(ids))])
    real = inputs.cpu().ne(PAD)  # positions of a shorter sentence that are PAD padding are not compared
    checks["logits_shape"] = list(logits.shape)
    checks["logits_batch_vs_single_max_abs_diff"] = (logits - singles).abs().amax(-1)[real].max().item()
    first_step = logits[:, 0]  # same BOS prefix for every image, so any difference comes from the regions
    checks["untrained_logits_depend_on_image_max_diff"] = (first_step - first_step[0]).abs().max().item()
    untrained = generate_tokens(decoder, regions.to(device), args.max_len).cpu()
    untrained_captions = [vocabulary.decode(row) for row in untrained.tolist()]

    # 4. gradients: the loss flows into the new projection and none reaches CLIP.
    decoder.train()
    loss = torch.nn.functional.cross_entropy(decoder(regions.to(device), inputs).flatten(0, 1), labels.flatten(), ignore_index=PAD)
    loss.backward()
    checks["initial_loss"] = loss.item()
    checks["ln_vocab_size"] = float(torch.log(torch.tensor(float(len(vocabulary.tokens)))))
    checks["projection_gradient_norm"] = decoder.regions.linear.weight.grad.norm().item()
    checks["encoder_gradients_none"] = all(p.grad is None for p in encoder.parameters())

    # 5. tiny overfit on these images only, to show the regions carry information (not a training result).
    references = [row["captions"][0] for row in picked]
    overfit = [None] * len(ids)
    checks.update(overfit_steps=args.overfit_steps, overfit_final_loss=None,
                  overfit_exact_matches=None, overfit_distinct_captions=None)
    if args.overfit_steps:
        torch.manual_seed(args.seed)
        tiny = TransformerCaptionDecoder(**{**config, "dropout": 0.0}).to(device)
        optimizer = torch.optim.AdamW(tiny.parameters(), lr=1e-3, weight_decay=0.0)
        gpu_regions = regions.to(device)
        for _ in range(args.overfit_steps):
            tiny.train()
            optimizer.zero_grad()
            value = torch.nn.functional.cross_entropy(tiny(gpu_regions, inputs).flatten(0, 1), labels.flatten(), ignore_index=PAD)
            value.backward()
            optimizer.step()
        overfit = [vocabulary.decode(row) for row in generate_tokens(tiny, gpu_regions, args.max_len).cpu().tolist()]
        checks["overfit_final_loss"] = value.item()
        checks["overfit_exact_matches"] = sum(a == b for a, b in zip(overfit, references))
        checks["overfit_distinct_captions"] = len(set(overfit))

    report = {
        "split": args.split, "num_images": len(ids), "decoder_parameters": parameters, "config": config,
        "checks": checks,
        "images": [{
            "image_id": row["image_id"], "image_path": row["image_path"], "original_size_wh": size,
            "caption_untrained_decoder": untrained_captions[i], "caption_after_tiny_overfit": overfit[i],
            "reference_caption_used": references[i],
        } for i, (row, size) in enumerate(zip(picked, sizes))],
    }
    report["errors"] = validate_pipeline_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("decoder_parameters:", parameters)
    for item in report["images"]:
        print(f"[{item['image_id']}] {item['original_size_wh']} chưa train: {item['caption_untrained_decoder']!r}\n"
              f"      sau overfit: {item['caption_after_tiny_overfit']!r}\n      tham chiếu : {item['reference_caption_used']!r}")
    print(f"\nĐã ghi {args.output}")
    if report["errors"]:
        raise SystemExit("CLIP pipeline check failed:\n  " + "\n  ".join(report["errors"]))


if __name__ == "__main__":
    main()
