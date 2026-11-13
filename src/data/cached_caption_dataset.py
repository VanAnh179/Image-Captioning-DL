"""Caption pairs and batching over the shared, ID-keyed image feature cache."""

import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import BOS, EOS, PAD


class CachedCaptionDataset(Dataset):
    def __init__(self, rows, vocabulary, store, first_caption_only=False):
        self.rows, self.vocabulary, self.store = rows, vocabulary, store
        self.samples = []
        for row in rows:
            captions = row["captions"][:1] if first_caption_only else row["captions"]
            for caption in captions:
                self.samples.append((row["image_id"], vocabulary.encode(caption)))
        if not self.samples:
            raise ValueError("Caption dataset must contain at least one pair")

    @classmethod
    def load(cls, processed, features, split, vocabulary):
        manifest = Path(processed) / f"{split}_manifest.jsonl"
        rows = read_manifest(manifest)
        store = RegionFeatureStore.load(features, split)
        ids = json.loads((Path(processed) / f"{split}_ids.json").read_text(encoding="utf-8"))
        if [row["image_id"] for row in rows] != ids or store.image_ids.tolist() != ids:
            raise ValueError(f"{split}: manifest, split IDs and feature cache disagree")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if store.meta.get("manifest_sha256") != digest:
            raise ValueError(f"{split}: feature cache was built from a different manifest")
        return cls(rows, vocabulary, store)

    def signature(self):
        content = {"samples": self.samples, "vocabulary": self.vocabulary.tokens, "feature_meta": self.store.meta}
        return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def collate(self, samples):
        ids = [image_id for image_id, _ in samples]
        inputs = [torch.tensor([BOS, *tokens]) for _, tokens in samples]
        labels = [torch.tensor([*tokens, EOS]) for _, tokens in samples]
        return {"image_ids": ids, "regions": self.store.get_batch(ids),
                "decoder_input": pad_sequence(inputs, batch_first=True, padding_value=PAD),
                "labels": pad_sequence(labels, batch_first=True, padding_value=PAD)}
