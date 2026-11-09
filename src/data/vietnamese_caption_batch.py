"""Text and image inputs shared by the KTVIC caption decoders."""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD, BOS, EOS, UNK = range(4)


def normalize_caption(text):
    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())


class CaptionVocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        if tuple(self.tokens[:4]) != SPECIAL_TOKENS or len(self.ids) != len(self.tokens):
            raise ValueError("Invalid special tokens or duplicate vocabulary entries")

    @classmethod
    def from_training_captions(cls, captions):
        counts = Counter(token for text in captions for token in normalize_caption(text).split())
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls([*SPECIAL_TOKENS, *ordered])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def encode(self, caption):
        return [self.ids.get(token, UNK) for token in normalize_caption(caption).split()]

    def decode(self, token_ids):
        words = []
        for token_id in token_ids:
            index = int(token_id)
            if index == EOS:
                break
            if index not in (BOS, PAD):
                words.append(self.tokens[index])
        return " ".join(words)


def make_caption_batch(samples, vocabulary, image_root, image_size=224):
    """Collate (image_id, relative image_path, caption) tuples on CPU."""
    images, inputs, targets, image_ids = [], [], [], []
    for image_id, image_path, caption in samples:
        with Image.open(Path(image_root) / image_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
            pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            images.append(pixels.view(image_size, image_size, 3).permute(2, 0, 1))
        tokens = vocabulary.encode(caption)
        inputs.append(torch.tensor([BOS, *tokens], dtype=torch.long))
        targets.append(torch.tensor([*tokens, EOS], dtype=torch.long))
        image_ids.append(image_id)

    labels = pad_sequence(targets, batch_first=True, padding_value=PAD)
    return {
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "images": torch.stack(images).float().div_(255),
        "decoder_input": pad_sequence(inputs, batch_first=True, padding_value=PAD),
        "labels": labels,
        "valid_token_mask": labels.ne(PAD),
    }
