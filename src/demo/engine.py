"""Offline demo engine: caption a chosen image with model A, and find images from a Vietnamese sentence.

Everything runs from local files: the A checkpoint, the TF-IDF index built from A's captions, the KTVIC images and the
ResNet-50 weights already in the torch cache. Nothing is downloaded.
"""

from pathlib import Path

import numpy as np
import torch

from src.data.region_features import FrozenResNet50Grid, load_image
from src.data.vietnamese_caption_batch import CaptionVocabulary
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder
from src.retrieval.caption_search import CaptionSearchIndex
from src.training.caption_trainer import load_checkpoint

# A Kaggle epoch 5 is the checkpoint chosen for the demo; the first path is where the version list puts it.
DEFAULT_CHECKPOINTS = ("runs/kaggle_matched_inspection/project/runs/a_baseline_matched/best.pt", "best.pt")
DEFAULT_INDEX = "artifacts/retrieval/ktvic_a_val.npz"
TOP_WORDS = 5
# Beam search chosen on val: CIDEr 76.96 (greedy) -> 83.93; beam_size=1 gives the original greedy captions.
DEFAULT_BEAM_SIZE = 3
DEFAULT_LENGTH_PENALTY = 0.7


def first_existing(paths):
    for path in paths:
        if Path(path).is_file():
            return Path(path)
    raise FileNotFoundError("Không thấy checkpoint A. Truyền --checkpoint <đường dẫn best.pt>. Đã thử: " + ", ".join(paths))


def check_index_matches_checkpoint(index, checkpoint):
    """True if the index holds the captions this checkpoint wrote for every validation image (it may hold more images,
    such as test), False if one differs or is missing, None if it cannot be told."""
    saved = checkpoint.get("validation_predictions")
    if not saved:
        return None
    held = {row["image_id"]: row["caption"] for row in index.records}
    return all(held.get(row["image_id"]) == row["caption"] for row in saved)


class DemoEngine:
    def __init__(self, checkpoint_path, index_path, image_root, device="cpu", *, beam_size=DEFAULT_BEAM_SIZE,
                 length_penalty=DEFAULT_LENGTH_PENALTY):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["model_name"] not in ("A", "A0"):
            raise ValueError(f"Demo chỉ nạp checkpoint LSTM A hoặc A0, file này là {checkpoint['model_name']}")
        self.index = CaptionSearchIndex.load(index_path)
        self.index_checked = check_index_matches_checkpoint(self.index, checkpoint)
        if self.index_checked is False:
            raise ValueError(f"Chỉ mục {index_path} không được lập từ caption của checkpoint {checkpoint_path}. "
                             "Dùng chỉ mục ktvic_a_val.npz của A Kaggle (SHA-256 0ea7de4c…) hoặc lập lại bằng "
                             "python -m scripts.build_image_search_index.")
        self.device = torch.device(device)
        self.vocabulary = CaptionVocabulary(checkpoint["vocabulary"])
        self.model = LSTMAttentionCaptionDecoder(**checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        try:
            self.encoder = FrozenResNet50Grid(local_only=True).to(self.device)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Thiếu trọng số ResNet-50 dùng cho demo offline: {error.filename}") from error
        self.image_root = Path(image_root)
        self.checkpoint_epoch = checkpoint["epoch"]
        self.beam_size, self.length_penalty = beam_size, length_penalty

    @torch.no_grad()
    def caption_image(self, path, max_len=40, *, formats=None):
        """Caption of any image file, using the same preprocessing as the evaluation and beam search (1 = greedy)."""
        pixels, _ = load_image(path, formats=formats)
        features = self.encoder(pixels.unsqueeze(0).to(self.device)).to(torch.float16)  # the caches are float16
        if self.beam_size > 1:
            output = self.model.generate_beam(features, self.beam_size, max_len, self.length_penalty)
        else:
            output = self.model.generate(features, max_len=max_len)
        length = int(output["lengths"][0])
        return self.vocabulary.decode(output["token_ids"][0, :length].cpu())

    def search(self, query, k=5):
        """Up to k images with a positive score. Raises SearchInputError for an empty or unknown-word query."""
        hits = self.index.search(query, k, self.image_root)
        query_vector = self.index.vectorize([query], self.index.vocabulary, self.index.idf)[0]
        results = []
        for rank, hit in enumerate(hits, 1):
            row = int(np.flatnonzero(self.index.image_ids == hit["image_id"])[0])
            contribution = query_vector * self.index.vectors[row]
            order = np.argsort(-contribution)[:TOP_WORDS]
            words = [self.index.vocabulary[int(i)] for i in order if contribution[i] > 0]
            results.append({**hit, "rank": rank, "matched_words": words})
        return results
