"""Vietnamese text-to-image search with locally cached MiniLM and projected ResNet features."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.vietnamese_caption_batch import normalize_caption
from src.retrieval.caption_search import SearchInputError

TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
FORMAT_VERSION = 1
DEFAULT_INDEX_DIR = Path("artifacts/retrieval/visual")


class MiniLMEncoder:
    """Encode text using the model's mean pooling and a 128-token limit."""

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        options = {"revision": TEXT_REVISION, "local_files_only": True}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, **options)
            self.model = AutoModel.from_pretrained(TEXT_MODEL, **options, use_safetensors=True)
        except OSError as error:
            raise FileNotFoundError(f"Thiếu bản MiniLM trong cache Hugging Face: {TEXT_MODEL} ({TEXT_REVISION}). "
                                    "Chuẩn bị cache trước khi chạy offline.") from error
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size=128):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = [text.strip() for text in texts[start:start + batch_size]]
            inputs = self.tokenizer(batch, max_length=128, padding=True, truncation=True,
                                    return_tensors="pt").to(self.device)
            tokens = self.model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors.append(F.normalize(pooled.float(), dim=1))
        return torch.cat(vectors)


class VisualSearchIndex:
    """Image vectors in MiniLM space, addressed by image ID."""

    def __init__(self, image_ids, vectors, text_center, split, manifest_sha256):
        if (not torch.is_tensor(image_ids) or image_ids.ndim != 1 or image_ids.dtype != torch.int64
                or not len(image_ids) or len(torch.unique(image_ids)) != len(image_ids)
                or not torch.is_tensor(vectors) or vectors.shape != (len(image_ids), 384)
                or not torch.is_tensor(text_center) or text_center.shape != (384,)):
            raise ValueError("Chỉ mục ảnh có ID hoặc kích thước vector không hợp lệ.")
        if (not torch.isfinite(vectors).all() or not torch.isfinite(text_center).all()
                or split not in ("val", "test") or not isinstance(manifest_sha256, str)
                or len(manifest_sha256) != 64):
            raise ValueError("Chỉ mục ảnh có vector hoặc thông tin tập dữ liệu không hợp lệ.")
        self.image_ids = image_ids.cpu().numpy()
        self.vectors = vectors.float().cpu()
        self.text_center = text_center.float().cpu()
        self.split = split
        self.manifest_sha256 = manifest_sha256

    def scores(self, text_vectors):
        queries = F.normalize(text_vectors.float().cpu() - self.text_center, dim=1)
        return (queries @ self.vectors.T).clamp(-1, 1)

    @classmethod
    def load(cls, path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, EOFError) as error:
            raise ValueError(f"Không đọc được chỉ mục ảnh {path}.") from error
        if (not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION
                or data.get("text_model") != TEXT_MODEL or data.get("text_revision") != TEXT_REVISION):
            raise ValueError(f"Chỉ mục ảnh {path} dùng phiên bản không tương thích.")
        return cls(*(data.get(key) for key in ("image_ids", "vectors", "text_center", "split", "manifest_sha256")))


class VisualImageSearch:
    """Rank images by visual similarity; use the existing caption index for display metadata."""

    def __init__(self, visual_index, caption_index, encoder):
        records = {row["image_id"]: row for row in caption_index.records}
        manifest = caption_index.metadata.get(f"{visual_index.split}_manifest_sha256")
        if (set(visual_index.image_ids) != records.keys()
                or caption_index.metadata.get("candidate_split") != visual_index.split
                or (manifest is not None and manifest != visual_index.manifest_sha256)):
            raise ValueError("Chỉ mục ảnh và chỉ mục caption phải cùng tập, manifest và danh sách ID.")
        self.index = visual_index
        self.records = records
        self.encoder = encoder

    @torch.inference_mode()
    def search(self, query, k=5):
        if not isinstance(query, str):
            raise SearchInputError("invalid_query", "Câu truy vấn phải là văn bản.")
        if not normalize_caption(query):
            raise SearchInputError("empty_query", "Hãy nhập một câu mô tả có nội dung.")
        if not isinstance(k, int) or k <= 0:
            raise SearchInputError("invalid_top_k", "k phải là số nguyên dương.")
        scores = self.index.scores(self.encoder.encode([query]))[0].numpy()
        order = np.lexsort((self.index.image_ids, -scores))[:k]
        return [{**self.records[int(self.index.image_ids[row])], "rank": rank, "score": float(scores[row])}
                for rank, row in enumerate(order, 1)]
