"""CPU TF-IDF search fitted on train captions and indexed over model-generated captions."""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.vietnamese_caption_batch import normalize_caption
from src.evaluation.caption_scorer import validate


class SearchInputError(ValueError):
    """Invalid search input with a stable code for callers such as the UI."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class CaptionSearchIndex:
    def __init__(self, vocabulary, idf, vectors, records, metadata):
        self.vocabulary = list(vocabulary)
        self.idf = np.asarray(idf, dtype=np.float64)
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.records = list(records)
        self.metadata = dict(metadata)
        self.image_ids = np.array([row["image_id"] for row in records], dtype=np.int64)
        if (not self.records or not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary)
                or len(set(self.image_ids.tolist())) != len(self.records)
                or self.idf.shape != (len(self.vocabulary),)
                or self.vectors.shape != (len(self.records), len(self.vocabulary))):
            raise ValueError("Invalid index vocabulary, IDs or matrix dimensions")
        if not np.isfinite(self.vectors).all() or not np.isfinite(self.idf).all() or (self.idf <= 0).any():
            raise ValueError("Index weights must be finite and IDF must be positive")

    @staticmethod
    def vectorize(captions, vocabulary, idf):
        columns = {token: column for column, token in enumerate(vocabulary)}
        vectors = np.zeros((len(captions), len(vocabulary)), dtype=np.float64)
        for row, caption in enumerate(captions):
            for token, count in Counter(normalize_caption(caption).split()).items():
                if token in columns:
                    column = columns[token]
                    vectors[row, column] = count * idf[column]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1)

    @classmethod
    def build(cls, train_rows, predictions, candidate_rows, metadata=None, *, candidate_split="val"):
        references = {row["image_id"]: row["captions"] for row in candidate_rows}
        if len(references) != len(candidate_rows):
            raise ValueError("Candidate manifest contains duplicate IDs")
        chosen = validate(predictions, references)
        document_frequency = Counter()
        document_count = 0
        for row in train_rows:
            for caption in row["captions"]:
                document_frequency.update(set(normalize_caption(caption).split()))
                document_count += 1
        if not document_frequency:
            raise ValueError("Training captions have no tokens")
        vocabulary = sorted(document_frequency)
        idf = np.array([np.log((1 + document_count) / (1 + document_frequency[token])) + 1
                        for token in vocabulary], dtype=np.float64)
        paths = {row["image_id"]: row["image_path"] for row in candidate_rows}
        records = [{"image_id": image_id, "image_path": paths[image_id], "caption": caption}
                   for image_id, caption in chosen.items()]
        vectors = cls.vectorize([row["caption"] for row in records], vocabulary, idf)
        details = {**(metadata or {}), "format_version": 1, "fit_split": "train", "candidate_split": candidate_split,
                   "train_documents": document_count, "train_images": len(train_rows),
                   "candidate_images": len(records), "vocabulary_size": len(vocabulary),
                   "weighting": "raw TF * (log((1+n_train_documents)/(1+df_train))+1), L2 normalization",
                   "ranking": "cosine similarity descending, image_id ascending for ties"}
        return cls(vocabulary, idf, vectors, records, details)

    def search(self, query, k=5, image_root=None):
        """Return up to k positive matches with image_id, image_path, caption and score."""
        if not isinstance(query, str):
            raise SearchInputError("invalid_query", "Câu truy vấn phải là văn bản.")
        if not normalize_caption(query):
            raise SearchInputError("empty_query", "Hãy nhập một câu mô tả có nội dung.")
        if not isinstance(k, int) or k <= 0:
            raise SearchInputError("invalid_top_k", "k phải là số nguyên dương.")
        vector = self.vectorize([query], self.vocabulary, self.idf)[0]
        if not np.any(vector):
            raise SearchInputError("unknown_terms", "Chưa nhận ra từ nào trong câu. Hãy thử mô tả lại.")
        scores = self.vectors @ vector
        matches = np.flatnonzero(scores > 0)
        order = matches[np.lexsort((self.image_ids[matches], -scores[matches]))[:k]]
        results = []
        for position in order:
            record = dict(self.records[int(position)])
            if image_root is not None:
                record["image_path"] = str((Path(image_root) / record["image_path"]).resolve())
            results.append({**record, "score": float(scores[position])})
        return results

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, vocabulary=np.array(self.vocabulary), idf=self.idf,
                                    vectors=self.vectors, records=np.array(json.dumps(self.records, ensure_ascii=False)),
                                    metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported search index version")
            return cls(archive["vocabulary"].tolist(), archive["idf"], archive["vectors"],
                       json.loads(str(archive["records"])), metadata)
