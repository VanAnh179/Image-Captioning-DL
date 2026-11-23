import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_caption_retrieval_table import (check_caption_scores, check_recall_slot, leaders, read_ranks,
                                                   table_markdown)

COLUMNS = [("CIDEr", True), ("R@1", True), ("median_rank", False)]


def summary_for(ranks, prediction_sha256):
    recall = {f"R@{k}": {"hits_search_rule": sum(r is not None and r <= k for r in ranks)} for k in (1, 5, 10)}
    return {"models": {"A": {"candidate_images": 2, "queries": len(ranks), "recall": recall, "median_rank": 2.0,
                             "predictions_sha256": prediction_sha256}}}


class CaptionRetrievalTableTests(unittest.TestCase):
    def test_leaders_list_ties_and_prefer_low_median_rank(self):
        rows = [{"model": "A", "CIDEr": 78.0, "R@1": 5, "median_rank": 71.0},
                {"model": "B", "CIDEr": 68.0, "R@1": 5, "median_rank": 77.0},
                {"model": "C", "CIDEr": 76.0, "median_rank": 68.0}]
        self.assertEqual(leaders(rows, COLUMNS, {"A", "B", "C"}),
                         {"CIDEr": ["A"], "R@1": ["A", "B"], "median_rank": ["C"]})
        self.assertEqual(leaders(rows, COLUMNS, {"B", "C"}), {"CIDEr": ["C"], "R@1": ["B"], "median_rank": ["C"]})

    def test_recall_is_recomputed_from_ranks_and_caption_file_is_matched(self):
        with tempfile.TemporaryDirectory() as folder:
            prediction = Path(folder) / "A_test_predictions.json"
            prediction.write_bytes(b"[\r\n]\r\n")
            ranks_file = Path(folder) / "ranks_A.csv"
            ranks_file.write_text("rank,query\n1,a\n2,b\n,c\n3,d\n", encoding="utf-8")
            ranks = read_ranks(ranks_file)
            self.assertEqual(ranks, [1, 2, None, 3])
            lf_hash = hashlib.sha256(b"[\n]\n").hexdigest()
            summary = summary_for(ranks, lf_hash)
            summary["models"]["A"]["median_rank"] = 2.0
            self.assertEqual(check_recall_slot(summary, "A", ranks, prediction, 2, 4), "same bytes except line endings")
            summary["models"]["A"]["predictions_sha256"] = hashlib.sha256(prediction.read_bytes()).hexdigest()
            self.assertEqual(check_recall_slot(summary, "A", ranks, prediction, 2, 4), "identical")
            summary["models"]["A"]["recall"]["R@1"]["hits_search_rule"] += 1
            with self.assertRaisesRegex(ValueError, "R@1 recomputed"):
                check_recall_slot(summary, "A", ranks, prediction, 2, 4)
            summary = summary_for(ranks, "0" * 64)
            with self.assertRaisesRegex(ValueError, "different caption file"):
                check_recall_slot(summary, "A", ranks, prediction, 2, 4)

    def test_caption_file_must_be_the_one_the_scores_came_from(self):
        with tempfile.TemporaryDirectory() as folder:
            prediction = Path(folder) / "A_test_predictions.json"
            prediction.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the file the table was built from"):
                check_caption_scores({"model": "A", "predictions_sha256": "0" * 64, "test_scores": {}}, prediction, {})

    def test_table_has_both_tasks_and_leader_rows(self):
        rows = [{"model": "A", "BLEU-1": 64.5, "BLEU-2": 50.0, "BLEU-3": 37.9, "BLEU-4": 29.1, "CIDEr": 78.67,
                 "R@1": 107, "R@5": 360, "R@10": 541, "median_rank": 71.0},
                {"model": "C", "BLEU-1": 66.5, "BLEU-2": 51.7, "BLEU-3": 39.6, "BLEU-4": 30.5, "CIDEr": 75.98}]
        text = table_markdown(rows, {"CIDEr": ["A"]}, {"CIDEr": ["A"]}, ["A", "B"], 2790)
        self.assertIn("3,84% (107)", text)
        self.assertIn("| C | CLIP ViT-B/32 |", text)
        self.assertIn("–", text.splitlines()[3])
        self.assertIn("chỉ xét A và B", text)


if __name__ == "__main__":
    unittest.main()
