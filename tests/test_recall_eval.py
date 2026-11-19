import unittest

import numpy as np

from src.retrieval.caption_search import CaptionSearchIndex
from src.retrieval.recall_eval import paired_bootstrap, rank_queries, recall_table


def toy_index():
    # Images 10 and 20 have the same caption; 30 differs. Vocabulary is a, b, c.
    vocabulary = ["a", "b", "c"]
    idf = np.ones(3)
    captions = {10: "a b", 20: "a b", 30: "c"}
    records = [{"image_id": image_id, "image_path": f"{image_id}.jpg", "caption": caption}
               for image_id, caption in captions.items()]
    vectors = CaptionSearchIndex.vectorize([row["caption"] for row in records], vocabulary, idf)
    return CaptionSearchIndex(vocabulary, idf, vectors, records, {"format_version": 1})


class RecallEvalTests(unittest.TestCase):
    def test_ranks_follow_search_rule_and_report_tie_bounds(self):
        index = toy_index()
        rows = rank_queries(index, [(10, "a b"), (20, "a b"), (30, "c")])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 1])  # tie broken by ascending image_id
        self.assertEqual([row["rank_best"] for row in rows], [1, 1, 1])
        self.assertEqual([row["rank_worst"] for row in rows], [2, 2, 1])
        self.assertEqual([row["tied_with_answer"] for row in rows], [1, 1, 0])
        self.assertEqual([row["top1_image_id"] for row in rows], [10, 10, 30])

    def test_rank_matches_public_search(self):
        index = toy_index()
        for image_id, text in [(10, "a"), (20, "a b c"), (30, "b c")]:
            row = rank_queries(index, [(image_id, text)])[0]
            ids = [hit["image_id"] for hit in index.search(text, k=3)]
            self.assertEqual(row["rank"], ids.index(image_id) + 1)

    def test_query_without_vocabulary_tokens_is_counted_as_a_miss(self):
        rows = rank_queries(toy_index(), [(10, "xyz"), (10, "a b")])
        unranked = rows[0]
        self.assertTrue(unranked["zero_query"])
        for field in ("rank", "rank_best", "rank_worst", "correct_score", "top1_image_id"):
            self.assertIsNone(unranked[field])
        table = recall_table(rows, 3)
        self.assertEqual(table["queries"], 2)
        self.assertEqual(table["median_rank"], 1)
        self.assertEqual(table["recall"]["R@10"]["search_rule"], 0.5)

    def test_paired_bootstrap_counts_unranked_queries_as_misses(self):
        result = paired_bootstrap([1, 2, 3], [1, None, None], [None, 2, None], resamples=50)
        self.assertEqual((result["queries_a_better"], result["queries_b_better"], result["queries_equal_rank"]), (1, 1, 1))
        self.assertAlmostEqual(result["R@1"]["difference"], 1 / 3)
        self.assertEqual(result["R@10"]["difference"], 0)

    def test_recall_table(self):
        index = toy_index()
        rows = rank_queries(index, [(10, "a b"), (20, "a b"), (30, "c"), (30, "a")])
        table = recall_table(rows, 3)
        self.assertEqual(table["queries"], 4)
        self.assertAlmostEqual(table["recall"]["R@1"]["search_rule"], 2 / 4)
        self.assertAlmostEqual(table["recall"]["R@1"]["best_case"], 3 / 4)
        self.assertAlmostEqual(table["recall"]["R@5"]["search_rule"], 1.0)
        self.assertAlmostEqual(table["recall"]["R@1"]["chance"], 1 / 3)
        self.assertAlmostEqual(table["recall"]["R@10"]["chance"], 1.0)

    def test_paired_bootstrap_difference_and_counts(self):
        image_ids = [1, 1, 2, 2]
        result = paired_bootstrap(image_ids, [1, 2, 3, 20], [1, 9, 1, 1], cutoffs=(1, 5), resamples=200)
        self.assertEqual((result["queries_a_better"], result["queries_b_better"], result["queries_equal_rank"]), (1, 2, 1))
        self.assertAlmostEqual(result["R@1"]["difference"], -0.5)  # A hits 1 of 4, B hits 3 of 4
        self.assertAlmostEqual(result["R@5"]["difference"], 0.0)  # both hit 3 of 4
        self.assertLessEqual(result["R@1"]["ci95_low"], result["R@1"]["difference"] + 1e-9)
        self.assertGreaterEqual(result["R@1"]["ci95_high"], result["R@1"]["difference"] - 1e-9)

    def test_unknown_answer_image_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_queries(toy_index(), [(99, "a")])


if __name__ == "__main__":
    unittest.main()
