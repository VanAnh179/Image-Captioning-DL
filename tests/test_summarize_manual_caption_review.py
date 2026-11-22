import tempfile
import unittest
from pathlib import Path

from scripts.summarize_manual_caption_review import (FLAGS, agreement, attribute_models, check_rows, error_table,
                                                     merge_final, parse_sheet, table_markdown)


def row(image_id, item_id, caption, **flags):
    return {"image_id": image_id, "item_id": item_id, "caption": caption, **{f: flags.get(f, 0) for f in FLAGS}}


PREDICTIONS = {"A": {1: "a1", 2: "a2"}, "B": {1: "b1", 2: "same"}, "A0": {1: "c1", 2: "same"}}


class ManualReviewTests(unittest.TestCase):
    def test_sheet_rows_are_parsed_with_question_marks_kept(self):
        text = ("## Ảnh 7\n\n| item_id | Caption | VT | HD | QH | SL | BD | LT | Căn cứ |\n|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|\n"
                "| it-0a1b | có một cái cây | 1 | ? | 0 | 0 | 0 | 0 | “cây” → không có |\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.md"
            path.write_text(text, encoding="utf-8")
            rows = parse_sheet(path)
        self.assertEqual(rows, [{"image_id": 7, "item_id": "it-0a1b", "caption": "có một cái cây",
                                 "VT": 1, "HD": "?", "QH": 0, "SL": 0, "BD": 0, "LT": 0}])

    def test_sheet_must_cover_three_captions_per_listed_image(self):
        rows = [row(1, "it-1", "a1"), row(1, "it-2", "b1"), row(1, "it-3", "c1")]
        check_rows(rows, [1], "x")
        with self.assertRaises(ValueError):
            check_rows(rows[:2], [1], "x")
        with self.assertRaises(ValueError):
            check_rows(rows + [row(2, "it-4", "a2")], [1], "x")
        with self.assertRaises(ValueError):
            check_rows([*rows[:2], row(1, "it-3", "c1", VT=2)], [1], "x")

    def test_models_are_found_from_saved_captions_and_equal_captions_are_split(self):
        rows = [row(1, "it-1", "c1"), row(1, "it-2", "a1"), row(1, "it-3", "b1"),
                row(2, "it-4", "same"), row(2, "it-5", "a2"), row(2, "it-6", "same")]
        owner = attribute_models(rows, PREDICTIONS)
        self.assertEqual([owner[f"it-{i}"] for i in (1, 2, 3)], ["A0", "A", "B"])
        self.assertEqual(sorted(owner[i] for i in ("it-4", "it-5", "it-6")), ["A", "A0", "B"])
        with self.assertRaises(ValueError):
            attribute_models([row(1, "it-1", "not saved"), row(1, "it-2", "a1"), row(1, "it-3", "b1")], PREDICTIONS)

    def test_agreement_counts_only_clear_cells_and_lists_differences(self):
        first = [row(1, "it-1", "a", VT=1, HD="?", QH="?", BD=1)]
        second = [row(1, "it-1", "a", VT=1, HD="?", QH=0, BD=0)]
        result = agreement(first, second)
        self.assertEqual((result["cells"], result["both_clear"], result["same"], result["different"]), (6, 4, 3, 1))
        self.assertEqual((result["both_unclear"], result["one_unclear"]), (1, 1))
        self.assertEqual(result["different_cells"], [{"item_id": "it-1", "flag": "BD", "first": 1, "second": 0}])

    def test_final_rows_use_own_labels_and_agreed_cells_on_shared_images(self):
        linh = [row(1, "it-1", "a", BD=1), row(3, "it-3", "c")]
        vanh = [row(1, "it-1", "a", QH=1), row(2, "it-2", "b", VT=1)]
        final = merge_final(linh, vanh, {1}, {"it-1": {"QH": 1, "BD": 0}})
        self.assertEqual([(r["image_id"], r["QH"], r["BD"], r["VT"]) for r in final],
                         [(1, 1, 0, 0), (2, 0, 0, 1), (3, 0, 0, 0)])

    def test_error_table_keeps_unclear_out_of_the_denominator(self):
        rows = [row(1, "it-1", "c1", VT=1), row(1, "it-2", "a1", HD="?"), row(1, "it-3", "b1", BD=1, LT="?")]
        owner = attribute_models(rows, PREDICTIONS)
        table = error_table(rows, owner)
        self.assertEqual(table["A0"]["VT"], {"errors": 1, "clear": 1, "unclear": 0})
        self.assertEqual(table["A"]["HD"], {"errors": 0, "clear": 0, "unclear": 1})
        self.assertEqual(table["A"]["any"], {"errors": 0, "clear": 0, "unclear": 1})
        self.assertEqual(table["B"]["any"], {"errors": 1, "clear": 1, "unclear": 0})  # a 1 outweighs another "?"
        self.assertIn("1/1 (100,00%) · 0 ?", table_markdown(table))


if __name__ == "__main__":
    unittest.main()
