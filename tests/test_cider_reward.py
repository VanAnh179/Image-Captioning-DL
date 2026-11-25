import unittest

from pycocoevalcap.cider.cider import Cider

from src.training.cider_reward import CiderD

REFERENCES = {
    1: ["một người đang đi xe máy trên đường", "có một người chạy xe máy", "người đàn ông lái xe máy qua phố"],
    2: ["nhiều loại hàng hoá được bày trên kệ", "kệ hàng có nhiều sản phẩm", "một gian hàng bày nhiều đồ"],
    3: ["con mèo ngủ trên ghế", "một con mèo nằm trên ghế sofa", "mèo đang ngủ"],
    4: ["một ngôi nhà cao tầng ở bên đường", "toà nhà cao xuất hiện trong ảnh", "ngôi nhà nhiều tầng"],
}
HYPOTHESES = {1: "một người đang chạy xe máy trên đường", 2: "có nhiều hàng hoá trên kệ", 3: "con mèo nằm ngủ",
              4: "một ngôi nhà ở bên đường"}


class CiderRewardTests(unittest.TestCase):
    def test_scores_match_pycocoevalcap_when_the_table_comes_from_the_same_references(self):
        _, expected = Cider().compute_score({i: r for i, r in REFERENCES.items()}, {i: [h] for i, h in HYPOTHESES.items()})
        scorer = CiderD(REFERENCES)
        got = [scorer.score(HYPOTHESES[i], scorer.prepare(REFERENCES[i])) for i in sorted(REFERENCES)]
        for want, have in zip(expected, got):
            self.assertAlmostEqual(float(want), have, places=9)

    def test_better_match_scores_higher_and_empty_scores_zero(self):
        scorer = CiderD(REFERENCES)
        prepared = scorer.prepare(REFERENCES[1])
        exact = scorer.score(REFERENCES[1][0], prepared)
        partial = scorer.score("một người đang đi", prepared)
        self.assertGreater(exact, partial)
        self.assertGreater(partial, scorer.score("con mèo ngủ", prepared))
        self.assertEqual(scorer.score("", prepared), 0.0)


if __name__ == "__main__":
    unittest.main()
