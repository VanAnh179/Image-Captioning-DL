import unittest

import torch

from src.data.vietnamese_caption_batch import BOS, EOS, PAD
from src.models.beam_search import beam_search_stateless, generate_beam
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder
from src.models.transformer_decoder import TransformerCaptionDecoder
from src.training.caption_trainer import generate_tokens

VOCAB = 30


def lstm():
    torch.manual_seed(0)
    return LSTMAttentionCaptionDecoder(VOCAB, d_model=16, hidden_size=24, attention_size=16, dropout=0.0).eval()


def transformer():
    torch.manual_seed(0)
    return TransformerCaptionDecoder(VOCAB, d_model=16, num_heads=2, num_layers=2, d_ff=32, dropout=0.0, max_len=16,
                                     feature_dim=12).eval()


class BeamSearchTests(unittest.TestCase):
    def test_beam_one_equals_greedy_for_transformer(self):
        model, regions = transformer(), torch.randn(3, 49, 12)
        greedy = generate_tokens(model, regions, 8)
        beam = generate_beam(model, regions, beam_size=1, max_len=8)
        for row in range(3):
            length = int(beam["lengths"][row])
            self.assertEqual(beam["token_ids"][row, :length].tolist(), greedy[row, :length].tolist())
            self.assertTrue(length == 8 or greedy[row, length - 1].item() == EOS)

    def test_stateless_search_matches_the_incremental_lstm_search(self):
        model, regions = lstm(), torch.randn(3, 49, 2048)
        fast = model.generate_beam(regions, beam_size=3, max_len=8, length_penalty=0.7)
        slow = beam_search_stateless(model, regions, beam_size=3, max_len=8, length_penalty=0.7)
        self.assertEqual(fast["lengths"].tolist(), slow["lengths"].tolist())
        for row, length in enumerate(fast["lengths"].tolist()):
            self.assertEqual(fast["token_ids"][row, :length].tolist(), slow["token_ids"][row, :length].tolist())

    def test_wider_beams_keep_the_caption_contract(self):
        model, regions = transformer(), torch.randn(2, 49, 12)
        output = generate_beam(model, regions, beam_size=4, max_len=8)
        for row, length in enumerate(output["lengths"].tolist()):
            tokens = output["token_ids"][row, :length].tolist()
            self.assertTrue(1 <= length <= 8)
            self.assertFalse({PAD, BOS} & set(tokens))
            self.assertNotIn(EOS, tokens[:-1])
        with self.assertRaises(ValueError):
            generate_beam(model, regions, beam_size=0)


if __name__ == "__main__":
    unittest.main()
