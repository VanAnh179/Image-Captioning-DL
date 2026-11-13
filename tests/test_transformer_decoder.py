import unittest

import torch

from src.data.vietnamese_caption_batch import BOS, EOS, PAD
from src.models.transformer_decoder import TransformerCaptionDecoder, causal_mask, self_attention_mask, sinusoidal_positions
from src.models.transformer_decoder import scaled_dot_product_attention
from scripts.check_decoder_causality import run

VOCAB = 50


def sentence(*tokens):
    return torch.tensor([[BOS, *tokens]])


class TransformerDecoderTests(unittest.TestCase):
    def test_fully_masked_attention_has_zero_weights_and_finite_gradients(self):
        query = torch.randn(2, 1, 3, 4, requires_grad=True)
        mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
        context, weights = scaled_dot_product_attention(query, query, query, mask)
        self.assertEqual(weights.count_nonzero().item(), 0)
        self.assertEqual(context.count_nonzero().item(), 0)
        context.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_odd_dimension_and_invalid_configuration(self):
        self.assertEqual(sinusoidal_positions(4, 9).shape, (4, 9))
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerCaptionDecoder(VOCAB, d_model=31, num_heads=4)
        with self.assertRaisesRegex(ValueError, "decoder_input"):
            self.model(self.regions, torch.ones(1, 17, dtype=torch.long))

    def test_causality_check_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run(self.model, None, [], 32)

    def setUp(self):
        torch.manual_seed(0)
        self.model = TransformerCaptionDecoder(VOCAB, d_model=32, num_heads=4, num_layers=2, d_ff=64, max_len=16).eval()
        self.regions = torch.randn(1, 49, 2048)

    def test_output_shape(self):
        out = self.model(torch.randn(3, 49, 2048), torch.randint(4, VOCAB, (3, 7)))
        self.assertEqual(tuple(out.shape), (3, 7, VOCAB))

    def test_positions_differ_per_word_slot(self):
        table = sinusoidal_positions(16, 32)
        self.assertEqual(tuple(table.shape), (16, 32))
        self.assertEqual(len({tuple(row.tolist()) for row in table}), 16)

    def test_causal_mask_blocks_future_tokens(self):
        self.assertTrue(torch.equal(causal_mask(3), torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).bool()))

    def test_padding_positions_are_never_attended(self):
        mask = self_attention_mask(torch.tensor([[BOS, 5, 6, PAD, PAD], [BOS, 7, 8, 9, EOS]]))
        self.assertFalse(mask[0, 0, :, 3:].any())  # no query reads the two PAD keys
        self.assertTrue(mask[1, 0].equal(causal_mask(5)))  # a row without padding only has the causal restriction

    def test_padding_does_not_change_real_positions(self):
        ids = sentence(5, 6, 7)
        padded = torch.cat([ids, torch.full((1, 4), PAD)], dim=1)
        with torch.no_grad():
            alone, in_batch = self.model(self.regions, ids), self.model(self.regions, padded)
        self.assertTrue(torch.allclose(alone, in_batch[:, :ids.shape[1]], atol=1e-6))

    def test_decoder_reads_the_regions(self):
        ids = sentence(5, 6, 7)
        with torch.no_grad():
            a = self.model(self.regions, ids)
            b = self.model(torch.randn(1, 49, 2048), ids)
        self.assertFalse(torch.allclose(a[:, 0], b[:, 0], atol=1e-4))  # even the BOS position sees the image

    def test_changing_the_tail_keeps_head_predictions(self):
        original = sentence(5, 6, 7, 8, 9, 10, EOS)
        changed = sentence(5, 6, 7, 21, 22, 23, 24)  # tokens from input index 4 on replaced
        with torch.no_grad():
            a, b = self.model(self.regions, original), self.model(self.regions, changed)
        head = 4  # input positions 0..3 (BOS,5,6,7) only see themselves and earlier tokens
        self.assertTrue(torch.allclose(a[:, :head], b[:, :head], atol=1e-6))
        self.assertTrue(torch.equal(a[:, :head].argmax(-1), b[:, :head].argmax(-1)))
        self.assertFalse(torch.allclose(a[:, head:], b[:, head:], atol=1e-4))  # the tail itself does change


if __name__ == "__main__":
    unittest.main()
