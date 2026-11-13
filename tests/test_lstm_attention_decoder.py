import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.data.vietnamese_caption_batch import BOS, EOS, PAD
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder
from src.models.region_projection import RegionProjection
from src.models.transformer_decoder import RegionProjection as TransformerRegionProjection


class LSTMAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LSTMAttentionCaptionDecoder(20, d_model=16, hidden_size=24, attention_size=12, dropout=0).eval()
        self.regions = torch.randn(2, 49, 2048)
        self.inputs = torch.tensor([[BOS, 4, 5, 6], [BOS, 7, 8, PAD]])
        self.labels = torch.tensor([[4, 5, 6, EOS], [7, 8, EOS, PAD]])

    def test_shapes_attention_sum_and_formula(self):
        output = self.model(self.regions, self.inputs)
        self.assertEqual(output["logits"].shape, (2, 4, 20))
        self.assertEqual(output["attention"].shape, (2, 4, 49))
        self.assertTrue(torch.all(output["attention"] >= 0))
        torch.testing.assert_close(output["attention"].sum(-1), torch.ones(2, 4))
        memory, keys, state = self.model._prepare(self.regions)
        context, weights = self.model.attention(memory, keys, state[0])
        expected = self.model.attention.score(torch.tanh(keys + self.model.attention.hidden_key(state[0])[:, None])).squeeze(-1).softmax(-1)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(context, (memory * weights.unsqueeze(-1)).sum(1))

    def test_loss_backpropagates_to_image_attention_lstm_and_projection(self):
        regions = self.regions.requires_grad_()
        output = self.model(regions, self.inputs)
        loss = F.cross_entropy(output["logits"].flatten(0, 1), self.labels.flatten(), ignore_index=PAD)
        loss.backward()
        for name in ("regions.linear.weight", "regions.position.weight", "attention.score.weight",
                     "attention.hidden_key.weight", "cell.weight_ih", "embedding.weight", "output.weight"):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertGreater(regions.grad.abs().sum().item(), 0)

    def test_future_tokens_and_right_padding_do_not_change_prefix_or_loss(self):
        original = self.model(self.regions, self.inputs)
        changed = self.inputs.clone()
        changed[:, 2:] = 9
        output = self.model(self.regions, changed)
        torch.testing.assert_close(original["logits"][:, :2], output["logits"][:, :2])
        torch.testing.assert_close(original["attention"][:, :2], output["attention"][:, :2])
        padded = F.pad(self.inputs, (0, 3), value=PAD)
        labels = F.pad(self.labels, (0, 3), value=PAD)
        logits = self.model(self.regions, padded)["logits"]
        loss = F.cross_entropy(original["logits"].flatten(0, 1), self.labels.flatten(), ignore_index=PAD)
        padded_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=PAD)
        torch.testing.assert_close(loss, padded_loss)

    def test_generation_matches_teacher_forcing_for_generated_prefix(self):
        self.model.train()
        generated = self.model.generate(self.regions, max_len=6)
        self.assertTrue(self.model.training)
        self.assertFalse(generated["logits"].requires_grad)
        self.model.eval()
        inputs = torch.cat((torch.full((2, 1), BOS), generated["token_ids"][:, :-1]), dim=1)
        teacher = self.model(self.regions, inputs)
        for row, length in enumerate(generated["lengths"].tolist()):
            torch.testing.assert_close(generated["logits"][row, :length], teacher["logits"][row, :length])
            torch.testing.assert_close(generated["attention"][row, :length], teacher["attention"][row, :length])

    def test_mixed_eos_lengths_and_masked_special_tokens(self):
        step_number = 0

        def step(token, memory, keys, state):
            nonlocal step_number
            scores = torch.zeros(2, 20)
            scores[:, PAD] = 100
            scores[:, BOS] = 99
            scores[0, EOS] = 10
            scores[1, EOS if step_number else 5] = 10
            step_number += 1
            return scores, torch.full((2, 49), 1 / 49), state

        with patch.object(self.model, "_step", side_effect=step):
            output = self.model.generate(self.regions, max_len=6)
        self.assertEqual(output["token_ids"].tolist(), [[EOS, PAD], [5, EOS]])
        self.assertEqual(output["lengths"].tolist(), [1, 2])
        self.assertTrue(output["finished"].all())
        self.assertEqual(output["attention"][0, 1].count_nonzero().item(), 0)
        self.assertEqual(output["logits"][0, 1].count_nonzero().item(), 0)

    def test_max_length_and_checkpoint_round_trip(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[5] = 10
        generated = self.model.generate(self.regions, max_len=3)
        self.assertEqual(generated["token_ids"].tolist(), [[5, 5, 5], [5, 5, 5]])
        self.assertFalse(generated["finished"].any())
        checkpoint = io.BytesIO()
        torch.save(self.model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = LSTMAttentionCaptionDecoder(20, 16, 24, 12, 0).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(generated["attention"], restored.generate(self.regions, 3)["attention"])

    def test_projection_is_shared_and_accepts_float16_cache(self):
        self.assertIs(RegionProjection, TransformerRegionProjection)
        output = self.model(self.regions.half(), self.inputs)
        self.assertTrue(torch.isfinite(output["logits"]).all())
        with self.assertRaisesRegex(ValueError, "regions"):
            self.model(self.regions[:, :48], self.inputs)
        with self.assertRaisesRegex(ValueError, "max_len"):
            self.model.generate(self.regions, 0)


if __name__ == "__main__":
    unittest.main()
