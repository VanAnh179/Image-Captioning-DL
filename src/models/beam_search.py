"""Beam search for any caption decoder, and the dispatcher that picks the fastest implementation.

The LSTM decoder keeps its own incremental beam search. The Transformer decoders keep no key/value cache, so
beam_search_stateless re-reads the whole prefix at every step, like their greedy decoding does.
"""

import torch

from src.data.vietnamese_caption_batch import BOS, EOS, PAD
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder


def logits_of(output):
    return output["logits"] if isinstance(output, dict) else output


@torch.no_grad()
def beam_search_stateless(model, regions, beam_size=3, max_len=40, length_penalty=0.7):
    """Same contract as LSTMAttentionCaptionDecoder.generate_beam: token_ids [B,L] (PAD after EOS) and lengths."""
    if max_len <= 0 or beam_size <= 0:
        raise ValueError("max_len and beam_size must be positive")
    training = model.training
    model.eval()
    try:
        batch, beams, device = regions.shape[0], beam_size, regions.device
        regions = regions.repeat_interleave(beams, dim=0)
        sequence = torch.full((batch * beams, 1), BOS, dtype=torch.long, device=device)
        scores = torch.zeros(batch, beams, device=device)
        scores[:, 1:] = float("-inf")  # all beams start identical; keep one alive
        finished = torch.zeros(batch * beams, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch * beams, dtype=torch.long, device=device)
        offset = (torch.arange(batch, device=device) * beams).unsqueeze(1)
        for _ in range(max_len):
            log_probs = torch.log_softmax(logits_of(model(regions, sequence))[:, -1].float(), dim=-1)
            log_probs[:, [PAD, BOS]] = float("-inf")
            done = log_probs.new_full(log_probs.shape, float("-inf"))
            done[:, PAD] = 0  # a finished beam only continues with PAD at no cost
            log_probs = torch.where(finished.unsqueeze(1), done, log_probs)
            vocab = log_probs.shape[1]
            candidates = (scores.reshape(-1, 1) + log_probs).reshape(batch, beams * vocab)
            scores, chosen = candidates.topk(beams, dim=1)
            parent = (offset + chosen // vocab).reshape(-1)
            token = (chosen % vocab).reshape(-1)
            was_finished = finished[parent]
            lengths = lengths[parent] + (~was_finished).long()
            sequence = torch.cat((sequence[parent], token.unsqueeze(1)), dim=1)
            finished = was_finished | token.eq(EOS)
            if finished.all():
                break
        ranked = scores.reshape(-1) / lengths.clamp_min(1).float() ** length_penalty
        best = ranked.reshape(batch, beams).argmax(dim=1) + offset.squeeze(1)
        return {"token_ids": sequence[best, 1:], "lengths": lengths[best]}
    finally:
        model.train(training)


def generate_beam(model, regions, beam_size=3, max_len=40, length_penalty=0.7):
    """Beam search for A, A0, B and C. beam_size 1 equals greedy decoding."""
    if isinstance(model, LSTMAttentionCaptionDecoder):
        return model.generate_beam(regions, beam_size, max_len, length_penalty)
    return beam_search_stateless(model, regions, beam_size, max_len, length_penalty)
