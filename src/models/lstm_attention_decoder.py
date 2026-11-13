"""Model A: a one-layer LSTM with additive attention over 49 image regions."""

import torch
from torch import nn

from src.data.vietnamese_caption_batch import BOS, EOS, PAD, UNK
from src.models.region_projection import RegionProjection


class AdditiveAttention(nn.Module):
    """Score each region with the previous hidden state and return context plus weights."""

    def __init__(self, d_model, hidden_size, attention_size):
        super().__init__()
        self.region_key = nn.Linear(d_model, attention_size, bias=False)
        self.hidden_key = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, memory, projected_memory, hidden):
        energy = self.score(torch.tanh(projected_memory + self.hidden_key(hidden).unsqueeze(1))).squeeze(-1)
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


class LSTMAttentionCaptionDecoder(nn.Module):
    """Teacher forcing returns logits [B,T,V] and attention [B,T,49].

    Input position t contains the previous token. Its output predicts label t.
    Use cross_entropy with ignore_index=PAD. In generate(), token_ids[:,t],
    logits[:,t] and attention[:,t] all describe the same newly generated token.
    """

    def __init__(self, vocab_size, d_model=256, hidden_size=256, attention_size=256, dropout=0.1):
        super().__init__()
        if vocab_size <= UNK or min(d_model, hidden_size, attention_size) <= 0:
            raise ValueError("Vocabulary must include PAD/BOS/EOS/UNK and dimensions must be positive")
        self.regions = RegionProjection(d_model, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.attention = AdditiveAttention(d_model, hidden_size, attention_size)
        self.initial_hidden = nn.Linear(d_model, hidden_size)
        self.initial_cell = nn.Linear(d_model, hidden_size)
        self.cell = nn.LSTMCell(d_model + d_model, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def _prepare(self, regions):
        memory = self.regions(regions)
        pooled = memory.mean(dim=1)
        state = (torch.tanh(self.initial_hidden(pooled)), torch.tanh(self.initial_cell(pooled)))
        return memory, self.attention.region_key(memory), state

    def _step(self, token, memory, keys, state):
        context, weights = self.attention(memory, keys, state[0])
        inputs = torch.cat((self.dropout(self.embedding(token)), context), dim=-1)
        state = self.cell(inputs, state)
        logits = self.output(self.dropout(state[0]))
        return logits, weights, state

    def forward(self, regions, decoder_input):
        if decoder_input.ndim != 2 or decoder_input.shape[0] != regions.shape[0] or decoder_input.shape[1] == 0:
            raise ValueError("decoder_input must have shape [B,T], matching regions, with T > 0")
        memory, keys, state = self._prepare(regions)
        logits, attention = [], []
        for token in decoder_input.unbind(dim=1):
            scores, weights, state = self._step(token, memory, keys, state)
            logits.append(scores)
            attention.append(weights)
        return {"logits": torch.stack(logits, dim=1), "attention": torch.stack(attention, dim=1)}

    @torch.no_grad()
    def generate(self, regions, max_len=40):
        """Greedy decode from BOS until EOS or max_len, temporarily disabling dropout.

        lengths includes EOS when emitted. Finished rows have PAD tokens and zero
        logits/attention after their lengths. Raw logits precede the PAD/BOS mask
        used to choose a token. Reshape attention[...,49] to [...,7,7] to plot it.
        """
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        training = self.training
        self.eval()
        try:
            memory, keys, state = self._prepare(regions)
            batch_size = regions.shape[0]
            token = torch.full((batch_size,), BOS, dtype=torch.long, device=regions.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=regions.device)
            lengths = torch.zeros(batch_size, dtype=torch.long, device=regions.device)
            tokens, logits, attention = [], [], []
            for _ in range(max_len):
                scores, weights, next_state = self._step(token, memory, keys, state)
                active = ~finished
                state = tuple(torch.where(active.unsqueeze(1), new, old) for new, old in zip(next_state, state))
                choices = scores.clone()
                choices[:, [PAD, BOS]] = float("-inf")
                token = choices.argmax(dim=-1).masked_fill(finished, PAD)
                tokens.append(token)
                logits.append(scores.masked_fill(finished.unsqueeze(1), 0))
                attention.append(weights.masked_fill(finished.unsqueeze(1), 0))
                lengths += active.long()
                finished |= token.eq(EOS)
                if finished.all():
                    break
            return {
                "token_ids": torch.stack(tokens, dim=1),
                "logits": torch.stack(logits, dim=1),
                "attention": torch.stack(attention, dim=1),
                "lengths": lengths,
                "finished": finished,
            }
        finally:
            self.train(training)
