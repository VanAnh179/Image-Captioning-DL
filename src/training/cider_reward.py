"""CIDEr-D with a fixed document-frequency table, used as the reward for self-critical training.

Scores equal pycocoevalcap's Cider when the table is built from the same references (see tests); training uses the
table of the train references, so rewards do not change with the batch.
"""

import math
from collections import Counter, defaultdict

import numpy as np

N = 4
SIGMA = 6.0


def ngrams(tokens):
    counts = Counter()
    for n in range(1, N + 1):
        for start in range(len(tokens) - n + 1):
            counts[tuple(tokens[start:start + n])] += 1
    return counts


class CiderD:
    def __init__(self, references):
        """references: {image_id: [caption string, ...]} whose document frequencies are used for every score."""
        self.frequency = defaultdict(float)
        for captions in references.values():
            for gram in {gram for caption in captions for gram in ngrams(caption.split())}:
                self.frequency[gram] += 1.0
        self.log_images = math.log(float(len(references)))

    def vector(self, counts):
        vec, norm, length = [defaultdict(float) for _ in range(N)], [0.0] * N, 0
        for gram, frequency in counts.items():
            n = len(gram) - 1
            vec[n][gram] = float(frequency) * (self.log_images - math.log(max(1.0, self.frequency[gram])))
            norm[n] += vec[n][gram] ** 2
            if n == 1:
                length += frequency  # pycocoevalcap counts bigrams here; kept for identical scores
        return vec, [math.sqrt(value) for value in norm], length

    def prepare(self, captions):
        """Reference vectors of one image, computed once."""
        return [self.vector(ngrams(caption.split())) for caption in captions]

    def score(self, hypothesis, prepared):
        """CIDEr-D of one hypothesis string against prepare(references), on pycocoevalcap's scale (0-10)."""
        vec_hyp, norm_hyp, length_hyp = self.vector(ngrams(hypothesis.split()))
        total = np.zeros(N)
        for vec_ref, norm_ref, length_ref in prepared:
            delta = float(length_hyp - length_ref)
            for n in range(N):
                value = sum(min(count, vec_ref[n][gram]) * vec_ref[n][gram] for gram, count in vec_hyp[n].items())
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    value /= norm_hyp[n] * norm_ref[n]
                total[n] += value * math.exp(-(delta ** 2) / (2 * SIGMA ** 2))
        return float(np.mean(total)) / len(prepared) * 10.0
