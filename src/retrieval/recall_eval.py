"""Recall@k of caption-to-image search: every reference caption is a query, its own image is the answer."""

import numpy as np

CUTOFFS = (1, 5, 10)


def rank_queries(index, queries):
    """Rank the whole candidate pool for each (image_id, text) query.

    Usable queries follow CaptionSearchIndex.search: score descending, image_id ascending for ties.
    Queries with no vocabulary tokens have no rank and still count toward the recall denominator.
    rank_best/rank_worst bound the rank if the images tied with the answer were ordered in its favour
    or against it, so a result that depends on the tie rule is visible.
    """
    positions = {int(image_id): position for position, image_id in enumerate(index.image_ids)}
    missing = sorted({image_id for image_id, _ in queries if image_id not in positions})
    if missing:
        raise ValueError(f"Answer images are not in the index: {missing[:5]}")
    vectors = index.vectorize([text for _, text in queries], index.vocabulary, index.idf)
    rows = []
    for (image_id, _), vector in zip(queries, vectors):
        if not vector.any():
            rows.append({"rank": None, "rank_best": None, "rank_worst": None, "tied_with_answer": 0,
                         "correct_score": None, "top1_image_id": None, "zero_query": True})
            continue
        scores = index.vectors @ vector  # same product as CaptionSearchIndex.search
        answer = positions[image_id]
        order = np.lexsort((index.image_ids, -scores))
        correct = scores[answer]
        greater = int((scores > correct).sum())
        tied = int((scores == correct).sum()) - 1
        rows.append({"rank": int(np.flatnonzero(order == answer)[0]) + 1, "rank_best": greater + 1,
                     "rank_worst": greater + 1 + tied, "tied_with_answer": tied,
                     "correct_score": float(correct), "top1_image_id": int(index.image_ids[order[0]]),
                     "zero_query": False})
    return rows


def recall_table(rows, pool_size, cutoffs=CUTOFFS):
    """Summarize recall over all queries and median rank over queries with search results."""
    total = len(rows)
    if total == 0:
        raise ValueError("No queries")
    ranked_rows = [row for row in rows if row["rank"] is not None]
    result = {"queries": total, "candidate_images": pool_size,
              "queries_with_zero_vector": sum(row["zero_query"] for row in rows),
              "queries_tied_with_other_images": sum(row["tied_with_answer"] > 0 for row in rows),
              "median_rank": float(np.median([row["rank"] for row in ranked_rows])) if ranked_rows else None,
              "recall": {}}
    for k in cutoffs:
        entry = {name: sum(row[key] <= k for row in ranked_rows) / total
                 for name, key in (("search_rule", "rank"), ("best_case", "rank_best"), ("worst_case", "rank_worst"))}
        entry["hits_search_rule"] = sum(row["rank"] <= k for row in ranked_rows)
        entry["chance"] = min(k, pool_size) / pool_size
        result["recall"][f"R@{k}"] = entry
    return result


def paired_bootstrap(image_ids, ranks_a, ranks_b, cutoffs=CUTOFFS, resamples=2000, seed=0):
    """A minus B Recall@k with a 95% interval from resampling images (their five queries stay together)."""
    image_ids = np.asarray(image_ids)
    ranks_a = np.array([np.inf if rank is None else rank for rank in ranks_a])
    ranks_b = np.array([np.inf if rank is None else rank for rank in ranks_b])
    unique, inverse = np.unique(image_ids, return_inverse=True)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique), size=(resamples, len(unique)))
    result = {"image_resamples": resamples, "seed": seed,
              "queries_a_better": int((ranks_a < ranks_b).sum()), "queries_b_better": int((ranks_b < ranks_a).sum()),
              "queries_equal_rank": int((ranks_a == ranks_b).sum())}
    for k in cutoffs:
        gain = (ranks_a <= k).astype(float) - (ranks_b <= k).astype(float)
        per_image_sum = np.bincount(inverse, weights=gain, minlength=len(unique))
        per_image_count = np.bincount(inverse, minlength=len(unique))
        estimates = per_image_sum[draws].sum(axis=1) / per_image_count[draws].sum(axis=1)
        low, high = np.percentile(estimates, [2.5, 97.5])
        result[f"R@{k}"] = {"difference": float(gain.mean()), "ci95_low": float(low), "ci95_high": float(high)}
    return result
