"""Pick distractors so the key lands at a random rank among the four options.

A multiple-choice item whose distractors are all derived from fixed misconceptions has the key at one fixed rank
in value order (for example always the second-smallest). `pick_ranked` takes a pool of misconception values on both
sides of the key and draws the rank first. Not a generator.
"""

from __future__ import annotations


def pick_ranked(rng, key_text: str, key_value: float, pool: dict[str, float], *, forced=(), min_gap: float = 0.0):
    """Return three distractor texts from ``pool`` ({text: numeric value}), or None if none can be drawn.

    The rank of the key among the four options is drawn uniformly from the ranks the pool can supply. Texts in
    ``forced`` are always included. Every option is at least ``min_gap`` from every other in value and all texts
    differ.
    """
    pool = {t: v for t, v in pool.items() if t != key_text and abs(v - key_value) >= min_gap}
    if any(t not in pool for t in forced):
        return None
    below = [t for t, v in pool.items() if v < key_value]
    above = [t for t, v in pool.items() if v > key_value]
    f_below = [t for t in forced if t in below]
    f_above = [t for t in forced if t in above]
    ranks = [r for r in (1, 2, 3, 4) if len(f_below) <= r - 1 <= len(below) and len(f_above) <= 4 - r <= len(above)]
    rng.shuffle(ranks)
    for rank in ranks:
        for _ in range(8):
            rest_below = [t for t in below if t not in f_below]
            rest_above = [t for t in above if t not in f_above]
            rng.shuffle(rest_below)
            rng.shuffle(rest_above)
            chosen = f_below + rest_below[: rank - 1 - len(f_below)] + f_above + rest_above[: 4 - rank - len(f_above)]
            values = [key_value] + [pool[t] for t in chosen]
            if len(set(chosen)) == 3 and all(
                abs(values[i] - values[j]) >= max(min_gap, 1e-12) for i in range(4) for j in range(i + 1, 4)
            ):
                return chosen
    return None
