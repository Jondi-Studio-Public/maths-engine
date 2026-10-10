"""Pick wrong answers so that the right one does not sit at a predictable rank.

When every misconception candidate is a small slip from the key (one more, one less, a ten more, a ten less),
choosing three at random leaves the key in the middle of the sorted options most of the time, and "pick the
middle value" beats guessing. `pick_ranked_distractors` draws the number of wrong answers that fall below the key
first, uniformly over what the candidates allow, then fills each side from the misconception candidates and,
only when a side has too few, from plain neighbours of the key.

Underscore-prefixed, so the generator registry does not import it as a generator module.
"""

from __future__ import annotations

from ..core import check


def pick_ranked_distractors(rng, key: int, candidates, *, lo: int, hi: int, n: int = 3, fillers=(2, 3, 4)) -> list[int]:
    """`n` distinct whole numbers in [lo, hi], none equal to `key`, with the number below `key` chosen uniformly
    from the counts the candidates (topped up with `key +/- d` for each `d` in `fillers`) allow."""
    pool: list[int] = []
    for c in candidates:
        if lo <= c <= hi and c != key and c not in pool:
            pool.append(c)
    low = [c for c in pool if c < key]
    high = [c for c in pool if c > key]
    extra_low = [key - d for d in fillers if lo <= key - d and key - d not in pool]
    extra_high = [key + d for d in fillers if key + d <= hi and key + d not in pool]
    feasible = [r for r in range(n + 1) if r <= len(low) + len(extra_low) and n - r <= len(high) + len(extra_high)]
    check(bool(feasible), f"fewer than {n} distinct wrong values available around {key} in {lo}..{hi}")
    below = rng.choice(feasible)
    above = n - below
    chosen: list[int] = []
    for count, preferred, extras in ((below, low, extra_low), (above, high, extra_high)):
        picks = rng.sample(preferred, min(count, len(preferred)))
        picks += rng.sample(extras, count - len(picks))
        chosen += picks
    rng.shuffle(chosen)
    return chosen
