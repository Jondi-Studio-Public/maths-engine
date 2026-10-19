"""``w7f_trial_variants`` (AC9M3P01, ``f10-trial-tallies-compare-outcomes``): repeated chance experiments with a coin,
a die, or two groups pooling their tallies, so the skill is no longer a single spinner-and-tally shape.

Ladder (reasoning escalation, not size):
  easy:   a coin or die is tossed; the tally is shown; count the trials that were NOT one outcome.
  medium: two groups ran the same experiment; give the pooled count for one outcome.
  hard:   two pooled tallies; the gap between two outcomes across both groups, or how many of all the pooled trials
          were not the outcome that came up most often.

Every key is checked a second way, by summing the tally text back to numbers (``_untally``).
"""

from __future__ import annotations

from mathgen.core import check, generator, phrase

from ._fr2d_kit import int_d, need, redraw
from .fr2d_trial_tallies_build import _distinct_counts, _finish, _table, _tally, _untally

TOPIC = "w7f_trial_variants"
TIERS = ("easy", "medium", "hard")

_SETTINGS = (
    dict(
        lead="A coin was tossed {n} times and the side that landed face up was tallied.",
        lead2="Two groups tossed the same coin and tallied the side that landed face up.",
        cats=["heads", "tails"],
        noun="tosses",
    ),
    dict(
        lead="A six-sided die was rolled {n} times and the number on top was tallied.",
        lead2="Two groups rolled the same six-sided die and tallied the number on top.",
        cats=["1", "2", "3", "4", "5", "6"],
        noun="rolls",
    ),
    dict(
        lead="A card was drawn from a small pack, its suit written down and the card put back, {n} times.",
        lead2="Two groups drew a card from the same pack, wrote down its suit and put it back each time.",
        cats=["hearts", "spades", "clubs", "diamonds"],
        noun="draws",
    ),
    dict(
        lead="A bag held tiles lettered A, B, C and D. A tile was drawn and put back {n} times and its letter tallied.",
        lead2="Two groups drew a tile from the same bag, wrote its letter down and put it back each time.",
        cats=["A", "B", "C", "D"],
        noun="draws",
    ),
)


def _pick_cats(rng, st):
    pool = st["cats"]
    k = 2 if len(pool) == 2 else rng.randint(3, min(4, len(pool)))
    idx = sorted(rng.sample(range(len(pool)), k))
    return [pool[i] for i in idx], k


def _counts(rng, k, lo, hi, tlo, thi):
    if k == 2:
        for _ in range(500):
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if a != b and tlo <= a + b <= thi:
                return [a, b]
        return [lo + 3, hi]
    return _distinct_counts(rng, k, lo, hi, tlo, thi)


def _easy(rng):
    st = rng.choice(_SETTINGS)
    cats, k = _pick_cats(rng, st)
    counts = _counts(rng, k, 3, 14, 16, 40)
    total = sum(counts)
    i = rng.randrange(k)
    x, cx = cats[i], counts[i]
    key = total - cx
    check(sum(_untally(_tally(n)) for n in counts) == total, "tally text does not add to the total")
    ask = phrase(
        rng,
        (
            "How many of the {noun} did not give {x}?",
            "In how many {noun} was the result something other than {x}?",
        ),
        noun=st["noun"],
        x=x,
    )
    cands = [
        int_d(cx, "trial-gives-the-outcome-asked-not-the-rest"),
        int_d(total, "fr2d-reports-the-total"),
        int_d(key + 1, "fr2d-tally-off-by-one"),
        int_d(key - 1, "fr2d-tally-off-by-one"),
        int_d(max(c for j, c in enumerate(counts) if j != i), "fr2d-reads-the-wrong-outcome"),
    ]
    working = [f"Altogether: {' + '.join(map(str, counts))} = {total}.", f"Not {x}: {total} - {cx} = {key}."]
    stem = f"{st['lead'].format(n=total)} The tally: {_table(cats, counts)}. {ask}"
    return _finish(stem, key, cands, working)


def _two_groups(rng, tier):
    st = rng.choice(_SETTINGS)
    cats, k = _pick_cats(rng, st)
    lo, hi = (2, 11) if tier == "medium" else (2, 13)
    a = _counts(rng, k, lo, hi, 12, 30)
    b = _counts(rng, k, lo, hi, 12, 30)
    return st, cats, k, a, b


def _medium(rng):
    st, cats, k, a, b = _two_groups(rng, "medium")
    i = rng.randrange(k)
    key = a[i] + b[i]
    check(_untally(_tally(a[i])) + _untally(_tally(b[i])) == key, "pooled count differs by the tally route")
    ask = phrase(
        rng,
        ("How many times did {x} come up in the two groups together?", "Altogether, how many results were {x}?"),
        x=cats[i],
    )
    cands = [
        int_d(a[i], "trial-uses-one-group-only"),
        int_d(b[i], "trial-uses-one-group-only"),
        int_d(abs(a[i] - b[i]), "fr2d-adds-instead-of-compares"),
        int_d(sum(a) + sum(b), "fr2d-reports-the-total"),
        int_d(key + 1, "fr2d-tally-off-by-one"),
        int_d(key - 1, "fr2d-tally-off-by-one"),
    ]
    working = [f"Group 1: {a[i]}. Group 2: {b[i]}.", f"{a[i]} + {b[i]} = {key}."]
    stem = f"{st['lead2']} Group 1: {_table(cats, a)}. Group 2: {_table(cats, b)}. {ask}"
    return _finish(stem, key, cands, working)


def _hard(rng):
    st, cats, k, a, b = _two_groups(rng, "hard")
    pooled = [x + y for x, y in zip(a, b)]
    if len(set(pooled)) < k:
        need([])
    order = sorted(range(k), key=lambda i: pooled[i])
    hi, lo = order[-1], order[0]
    if rng.random() < 0.6:
        key = pooled[hi] - pooled[lo]
        check(
            key == (_untally(_tally(a[hi])) + _untally(_tally(b[hi]))) - (a[lo] + b[lo]),
            "gap differs by the tally route",
        )
        ask = phrase(
            rng,
            (
                "Across both groups, how many more times did {x} come up than {y}?",
                "Put the two groups together. How many more results were {x} than {y}?",
            ),
            x=cats[hi],
            y=cats[lo],
        )
        cands = [
            int_d(abs(a[hi] - a[lo]), "fr2d-gap-over-first-trials"),
            int_d(abs(b[hi] - b[lo]), "fr2d-gap-over-first-trials"),
            int_d(pooled[hi] + pooled[lo], "fr2d-adds-instead-of-compares"),
            int_d(pooled[hi], "fr2d-gap-for-count"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
            int_d(key - 1, "fr2d-tally-off-by-one"),
        ]
        working = [
            f"{cats[hi]}: {a[hi]} + {b[hi]} = {pooled[hi]}. {cats[lo]}: {a[lo]} + {b[lo]} = {pooled[lo]}.",
            f"{pooled[hi]} - {pooled[lo]} = {key}.",
        ]
    else:
        total = sum(pooled)
        key = total - pooled[hi]
        ask = phrase(
            rng,
            (
                "Across both groups, how many results were not the outcome that came up most often?",
                "Put the two groups together. How many {noun} did not give the most common outcome?",
            ),
            noun=st["noun"],
        )
        cands = [
            int_d(pooled[hi], "trial-gives-the-outcome-asked-not-the-rest"),
            int_d(total, "fr2d-reports-the-total"),
            int_d(sum(a) - a[hi], "trial-uses-one-group-only"),
            int_d(sum(b) - b[hi], "trial-uses-one-group-only"),
            int_d(key + 1, "fr2d-tally-off-by-one"),
            int_d(key - 1, "fr2d-tally-off-by-one"),
        ]
        working = [
            f"Together: {' + '.join(str(p) for p in pooled)} = {total}. Most common: {cats[hi]} with {pooled[hi]}.",
            f"{total} - {pooled[hi]} = {key}.",
        ]
    stem = f"{st['lead2']} Group 1: {_table(cats, a)}. Group 2: {_table(cats, b)}. {ask}"
    return _finish(stem, key, cands, working)


@generator(TOPIC, level=4, presets={t: {"tier": t} for t in TIERS})
@redraw
def w7f_trial_variants(rng, tier: str = "easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
