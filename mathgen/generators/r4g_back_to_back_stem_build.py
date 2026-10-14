"""Back-to-back stem plots compared by median and interquartile range: `r4g_back_to_back_stem`.

``back-to-back-stem-plot-compare`` (``statistics-probability``, answer ``number``). The plot is text, laid out like
the one in `f10_stem_and_leaf_build` (which this module reuses for the group contexts and the row layout): each row
is `leaves of group 1 | stem | leaves of group 2`, the left leaves written with the smallest nearest the stem.

* easy: the median of one named group (an odd or even number of values).
* medium: the interquartile range of one named group. Q1 and Q3 are the medians of the lower and upper halves of
  the ordered data, leaving the median out of both halves when the count is odd (the question says so).
* hard: how much larger one group's IQR is than the other's (the less consistent group's spread minus the more
  consistent group's), or how many values of one group lie above the other group's upper quartile.

Every answer is recomputed from the values parsed back out of the stem text (`_reread`).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import number_text, single_number, retry
from .f10_stem_and_leaf_build import _PAIRS, _bb_text

QUARTILE_NOTE = (
    "Take Q1 and Q3 as the medians of the lower and upper halves of the ordered data, leaving the median out of "
    "both halves when there is an odd number of values."
)


def median(values) -> Fraction:
    s = sorted(values)
    n = len(s)
    return Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)


def quartiles(values) -> tuple[Fraction, Fraction]:
    s = sorted(values)
    n = len(s)
    return median(s[: n // 2]), median(s[(n + 1) // 2 :])


def iqr(values) -> Fraction:
    q1, q3 = quartiles(values)
    return q3 - q1


def _draw_group(rng, stems, n, tight):
    """`n` values over `stems`; a tight group packs most values on the middle stems."""
    centre = len(stems) / 2 - 0.5
    weights = [
        (1 if tight else 3) + (6 if tight and abs(i - centre) <= 0.5 else 0) + rng.randint(0, 2)
        for i in range(len(stems))
    ]
    counts = [1] * len(stems)
    for _ in range(n - len(stems)):
        counts[rng.choices(range(len(stems)), weights=weights)[0]] += 1
    values = []
    for s, c in zip(stems, counts):
        leaves = [rng.randint(0, 9) for _ in range(c)]
        if tight and abs(stems.index(s) - centre) <= 0.5:
            leaves = [rng.randint(2, 7) for _ in range(c)]
        values += [10 * s + x for x in leaves]
    return sorted(values)


def draw_plot(rng):
    """(context, stems, group1 values, group2 values, tight group index), with every row non-empty."""
    ctx = rng.choice([c for c in _PAIRS if c["groups"][0] != "Fertiliser X"])
    lo, hi = ctx["stems"]
    for _ in attempts(500, "two groups with different spreads and medians"):
        nstems = min(rng.randint(4, 5), hi - lo + 1)
        s0 = rng.randint(lo, hi - nstems + 1)
        stems = list(range(s0, s0 + nstems))
        tight = rng.randrange(2)
        a = _draw_group(rng, stems, rng.randint(9, 15), tight == 0)
        b = _draw_group(rng, stems, rng.randint(9, 15), tight == 1)
        rows_a = {s: [v % 10 for v in a if v // 10 == s] for s in stems}
        rows_b = {s: [v % 10 for v in b if v // 10 == s] for s in stems}
        if any(not rows_a[s] or not rows_b[s] for s in stems):
            continue
        if abs(iqr(a) - iqr(b)) < 3 or median(a) == median(b):
            continue
        return ctx, stems, rows_a, rows_b, tight
    raise AssertionError("unreachable")


def plot_text(ctx, stems, rows_a, rows_b):
    ga, gb = ctx["groups"]
    ks = stems[len(stems) // 2]
    la, lb = rows_a[ks][0], rows_b[ks][-1]
    key = f"Key: {la} | {ks} | {lb} means {10 * ks + la} for {ga} and {10 * ks + lb} for {gb}."
    return (
        f"{ctx['stmt']} {key} Plot, rows written as {ga} leaves | stem | {gb} leaves: {_bb_text(rows_a, rows_b)}. "
        "Leaves in each row are in order, smallest nearest the stem."
    )


def reread(text: str, ctx):
    """(values of group 1, values of group 2) parsed back out of the plot text."""
    ga, gb = ctx["groups"]
    body = text.split(f"{ga} leaves | stem | {gb} leaves: ", 1)[1].split(". ", 1)[0]
    a, b = [], []
    for row in body.split("; "):
        left, stem, right = row.split(" | ")
        a += [10 * int(stem) + int(x) for x in left.split()]
        b += [10 * int(stem) + int(x) for x in right.split()]
    return sorted(a), sorted(b)


def _setup(rng):
    ctx, stems, rows_a, rows_b, tight = draw_plot(rng)
    base = plot_text(ctx, stems, rows_a, rows_b)
    a, b = reread(base, ctx)
    check(a == sorted(10 * s + x for s in stems for x in rows_a[s]), "group 1 re-read differs")
    check(b == sorted(10 * s + x for s in stems for x in rows_b[s]), "group 2 re-read differs")
    return ctx, base, a, b


def _places(*vals):
    return 0 if all(Fraction(v).denominator == 1 for v in vals) else 1


def _easy(rng):
    ctx, base, a, b = _setup(rng)
    pick = rng.randrange(2)
    name = ctx["groups"][pick]
    mine, other = (a, b) if pick == 0 else (b, a)
    ans = median(mine)
    s = sorted(mine)
    n = len(s)
    places = _places(ans)
    text = f"{base} Find the median {ctx['val']} for {name}."
    slips = [
        (median(other), "r4g-other-group-read"),
        (Fraction(sum(mine), n), "r4g-mean-for-median"),
        (Fraction(min(mine) + max(mine), 2), "r4g-midrange-for-median"),
        (s[n // 2 - 1] if n % 2 == 0 else s[n // 2 - 1], "r4g-one-middle-value-only"),
        (s[n // 2] % 10, "r4g-leaf-of-median-only"),
        (Fraction(n + 1, 2), "r4g-position-for-median"),
    ]
    working = [
        f"{name} has {n} values; in order they are {', '.join(str(v) for v in s)}.",
        f"The median is {'the middle value' if n % 2 else 'halfway between the two middle values'}: {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


def _medium(rng):
    ctx, base, a, b = _setup(rng)
    pick = rng.randrange(2)
    name = ctx["groups"][pick]
    mine, other = (a, b) if pick == 0 else (b, a)
    q1, q3 = quartiles(mine)
    ans = q3 - q1
    check(ans == iqr(mine), "IQR re-derivation differs")
    s = sorted(mine)
    n = len(s)
    med = median(s)
    places = _places(ans)
    text = f"{base} {QUARTILE_NOTE} Find the interquartile range of the {ctx['val']}s for {name}."
    alt_lo, alt_hi = median(s[: (n + 1) // 2]), median(s[n // 2 :])  # median kept in both halves
    slips = [
        (max(mine) - min(mine), "box-iqr-as-range"),
        (q3, "box-quartile-for-iqr"),
        (q3 - med, "box-iqr-from-median"),
        (alt_hi - alt_lo, "r4g-median-kept-in-both-halves") if n % 2 else (iqr(other), "r4g-other-group-read"),
        (iqr(other), "r4g-other-group-read"),
        ((q3 - q1) / 2, "r4g-semi-interquartile-range"),
    ]
    working = [
        f"In order: {', '.join(str(v) for v in s)}.",
        f"Q1 = {number_text(q1, 1)} and Q3 = {number_text(q3, 1)}.",
        f"IQR = {number_text(q3, 1)} - {number_text(q1, 1)} = {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


def _hard_iqr_gap(rng):
    ctx, base, a, b = _setup(rng)
    ia, ib = iqr(a), iqr(b)
    ans = abs(ia - ib)
    ga, gb = ctx["groups"]
    places = _places(ans)
    text = (
        f"{base} {QUARTILE_NOTE} The group with the smaller interquartile range has the more consistent "
        f"{ctx['val']}s. By how much is the interquartile range of the less consistent group greater than that of the more consistent group?"
    )
    ra, rb = max(a) - min(a), max(b) - min(b)
    slips = [
        (abs(median(a) - median(b)), "box-iqr-difference-of-medians"),
        (abs(ra - rb), "box-iqr-difference-of-ranges"),
        (ia + ib, "box-iqr-difference-summed"),
        (abs(quartiles(a)[1] - quartiles(b)[1]), "box-iqr-difference-of-upper-quartiles"),
        (max(ia, ib), "r4g-one-iqr-not-the-difference"),
    ]
    working = [
        f"IQR of {ga} = {number_text(ia, 1)}; IQR of {gb} = {number_text(ib, 1)}.",
        f"{ga if ia < ib else gb} is more consistent; the difference is {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


def _hard_count_above(rng):
    for _ in attempts(500, "a count of at least 1 and not all"):
        ctx, base, a, b = _setup(rng)
        pick = rng.randrange(2)
        mine, other = (a, b) if pick == 0 else (b, a)
        q3 = quartiles(other)[1]
        count = sum(1 for v in mine if v > q3)
        at = sum(1 for v in mine if v >= q3)
        over_median = sum(1 for v in mine if v > median(other))
        if 1 <= count <= len(mine) - 2 and len({count, at, over_median}) == 3:
            break
    name, oname = ctx["groups"][pick], ctx["groups"][1 - pick]
    text = (
        f"{base} {QUARTILE_NOTE} How many of the {len(mine)} {ctx['val']}s for {name} are greater than the "
        f"upper quartile (Q3) of the {ctx['val']}s for {oname}?"
    )
    slips = [
        (over_median, "r4g-median-for-upper-quartile"),
        (at, "r4g-value-equal-to-quartile-counted"),
        (sum(1 for v in mine if v > quartiles(mine)[1]), "r4g-own-quartile-used"),
        (len(mine) - count, "r4g-count-below-not-above"),
        (count + 1, "r4g-count-off-by-one"),
    ]
    working = [
        f"Q3 of {oname} is {number_text(q3, 1)}.",
        f"{count} of the values for {name} are greater than {number_text(q3, 1)}.",
    ]
    return single_number(text, count, 0, working, slips)


def _hard(rng):
    return _hard_iqr_gap(rng) if rng.random() < 0.6 else _hard_count_above(rng)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_back_to_back_stem",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_back_to_back_stem(rng, tier="easy"):
    return retry(_TIER[tier], rng)
