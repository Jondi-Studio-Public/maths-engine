"""Grouped frequency tables with class intervals: `r4g_grouped_frequency` (easy, medium, hard).

``grouped-frequency-table-class-intervals`` (``statistics-probability``, answer ``number``). The table is written
in the question as `0 to under 10: 3; 10 to under 20: 8; ...` (class limits, then the frequency); nothing is drawn.

* easy: the midpoint of the modal class (the class with the greatest frequency; it is always unique).
* medium: the estimated mean, from class midpoints, to 1 decimal place.
* hard: the lower limit of the class that contains the median, or the missing frequency of one class when the
  estimated mean is stated.

Every key is re-derived from the table text by `_reread` (limits and frequencies parsed back out of the stem).
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import clear_of_tie, number_text, single_number, retry

CONTEXTS = [
    dict(unit="minutes", what="the time {who} took to travel to school", who="students", lo=0, widths=(5, 10)),
    dict(unit="dollars", what="the weekly pocket money of {who}", who="students", lo=0, widths=(5, 10)),
    dict(unit="kilograms", what="the mass of {who}", who="parcels at a depot", lo=0, widths=(5, 10)),
    dict(unit="millimetres", what="the height of {who}", who="seedlings", lo=100, widths=(10, 20)),
    dict(unit="kilometres", what="the distance cycled in a week by {who}", who="club members", lo=0, widths=(10, 20)),
    dict(unit="seconds", what="the time {who} took to solve a puzzle", who="competitors", lo=0, widths=(20, 10)),
    dict(unit="minutes", what="the waiting time of {who}", who="patients at a clinic", lo=0, widths=(5, 10)),
]


class Table:
    """Class limits `lows[i]` to under `lows[i] + width` with frequencies `freqs[i]`."""

    def __init__(self, lows, width, freqs):
        self.lows, self.width, self.freqs = list(lows), width, list(freqs)

    @property
    def mids(self):
        return [Fraction(2 * lo + self.width, 2) for lo in self.lows]

    @property
    def n(self):
        return sum(self.freqs)

    @property
    def total(self):
        return sum(f * m for f, m in zip(self.freqs, self.mids))

    @property
    def mean(self):
        return self.total / self.n

    def mode_index(self):
        top = max(self.freqs)
        return self.freqs.index(top)

    def median_index(self):
        n = self.n
        first, second = (n + 1) // 2, n // 2 + 1
        cum = 0
        for i, f in enumerate(self.freqs):
            cum += f
            if first <= cum:
                return i if second <= cum else None
        return None

    def text(self, hidden=None):
        parts = []
        for i, (lo, f) in enumerate(zip(self.lows, self.freqs)):
            shown = "x" if i == hidden else str(f)
            parts.append(f"{lo} to under {lo + self.width}: {shown}")
        return "; ".join(parts)


def _reread(text: str):
    """(lows, width, freqs) parsed back out of a stem; the hidden class (if any) reads as None."""
    body = text.split("): frequency: ", 1)[1].split(". ", 1)[0]
    lows, highs, freqs = [], [], []
    for row in body.split("; "):
        m = re.fullmatch(r"(\d+) to under (\d+): (\d+|x)", row)
        lows.append(int(m.group(1)))
        highs.append(int(m.group(2)))
        freqs.append(None if m.group(3) == "x" else int(m.group(3)))
    return lows, highs[0] - lows[0], freqs


def draw_table(rng, *, classes=(5, 7), n_range=None, unique_mode=True, same_median_class=False):
    ctx = rng.choice(CONTEXTS)
    width = rng.choice(ctx["widths"])
    k = rng.randint(*classes)
    lo0 = ctx["lo"] + width * rng.randint(0, 3)
    for _ in attempts(500, "a table with a single modal class"):
        peak = rng.randint(1, k - 2)
        freqs = [max(1, 2 + rng.randint(0, 4) + 3 * min(i, k - 1 - i, 2) - abs(i - peak) // 2) for i in range(k)]
        freqs = [f + rng.randint(0, 3) for f in freqs]
        table = Table([lo0 + width * i for i in range(k)], width, freqs)
        if unique_mode and sorted(freqs)[-1] == sorted(freqs)[-2]:
            continue
        if n_range and not n_range[0] <= table.n <= n_range[1]:
            continue
        if same_median_class and table.median_index() is None:
            continue
        return ctx, table
    raise AssertionError("unreachable")


def _intro(ctx):
    return f"The grouped frequency table shows {ctx['what'].format(who=ctx['who'])}, in {ctx['unit']}."


def _table_sentence(ctx, table, hidden=None):
    return f"Class ({ctx['unit']}): frequency: {table.text(hidden)}."


def _easy(rng):
    ctx, t = draw_table(rng)
    i = t.mode_index()
    mid = t.mids[i]
    places = 0 if mid.denominator == 1 else 1
    mid_row = len(t.freqs) // 2
    text = f"{_intro(ctx)} {_table_sentence(ctx, t)} Find the midpoint of the modal class, in {ctx['unit']}."
    lows, width, freqs = _reread(text)
    check(freqs.index(max(freqs)) == i and Fraction(2 * lows[i] + width, 2) == mid, "modal class re-read differs")
    slips = [
        (t.lows[i], "r4g-class-limit-for-midpoint"),
        (t.lows[i] + t.width, "r4g-class-limit-for-midpoint"),
        (t.freqs[i], "r4g-frequency-for-modal-class"),
        (t.mids[mid_row], "r4g-middle-row-for-modal-class"),
        (t.mids[i] + t.width, "r4g-class-limit-for-midpoint"),
    ]
    working = [
        f"The greatest frequency is {t.freqs[i]}, so the modal class is {t.lows[i]} to under {t.lows[i] + t.width}.",
        f"Midpoint = ({t.lows[i]} + {t.lows[i] + t.width}) / 2 = {number_text(mid, places)}.",
    ]
    return single_number(text, mid, places, working, slips)


def _medium(rng):
    for _ in attempts(500, "a mean clear of a rounding tie"):
        ctx, t = draw_table(rng, classes=(5, 7))
        if clear_of_tie(t.mean, 1):
            break
    mean = t.mean
    check(
        sum(Fraction(f) * (lo + Fraction(t.width, 2)) for f, lo in zip(t.freqs, t.lows)) / sum(t.freqs) == mean,
        "mean re-derivation differs",
    )
    text = f"{_intro(ctx)} {_table_sentence(ctx, t)} Use the class midpoints to estimate the mean, correct to 1 decimal place."
    slips = [
        (sum(t.mids) / len(t.mids), "r4g-midpoints-averaged-without-frequencies"),
        (sum(f * lo for f, lo in zip(t.freqs, t.lows)) / t.n, "r4g-lower-limits-used-as-midpoints"),
        (t.total / len(t.freqs), "r4g-divided-by-number-of-classes"),
        (sum(f * (lo + t.width) for f, lo in zip(t.freqs, t.lows)) / t.n, "r4g-upper-limits-used-as-midpoints"),
        (Fraction(t.n, len(t.freqs)), "r4g-mean-of-the-frequencies"),
    ]

    def plain(v):
        return str(int(v)) if v == int(v) else number_text(v, 1)

    working = [
        "Multiply each midpoint by its frequency and add: "
        + " + ".join(f"{plain(m)} × {f}" for m, f in zip(t.mids, t.freqs))
        + f" = {plain(t.total)}.",
        f"Total frequency = {t.n}, so the estimated mean = {plain(t.total)} / {t.n} = {number_text(mean, 1)}.",
    ]
    return single_number(text, mean, 1, working, slips)


def _median_class(rng):
    ctx, t = draw_table(rng, classes=(5, 7), same_median_class=True)
    i = t.median_index()
    text = (
        f"{_intro(ctx)} {_table_sentence(ctx, t)} "
        f"Find the lower limit of the class that contains the median, in {ctx['unit']}."
    )
    lows, width, freqs = _reread(text)
    cum, run = [], 0
    for f in freqs:
        run += f
        cum.append(run)
    check(next(j for j, c in enumerate(cum) if c >= (sum(freqs) + 1) // 2) == i, "median class re-read differs")
    mode_i = t.mode_index()
    slips = [
        (t.lows[mode_i], "hist-modal-class-for-median-class"),
        (t.lows[len(t.lows) // 2], "hist-middle-class-for-median-class"),
        (t.lows[min(i + 1, len(t.lows) - 1)], "r4g-median-class-off-by-one"),
        (t.lows[max(i - 1, 0)], "r4g-median-class-off-by-one"),
        (t.mids[i], "r4g-midpoint-for-lower-limit"),
    ]
    working = [
        f"There are {t.n} values, so the median is at position {(t.n + 1) / 2:g}.",
        "Cumulative frequencies: " + ", ".join(str(c) for c in cum) + ".",
        f"The cumulative frequency first reaches {(t.n + 1) // 2} in the class {t.lows[i]} to under {t.lows[i] + t.width}.",
    ]
    return single_number(text, t.lows[i], 0, working, slips)


def _missing_frequency(rng):
    for _ in attempts(2000, "a table whose mean is a short decimal"):
        n = rng.choice((20, 25, 50))
        ctx = rng.choice(CONTEXTS)
        width = rng.choice(ctx["widths"])
        k = rng.randint(4, 6)
        lo0 = ctx["lo"] + width * rng.randint(0, 3)
        freqs = [rng.randint(1, 9) for _ in range(k)]
        scale = Fraction(n, sum(freqs))
        freqs = [max(1, round(f * scale)) for f in freqs]
        if sum(freqs) != n:
            continue
        hidden = rng.randrange(1, k - 1)
        if freqs[hidden] < 3:
            continue
        t = Table([lo0 + width * i for i in range(k)], width, freqs)
        mean = t.mean
        if (mean * 100).denominator != 1:
            continue
        # The mean must pin the hidden frequency down (the hidden class midpoint is not the mean).
        if t.mids[hidden] == mean:
            continue
        break
    known_n = n - freqs[hidden]
    known_s = t.total - freqs[hidden] * t.mids[hidden]
    mid_h = t.mids[hidden]
    x = (mean * known_n - known_s) / (mid_h - mean)
    check(x == freqs[hidden], "missing frequency re-derivation differs")
    places = 2 if (mean * 10).denominator != 1 else 1
    mean_text = number_text(mean, places)
    text = (
        f"{_intro(ctx)} {_table_sentence(ctx, t, hidden)} The frequency x of one class is not shown. "
        f"The estimated mean from the class midpoints is {mean_text} {ctx['unit']}. Find x."
    )
    slips = [
        ((mean * n - known_s) / mid_h, "r4g-hidden-class-left-out-of-the-total"),
        ((mean * known_n - known_s) / mid_h, "r4g-hidden-class-left-out-of-the-total"),
        (x + 1, "r4g-frequency-off-by-one"),
        (x - 1, "r4g-frequency-off-by-one"),
        (x + 2, "r4g-frequency-off-by-one"),
        (x * 2, "r4g-frequency-off-by-one"),
    ]
    low_h = t.lows[hidden]
    known_low_s = sum(f * lo for i, (f, lo) in enumerate(zip(t.freqs, t.lows)) if i != hidden)
    if low_h != mean:
        slips.insert(1, ((mean * known_n - known_low_s) / (low_h - mean), "r4g-lower-limits-used-as-midpoints"))
    slips = [s for s in slips if Fraction(s[0]) >= 1]
    working = [
        f"Let x be the frequency. The sum of frequency × midpoint is {number_text(known_s, 1)} + {number_text(mid_h, 0 if Fraction(mid_h).denominator == 1 else 1)}x.",
        f"The total frequency is {known_n} + x, so ({number_text(known_s, 1)} + {number_text(mid_h, 0 if Fraction(mid_h).denominator == 1 else 1)}x) / ({known_n} + x) = {mean_text}.",
        f"Solving gives x = {freqs[hidden]}.",
    ]
    return single_number(text, x, 0, working, slips)


def _hard(rng):
    return _median_class(rng) if rng.random() < 0.5 else _missing_frequency(rng)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_grouped_frequency",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_grouped_frequency(rng, tier="easy"):
    return retry(_TIER[tier], rng)
