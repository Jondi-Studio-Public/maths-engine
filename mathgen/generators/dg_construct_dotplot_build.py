"""Generates `dg_construct_dotplot` questions (easy, medium, hard) and the exam chain `dg_construct_dotplot_chain`.

``construct-dot-plots-from-data`` (``statistics-probability``, answer ``bars``): the learner builds a dot plot of a
numerical variable from a list of values and replies with the number of dots above each value on the axis, in order,
written ``Value = dots``. A value on the axis that nobody gave has 0 dots, and a repeated value stacks one dot per
observation (not one dot per distinct value). The `bars` grader reads the reply exactly.

* easy: whole-number data (siblings, shoe sizes) on an axis that runs from the lowest to the highest value.
* medium: data to one decimal place (sprint times, heights in metres), so the ticks step by 0.1.
* hard: data with a gap and a far-out value, so the axis has several empty ticks, and the learner must still give 0 there.
* exam (``dg_construct_dotplot_chain``): from one list of whole numbers: (a) the dots above each value, (b) the mode,
  (c) the median (a half when the count is even), (d) the median minus the mode, following through from (b) and (c).

The counts are chosen first and the list is built from them, then recounted from the question text; the median and mode
are re-derived with `statistics`.
"""

from __future__ import annotations

import statistics
from collections import Counter
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._spread_kit import uses_rng
from ._dg_kit import bars_dist, bars_fields, numd, numtext, p_bars, p_number

AREA = "construct-dot-plots-from-data"

MISCONCEPTIONS = [
    ("dg-dp-one-dot-per-value", "Draws one dot above each different value, so a repeated value is not stacked.", AREA),
    (
        "dg-dp-dots-shifted",
        "Puts the dots one tick to the side of the value they belong to, so every stack is on the wrong value.",
        AREA,
    ),
    (
        "dg-dp-value-as-dots",
        "Uses the value itself as the number of dots above it instead of counting how many times it occurs.",
        AREA,
    ),
    (
        "dg-dp-gap-closed",
        "Squeezes the empty ticks out of the axis, so the stacks after a gap sit one value too low.",
        AREA,
    ),
    ("dg-dp-miscount-one", "Miscounts one stack by one dot when tallying the list.", AREA),
    (
        "dg-dp-mode-as-count",
        "Gives the number of dots in the tallest stack where the value of the mode was asked for.",
        AREA,
    ),
    ("dg-dp-mode-second", "Picks the second tallest stack, or the highest value, as the mode.", AREA),
    (
        "dg-dp-median-as-middle-value",
        "Takes the middle value of the axis, not the middle of the ordered data, as the median.",
        AREA,
    ),
    (
        "dg-dp-median-as-mean-of-extremes",
        "Takes the halfway point between the lowest and highest value as the median.",
        AREA,
    ),
    (
        "dg-dp-median-even-lower",
        "Takes the lower of the two middle values as the median when the number of values is even.",
        AREA,
    ),
    (
        "dg-dp-difference-reversed",
        "Subtracts the median from the mode when the median minus the mode was asked for.",
        AREA,
    ),
    (
        "dg-dp-difference-uses-count",
        "Uses the number of dots at the mode instead of the mode itself in the difference.",
        AREA,
    ),
]

_INT_THEMES = (
    ("Students in a Year 7 class were asked how many brothers and sisters they have. The answers were: {data}.", 0, 6),
    ("A netball club recorded the shoe size of each player: {data}.", 4, 11),
    ("A teacher recorded how many questions each student got wrong in a ten-question quiz: {data}.", 0, 8),
    ("A bus company recorded how many minutes late the morning bus was on each school day: {data}.", 0, 9),
    ("A netballer recorded how many goals she scored in each game of the season: {data}.", 0, 9),
)
_DEC_THEMES = (
    ("Students ran a 40 metre sprint and their times in seconds were: {data}.", 6.0, 0.1),
    ("A class measured the height of each bean seedling in centimetres: {data}.", 4.0, 0.1),
    ("Students were timed for the 50 metre freestyle, in minutes: {data}.", 1.0, 0.1),
)


def _list_text(rng, values, counts, fmt):
    items = [fmt(v) for v, c in zip(values, counts) for _ in range(c)]
    rng.shuffle(items)
    return ", ".join(items)


def _recount(text, cats):
    data = text.split(": ", 1)[1].rstrip(".")
    words = Counter(w.strip() for w in data.split(","))
    return [words[c] for c in cats]


def _ask(cats, what="number of dots above each value"):
    return f"Give the {what} on the axis from {cats[0]} to {cats[-1]}, in order, written like {cats[0]} = 2."


def _bump(counts, i, d):
    c = list(counts)
    c[i] += d
    return c if c[i] >= 0 else None


def _candidates(rng, counts):
    """The usual slips for a dot plot: one dot per distinct value, every stack a tick out, a miscount."""
    nonzero = [i for i, c in enumerate(counts) if c]
    out = [([1 if c else 0 for c in counts], "dg-dp-one-dot-per-value")]
    out.append(([0, *counts[:-1]], "dg-dp-dots-shifted"))
    out.append(([*counts[1:], 0], "dg-dp-dots-shifted"))
    j = rng.choice(nonzero)
    out.append((_bump(counts, j, 1), "dg-dp-miscount-one"))
    out.append((_bump(counts, j, -1), "dg-dp-miscount-one"))
    if 0 in counts[1:-1]:
        z = counts.index(0, 1)
        out.append(([*counts[:z], *counts[z + 1 :], 0], "dg-dp-gap-closed"))
    return out


# ---------------------------------------------------------------------------------------------- easy


def _counts(rng, k, lo=1, hi=5, zero=False, total=(12, 22)):
    for _ in attempts(300, "dot counts"):
        counts = [rng.randint(lo, hi) for _ in range(k)]
        if zero:
            counts[rng.randint(1, k - 2)] = 0
        if total[0] <= sum(counts) <= total[1] and len({c for c in counts if c}) >= 3 and counts[0] and counts[-1]:
            return counts
    raise AssertionError  # pragma: no cover


def _easy(rng):
    stem_t, lo, hi = rng.choice(_INT_THEMES)
    k = rng.randint(5, 7)
    low = rng.randint(lo, max(lo, hi - k + 1))
    values = list(range(low, low + k))
    counts = _counts(rng, k)
    cats = [str(v) for v in values]
    stem = stem_t.format(data=_list_text(rng, values, counts, str))
    check(_recount(stem, cats) == counts, "recount of the list")
    ds = bars_dist(cats, counts, _candidates(rng, counts))
    if len(ds) < 3:
        return None
    return dict(
        text=f"{stem} The data is shown in a dot plot with one dot for each value. {_ask(cats)}",
        answer_kind="bars",
        **bars_fields(cats, counts),
        distractors=ds,
        working=[
            "Tally each value: " + ", ".join(f"{c}: {h}" for c, h in zip(cats, counts)) + ".",
            "Each observation is its own dot, so a repeated value stacks.",
        ],
    )


# ---------------------------------------------------------------------------------------------- medium


def _medium(rng):
    stem_t, start, step = rng.choice(_DEC_THEMES)
    k = rng.randint(5, 6)
    shift = rng.randint(0, 9) / 10  # the first tick is a random tenth, so the data is not always on the same ticks
    values = [round(start + shift + step * i, 1) for i in range(k)]
    counts = _counts(rng, k)
    cats = [f"{v:.1f}" for v in values]
    stem = stem_t.format(data=_list_text(rng, values, counts, lambda v: f"{v:.1f}"))
    check(_recount(stem, cats) == counts, "recount of the list")
    ds = bars_dist(cats, counts, _candidates(rng, counts))
    if len(ds) < 3:
        return None
    return dict(
        text=f"{stem} The data is shown in a dot plot whose axis steps up by 0.1. {_ask(cats)}",
        answer_kind="bars",
        **bars_fields(cats, counts),
        distractors=ds,
        working=[
            "Tally each value: " + ", ".join(f"{c}: {h}" for c, h in zip(cats, counts)) + ".",
            "Match on the whole number and the tenth together: 6.1 and 6.2 are different ticks.",
        ],
    )


# ---------------------------------------------------------------------------------------------- hard


def _hard(rng):
    stem_t, lo, hi = rng.choice([t for t in _INT_THEMES if t[2] - t[1] >= 8])
    k = rng.randint(8, 10)
    low = rng.randint(lo, max(lo, hi - k + 1))
    values = list(range(low, low + k))
    body = rng.randint(3, 4)
    counts = [rng.randint(2, 5) for _ in range(body)] + [0] * (k - body - 1) + [1]  # a far-out value after a gap
    if rng.random() < 0.5:
        counts[rng.randint(1, body - 2)] = 0  # an empty tick inside the main body too
    cats = [str(v) for v in values]
    if sum(counts) < 9 or counts[0] == 0:
        return None
    stem = stem_t.format(data=_list_text(rng, values, counts, str))
    check(_recount(stem, cats) == counts, "recount of the list")
    ds = bars_dist(cats, counts, _candidates(rng, counts))
    if len(ds) < 3:
        return None
    empty = [c for c, h in zip(cats, counts) if h == 0]
    return dict(
        text=f"{stem} The dot plot's axis shows every whole number from {cats[0]} to {cats[-1]}. {_ask(cats)}",
        answer_kind="bars",
        **bars_fields(cats, counts),
        distractors=ds,
        working=[
            "Tally each value: " + ", ".join(f"{c}: {h}" for c, h in zip(cats, counts)) + ".",
            f"The values {', '.join(empty)} have no dots but still have a place on the axis.",
        ],
    )


@generator("dg_construct_dotplot", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
@uses_rng
def dg_construct_dotplot(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(500, f"a {tier} dot plot question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_construct_dotplot_chain", level=12, presets={"exam": {"tier": "exam"}})
@uses_rng
def dg_construct_dotplot_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(600, "a dot plot chain"):
        stem_t, lo, hi = rng.choice(_INT_THEMES)
        k = rng.randint(5, 7)
        low = rng.randint(lo, max(lo, hi - k + 1))
        values = list(range(low, low + k))
        counts = _counts(rng, k, total=(14, 24))
        data = [v for v, c in zip(values, counts) for _ in range(c)]
        modes = statistics.multimode(data)
        if len(modes) != 1:
            continue
        mode = modes[0]
        med = Fraction(statistics.median(data)).limit_denominator(2)
        if med == mode or med.denominator not in (1, 2):
            continue
        cats = [str(v) for v in values]
        stem = stem_t.format(data=_list_text(rng, values, counts, str))
        check(_recount(stem, cats) == counts, "recount of the list")
        n = len(data)
        # the median again by position in the stacks
        ordered = sorted(data)
        by_pos = Fraction(ordered[(n - 1) // 2] + ordered[n // 2], 2)
        check(by_pos == med, "median two ways")
        da = bars_dist(cats, counts, _candidates(rng, counts))
        top = max(counts)
        db = numd(
            mode,
            [
                (top, "dg-dp-mode-as-count"),
                (values[counts.index(sorted(counts)[-2])], "dg-dp-mode-second"),
                (values[-1], "dg-dp-mode-second"),
                (values[0], "dg-dp-mode-second"),
            ],
            lo=0,
        )
        mid_axis = Fraction(values[0] + values[-1], 2)
        lower = ordered[(n - 1) // 2]
        dc = numd(
            med,
            [
                (mid_axis, "dg-dp-median-as-middle-value"),
                (Fraction(ordered[0] + ordered[-1], 2), "dg-dp-median-as-mean-of-extremes"),
                (lower if n % 2 == 0 else None, "dg-dp-median-even-lower"),
                (med + 1, "dg-dp-median-as-middle-value"),
                (med - 1, "dg-dp-median-as-middle-value"),
            ],
            lo=0,
        )
        diff = med - mode
        dd = numd(
            diff,
            [
                (mode - med, "dg-dp-difference-reversed"),
                (med - top, "dg-dp-difference-uses-count"),
                (diff + 1, "dg-dp-miscount-one"),
                (diff - 1, "dg-dp-miscount-one"),
            ],
        )
        if min(len(x) for x in (da, db, dc, dd)) < 2:
            continue
        return dict(
            text=f"{stem} The data is shown in a dot plot with one dot for each value.",
            parts=[
                p_bars(
                    _ask(cats),
                    cats,
                    counts,
                    3,
                    ["Tally each value: " + ", ".join(f"{c}: {h}" for c, h in zip(cats, counts)) + "."],
                    da,
                ),
                p_number(
                    "What is the mode of the data?",
                    mode,
                    1,
                    [f"The tallest stack is at {mode}, with {top} dots."],
                    db,
                    sensible_range=(values[0] - 1, values[-1] + 1),
                ),
                p_number(
                    "What is the median of the data?",
                    med,
                    2,
                    [
                        f"There are {n} values. The middle of the ordered list is at position {numtext(Fraction(n + 1, 2))}, which is {numtext(med)}."
                    ],
                    dc,
                    sensible_range=(values[0] - 1, values[-1] + 1),
                ),
                p_number(
                    "Find the median minus the mode.",
                    diff,
                    1,
                    [f"{numtext(med)} - {mode} = {numtext(diff)}."],
                    dd,
                    follow_through="part_c - part_b",
                    sensible_range=(-20, 20),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover
