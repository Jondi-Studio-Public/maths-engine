"""Generates `dg_compare_displays` questions (easy, medium, hard) and the exam chain `dg_compare_displays_chain`.

``compare-two-data-displays`` (``statistics-probability``, answer ``number``): two groups measured on one variable are
shown in two different displays, Class 7A as a drawn dot plot and Class 7B as a stem-and-leaf plot written in the
question, so the learner has to read both before comparing their centre (mode, median) and spread (range).

* easy: compare one measure from each display: the mode, the range or the number of values (a difference of two readings).
* medium: compare the medians (a half when a group has an even number of values).
* hard: use one group's median to count values in the other group, or find the range of the two groups combined.
* exam (``dg_compare_displays_chain``): (a) the median of 7A from the dot plot, (b) the median of 7B from the stem plot,
  (c) how much higher 7B's median is, following through from (a) and (b), (d) how much greater one range is than the other.

The dot plot is the figure; a registered reader recomputes the answer from the figure's dots and the stem plot printed in
the question, never from the stored key. Medians are also re-derived with `statistics`.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds.data import dot_plot
from ._dg_kit import numd, numtext, p_number

AREA = "compare-two-data-displays"

MISCONCEPTIONS = [
    (
        "dg-cd-mode-as-frequency",
        "Gives how many times the mode occurs (the height of the tallest stack or the number of leaves) instead of the mode itself.",
        AREA,
    ),
    (
        "dg-cd-difference-reversed",
        "Subtracts the larger measure from the smaller, so the difference has the wrong sign.",
        AREA,
    ),
    (
        "dg-cd-difference-added",
        "Adds the two measures instead of finding how much greater one is than the other.",
        AREA,
    ),
    ("dg-cd-range-as-highest", "Gives the highest value as the range instead of taking the lowest away from it.", AREA),
    (
        "dg-cd-leaf-as-value",
        "Reads a leaf as the whole value, ignoring its stem, when reading the stem-and-leaf plot.",
        AREA,
    ),
    (
        "dg-cd-median-axis-middle",
        "Takes the middle of the axis (or of the stems) as the median instead of the middle of the ordered values.",
        AREA,
    ),
    (
        "dg-cd-median-even-lower",
        "Takes the lower of the two middle values as the median when there is an even number of values.",
        AREA,
    ),
    (
        "dg-cd-median-halfway-extremes",
        "Takes the value halfway between the lowest and highest values as the median.",
        AREA,
    ),
    (
        "dg-cd-count-total",
        "Adds the two numbers of values when how many more one group has than the other was asked.",
        AREA,
    ),
    ("dg-cd-count-off-one", "Miscounts the values by one when reading a display.", AREA),
    (
        "dg-cd-ticks-not-dots",
        "Counts the ticks on the dot plot's axis where the number of values (dots) was needed.",
        AREA,
    ),
    (
        "dg-cd-above-includes-equal",
        "Counts values equal to the median as well as the values above it when only values above were asked.",
        AREA,
    ),
    (
        "dg-cd-below-includes-equal",
        "Counts values equal to the median as well as the values below it when only values below were asked.",
        AREA,
    ),
    (
        "dg-cd-combined-range-of-one",
        "Gives the larger of the two groups' ranges as the range of the combined data.",
        AREA,
    ),
    ("dg-cd-combined-range-added", "Adds the two groups' ranges to get the range of the combined data.", AREA),
]

_VARIABLES = (
    ("the number of minutes each student spent on homework last night", "Minutes of homework"),
    ("the time, in seconds, each student took to solve a puzzle", "Time (seconds)"),
    ("the number of push-ups each student did in one minute", "Push-ups"),
    ("the number of points each team scored in a game", "Points scored"),
    ("the number of sit-ups each student did in one minute", "Sit-ups"),
)
_GROUPS = (("Class 7A", "Class 7B"), ("Team Red", "Team Blue"), ("Group A", "Group B"), ("Class 7C", "Class 7D"))


# ---------------------------------------------------------------------------------------------- data


def _expand(counts, low):
    return [low + i for i, c in enumerate(counts) for _ in range(c)]


def _median(values):
    m = statistics.median(values)
    return Fraction(m).limit_denominator(2)


def _unique_mode(values):
    modes = statistics.multimode(values)
    return modes[0] if len(modes) == 1 else None


def _draw_data(rng, higher_b=False):
    """Two data sets on one scale. A is drawn as stacks (so counts), B as a list. Returns (a_values, b_values)."""
    for _ in attempts(500, "two data sets"):
        low = rng.randint(12, 22)
        k = rng.randint(7, 9)
        counts = [rng.randint(0, 4) for _ in range(k)]
        counts[0] = max(counts[0], 1)
        counts[-1] = max(counts[-1], 1)
        a = _expand(counts, low)
        shift = rng.randint(-2, 4)
        n_b = rng.randint(11, 17)
        b = [low + shift + rng.randint(0, k - 1) + rng.choice((0, 0, 1, 2)) for _ in range(n_b)]
        if not (11 <= len(a) <= 16) or max(counts) > 6:
            continue
        if _unique_mode(a) is None or _unique_mode(b) is None:
            continue
        if min(b) < 10 or max(b) > 39 or min(a) < 10:
            continue
        if not 2 <= len({x // 10 for x in b}) <= 3 or len(set(a)) < 5 or len(set(b)) < 5:
            continue
        ma, mb = _median(a), _median(b)
        if higher_b and not mb > ma:
            continue
        if max(a) - min(a) == max(b) - min(b):
            continue
        return counts, low, a, sorted(b)
    raise AssertionError  # pragma: no cover


def _stem_text(values):
    rows: dict[int, list[int]] = {}
    for v in sorted(values):
        rows.setdefault(v // 10, []).append(v % 10)
    stems = sorted(rows)
    full = list(range(stems[0], stems[-1] + 1))
    return "; ".join(f"{s} | {' '.join(str(x) for x in rows.get(s, []))}".rstrip() for s in full)


def _figure(counts, low, names, var):
    high = low + len(counts)
    dots = {low + i: c for i, c in enumerate(counts) if c}
    return dot_plot(
        low=low - 1,
        high=high,
        step=1,
        dots=dots,
        alt=f"A dot plot of the values for {names[0]}, with a stack of dots above each value on a number line.",
        x_label=var[1],
        title=names[0],
    )


def _stem_sentence(names, b):
    first = sorted(b)[0]
    return f"The values for {names[1]} are in this stem-and-leaf plot. Key: {first // 10} | {first % 10} means {first}. {_stem_text(b)}."


def _setup(rng, higher_b=False):
    counts, low, a, b = _draw_data(rng, higher_b)
    var = rng.choice(_VARIABLES)
    names = rng.choice(_GROUPS)
    fig = _figure(counts, low, names, var)
    intro = f"The dot plot shows {var[0]} for {names[0]}. {_stem_sentence(names, b)} "
    return counts, low, a, b, names, fig, intro


def _q(text, fig, answer, working, ds):
    if len(ds) < 3:
        return None
    return dict(
        text=text,
        figure=fig,
        answer_kind="number",
        answer=int(answer) if Fraction(answer).denominator == 1 else float(answer),
        answer_text=numtext(answer),
        answer_latex=numtext(answer),
        working=working,
        distractors=ds,
    )


def _nonneg(answer, cands):
    return numd(answer, cands, lo=0 if answer >= 0 else None)


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    counts, low, a, b, names, fig, intro = _setup(rng)
    na, nb = names
    shape = rng.choice(("mode", "range", "count"))
    if shape == "mode":
        ma, mb = _unique_mode(a), _unique_mode(b)
        if ma == mb:
            return None
        hi_, lo_ = (na, nb) if ma > mb else (nb, na)
        ans = abs(ma - mb)
        ask = f"By how much is the mode of {hi_} greater than the mode of {lo_}?"
        fa, fb = max(counts), sorted(b).count(mb)
        cands = [
            (abs(fa - fb), "dg-cd-mode-as-frequency"),
            (ma + mb, "dg-cd-difference-added"),
            (-ans, "dg-cd-difference-reversed"),
            (abs(ma % 10 - mb % 10), "dg-cd-leaf-as-value"),
            (ans + 1, "dg-cd-count-off-one"),
        ]
        working = [
            f"{na}: the tallest stack is at {ma}. {nb}: the most repeated value is {mb}.",
            f"{max(ma, mb)} - {min(ma, mb)} = {ans}.",
        ]
    elif shape == "range":
        ra, rb = max(a) - min(a), max(b) - min(b)
        hi_, lo_ = (na, nb) if ra > rb else (nb, na)
        ans = abs(ra - rb)
        ask = f"By how much is the range of {hi_} greater than the range of {lo_}?"
        cands = [
            (ra + rb, "dg-cd-difference-added"),
            (-ans, "dg-cd-difference-reversed"),
            (abs(max(a) - max(b)), "dg-cd-range-as-highest"),
            (ans + 1, "dg-cd-count-off-one"),
            (ans - 1, "dg-cd-count-off-one"),
        ]
        working = [f"{na}: {max(a)} - {min(a)} = {ra}. {nb}: {max(b)} - {min(b)} = {rb}.", f"The difference is {ans}."]
    else:
        if len(a) == len(b):
            return None
        hi_, lo_ = (na, nb) if len(a) > len(b) else (nb, na)
        ans = abs(len(a) - len(b))
        ask = f"How many more values does {hi_} have than {lo_}?"
        ticks = len(counts)
        cands = [
            (len(a) + len(b), "dg-cd-count-total"),
            (-ans, "dg-cd-difference-reversed"),
            (abs(ticks - len(b)), "dg-cd-ticks-not-dots"),
            (ans + 1, "dg-cd-count-off-one"),
            (ans - 1, "dg-cd-count-off-one"),
        ]
        working = [
            f"{na} has {len(a)} dots and {nb} has {len(b)} leaves.",
            f"{max(len(a), len(b))} - {min(len(a), len(b))} = {ans}.",
        ]
    return _q(intro + ask, fig, ans, working, _nonneg(ans, cands))


# ---------------------------------------------------------------------------------------------- medium


def _medium(rng):
    counts, low, a, b, names, fig, intro = _setup(rng)
    na, nb = names
    ma, mb = _median(a), _median(b)
    if ma == mb:
        return None
    hi_, lo_ = (na, nb) if ma > mb else (nb, na)
    ans = abs(ma - mb)
    oa, ob = sorted(a), sorted(b)
    mid_axis_a = Fraction(low + low + len(counts) - 1, 2)
    mid_stem_b = Fraction(10 * ((ob[0] // 10 + ob[-1] // 10) // 2) + 5)
    lower_a = Fraction(oa[(len(a) - 1) // 2])
    lower_b = Fraction(ob[(len(b) - 1) // 2])
    ext_a, ext_b = Fraction(oa[0] + oa[-1], 2), Fraction(ob[0] + ob[-1], 2)
    cands = [
        (abs(mid_axis_a - mb), "dg-cd-median-axis-middle"),
        (abs(ma - mid_stem_b), "dg-cd-median-axis-middle"),
        (abs(lower_a - lower_b) if (len(a) % 2 == 0 or len(b) % 2 == 0) else None, "dg-cd-median-even-lower"),
        (abs(ext_a - ext_b), "dg-cd-median-halfway-extremes"),
        (-ans, "dg-cd-difference-reversed"),
        (ma + mb, "dg-cd-difference-added"),
        (ans + 1, "dg-cd-count-off-one"),
    ]
    if len(a) % 2 == 0 or len(b) % 2 == 0:
        cands = [c for c in cands if c[0] is not None]
    ds = _nonneg(ans, cands)
    ask = f"By how much is the median of {hi_} greater than the median of {lo_}?"
    working = [
        f"{na} has {len(a)} values, so its median is at position {numtext(Fraction(len(a) + 1, 2))}: {numtext(ma)}.",
        f"{nb} has {len(b)} values, so its median is at position {numtext(Fraction(len(b) + 1, 2))}: {numtext(mb)}.",
        f"{numtext(max(ma, mb))} - {numtext(min(ma, mb))} = {numtext(ans)}.",
    ]
    check(ma == Fraction(statistics.median(a)).limit_denominator(2), "median of A two ways")
    return _q(intro + ask, fig, ans, working, ds)


# ---------------------------------------------------------------------------------------------- hard


def _hard(rng):
    counts, low, a, b, names, fig, intro = _setup(rng)
    na, nb = names
    shape = rng.choice(("b_above_a", "a_below_b", "combined"))
    ma, mb = _median(a), _median(b)
    if shape == "b_above_a":
        ans = sum(1 for v in b if v > ma)
        eq = sum(1 for v in b if v == ma)
        ask = f"How many values of {nb} are greater than the median of {na}?"
        cands = [
            (ans + eq if eq else None, "dg-cd-above-includes-equal"),
            (len(b) - ans, "dg-cd-below-includes-equal"),
            (len(b) // 2, "dg-cd-median-axis-middle"),
            (ans + 1, "dg-cd-count-off-one"),
            (ans - 1, "dg-cd-count-off-one"),
        ]
        working = [
            f"The median of {na} is {numtext(ma)}.",
            f"In {nb}, the values above {numtext(ma)}: {', '.join(str(v) for v in b if v > ma)}; that is {ans}.",
        ]
    elif shape == "a_below_b":
        ans = sum(1 for v in a if v < mb)
        eq = sum(1 for v in a if v == mb)
        ask = f"How many values of {na} are less than the median of {nb}?"
        cands = [
            (ans + eq if eq else None, "dg-cd-below-includes-equal"),
            (len(a) - ans, "dg-cd-above-includes-equal"),
            (len(a) // 2, "dg-cd-median-axis-middle"),
            (ans + 1, "dg-cd-count-off-one"),
            (ans - 1, "dg-cd-count-off-one"),
        ]
        working = [
            f"The median of {nb} is {numtext(mb)}.",
            f"In {na}, the values below {numtext(mb)} add up to {ans} dot{'' if ans == 1 else 's'}.",
        ]
    else:
        ra, rb = max(a) - min(a), max(b) - min(b)
        ans = max(max(a), max(b)) - min(min(a), min(b))
        if ans in (ra, rb):
            return None
        ask = f"What is the range of all the values of {na} and {nb} together?"
        cands = [
            (max(ra, rb), "dg-cd-combined-range-of-one"),
            (ra + rb, "dg-cd-combined-range-added"),
            (max(max(a), max(b)), "dg-cd-range-as-highest"),
            (ans + 1, "dg-cd-count-off-one"),
            (ans - 1, "dg-cd-count-off-one"),
        ]
        working = [
            f"The highest value in either display is {max(max(a), max(b))} and the lowest is {min(min(a), min(b))}.",
            f"{max(max(a), max(b))} - {min(min(a), min(b))} = {ans}.",
        ]
    if ans == 0:
        return None
    return _q(intro + ask, fig, ans, working, _nonneg(ans, [c for c in cands if c[0] is not None]))


@generator("dg_compare_displays", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def dg_compare_displays(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(500, f"a {tier} compare-displays question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_compare_displays_chain", level=12, presets={"exam": {"tier": "exam"}})
def dg_compare_displays_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a compare-displays chain"):
        counts, low, a, b, names, fig, intro = _setup(rng, higher_b=True)
        na, nb = names
        ma, mb = _median(a), _median(b)
        diff = mb - ma
        ra, rb = max(a) - min(a), max(b) - min(b)
        gap = abs(ra - rb)
        hi_r, lo_r = (na, nb) if ra > rb else (nb, na)
        oa, ob = sorted(a), sorted(b)
        da = numd(
            ma,
            [
                (Fraction(low * 2 + len(counts) - 1, 2), "dg-cd-median-axis-middle"),
                (Fraction(oa[(len(a) - 1) // 2]) if len(a) % 2 == 0 else None, "dg-cd-median-even-lower"),
                (Fraction(oa[0] + oa[-1], 2), "dg-cd-median-halfway-extremes"),
                (ma + 1, "dg-cd-count-off-one"),
                (ma - 1, "dg-cd-count-off-one"),
            ],
            lo=0,
        )
        db = numd(
            mb,
            [
                (Fraction(10 * ((ob[0] // 10 + ob[-1] // 10) // 2) + 5), "dg-cd-median-axis-middle"),
                (Fraction(ob[(len(b) - 1) // 2]) if len(b) % 2 == 0 else None, "dg-cd-median-even-lower"),
                (Fraction(ob[0] + ob[-1], 2), "dg-cd-median-halfway-extremes"),
                (Fraction(ob[(len(b) - 1) // 2] % 10), "dg-cd-leaf-as-value"),
                (mb + 1, "dg-cd-count-off-one"),
            ],
            lo=0,
        )
        dc = numd(
            diff,
            [
                (-diff, "dg-cd-difference-reversed"),
                (ma + mb, "dg-cd-difference-added"),
                (diff + 1, "dg-cd-count-off-one"),
                (diff - 1, "dg-cd-count-off-one"),
                (diff + Fraction(1, 2), "dg-cd-median-even-lower"),
            ],
            lo=None,
        )
        dd = numd(
            gap,
            [
                (ra + rb, "dg-cd-difference-added"),
                (-gap, "dg-cd-difference-reversed"),
                (abs(max(a) - max(b)), "dg-cd-range-as-highest"),
                (gap + 1, "dg-cd-count-off-one"),
                (gap - 1, "dg-cd-count-off-one"),
            ],
            lo=0,
        )
        if min(len(x) for x in (da, db, dc, dd)) < 2 or gap == 0 or diff <= 0:
            continue
        return dict(
            text=intro.rstrip(),
            figure=fig,
            parts=[
                p_number(
                    f"Find the median of {na}.",
                    ma,
                    1,
                    [
                        f"{len(a)} values: the median is at position {numtext(Fraction(len(a) + 1, 2))}, which is {numtext(ma)}."
                    ],
                    da,
                    sensible_range=(low - 1, low + len(counts)),
                ),
                p_number(
                    f"Find the median of {nb}.",
                    mb,
                    2,
                    [
                        f"{len(b)} values: the median is at position {numtext(Fraction(len(b) + 1, 2))}, which is {numtext(mb)}."
                    ],
                    db,
                    sensible_range=(9, 40),
                ),
                p_number(
                    f"By how much is the median of {nb} greater than the median of {na}?",
                    diff,
                    1,
                    [f"{numtext(mb)} - {numtext(ma)} = {numtext(diff)}."],
                    dc,
                    follow_through="part_b - part_a",
                    sensible_range=(-30, 30),
                ),
                p_number(
                    f"By how much is the range of {hi_r} greater than the range of {lo_r}?",
                    gap,
                    2,
                    [
                        f"{na}: {max(a)} - {min(a)} = {ra}. {nb}: {max(b)} - {min(b)} = {rb}.",
                        f"The difference is {gap}.",
                    ],
                    dd,
                    sensible_range=(0, 30),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- the reader

_ROW = re.compile(r"(\d+) \|((?: \d+)*)")


def _read_b(text):
    body = text.split("means", 1)[1].split(". ", 1)[1]
    values = []
    for stem, leaves in _ROW.findall(body):
        values += [int(stem) * 10 + int(x) for x in leaves.split()]
    return values


def _read_a(fig):
    return [d["at"] for d in fig["dots"] for _ in range(d["count"])]


def _read_compare(question) -> str:
    """The answer recomputed from the dot plot's dots and the stem plot printed in the question."""
    text, fig = question.text, question.figure
    a, b = _read_a(fig), _read_b(text)
    if "mode of" in text:
        return numtext(abs(_unique_mode(a) - _unique_mode(b)))
    if "range of all the values" in text:
        return numtext(max(max(a), max(b)) - min(min(a), min(b)))
    if "range of" in text:
        return numtext(abs((max(a) - min(a)) - (max(b) - min(b))))
    if "How many more values" in text:
        return numtext(abs(len(a) - len(b)))
    if "median of" in text and "By how much" in text:
        return numtext(abs(_median(a) - _median(b)))
    if "are greater than the median" in text:
        m = _median(a)
        return str(sum(1 for v in b if v > m))
    if "are less than the median" in text:
        m = _median(b)
        return str(sum(1 for v in a if v < m))
    raise AssertionError(f"unrecognised question: {text}")


figures.register_reader("dg_compare_displays", _read_compare)
