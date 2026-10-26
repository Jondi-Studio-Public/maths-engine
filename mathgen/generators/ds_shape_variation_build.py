"""Generates `f10-shape-and-variation-of-a-display` questions (Year 4, AC9M4ST02: analyse the effectiveness of
different displays; discuss the shape of the data and its variation). Three topics, one per tier:

* ``ds_shape_var_read`` (easy): one drawn display, a dot plot or a column graph of how often each value occurs. The
  learner reads one feature of it as a number: the mode, the range, how many values are shown, how many have the mode
  value, or (when one value sits far from the rest) the value that is the outlier.
* ``ds_shape_var_shape`` (medium): the same kind of display, but the shape matters. Name the true statements about it
  (mode, range, how many values, symmetric, bunched at one end, an outlier), find the range once the outlier is left
  out, or count the values above the mode.
* ``ds_shape_var_compare`` (hard): two groups measured on one variable and drawn in one column graph, a pair of
  columns for each value. Name the true statements comparing them (spread, mode, outlier, symmetry, number of values),
  find how much greater one range is than the other, or find the same gap once an outlier is left out.

Not asked: a written discussion of why one display suits a purpose, or of what a shape means in context. That stays
open-response (rubric) work. Every answer here is a number or a set of statement letters.

A question is built backwards from a chosen shape: ``_sym``, ``_low``, ``_high`` and ``_outlier`` draw the column
heights (frequencies) of that shape; ``_shape_of`` classifies the heights by a plain rule and the generator checks
the class it drew. The registered reader recomputes every answer from the drawn figure and the statements printed
in the question, never from the stored key.
"""

from __future__ import annotations

import itertools
import random
import re
import statistics

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds.data import bar_chart, dot_plot
from ..answers import Distractor
from ._dg_kit import NEAR
from ._spread_kit import _RNG, uses_rng
from .dg_graph_terms_build import _set_dist, _set_fields

AREA = "shape-and-variation-of-a-display"

MISCONCEPTIONS = [
    (
        "ds-sv-mode-as-frequency",
        "Gives how many times the mode occurs (the height of the tallest stack or column) instead of the mode itself.",
    ),
    (
        "ds-sv-mode-as-highest",
        "Takes the highest value shown as the mode instead of the value that occurs most often.",
    ),
    ("ds-sv-range-as-highest", "Gives the highest value as the range instead of taking the lowest away from it."),
    (
        "ds-sv-range-counts-columns",
        "Counts the stacks or columns shown as the range, which also miscounts when a value in between has none.",
    ),
    (
        "ds-sv-total-as-columns",
        "Counts the stacks or columns as the number of values, instead of adding up how many each one holds.",
    ),
    (
        "ds-sv-outlier-kept",
        "Includes the outlier when the range of the rest of the values was asked for.",
    ),
    (
        "ds-sv-outlier-gap-as-range",
        "Gives the gap between the outlier and the rest as the range of the rest of the values.",
    ),
    (
        "ds-sv-outlier-as-neighbour",
        "Names the value next to the outlier, at the edge of the main group, instead of the outlier itself.",
    ),
    (
        "ds-sv-extreme-as-outlier",
        "Calls the highest or lowest value an outlier even though it sits next to the other values.",
    ),
    (
        "ds-sv-symmetric-assumed",
        "Calls the data symmetric because the tallest column is near the middle, without checking both sides match.",
    ),
    (
        "ds-sv-bunch-side-swapped",
        "Mixes up which end the values are bunched at: the tall columns, not the long tail, show where they bunch.",
    ),
    (
        "ds-sv-above-includes-mode",
        "Counts the values equal to the mode as well as the values above it, when only values above it were asked.",
    ),
    (
        "ds-sv-below-for-above",
        "Counts the values below the mode where the values above it were asked.",
    ),
    (
        "ds-sv-spread-as-count",
        "Judges which group has the greater spread by which has more values, not by the distance from lowest to highest.",
    ),
    (
        "ds-sv-spread-as-mode",
        "Judges spread by the difference between the modes, which measures centre, not how spread out the values are.",
    ),
    (
        "ds-sv-ranges-added",
        "Adds the two ranges where how much greater one range is than the other was asked.",
    ),
    (
        "ds-sv-totals-added",
        "Adds the two numbers of values where how many more one group has than the other was asked.",
    ),
    (
        "ds-sv-combined-range-of-one",
        "Gives the larger of the two ranges as the range of both groups together.",
    ),
    (
        "ds-sv-missed-true-statement",
        "Leaves out a statement that is true of the display.",
    ),
]

_CONTEXTS = (
    dict(
        unit="book",
        desc="the number of books each student read last month",
        axis="Books read",
        who="students",
        groups=("Class 4A", "Class 4B"),
    ),
    dict(
        unit="goal",
        desc="the number of goals each player scored this season",
        axis="Goals scored",
        who="players",
        groups=("Team Red", "Team Blue"),
    ),
    dict(
        unit="pet",
        desc="the number of pets each child has at home",
        axis="Pets at home",
        who="children",
        groups=("Class 4C", "Class 4D"),
    ),
    dict(
        unit="time",
        desc="the number of times each student was late to school this term",
        axis="Times late",
        who="students",
        groups=("Class 4E", "Class 4F"),
    ),
    dict(
        unit="sticker",
        desc="the number of stickers each child collected at the fair",
        axis="Stickers collected",
        who="children",
        groups=("Group 1", "Group 2"),
    ),
)


# ---------------------------------------------------------------------------------------------- shapes


def _sym(rng, k):
    """Heights that rise to a single middle column and fall away the same way."""
    half, v, out = (k + 1) // 2, rng.randint(1, 2), []
    for _ in range(half):
        out.append(v)
        v += rng.randint(1, 2)
    return out + out[-2::-1]


def _low(rng, k):
    """Tall at the low end and thinning to a single value at the high end."""
    for _ in attempts(200, "a bunched-low shape"):
        c = [rng.randint(6, 9)]
        for _ in range(k - 1):
            c.append(max(1, c[-1] - rng.randint(1, 3)))
        if c[-1] == 1 and c[0] > c[1] and sum(c) <= 36:
            return c
    raise AssertionError  # pragma: no cover


def _high(rng, k):
    return _low(rng, k)[::-1]


def _cluster(rng, w):
    """The main group beside an outlier: heights that are not mirror images and have one tallest column."""
    for _ in attempts(200, "a main group"):
        c = [rng.randint(1, 6) for _ in range(w)]
        if c != c[::-1] and c.count(max(c)) == 1:
            return c
    raise AssertionError  # pragma: no cover


def _outlier(rng, w, gap, side):
    c = _cluster(rng, w) + [0] * gap + [1]
    return c if side == "high" else c[::-1]


def _trim(counts):
    """(first value's offset, heights with the empty ends cut off)."""
    first = next(i for i, c in enumerate(counts) if c)
    last = max(i for i, c in enumerate(counts) if c)
    return first, counts[first : last + 1]


def _shape_of(counts):
    """The shape class of a list of heights by a plain rule: outlier, sym, low, high or other."""
    _, c = _trim(counts)
    occ = [i for i, v in enumerate(c) if v]
    if len(occ) >= 3 and c[-1] == 1 and occ[-1] - occ[-2] >= 3:
        return "outlier"
    if len(occ) >= 3 and c[0] == 1 and occ[1] - occ[0] >= 3:
        return "outlier"
    if c == c[::-1]:
        return "sym"
    down = all(a >= b for a, b in zip(c, c[1:]))
    up = all(a <= b for a, b in zip(c, c[1:]))
    if down and c[0] > c[1] and c[-1] == 1 and 0 not in c:
        return "low"
    if up and c[-1] > c[-2] and c[0] == 1 and 0 not in c:
        return "high"
    return "other"


def _draw_counts(rng, shape):
    if shape == "sym":
        return _sym(rng, rng.choice((5, 7)))
    if shape == "low":
        return _low(rng, rng.choice((5, 6, 7)))
    if shape == "high":
        return _high(rng, rng.choice((5, 6, 7)))
    if shape == "outlier":
        return _outlier(rng, rng.choice((3, 5)), rng.choice((2, 3)), rng.choice(("high", "low")))
    raise ValueError(f"unknown shape {shape!r}")


# ---------------------------------------------------------------------------------------------- facts


class Data:
    """A group's heights from value ``low`` upwards, and everything read off them."""

    def __init__(self, counts, low):
        self.counts, self.low = list(counts), low
        self.values = [low + i for i, c in enumerate(counts) for _ in range(c)]
        peak = max(counts)
        self.mode = low + counts.index(peak)
        self.freq = peak
        self.total = sum(counts)
        self.lo, self.hi = min(self.values), max(self.values)
        self.range = self.hi - self.lo
        self.columns = sum(1 for c in counts if c)
        self.shape = _shape_of(counts)
        if self.shape == "outlier":
            _, c = _trim(counts)
            at_high = c[-1] == 1 and sum(1 for v in c[-3:] if v) == 1
            self.out_val = self.hi if at_high else self.lo
            rest = [v for v in self.values if v != self.out_val]
            self.rest_range = max(rest) - min(rest)
            self.neighbour = max(rest) if at_high else min(rest)
        check(counts.count(peak) == 1, "a single mode")
        check(statistics.multimode(self.values) == [self.mode], "the mode two ways")
        check(self.range == max(self.values) - min(self.values) and self.total == len(self.values), "range and total")


def _setup_one(rng, shape):
    ctx = rng.choice(_CONTEXTS)
    counts = _draw_counts(rng, shape)
    check(_shape_of(counts) == shape, "the shape that was drawn is the shape that is classed")
    return ctx, Data(counts, rng.randint(1, 4))


def _label(ctx, v):
    """A column's label: the value with its unit ("1 book", "3 books"), so no label is a bare number an answer can match."""
    return f"{v} {ctx['unit']}" + ("" if v == 1 else "s")


def _single_figure(rng, ctx, d, kind):
    k = len(d.counts)
    if kind == "dot":
        dots = {d.low + i: c for i, c in enumerate(d.counts) if c}
        fig = dot_plot(
            low=d.low - 1,
            high=d.low + k,
            step=1,
            dots=dots,
            alt=f"A dot plot of {ctx['desc']}, with a stack of dots above each value on a number line.",
            x_label=ctx["axis"],
        )
        intro = f"The dot plot shows {ctx['desc']}. Each dot is one of the {ctx['who']}."
    else:
        fig = bar_chart(
            categories=[_label(ctx, d.low + i) for i in range(k)],
            series=[{"name": "", "values": d.counts}],
            value_max=max(d.counts) + 1,
            value_step=1,
            alt=f"A column graph of {ctx['desc']}: one column for each value, as tall as the number of {ctx['who']} with it.",
            x_label=ctx["axis"],
            y_label=f"Number of {ctx['who']}",
        )
        intro = f"The column graph shows {ctx['desc']}. The height of each column is the number of {ctx['who']} with that value."
    return fig, intro


_STD_DEGREES = (30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360)
WEIGHTS = (0.5, 1, 0.5, 0.5, 2, 0.5, 0.5, 0.5)


def _lone_count(values):
    """How many (feature, option) pairs have an option alone in its value of a feature a learner could read without
    working: parity, being a power of two, a multiple of 5, digits, how many others are within 1 or 2, centrality."""
    spread = [sum(abs(v - w) for w in values) for v in values]
    feats = [
        [v % 2 for v in values],
        [v > 0 and v & (v - 1) == 0 for v in values],
        [v % 5 == 0 for v in values],
        [v in _STD_DEGREES for v in values],
        [len(str(v)) for v in values],
        [min(sum(abs(v - w) <= 1 for w in values) - 1, 3) for v in values],
        [min(sum(abs(v - w) <= 2 for w in values) - 1, 3) for v in values],
        [
            "central"
            if x == min(spread) and spread.count(x) == 1
            else "outlier"
            if x == max(spread) and spread.count(x) == 1
            else "mid"
            for x in spread
        ],
    ]
    return sum((w if i == 0 else 1) * (f.count(x) == 1) for f, w in zip(feats, WEIGHTS) for i, x in enumerate(f))


def _nums(ans, cands, lo=0):
    """Three distractors for a whole-number answer, chosen so that no option (the key or a wrong one) is the only one
    of its kind in a feature a learner could read without working, and so that the number below the key is drawn at
    random. The named slips and plain near misses are the pool."""
    rng = _RNG.get() or random.Random(ans)
    seen, pool = {ans}, []
    for value, tag in cands:
        if value is not None and value >= lo and value not in seen:
            seen.add(value)
            pool.append((value, tag))
    extra = [ans + d for d in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10)]
    extra += [ans * 2, ans * 4] + ([ans // 2] if ans % 2 == 0 else [])
    for v in extra:
        if v >= lo and v not in seen:
            seen.add(v)
            pool.append((v, NEAR))
    # a draw of trios from the pool: all of them when the pool is small, else a sample (the choice is the cost)
    trios = list(itertools.combinations(pool, 3))
    if len(trios) > 220:
        trios = rng.sample(trios, 220)
    scored = []
    for trio in trios:
        below = sum(v < ans for v, _ in trio)
        slip_credit = 0.3 * sum(t != NEAR for _, t in trio)
        scored.append((_lone_count([ans, *(v for v, _ in trio)]) - slip_credit, below, trio))
    if not scored:
        return []
    best = min(sc for sc, _, _ in scored)
    ranks = []
    for r in range(4):
        in_rank = [sc for sc, below, _ in scored if below == r]
        if in_rank and min(in_rank) <= best + 2:
            ranks.append(r)
    rank = rng.choice(ranks)
    top = min(sc for sc, below, _ in scored if below == rank)
    pick = [t for sc, below, t in scored if below == rank and sc <= top + 1]
    return [Distractor(str(v), tag, v) for v, tag in rng.choice(pick)]


def _num_q(text, fig, ans, working, ds, **extra):
    if len(ds) < 3:
        return None
    try:
        figures.assert_no_leak(fig, [str(ans)])
    except AssertionError:
        return None  # a column's label names the answer
    return dict(
        text=text,
        figure=fig,
        answer_kind="number",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        working=working,
        distractors=ds,
        **extra,
    )


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    ask = rng.choice(("mode", "range", "total", "at_mode", "outlier"))
    shape = "outlier" if ask == "outlier" else rng.choice(("sym", "low", "high", "outlier"))
    ctx, d = _setup_one(rng, shape)
    fig, intro = _single_figure(rng, ctx, d, rng.choice(("dot", "column")))
    who = ctx["who"]
    if ask == "mode":
        text, ans = "What is the mode?", d.mode
        cands = [
            (d.freq, "ds-sv-mode-as-frequency"),
            (d.hi, "ds-sv-mode-as-highest"),
            (d.mode + 1, NEAR),
            (d.mode - 1, NEAR),
            (d.mode + 2, NEAR),
        ]
        working = [f"The mode is the value that occurs most often: the tallest stack is at {d.mode}."]
        if d.freq == d.mode:
            return None
    elif ask == "range":
        text, ans = "What is the range of the values?", d.range
        cands = [
            (d.hi, "ds-sv-range-as-highest"),
            (d.columns, "ds-sv-range-counts-columns"),
            (d.total, "ds-sv-total-as-columns"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        working = [f"Highest value {d.hi}, lowest value {d.lo}: {d.hi} - {d.lo} = {ans}."]
    elif ask == "total":
        text, ans = f"How many {who} are shown altogether?", d.total
        cands = [
            (d.columns, "ds-sv-total-as-columns"),
            (d.freq, "ds-sv-mode-as-frequency"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
            (ans - 2, NEAR),
        ]
        working = ["Add up the " + ("dots" if "dot plot" in intro else "column heights") + f": {ans}."]
    elif ask == "at_mode":
        text, ans = f"How many {who} have the mode value?", d.freq
        cands = [
            (d.mode, "ds-sv-mode-as-frequency"),
            (d.total, "ds-sv-total-as-columns"),
            (d.columns, "ds-sv-total-as-columns"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        working = [f"The mode is {d.mode}; its stack or column holds {ans}."]
        if d.mode == d.freq:
            return None
    else:
        text = "One value is far away from all the others. What is that value?"
        ans = d.out_val
        cands = [
            (d.neighbour, "ds-sv-outlier-as-neighbour"),
            (d.mode, "ds-sv-mode-as-highest"),
            (d.range, "ds-sv-range-as-highest"),
            (ans + 1 if ans + 1 not in (d.hi, d.lo) else ans - 1, NEAR),
            (ans + 2, NEAR),
            (ans - 2, NEAR),
        ]
        working = [f"The value {ans} is separated from the main group by a gap, so it is the outlier."]
    return _num_q(f"{intro} {text}", fig, ans, working, _nums(ans, cands, lo=0))


# ---------------------------------------------------------------------------------------------- statements


def _stmt(text, truth, tag):
    return dict(text=text, truth=truth, tag=tag)


def _single_statements(d, ctx, rng):
    """A true and a false version of each statement about one display; the shape ones are true only for their shape."""
    who = ctx["who"]
    pool = {}
    pool["mode"] = [
        _stmt(f"The mode is {d.mode}.", True, ""),
        _stmt(f"The mode is {d.freq}.", d.freq == d.mode, "ds-sv-mode-as-frequency"),
        _stmt(f"The mode is {d.hi}.", d.hi == d.mode, "ds-sv-mode-as-highest"),
    ]
    pool["range"] = [
        _stmt(f"The range is {d.range}.", True, ""),
        _stmt(f"The range is {d.hi}.", d.hi == d.range, "ds-sv-range-as-highest"),
        _stmt(f"The range is {d.columns}.", d.columns == d.range, "ds-sv-range-counts-columns"),
    ]
    pool["total"] = [
        _stmt(f"A total of {d.total} {who} are shown.", True, ""),
        _stmt(f"A total of {d.columns} {who} are shown.", d.columns == d.total, "ds-sv-total-as-columns"),
    ]
    shape = d.shape
    shapes = [
        ("sym", "The data are symmetric.", "ds-sv-symmetric-assumed"),
        ("low", "The values are bunched at the low end.", "ds-sv-bunch-side-swapped"),
        ("high", "The values are bunched at the high end.", "ds-sv-bunch-side-swapped"),
        ("outlier", "One value is an outlier, far from all the others.", "ds-sv-extreme-as-outlier"),
    ]
    pool["shape"] = [
        _stmt(t, shape == key, tag)
        for key, t, tag in shapes
        if shape != "outlier" or key in ("sym", "outlier")  # beside an outlier, 'bunched' has no clear answer
    ]
    return pool


def _pick_statements(rng, pool):
    """Five statements (one of each of mode, range, total and two shapes) with one to three true, in random order."""
    for _ in attempts(200, "a mix of true and false statements"):
        chosen = []
        for key in ("mode", "range", "total"):
            truths = [s for s in pool[key] if s["truth"]]
            falses = [s for s in pool[key] if not s["truth"]]
            chosen.append(rng.choice(truths if rng.random() < 0.5 or not falses else falses))
        chosen += rng.sample(pool["shape"], 2)
        n_true = sum(s["truth"] for s in chosen)
        if 1 <= n_true <= 3 and len({s["text"] for s in chosen}) == 5:
            rng.shuffle(chosen)
            return chosen
    raise AssertionError  # pragma: no cover


def _set_question(rng, stmts, intro, fig, ask_text):
    letters = "ABCDE"
    lettered = list(zip(letters, stmts))
    key = [letter for letter, s in lettered if s["truth"]]
    lines = " ".join(f"{letter}. {s['text']}" for letter, s in lettered)
    cands = []
    for letter, s in lettered:
        if not s["truth"]:
            cands.append((sorted({*key, letter}), s["tag"] or NEAR))
        else:
            cands.append((sorted(set(key) - {letter}), "ds-sv-missed-true-statement"))
    false_letters = [letter for letter, s in lettered if not s["truth"]]
    true_letters = list(key)
    for f in false_letters:
        for t in true_letters:
            tag = dict(lettered)[f]["tag"] or NEAR
            cands.append((sorted((set(key) - {t}) | {f}), tag))
    ds = _set_dist(key, cands, universe=list(letters))
    if len(ds) < 3:
        return None
    working = [f"{letter} is {'true' if s['truth'] else 'false'}: {s['text']}" for letter, s in lettered]
    part = dict(
        prompt=f"{ask_text} {lines}",
        marks=2,
        working=working,
        distractors=ds,
        **_set_fields(key),
    )
    return dict(text=intro, figure=fig, parts=[part])


# ---------------------------------------------------------------------------------------------- medium


def _medium(rng):
    ask = rng.choice(("statements", "statements", "rest_range", "above_mode"))
    if ask == "rest_range":
        ctx, d = _setup_one(rng, "outlier")
        fig, intro = _single_figure(rng, ctx, d, rng.choice(("dot", "column")))
        ans = d.rest_range
        cands = [
            (d.range, "ds-sv-outlier-kept"),
            (d.range - d.rest_range, "ds-sv-outlier-gap-as-range"),
            (d.total - 1, "ds-sv-total-as-columns"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        working = [
            f"The outlier is {d.out_val}. Leave it out and the rest run from "
            f"{min(v for v in d.values if v != d.out_val)} to {max(v for v in d.values if v != d.out_val)}.",
            f"The range of the rest is {ans}.",
        ]
        return _num_q(
            f"{intro} One value is an outlier. Leave it out. What is the range of the rest of the values?",
            fig,
            ans,
            working,
            _nums(ans, cands, lo=0),
        )
    if ask == "above_mode":
        for _ in attempts(60, "a display with values above the mode"):
            ctx, d = _setup_one(rng, rng.choice(("sym", "low", "high")))
            below = sum(c for i, c in enumerate(d.counts) if d.low + i < d.mode)
            above = d.total - below - d.freq
            if above >= 2:
                break
        else:  # pragma: no cover
            return None
        fig, intro = _single_figure(rng, ctx, d, rng.choice(("dot", "column")))
        ans = above
        cands = [
            (above + d.freq, "ds-sv-above-includes-mode"),
            (below, "ds-sv-below-for-above"),
            (d.total - 1, NEAR),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        working = [
            f"The mode is {d.mode}. Add the {ctx['who']} with a value greater than {d.mode}: {ans}.",
        ]
        return _num_q(
            f"{intro} How many {ctx['who']} have a value greater than the mode?",
            fig,
            ans,
            working,
            _nums(ans, cands, lo=0),
        )
    shape = rng.choice(("sym", "low", "high", "outlier"))
    ctx, d = _setup_one(rng, shape)
    fig, intro = _single_figure(rng, ctx, d, rng.choice(("dot", "column")))
    stmts = _pick_statements(rng, _single_statements(d, ctx, rng))
    return _set_question(rng, stmts, intro, fig, "Which statements are true? Give the letters.")


# ---------------------------------------------------------------------------------------------- hard

_PAIR_SHAPES = ("sym", "low", "high", "outlier")


def _pair_counts(rng, shape):
    """Heights for a group inside a window of six values, with empty columns where the group has no values."""
    if shape == "outlier":
        c = _outlier(rng, 3, 2, rng.choice(("high", "low")))
    elif shape == "sym":
        c = _sym(rng, rng.choice((3, 5)))
    else:
        c = _draw_counts_small(rng, shape)
    pad = rng.randint(0, 6 - len(c))
    return [0] * pad + c + [0] * (6 - len(c) - pad)


def _draw_counts_small(rng, shape):
    k = rng.choice((4, 5, 6))
    return _low(rng, k) if shape == "low" else _high(rng, k)


def _pair_figure(ctx, a, b, names):
    k = 6
    low = a.low
    return bar_chart(
        categories=[_label(ctx, low + i) for i in range(k)],
        series=[{"name": names[0], "values": a.counts}, {"name": names[1], "values": b.counts}],
        value_max=max(max(a.counts), max(b.counts)) + 1,
        value_step=1,
        alt=f"A column graph of {ctx['desc']} for {names[0]} and {names[1]}: a pair of columns for each value, one column for each group.",
        x_label=ctx["axis"],
        y_label=f"Number of {ctx['who']}",
    )


def _pair_setup(rng, shapes=None):
    ctx = rng.choice(_CONTEXTS)
    names = ctx["groups"]
    low = rng.randint(1, 4)
    for _ in attempts(500, "two groups"):
        sa, sb = shapes or (rng.choice(_PAIR_SHAPES), rng.choice(_PAIR_SHAPES))
        a, b = Data(_pair_counts(rng, sa), low), Data(_pair_counts(rng, sb), low)
        if a.range == b.range:
            continue
        return ctx, names, a, b
    raise AssertionError  # pragma: no cover


def _pair_statements(a, b, names):
    na, nb = names
    ra, rb = a.range, b.range
    wide, narrow = (na, nb) if ra > rb else (nb, na)
    swapped = narrow
    more, fewer = (na, nb) if a.total > b.total else (nb, na)
    pool = {}
    pool["spread"] = [
        _stmt(f"{wide} has the greater range.", True, ""),
        _stmt(f"{swapped} has the greater range.", False, "ds-sv-spread-as-count"),
    ]
    if a.total != b.total:
        # the group with more values is not always the more spread out one
        pool["spread"][1] = _stmt(
            f"{more} has the greater range.",
            more == wide,
            "ds-sv-spread-as-count",
        )
    pool["mode"] = [
        _stmt(f"{na} and {nb} have the same mode.", a.mode == b.mode, "ds-sv-spread-as-mode"),
        _stmt(f"The mode of {na} is greater than the mode of {nb}.", a.mode > b.mode, "ds-sv-spread-as-mode"),
    ]
    pool["outlier"] = [
        _stmt(f"{n} has an outlier.", d.shape == "outlier", "ds-sv-extreme-as-outlier") for n, d in ((na, a), (nb, b))
    ]
    pool["sym"] = [
        _stmt(f"The data for {n} are symmetric.", d.shape == "sym", "ds-sv-symmetric-assumed")
        for n, d in ((na, a), (nb, b))
    ]
    pool["count"] = ([_stmt(f"{more} has more values than {fewer}.", True, "")] if a.total != b.total else []) + [
        _stmt(f"{na} and {nb} have the same number of values.", a.total == b.total, "ds-sv-total-as-columns")
    ]
    return pool


def _hard(rng):
    ask = rng.choice(("statements", "statements", "gap", "count", "combined"))
    ctx, names, a, b = _pair_setup(rng)
    fig = _pair_figure(ctx, a, b, names)
    intro = (
        f"The column graph shows {ctx['desc']} for {names[0]} and {names[1]}. "
        f"The height of each column is the number of {ctx['who']} with that value."
    )
    if ask == "gap":
        ans = abs(a.range - b.range)
        cands = [
            (a.range + b.range, "ds-sv-ranges-added"),
            (abs(a.mode - b.mode) if a.mode != b.mode else None, "ds-sv-spread-as-mode"),
            (abs(a.total - b.total) if a.total != b.total else None, "ds-sv-spread-as-count"),
            (abs(a.hi - b.hi) if a.hi != b.hi else None, "ds-sv-range-as-highest"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        wide = names[0] if a.range > b.range else names[1]
        text = f"{intro} One group has the greater spread. How much greater is its range than the range of the other group?"
        working = [
            f"{names[0]}: {a.hi} - {a.lo} = {a.range}. {names[1]}: {b.hi} - {b.lo} = {b.range}.",
            f"{wide} is more spread out; {max(a.range, b.range)} - {min(a.range, b.range)} = {ans}.",
        ]
        return _num_q(text, fig, ans, working, _nums(ans, cands, lo=0))
    if ask == "count":
        if a.total == b.total:
            return None
        ans = abs(a.total - b.total)
        more, fewer = (names[0], names[1]) if a.total > b.total else (names[1], names[0])
        cands = [
            (a.total + b.total, "ds-sv-totals-added"),
            (abs(a.columns - b.columns) if a.columns != b.columns else None, "ds-sv-total-as-columns"),
            (abs(a.freq - b.freq) if a.freq != b.freq else None, "ds-sv-mode-as-frequency"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        text = f"{intro} How many more {ctx['who']} are in {more} than in {fewer}?"
        working = [
            f"{names[0]}: {a.total}. {names[1]}: {b.total}.",
            f"{max(a.total, b.total)} - {min(a.total, b.total)} = {ans}.",
        ]
        return _num_q(text, fig, ans, working, _nums(ans, cands, lo=1))
    if ask == "combined":
        ans = max(a.hi, b.hi) - min(a.lo, b.lo)
        if ans in (a.range, b.range):
            return None
        cands = [
            (max(a.range, b.range), "ds-sv-combined-range-of-one"),
            (a.range + b.range, "ds-sv-ranges-added"),
            (max(a.hi, b.hi), "ds-sv-range-as-highest"),
            (ans + 1, NEAR),
            (ans - 1, NEAR),
            (ans + 2, NEAR),
        ]
        text = f"{intro} What is the range of all the values of {names[0]} and {names[1]} together?"
        working = [
            f"Highest value {max(a.hi, b.hi)}, lowest value {min(a.lo, b.lo)}.",
            f"{max(a.hi, b.hi)} - {min(a.lo, b.lo)} = {ans}.",
        ]
        return _num_q(text, fig, ans, working, _nums(ans, cands, lo=0))
    pool = _pair_statements(a, b, names)
    for _ in attempts(200, "five comparing statements"):
        chosen = [rng.choice(pool[k]) for k in ("spread", "mode", "outlier", "sym", "count")]
        if len({s["text"] for s in chosen}) == 5 and 1 <= sum(s["truth"] for s in chosen) <= 3:
            rng.shuffle(chosen)
            break
    else:  # pragma: no cover
        return None
    return _set_question(rng, chosen, intro, fig, "Which statements are true? Give the letters.")


# ---------------------------------------------------------------------------------------------- topics


def _build(rng, tier, make_one):
    for _ in attempts(500, f"a {tier} shape-and-variation question"):
        d = make_one(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


@generator("ds_shape_var_read", level=4, presets={"easy": {"tier": "easy"}})
@uses_rng
def ds_shape_var_read(rng, tier="easy"):
    if tier != "easy":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng, tier, _easy)


@generator("ds_shape_var_shape", level=4, presets={"medium": {"tier": "medium"}})
@uses_rng
def ds_shape_var_shape(rng, tier="medium"):
    if tier != "medium":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng, tier, _medium)


@generator("ds_shape_var_compare", level=4, presets={"hard": {"tier": "hard"}})
@uses_rng
def ds_shape_var_compare(rng, tier="hard"):
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng, tier, _hard)


# ---------------------------------------------------------------------------------------------- the reader


def _groups_from(fig):
    """[(name, low, counts)] read off a drawn figure: one dot plot, or each series of a column graph."""
    if fig["kind"] == "dot_plot":
        at = {d["at"]: d["count"] for d in fig["dots"]}
        low, high = min(at), max(at)
        return [("", low, [at.get(v, 0) for v in range(low, high + 1)])]
    low = int(fig["categories"][0].split()[0])
    return [(s["name"], low, [int(v) for v in s["values"]]) for s in fig["series"]]


def _statement_truth(text, groups, who_words):
    """True or False for one printed statement, from the figure's groups alone."""
    ds = {n: Data(c, low) for n, low, c in groups}
    first = next(iter(ds.values()))
    if m := re.fullmatch(r"The mode is (\d+)\.", text):
        return int(m.group(1)) == first.mode
    if m := re.fullmatch(r"The range is (\d+)\.", text):
        return int(m.group(1)) == first.range
    if m := re.fullmatch(r"A total of (\d+) \w+ are shown\.", text):
        return int(m.group(1)) == first.total
    if text == "The data are symmetric.":
        return first.shape == "sym"
    if text == "The values are bunched at the low end.":
        return first.shape == "low"
    if text == "The values are bunched at the high end.":
        return first.shape == "high"
    if text == "One value is an outlier, far from all the others.":
        return first.shape == "outlier"
    names = list(ds)
    if m := re.fullmatch(r"(.+) has the greater range\.", text):
        other = next(n for n in names if n != m.group(1))
        return ds[m.group(1)].range > ds[other].range
    if m := re.fullmatch(r"(.+) and (.+) have the same mode\.", text):
        return ds[m.group(1)].mode == ds[m.group(2)].mode
    if m := re.fullmatch(r"The mode of (.+) is greater than the mode of (.+)\.", text):
        return ds[m.group(1)].mode > ds[m.group(2)].mode
    if m := re.fullmatch(r"(.+) has an outlier\.", text):
        return ds[m.group(1)].shape == "outlier"
    if m := re.fullmatch(r"The data for (.+) are symmetric\.", text):
        return ds[m.group(1)].shape == "sym"
    if m := re.fullmatch(r"(.+) has more values than (.+)\.", text):
        return ds[m.group(1)].total > ds[m.group(2)].total
    if m := re.fullmatch(r"(.+) and (.+) have the same number of values\.", text):
        return ds[m.group(1)].total == ds[m.group(2)].total
    raise AssertionError(f"unrecognised statement: {text}")


def split_statements(text):
    """[(letter, statement)] for the lettered statements A. to E. printed at the end of a question, else []."""
    start = re.search(r"(?:^|\s)A\. ", text)
    if not start:
        return []
    pieces = re.split(r"(?<=\.) (?=[B-E]\. )", text[start.end() :])
    return [(letter, piece[3:] if i else piece) for i, (letter, piece) in enumerate(zip("ABCDE", pieces))]


def _read_shape_var(question) -> str:
    text, fig = question.text, question.figure
    groups = _groups_from(fig)
    # A trimmed copy of each group (empty ends cut off) is what Data reads; offsets are kept in `low`.
    trimmed = []
    for n, low, c in groups:
        off, cut = _trim(c)
        trimmed.append((n, low + off, cut))
    lines = split_statements(text)
    if lines:
        true = sorted(letter for letter, s in lines if _statement_truth(s, trimmed, None))
        return "{" + ", ".join(true) + "}"
    d = Data(trimmed[0][2], trimmed[0][1])
    if len(trimmed) == 1 and "range of the rest of the values" in text:
        return str(d.rest_range)
    if "What is the mode" in text:
        return str(d.mode)
    if len(trimmed) == 1 and "What is the range" in text:
        return str(d.range)
    if "shown altogether" in text:
        return str(d.total)
    if "have the mode value" in text:
        return str(d.freq)
    if "What is that value" in text:
        return str(d.out_val)
    if "greater than the mode" in text:
        return str(sum(v > d.mode for v in d.values))
    ds = {n: Data(c, low) for n, low, c in trimmed}
    a, b = ds[groups[0][0]], ds[groups[1][0]]
    if "One group has the greater spread" in text:
        return str(abs(a.range - b.range))
    if "How many more" in text:
        return str(abs(a.total - b.total))
    if "all the values of" in text:
        return str(max(a.hi, b.hi) - min(a.lo, b.lo))
    raise AssertionError(f"unrecognised question: {text}")


for _topic in ("ds_shape_var_read", "ds_shape_var_shape", "ds_shape_var_compare"):
    figures.register_reader(_topic, _read_shape_var)
