"""Generates `misleading-graph-figure` questions: read a line graph, then judge what its vertical axis does.

A newsletter prints a line graph of a quantity over some years, drawn as a ``cartesian`` figure, with a headline. On
most graphs the vertical axis starts well above zero, which makes a small rise look large; on the others it starts at
zero and the headline is fair. The learner reads values off the graph, works out the real rise, and (on the harder
tiers) how much bigger the last point looks than the first, then names what is misleading from five options.

The values and the axis are chosen together so that every plotted value lies on a gridline, the real percentage rise
has at most one decimal place, and the apparent ratio (the height of the last point above the bottom of the graph
divided by the height of the first) is a whole number on a truncated graph.

Ladder (reasoning escalation, not size):
  easy:   read the last value off the graph, then say what the vertical axis does (honest or truncated).
  medium: read the last value, work out the real percentage rise from the first, then name the flaw.
  hard:   the real percentage rise, how many times higher the last point looks than the first on the drawn axis, then
          the flaw; the first and last values are not read for the learner as separate steps.
  exam:   read the first value, read the last value, the real percentage rise (follow-through), then judge the headline.

The plausibility reader re-derives every answer from the figure and the question's own wording, never from the key.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import attempts, check, generator

from ._ir2a_kit import NAMES, chain, mcq_part, number_part

TOPIC = "ir2a_misleading_graph_figure"
EXAM_TOPIC = "exd_x_ir2a_misleading_graph_figure"
_TIERS = ("easy", "medium", "hard")

_CONTEXTS = (
    dict(
        what="sales",
        axis="Sales ($ thousands)",
        headline="Sales have rocketed!",
        noun="the company's sales",
        body="A company",
    ),
    dict(
        what="membership",
        axis="Members",
        headline="Membership soars to new heights!",
        noun="the club's membership",
        body="A sports club",
    ),
    dict(
        what="profit",
        axis="Profit ($ thousands)",
        headline="Profits are through the roof!",
        noun="the cafe's profit",
        body="A cafe",
    ),
    dict(
        what="average maths mark",
        axis="Average maths mark (%)",
        headline="Maths marks shoot up!",
        noun="the school's average maths mark",
        body="A school",
    ),
    dict(
        what="visitors",
        axis="Visitors (thousands)",
        headline="Visitors flood in!",
        noun="the museum's number of visitors",
        body="A museum",
    ),
)


# -- the five flaw options --------------------------------------------------------------------------------------------


def flaw_options(y0: int) -> dict[str, str]:
    return {
        "truncated": (
            f"The vertical axis starts at {y0}, not at 0, so the rise looks larger than it really is."
            if y0
            else "The vertical axis starts too low, so the rise looks smaller than it really is."
        ),
        "honest": "Nothing is misleading: the vertical axis starts at 0 and its steps are equal.",
        "uneven_x": "The horizontal axis is unevenly spaced, so the years look the wrong distance apart.",
        "uneven_y": "The steps on the vertical axis are not equal, so the line is bent out of shape.",
        "wrong_type": "A line graph should never be used for values that change from year to year.",
        "no_values": "The graph shows no values, so nothing can be read from it.",
    }


def _flaw_part(
    y0: int, truncated: bool, marks: int = 2, prompt: str = "What, if anything, is misleading about the graph?"
) -> dict:
    opts = flaw_options(y0)
    key = opts["truncated"] if truncated else opts["honest"]
    wrongs = []
    if truncated:
        wrongs.append((opts["honest"], "ir2a-trusts-a-graph-that-does-not-start-at-zero"))
    else:
        wrongs.append((opts["truncated"], "ir2a-sees-a-flaw-in-an-honest-axis"))
    wrongs += [
        (opts["uneven_x"], "ir2a-blames-the-horizontal-axis"),
        (opts["uneven_y"], "ir2a-blames-the-vertical-steps"),
        (opts["wrong_type"], "ir2a-says-a-line-graph-is-the-wrong-display"),
        (opts["no_values"], "ir2a-says-the-graph-cannot-be-read"),
    ]
    return mcq_part(
        prompt,
        key,
        wrongs,
        marks=marks,
        working=[
            f"The vertical axis starts at {y0}."
            if truncated
            else "The vertical axis starts at 0, so heights are in proportion."
        ],
        keep=4,
    )


# -- drawing the graph ----------------------------------------------------------------------------------------------


def _figure(ctx: dict, years: list[int], values: list[int], y0: int, y1: int, g: int) -> dict:
    points = [{"at": [x, v]} for x, v in zip(years, values)]
    segments = [{"from": [years[i], values[i]], "to": [years[i + 1], values[i + 1]]} for i in range(len(years) - 1)]
    return figures.cartesian(
        x_range=(years[0] - 1, years[-1] + 1),
        y_range=(y0, y1),
        grid=1,
        x_grid=1,
        y_grid=g,
        equal_scale=False,
        points=points,
        segments=segments,
        x_label="Year",
        y_label=ctx["axis"],
        alt="a line graph with one point for each year",
    )


def _draw(rng: random.Random, truncated: bool):
    """(values, y0, y1, g) with every value on a gridline, a clean real percentage rise, and (when truncated) a whole
    apparent ratio of at least 2."""
    for _ in attempts(5000, "a graph with a clean rise"):
        n = rng.choice([5, 6])
        if truncated:
            g = rng.choice([1, 2, 5, 10])
            rows = rng.randint(8, 11)
            y0 = rng.choice(range(60, 400, g)) // g * g
            y1 = y0 + rows * g
            a = rng.randint(1, 2)
            first = y0 + a * g
            last = first + g * rng.randint(3, rows - a - 2)
            ratio = Fraction(last - y0, first - y0)
            if ratio.denominator != 1 or ratio < 2:
                continue
        else:
            g = rng.choice([5, 10, 20, 25, 50])
            y0 = 0
            m1 = rng.randint(2, 6)
            first = g * m1
            last = first + g * rng.randint(2, 5)
            y1 = (last // g + 1) * g
            if y1 // g > 12:
                continue
        pct = Fraction((last - first) * 100, first)
        if (pct * 10).denominator != 1 or pct < 1:
            continue
        middle = sorted(rng.randint(first // g, last // g) * g for _ in range(n - 2))
        values = [first, *middle, last]
        if len(set(values)) < 3:
            continue
        return values, y0, y1, g
    raise AssertionError("unreachable")


def _years(rng, n):
    start = rng.choice([2015, 2016, 2017, 2018, 2019])
    return list(range(start, start + n))


def _pct_part(first, last, y_first, y_last, marks=2, follow=None):
    pct = Fraction((last - first) * 100, first)
    return number_part(
        prompt=f"By what percentage did the value rise from {y_first} to {y_last}? Give your answer to 1 decimal place.",
        exact=pct,
        places=1,
        marks=marks,
        working=[f"({last} - {first}) / {first} x 100 = {float(pct):.1f}%."],
        wrong=[
            (last - first, "ir2a-gives-the-rise-as-a-percentage"),
            (Fraction(last * 100, first), "ir2a-gives-the-final-value-as-a-percentage-of-the-first"),
            (Fraction((last - first) * 100, last), "ir2a-divides-by-the-final-value"),
        ],
        follow_through=follow,
        sensible_range=(0, None),
    )


def _ratio_part(first, last, y0, marks=2):
    ratio = Fraction(last - y0, first - y0)
    return number_part(
        prompt="On the graph as drawn, the height of the last point above the bottom of the graph is how many times the height of the first point above the bottom? Give your answer to 1 decimal place.",
        exact=ratio,
        places=1,
        marks=marks,
        working=[
            f"Heights above the bottom of the graph: {last} - {y0} = {last - y0} and {first} - {y0} = {first - y0}.",
            f"{last - y0} / {first - y0} = {float(ratio):.1f}.",
        ],
        wrong=[
            (Fraction(last, first), "ir2a-measures-heights-from-zero-not-from-the-axis"),
            (last - y0, "ir2a-gives-one-height-not-the-ratio"),
            (Fraction(first - y0, last - y0), "inverted-ratio"),
        ],
        sensible_range=(0, None),
    )


def _stem(ctx, name, years, headline_tail=""):
    return (
        f"{ctx['body']} prints the line graph shown, with the headline “{ctx['headline']}”. "
        f"The graph shows {ctx['noun']} for each year from {years[0]} to {years[-1]}.{headline_tail}"
    )


def _build(rng, tier):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    truncated = rng.random() < (0.5 if tier == "easy" else 0.8)
    values, y0, y1, g = _draw(rng, truncated)
    years = _years(rng, len(values))
    fig = _figure(ctx, years, values, y0, y1, g)
    first, last = values[0], values[-1]
    stem = _stem(ctx, name, years)
    read = number_part(
        prompt=f"What value does the graph show for {years[-1]}?",
        exact=last,
        places=0,
        marks=1,
        working=[f"The point for {years[-1]} is at {last}."],
        wrong=[
            (values[-2], "ir2a-reads-the-neighbouring-value"),
            (last - g, "ir2a-reads-the-gridline-below"),
            (last + g, "ir2a-reads-the-gridline-above"),
            (y0, "ir2a-reads-the-bottom-of-the-axis"),
        ],
        sensible_range=None,
    )
    if tier == "easy":
        parts = [read, _flaw_part(y0, truncated)]
    elif tier == "medium":
        parts = [read, _pct_part(first, last, years[0], years[-1]), _flaw_part(y0, truncated)]
    else:
        parts = [
            _pct_part(first, last, years[0], years[-1]),
            _ratio_part(first, last, y0),
            _flaw_part(y0, truncated, 2),
        ]
    d = chain(stem, *parts)
    d["figure"] = fig
    return d


@generator(TOPIC, level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_misleading_graph_figure(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return _build(rng, tier)


@generator(EXAM_TOPIC, level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_misleading_graph_figure(rng: random.Random, tier: str = "exam") -> dict:
    """Read the first and last values, the real percentage rise, then judge the headline."""
    check(tier == "exam", "only the exam tier exists")
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    truncated = rng.random() < 0.75
    values, y0, y1, g = _draw(rng, truncated)
    years = _years(rng, len(values))
    fig = _figure(ctx, years, values, y0, y1, g)
    first, last = values[0], values[-1]
    stem = _stem(ctx, name, years)
    pa = number_part(
        prompt=f"What value does the graph show for {years[0]}?",
        exact=first,
        places=0,
        marks=1,
        working=[f"The point for {years[0]} is at {first}."],
        wrong=[
            (values[1], "ir2a-reads-the-neighbouring-value"),
            (first - g, "ir2a-reads-the-gridline-below"),
            (first + g, "ir2a-reads-the-gridline-above"),
            (y0, "ir2a-reads-the-bottom-of-the-axis"),
        ],
        sensible_range=None,
    )
    pb = number_part(
        prompt=f"What value does the graph show for {years[-1]}?",
        exact=last,
        places=0,
        marks=1,
        working=[f"The point for {years[-1]} is at {last}."],
        wrong=[
            (values[-2], "ir2a-reads-the-neighbouring-value"),
            (last - g, "ir2a-reads-the-gridline-below"),
            (last + g, "ir2a-reads-the-gridline-above"),
            (y0, "ir2a-reads-the-bottom-of-the-axis"),
        ],
        sensible_range=None,
    )
    pc = _pct_part(first, last, years[0], years[-1], marks=2, follow="(part_b - part_a)/part_a*100")
    pct = Fraction((last - first) * 100, first)
    ratio = Fraction(last - y0, first - y0)
    if truncated:
        key = f"The headline is misleading: the axis starts at {y0}, so the last point looks {float(ratio):.1f} times as high as the first, but the real rise is only {float(pct):.1f}%."
        wrongs = [
            (
                f"The headline is fair: the graph shows the last point {float(ratio):.1f} times as high as the first.",
                "ir2a-trusts-a-graph-that-does-not-start-at-zero",
            ),
            (
                f"The headline is misleading: the horizontal axis is unevenly spaced, and the real rise is {float(pct):.1f}%.",
                "ir2a-blames-the-horizontal-axis",
            ),
            (
                f"The headline is fair: the value rose by {float(pct):.1f}%, which is large.",
                "ir2a-calls-a-small-rise-large",
            ),
            (
                f"The headline is misleading: the real rise is {float(Fraction(last * 100, first)):.1f}%, not what the graph shows.",
                "ir2a-gives-the-final-value-as-a-percentage-of-the-first",
            ),
        ]
    else:
        key = f"The headline is fair: the axis starts at 0, so the last point is {float(ratio):.1f} times as high as the first, a rise of {float(pct):.1f}%."
        wrongs = [
            (
                f"The headline is misleading: the axis starts at {y0}, so the rise looks larger than the real {float(pct):.1f}%.",
                "ir2a-sees-a-flaw-in-an-honest-axis",
            ),
            (
                f"The headline is misleading: the horizontal axis is unevenly spaced, and the real rise is {float(pct):.1f}%.",
                "ir2a-blames-the-horizontal-axis",
            ),
            (
                f"The headline is fair: the value rose by {float(Fraction(last * 100, first)):.1f}%.",
                "ir2a-gives-the-final-value-as-a-percentage-of-the-first",
            ),
            (
                f"The headline is fair: the last point is {float(Fraction(first, last)):.1f} times as high as the first.",
                "inverted-ratio",
            ),
        ]
    pd = mcq_part(
        "Which statement is the best judgement of the headline?",
        key,
        wrongs,
        marks=2,
        working=["Compare the real rise with how the graph looks."],
        keep=4,
    )
    d = chain(stem, pa, pb, pc, pd)
    d["figure"] = fig
    return d


# -- the figure reader -----------------------------------------------------------------------------------------------


def _read(question) -> str:
    """Every answer re-derived from the figure and the question's wording, in the labelled form a reply takes."""
    fig = question.figure
    values = {p["at"][0]: p["at"][1] for p in fig["points"]}
    xs = sorted(values)
    y0 = fig["y_range"][0]
    first, last = values[xs[0]], values[xs[-1]]
    truncated = y0 != 0
    pct = Fraction((last - first) * 100, first)
    ratio = Fraction(last - y0, first - y0)
    labels = {p.label: p.prompt for p in question.parts}
    out = []
    for label in sorted(labels):
        prompt = labels[label]
        if prompt.startswith("What value does the graph show for"):
            year = int(re.search(r"for (\d{4})", prompt).group(1))
            out.append(f"({label}) {values[year]}")
        elif prompt.startswith("By what percentage"):
            out.append(f"({label}) {float(pct):.1f}")
        elif prompt.startswith("On the graph as drawn"):
            out.append(f"({label}) {float(ratio):.1f}")
        elif prompt.startswith("What, if anything"):
            opts = flaw_options(y0)
            out.append(f"({label}) {opts['truncated'] if truncated else opts['honest']}")
        else:
            if truncated:
                text = f"The headline is misleading: the axis starts at {y0}, so the last point looks {float(ratio):.1f} times as high as the first, but the real rise is only {float(pct):.1f}%."
            else:
                text = f"The headline is fair: the axis starts at 0, so the last point is {float(ratio):.1f} times as high as the first, a rise of {float(pct):.1f}%."
            out.append(f"({label}) {text}")
    return "; ".join(out)


figures.register_reader(TOPIC, _read)
figures.register_reader(EXAM_TOPIC, _read)
