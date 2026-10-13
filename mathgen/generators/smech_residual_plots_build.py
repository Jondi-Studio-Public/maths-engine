"""Residual plots and the choice of a linear model: `smech_residual_plots` (easy, medium, hard) and
`smech_residual_plots_exam`.

Skill: residual-plot-linear-model-check. A least-squares line is appropriate only when the residuals show no pattern.
The skill asks for that judgement, always as a multiple-choice verdict with its reason, and in the exam chain after two
typed computations.

Tiers, each a new decision:

* easy: a residual plot (a coordinate figure) that is either scattered at random about zero or forms a curve.
* medium: the same figure with a third pattern, residuals that fan out as x grows, so the plot has to be read, not just
  checked for a shape.
* hard: no figure. The residuals are listed in a table, with the coefficient of determination r² stated. A high r²
  beside curved residuals (or a low r² beside patternless ones) tempts the wrong verdict.
* exam: five observations and a stated line. Predict, find the residual (follow-through), judge the fit from all five
  residuals, then find the sum of the squares of the residuals.

## Construction

Residuals are built to a named pattern and then classified again by `_classify`, a different route that looks only at
the signs and sizes (the figure's reader uses it on the plotted points alone). A pattern is kept only if
`_classify` agrees, so the stated verdict is the one the plot shows. The figure's alt text describes only the axes,
never the pattern. The exam chain reuses the exact least-squares data of `smech_residuals`.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .. import figures
from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from . import smech_residuals_build as _res
from ._smech_kit import dec, number_fields, number_part

_N = 9
_RANDOM = "Appropriate: the residuals are scattered at random about zero, with no pattern"
_CURVED = "Not appropriate: the residuals form a curved pattern"
_FAN = "Not appropriate: the residuals fan out as x increases"
_PATTERN_GOOD = "Appropriate: the residuals follow a clear pattern"
_NONZERO = "Not appropriate: some of the residuals are not zero"
_OPTIONS = {"random": _RANDOM, "curved": _CURVED, "fan": _FAN}


def _tenths(x: float) -> Fraction:
    return Fraction(round(x * 10), 10)


def _runs(res) -> int:
    signs = [r > 0 for r in res]
    return 1 + sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _growth(res) -> float:
    third = max(1, len(res) // 3)
    first = sum(abs(r) for r in res[:third]) / third
    last = sum(abs(r) for r in res[-third:]) / third
    return float(last / first)


def _classify(res) -> str:
    """The pattern a residual list shows, from its signs and sizes alone."""
    if any(r == 0 for r in res):
        return "unclear"
    runs, growth = _runs(res), _growth(res)
    if runs <= 3:
        return "curved"
    if growth >= 2.5:
        return "fan"
    if runs >= 5 and 0.5 <= growth <= 2.0:
        return "random"
    return "unclear"


def _draw_pattern(rng, pattern: str, n: int = _N):
    """Residuals (in tenths, no zeros) that show `pattern`, accepted only if `_classify` agrees."""
    for _ in attempts(2000, f"residuals showing a {pattern} pattern"):
        if pattern == "random":
            res = [_tenths(rng.choice((-1, 1)) * rng.choice((3, 5, 8, 10, 12, 15, 18, 22, 24)) / 10) for _ in range(n)]
        elif pattern == "curved":
            c, flip = rng.choice((0.2, 0.3, 0.4, 0.5)), rng.choice((-1, 1))
            mid = (n + 1) / 2
            res = [_tenths(flip * (c * ((i - mid) ** 2 - 6) + rng.uniform(-0.25, 0.25))) for i in range(1, n + 1)]
        else:
            g = rng.choice((0.25, 0.3, 0.35))
            res = [_tenths(rng.choice((-1, 1)) * (0.2 + g * i * rng.uniform(0.8, 1.2))) for i in range(1, n + 1)]
        if _classify(res) == pattern and abs(sum(res)) <= 2:
            return res
    raise AssertionError("unreachable")  # pragma: no cover


_CONTEXTS = (
    ("hours of study per week", "test score"),
    ("daily maximum temperature", "ice-cream sales"),
    ("age of a car in years", "advertised price"),
    ("hours of sunshine", "solar power output"),
    ("training sessions per week", "5 km running time"),
    ("rainfall in a month", "dam inflow"),
    ("length of a fish", "mass of the fish"),
    ("weeks since planting", "height of a seedling"),
    ("distance from the city centre", "rent for a flat"),
    ("number of staff on a shift", "orders filled"),
    ("hours of practice", "words typed per minute"),
    ("size of a house block", "council rates"),
)
_STEMS = {
    "easy": (
        "A least-squares line is fitted to data on {y} against {x}, and the residuals are plotted against {x}. "
        "Which statement about using a linear model for these data is correct?",
        "The residuals from a least-squares line fitted to {y} and {x} are plotted against {x}. "
        "Which statement about fitting a linear model to these data is correct?",
        "A student fits a least-squares line to data on {y} against {x} and plots the residuals against {x}. "
        "What can be concluded about using a linear model?",
    ),
    "medium": (
        "Residuals from a least-squares line for {y} against {x} are plotted against {x}. "
        "Which conclusion about the linear model does the plot support?",
        "An analyst plots the residuals after fitting a line to {y} and {x}. "
        "Which statement about the suitability of a linear model is correct?",
        "The residuals for a line fitted to {y} against {x} are plotted against {x}. "
        "Which statement best describes whether a linear model is appropriate?",
    ),
}


def _figure(res, x_label="x"):
    top = math.ceil(float(max(abs(r) for r in res))) + 1
    return figures.cartesian(
        x_range=(0, len(res) + 1),
        y_range=(-top, top),
        grid=1,
        y_grid=1 if top <= 6 else 2,
        equal_scale=False,
        x_label=x_label,
        y_label="Residual",
        title="Residual plot",
        points=[{"at": [i + 1, float(r)]} for i, r in enumerate(res)],
        lines=[{"through": [[0, 0], [1, 0]]}],
        alt=(
            "A residual plot: x on the horizontal axis from 0 to "
            f"{len(res) + 1}, the residual on the vertical axis, a horizontal line at zero and {len(res)} plotted points."
        ),
    )


def _read_figure(question) -> str:
    """Re-derive the verdict from the plotted points alone."""
    pts = sorted(question.figure["points"], key=lambda p: p["at"][0])
    res = [Fraction(str(p["at"][1])) for p in pts]
    return _OPTIONS[_classify(res)]


_MISREAD = {  # (true pattern, chosen option) -> the error the choice stands for
    ("random", "curved"): "mech-pattern-seen-in-random-scatter",
    ("random", "fan"): "mech-pattern-seen-in-random-scatter",
    ("curved", "random"): "mech-pattern-read-as-random-scatter",
    ("fan", "random"): "mech-pattern-read-as-random-scatter",
    ("curved", "fan"): "mech-fan-and-curve-confused",
    ("fan", "curved"): "mech-fan-and-curve-confused",
}


def _plot_question(rng, patterns, tier):
    truth = rng.choice(patterns)
    res = _draw_pattern(rng, truth)
    x_name, y_name = rng.choice(_CONTEXTS)
    fig = _figure(res, x_name)
    key = _OPTIONS[truth]
    wrong = [Distractor(_OPTIONS[p], _MISREAD[(truth, p)]) for p in patterns if p != truth]
    wrong.append(Distractor(_PATTERN_GOOD, "mech-pattern-taken-as-good-fit"))
    if truth == "random":
        wrong.append(Distractor(_NONZERO, "mech-nonzero-residuals-mean-poor-fit"))
    else:
        wrong.append(Distractor("Appropriate: the residuals add to about zero", "mech-sum-zero-taken-as-linear"))
    wrong = wrong[:3]
    text = phrase(rng, _STEMS[tier], x=x_name, y=y_name)
    check(_read_figure(type("Q", (), {"figure": fig})) == key, "the figure does not show the keyed pattern")
    working = [f"The residuals show a {truth} pattern, so: {key[0].lower()}{key[1:]}."]
    return dict(
        text=text, answer=key, answer_text=key, answer_kind="mcq", distractors=wrong, figure=fig, working=working
    )


# ------------------------------------------------------------------------------------------------ hard


_TABLE_STEMS = (
    "A least-squares line is fitted to nine observations, and the coefficient of determination is r² = {r2}. "
    "The residuals, in order of x, are: {listed}. Which statement about using a linear model for these data "
    "is correct?",
    "Nine observations are modelled by a least-squares line with r² = {r2}. The residuals, in order of x, are: "
    "{listed}. Which conclusion about the linear model is correct?",
    "A least-squares line fitted to nine data points has a coefficient of determination of r² = {r2}. Its "
    "residuals, in order of x, are: {listed}. What can be concluded about using a linear model?",
)


def _table_question(rng):
    truth = rng.choice(("curved", "random"))
    res = _draw_pattern(rng, truth)
    r2 = Fraction(rng.randint(90, 98), 100) if truth == "curved" else Fraction(rng.randint(30, 55), 100)
    listed = ", ".join(f"x = {i + 1}: {dec(r)}" for i, r in enumerate(res))
    check(_classify(res) == truth, "table disagrees with the pattern")
    if truth == "curved":
        key = f"Not appropriate: the residuals form a curved pattern, even though r² = {dec(r2)} is high"
        wrong = [
            Distractor(f"Appropriate: r² = {dec(r2)} is high, so the line fits well", "mech-r-squared-trusted"),
            Distractor("Appropriate: the residuals add to about zero", "mech-sum-zero-taken-as-linear"),
            Distractor("Not appropriate: some of the residuals are large", "mech-large-residuals-mean-poor-fit"),
        ]
    else:
        key = f"Appropriate: the residuals show no pattern; r² = {dec(r2)} only means the predictions are less precise"
        wrong = [
            Distractor(f"Not appropriate: r² = {dec(r2)} is low", "mech-low-r-squared-rejects"),
            Distractor("Not appropriate: the residuals form a curved pattern", "mech-pattern-seen-in-random-scatter"),
            Distractor(
                "Not appropriate: some residuals are positive and some are negative", "mech-sign-mix-taken-as-pattern"
            ),
        ]
    text = phrase(rng, _TABLE_STEMS, r2=dec(r2), listed=listed)
    signs = "".join("+" if r > 0 else "-" for r in res)
    working = [f"Signs of the residuals in order of x: {signs}.", f"The residuals show a {truth} pattern, so: {key}."]
    return dict(text=text, answer=key, answer_text=key, answer_kind="mcq", distractors=wrong, working=working)


_TIERS = {
    "easy": lambda rng: _plot_question(rng, ("random", "curved"), "easy"),
    "medium": lambda rng: _plot_question(rng, ("random", "curved", "fan"), "medium"),
    "hard": _table_question,
}


@generator("smech_residual_plots", level=17, presets={t: {"tier": t} for t in _TIERS})
def smech_residual_plots(rng, tier: str = "easy"):
    """Is a linear model appropriate? Read the residual plot (or table) for a pattern."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


figures.register_reader("smech_residual_plots", _read_figure)


# ------------------------------------------------------------------------------------------------ exam


@generator("smech_residual_plots_exam", level=17, presets={"exam": {}})
def smech_residual_plots_exam(rng):
    """Five observations: predicted value, residual, the verdict from the five residuals, sum of squares."""
    want = rng.choice(("curved", "random"))
    for _ in attempts(2000, f"a five-point data set with {want} residuals"):
        ctx, xs, ys, a, b, res = _res._dataset(rng)
        runs = _runs(res)
        if (want == "curved" and runs == 3 and res[2] * res[1] > 0) or (want == "random" and runs >= 4):
            break
    k = rng.choice((0, 1, 3, 4)) if want == "random" else 2
    k = 2 if want == "curved" else k
    pred = a + b * xs[k]
    others = [(x, r) for i, (x, r) in enumerate(zip(xs, res)) if i != k]
    listed = ", ".join(f"x = {x}: {dec(r)}" for x, r in others)
    sse = sum(r * r for r in res)
    other_sse = sum(r * r for _, r in others)
    check(ys[k] - pred == res[k] and sse == res[k] ** 2 + other_sse, "second route disagrees")
    stem = (
        f"{ctx[0][0].upper() + ctx[0][1:]} (x) and {ctx[1]} (y) are recorded for five observations: {_res._points(xs, ys)}. "
        f"The least-squares line is {_res._line_text(a, b, ctx)}. A residual is the observed value minus the "
        f"predicted value. The residuals for four of the observations are {listed}."
    )
    part_a = number_part(
        f"Find the predicted {ctx[1]} when {ctx[0]} is {xs[k]}, correct to 1 decimal place.",
        pred,
        1,
        1,
        [f"{dec(a)} + ({dec(b)}) × {xs[k]} = {dec(pred)}."],
        [
            (Fraction(ys[k]), "mech-observed-used-as-predicted"),
            (b * xs[k], "mech-intercept-left-out"),
            (a + b, "mech-x-value-replaced-by-one"),
            (b + a * xs[k], "mech-intercept-and-slope-swapped"),
        ],
        sensible=(0, None),
    )
    part_b = number_part(
        f"Find the residual for the observation with {ctx[0]} equal to {xs[k]}, correct to 1 decimal place.",
        res[k],
        1,
        1,
        [f"Residual = {ys[k]} - {dec(pred)} = {dec(res[k])}."],
        [
            (pred - ys[k], "residual-sign-reversed"),
            (pred, "mech-predicted-reported-as-residual"),
            (Fraction(ys[k]), "mech-observed-reported-as-residual"),
            (ys[k] - b * xs[k], "mech-intercept-left-out"),
        ],
        follow=f"{ys[k]} - part_a",
    )
    if want == "curved":
        key = _CURVED
        wrong = [
            Distractor(_RANDOM, "mech-pattern-read-as-random-scatter"),
            Distractor(_PATTERN_GOOD, "mech-pattern-taken-as-good-fit"),
            Distractor("Appropriate: the residuals add to zero", "mech-sum-zero-taken-as-linear"),
        ]
    else:
        key = _RANDOM
        wrong = [
            Distractor(_CURVED, "mech-pattern-seen-in-random-scatter"),
            Distractor(_NONZERO, "mech-nonzero-residuals-mean-poor-fit"),
            Distractor(
                "Not appropriate: some residuals are positive and some are negative", "mech-sign-mix-taken-as-pattern"
            ),
        ]
    ordered = ", ".join(dec(r) for r in res)
    part_c = dict(
        prompt="Using the five residuals in order of x, which statement about using a linear model is correct?",
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        marks=2,
        working=[f"Residuals in order of x: {ordered}.", f"They show a {want} pattern, so: {key}."],
        distractors=wrong,
    )
    part_d = number_part(
        "Find the sum of the squares of the five residuals, correct to 2 decimal places.",
        sse,
        2,
        2,
        [f"{' + '.join(f'({dec(r)})²' for r in res)} = {dec(sse)}."],
        [
            (sum(abs(r) for r in res), "mech-absolute-not-squared"),
            (other_sse, "mech-one-residual-left-out"),
            (sse / 5, "mech-mean-instead-of-sum"),
            (res[k] ** 2, "mech-one-residual-squared"),
        ],
        follow=f"part_b**2 + {dec(other_sse)}",
        sensible=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
