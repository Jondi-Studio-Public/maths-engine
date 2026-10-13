"""Generates `pv1c_period` figure questions: find the period change from a drawn sine curve.

Skill ``senior-examine-period-changes-and-the-graphs-of-y``, topic ``pv1c_period``. The learner
sees a sine curve drawn on an x-axis ticked and labelled in multiples of pi (b is never
printed, no point is marked) and types the period as an expression in pi.

Answer first. The period k*pi is drawn from the tier's pool (k a Fraction), the curve
sign * sin(2*pi*u / k) is sampled so every zero, extremum and gridline is an exact sample, and
the figure is built around it. The x values of the figure are in units of pi (``x_tick_format``
"pi"). The graded string is the ``*`` form ("2*pi/3"); the learner-facing text is "2pi/3".

Tiers differ in the learner's DECISION:

- easy   -- ``peaks``: the period is the distance between successive peaks (one full cycle).
- medium -- ``zeros``: only the zero crossings sit on the labelled gridlines, and consecutive
            zeros are HALF a period apart, so the learner doubles the spacing. The prompt states
            the reading of "period" (upward crossing to the next upward crossing) without the answer.
- hard   -- ``count_cycles``: the window is fixed at 0 to 2pi; the learner counts complete cycles
            and divides (b a nonzero whole number, negative on half the draws: the curve falls
            first, the period is still positive).
- exam   -- ``partial_cycle``: the window shows about one cycle or less, so the period is read
            from the first quarter (4 times the x of the first extreme), never by counting cycles;
            half the draws fall first, so the first extreme is a trough.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction as F
from types import SimpleNamespace

import sympy

from .. import figures
from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    format_pi_multiple,
    generator,
    latex_from_inline,
    latex_pi_multiple,
    rational_text,
)

TOPIC = "pv1c_period"

_ALT = "the graph of a sine curve on a coordinate grid"

# k = period / pi.  x_grid denominators must be among those the renderer reduces (1,2,3,4,6,8,12).
_K_EASY = (F(1, 3), F(1, 2), F(2, 3), F(1), F(4, 3), F(2), F(3), F(4))
_N_HARD = (1, 2, 3, 4, 6, 8)  # b = n, k = 2/n
_K_EXAM = (F(4), F(6), F(8), F(10), F(12), F(14))

#: probability that the drawn curve falls first (b < 0), hard and exam only
_NEG_SHARE = 0.5

#: distractor priority per variant (names are the skill's misconception ids)
_PRIORITY = {
    "peaks": ("half-period", "double-period", "frequency-b", "quarter-period"),
    "zeros": ("half-period", "double-period", "quarter-period", "frequency-b"),
    "count_cycles": ("frequency-b", "half-period", "double-period", "quarter-period"),
    "partial_cycle": ("quarter-period", "half-period", "double-period", "frequency-b"),
}

# ---------------------------------------------------------------- prompts
# {EQ}: y = sin(bx); {B}: b; {PI}: pi; {EX}: the notation example. Text and latex twins share a
# bank and differ only in how these four slots are written.

_OPEN = {
    "peaks": [
        "The graph of {EQ} is drawn below, where {B} is a positive constant.",
        "Here is the graph of the curve {EQ} for some positive constant {B}.",
        "A curve with equation {EQ}, where {B} is a positive constant, is shown.",
        "The diagram shows the graph of {EQ} for a positive value of {B}.",
    ],
    "zeros": [
        "The graph of {EQ} is drawn below, where {B} is a positive constant.",
        "Below is a sketch of {EQ} for some positive constant {B}.",
        "A sine curve {EQ}, with {B} a positive constant, has been plotted.",
        "The curve {EQ} is plotted below for a positive value of {B}.",
    ],
    "count_cycles": [
        "The graph of {EQ} is drawn below, where {B} is a nonzero whole number.",
        "Here is the graph of {EQ} for some nonzero whole number {B}.",
        "A curve with equation {EQ}, where {B} is a nonzero whole number, is shown.",
        "The diagram shows {EQ} drawn for a nonzero whole-number value of {B}.",
    ],
    "partial_cycle": [
        "The graph of {EQ} is drawn below, where {B} is a nonzero constant.",
        "Here is part of the graph of {EQ} for some nonzero constant {B}.",
        "A curve with equation {EQ}, where {B} is a nonzero constant, is shown.",
        "The diagram shows the curve {EQ} for a nonzero value of {B}.",
    ],
}

_MIDDLE = {
    "peaks": [
        "The x-axis is marked in multiples of {PI}, and the curve reaches its highest value more than once.",
        "Use the peaks of the curve and the x-axis, which is marked in multiples of {PI}.",
        "Each gridline on the x-axis is labelled as a multiple of {PI}; look at where the curve peaks.",
    ],
    "zeros": [
        "The x-axis is marked in multiples of {PI}. One full cycle runs from a point where the curve crosses the x-axis going upward to the next point where it crosses the x-axis going upward.",
        "The gridlines are labelled in multiples of {PI}. Take a full cycle to run from one point where the curve crosses the x-axis going upward to the next point where it crosses the x-axis going upward.",
        "The x-axis is labelled in multiples of {PI}. A full cycle starts where the curve crosses the x-axis going upward and ends at the next place it crosses the x-axis going upward.",
    ],
    "count_cycles": [
        "The graph covers the x-values from 0 to 2{PI}, and the x-axis is labelled in multiples of {PI}.",
        "The drawn part of the curve runs from x = 0 to x = 2{PI}; the x-axis is labelled in multiples of {PI}.",
        "Only the interval from 0 to 2{PI} is drawn. The x-axis is labelled in multiples of {PI}.",
    ],
    "partial_cycle": [
        "The graph is drawn only for x from 0 to 4{PI}, and the x-axis is labelled in multiples of {PI}. It may not show a complete cycle.",
        "Only the interval from 0 to 4{PI} is drawn, so the curve might not complete a cycle. The gridlines are labelled in multiples of {PI}.",
        "The x-axis is labelled in multiples of {PI}, and the curve is shown for x from 0 to 4{PI} only. Not every cycle of the curve is necessarily visible.",
    ],
}

_CLOSE = {
    "peaks": [
        "Give the period of the curve in terms of {PI} (for example {EX}).",
        "What is the period of this curve? Write it in terms of {PI}, for example {EX}.",
        "State the period as a multiple of {PI}, such as {EX}.",
    ],
    "zeros": [
        "State the period of the curve as a multiple of {PI} (for example {EX}).",
        "Find the period of the curve and give it in terms of {PI}, for example {EX}.",
        "What is the period? Write your answer in terms of {PI} (for example {EX}).",
    ],
    "count_cycles": [
        "Find the period of the curve in terms of {PI} (for example {EX}). The period is a positive length.",
        "What is the period of the curve? Give it as a positive multiple of {PI}, such as {EX}.",
        "Give the period in terms of {PI} (for example {EX}); a period is always positive.",
    ],
    "partial_cycle": [
        "Find the period of the whole curve in terms of {PI} (for example {EX}). The period is positive.",
        "What is the period of the curve? Give a positive answer in terms of {PI}, for example {EX}.",
        "State the period as a positive multiple of {PI}, such as {EX}.",
    ],
}

_TEXT_SLOTS = {"EQ": "y = sin(bx)", "B": "b", "PI": "pi", "EX": "5pi/6"}
_LATEX_SLOTS = {"EQ": r"\(y = \sin(bx)\)", "B": r"\(b\)", "PI": r"\(\pi\)", "EX": r"\(\frac{5\pi}{6}\)"}
# "2{PI}" / "4{PI}" are written with a digit glued to pi in text ("2pi") and as one maths span in latex.
_LATEX_GLUED = {"2": r"\(2\pi\)", "4": r"\(4\pi\)"}


def _fill(pattern: str, latex: bool) -> str:
    if latex:
        pattern = re.sub(r"([24])\{PI\}", lambda m: _LATEX_GLUED[m.group(1)], pattern)
        return pattern.format(**_LATEX_SLOTS)
    return pattern.format(**_TEXT_SLOTS)


# ---------------------------------------------------------------- the curve


class Spec:
    """One drawn question: period k (units of pi), direction sign, window, gridline step."""

    def __init__(self, variant, k: F, sign: int, lo: F, hi: F, x_grid: F, step: F, b: F):
        self.variant, self.k, self.sign = variant, k, sign
        self.lo, self.hi, self.x_grid, self.step, self.b = lo, hi, x_grid, step, b

    @property
    def n_points(self) -> int:
        return int((self.hi - self.lo) / self.step) + 1


def _draw_spec(rng, variant: str) -> Spec:
    sign = -1 if (variant in ("count_cycles", "partial_cycle") and rng.random() < _NEG_SHARE) else 1
    if variant in ("peaks", "zeros"):
        k = rng.choice(_K_EASY)
        grid = k / 4 if variant == "peaks" else k / 2
        return Spec(variant, k, sign, F(0), 2 * k, grid, k / 64, F(2) / k)
    if variant == "count_cycles":
        n = rng.choice(_N_HARD)
        k = F(2, n)
        grid = F(1, 2 * n) if n <= 6 else F(1, n)  # n = 8: zeros-only gridlines (1/16 is not a tick)
        return Spec(variant, k, sign, F(0), F(2), grid, k / 32 if n <= 6 else k / 16, F(n) * sign)
    if variant == "partial_cycle":
        k = rng.choice(_K_EXAM)
        return Spec(variant, k, sign, F(0), F(4), F(1, 2), F(1, 32), F(2) / k * sign)
    raise ValueError(f"unknown variant {variant!r}")


def _sine_curve(spec: Spec):
    """The sampled curve; u is in units of pi. Sampled so the quarter-period is an exact sample."""
    two_pi_over_k = 2 * math.pi / float(spec.k)
    s = spec.sign
    n = spec.n_points
    check(2 <= n <= 200, f"{TOPIC}: sample count {n} out of range")
    return figures.sample_curve(lambda u: s * math.sin(two_pi_over_k * u), float(spec.lo), float(spec.hi), (-2, 2), n=n)


# ---------------------------------------------------------------- reading a figure


def _samples(fig):
    pts = [tuple(p) for c in fig["curves"] for piece in c["pieces"] for p in piece]
    return sorted(pts)


def read_period(question) -> str:
    """Re-derive the period from the figure alone: the quarter period is the distance from the
    first zero crossing to the first extremum, so the period is four times that. The zero
    crossings and the extrema found elsewhere on the curve must agree."""
    fig = question.figure
    pts = _samples(fig)
    zeros = [x for x, y in pts if abs(y) < 1e-9]
    check(bool(zeros), f"{TOPIC} reader: no zero crossing on the curve")
    z0 = zeros[0]
    ext = [x for x, y in pts if abs(y) > 1 - 1e-9 and x > z0]
    check(bool(ext), f"{TOPIC} reader: no extremum on the curve")
    quarter = ext[0] - z0
    k = F(4 * quarter).limit_denominator(12)
    check(abs(float(k) - 4 * quarter) < 1e-4, f"{TOPIC} reader: quarter period is not a clean fraction")
    # every consecutive pair of zeros is half a period apart
    for a, b in zip(zeros, zeros[1:]):
        check(abs((b - a) - float(k) / 2) < 1e-4, f"{TOPIC} reader: zeros are not half a period apart")
    # every extremum is an odd multiple of a quarter period from the first zero
    for x in ext:
        q = (x - z0) / quarter
        check(
            abs(q - round(q)) < 1e-3 and round(q) % 2 == 1,
            f"{TOPIC} reader: an extremum is not an odd number of quarter periods along",
        )
    return format_pi_multiple(k)


figures.register_reader(TOPIC, read_period)


# ---------------------------------------------------------------- building


def _distractors(spec: Spec):
    k = spec.k
    cands = {
        "half-period": format_pi_multiple(k / 2),
        "double-period": format_pi_multiple(2 * k),
        "quarter-period": format_pi_multiple(k / 4),
        "frequency-b": rational_text(spec.b),
    }
    return [(name, cands[name]) for name in _PRIORITY[spec.variant]][:3]


def _star(display: str) -> str:
    """The grader-facing spelling: a '*' between a digit and 'pi' ("2pi/3" -> "2*pi/3")."""
    return re.sub(r"(\d)pi", r"\1*pi", display)


def _verify(spec: Spec, fig, answer_text: str, answer: str, dist) -> None:
    # 1. the registered reader, on the figure as built, reaches the key.
    check(read_period(SimpleNamespace(figure=fig)) == answer_text, f"{TOPIC}: figure reader disagrees with the key")
    # 2. the graded string is the stated multiple of pi, by sympy, not by the formatter.
    check_equivalent(
        sympy.sympify(answer),
        sympy.Rational(spec.k.numerator, spec.k.denominator) * sympy.pi,
        f"{TOPIC}: graded string is not k*pi",
    )
    # 3. zero-crossing route: the curve crosses zero every half period and the window is long enough.
    zeros = [x for x, y in _samples(fig) if abs(y) < 1e-9]
    check(
        bool(zeros) and (len(zeros) < 2 or abs((zeros[1] - zeros[0]) - float(spec.k) / 2) < 1e-4),
        f"{TOPIC}: zero spacing is not half the keyed period",
    )
    # 4. every option is a different value from every other (the key included).
    vals = [sympy.sympify(_star(answer_text))] + [sympy.sympify(_star(t)) for _, t in dist]
    check_pairwise_distinct(*vals)


def _working(spec: Spec, answer_text: str) -> list[str]:
    k = spec.k
    if spec.variant == "peaks":
        return [
            f"One full cycle runs from one peak to the next, which is {answer_text} along the x-axis.",
            f"The period is {answer_text}.",
        ]
    if spec.variant == "zeros":
        return [
            f"Consecutive points where the curve meets the x-axis are half a cycle apart: {format_pi_multiple(k / 2)}.",
            f"The period is twice that: {answer_text}.",
        ]
    if spec.variant == "count_cycles":
        n = int(2 / k)
        return [
            f"The graph runs from 0 to 2pi and shows {n} complete cycle" + ("" if n == 1 else "s") + ".",
            f"The period is 2pi / {n} = {answer_text}.",
        ]
    return [
        f"The first extreme is {format_pi_multiple(k / 4)} along the x-axis, a quarter of a cycle.",
        f"The period is 4 x {format_pi_multiple(k / 4)} = {answer_text}.",
    ]


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "peaks"},
        "medium": {"variant": "zeros"},
        "hard": {"variant": "count_cycles"},
        "exam": {"variant": "partial_cycle"},
    },
)
def pv1c_period(rng, variant: str = "peaks") -> dict:
    """The period of y = sin(bx) from a drawn curve. Answer first: draw k, then the figure."""
    spec = _draw_spec(rng, variant)
    fig = figures.cartesian(
        x_range=(float(spec.lo), float(spec.hi)),
        y_range=(-2, 2),
        alt=_ALT,
        x_grid=float(spec.x_grid),
        y_grid=1,
        equal_scale=False,
        x_tick_format="pi",
        curves=[{"pieces": _sine_curve(spec)}],
    )
    answer_text = format_pi_multiple(spec.k)
    answer = _star(answer_text)
    dist = _distractors(spec)
    texts = [t for _, t in dist]
    _verify(spec, fig, answer_text, answer, dist)
    figures.assert_no_leak(fig, [answer_text, answer, *texts])

    o = rng.randrange(len(_OPEN[variant]))
    m = rng.randrange(len(_MIDDLE[variant]))
    c = rng.randrange(len(_CLOSE[variant]))
    pattern = f"{_OPEN[variant][o]} {_MIDDLE[variant][m]} {_CLOSE[variant][c]}"
    return {
        "text": _fill(pattern, latex=False),
        "latex": latex_from_inline(_fill(pattern, latex=True)),
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_pi_multiple(spec.k),
        "distractors": texts,
        "working": _working(spec, answer_text),
        "figure": fig,
    }
