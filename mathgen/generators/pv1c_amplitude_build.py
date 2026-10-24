"""Generates `pv1c_amplitude` figure questions: read the signed amplitude multiplier a from a drawn sine curve.

Skill ``senior-examine-amplitude-changes-and-the-graphs-of-y``, topic ``pv1c_amplitude``.
The learner sees one drawn sine curve (a is never printed) and types a, the signed integer
multiplier, read off the integer y grid.

Answer first. a is drawn from the tier's set, then the curve a sin(pi x) is sampled with
step 1/16 so every half-integer x (the peaks and troughs) and every integer x (the zeros) is
an exact sample, and every peak height is a grid intersection.

Tiers differ in the learner's DECISION:

- easy   -- a > 0, origin in view: the height of the highest point.
- medium -- a may be negative: the curve falls first, so a is the signed first-extreme height.
- hard   -- the window does not contain x = 0 (x from s to s + 3): the sign of a comes from
            WHERE on the labelled x axis the window sits, since sin x is negative on (pi, 2 pi).
- exam   -- one lobe only (x from s to s + 1): the sign of a is decided from which interval of
            the x axis the single lobe occupies, the sign of sin there being (-1)^s.
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from types import SimpleNamespace

from .. import figures
from ..core import check, check_pairwise_distinct, generator, rational_text, latex_from_inline

TOPIC = "pv1c_amplitude"

YLO, YHI = -7, 7
STEP_DEN = 16  # every sample x is a multiple of 1/16

_POS = (1, 2, 3, 4, 5, 6)
_SIGNED = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
_S_HARD = (1, 2, 3, 4, 5, -6, -5, -4)  # -3 would end the window at x = 0 (the y-axis would be drawn)
_S_EXAM = (-6, -5, -4, -3, -2, 1, 2, 3, 4, 5)

_ALT = "the graph of a sine curve on a coordinate grid"

# ---------------------------------------------------------------- prompts
# {Y} the family, {A} the parameter, {PI} pi: filled differently for text and latex.

_SCALE = [
    "The x-axis is marked in multiples of {PI} and each square up the y-axis is 1 unit.",
    "Along the x-axis the marks are multiples of {PI}; up the y-axis, one square is 1 unit.",
    "Read the x-axis in multiples of {PI} (radians) and the y-axis at 1 unit per grid square.",
]

_OPEN = {
    "peak_up": [
        "The graph of {Y} is drawn below, where {A} is a positive constant.",
        "Below is the graph of {Y} for one particular positive value of {A}.",
        "A curve with equation {Y} is shown, and {A} is positive.",
        "Here is the graph of {Y}, where {A} is a positive whole number.",
    ],
    "signed_near_origin": [
        "The graph of {Y} is drawn below, where {A} can be positive or negative.",
        "Below is the graph of {Y} for one particular whole-number value of {A}, which may be negative.",
        "A curve with equation {Y} is shown, where {A} is a nonzero whole number that may be negative.",
        "Here is the graph of {Y}, with the sign of {A} not given.",
    ],
    "offset_window": [
        "Part of the graph of {Y} is drawn below, where {A} can be positive or negative.",
        "The graph below shows a section of the curve {Y}, where {A} is a whole number that may be negative.",
        "A section of the curve {Y} is drawn, and {A} is either positive or negative.",
        "Here is part of the graph of {Y}, where the sign of {A} is not given.",
    ],
    "single_lobe": [
        "Only one arch of the curve {Y} is drawn below, where {A} can be positive or negative.",
        "The graph below shows a single arch of {Y}, where {A} is a whole number that may be negative.",
        "A single arch of the curve {Y} is drawn, and {A} is either positive or negative.",
        "Here is one arch of the graph of {Y}, where the sign of {A} is not given.",
    ],
}

_ASK = {
    "peak_up": ["State the value of {A}.", "What is {A}?", "Use the graph to find {A}."],
}
_ASK["signed_near_origin"] = [
    "State the value of {A}, with its sign.",
    "What is {A}?",
    "Use the graph to find {A}, including its sign.",
]
_ASK["offset_window"] = [
    "State the value of {A}, with its sign.",
    "What is {A}?",
    "Use the graph to find the value of {A}.",
]
_ASK["single_lobe"] = _ASK["offset_window"]

_FILL_TEXT = {"Y": "y = a sin x", "A": "a", "PI": "π"}
_FILL_TEX = {"Y": r"\(y = a \sin x\)", "A": r"\(a\)", "PI": r"\(\pi\)"}


def _prompt(rng, variant: str) -> tuple[str, str]:
    o = rng.choice(_OPEN[variant])
    s = rng.choice(_SCALE)
    k = rng.choice(_ASK[variant])
    pattern = f"{o} {s} {k}"
    return pattern.format(**_FILL_TEXT), pattern.format(**_FILL_TEX)


def _pi_multiple(k: int) -> str:
    """k·π as written on paper: 0, π, -π, 2π."""
    return "0" if k == 0 else "π" if k == 1 else "-π" if k == -1 else f"{k}π"


# ---------------------------------------------------------------- curve and figure


def _sample_fn(a: int):
    return lambda x: a * math.sin(math.pi * x)


def _window(variant: str, rng) -> tuple[int, int, F]:
    """(x_lo, x_hi, x_grid) in units of pi."""
    if variant in ("peak_up", "signed_near_origin"):
        return -2, 2, F(1, 2)
    if variant == "offset_window":
        s = rng.choice(_S_HARD)
        return s, s + 3, F(1, 2)
    s = rng.choice(_S_EXAM)
    return s, s + 1, F(1, 4)


def _all_samples(fig):
    return [tuple(p) for c in fig["curves"] for piece in c["pieces"] for p in piece]


# ---------------------------------------------------------------- reading a figure


def read_amplitude(question) -> str:
    """Re-derive a from the figure alone. The prompt names the family y = a sin x with x in
    units of pi (the figure's own tick format), so a = y / sin(pi x) at every sample where
    |sin(pi x)| > 1/2; they must agree and be an integer, and the largest |y| sample must be
    |a|."""
    fig = question.figure
    check(fig.get("x_tick_format") == "pi", "amplitude reader: x axis is not in units of pi")
    pts = _all_samples(fig)
    ratios = []
    for x, y in pts:
        sx = math.sin(math.pi * x)
        if abs(sx) > 0.5:
            ratios.append(y / sx)
    check(len(ratios) >= 4, "amplitude reader: too few samples away from the zeros")
    a = round(ratios[0])
    check(all(abs(r - a) < 1e-4 for r in ratios), "amplitude reader: samples disagree on a")
    check(a != 0, "amplitude reader: a flat curve")
    peak = max(abs(y) for _, y in pts)
    check(abs(peak - abs(a)) < 1e-4, "amplitude reader: peak height is not |a|")
    return rational_text(a)


figures.register_reader(TOPIC, read_amplitude)

# ---------------------------------------------------------------- verification


def _verify(fig, a: int, x_lo: int, x_hi: int) -> None:
    """Independent checks on a built question (raises VerificationError)."""
    # 1. exact route: every half-integer sample is a signed peak, (-1)^floor(x) * a.
    halves = 0
    for x, y in _all_samples(fig):
        if abs(2 * x - round(2 * x)) < 1e-9 and round(2 * x) % 2 == 1:
            halves += 1
            check(abs(y - a * (-1) ** math.floor(x)) < 1e-9, "amplitude: a half-integer sample is not the signed peak")
    check(halves >= 1, "amplitude: no peak sample in the window")
    # 2. the registered reader agrees with the key on the figure as built.
    check(
        read_amplitude(SimpleNamespace(figure=fig)) == rational_text(a),
        "amplitude: figure reader disagrees with the key",
    )
    # 3. whole peak shown, one unclipped piece, no x = 0 where the tier forbids it.
    check(fig["y_range"] == [YLO, YHI] and abs(a) < YHI, "amplitude: peak not inside the y window")
    check(len(fig["curves"]) == 1 and len(fig["curves"][0]["pieces"]) == 1, "amplitude: the curve is clipped")
    check(len(_all_samples(fig)) == (x_hi - x_lo) * STEP_DEN + 1, "amplitude: sample count")


# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "peak_up": ("peak-to-trough", "off-by-one-up", "off-by-one-down", "sign-flipped"),
    "signed_near_origin": ("sign-flipped", "peak-to-trough", "off-by-one-up", "off-by-one-down"),
    "offset_window": ("sign-flipped", "peak-to-trough", "off-by-one-up", "off-by-one-down"),
    "single_lobe": ("sign-flipped", "peak-to-trough", "off-by-one-up", "off-by-one-down"),
}


def _distractors(variant: str, a: int) -> list[tuple[str, int]]:
    sg = 1 if a > 0 else -1
    cands = {"sign-flipped": -a, "peak-to-trough": 2 * a, "off-by-one-up": a + sg, "off-by-one-down": a - sg}
    out, seen = [], {a}
    for name in _PRIORITY[variant]:
        v = cands[name]
        if v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out[:3]


# ---------------------------------------------------------------- building


def _working(variant: str, a: int, x_lo: int, x_hi: int) -> list[str]:
    if variant == "peak_up":
        return [
            f"y = a sin x is largest when sin x = 1, where y = a. The highest point of the "
            f"curve is at height {a} on the y-axis.",
            f"So a = {a}.",
        ]
    if variant == "signed_near_origin":
        way = "rises" if a > 0 else "falls"
        return [
            f"The curve {way} first from the origin, and its first extreme is at height {a}.",
            f"Near the origin sin x is positive, so the sign of a is the direction of the first extreme: a = {a}.",
        ]
    sgn = "positive" if (-1) ** x_lo > 0 else "negative"
    lobe = "above" if a * (-1) ** x_lo > 0 else "below"
    return [
        f"The first arch runs from x = {_pi_multiple(x_lo)} to x = {_pi_multiple(x_lo + 1)}, where sin x is {sgn}.",
        f"The curve is {lobe} the x-axis there, so a is {'positive' if a > 0 else 'negative'}, "
        f"and its extreme height is {abs(a)}: a = {a}.",
    ]


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "peak_up"},
        "medium": {"variant": "signed_near_origin"},
        "hard": {"variant": "offset_window"},
        "exam": {"variant": "single_lobe"},
    },
)
def pv1c_amplitude(rng, variant: str = "peak_up") -> dict:
    """a read off the graph of y = a sin x. Answer first: draw a, then sample the curve."""
    if variant not in _OPEN:
        raise ValueError(f"unknown variant {variant!r}")
    a = rng.choice(_POS if variant == "peak_up" else _SIGNED)
    x_lo, x_hi, x_grid = _window(variant, rng)
    n = (x_hi - x_lo) * STEP_DEN + 1
    fig = figures.cartesian(
        x_range=(x_lo, x_hi),
        y_range=(YLO, YHI),
        alt=_ALT,
        x_grid=x_grid,
        y_grid=1,
        equal_scale=False,
        x_tick_format="pi",
        curves=[{"pieces": figures.sample_curve(_sample_fn(a), x_lo, x_hi, (YLO, YHI), n=n)}],
    )
    _verify(fig, a, x_lo, x_hi)

    dist = _distractors(variant, a)
    answer_text = rational_text(a)
    texts = [rational_text(v) for _, v in dist]
    check_pairwise_distinct(a, *[v for _, v in dist])
    figures.assert_no_leak(fig, [answer_text, *texts])
    text, latex = _prompt(rng, variant)
    return {
        "text": text,
        "latex": latex_from_inline(latex),
        "answer": a,
        "answer_text": answer_text,
        "answer_latex": str(a),
        "distractors": texts,
        "working": _working(variant, a, x_lo, x_hi),
        "figure": fig,
    }
