"""Generates `pv1e_exp_asymptote` figure questions: read the asymptote of a drawn exponential curve.

Skill ``senior-recognise-the-qualitative-features-of-the-graph-of``, topic ``pv1e_exp_asymptote``.
The learner sees an exponential curve on a labelled integer grid and types the equation of its
horizontal asymptote, ``y = k`` (``answer: pair``, one labelled component). The asymptote is not
drawn: the curve is sampled at step 1/16 and one end of it runs level along y = k (within 2^-6 of
it) while the other end climbs out of the window.

Tiers differ in the learner's DECISION:

- easy   -- a > 1, so the curve levels off on the LEFT; k is any non-zero whole number, so the
            curve often crosses the x-axis and the x-axis is a tempting wrong answer.
- medium -- 0 < a < 1 is possible, so the level end may be on the RIGHT, and k may be 0 (the
            asymptote is the x-axis itself): the learner first decides which end is the level one.
- hard   -- one grid square is 2 units up: the learner reads y = k in units from the tick numbers
            (counting squares gives k/2).
- exam   -- the x-axis is not in the window, so no axis is drawn: the learner reads the level
            end against the numbers beside the y-axis (the y-axis is drawn, the x-axis is not).
"""

from __future__ import annotations

from types import SimpleNamespace

from .. import figures
from ..core import check, generator
from . import pv1e_curve_figure_build as cf

TOPIC = "pv1e_exp_asymptote"
_TIERS = ("easy", "medium", "hard", "exam")
_ALT = "a curve rising steeply at one end on a coordinate grid"

_UP = (2.0, 3.0)
_DOWN = (0.5, 1 / 3)
_NONZERO = tuple(k for k in range(-5, 6) if k != 0)
_ALL = tuple(range(-5, 6))
_OFF_Y0 = tuple(range(2, 9)) + tuple(range(-20, -13))  # a 12-tall window that excludes 0

#: (plain text, latex) opening sentences; a and k are named, never their values
_OPEN = [
    (
        "The graph of y = a^x + k is drawn on the grid, where a > 0 and k is an integer.",
        r"The graph of \(y = a^{x} + k\) is drawn on the grid, where \(a > 0\) and \(k\) is an integer.",
    ),
    (
        "A curve with equation y = a^x + k is shown, with a > 0 (a is not 1) and k an integer.",
        r"A curve with equation \(y = a^{x} + k\) is shown, with \(a > 0\) (\(a\) is not \(1\)) and \(k\) an integer.",
    ),
    (
        "The graph shows y = a^x + k for a positive number a and an integer k.",
        r"The graph shows \(y = a^{x} + k\) for a positive number \(a\) and an integer \(k\).",
    ),
    (
        "An exponential curve y = a^x + k, with a > 0 and k an integer, is drawn below.",
        r"An exponential curve \(y = a^{x} + k\), with \(a > 0\) and \(k\) an integer, is drawn below.",
    ),
]
_NOTES = {
    "easy": [
        "Each grid square is 1 unit on both axes.",
        "One grid square is 1 unit across and 1 unit up.",
        "Both axes have 1 unit per grid square.",
    ],
    "hard": [
        "Each grid square is 1 unit across and 2 units up.",
        "The horizontal scale is 1 unit per grid square and the vertical scale is 2 units per grid square.",
        "Across, one grid square is 1 unit; up, one grid square is 2 units.",
    ],
    "exam": [
        "The x-axis is not shown, so read the value from the numbers beside the y-axis. Each grid square is 1 unit.",
        "No x-axis is drawn: use the numbers marked along the y-axis, where one square is 1 unit each way.",
        "The window is above (or below) the x-axis, so use the numbers along the y-axis; one grid "
        "square is 1 unit on both scales.",
    ],
}
_NOTES["medium"] = _NOTES["easy"]
_ASK = [
    "State the equation of its horizontal asymptote.",
    "Find the equation of the horizontal asymptote.",
    "Give the equation of its horizontal asymptote.",
]


def _tex_note(note: str) -> str:
    return note.replace("x-axis", r"\(x\)-axis").replace("y-axis", r"\(y\)-axis")


# ---------------------------------------------------------------- drawing the answer first


def _draw(rng, tier: str):
    """-> (a, k, x_range, y_range, y_grid, equal_scale)."""
    x_range = (-6, 6)
    if tier == "easy":
        return rng.choice(_UP), rng.choice(_NONZERO), x_range, (-6, 6), 1, True
    if tier == "medium":
        return rng.choice(_UP + _DOWN), rng.choice(_ALL), x_range, (-6, 6), 1, True
    if tier == "hard":
        return (
            rng.choice(_UP + _DOWN),
            rng.choice((-10, -8, -6, -4, -2, 2, 4, 6, 8, 10)),
            x_range,
            (-12, 12),
            2,
            False,
        )
    if tier == "exam":
        y0 = rng.choice(_OFF_Y0)
        return rng.choice(_UP + _DOWN), y0 + rng.randint(1, 8), x_range, (y0, y0 + cf.SPAN), 1, True
    raise ValueError(f"unknown tier {tier!r}")


def _visible_share(pcs, n_total=cf.N_SAMPLES) -> float:
    return sum(len(p) for p in pcs) / n_total


def _shows_a_climb(pcs, y_range) -> bool:
    """Enough of the curve is in the window and it rises or falls by a fifth of the window's height."""
    ys = [p[1] for pc in pcs for p in pc]
    return _visible_share(pcs) >= 0.3 and max(ys) - min(ys) >= 0.2 * (y_range[1] - y_range[0])


# ---------------------------------------------------------------- reading a figure


def read_exp_asymptote(question) -> str:
    """Re-derive ``y = k`` from the figure alone: the whole number the level end runs along."""
    pcs = cf.curve_pieces(question.figure)
    check(len(pcs) == 1, "exp reader: expected one unbroken curve")
    pts = pcs[0]
    step_left = abs(pts[1][1] - pts[0][1])
    step_right = abs(pts[-1][1] - pts[-2][1])
    check(
        min(step_left, step_right) * 8 < max(step_left, step_right) or max(step_left, step_right) < 0.05,
        "exp reader: neither end is clearly the level one",
    )
    level = pts[0][1] if step_left < step_right else pts[-1][1]
    k = round(level)
    check(abs(level - k) < 0.03, "exp reader: the level end is not on a whole-number line")
    return f"y = {k}"


figures.register_reader(TOPIC, read_exp_asymptote)


def _verify(fig, a, k) -> None:
    got = read_exp_asymptote(SimpleNamespace(figure=fig))
    check(got == f"y = {k}", "exp: figure reader disagrees with the key")
    check(_shows_a_climb(cf.curve_pieces(fig), fig["y_range"]), "exp: the curve is not plausibly shown")


# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "easy": ("y-intercept", "x-axis", "sign-flipped", "off-by-one", "off-by-two"),
    "medium": ("y-intercept", "x-axis", "sign-flipped", "off-by-one", "off-by-two"),
    "hard": ("squares-not-units", "y-intercept", "x-axis", "sign-flipped", "off-by-one"),
    "exam": ("y-intercept", "off-by-one", "sign-flipped", "off-by-two", "x-axis"),
}


def _distractors(tier: str, k: int):
    cands = {
        "y-intercept": k + 1,
        "x-axis": 0,
        "sign-flipped": -k,
        "off-by-one": k - 1,
        "off-by-two": k + 2,
        "squares-not-units": k // 2,
    }
    out, seen = [], {k}
    for name in _PRIORITY[tier]:
        v = cands[name]
        if v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out[:3]


# ---------------------------------------------------------------- building


@generator(TOPIC, level=13, presets={t: {"tier": t} for t in _TIERS})
def pv1e_exp_asymptote(rng, tier: str = "easy") -> dict:
    """The horizontal asymptote of y = a^x + k from its graph. Answer first, then the window."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(40):
        a, k, x_range, y_range, y_grid, equal_scale = _draw(rng, tier)
        pcs = cf.pieces(lambda x: a**x + k, x_range, y_range)
        if pcs and _shows_a_climb(pcs, y_range):
            break
    else:
        raise RuntimeError(f"{TOPIC}/{tier}: could not draw a visible curve")
    fig = cf.plane(
        x_range=x_range, y_range=y_range, alt=_ALT, y_grid=y_grid, equal_scale=equal_scale, curves=[{"pieces": pcs}]
    )
    _verify(fig, a, k)

    answer_text = f"y = {k}"
    dist = _distractors(tier, k)
    check(len(dist) >= 2, "exp: fewer than two distinct distractors")
    texts = [f"y = {v}" for _, v in dist]
    check(len({k, *[v for _, v in dist]}) == len(dist) + 1, "exp: a distractor repeats the key")
    figures.assert_no_leak(fig, [answer_text, *texts])

    open_text, open_latex = rng.choice(_OPEN)
    note = rng.choice(_NOTES[tier])
    ask = rng.choice(_ASK)
    text = " ".join((open_text, note, ask))
    latex = " ".join((open_latex, _tex_note(note), ask))
    side = "left" if a > 1 else "right"
    working = [
        f"The curve levels off along a horizontal line at the {side} without ever reaching it: that line is y = {k}.",
        f"So the horizontal asymptote is y = {k}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": {"y": k},
        "answer_text": answer_text,
        "answer_latex": f"y = {k}",
        "distractors": texts,
        "working": working,
        "figure": fig,
    }
