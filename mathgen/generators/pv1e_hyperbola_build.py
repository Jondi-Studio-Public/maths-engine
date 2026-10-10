"""Generates `pv1e_hyperbola_asymptotes` figure questions: read the asymptotes of a drawn hyperbola.

Skill ``senior-recognise-features-of-the-graphs-of-y-frac1x``, topic ``pv1e_hyperbola_asymptotes``.
The learner sees a hyperbola on a labelled integer grid and types its two asymptotes as
``x = h, y = k`` (``answer: pair``, labelled, so the order they are written in does not matter).
Neither asymptote is drawn: the two branches are sampled so that they break at the integer x = h
(``1/(x - h)`` is undefined there) and level off along y = k at the window's edges. The prompt
names the family ``y = 1/(x - h) + k`` and never the values of h and k.

"Unambiguous", as a construction constraint: h is at least 3 from both side edges of the
window, so the far ends sit within 1/3 of y = k (round-to-integer is safe), and k is at least 3
from the top and bottom edges so both branches are visible.

Tiers differ in the learner's DECISION:

- easy   -- h and k are non-zero: two asymptotes, away from both axes, read off the grid.
- medium -- an asymptote may be an axis itself (x = 0, or y = 0, or both): the learner names the
            axis as the asymptote instead of looking for a separate line.
- hard   -- one grid square is 2 units up: the learner reads y = k in units from the tick numbers
            (counting squares gives k/2), while x is still 1 per square.
- exam   -- the window is shifted away from the origin so no axis is drawn: both asymptotes are
            read from the numbers along the edges of the grid, not relative to an axis.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from .. import figures
from ..core import check, generator
from . import pv1e_curve_figure_build as cf

TOPIC = "pv1e_hyperbola_asymptotes"
_TIERS = ("easy", "medium", "hard", "exam")

_ALT = "a curve with two branches on a coordinate grid"

_FORM = "in the form x = h, y = k"

_OPEN = [
    "The graph of y = 1/(x - h) + k is drawn on the grid, where h and k are integers.",
    "A curve with equation y = 1/(x - h) + k is shown, with h and k integers.",
    "The graph shows y = 1/(x - h) + k for whole numbers h and k.",
    "The hyperbola y = 1/(x - h) + k is drawn below, where h and k are integers.",
]

_ASK = [
    "State the equations of its vertical and horizontal asymptotes, " + _FORM + ".",
    "Find both asymptotes and give them " + _FORM + ".",
    "Write down the vertical asymptote and the horizontal asymptote, " + _FORM + ".",
]

_NOTES = {
    "easy": [
        "Each grid square is 1 unit on both axes.",
        "One grid square is 1 unit across and 1 unit up.",
        "Both axes have 1 unit per grid square.",
    ],
    "medium": [
        "Each grid square is 1 unit on both axes. Either h or k may be 0.",
        "One grid square is 1 unit across and 1 unit up. Either h or k may be 0.",
        "Both axes have 1 unit per grid square, and h or k may be 0.",
    ],
    "hard": [
        "Each grid square is 1 unit across and 2 units up.",
        "The horizontal scale is 1 unit per grid square and the vertical scale is 2 units per grid square.",
        "Across, one grid square is 1 unit; up, one grid square is 2 units.",
    ],
    "exam": [
        "The axes are not shown, so read the values from the numbers along the edges of the grid. "
        "Each grid square is 1 unit.",
        "No axes are drawn: use the numbers on the edges of the grid, where one square is 1 unit each way.",
        "The window is away from the origin, so use the numbers on its edges; one grid square is "
        "1 unit on both scales.",
    ],
}


def _prompt(rng, tier: str) -> str:
    return " ".join((rng.choice(_OPEN), rng.choice(_NOTES[tier]), rng.choice(_ASK)))


def _tex(pattern: str) -> str:
    """The LaTeX twin: instructions stay prose, only the mathematics is delimited."""
    pattern = pattern.replace("y = 1/(x - h) + k", "\x00A").replace("x = h, y = k", "\x00B")
    pattern = re.sub(r"\b([hk])\b", r"\\(\1\\)", pattern)
    return pattern.replace("\x00A", r"\(y = \frac{1}{x - h} + k\)").replace("\x00B", r"\(x = h,\ y = k\)")


# ---------------------------------------------------------------- drawing the answer first

_NONZERO = (-3, -2, -1, 1, 2, 3)
_OFF_X0 = tuple(range(1, 9)) + tuple(range(-20, -12))  # a 12-wide window that excludes 0


def _draw(rng, tier: str):
    """(h, k, x_range, y_range, y_grid, equal_scale)."""
    if tier == "easy":
        h, k = rng.choice(_NONZERO), rng.choice(_NONZERO)
        while k == h:
            k = rng.choice(_NONZERO)
        return h, k, (-6, 6), (-6, 6), 1, True
    if tier == "medium":
        cls = rng.choices(("y_axis", "x_axis", "both"), weights=(45, 45, 10))[0]
        if cls == "y_axis":
            h, k = 0, rng.choice(_NONZERO)
        elif cls == "x_axis":
            h, k = rng.choice(_NONZERO), 0
        else:
            h, k = 0, 0
        return h, k, (-6, 6), (-6, 6), 1, True
    if tier == "hard":
        h, k = rng.choice(_NONZERO), rng.choice((-4, -2, 2, 4))
        return h, k, (-6, 6), (-8, 8), 2, False
    if tier == "exam":
        x0, y0 = rng.choice(_OFF_X0), rng.choice(_OFF_X0)
        h, k = x0 + rng.randint(3, 9), y0 + rng.randint(3, 9)
        return h, k, (x0, x0 + cf.SPAN), (y0, y0 + cf.SPAN), 1, True
    raise ValueError(f"unknown tier {tier!r}")


def _build_figure(h: int, k: int, x_range, y_range, y_grid, equal_scale) -> dict:
    pcs = cf.pieces(lambda x: 1 / (x - h) + k, x_range, y_range)
    return cf.plane(
        x_range=x_range, y_range=y_range, alt=_ALT, y_grid=y_grid, equal_scale=equal_scale, curves=[{"pieces": pcs}]
    )


# ---------------------------------------------------------------- reading a figure


def read_hyperbola(question) -> str:
    """Re-derive ``x = h, y = k`` from the figure alone.

    The vertical asymptote is where the two branches break (the middle of the gap between the
    left piece's last sample and the right piece's first). The horizontal one is the whole
    number both outer ends are level with.
    """
    pcs = cf.curve_pieces(question.figure)
    check(len(pcs) == 2, "hyperbola reader: expected two branches")
    left, right = pcs
    h = round((left[-1][0] + right[0][0]) / 2)
    y_left, y_right = left[0][1], right[-1][1]
    k = round(y_left)
    check(round(y_right) == k, "hyperbola reader: the two outer ends are not level with one line")
    check(
        abs(y_left - k) < 0.4 and abs(y_right - k) < 0.4,
        "hyperbola reader: an outer end is not close to a whole-number line",
    )
    return f"x = {h}, y = {k}"


figures.register_reader(TOPIC, read_hyperbola)


def _verify(fig, h, k, x_range) -> None:
    left_edge, right_edge = x_range
    check(h - left_edge >= 3 and right_edge - h >= 3, "hyperbola: h is too near a side edge")
    got = read_hyperbola(SimpleNamespace(figure=fig))
    check(got == f"x = {h}, y = {k}", "hyperbola: figure reader disagrees with the key")
    left, right = cf.curve_pieces(fig)
    check(left[-1][0] < h < right[0][0], "hyperbola: the break does not straddle x = h")
    check(all(abs(p[0] - h) > 1e-9 for p in left + right), "hyperbola: a sample sits on x = h")


# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "easy": ("swapped", "sign-h", "sign-k", "branch-point"),
    "medium": ("swapped", "sign-h", "sign-k", "branch-point", "axis-next-to"),
    "hard": ("squares-not-units", "swapped", "sign-h", "sign-k", "branch-point"),
    "exam": ("swapped", "sign-h", "sign-k", "branch-point"),
}


def _candidates(tier: str, h: int, k: int) -> list[tuple[str, int, int]]:
    cands = {
        "swapped": (k, h),
        "sign-h": (-h, k),
        "sign-k": (h, -k),
        "branch-point": (h + 1, k + 1),
        "axis-next-to": (h + 1, k) if h == 0 else (h, k + 1),
        "squares-not-units": (h, k // 2),
    }
    return [(name, *cands[name]) for name in _PRIORITY[tier]]


def _distractors(tier: str, h: int, k: int):
    out, seen = [], {(h, k)}
    for name, a, b in _candidates(tier, h, k):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((name, a, b))
    return out[:3]


def _text(h: int, k: int) -> str:
    return f"x = {h}, y = {k}"


# ---------------------------------------------------------------- building


@generator(TOPIC, level=13, presets={t: {"tier": t} for t in _TIERS})
def pv1e_hyperbola_asymptotes(rng, tier: str = "easy") -> dict:
    """The asymptotes of y = 1/(x - h) + k from its graph. Answer first, then the window."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    h, k, x_range, y_range, y_grid, equal_scale = _draw(rng, tier)
    fig = _build_figure(h, k, x_range, y_range, y_grid, equal_scale)
    _verify(fig, h, k, x_range)

    answer_text = _text(h, k)
    dist = _distractors(tier, h, k)
    check(len(dist) >= 2, "hyperbola: fewer than two distinct distractors")
    texts = [_text(a, b) for _, a, b in dist]
    check(
        len({(h, k), *[(a, b) for _, a, b in dist]}) == len(dist) + 1,
        "hyperbola: a distractor repeats the key or another distractor",
    )
    figures.assert_no_leak(fig, [answer_text, *texts, f"x = {h}", f"y = {k}"])

    pattern = _prompt(rng, tier)
    working = [
        f"The two branches run alongside the vertical line x = {h} without meeting it, so the "
        f"vertical asymptote is x = {h}.",
        f"Far from that line the branches level off along y = {k}, so the horizontal asymptote is y = {k}.",
    ]
    return {
        "text": pattern,
        "latex": _tex(pattern),
        "answer": {"x": h, "y": k},
        "answer_text": answer_text,
        "answer_latex": rf"x = {h},\ y = {k}",
        "distractors": texts,
        "working": working,
        "figure": fig,
    }
