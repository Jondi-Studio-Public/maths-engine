"""Generates `senior-recognise-features-of-the-graph-of-y-mx` questions (ACMMM003): decide the sign of the
gradient from a drawn line, `answer: mcq` with three options.

One skill: ``senior-recognise-features-of-the-graph-of-y-mx`` (ACMMM003),
``domain: functions-sequences``, ``answer: mcq`` with exactly three options,
``positive`` / ``negative`` / ``zero``. The figure is the whole stimulus: no
equation, coordinate or rate is printed anywhere. The CLASS (positive,
negative, zero) is drawn uniformly first, then a line of that class is drawn,
so each label is the key about a third of the time. The zero class is a
horizontal line through two grid intersections.

Tiers escalate by what the learner DECIDES:

- ``easy``: a centred window with the y-axis drawn and a clearly rising,
  falling or flat line (gradients of 1 or 2 in magnitude). Decision: read the
  direction of the line, left to right.
- ``medium``: the y-axis is drawn and the line meets it on the side that
  CONTRADICTS the gradient (a rising line meets it below the origin, a falling
  line above, a flat line off the axis), so judging by the intercept fails.
  Decision: ignore the intercept, use the direction.
- ``hard``: the window is away from x = 0, the y-axis is not drawn and the
  intercept is not visible. Decision: direction alone.
- ``exam``: the axes use different scales and the steepness of the picture is
  made independent of the sign (a rising line may look nearly flat and a
  falling one steep). Decision: the sign is still the direction, whatever the
  slope looks like.

The plausibility reader re-derives the label from the figure's two through
points alone and returns its position among the options shown.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import figures
from mathgen.core import check, generator, latex_answer, phrase
from mathgen.generators.pv1a_line_figure_build import (
    Window,
    line_figure,
    lattice_points,
    pick_two_points,
    read_line,
)

_F = Fraction
_ALT = "a straight line on a coordinate grid"
_CLASSES = ("positive", "negative", "zero")
_ATTEMPTS = 2000

_OPEN = [
    "The graph shows a straight line on a coordinate grid.",
    "A straight line is drawn on the coordinate grid below.",
    "Look at the straight line graphed on this grid.",
    "The diagram shows the graph of a straight line.",
    "Study the straight line drawn on this coordinate grid.",
    "The grid below has the graph of a straight line drawn on it.",
]
_ASK = [
    "Is the gradient m of the line positive, negative or zero?",
    "Which of the following describes the gradient m of this line?",
    "What is the sign of the gradient m of the line?",
    "Decide whether the gradient m of the line is positive, negative or zero.",
    "Choose the statement that is true about the gradient m of the line.",
    "State whether the gradient of the line is positive, negative or zero.",
]
_TIER_NOTE = {
    "easy": ["", "Read the line from left to right."],
    "medium": ["Read the line from left to right.", "Look at the line as x increases from left to right."],
    "hard": [
        "Use the line itself, read from left to right.",
        "Judge the line alone, as x increases from left to right.",
    ],
    "exam": [
        "The two axes use different scales, so judge the line by its direction.",
        "Note that one grid step is not the same on both axes; read the line from left to right.",
    ],
}
_EXAM_SCALES = ((1, 2), (1, 5), (1, 10), (2, 5), (2, 10))  # (xs, ys) with ys >= 2 xs

_NOTES = {
    # (key class, offered label) -> misconception id
    ("positive", "negative"): "reads-rising-line-as-falling",
    ("positive", "zero"): "reads-shallow-line-as-flat",
    ("negative", "positive"): "reads-falling-line-as-rising",
    ("negative", "zero"): "reads-shallow-line-as-flat",
    ("zero", "positive"): "reads-height-above-axis-as-positive-gradient",
    ("zero", "negative"): "reads-height-below-axis-as-negative-gradient",
}


def _note(tier: str, cls: str, label: str) -> str:
    """The misconception behind offering ``label`` when the key is ``cls``.
    Where the y-intercept contradicts the gradient (medium) the rising/falling
    swap is the intercept-sign reading."""
    if tier == "medium" and {cls, label} == {"positive", "negative"}:
        return "sign-of-intercept-as-sign-of-gradient"
    return _NOTES[(cls, label)]


def _draw(rng, tier: str, cls: str):
    """A line of class ``cls`` in a window fitting the tier; (win, pa, pb, M, C)."""
    for _ in range(_ATTEMPTS):
        nu, nv = rng.randint(8, 10), rng.randint(8, 10)
        xs, ys = 1, 1
        if tier == "easy":
            nu = nv = 9
            u0, v0 = -4, -4
            slopes = [_F(1), _F(2), _F(-1), _F(-2)]
        elif tier == "medium":
            u0 = rng.randint(-(nu - 2), -1)  # x = 0 drawn, off the edge
            v0 = rng.randint(-(nv - 2), -1)
            slopes = [_F(s) for s in (-3, -2, -1, 1, 2, 3)] + [_F(1, 2), _F(-1, 2)]
        elif tier == "hard":
            k = rng.randint(1, 6)  # x = 0 NOT drawn
            u0 = k if rng.random() < 0.5 else -k - (nu - 1)
            v0 = rng.randint(-(nv - 1) - 4, 4)
            slopes = [_F(s) for s in (-3, -2, -1, 1, 2, 3)] + [_F(1, 2), _F(-1, 2), _F(1, 3), _F(-1, 3)]
        else:
            u0 = rng.randint(-(nu - 1), 1)
            v0 = rng.randint(-(nv - 1), 1)
            xs, ys = rng.choice(_EXAM_SCALES)
            slopes = (
                [_F(s) for s in (1, 2, 3, 4)]
                + [_F(-s) for s in (1, 2, 3, 4)]
                + [_F(1, 2), _F(1, 3), _F(-1, 2), _F(-1, 3)]
            )
        win = Window(u0, u0 + nu - 1, v0, v0 + nv - 1, xs=xs, ys=ys)
        if cls == "zero":
            M = _F(0)
        else:
            want = 1 if cls == "positive" else -1
            pool = [s for s in slopes if (s > 0) == (want > 0)]
            if tier == "exam":
                # the picture's steepness is independent of the sign: a flat-looking
                # (|M| < 1) or steep-looking (|M| >= 2) line is equally likely.
                flat = rng.random() < 0.5
                pool = [s for s in pool if (abs(s) < 1) == flat]
            M = rng.choice(pool)
        # intercept (grid units): where the line meets u = 0
        C = rng.randint(win.v0 - 6, win.v1 + 6) if not win.shows_y_axis else rng.randint(win.v0, win.v1)
        if tier == "easy":
            C = rng.randint(-2, 2) if cls != "zero" else rng.choice((-3, -2, -1, 1, 2, 3))
            if cls != "zero" and C != 0 and (C > 0) != (M > 0):
                continue
        elif tier == "medium":
            C = rng.randint(1, 4) * (-1 if cls == "positive" else (1 if cls == "negative" else rng.choice((-1, 1))))
            if not (win.v0 <= C <= win.v1):
                continue
        elif cls == "zero" and C == 0:
            continue
        if tier == "hard" and win.shows_y_axis:
            continue
        pts = lattice_points(win, M, C)
        try:
            pa, pb = pick_two_points(rng, pts, min_du=2)
        except ValueError:
            continue
        if cls != "zero" and pa[1] == pb[1]:
            continue
        return win, pa, pb, M, C
    raise ValueError(f"no plausible draw for tier {tier!r}, class {cls!r}")


@generator(
    "pv1a_gradient_sign",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv1a_gradient_sign(rng, tier="easy", notes=False):
    if tier not in _TIER_NOTE:
        raise ValueError(f"unknown tier {tier!r}")
    cls = rng.choice(_CLASSES)  # the class is drawn first, uniformly
    for _ in range(50):
        win, pa, pb, M, C = _draw(rng, tier, cls)
        fig = line_figure(win, [(pa, pb)], _ALT)
        rm, rc = read_line(fig)
        got = "positive" if rm > 0 else ("negative" if rm < 0 else "zero")
        check(got == cls, "the drawn line does not match its class")
        check(fig["equal_scale"] is (tier != "exam"), "unequal scales belong to the exam tier alone")
        distractors = [c for c in _CLASSES if c != cls]
        check(len(set([cls, *distractors])) == 3, "options must be three distinct labels")
        try:
            figures.assert_no_leak(fig, [cls, *distractors])
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a question")

    rng.shuffle(distractors)
    text = " ".join(p for p in (phrase(rng, _OPEN), phrase(rng, _ASK), phrase(rng, _TIER_NOTE[tier])) if p)
    (x1, y1), (x2, y2) = [(u * win.xs, v * win.ys) for (u, v) in (pa, pb)]
    working = [
        f"Read two points on the line: ({x1}, {y1}) and ({x2}, {y2}).",
        f"As x goes from {x1} to {x2}, y goes from {y1} to {y2}.",
        f"So the gradient is {cls}.",
    ]
    return dict(
        text=text,
        answer=cls,
        answer_text=cls,
        answer_latex=latex_answer(cls, plain=True),
        working=working,
        distractors=distractors,
        **({"option_notes": {d: _note(tier, cls, d) for d in distractors}} if notes else {}),
        figure=fig,
    )


def _read_gradient_sign(question) -> str:
    """Plausibility reader: the label from the figure's through points alone
    (the regrade sweep maps an mcq reader's option text to its index)."""
    m, _ = read_line(question.figure)
    return "positive" if m > 0 else ("negative" if m < 0 else "zero")


figures.register_reader("pv1a_gradient_sign", _read_gradient_sign)
