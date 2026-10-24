"""Generates slope-and-intercept questions (ACMGM041): read (m, c) off a drawn straight line, answered as a pair.

One skill: ``senior-determine-the-slope-and-intercepts-of-a-straight``
(ACMGM041), ``domain: linear-relations``, ``answer: pair``, answer text
``(m, c)``. The figure is the whole stimulus: a straight line drawn on a
labelled grid. Nothing the key needs is printed (no equation, no coordinate,
no rate). Two points on the line are grid intersections at least two columns
apart, so both parts are read from the grid.

Four tiers, escalating by what the learner DECIDES (not by reading precision):

- ``easy``: the y-axis is drawn and the line crosses it at a grid
  intersection; integer gradient. Decision: read c at the axis, rise over run
  for m.
- ``medium``: the window excludes x = 0 (the y-axis is not drawn), so c is
  found by running the line back to x = 0 using the gradient; the gradient is
  an integer or a half.
- ``hard``: a fractional gradient (halves, thirds, quarters, three halves ...)
  and the y-intercept lies outside the visible window, so it is extrapolated
  from the gradient, never read.
- ``exam``: the two axes use different scales (a grid step is not one unit on
  both axes). Decision: convert rise and run to the units on the axis labels
  instead of counting squares; c is in y-axis units.

Distractors are named misconceptions: swapped (c, m); m counted in squares;
sign error on m; c read as the y of one through point; inverted gradient
(run over rise); sign error on c. Every one is checked against the real
grader's rules (an integer c is compared exactly, a fractional m to 0.051),
so none can be accepted.

The plausibility reader (``figures.register_reader``) re-derives ``(m, c)``
from the figure's two through points and axis settings alone.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import figures
from mathgen.core import (
    check,
    check_pairwise_distinct,
    generator,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)
from mathgen.generators.pv1a_line_figure_build import (
    Window,
    line_figure,
    read_line,
    real_line,
)

_ATTEMPTS = 2000
_ALT = "a straight line on a coordinate grid"
# A typed m is compared to 0.051 when it is not an exact decimal/fraction, so a
# distractor must differ from the key by more than this in m, or in c at all.
_M_CLEARANCE = Fraction(6, 100)

_F = Fraction
_INT_SLOPES = tuple(_F(s) for s in (-3, -2, -1, 1, 2, 3))
_HALF_SLOPES = tuple(_F(n, 2) for n in (-3, -1, 1, 3))
_FRAC_SLOPES = tuple(
    _F(n, d) * s
    for (n, d) in ((1, 2), (1, 3), (2, 3), (3, 2), (3, 4), (4, 3), (5, 2), (1, 4), (5, 3), (4, 5))
    for s in (1, -1)
)
_EXAM_SLOPES = tuple(_F(s) for s in (-4, -3, -2, -1, 1, 2, 3, 4))
_EXAM_SCALES = ((1, 2), (1, 5), (2, 1), (5, 1), (2, 5), (5, 2))

_TIERS = {
    "easy": dict(slopes=_INT_SLOPES, c_range=(-5, 5), xs_ys=((1, 1),)),
    "medium": dict(slopes=_INT_SLOPES + _HALF_SLOPES, c_range=(-6, 6), xs_ys=((1, 1),)),
    "hard": dict(slopes=_FRAC_SLOPES, c_range=(-9, 9), xs_ys=((1, 1),)),
    "exam": dict(slopes=_EXAM_SLOPES, c_range=(-6, 6), xs_ys=_EXAM_SCALES),
}

_OPEN = [
    "The graph shows a straight line on a coordinate grid.",
    "A straight line is drawn on the coordinate grid below.",
    "Look at the straight line graphed on this grid.",
    "The diagram shows the graph of a straight line.",
    "Study the straight line drawn on this coordinate grid.",
    "The grid below has a straight line drawn on it.",
]
_ASK = [
    "State its gradient m and its y-intercept c as the ordered pair (m, c).",
    "Find the gradient and the y-intercept of the line, and give them as (m, c).",
    "What are the gradient m and the y-intercept c of the line? Answer in the form (m, c).",
    "Write down (m, c), where m is the gradient and c is the y-intercept of the line.",
    "Determine the slope and the y-intercept of the line. Give your answer as (m, c).",
    "Read the gradient and the y-intercept from this line and write them as the pair (m, c).",
]
_CONVENTION = {
    "easy": [
        "Write m as a whole number.",
        "The gradient is a whole number.",
    ],
    "medium": [
        "Write m as a whole number or as a fraction.",
        "Give m as an integer or as a fraction.",
    ],
    "hard": [
        "Write m as a whole number or as a fraction in lowest terms.",
        "Give m exactly, as an integer or a fraction in lowest terms.",
    ],
    "exam": [
        "The two axes use different scales, so give m and c in the units on the axes.",
        "Note that one grid step is not the same on both axes; use the scale on each axis for m and c.",
    ],
}


def _rational_latex(v: Fraction) -> str:
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v < 0 else ""
    return rf"{sign}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def _draw(rng, tier: str):
    """Sample a line and a window. Returns the pieces or raises if no draw fits."""
    cfg = _TIERS[tier]
    for _ in range(_ATTEMPTS):
        M = rng.choice(cfg["slopes"])
        C = rng.randint(*cfg["c_range"])
        nu, nv = rng.randint(8, 10), rng.randint(8, 10)
        if tier == "easy":
            u0 = rng.randint(-(nu - 1), 0)  # x = 0 is drawn
        elif tier == "medium":
            k = rng.randint(1, 4)  # x = 0 is NOT drawn
            u0 = k if rng.random() < 0.5 else -k - (nu - 1)
        else:
            u0 = rng.randint(-(nu - 1) - 3, 4)
        us = [u for u in range(u0, u0 + nu) if (M * u).denominator == 1]
        pairs = [(a, b) for i, a in enumerate(us) for b in us[i + 1 :] if b - a >= 2]
        if not pairs:
            continue
        ua, ub = rng.choice(pairs)
        va, vb = int(M * ua + C), int(M * ub + C)
        if abs(va - vb) > nv - 1:
            continue
        lo, hi = max(va, vb) - (nv - 1), min(va, vb)
        v0 = rng.randint(lo, hi)
        c_visible = v0 <= C <= v0 + nv - 1
        if tier in ("easy", "medium") and not c_visible:
            continue
        if tier == "hard" and c_visible:
            continue
        xs, ys = rng.choice(cfg["xs_ys"])
        win = Window(u0, u0 + nu - 1, v0, v0 + nv - 1, xs=xs, ys=ys)
        if tier == "easy" and not win.shows_y_axis:
            continue
        if tier == "medium" and win.shows_y_axis:
            continue
        return M, C, win, (ua, va), (ub, vb)
    raise ValueError(f"no plausible draw for tier {tier!r}")


def _candidates(rng, m: Fraction, c: int, M: Fraction, C: int, first_y: int, tier: str):
    """(text, note id) per named misconception. The four main ones come
    first, in a seed-shuffled order (so no misconception is always the first
    distractor offered); the extras only fill in when a named one collides with
    the key. The text is ``(m, c)`` spelled like the key."""

    def t(a, b):
        return f"({rational_text(a)}, {rational_text(b)})"

    named = [(t(c, m), "swapped-m-and-c"), (t(-m, c), "sign-error-on-m"), (t(m, first_y), "c-read-from-a-point")]
    extras = [(t(1 / m, c), "inverted-gradient"), (t(m, -c), "sign-error-on-c")]
    if tier == "exam":
        named.append((t(M, c), "m-counted-in-squares"))
        extras.insert(0, (t(m, C), "c-counted-in-squares"))
    rng.shuffle(named)
    return named + extras


def _clears(m: Fraction, c: int, cand_text: str) -> bool:
    """True when the real pair grader cannot accept ``cand_text`` for key (m, c):
    c is an integer compared exactly; m is compared to at most 0.051."""
    body = cand_text.strip("()")
    a, b = (Fraction(p.strip()) for p in body.split(", "))
    return b != c or abs(a - m) > _M_CLEARANCE


@generator(
    "pv1a_slope_intercept",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv1a_slope_intercept(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(400):
        M, C, win, pa, pb = _draw(rng, tier)
        m, c_frac = real_line(win, M, C)
        check(c_frac.denominator == 1, "c must be an integer")
        c = int(c_frac)
        fig = line_figure(win, [(pa, pb)], _ALT)

        # Independent route: re-derive from the figure alone and compare.
        rm, rc = read_line(fig)
        check(rm == m and rc == c, "the drawn line does not match (m, c)")
        if tier == "exam":
            check(win.xs != win.ys and fig["equal_scale"] is False, "exam tier needs unequal scales")
        else:
            check(fig["equal_scale"] is True, "only the exam tier uses unequal scales")

        answer_text = f"({rational_text(m)}, {c})"
        first_y = pa[1] * win.ys
        cands = [(s, n) for s, n in _candidates(rng, m, c, M, C, first_y, tier) if _clears(m, c, s)]
        texts = pick_distractors(answer_text, [s for s, _ in cands], n=3)
        if len(texts) < 3:
            continue
        check_pairwise_distinct(*[_pair_value(x) for x in [answer_text, *texts]])
        for x in texts:
            check(_clears(m, c, x), "a distractor would be accepted by the grader")
        try:
            figures.assert_no_leak(fig, [answer_text, *texts])
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a question with three distractors")

    note_of = dict(cands)
    text = " ".join(
        [
            phrase(rng, _OPEN),
            phrase(rng, _ASK),
            phrase(rng, _CONVENTION[tier]),
        ]
    )
    (x1, y1), (x2, y2) = [(u * win.xs, v * win.ys) for (u, v) in (pa, pb)]
    working = [
        f"Read two points on the line: ({x1}, {y1}) and ({x2}, {y2}).",
        f"m = ({y2} - {paren_negative(y1)}) / ({x2} - {paren_negative(x1)}) = {rational_text(m)}.",
        f"Running the line to x = 0 (or substituting ({x1}, {y1})) gives c = {c}.",
    ]
    return dict(
        text=text,
        answer={"m": float(m), "c": c},
        answer_text=answer_text,
        answer_latex=rf"\left({_rational_latex(m)}, {c}\right)",
        working=working,
        distractors=texts,
        **({"option_notes": {x: note_of[x] for x in texts}} if notes else {}),
        figure=fig,
    )


def _pair_value(text: str):
    a, b = (Fraction(p.strip()) for p in text.strip("()").split(", "))
    return (a, b)


def _read_slope_intercept(question) -> str:
    """Plausibility reader: ``(m, c)`` from the figure's through points alone."""
    m, c = read_line(question.figure)
    return f"({rational_text(m)}, {rational_text(c)})"


figures.register_reader("pv1a_slope_intercept", _read_slope_intercept)
