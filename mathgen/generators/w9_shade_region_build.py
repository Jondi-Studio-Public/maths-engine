"""Generates `w9_shade_region` questions (easy, medium, hard) and the exam chain `exd_x_w9_shade_region`.

``shade-linear-inequality-region`` (``linear-relations``, answer ``half_planes``): the learner shades the region of the
coordinate plane that a linear inequality (or a system of them) describes. For each inequality the front end sends the
boundary line as two points, whether it is solid or dashed, and one point inside the shaded side
(``{"half_planes": [{"through": [[0, -1], [1, 1]], "boundary": "dashed", "shade": [0, 3]}]}``). The grader compares
each half-plane exactly (any two points on the line, any point strictly on the right side, solid for ``≤ ≥`` and dashed
for ``< >``). The typed form is the inequality itself: ``y > 2x - 1``, or several joined by "and".

* easy: ``y`` against ``mx + c`` with a whole gradient, or a vertical or horizontal boundary (``x ≥ 2``, ``y < -3``).
* medium: an inequality in standard form (``2x + 3y ≤ 6``, ``x - 2y > 4``) or with ``y`` first (``y - 2x ≥ 3``): the
  boundary comes from the intercepts, and a negative coefficient of ``y`` turns the shading round.
* hard: a system: two sloping boundaries, the triangle cut off by ``x ≥ 0``, ``y ≥ 0`` and a sloping line, or a strip
  between two parallel lines (``y - 2x > -1`` and ``y - 2x ≤ 3``).
* exam (``exd_x_w9_shade_region``): the gradient of the boundary, the region, how many given points lie in it, and the
  region cut by ``x ≥ 0`` and ``y ≥ 0``.

Each region is checked a second way: every whole-number point of the grid is tested against the inequalities as written
in the question, and against the half-planes of the answer, and the two must agree. Every wrong region is checked against
the `half_planes` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..core import attempts, check, generator, phrase
from ..grading import interactive as ia
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import HALF, dist, enough, frac_text, ineq_text, part, plane, point_text, retry_draw

TOPIC = "w9_shade_region"
EXAM_TOPIC = "exd_x_w9_shade_region"

AREA = "shade-linear-inequality-region"
MISCONCEPTIONS = [
    ("w9-region-wrong-side", "Shades the side of the boundary line that does not satisfy the inequality.", AREA),
    (
        "w9-region-boundary-style",
        "Draws a solid boundary for a strict inequality, or a dashed boundary for one that includes the line.",
        AREA,
    ),
    (
        "w9-region-intercept-sign",
        "Puts the boundary line through the y-axis at the opposite end: the sign of the constant is wrong.",
        AREA,
    ),
    ("w9-region-gradient-sign", "Draws the boundary sloping the wrong way: the sign of the gradient is wrong.", AREA),
    (
        "w9-region-gradient-inverted",
        "Draws a boundary with the reciprocal of the gradient, going across by the rise and up by the run.",
        AREA,
    ),
    ("w9-region-axes-swapped", "Swaps x and y when drawing the boundary, which reflects it in the line y = x.", AREA),
    (
        "w9-region-sign-not-flipped",
        "Rearranges to make y positive by dividing by a negative number without reversing the inequality sign, so the shading is on the wrong side.",
        AREA,
    ),
    (
        "w9-region-intercepts-swapped",
        "Swaps the x- and y-intercepts of the boundary when the equation is in standard form.",
        AREA,
    ),
    (
        "w9-region-intercept-not-divided",
        "Uses the right-hand constant as an intercept without dividing by the coefficient of the variable.",
        AREA,
    ),
    (
        "w9-region-one-condition-only",
        "Shades the region for one of the inequalities in a system and ignores the others.",
        AREA,
    ),
    (
        "w9-region-union-for-intersection",
        "Shades everywhere that satisfies at least one inequality, when the system needs all of them at once.",
        AREA,
    ),
    (
        "w9-region-strip-one-side",
        "Shades one side of only one of the two parallel boundaries instead of the strip between them.",
        AREA,
    ),
    ("w9-region-strip-outside", "Shades outside the two parallel boundaries instead of between them.", AREA),
    (
        "w9-region-quadrant-ignored",
        "Leaves out the conditions x ≥ 0 and y ≥ 0, so the region extends beyond the first quadrant.",
        AREA,
    ),
    (
        "w9-region-gradient-from-coefficients",
        "Reads the gradient of ax + by = c as a/b, forgetting the minus sign.",
        AREA,
    ),
    ("w9-region-gradient-as-intercept", "Gives the y-intercept c/b where the gradient was asked.", AREA),
    ("w9-region-gradient-inverted-number", "Gives -b/a, the reciprocal of the gradient.", AREA),
    ("w9-region-count-boundary-strict", "Counts a point on a dashed boundary as inside the region.", AREA),
    ("w9-region-count-boundary-missed", "Leaves out a point on a solid boundary, which is part of the region.", AREA),
    ("w9-region-count-complement", "Counts the points outside the region instead of inside it.", AREA),
]

_ALT = "A blank coordinate grid with x and y axes, for shading the region of an inequality."


def _figure(alt=_ALT):
    return plane(reply="half_planes", tools=["half_plane"], alt=alt, snap=1, max_points=6)


def _lattice():
    r = range(-HALF, HALF + 1)
    return sorted(((x, y) for x in r for y in r), key=lambda p: (abs(p[0]) + abs(p[1]), p))


_LATTICE = _lattice()


def _item(a: int, b: int, c: int, op: str):
    """The answer item for the inequality ``ax + by op c``: two lattice points on the boundary and a lattice point
    strictly inside the shaded side. None when the boundary has fewer than two lattice points on the grid, or no
    lattice point is strictly on the shaded side."""
    if a == 0 and b == 0:
        return None
    on = [p for p in _LATTICE if a * p[0] + b * p[1] == c]
    if len(on) < 2:
        return None
    on.sort()
    sign = 1 if op in (">", "≥") else -1
    shade = next((p for p in _LATTICE if sign * (a * p[0] + b * p[1] - c) > 0), None)
    if shade is None:
        return None
    return (on[0], on[-1], shade, "solid" if op in ("≤", "≥") else "dashed")


def _answer(items) -> dict:
    return ia.half_planes_answer(items, x_range=(-HALF, HALF), y_range=(-HALF, HALF), snap=1)


def _holds(a: int, b: int, c: int, op: str, x, y) -> bool:
    v = a * x + b * y
    return {"<": v < c, "≤": v <= c, ">": v > c, "≥": v >= c}[op]


def _fields(specs, text: str) -> dict:
    """Answer fields for a region given as ``[(a, b, c, op), ...]``; the grid test of the answer is the second way."""
    items = [_item(*s) for s in specs]
    if any(i is None for i in items):
        raise ValueError("a boundary has no lattice points on the grid")
    answer = _answer(items)
    keys = ia._expected_half_planes(answer)
    for x, y in _LATTICE:
        direct = all(_holds(*s, x, y) for s in specs)
        via_keys = all(
            (ka * x + kb * y + kc >= 0) if closed else (ka * x + kb * y + kc > 0) for ka, kb, kc, closed in keys
        )
        check(direct == via_keys, f"the answer and the inequalities disagree at ({x}, {y})")
    return dict(answer=answer, answer_text=text, answer_latex="plain:" + text, answer_kind="half_planes")


def _std_text(a: int, b: int, c: int, op: str, y_first: bool = False) -> str:
    if y_first and b == 1 and a != 0:
        left = f"y - {abs(a)}x" if a < 0 else f"y + {a}x"
        left = left.replace(" 1x", " x")
        return f"{left} {op} {c}"
    parts = []
    for coef, var in ((a, "x"), (b, "y")):
        if coef == 0:
            continue
        mag = "" if abs(coef) == 1 else str(abs(coef))
        parts.append(("- " if coef < 0 else "+ ") + mag + var if parts else ("-" if coef < 0 else "") + mag + var)
    return f"{' '.join(parts)} {op} {c}"


def _dist(fields, candidates, n=4):
    """`candidates` is ``[(specs, text, tag), ...]``: wrong regions as inequality specs with the text a learner would
    type. Regions with no lattice boundary are skipped."""
    cands = []
    for specs, text, tag in candidates:
        items = [_item(*s) for s in specs]
        if any(i is None for i in items):
            continue
        cands.append((text, _answer(items), tag))
    return dist("half_planes", fields["answer"], fields["answer_text"], cands, n)


_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_STRICTNESS = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    kind = rng.choice(("slope", "slope", "slope", "vertical", "horizontal"))
    op = rng.choice(list(_FLIP))
    if kind == "slope":
        for _ in attempts(100, "a boundary with lattice points on the grid"):
            m = rng.choice((-2, -1, 1, 2))
            c = rng.randint(-4, 4)
            if not (c == 0 and abs(m) == 1) and _item(-m, 1, c, op):
                break
        specs = [(-m, 1, c, op)]
        text_ineq = ineq_text(m, c, op)
        swapped_rhs = ineq_text(m, c, op)[4:].replace("x", "y")
        wrong = [
            ([(-m, 1, c, _FLIP[op])], ineq_text(m, c, _FLIP[op]), "w9-region-wrong-side"),
            ([(-m, 1, c, _STRICTNESS[op])], ineq_text(m, c, _STRICTNESS[op]), "w9-region-boundary-style"),
            ([(-m, 1, -c, op)], ineq_text(m, -c, op), "w9-region-intercept-sign") if c else None,
            ([(m, 1, c, op)], ineq_text(-m, c, op), "w9-region-gradient-sign"),
            ([(1, -m, c, op)], f"x {op} {swapped_rhs}", "w9-region-axes-swapped"),
        ]
        wrong = [w for w in wrong if w]
    elif kind == "vertical":
        k = rng.choice([v for v in range(-6, 7) if v])
        specs = [(1, 0, k, op)]
        text_ineq = f"x {op} {k}"
        wrong = [
            ([(1, 0, k, _FLIP[op])], f"x {_FLIP[op]} {k}", "w9-region-wrong-side"),
            ([(1, 0, k, _STRICTNESS[op])], f"x {_STRICTNESS[op]} {k}", "w9-region-boundary-style"),
            ([(1, 0, -k, op)], f"x {op} {-k}", "w9-region-intercept-sign"),
            ([(0, 1, k, op)], f"y {op} {k}", "w9-region-axes-swapped"),
            ([(0, 1, k, _FLIP[op])], f"y {_FLIP[op]} {k}", "w9-region-axes-swapped"),
        ]
    else:
        k = rng.choice([v for v in range(-6, 7) if v])
        specs = [(0, 1, k, op)]
        text_ineq = f"y {op} {k}"
        wrong = [
            ([(0, 1, k, _FLIP[op])], f"y {_FLIP[op]} {k}", "w9-region-wrong-side"),
            ([(0, 1, k, _STRICTNESS[op])], f"y {_STRICTNESS[op]} {k}", "w9-region-boundary-style"),
            ([(0, 1, -k, op)], f"y {op} {-k}", "w9-region-intercept-sign"),
            ([(1, 0, k, op)], f"x {op} {k}", "w9-region-axes-swapped"),
            ([(1, 0, k, _FLIP[op])], f"x {_FLIP[op]} {k}", "w9-region-axes-swapped"),
        ]
    fields = _fields(specs, text_ineq)
    d = _dist(fields, wrong)
    enough(d, 3)
    text = phrase(
        rng,
        (
            "Shade the region of the grid where {ineq}. Use a dashed boundary for < and >, and a solid boundary for ≤ and ≥.",
            "Draw the boundary line and shade the region {ineq}. A strict inequality has a dashed boundary; one that includes the line has a solid boundary.",
        ),
        ineq=text_ineq,
    )
    solid = op in ("≤", "≥")
    working = [
        f"The boundary line is {text_ineq.replace(op, '=', 1)}. It is {'solid, because the line is included' if solid else 'dashed, because the line is not included'}.",
        "Test a point not on the line, such as (0, 0) if it is clear of the line, and shade the side where the inequality is true.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


# ---------------------------------------------------------------------------------------------- medium


def _intercept_line(rng):
    """A line ax + by = c with whole intercepts p and q on the grid, in lowest terms, with a > 0."""
    for _ in attempts(300, "a standard-form line with whole intercepts"):
        p = rng.choice([v for v in range(-6, 7) if v])
        q = rng.choice([v for v in range(-6, 7) if v])
        if abs(p) == abs(q):
            continue
        a, b, c = q, p, p * q
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        a, b, c = a // g, b // g, c // g
        if a < 0:
            a, b, c = -a, -b, -c
        if abs(a) <= 9 and abs(b) <= 9:
            return a, b, c, p, q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    if rng.random() < 0.7:
        a, b, c, p, q = _intercept_line(rng)
        op = rng.choice(list(_FLIP))
        text_ineq = _std_text(a, b, c, op)
        specs = [(a, b, c, op)]
        pq_text = f"({p}, 0) and (0, {q})"
        wrong = [
            (
                [(a, b, c, _FLIP[op])],
                _std_text(a, b, c, _FLIP[op]),
                "w9-region-sign-not-flipped" if b < 0 else "w9-region-wrong-side",
            ),
            ([(a, b, c, _STRICTNESS[op])], _std_text(a, b, c, _STRICTNESS[op]), "w9-region-boundary-style"),
            ([(b, a, c, op)], _std_text(b, a, c, op), "w9-region-intercepts-swapped"),
            ([(a, b, -c, op)], _std_text(a, b, -c, op), "w9-region-intercept-sign"),
            ([(-a, b, c, op)], _std_text(-a, b, c, op), "w9-region-gradient-sign"),
        ]
        working = [
            f"Boundary: put x = 0 to get y = {frac_text(Fraction(c, b))}, and y = 0 to get x = {frac_text(Fraction(c, a))}. The line passes through {pq_text}.",
            f"The line is {'solid' if op in '≤≥' else 'dashed'}. Test (0, 0): {a}(0) + {b}(0) = 0, and 0 {op} {c} is {'true' if _holds(a, b, c, op, 0, 0) else 'false'}, so "
            + (
                "shade the side that contains the origin."
                if _holds(a, b, c, op, 0, 0)
                else "shade the side that does not contain the origin."
            ),
        ]
    else:
        for _ in attempts(200, "a y-first inequality with lattice points"):
            m = rng.choice((-3, -2, -1, 2, 3))
            k = rng.randint(-5, 5)
            op = rng.choice(list(_FLIP))
            if k and _item(-m, 1, k, op):
                break
        a, b, c = -m, 1, k
        text_ineq = _std_text(a, b, c, op, y_first=True)
        specs = [(a, b, c, op)]
        wrong = [
            ([(a, b, c, _FLIP[op])], _std_text(a, b, c, _FLIP[op], True), "w9-region-wrong-side"),
            ([(a, b, c, _STRICTNESS[op])], _std_text(a, b, c, _STRICTNESS[op], True), "w9-region-boundary-style"),
            ([(-a, b, c, op)], _std_text(-a, b, c, op, True), "w9-region-gradient-sign"),
            ([(a, b, -c, op)], _std_text(a, b, -c, op, True), "w9-region-intercept-sign"),
            ([(b, a, c, op)], _std_text(b, a, c, op), "w9-region-axes-swapped"),
        ]
        working = [
            f"Rearrange: y {op} {ineq_text(m, k, op)[4:]}. The boundary is y = {ineq_text(m, k, '=')[4:]}, with gradient {m} and y-intercept {k}.",
            f"The boundary is {'solid' if op in '≤≥' else 'dashed'}; shade {'above' if op in '>≥' else 'below'} it.",
        ]
    fields = _fields(specs, text_ineq)
    d = _dist(fields, wrong)
    enough(d, 3)
    text = phrase(
        rng,
        (
            "Shade the region {ineq}. Use a dashed boundary for < and >, and a solid boundary for ≤ and ≥.",
            "Show the inequality {ineq} on the grid by shading its region. A strict inequality has a dashed boundary.",
        ),
        ineq=text_ineq,
    )
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


# ---------------------------------------------------------------------------------------------- hard


def _hard_two_lines(rng):
    for _ in attempts(500, "two sloping boundaries that cross on the grid"):
        m1 = rng.choice((-2, -1, 1, 2))
        c1 = rng.randint(-4, 4)
        m2 = rng.choice((-2, -1, 1, 2))
        c2 = rng.randint(-4, 4)
        if m1 == m2 or (c1 == 0 and abs(m1) == 1) or (c2 == 0 and abs(m2) == 1):
            continue
        x = Fraction(c2 - c1, m1 - m2)
        y = m1 * x + c1
        op1, op2 = rng.choice(list(_FLIP)), rng.choice(list(_FLIP))
        if x.denominator == 1 and abs(x) <= 6 and abs(y) <= 7 and _item(-m1, 1, c1, op1) and _item(-m2, 1, c2, op2):
            break
    specs = [(-m1, 1, c1, op1), (-m2, 1, c2, op2)]
    t1, t2 = ineq_text(m1, c1, op1), ineq_text(m2, c2, op2)
    text_ineq = f"{t1} and {t2}"
    fields = _fields(specs, text_ineq)
    cands = [
        ([specs[0]], t1, "w9-region-one-condition-only"),
        ([specs[1]], t2, "w9-region-one-condition-only"),
        ([(-m1, 1, c1, _FLIP[op1]), specs[1]], f"{ineq_text(m1, c1, _FLIP[op1])} and {t2}", "w9-region-wrong-side"),
        ([specs[0], (-m2, 1, c2, _FLIP[op2])], f"{t1} and {ineq_text(m2, c2, _FLIP[op2])}", "w9-region-wrong-side"),
        (
            [(-m1, 1, c1, _STRICTNESS[op1]), specs[1]],
            f"{ineq_text(m1, c1, _STRICTNESS[op1])} and {t2}",
            "w9-region-boundary-style",
        ),
        (
            [specs[0], (-m2, 1, c2, _STRICTNESS[op2])],
            f"{t1} and {ineq_text(m2, c2, _STRICTNESS[op2])}",
            "w9-region-boundary-style",
        ),
        ([(m1, 1, c1, op1), specs[1]], f"{ineq_text(-m1, c1, op1)} and {t2}", "w9-region-gradient-sign"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    text = (
        f"Shade the region where both {t1} and {t2} are true. Use a dashed boundary for < and >, "
        "and a solid boundary for ≤ and ≥."
    )
    working = [
        f"Draw the boundary y = {ineq_text(m1, c1, '=')[4:]} ({'solid' if op1 in '≤≥' else 'dashed'}) and shade {'above' if op1 in '>≥' else 'below'} it.",
        f"Draw the boundary y = {ineq_text(m2, c2, '=')[4:]} ({'solid' if op2 in '≤≥' else 'dashed'}) and shade {'above' if op2 in '>≥' else 'below'} it.",
        "The answer is the region where the two shadings overlap.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _hard_triangle(rng):
    for _ in attempts(300, "a triangle cut off by the axes"):
        p, q = rng.randint(2, 7), rng.randint(2, 7)
        if p == q:
            continue
        a, b, c = q, p, p * q
        g = gcd(gcd(a, b), c)
        a, b, c = a // g, b // g, c // g
        op = rng.choice(("≤", "<"))
        if _item(a, b, c, op):
            break
    specs = [(1, 0, 0, "≥"), (0, 1, 0, "≥"), (a, b, c, op)]
    tri = _std_text(a, b, c, op)
    text_ineq = f"x ≥ 0 and y ≥ 0 and {tri}"
    fields = _fields(specs, text_ineq)
    cands = [
        ([(a, b, c, op)], tri, "w9-region-quadrant-ignored"),
        ([specs[0], specs[2]], f"x ≥ 0 and {tri}", "w9-region-one-condition-only"),
        ([specs[1], specs[2]], f"y ≥ 0 and {tri}", "w9-region-one-condition-only"),
        (
            [specs[0], specs[1], (a, b, c, _FLIP[op])],
            f"x ≥ 0 and y ≥ 0 and {_std_text(a, b, c, _FLIP[op])}",
            "w9-region-wrong-side",
        ),
        (
            [specs[0], specs[1], (a, b, c, _STRICTNESS[op])],
            f"x ≥ 0 and y ≥ 0 and {_std_text(a, b, c, _STRICTNESS[op])}",
            "w9-region-boundary-style",
        ),
        (
            [specs[0], specs[1], (b, a, c, op)],
            f"x ≥ 0 and y ≥ 0 and {_std_text(b, a, c, op)}",
            "w9-region-intercepts-swapped",
        ),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    text = (
        f"Shade the region where x ≥ 0, y ≥ 0 and {tri} are all true. Use a dashed boundary for < and >, "
        "and a solid boundary for ≤ and ≥."
    )
    working = [
        "x ≥ 0 keeps the right of the y-axis (solid) and y ≥ 0 keeps the part above the x-axis (solid).",
        f"For {tri}: the boundary crosses the axes at ({frac_text(Fraction(c, a))}, 0) and (0, {frac_text(Fraction(c, b))}); test (0, 0) and shade the side that contains the origin.",
        "The answer is the triangle where all three shadings overlap.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _hard_strip(rng):
    for _ in attempts(300, "a strip between two parallel lines"):
        m = rng.choice((-2, -1, 1, 2, 3))
        lo = rng.randint(-5, 1)
        hi = lo + rng.randint(2, 5)
        op1, op2 = rng.choice((">", "≥")), rng.choice(("<", "≤"))
        if lo != hi and abs(hi) <= 6 and abs(lo) <= 6 and _item(-m, 1, lo, op1) and _item(-m, 1, hi, op2):
            break
    specs = [(-m, 1, lo, op1), (-m, 1, hi, op2)]
    left = f"y - {m}x" if m > 0 else f"y + {-m}x"
    left = left.replace(" 1x", " x")
    t_lo, t_hi = f"{left} {op1} {lo}", f"{left} {op2} {hi}"
    text_ineq = f"{t_lo} and {t_hi}"
    fields = _fields(specs, text_ineq)
    cands = [
        ([specs[0]], t_lo, "w9-region-strip-one-side"),
        ([specs[1]], t_hi, "w9-region-strip-one-side"),
        (
            [(-m, 1, lo, _FLIP[op1]), (-m, 1, hi, _FLIP[op2])],
            f"{left} {_FLIP[op1]} {lo} and {left} {_FLIP[op2]} {hi}",
            "w9-region-strip-outside",
        ),
        (
            [(-m, 1, lo, op1), (-m, 1, hi, _STRICTNESS[op2])],
            f"{t_lo} and {left} {_STRICTNESS[op2]} {hi}",
            "w9-region-boundary-style",
        ),
        (
            [(-m, 1, -lo, op1), (-m, 1, -hi, op2)],
            f"{left} {op1} {-lo} and {left} {op2} {-hi}",
            "w9-region-intercept-sign",
        ),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    text = (
        f"Shade the region where {t_lo} and {t_hi}. Use a dashed boundary for < and >, "
        "and a solid boundary for ≤ and ≥."
    )
    working = [
        f"Both boundaries have gradient {m}: y = {ineq_text(m, lo, '=')[4:]} and y = {ineq_text(m, hi, '=')[4:]}. They are parallel.",
        f"Shade above the lower line ({'solid' if op1 == '≥' else 'dashed'}) and below the upper line ({'solid' if op2 == '≤' else 'dashed'}): the strip between them.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


_HARD = (_hard_two_lines, _hard_triangle, _hard_strip)


@generator(
    TOPIC,
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_shade_region(rng, tier="easy"):
    """Shade the region of a linear inequality; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=10, presets={"exam": dict(tier="exam")})
def exd_x_w9_shade_region(rng, tier="exam"):
    """Exam chain: gradient of the boundary; the region; how many points lie in it; the region cut by x >= 0, y >= 0."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(500, "an inequality with positive intercepts and mixed test points"):
        p, q = rng.randint(2, 7), rng.randint(2, 7)
        if p == q:
            continue
        a, b, c = q, p, p * q
        g = gcd(gcd(a, b), c)
        a, b, c = a // g, b // g, c // g
        op = rng.choice(("≤", "<", "≥", ">"))
        pts = []
        # five points: some inside, one on the boundary
        pool = [pt for pt in _LATTICE if pt[0] != pt[1] and 0 <= pt[0] <= 7 and 0 <= pt[1] <= 7]
        on = [pt for pt in pool if a * pt[0] + b * pt[1] == c]
        if not on:
            continue
        pts = [rng.choice(on)] + rng.sample([pt for pt in pool if pt not in on], 4)
        rng.shuffle(pts)
        inside = sum(_holds(a, b, c, op, x, y) for x, y in pts)
        if 1 <= inside <= 3 and _item(a, b, c, op) and _item(1, 0, 0, "≥") and _item(0, 1, 0, "≥"):
            break
    gradient = Fraction(-a, b)
    tri = _std_text(a, b, c, op)
    boundary_pt = next(pt for pt in pts if a * pt[0] + b * pt[1] == c)
    strict = op in ("<", ">")
    check(_holds(a, b, c, op, *boundary_pt) == (not strict), "boundary point verdict is wrong")
    names = "ABCDE"
    plist = ", ".join(f"{n}{point_text(pt)}" for n, pt in zip(names, pts))

    f_b = _fields([(a, b, c, op)], tri)
    both = [(1, 0, 0, "≥"), (0, 1, 0, "≥"), (a, b, c, op)]
    f_d = _fields(both, f"x ≥ 0 and y ≥ 0 and {tri}")
    d_a = num_dist(
        gradient,
        [
            (Fraction(a, b), "w9-region-gradient-from-coefficients"),
            (Fraction(c, b), "w9-region-gradient-as-intercept"),
            (Fraction(-b, a), "w9-region-gradient-inverted-number"),
            (Fraction(b, a), "w9-region-gradient-inverted-number"),
        ],
    )
    d_b = _dist(
        f_b,
        [
            ([(a, b, c, _FLIP[op])], _std_text(a, b, c, _FLIP[op]), "w9-region-wrong-side"),
            ([(a, b, c, _STRICTNESS[op])], _std_text(a, b, c, _STRICTNESS[op]), "w9-region-boundary-style"),
            ([(b, a, c, op)], _std_text(b, a, c, op), "w9-region-intercepts-swapped"),
            ([(-a, b, c, op)], _std_text(-a, b, c, op), "w9-region-gradient-sign"),
        ],
    )
    d_d = _dist(
        f_d,
        [
            ([(a, b, c, op)], tri, "w9-region-quadrant-ignored"),
            ([both[0], both[2]], f"x ≥ 0 and {tri}", "w9-region-one-condition-only"),
            (
                [both[0], both[1], (a, b, c, _FLIP[op])],
                f"x ≥ 0 and y ≥ 0 and {_std_text(a, b, c, _FLIP[op])}",
                "w9-region-wrong-side",
            ),
            (
                [both[0], both[1], (a, b, c, _STRICTNESS[op])],
                f"x ≥ 0 and y ≥ 0 and {_std_text(a, b, c, _STRICTNESS[op])}",
                "w9-region-boundary-style",
            ),
        ],
    )
    d_c = num_dist(
        inside,
        [
            (
                inside + 1 if strict else inside - 1,
                "w9-region-count-boundary-strict" if strict else "w9-region-count-boundary-missed",
            ),
            (len(pts) - inside, "w9-region-count-complement"),
            (
                inside + 1 if not strict else inside - 1,
                "w9-region-count-boundary-missed" if not strict else "w9-region-count-boundary-strict",
            ),
            (len(pts), "w9-region-count-complement"),
        ],
    )
    for dl in (d_a, d_b, d_c, d_d):
        enough(dl)
    stem = f"The inequality {tri} is given."
    working_b = [
        f"Boundary: ({frac_text(Fraction(c, a))}, 0) and (0, {frac_text(Fraction(c, b))}); {'solid' if not strict else 'dashed'} line.",
        f"Test (0, 0): 0 {op} {c} is {'true' if _holds(a, b, c, op, 0, 0) else 'false'}.",
    ]
    return dict(
        text=stem,
        figure=_figure(),
        parts=[
            part(
                "Find the gradient of the boundary line.",
                "number",
                num_fields(gradient),
                1,
                [
                    f"{b}y = {c} - {a}x, so y = ({frac_text(Fraction(-a, b))})x + {frac_text(Fraction(c, b))}. The gradient is {frac_text(gradient)}."
                ],
                d_a,
            ),
            part("Shade the region described by the inequality.", "half_planes", f_b, 2, working_b, d_b),
            part(
                f"How many of the points {plist} lie in the shaded region?",
                "number",
                num_fields(inside),
                1,
                [
                    f"{n}: {a}({x}) + {b}({y}) = {a * x + b * y}, and {a * x + b * y} {op} {c} is {'true' if _holds(a, b, c, op, x, y) else 'false'}."
                    for n, (x, y) in zip(names, pts)
                ],
                d_c,
                sensible_range=(-1, None),
            ),
            part(
                f"Shade the region where {tri}, x ≥ 0 and y ≥ 0 are all true.",
                "half_planes",
                f_d,
                3,
                ["x ≥ 0 and y ≥ 0 keep the first quadrant; the sloping boundary cuts it into a triangle."],
                d_d,
            ),
        ],
    )
