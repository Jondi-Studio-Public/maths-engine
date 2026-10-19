"""Generates `w9_parabola_draw` questions (easy, medium, hard) and the exam chain `exd_x_w9_parabola_draw`.

``parabola-from-vertex-and-point`` (``functions-sequences``, answer ``curve``): the learner draws a parabola by placing its
vertex and a second point on it. The front end sends ``{"vertex": [1, -3], "point": [2, -1]}``; the ``curve`` grader
(`mathgen.grading.interactive`) needs the vertex exactly where it is and the second point exactly on the curve, with
a different x, all in exact fractions. Three points on the curve with different x (``{"points": [...]}``) are also
right, so a learner who prefers three points is not penalised. The typed form is ``vertex (1, -3) point (2, -1)``, or
three points ``(0, -1) (2, -1) (3, 5)``.

The question never prints a point to copy that would stand in for the work: the equation, or roots and an intercept,
are given, and the learner finds the vertex and a second point.

* easy: ``y = (x - h)² + k`` or ``y = -(x - h)² + k``.
* medium: ``y = a(x - h)² + k`` with a stretch ``a`` of 2, 3, 1/2 or the same negated.
* hard: the equation expanded (``y = 2x² - 8x + 5``: the vertex comes from -b/2a), factorised (``y = (x - 1)(x - 5)``:
  the vertex is halfway between the roots), or given as two x-intercepts and a y-intercept (a has to be found).
* exam (``exd_x_w9_parabola_draw``): from an expanded equation, the x- and y-coordinates of the vertex, the drawing, and
  the y-intercept.

Each parabola is built from its vertex and stretch, expanded or factorised for the question, and checked a second way:
the vertex form, the expanded form and the factorised form are evaluated at several whole numbers and must agree, and
the curve is symmetric about the vertex. Every wrong parabola is checked against the `curve` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, fmt_signed, generator, phrase
from ..grading import interactive as ia
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import HALF, dist, enough, coef_text, frac_text, part, plane, point_text, retry_draw, sum_text

TOPIC = "w9_parabola_draw"
EXAM_TOPIC = "exd_x_w9_parabola_draw"

AREA = "parabola-from-vertex-and-point"
MISCONCEPTIONS = [
    (
        "w9-parabola-vertex-x-sign",
        "Reads the vertex of y = a(x - h)² + k as (-h, k): the sign inside the bracket is taken at face value.",
        AREA,
    ),
    (
        "w9-parabola-vertex-y-sign",
        "Reads the vertex of y = a(x - h)² + k as (h, -k): the sign of the constant is flipped.",
        AREA,
    ),
    ("w9-parabola-vertex-swapped", "Swaps the coordinates of the vertex, giving (k, h).", AREA),
    (
        "w9-parabola-opens-wrong-way",
        "Draws the parabola opening upwards when a is negative, or downwards when a is positive.",
        AREA,
    ),
    (
        "w9-parabola-stretch-ignored",
        "Ignores the stretch a, so the second point is only 1 up (or down) for each 1 across.",
        AREA,
    ),
    ("w9-parabola-stretch-inverted", "Uses 1/a for the stretch instead of a.", AREA),
    (
        "w9-parabola-stretch-not-squared",
        "Multiplies the distance from the axis of symmetry by a without squaring it, so the second point is wrong.",
        AREA,
    ),
    (
        "w9-parabola-vertex-formula-no-half",
        "Takes the x-coordinate of the vertex of ax² + bx + c as -b, forgetting to divide by 2a.",
        AREA,
    ),
    (
        "w9-parabola-vertex-formula-sign",
        "Takes the x-coordinate of the vertex of ax² + bx + c as b/2a, forgetting the minus sign.",
        AREA,
    ),
    (
        "w9-parabola-vertex-formula-no-a",
        "Takes the x-coordinate of the vertex of ax² + bx + c as -b/2, forgetting to divide by a.",
        AREA,
    ),
    (
        "w9-parabola-vertex-y-is-intercept",
        "Takes the constant term c of the expanded equation as the y-coordinate of the vertex.",
        AREA,
    ),
    (
        "w9-parabola-vertex-x-is-root",
        "Puts the vertex at one of the x-intercepts instead of halfway between them.",
        AREA,
    ),
    (
        "w9-parabola-roots-sum-not-halved",
        "Adds the two x-intercepts to find the axis of symmetry but does not halve the sum.",
        AREA,
    ),
    (
        "w9-parabola-vertex-on-axis",
        "Puts the vertex on the x-axis, at the midpoint of the roots, instead of at the minimum or maximum height.",
        AREA,
    ),
    (
        "w9-parabola-roots-sign",
        "Reads the factor (x - p) as a root at -p, so the x-intercepts are on the wrong side of zero.",
        AREA,
    ),
    (
        "w9-parabola-stretch-from-intercept",
        "Takes the y-intercept as the stretch a, instead of dividing it by the product of the roots.",
        AREA,
    ),
    (
        "w9-parabola-y-intercept-wrong-sign",
        "Gives the y-intercept of a(x - p)(x - q) as -apq, forgetting that two negative signs multiply.",
        AREA,
    ),
    (
        "w9-parabola-y-intercept-coefficient",
        "Gives the coefficient of x as the y-intercept of the expanded equation.",
        AREA,
    ),
    ("w9-parabola-y-intercept-vertex-y", "Gives the y-coordinate of the vertex as the y-intercept.", AREA),
    (
        "w9-parabola-y-intercept-from-vertex-form",
        "Gives the constant k of the vertex form as the y-intercept, without putting x = 0 into the squared bracket.",
        AREA,
    ),
]

Q = Fraction


def _sup(coef, h) -> str:
    return "x" if h == 0 else (f"x - {h}" if h > 0 else f"x + {-h}")


def vertex_form_text(a: Fraction, h: int, k: int) -> str:
    inner = _sup(a, h)
    sq = "x²" if h == 0 else f"({inner})²"
    if a == 1:
        lead = ""
    elif a == -1:
        lead = "-"
    elif a.denominator == 1:
        lead = str(a.numerator)
    else:
        lead = f"({'-' if a < 0 else ''}{abs(a.numerator)}/{a.denominator})"
    body = f"{lead}{sq}"
    tail = "" if k == 0 else (f" + {k}" if k > 0 else f" - {-k}")
    return f"y = {body}{tail}"


def expanded_text(a: int, b: int, c: int) -> str:
    out = ""
    for coef, var in ((a, "x²"), (b, "x"), (c, "")):
        if coef == 0:
            continue
        mag = "" if (abs(coef) == 1 and var) else str(abs(coef))
        piece = f"{mag}{var}"
        if not out:
            out = ("-" if coef < 0 else "") + piece
        else:
            out += (" - " if coef < 0 else " + ") + piece
    return f"y = {out}"


def _fac(r):
    return "x" if r == 0 else (f"(x - {r})" if r > 0 else f"(x + {-r})")


def factorised_text(a: int, p: int, q: int) -> str:
    fac = _fac

    lead = "" if a == 1 else ("-" if a == -1 else str(a))
    return f"y = {lead}{fac(p)}{fac(q)}"


def f_vertex(a, h, k, x):
    return Q(a) * (x - h) ** 2 + k


def _in(p) -> bool:
    return all(abs(v) <= HALF for v in p)


def _second_point(a: Fraction, h: int, k: int):
    """A whole-number point on the parabola other than the vertex, on the grid: 1 across if `a` is whole, else 2."""
    for dx in (1, 2, -1, -2, 3, -3):
        y = f_vertex(a, h, k, h + dx)
        if y.denominator == 1 and _in((h + dx, int(y))):
            return (h + dx, int(y))
    return None


def _answer(a: Fraction, h: int, k: int) -> dict:
    return ia.curve_answer("quadratic", a=a, h=h, k=k, x_range=(-HALF, HALF), y_range=(-HALF, HALF), snap=1)


def _fields(a: Fraction, h: int, k: int) -> dict:
    pt = _second_point(a, h, k)
    if pt is None:
        raise ValueError("no whole-number second point on the grid")
    params = ia._curve_params(_answer(a, h, k))
    # second way: the symmetric points, and the point itself, are on the curve; the vertex is the extreme
    check(ia._on_curve(params, pt) and ia._on_curve(params, (2 * h - pt[0], pt[1])), "the curve is not symmetric")
    check(all((f_vertex(a, h, k, h + t) - k) * a > 0 for t in (1, 2, 3)), "the vertex is not the extreme point")
    text = f"vertex {point_text((h, k))} point {point_text(pt)}"
    return dict(answer=_answer(a, h, k), answer_text=text, answer_latex="plain:" + text, answer_kind="curve")


def _wrong_text(a: Fraction, h: int, k: int):
    """``(text, value)`` for the parabola with these parameters, as vertex and a second point, or None."""
    if a == 0 or not _in((h, k)):
        return None
    pt = _second_point(a, h, k)
    if pt is None or pt == (h, k):
        return None
    return f"vertex {point_text((h, k))} point {point_text(pt)}", _answer(a, h, k)


def _dist(fields, candidates, n=4):
    cands = []
    for (a, h, k), tag in candidates:
        wrong = _wrong_text(Q(a), h, k)
        if wrong:
            cands.append((wrong[0], wrong[1], tag))
    return dist("curve", fields["answer"], fields["answer_text"], cands, n)


def _figure(alt="A blank coordinate grid with x and y axes, for drawing a parabola."):
    return plane(reply="curve", tools=["parabola_vertex_point"], alt=alt, snap=1, max_points=3)


def _common_wrong(a: Fraction, h: int, k: int):
    out = [
        ((a, -h, k), "w9-parabola-vertex-x-sign"),
        ((a, h, -k), "w9-parabola-vertex-y-sign"),
        ((-a, h, k), "w9-parabola-opens-wrong-way"),
        ((a, k, h), "w9-parabola-vertex-swapped"),
    ]
    if abs(a) != 1:
        out.append(((Q(1) if a > 0 else Q(-1), h, k), "w9-parabola-stretch-ignored"))
        out.append(((1 / a, h, k), "w9-parabola-stretch-inverted"))
    return out


# ---------------------------------------------------------------------------------------------- easy / medium


def _vertex_form(rng, stretches):
    for _ in attempts(500, "a parabola with a second lattice point on the grid"):
        a = Q(rng.choice(stretches))
        h = rng.choice([v for v in range(-4, 5) if v])
        k = rng.choice([v for v in range(-5, 6)])
        if _second_point(a, h, k) and a != 0 and abs(k) <= 5 and k != -h and k != h:
            break
    fields = _fields(a, h, k)
    eq = vertex_form_text(a, h, k)
    text = phrase(
        rng,
        (
            "Draw the graph of {eq}: place the vertex and one other point on the curve.",
            "The parabola {eq} is to be drawn. Place its vertex and one other point on it.",
        ),
        eq=eq,
    )
    d = _dist(fields, _common_wrong(a, h, k))
    enough(d, 3)
    pt = _second_point(a, h, k)
    working = [
        f"In y = a(x - h)² + k the vertex is (h, k). Here the vertex is {point_text((h, k))}.",
        f"The curve opens {'upwards' if a > 0 else 'downwards'}. Put x = {pt[0]}: y = {coef_text(a)}({pt[0] - h})²{'' if k == 0 else f' + {k}' if k > 0 else f' - {-k}'} = {pt[1]}, so {point_text(pt)} is on the curve.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _easy(rng):
    return _vertex_form(rng, (1, 1, -1))


def _medium(rng):
    return _vertex_form(rng, (2, 3, -2, -3, Q(1, 2), Q(-1, 2)))


# ---------------------------------------------------------------------------------------------- hard


def _hard_expanded(rng):
    for _ in attempts(500, "an expanded parabola with integer vertex on the grid"):
        a = rng.choice((1, -1, 2, -2, 1, -1))
        h = rng.choice([v for v in range(-4, 5) if v])
        k = rng.randint(-5, 5)
        b, c = -2 * a * h, a * h * h + k
        if _second_point(Q(a), h, k) and abs(c) <= 25 and abs(b) <= 20 and k != 0 and c != k:
            break
    for x in range(-4, 5):
        check(f_vertex(a, h, k, x) == a * x * x + b * x + c, "expanded and vertex forms disagree")
    fields = _fields(Q(a), h, k)
    eq = expanded_text(a, b, c)
    text = phrase(
        rng,
        (
            "Draw the graph of {eq}: place the vertex and one other point on the curve.",
            "The parabola {eq} is to be drawn. Place its vertex and one other point on it.",
        ),
        eq=eq,
    )
    cands = [
        ((a, -b, k), "w9-parabola-vertex-formula-no-half") if abs(b) <= HALF else None,
        ((a, Fraction(b, 2 * a), k), "w9-parabola-vertex-formula-sign"),
        ((a, Fraction(-b, 2), k), "w9-parabola-vertex-formula-no-a") if abs(a) != 1 else None,
        ((a, h, c), "w9-parabola-vertex-y-is-intercept"),
        ((a, -h, k), "w9-parabola-vertex-x-sign"),
        ((-a, h, k), "w9-parabola-opens-wrong-way"),
    ]
    cands = [c_ for c_ in cands if c_ and Fraction(c_[0][1]).denominator == 1]
    cands = [((c_[0][0], int(c_[0][1]), c_[0][2]), c_[1]) for c_ in cands]
    d = _dist(fields, cands)
    enough(d, 3)
    pt = _second_point(Q(a), h, k)
    working = [
        f"The axis of symmetry is x = -b / 2a = {-b} / {2 * a} = {h}.",
        f"Put x = {h}: y = {k}. The vertex is {point_text((h, k))}.",
        f"Put x = {pt[0]}: y = {pt[1]}, so {point_text(pt)} is on the curve.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _hard_factorised(rng):
    for _ in attempts(500, "a factorised parabola with integer vertex on the grid"):
        a = rng.choice((1, -1, 2, -2))
        p = rng.randint(-5, 3)
        q = p + rng.choice((2, 4))
        h = (p + q) // 2
        k = -a * (q - p) ** 2 // 4
        if _second_point(Q(a), h, k) and abs(k) <= 6 and p != q and h != 0 and p * q != 0:
            break
    for x in range(-6, 7):
        check(f_vertex(a, h, k, x) == a * (x - p) * (x - q), "factorised and vertex forms disagree")
    fields = _fields(Q(a), h, k)
    eq = factorised_text(a, p, q)
    text = phrase(
        rng,
        (
            "Draw the graph of {eq}: place the vertex and one other point on the curve.",
            "The parabola {eq} is to be drawn. Place its vertex and one other point on it.",
        ),
        eq=eq,
    )
    cands = [
        ((a, p, 0), "w9-parabola-vertex-x-is-root"),
        ((a, p + q, k), "w9-parabola-roots-sum-not-halved"),
        ((a, h, 0), "w9-parabola-vertex-on-axis"),
        ((a, -h, k), "w9-parabola-roots-sign"),
        ((-a, h, k), "w9-parabola-opens-wrong-way"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    pt = _second_point(Q(a), h, k)
    working = [
        f"The x-intercepts are x = {p} and x = {q}. The axis of symmetry is halfway between them: x = ({sum_text(p, q)}) / 2 = {h}.",
        f"Put x = {h}: y = {coef_text(Q(a))}({h - p})({h - q}) = {k}. The vertex is {point_text((h, k))}.",
        f"Put x = {pt[0]}: y = {pt[1]}, so {point_text(pt)} is on the curve.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _hard_roots_intercept(rng):
    for _ in attempts(1000, "roots and an intercept that give an integer vertex on the grid"):
        a = rng.choice((1, -1, 2, -2, Q(1, 2), Q(-1, 2)))
        p = rng.randint(-5, 2)
        q = p + rng.choice((2, 4, 6))
        if (p + q) % 2 or p * q == 0:
            continue
        h = (p + q) // 2
        k = -Q(a) * (q - p) ** 2 / 4
        c = Q(a) * p * q
        if (
            k.denominator == 1
            and c.denominator == 1
            and abs(c) <= 8
            and abs(k) <= 6
            and h != 0
            and _second_point(Q(a), h, int(k))
        ):
            break
    a, k, c = Q(a), int(k), int(c)
    for x in range(-6, 7):
        check(f_vertex(a, h, k, x) == a * (x - p) * (x - q), "factorised and vertex forms disagree")
    check(a * p * q == c, "the y-intercept is wrong")
    fields = _fields(a, h, k)
    text = (
        f"A parabola crosses the x-axis at x = {p} and x = {q}, and crosses the y-axis at y = {c}. "
        "Draw it: place the vertex and one other point on the curve."
    )
    cands = [
        ((a, p, 0), "w9-parabola-vertex-x-is-root"),
        ((a, p + q, k), "w9-parabola-roots-sum-not-halved"),
        ((a, h, 0), "w9-parabola-vertex-on-axis"),
        ((-a, h, k), "w9-parabola-opens-wrong-way"),
        ((Q(c) if c else Q(1), h, k), "w9-parabola-stretch-from-intercept"),
        ((a, -h, k), "w9-parabola-roots-sign"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    pt = _second_point(a, h, k)
    working = [
        f"The axis of symmetry is halfway between the roots: x = ({sum_text(p, q)}) / 2 = {h}.",
        f"y = a{_fac(p)}{_fac(q)}. Put x = 0: {c} = a({-p})({-q}) = {fmt_signed(p * q, 'a')}, so a = {frac_text(a)}.",
        f"Put x = {h}: y = {coef_text(a)}({h - p})({h - q}) = {k}. The vertex is {point_text((h, k))}; {point_text(pt)} is on the curve.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


_HARD = (_hard_expanded, _hard_factorised, _hard_roots_intercept)


@generator(
    TOPIC,
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_parabola_draw(rng, tier="easy"):
    """Draw a parabola from its vertex and a point; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=10, presets={"exam": dict(tier="exam")})
def exd_x_w9_parabola_draw(rng, tier="exam"):
    """Exam chain from an expanded equation: vertex x, vertex y, the drawing, the y-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(500, "an expanded parabola with integer vertex on the grid"):
        a = rng.choice((1, -1, 2, -2))
        h = rng.choice([v for v in range(-4, 5) if v])
        k = rng.choice([v for v in range(-5, 6) if v])
        b, c = -2 * a * h, a * h * h + k
        if _second_point(Q(a), h, k) and abs(c) <= 25 and c != k and abs(b) != abs(c) and h != k:
            break
    for x in range(-4, 5):
        check(f_vertex(a, h, k, x) == a * x * x + b * x + c, "expanded and vertex forms disagree")
    check(f_vertex(a, h, k, 0) == c, "the y-intercept is wrong")
    fields = _fields(Q(a), h, k)
    d_h = num_dist(
        h,
        [
            (-b, "w9-parabola-vertex-formula-no-half"),
            (Fraction(b, 2 * a), "w9-parabola-vertex-formula-sign"),
            (Fraction(-b, 2), "w9-parabola-vertex-formula-no-a"),
            (b, "w9-parabola-vertex-formula-no-half"),
        ],
    )
    d_k = num_dist(
        k,
        [
            (c, "w9-parabola-vertex-y-is-intercept"),
            (-k, "w9-parabola-vertex-y-sign"),
            (h, "w9-parabola-vertex-swapped"),
            (c - b, "w9-parabola-vertex-y-is-intercept"),
        ],
    )
    d_c = _dist(
        fields,
        [
            ((a, -h, k), "w9-parabola-vertex-x-sign"),
            ((a, h, c), "w9-parabola-vertex-y-is-intercept"),
            ((-a, h, k), "w9-parabola-opens-wrong-way"),
            ((a, k, h), "w9-parabola-vertex-swapped"),
            ((a, Fraction(-b, 1) if abs(b) <= 4 else h, k), "w9-parabola-vertex-formula-no-half"),
        ],
    )
    d_i = num_dist(
        c,
        [
            (b, "w9-parabola-y-intercept-coefficient"),
            (k, "w9-parabola-y-intercept-vertex-y"),
            (a, "w9-parabola-y-intercept-from-vertex-form"),
            (-c, "w9-parabola-y-intercept-wrong-sign"),
        ],
    )
    for dl in (d_h, d_k, d_c, d_i):
        enough(dl)
    eq = expanded_text(a, b, c)
    pt = _second_point(Q(a), h, k)
    return dict(
        text=f"A parabola has the equation {eq}.",
        figure=_figure(),
        parts=[
            part(
                "Find the x-coordinate of the vertex.",
                "number",
                num_fields(h),
                1,
                [f"x = -b / 2a = {-b} / {2 * a} = {h}."],
                d_h,
            ),
            part(
                "Find the y-coordinate of the vertex.",
                "number",
                num_fields(k),
                1,
                [f"Put x = {h} into the equation: y = {k}."],
                d_k,
            ),
            part(
                "Draw the parabola: place its vertex and one other point on the curve.",
                "curve",
                {x: fields[x] for x in ("answer", "answer_text", "answer_latex")},
                3,
                [f"The vertex is {point_text((h, k))} and {point_text(pt)} is on the curve."],
                d_c,
            ),
            part("Find the y-intercept.", "number", num_fields(c), 1, [f"Put x = 0: y = {c}."], d_i),
        ],
    )
