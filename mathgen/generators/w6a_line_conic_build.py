"""Generates `w6a_line_conic` questions (easy, medium, hard) and the exam chain `exd_x_w6a_line_conic`.

``line-meets-parabola-circle`` (``quadratics``, answer ``number``): a line against a parabola ``y = ax² + bx + d`` or a
circle ``x² + y² = r²``. Substituting the line gives a quadratic, and its discriminant says how many points the two
share; a zero discriminant is the tangent condition.

* easy: a line cuts a parabola at two points with whole-number x. The larger x, the y of the left-hand point, or the sum
  of the x-coordinates.
* medium: the intercept that makes a line of given gradient tangent to a parabola (discriminant zero); the larger
  x-coordinate where a line meets a circle; the x-coordinate of the midpoint of the chord a line cuts from a parabola
  when the two x-coordinates are irrational (so the sum of roots, not the roots, is what is needed).
* hard: the positive intercept (or gradient) that makes a line tangent to a circle, how many whole-number intercepts
  give a line with two intersection points, or the larger gradient of the two tangents to a parabola or circle from a
  point on the y-axis.
* exam (``exd_x_w6a_line_conic``): a line against a parabola or a circle: (a) the larger x of the intersections,
  (b) the intercept that makes a parallel line tangent, (c) the x of the point of contact, (d) a value that follows
  through from (b) or (c).

Each answer is derived by solving the substituted quadratic exactly, and re-checked by a SymPy solve of the pair of
equations (or, for a count, a Sturm count of the real roots of the substituted quadratic for every candidate).
"""

from __future__ import annotations

import functools
from fractions import Fraction
from math import gcd, isqrt

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w6a_line_conic"
EXAM_TOPIC = "exd_x_w6a_line_conic"

AREA = "line-and-conic"
MISCONCEPTIONS = [
    (
        "lcv-other-root",
        "Gives the x-coordinate of the other intersection point, the smaller where the larger was asked.",
        AREA,
    ),
    ("lcv-x-for-y", "Gives an x-coordinate where a y-coordinate was asked, or the reverse.", AREA),
    (
        "lcv-root-sign",
        "Reads a root of the substituted quadratic with the wrong sign, as x = -2 from the factor (x - 2).",
        AREA,
    ),
    ("lcv-product-for-sum", "Gives the product of the x-coordinates where their sum was asked.", AREA),
    ("lcv-difference-for-sum", "Gives the distance between the two x-coordinates where their sum was asked.", AREA),
    (
        "lcv-disc-sign",
        "Gets the sign of the discriminant condition wrong, so the intercept moves the wrong way from the vertex.",
        AREA,
    ),
    (
        "lcv-disc-drops-a",
        "Leaves the leading coefficient out of the discriminant, using b² - 4c in place of b² - 4ac.",
        AREA,
    ),
    (
        "lcv-slope-sign",
        "Subtracts the gradient from the x coefficient with the wrong sign when forming the substituted quadratic.",
        AREA,
    ),
    ("lcv-disc-factor", "Uses 2a in place of 4a in the zero-discriminant condition.", AREA),
    ("lcv-midpoint-is-vertex", "Gives the x-coordinate of the parabola's vertex as the midpoint of the chord.", AREA),
    ("lcv-sum-not-halved", "Gives the sum of the two x-coordinates where their mean (the midpoint) was asked.", AREA),
    (
        "lcv-tangent-radius-only",
        "Takes the intercept of a tangent to a circle to be the radius, ignoring the gradient.",
        AREA,
    ),
    (
        "lcv-tangent-gradient-unsquared",
        "Uses r(1 + m) in place of r times the square root of (1 + m²) for the tangent intercept.",
        AREA,
    ),
    (
        "lcv-tangent-unscaled",
        "Forgets to divide by the sum of the squares of the coefficients when finding the point of contact.",
        AREA,
    ),
    (
        "lcv-tangent-counted-as-two",
        "Counts the intercept that gives a tangent as a line with two intersection points.",
        AREA,
    ),
    (
        "lcv-no-intersection-counted",
        "Counts the intercepts that give no intersection instead of those that give two points.",
        AREA,
    ),
    (
        "lcv-count-misses-end",
        "Leaves out one end of the range, or counts both ends, when counting whole-number values.",
        AREA,
    ),
    ("lcv-gradient-inverted", "Gives the reciprocal of the tangent gradient, run over rise.", AREA),
    ("lcv-gradient-sign", "Gives the negative gradient, taking the smaller of the two tangent gradients.", AREA),
    (
        "lcv-gap-sign",
        "Adds the two intercepts instead of subtracting them to find the gap between parallel lines.",
        AREA,
    ),
    (
        "lcv-gap-from-intercept",
        "Measures the gap from the parabola's own y-intercept rather than from the tangent line.",
        AREA,
    ),
    (
        "lcv-line-value-in-parabola",
        "Substitutes the x-coordinate into the line's gradient instead of its equation, so the y-value is wrong.",
        AREA,
    ),
]


# -- shared pieces ------------------------------------------------------------------------------------------------------


def _fx(value) -> Fraction:
    return Fraction(value)


def _k_line(m: int) -> str:
    """The line y = mx + k as text, with a gradient of 1 or -1 written without the 1."""
    return "y = x + k" if m == 1 else "y = -x + k" if m == -1 else f"y = {m}x + k"


def _poly_text(a: int, b: int, d: int) -> str:
    return "y = " + join_terms([(a, "x²"), (b, "x"), (d, "")])


def _line_text(m: int, c: int) -> str:
    return "y = " + join_terms([(m, "x"), (c, "")])


def _general_line_text(p: int, q: int, c: int) -> str:
    """The line q y = p x + c, as the text a learner sees."""
    left = "y" if q == 1 else f"{q}y"
    return f"{left} = " + join_terms([(p, "x"), (c, "")])


def _circle_text(r: int) -> str:
    return f"x² + y² = {r * r}"


def _poly_rhs(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x²"), (b, "x"), (c, "")])


def _lin_rhs(m: int, c: int) -> str:
    return join_terms([(m, "x"), (c, "")])


def _frac_roots(a: int, b: int, c: int):
    """The real roots of a x² + b x + c = 0 as exact SymPy numbers (sorted)."""
    x = sympy.Symbol("x")
    return sorted(sympy.solve(a * x**2 + b * x + c, x), key=lambda v: float(v))


def _pick_parabola(rng, a_pool=(1, 1, 2, -1, -2, 3)):
    a = rng.choice(a_pool)
    b = rng.randint(-4, 4)
    d = rng.randint(-5, 5)
    return a, b, d


def _line_through_integer_points(rng, a, b, d):
    """(m, c, r1, r2): a line y = mx + c meeting y = ax² + bx + d at x = r1 < r2, both whole numbers."""
    for _ in attempts(500, "a line with two whole-number intersections"):
        r1, r2 = sorted(rng.sample(range(-5, 7), 2))
        m = b + a * (r1 + r2)
        c = d - a * r1 * r2
        if 0 < abs(m) <= 12 and abs(c) <= 20 and not (m == 0 and c == 0):
            return m, c, r1, r2
    raise AssertionError  # pragma: no cover


def _check_intersections(a, b, d, m, c, expected):
    """SymPy: solving the pair of equations gives exactly the expected x-values."""
    x, y = sympy.symbols("x y")
    sols = sympy.solve([sympy.Eq(y, a * x**2 + b * x + d), sympy.Eq(y, m * x + c)], [x, y], dict=True)
    got = sorted(float(s[x]) for s in sols)
    check(got == sorted(float(e) for e in expected), "SymPy's intersections differ")


@functools.cache
def _circle_cases():
    """(p, q, C, r, s2): lines q y = p x + C against x² + y² = r² meeting in rational points
    (the square of the half-discriminant s2 = (p²+q²) r² - C² is a perfect square), s2 >= 0."""
    out = []
    for q in (1, 2, 3, 4, 5, 12):
        for p in (-12, -8, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 8, 12, 15):
            if gcd(abs(p), q) != 1:
                continue
            for r in range(3, 14):
                for cc in range(-30, 31):
                    if cc == 0:
                        continue
                    s2 = (p * p + q * q) * r * r - cc * cc
                    if s2 > 0 and isqrt(s2) ** 2 == s2:
                        out.append((p, q, cc, r, isqrt(s2)))
    return tuple(out)


_PYTH = {(3, 4), (4, 3), (5, 12), (12, 5), (8, 15), (15, 8)}


def _circle_case(rng, pythagorean=False):
    pool = _circle_cases()
    if pythagorean:
        pool = [t for t in pool if (abs(t[0]), t[1]) in _PYTH]
    return rng.choice(pool)


def _circle_xs(p, q, cc, r, s):
    """Both x-coordinates where q y = p x + C meets x² + y² = r², exactly."""
    den = p * p + q * q
    return sorted([Fraction(-p * cc - q * s, den), Fraction(-p * cc + q * s, den)])


def _check_circle(p, q, cc, r, xs):
    x, y = sympy.symbols("x y")
    sols = sympy.solve([sympy.Eq(x**2 + y**2, r * r), sympy.Eq(q * y, p * x + cc)], [x, y], dict=True)
    got = sorted(float(s[x]) for s in sols)
    check(got == sorted(float(v) for v in xs), "SymPy's circle intersections differ")


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy(rng):
    a, b, d = _pick_parabola(rng)
    m, c, r1, r2 = _line_through_integer_points(rng, a, b, d)
    _check_intersections(a, b, d, m, c, [r1, r2])
    y1 = m * r1 + c
    kind = rng.choice(("larger", "y_of_smaller", "sum"))
    pair = f"the parabola {_poly_text(a, b, d)} and the line {_line_text(m, c)}"
    if kind == "larger":
        value = r2
        cands = [
            (r1, "lcv-other-root"),
            (m * r2 + c, "lcv-x-for-y"),
            (-r2, "lcv-root-sign"),
            (r1 + r2, "lcv-product-for-sum"),
        ]
        stem = phrase(
            rng,
            (
                f"The line {_line_text(m, c)} and the parabola {_poly_text(a, b, d)} meet at two points. Find the larger x-coordinate of the two points.",
                f"Find the larger x-coordinate where {pair} meet.",
                f"The graphs of {pair} intersect twice. What is the x-coordinate of the right-hand point of intersection?",
            ),
        )
        working = [
            f"Equate: {_poly_rhs(a, b, d)} = {_lin_rhs(m, c)}, a quadratic with roots x = {r1} and x = {r2}.",
            f"The larger is x = {r2}.",
        ]
    elif kind == "y_of_smaller":
        value = y1
        cands = [
            (r1, "lcv-x-for-y"),
            (m * r2 + c, "lcv-other-root"),
            (-y1, "lcv-root-sign"),
            (m * r1, "lcv-line-value-in-parabola"),
        ]
        stem = phrase(
            rng,
            (
                f"The line {_line_text(m, c)} and the parabola {_poly_text(a, b, d)} meet at two points. Find the y-coordinate of the point with the smaller x-coordinate.",
                f"{pair.capitalize()} intersect at two points. What is the y-coordinate of the left-hand point?",
                f"Find the y-coordinate of the left-hand point where {pair} meet.",
            ),
        )
        working = [
            f"The x-coordinates are {r1} and {r2}; the left-hand point has x = {r1}.",
            f"y = {m}({r1}) {'+' if c >= 0 else '-'} {abs(c)} = {y1}.",
        ]
    else:
        value = r1 + r2
        cands = [
            (r1 * r2, "lcv-product-for-sum"),
            (r2 - r1, "lcv-difference-for-sum"),
            (-(r1 + r2), "lcv-root-sign"),
            (r2, "lcv-other-root"),
        ]
        stem = phrase(
            rng,
            (
                f"The line {_line_text(m, c)} and the parabola {_poly_text(a, b, d)} meet at two points. Find the sum of their x-coordinates.",
                f"Find the sum of the x-coordinates of the two points where {pair} meet.",
                f"{pair.capitalize()} intersect twice. Add the x-coordinates of the two points of intersection.",
            ),
        )
        working = [f"The x-coordinates are {r1} and {r2}.", f"Their sum is {r1 + r2}."]
    dist = num_dist(value, cands)
    enough(dist)
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(value))


# -- medium --------------------------------------------------------------------------------------------------------------


def _tangent_k(a, b, d, m) -> Fraction:
    return _fx(d) - Fraction((b - m) ** 2, 4 * a)


def _medium_tangent(rng):
    for _ in attempts(500, "a tangent intercept that is not the vertex value"):
        a, b, d = _pick_parabola(rng, (1, 2, -1, -2, 3, 4))
        m = rng.choice([v for v in range(-6, 7) if v != 0 and v != b])
        k = _tangent_k(a, b, d, m)
        if abs(k) <= 40:
            break
    # Second route: with this k the substituted quadratic has a repeated root, and with k + 1 or k - 1 it does not.
    x = sympy.Symbol("x")
    kk = sympy.Rational(k.numerator, k.denominator)
    zero = sympy.discriminant(a * x**2 + b * x + d - (m * x + kk), x)
    check(zero == 0, "the substituted quadratic has no repeated root")
    check(sympy.discriminant(a * x**2 + b * x + d - (m * x + kk + 1), x) != 0, "k + 1 also gives a repeated root")
    cands = [
        (_fx(d) + Fraction((b - m) ** 2, 4 * a), "lcv-disc-sign"),
        (_fx(d) - Fraction((b - m) ** 2, 4), "lcv-disc-drops-a"),
        (_fx(d) - Fraction((b + m) ** 2, 4 * a), "lcv-slope-sign"),
        (_fx(d) - Fraction((b - m) ** 2, 2 * a), "lcv-disc-factor"),
    ]
    dist = num_dist(k, cands)
    enough(dist)
    line = _k_line(m)
    stem = phrase(
        rng,
        (
            f"The line {line} is a tangent to the parabola {_poly_text(a, b, d)}. Find k.",
            f"Find the value of k for which the line {line} touches the parabola {_poly_text(a, b, d)} at exactly one point.",
            f"For which value of k does the line {line} meet the parabola {_poly_text(a, b, d)} at exactly one point?",
        ),
    )
    working = [
        f"Equate: {_poly_rhs(a, b, d)} = {_lin_rhs(m, 0)} + k, so {a}x² + ({b - m})x + ({d} - k) = 0.",
        f"Tangency needs the discriminant to be zero: ({b - m})² - 4({a})({d} - k) = 0.",
        f"So k = {k}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(k))


def _medium_circle(rng):
    p, q, cc, r, s = _circle_case(rng)
    xs = _circle_xs(p, q, cc, r, s)
    _check_circle(p, q, cc, r, xs)
    larger, smaller = xs[1], xs[0]
    den = p * p + q * q
    cands = [
        (smaller, "lcv-other-root"),
        (Fraction(-p * cc, den), "lcv-midpoint-is-vertex"),
        (Fraction(p * cc, den) + Fraction(q * s, den), "lcv-root-sign"),
        (Fraction(-p * cc + q * s, 1), "lcv-tangent-unscaled"),
    ]
    dist = num_dist(larger, cands)
    enough(dist)
    line = _general_line_text(p, q, cc)
    stem = phrase(
        rng,
        (
            f"The line {line} and the circle {_circle_text(r)} meet at two points. Find the larger x-coordinate of the two points.",
            f"Find the x-coordinate of the right-hand point where the line {line} meets the circle {_circle_text(r)}.",
            f"The circle {_circle_text(r)} and the line {line} intersect twice. What is the greater x-coordinate of a point of intersection?",
        ),
    )
    working = [
        f"Substituting y = ({_lin_rhs(p, cc)})/{q} gives {_poly_rhs(den, 2 * p * cc, cc * cc - q * q * r * r)} = 0.",
        f"The roots are x = {smaller} and x = {larger}; the larger is {larger}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(larger))


def _medium_midpoint(rng):
    """A chord whose end points are irrational, so the answer needs the sum of the roots."""
    for _ in attempts(1000, "a chord with irrational ends"):
        a, b, d = _pick_parabola(rng, (1, 1, 2, -1, -2))
        m = rng.randint(-6, 6)
        c = rng.randint(-8, 8)
        # a x² + (b - m) x + (d - c) = 0
        disc = (b - m) ** 2 - 4 * a * (d - c)
        if disc > 0 and isqrt(disc) ** 2 != disc and m != 0:
            break
    mid = Fraction(m - b, 2 * a)
    # Second route: SymPy's roots, averaged.
    x = sympy.Symbol("x")
    roots = sympy.solve(a * x**2 + (b - m) * x + (d - c), x)
    check(len(roots) == 2 and abs(float(sum(roots) / 2) - float(mid)) < 1e-9, "SymPy's mean of the roots differs")
    vertex = Fraction(-b, 2 * a)
    cands = [
        (vertex, "lcv-midpoint-is-vertex"),
        (2 * mid, "lcv-sum-not-halved"),
        (-mid, "lcv-root-sign"),
        (Fraction(m, 2 * a), "lcv-slope-sign"),
    ]
    dist = num_dist(mid, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            f"The line {_line_text(m, c)} and the parabola {_poly_text(a, b, d)} intersect at the points A and B. Find the x-coordinate of the midpoint of AB.",
            f"A and B are the two points of intersection of the parabola {_poly_text(a, b, d)} and the line {_line_text(m, c)}. What is the x-coordinate of the midpoint of the chord AB?",
        ),
    )
    working = [
        f"Equating gives {a}x² + ({b - m})x + ({d - c}) = 0, whose roots are the x-coordinates of A and B.",
        f"The roots are not whole numbers, but their sum is {-Fraction(b - m, a)}, so the midpoint has x = {mid}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(mid))


# -- hard ----------------------------------------------------------------------------------------------------------------


def _hard_circle_tangent(rng):
    """Intercept (positive) of a tangent to x² + y² = r² of the form q y = p x + C, with p²+q² a perfect square."""
    p, q = rng.choice(sorted(_PYTH))
    p = p * rng.choice((1, -1))
    r = rng.randint(2, 12)
    root = isqrt(p * p + q * q)
    cc = r * root
    # Second route: with this C the substituted quadratic has a repeated root.
    x = sympy.Symbol("x")
    y = (p * x + cc) / sympy.Integer(q)
    check(sympy.discriminant(sympy.expand(x**2 + y**2 - r * r), x) == 0, "not tangent")
    cands = [
        (r, "lcv-tangent-radius-only"),
        (Fraction(r * (abs(p) + q), 1), "lcv-tangent-gradient-unsquared"),
        (r * (p * p + q * q), "lcv-tangent-gradient-unsquared"),
        (r * q, "lcv-tangent-radius-only"),
        (cc + r, "lcv-tangent-gradient-unsquared"),
    ]
    dist = num_dist(cc, cands)
    enough(dist)
    left = "y" if q == 1 else f"{q}y"
    line = f"{left} = {fmt_x(p)} + C"
    stem = phrase(
        rng,
        (
            f"The line {line} is a tangent to the circle {_circle_text(r)}, and C > 0. Find C.",
            f"Find the positive value of C for which the line {line} touches the circle {_circle_text(r)} at exactly one point.",
            f"For the line {line} to be a tangent to {_circle_text(r)}, C must take two values. What is the positive one?",
        ),
    )
    working = [
        f"Substitute y = ({fmt_x(p)} + C)/{q} into x² + y² = {r * r}.",
        f"The discriminant of the quadratic in x is zero when C² = {r * r}({p * p} + {q * q}), so C = {cc}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(cc))


def fmt_x(p: int) -> str:
    return "x" if p == 1 else "-x" if p == -1 else f"{p}x"


def _hard_count(rng):
    circle = rng.random() < 0.5
    if circle:
        r = rng.randint(2, 8)
        m = rng.choice((1, 2, -1, -2, 3))
        bound2 = r * r * (1 + m * m)
        two = lambda k: k * k < bound2  # noqa: E731
        tangent = lambda k: k * k == bound2  # noqa: E731
        edge = isqrt(bound2)
        lo, hi = -(edge + rng.randint(2, 4)), edge + rng.randint(2, 4)
        target = f"the circle {_circle_text(r)}"
    else:
        a, b, d = _pick_parabola(rng, (1, 1, 2, -1, -2))
        m = rng.choice([v for v in range(-5, 6) if v != 0])
        two = lambda k: (b - m) ** 2 - 4 * a * (d - k) > 0  # noqa: E731
        tangent = lambda k: (b - m) ** 2 - 4 * a * (d - k) == 0  # noqa: E731
        boundary = Fraction(d) - Fraction((b - m) ** 2, 4 * a)
        lo, hi = int(boundary) - rng.randint(3, 7), int(boundary) + rng.randint(3, 7)
        target = f"the parabola {_poly_text(a, b, d)}"
    line = _k_line(m)
    ks = list(range(lo, hi + 1))
    two_ks = [k for k in ks if two(k)]
    tangent_ks = [k for k in ks if tangent(k)]
    none_ks = [k for k in ks if not two(k) and not tangent(k)]
    count = len(two_ks)
    check(0 < count < len(ks), "all or none of the intercepts give two points")
    # Second route: a Sturm count of the real roots of the substituted quadratic for every k.
    x = sympy.Symbol("x")
    for k in ks:
        if circle:
            poly = sympy.Poly((1 + m * m) * x**2 + 2 * m * k * x + k * k - r * r, x)
        else:
            poly = sympy.Poly(a * x**2 + (b - m) * x + (d - k), x)
        roots = poly.count_roots()
        check((roots == 2 and poly.discriminant() > 0) == (k in two_ks), "Sturm count differs")
    cands = [
        (count + len(tangent_ks), "lcv-tangent-counted-as-two"),
        (len(none_ks), "lcv-no-intersection-counted"),
        (count - 1, "lcv-count-misses-end"),
        (count + 1, "lcv-count-misses-end"),
    ]
    dist = num_dist(count, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            f"The line {line} and {target} are drawn together. For how many whole-number values of k from {lo} to {hi} inclusive does the line meet it at two different points?",
            f"How many integers k with {lo} <= k <= {hi} make the line {line} cross {target} twice?",
            f"Consider the line {line} and {target}. Count the whole numbers k, from {lo} to {hi} inclusive, for which the line has two intersection points with it.",
        ),
    )
    working = [
        "Substitute the line into the equation and write the discriminant of the resulting quadratic in x.",
        f"Two intersection points need a positive discriminant; that holds for {count} of the {len(ks)} whole numbers from {lo} to {hi}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(count))


@functools.cache
def _tangent_triples():
    """(c, r, s) with c² - r² = s² and 0 < r < c: the lines through (0, c) tangent to x² + y² = r² have gradient ±s/r."""
    return tuple(
        (c, r, isqrt(c * c - r * r))
        for c in range(4, 26)
        for r in range(2, c)
        if isqrt(c * c - r * r) ** 2 == c * c - r * r
    )


def _hard_two_tangents(rng):
    if rng.random() < 0.5:
        c, r, s = rng.choice(_tangent_triples())
        value = Fraction(s, r)
        x = sympy.Symbol("x")
        m = sympy.Symbol("m")
        sols = sympy.solve(sympy.discriminant(sympy.expand(x**2 + (m * x + c) ** 2 - r * r), x), m)
        check(
            sorted(float(v) for v in sols) == sorted([-float(value), float(value)]), "SymPy's tangent gradients differ"
        )
        cands = [
            (Fraction(r, s), "lcv-gradient-inverted"),
            (-value, "lcv-gradient-sign"),
            (Fraction(s, c), "lcv-tangent-radius-only"),
            (Fraction(c, r), "lcv-tangent-gradient-unsquared"),
        ]
        dist = num_dist(value, cands)
        enough(dist)
        stem = phrase(
            rng,
            (
                f"Two tangents to the circle {_circle_text(r)} pass through the point (0, {c}). Find the larger of their gradients.",
                f"From the point (0, {c}), two tangent lines are drawn to {_circle_text(r)}. What is the gradient of the steeper one?",
            ),
        )
        working = [
            f"A line y = mx {'+' if c >= 0 else '-'} {abs(c)} is tangent when {c}² = {r}²(1 + m²).",
            f"So m² = {c * c - r * r}/{r * r} and m = ±{value}; the larger gradient is {value}.",
        ]
        return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(value))
    for _ in attempts(500, "tangents from a point above a parabola"):
        a, b, d = _pick_parabola(rng, (1, 1, 1, 2, -1))
        t = rng.randint(1, 4)
        # a (d - c) = t²  ->  c = d - t²/a  (whole number needs a | t²)
        if (t * t) % abs(a) != 0:
            continue
        c = d - (t * t) // a
        if abs(c) <= 20 and (t > 0):
            break
    value = b + 2 * t
    m = sympy.Symbol("m")
    x = sympy.Symbol("x")
    sols = sympy.solve(sympy.discriminant(a * x**2 + b * x + d - (m * x + c), x), m)
    check(sorted(float(v) for v in sols) == sorted([b - 2 * t, b + 2 * t]), "SymPy's tangent gradients differ")
    cands = [
        (b - 2 * t, "lcv-gradient-sign"),
        (b + t, "lcv-disc-factor"),
        (2 * t - b, "lcv-slope-sign"),
        (-(b + 2 * t), "lcv-root-sign"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            f"Two tangents to the parabola {_poly_text(a, b, d)} pass through the point (0, {c}). Find the larger of their gradients.",
            f"From the point (0, {c}), two lines are drawn tangent to {_poly_text(a, b, d)}. What is the greater gradient?",
        ),
    )
    working = [
        f"A line y = mx {'+' if c >= 0 else '-'} {abs(c)} is tangent when {a}x² + ({b} - m)x + ({d - c}) has a zero discriminant: ({b} - m)² = {4 * a * (d - c)}.",
        f"So m = {b} ± {2 * t}, and the larger gradient is {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="number", **num_fields(value))


_TIERS = {
    "easy": (_easy,),
    "medium": (_medium_tangent, _medium_circle, _medium_midpoint),
    "hard": (_hard_circle_tangent, _hard_count, _hard_two_tangents),
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6a_line_conic(rng, tier="easy"):
    """A line against a parabola or circle: intersections, tangent conditions and counts; see the module docstring."""
    pool = _TIERS.get(tier)
    check(pool is not None, f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------------


def _exam_parabola(rng):
    a, b, d = _pick_parabola(rng, (1, 1, 2, -1, -2))
    m, c, r1, r2 = _line_through_integer_points(rng, a, b, d)
    _check_intersections(a, b, d, m, c, [r1, r2])
    k = _tangent_k(a, b, d, m)
    xt = Fraction(m - b, 2 * a)
    # Second route for the contact point: the tangent line meets the parabola only at x = xt.
    x = sympy.Symbol("x")
    kk = sympy.Rational(k.numerator, k.denominator)
    check(
        sympy.solve(a * x**2 + b * x + d - (m * x + kk), x) == [sympy.Rational(xt.numerator, xt.denominator)],
        "contact point differs",
    )
    gap = abs(_fx(c) - k)
    stem = f"The line L: {_line_text(m, c)} and the parabola P: {_poly_text(a, b, d)} intersect at two points."
    a_dist = num_dist(
        r2,
        [(r1, "lcv-other-root"), (m * r2 + c, "lcv-x-for-y"), (-r2, "lcv-root-sign"), (r1 + r2, "lcv-product-for-sum")],
    )
    b_dist = num_dist(
        k,
        [
            (_fx(d) + Fraction((b - m) ** 2, 4 * a), "lcv-disc-sign"),
            (_fx(d) - Fraction((b - m) ** 2, 4), "lcv-disc-drops-a"),
            (_fx(d) - Fraction((b + m) ** 2, 4 * a), "lcv-slope-sign"),
            (_fx(d) - Fraction((b - m) ** 2, 2 * a), "lcv-disc-factor"),
        ],
        3,
    )
    c_dist = num_dist(
        xt,
        [
            (Fraction(-b, 2 * a), "lcv-midpoint-is-vertex"),
            (Fraction(b - m, 2 * a), "lcv-root-sign"),
            (Fraction(m - b, a), "lcv-sum-not-halved"),
            (Fraction(m, 2 * a), "lcv-slope-sign"),
        ],
        3,
    )
    d_dist = num_dist(
        gap,
        [
            (abs(_fx(c) + k), "lcv-gap-sign"),
            (abs(_fx(c) - d), "lcv-gap-from-intercept"),
            (abs(_fx(d) - k), "lcv-gap-from-intercept"),
        ],
        3,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        enough(dist)
    return dict(
        text=stem,
        parts=[
            part(
                "Find the larger x-coordinate of the two points where L meets P.",
                "number",
                num_fields(r2),
                1,
                [f"The intersections are at x = {r1} and x = {r2}."],
                a_dist,
            ),
            part(
                f"The line {_k_line(m)} is a tangent to P. Find k.",
                "number",
                num_fields(k),
                2,
                [f"A zero discriminant gives k = {k}."],
                b_dist,
            ),
            part(
                "Find the x-coordinate of the point where that tangent touches P.",
                "number",
                num_fields(xt),
                2,
                [f"The repeated root is x = {xt}."],
                c_dist,
            ),
            part(
                "Find the vertical distance between L and that tangent.",
                "fraction",
                num_fields(gap),
                1,
                [f"The lines are parallel, so the distance is |{c} - ({k})| = {gap}."],
                d_dist,
                follow_through=f"Abs({c} - part_b)",
                sensible_range=(0, None),
            ),
        ],
    )


def _exam_circle(rng):
    p, q, cc, r, s = _circle_case(rng, pythagorean=True)
    xs = _circle_xs(p, q, cc, r, s)
    _check_circle(p, q, cc, r, xs)
    den = p * p + q * q
    root = isqrt(den)
    tangent_c = r * root
    xt = Fraction(-p * tangent_c, den)
    yt = Fraction(p, q) * xt + Fraction(tangent_c, q)
    # Second route: the point of contact lies on the circle and on the tangent line.
    check(xt * xt + yt * yt == r * r and q * yt == p * xt + tangent_c, "contact point is off the circle or the line")
    line = _general_line_text(p, q, cc)
    left = "y" if q == 1 else f"{q}y"
    tline = f"{left} = {fmt_x(p)} + C"
    stem = f"The line L: {line} and the circle S: {_circle_text(r)} intersect at two points."
    a_dist = num_dist(
        xs[1],
        [
            (xs[0], "lcv-other-root"),
            (Fraction(-p * cc, den), "lcv-midpoint-is-vertex"),
            (Fraction(p * cc + q * s, den), "lcv-root-sign"),
            (Fraction(-p * cc + q * s, 1), "lcv-tangent-unscaled"),
        ],
        3,
    )
    b_dist = num_dist(
        tangent_c,
        [
            (r, "lcv-tangent-radius-only"),
            (r * (abs(p) + q), "lcv-tangent-gradient-unsquared"),
            (r * den, "lcv-tangent-gradient-unsquared"),
            (r * q, "lcv-tangent-radius-only"),
        ],
        3,
    )
    c_dist = num_dist(
        xt,
        [
            (Fraction(-p * tangent_c, 1), "lcv-tangent-unscaled"),
            (Fraction(p * tangent_c, den), "lcv-root-sign"),
            (Fraction(-tangent_c, den), "lcv-slope-sign"),
            (Fraction(-q * tangent_c, den), "lcv-x-for-y"),
        ],
        3,
    )
    d_dist = num_dist(
        yt,
        [
            (xt, "lcv-x-for-y"),
            (-yt, "lcv-root-sign"),
            ((p * xt) / q, "lcv-line-value-in-parabola"),
            (Fraction(tangent_c, q), "lcv-line-value-in-parabola"),
        ],
        3,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        enough(dist)
    return dict(
        text=stem,
        parts=[
            part(
                "Find the larger x-coordinate of the two points where L meets S.",
                "number",
                num_fields(xs[1]),
                2,
                [f"The intersections have x = {xs[0]} and x = {xs[1]}."],
                a_dist,
            ),
            part(
                f"The line {tline} is a tangent to S, with C > 0. Find C.",
                "number",
                num_fields(tangent_c),
                2,
                [f"C² = {r * r}({p * p} + {q * q}), so C = {tangent_c}."],
                b_dist,
            ),
            part(
                "Find the x-coordinate of the point where that tangent touches S.",
                "fraction",
                num_fields(xt),
                2,
                [f"The repeated root of the substituted quadratic is x = {xt}."],
                c_dist,
                follow_through=f"{-p}*part_b/{den}",
            ),
            part(
                "Find the y-coordinate of that point of contact.",
                "fraction",
                num_fields(yt),
                1,
                [f"From the tangent line, y = ({p}({xt}) + {tangent_c})/{q} = {yt}."],
                d_dist,
                follow_through=f"({p}*part_c + part_b)/{q}",
            ),
        ],
    )


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w6a_line_conic(rng, tier="exam"):
    """Exam chain: a line against a parabola or a circle, then the parallel tangent and its point of contact."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice((_exam_parabola, _exam_circle)))
