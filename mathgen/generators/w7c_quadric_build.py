"""Generates `w7c_quadric` questions (easy, medium, hard) and the exam chain `exd_x_w7c_quadric`.

``uc2-quadric-surfaces-level-sets`` (``multivariable-and-vector-calculus``): name a quadric surface from its equation,
describe a level set f(x, y, z) = c, and read the curve in which a plane cuts a quadric. Surfaces are built backwards
from a named type and its semi-axes, so the key is known by construction; each is then re-derived by an independent
classifier (`classify`) that works only from the expanded polynomial: its Hessian and gradient at the origin, the centre
(if there is one), and the exact inertia of the Hessian. Every cross-section is re-derived from the substituted polynomial
by the determinant test for a conic (`conic_type`).

* easy (multiple choice): the surface named by an axis-aligned equation with every type present (ellipsoid, the two
  hyperboloids, cone, the paraboloids, the cylinders); the level set f = c of a quadratic form, where the sign of c
  changes the answer and a definite form gives a single point or the empty set; the curve in which a coordinate plane
  or a plane parallel to one cuts a surface.
* medium: complete the square to name a shifted quadric (multiple choice) or to find a coordinate of its centre
  (number); a trace that is degenerate (a single point, no points, a pair of lines); the semi-axis of the ellipse
  in which a plane z = k cuts an ellipsoid or a hyperboloid.
* hard: a plane z = mx + c through the cone or a hyperboloid of one sheet, named by comparing the slope with the
  generators; the radius of the circle in which a plane cuts z = x² + y²; the intercept of the level surface through a
  given point; a quadric with an xy term (named from the sign of the 2×2 determinant, not from the diagonal), or a
  paraboloid with an xy term (named from b² − 4ac).
* exam (``exd_x_w7c_quadric``): a quadric x²/a² + y²/b² ± z²/c² = 1 and a plane z = k. (a) the surface, (b) and (c) the
  semi-axes of the cross-section, (d) its area as a multiple of π (follows through from (b) and (c)).
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import gcd, isqrt, lcm

import sympy
from sympy import Rational as Q

from ..answers import Distractor
from ..core import check, generator, paren_negative, phrase
from ._ucv1_linalg import rref
from ._w6c_kit import Redraw, choice_part, math_of, redrawn, value_fields, value_part

X, Y, Z = sympy.symbols("x y z")
VARS = (X, Y, Z)

ELL = "ellipsoid"
H1 = "hyperboloid of one sheet"
H2 = "hyperboloid of two sheets"
CONE = "cone"
EP = "elliptic paraboloid"
HP = "hyperbolic paraboloid"
EC = "elliptic cylinder"
HC = "hyperbolic cylinder"
PC = "parabolic cylinder"
POINT = "single point"
EMPTY = "empty set"
LINES = "pair of intersecting lines"
CIRCLE = "circle"
ELLIPSE = "ellipse"
HYPERBOLA = "hyperbola"
PARABOLA = "parabola"


# -- text -------------------------------------------------------------------------------------------------------


def show(expr) -> str:
    """A SymPy expression as plain text with a true minus sign: ``x²/4 − y²/9``."""
    text = math_of(expr)
    text = str(text)
    if text.startswith("-"):
        text = "−" + text[1:]
    return text.replace("(-", "(−")


def equation(lhs, rhs) -> str:
    return f"{show(lhs)} = {show(rhs)}"


# -- an independent classifier ---------------------------------------------------------------------------------


def _fr(value) -> Fraction:
    value = sympy.nsimplify(value) if not isinstance(value, (int, sympy.Rational)) else sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def _parts(f):
    """(H, g, d): the Hessian, the gradient at the origin and the constant term of a quadratic polynomial."""
    poly = sympy.Poly(sympy.expand(f), *VARS)
    check(poly.total_degree() <= 2, "not a quadric")
    coeff = poly.coeff_monomial
    h = [
        [_fr(2 * coeff(v * v)) if i == j else _fr(coeff(VARS[i] * VARS[j])) for j, v in enumerate(VARS)]
        for i, v in enumerate(VARS)
    ]
    g = [_fr(coeff(v)) for v in VARS]
    return h, g, _fr(coeff(1))


def inertia(h) -> tuple[int, int]:
    """(positive, negative) counts of a symmetric Fraction matrix, by exact congruence diagonalisation."""
    a = [row[:] for row in h]
    pos = neg = 0
    for _ in range(len(h)):
        n = len(a)
        i = next((i for i in range(n) if a[i][i] != 0), None)
        if i is None:
            pair = next(((r, c) for r in range(n) for c in range(r + 1, n) if a[r][c] != 0), None)
            if pair is None:
                break
            r, c = pair
            # e_r <- e_r + e_c turns a zero diagonal entry into 2 a_rc.
            a[r] = [x + y for x, y in zip(a[r], a[c])]
            for row in a:
                row[r] += row[c]
            i = r
        pivot = a[i][i]
        if pivot > 0:
            pos += 1
        else:
            neg += 1
        for j in range(n):
            if j != i and a[j][i] != 0:
                factor = a[j][i] / pivot
                a[j] = [x - factor * y for x, y in zip(a[j], a[i])]
        a = [[v for c, v in enumerate(row) if c != i] for r, row in enumerate(a) if r != i]
    return pos, neg


def classify(f) -> str:
    """The name of the quadric surface f = 0, from its polynomial alone."""
    h, g, d = _parts(f)
    pos, neg = inertia(h)
    rank = pos + neg
    augmented = [row + [-g[i]] for i, row in enumerate(h)]
    reduced, pivots = rref(augmented)
    if 3 in pivots:  # the system H c = -g has no solution: there is no centre
        if rank == 2:
            return EP if pos * neg == 0 else HP
        check(rank == 1, "unexpected rank for a quadric with no centre")
        return PC
    centre = [Fraction(0)] * 3
    for row, col in zip(reduced, pivots):
        centre[col] = row[3]
    value = Fraction(1, 2) * sum(centre[i] * h[i][j] * centre[j] for i in range(3) for j in range(3))
    value += sum(g[i] * centre[i] for i in range(3)) + d
    return central_name(pos, neg, -value, rank)


def central_name(pos: int, neg: int, v, rank: int = 3) -> str:
    """The surface ``q(y) = v`` for a quadratic form of inertia (pos, neg): v is the constant on the right."""
    if v < 0:
        pos, neg, v = neg, pos, -v
    if rank == 3:
        if v == 0:
            return CONE if pos * neg else POINT
        return {(3, 0): ELL, (2, 1): H1, (1, 2): H2, (0, 3): EMPTY}[(pos, neg)]
    check(rank == 2, "unsupported rank")
    if v == 0:
        return LINES if pos * neg else "line"
    return {(2, 0): EC, (1, 1): HC, (0, 2): EMPTY}[(pos, neg)]


def central_from_signs(signs, v) -> str:
    """Name of ``Σ s_i x_i² = v`` from the signs alone (the reading a learner does)."""
    pos = sum(1 for s in signs if s > 0)
    return central_name(pos, len(signs) - pos, v)


def conic_type(expr, s, t) -> str:
    """The curve ``expr = 0`` in the (s, t) plane, by the determinant test."""
    poly = sympy.Poly(sympy.expand(expr), s, t)
    coeff = poly.coeff_monomial
    a, b, c = coeff(s * s), coeff(s * t), coeff(t * t)
    d, e, f = coeff(s), coeff(t), coeff(1)
    check(a != 0 or b != 0 or c != 0, "not a conic")
    big = sympy.Matrix([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]]).det()
    delta = b * b - 4 * a * c
    if big != 0:
        if delta < 0:
            if big * (a + c) > 0:
                return EMPTY
            return CIRCLE if b == 0 and a == c else ELLIPSE
        return PARABOLA if delta == 0 else HYPERBOLA
    if delta < 0:
        return POINT
    return LINES if delta > 0 else "parallel lines"


# -- standard surfaces ------------------------------------------------------------------------------------------


class Surface:
    """A named quadric: ``f`` is the polynomial of f = 0, ``text`` the equation, ``u, v, w`` the variables in their
    roles (w is the odd one: the axis of a paraboloid, the minus-signed or positive-signed variable of a hyperboloid)."""

    def __init__(self, name, f, text, roles, semi=None):
        self.name, self.f, self.text, self.roles, self.semi = name, f, text, roles, semi


def _std(rng, name, *, semi=None, roles=None, flip=None, cleared=None) -> Surface:
    """A surface of type ``name`` in standard position. ``semi`` = (a, b, c) for the roles (u, v, w)."""
    u, v, w = roles or rng.sample(VARS, 3)
    if semi is None:
        semi = tuple(rng.randint(2, 7) for _ in range(3))
        if name in (EC, HC):
            semi = (semi[0], semi[1] if semi[1] != semi[0] else semi[0] + 1, semi[2])
        if name == ELL and len(set(semi)) == 1:
            semi = (semi[0], semi[1] + 1, semi[2])
    a, b, c = semi
    flip = rng.random() < 0.3 if flip is None else flip
    signs = {
        ELL: (1, 1, 1),
        H1: (1, 1, -1),
        H2: (-1, -1, 1),
        CONE: (1, 1, -1),
        EC: (1, 1, 0),
        HC: (1, -1, 0),
    }.get(name)
    if signs is not None:
        rhs = 0 if name == CONE else 1
        denominators = [a * a, b * b, c * c]
        if cleared is None:
            cleared = lcm(*[d for d, s in zip(denominators, signs) if s]) <= 144 and rng.random() < 0.5
        scale = lcm(*[d for d, s in zip(denominators, signs) if s]) if cleared else 1
        lhs = sum(Q(s * scale, d) * var**2 for s, d, var in zip(signs, denominators, (u, v, w)) if s)
        right = Q(rhs * scale)
        if flip:
            lhs, right = -lhs, -right
        text = equation(lhs, right)
        return Surface(name, lhs - right, text, (u, v, w), semi)
    # paraboloids and the parabolic cylinder: w = ...
    sign = (1, 1) if name == EP else (1, -1) if name == HP else (1, 0)
    denominators = (a * a, b * b)
    if cleared is None:
        cleared = lcm(*[d for d, s in zip(denominators, sign) if s]) <= 144 and rng.random() < 0.5
    scale = lcm(*[d for d, s in zip(denominators, sign) if s]) if cleared else 1
    rhs = sum(Q(s * scale, d) * var**2 for s, d, var in zip(sign, denominators, (u, v)) if s)
    left = scale * w
    if flip:
        rhs = -rhs
    return Surface(name, left - rhs, equation(left, rhs), (u, v, w), semi)


# -- multiple choice --------------------------------------------------------------------------------------------

_CONFUSIONS = {
    ELL: [("quad-ellipsoid-mixed-up", n) for n in (H1, EP, CONE, H2, EC)],
    H1: [
        ("quad-sheets-swapped", H2),
        ("quad-zero-rhs-ignored", CONE),
        ("quad-elliptical-sections-means-ellipsoid", ELL),
        ("quad-hyperbolic-sections-means-saddle", HP),
    ],
    H2: [
        ("quad-sheets-swapped", H1),
        ("quad-zero-rhs-ignored", CONE),
        ("quad-elliptical-sections-means-ellipsoid", ELL),
        ("quad-hyperbolic-sections-means-saddle", HP),
    ],
    CONE: [
        ("quad-zero-rhs-ignored", H1),
        ("quad-zero-rhs-ignored", H2),
        ("quad-elliptical-sections-means-ellipsoid", ELL),
        ("quad-squared-axis-confused", EP),
    ],
    EP: [
        ("quad-paraboloid-sign-mixed", HP),
        ("quad-squared-axis-confused", CONE),
        ("quad-elliptical-sections-means-ellipsoid", ELL),
        ("quad-squared-axis-confused", H2),
    ],
    HP: [
        ("quad-paraboloid-sign-mixed", EP),
        ("quad-minus-sign-means-hyperboloid", H1),
        ("quad-squared-axis-confused", CONE),
        ("quad-squared-axis-confused", H2),
    ],
    EC: [
        ("quad-missing-variable-ignored", ELL),
        ("quad-cylinder-kind-mixed", HC),
        ("quad-cylinder-kind-mixed", PC),
        ("quad-missing-variable-ignored", EP),
    ],
    HC: [
        ("quad-missing-variable-ignored", H1),
        ("quad-cylinder-kind-mixed", EC),
        ("quad-cylinder-kind-mixed", PC),
        ("quad-missing-variable-ignored", HP),
    ],
    PC: [
        ("quad-missing-variable-ignored", EP),
        ("quad-cylinder-kind-mixed", HC),
        ("quad-cylinder-kind-mixed", EC),
        ("quad-missing-variable-ignored", HP),
    ],
    POINT: [("quad-level-zero-empty", EMPTY), ("quad-zero-rhs-ignored", CONE), ("quad-level-sign-ignored", ELL)],
    EMPTY: [
        ("quad-level-sign-ignored", ELL),
        ("quad-level-negative-point", POINT),
        ("quad-zero-rhs-ignored", CONE),
        ("quad-level-sign-ignored", H1),
    ],
}

_CONIC_WRONG = {
    CIRCLE: [
        ("quad-trace-closed-vs-open", HYPERBOLA),
        ("quad-trace-closed-vs-open", PARABOLA),
        ("quad-trace-degenerate-missed", POINT),
        ("quad-trace-degenerate-missed", LINES),
    ],
    ELLIPSE: [
        ("quad-trace-closed-vs-open", HYPERBOLA),
        ("quad-trace-closed-vs-open", PARABOLA),
        ("quad-trace-circle-vs-ellipse", CIRCLE),
        ("quad-trace-degenerate-missed", POINT),
    ],
    HYPERBOLA: [
        ("quad-trace-closed-vs-open", ELLIPSE),
        ("quad-trace-hyperbola-vs-parabola", PARABOLA),
        ("quad-trace-degenerate-missed", LINES),
        ("quad-trace-closed-vs-open", CIRCLE),
    ],
    PARABOLA: [
        ("quad-trace-closed-vs-open", ELLIPSE),
        ("quad-trace-hyperbola-vs-parabola", HYPERBOLA),
        ("quad-trace-closed-vs-open", CIRCLE),
        ("quad-trace-degenerate-missed", LINES),
    ],
    LINES: [
        ("quad-trace-degenerate-missed", HYPERBOLA),
        ("quad-trace-degenerate-missed", ELLIPSE),
        ("quad-trace-degenerate-missed", PARABOLA),
        ("quad-trace-point-vs-empty", POINT),
    ],
    POINT: [
        ("quad-trace-point-vs-empty", EMPTY),
        ("quad-trace-degenerate-missed", ELLIPSE),
        ("quad-trace-degenerate-missed", LINES),
        ("quad-trace-degenerate-missed", CIRCLE),
    ],
    EMPTY: [
        ("quad-trace-point-vs-empty", POINT),
        ("quad-trace-degenerate-missed", ELLIPSE),
        ("quad-trace-degenerate-missed", HYPERBOLA),
        ("quad-trace-degenerate-missed", PARABOLA),
    ],
}


def _options(rng, key, candidates, *, forbid=(), extra=(), n=4, minimum=3):
    """Up to ``n`` distinct wrong options (name, tag) from ``extra`` (priority) then a shuffled ``candidates``."""
    pool = list(extra) + rng.sample(candidates, len(candidates))
    wrong, seen = [], {key, *forbid}
    for tag, name in pool:
        if name in seen:
            continue
        seen.add(name)
        wrong.append((tag, name))
        if len(wrong) == n:
            break
    if len(wrong) < minimum:
        raise Redraw(f"only {len(wrong)} wrong options for {key}")
    return wrong


def _choice(text, key, wrong, working) -> dict:
    return dict(
        text=text,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        working=working,
        distractors=[Distractor(name, tag, name) for tag, name in wrong],
    )


def _conic_wrong(rng, key, *, n=4, minimum=3):
    forbid = {ELLIPSE} if key == CIRCLE else set()
    return _options(rng, key, _CONIC_WRONG[key], forbid=forbid, n=n, minimum=minimum)


def _confusion_wrong(rng, key, *, extra=(), n=4, minimum=3):
    return _options(rng, key, _CONFUSIONS[key], extra=extra, n=n, minimum=minimum)


# -- planes and traces ------------------------------------------------------------------------------------------


def _trace(f, var, k):
    """(expr, s, t, label): the cross-section f = 0 with ``var`` = k, as a conic in the other two variables."""
    s, t = [v for v in VARS if v != var]
    expr = sympy.expand(f.subs(var, k))
    return expr, s, t, conic_type(expr, s, t)


# -- easy -------------------------------------------------------------------------------------------------------

_NAME_STEMS = (
    "Which quadric surface is given by {eq}?",
    "Name the quadric surface {eq}.",
    "What type of surface is {eq}?",
)
_EASY_TYPES = (ELL, H1, H2, CONE, EP, HP, EC, HC, PC)


@redrawn
def _easy_name(rng):
    name = rng.choice(_EASY_TYPES)
    s = _std(rng, name)
    check(classify(s.f) == name, f"the classifier disagrees on {s.text}")
    return _choice(
        phrase(rng, _NAME_STEMS, eq=s.text),
        name,
        _confusion_wrong(rng, name),
        _name_working(s),
    )


def _name_working(s: Surface):
    lines = {
        ELL: "Every squared term is positive and the right side is a positive constant: an ellipsoid.",
        H1: "One squared term has the opposite sign to the other two and the right side is not zero, so each cut by a plane at the minus-signed variable is an ellipse and the surface is in one piece: one sheet.",
        H2: "Two squared terms have the opposite sign to the third and the right side is not zero, so the surface has two separate pieces: two sheets.",
        CONE: "The squared terms have mixed signs and the right side is 0: a cone.",
        EP: "One variable appears to the first power and the other two are squared with the same sign: an elliptic paraboloid.",
        HP: "One variable appears to the first power and the other two are squared with opposite signs: a hyperbolic paraboloid.",
        EC: "One variable is missing, and the other two are squared with the same sign: an elliptic cylinder.",
        HC: "One variable is missing, and the other two are squared with opposite signs: a hyperbolic cylinder.",
        PC: "Two variables are missing from the squares: one is squared and the other is to the first power, a parabolic cylinder.",
    }
    return [lines[s.name]]


_LEVEL_STEMS = (
    "Let f(x, y, z) = {f}. What is the level set f(x, y, z) = {c}?",
    "What is the level set f(x, y, z) = {c} of f(x, y, z) = {f}?",
    "For f(x, y, z) = {f}, describe the set of points where f = {c}.",
)

_LEVEL_SIGNS = ([1, 1, 1], [-1, -1, -1], [1, 1, -1], [1, -1, -1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1])


@redrawn
def _easy_level(rng):
    signs = list(rng.choice(_LEVEL_SIGNS))
    mags = [rng.randint(1, 4) for _ in range(3)]
    if len(set(signs)) == 1 and len(set(mags)) == 1:
        mags[rng.randrange(3)] += 1  # not a sphere
    coeffs = [s * m for s, m in zip(signs, mags)]
    f = sum(c * v**2 for c, v in zip(coeffs, VARS))
    c = rng.choice([-16, -9, -4, -1, 0, 0, 1, 4, 9, 16])
    key = classify(f - c)
    check(key == central_from_signs(coeffs, c), "the sign reading and the classifier differ")
    if c != 0:
        extra = [
            ("quad-level-sign-flipped", central_from_signs(coeffs, -c)),
            ("quad-zero-rhs-ignored", central_from_signs(coeffs, 0)),
            ("quad-level-sign-ignored", central_from_signs(coeffs, abs(c))),
        ]
    else:
        extra = [
            ("quad-zero-rhs-ignored", central_from_signs(coeffs, 1)),
            ("quad-zero-rhs-ignored", central_from_signs(coeffs, -1)),
        ]
        if key == POINT:
            extra.append(("quad-level-zero-empty", EMPTY))
    wrong = _options(rng, key, _CONFUSIONS[key], extra=extra)
    working = [
        f"The signs of the squared terms are {_sign_text(coeffs)}.",
        {
            ELL: "All three are positive and c > 0, so f = c is an ellipsoid.",
            POINT: "All three have one sign and c = 0, so only the origin satisfies f = 0.",
            EMPTY: "A sum of squares with one sign cannot take a value of the opposite sign, so there are no points.",
            CONE: "The signs are mixed and c = 0, so f = 0 is a cone.",
            H1: "The signs are mixed and c has the sign that appears twice, so the surface is in one piece: a hyperboloid of one sheet.",
            H2: "The signs are mixed and c has the sign that appears once, so the surface has two sheets.",
        }[key],
    ]
    return _choice(phrase(rng, _LEVEL_STEMS, f=show(f), c=show(sympy.Integer(c))), key, wrong, working)


def _sign_text(coeffs):
    return ", ".join("positive" if c > 0 else "negative" for c in coeffs)


_TRACE_STEMS = (
    "The surface {eq} is cut by the plane {plane}. What curve is the cross-section?",
    "What kind of curve is the trace of the surface {eq} in the plane {plane}?",
    "Describe the intersection of the surface {eq} with the plane {plane}.",
)


@redrawn
def _easy_trace(rng):
    name = rng.choice([ELL, H1, H2, EP, HP, CONE])
    equal = rng.random() < 0.2
    s = _std(rng, name)
    if equal and name in (ELL, H1, EP, CONE):
        a, _b, c = s.semi
        s = _std(rng, name, semi=(a, a, c if c != a or name != ELL else a + 1), roles=s.roles)
    var = rng.choice(VARS)
    k = _plane_value(rng, s, var)
    return _trace_question(rng, s, var, k, allow=(CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA))


def _plane_value(rng, s: Surface, var):
    """A value of ``var`` for which the trace is a proper conic (easy), found by testing the substituted polynomial."""
    for k in rng.sample([0, 0, 1, -1, 2, -2, 3, -3], 8):
        _expr, _s, _t, label = _trace(s.f, var, k)
        if label in (CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA):
            return k
    raise Redraw("no proper conic for this plane")


def _trace_question(rng, s: Surface, var, k, *, allow=None, minimum=3):
    expr, a, b, label = _trace(s.f, var, k)
    check(allow is None or label in allow, f"trace {label} not allowed")
    wrong = _conic_wrong(rng, label, minimum=minimum)
    plane = f"{var} = {k}"
    working = [
        f"Put {plane} in the equation: {show(expr)} = 0 in the {a}{b}-plane.",
        _conic_reason(label),
    ]
    return _choice(phrase(rng, _TRACE_STEMS, eq=s.text, plane=plane.replace("-", "−")), label, wrong, working)


def _conic_reason(label):
    return {
        CIRCLE: "Both squared terms have the same sign and the same coefficient, so the curve is a circle.",
        ELLIPSE: "Both squared terms have the same sign and different coefficients, with a positive constant on the other side: an ellipse.",
        HYPERBOLA: "The squared terms have opposite signs and the constant is not zero: a hyperbola.",
        PARABOLA: "One variable is squared and the other is to the first power: a parabola.",
        LINES: "The squared terms have opposite signs and the constant is zero: two lines crossing at a point.",
        POINT: "A sum of squares equal to zero: only one point.",
        EMPTY: "A sum of squares cannot equal a negative number: no points.",
    }[label]


# -- medium: complete the square -------------------------------------------------------------------------------


_CENTRED_TYPES = (ELL, H1, H2, CONE)


def _minus(left, right) -> str:
    """``left − right``, written ``left + |right|`` when right is negative."""
    return f"{left} − {right}" if right >= 0 else f"{left} + {-right}"


def _plus(left, right) -> str:
    """``left + right``, written ``left − |right|`` when right is negative."""
    return f"{left} + {right}" if right >= 0 else f"{left} − {-right}"


def _shifted(rng, name):
    """A central quadric of type ``name`` with integer coefficients, shifted to an integer centre, as (alphas, centre, R, f,
    text). Σ α_i (x_i − h_i)² = R."""
    for _ in range(60):
        mags = [rng.randint(1, 4) for _ in range(3)]
        signs = {ELL: (1, 1, 1), H1: (1, 1, -1), H2: (1, -1, -1), CONE: (1, 1, -1)}[name]
        order = rng.sample(range(3), 3)
        alphas = [mags[i] * signs[order[i]] for i in range(3)]
        if name == ELL and len(set(alphas)) == 1:
            continue
        centre = [rng.randint(-4, 4) for _ in range(3)]
        if sum(1 for h in centre if h) < 2:
            continue
        r = 0 if name == CONE else rng.randint(2, 18)
        if r and rng.random() < 0.5:
            alphas, r = [-a for a in alphas], -r
        f = sum(a * (v - h) ** 2 for a, v, h in zip(alphas, VARS, centre)) - r
        expanded = sympy.expand(f)
        const = expanded.subs({X: 0, Y: 0, Z: 0})
        lhs = expanded - const
        return alphas, centre, r, f, equation(lhs, -const), -const
    raise Redraw("no shifted quadric")


_SQUARE_STEMS = (
    "Complete the square to identify the surface {eq}.",
    "Which quadric surface is {eq}?",
    "Rewrite {eq} by completing the square. What surface is it?",
)


@redrawn
def _medium_name(rng):
    if rng.random() < 0.25:
        return _medium_paraboloid(rng)
    name = rng.choice(_CENTRED_TYPES)
    alphas, centre, r, f, text, naive_rhs = _shifted(rng, name)
    key = classify(f)
    check(key == central_from_signs(alphas, r), "the sign reading and the classifier differ")
    extra = []
    naive = central_from_signs(alphas, naive_rhs)
    if naive != key:
        extra.append(("quad-forgot-completing-constant", naive))
    working = [
        f"Completing the square in each variable gives a sum of terms aᵢ(xᵢ − hᵢ)² equal to {r}, with a = {tuple(alphas)} and centre {tuple(centre)}.",
        {
            ELL: "All three signs agree with the right side: an ellipsoid.",
            H1: "The signs are mixed, the right side is not zero and has the sign that appears twice: one sheet.",
            H2: "The signs are mixed, the right side is not zero and has the sign that appears once: two sheets.",
            CONE: "The signs are mixed and the right side is 0: a cone.",
        }[key],
    ]
    return _choice(
        phrase(rng, _SQUARE_STEMS, eq=text), key, _confusion_wrong(rng, key, extra=extra, minimum=4), working
    )


def _medium_paraboloid(rng):
    name = rng.choice([EP, HP])
    a1 = rng.randint(1, 3)
    a2 = rng.randint(1, 3) * (1 if name == EP else -1)
    h, k, l = (rng.randint(-4, 4) for _ in range(3))
    if rng.random() < 0.4:
        a1, a2 = -a1, -a2
    w = rng.choice(VARS)
    u, v = [x for x in VARS if x != w]
    rhs = sympy.expand(a1 * (u - h) ** 2 + a2 * (v - k) ** 2 + l)
    f = w - rhs
    key = classify(f)
    check(key == name, "the classifier disagrees")
    text = equation(w, rhs)
    working = [
        f"Completing the square in {u} and {v} gives {_minus(w, l)} = {a1}({_minus(u, h)})² + ({a2})({_minus(v, k)})².",
        "One variable is to the first power and two are squared, "
        + (
            "with the same sign: an elliptic paraboloid."
            if name == EP
            else "with opposite signs: a hyperbolic paraboloid."
        ),
    ]
    return _choice(phrase(rng, _SQUARE_STEMS, eq=text), key, _confusion_wrong(rng, key, minimum=4), working)


_CENTRE_STEMS = (
    "The quadric surface {eq} has a centre. Find its {var}-coordinate.",
    "Complete the square in {eq} to find the {var}-coordinate of the centre of the surface.",
    "The centre of the surface {eq} is (h, k, l). Find the {var}-coordinate.",
)


@redrawn
def _medium_centre(rng):
    name = rng.choice(_CENTRED_TYPES)
    alphas, centre, r, f, text, _naive = _shifted(rng, name)
    options = [i for i in range(3) if centre[i] != 0 and abs(alphas[i]) >= 2]
    if not options:
        raise Redraw("no coordinate with a coefficient")
    i = rng.choice(options)
    var = "xyz"[i]
    # independent check: the gradient of f vanishes at the centre
    point = {v: h for v, h in zip(VARS, centre)}
    check(all(sympy.diff(f, v).subs(point) == 0 for v in VARS), "gradient does not vanish at the centre")
    a, h = alphas[i], centre[i]
    b = -2 * a * h  # the coefficient of the linear term in `var`
    wrong = [
        ("quad-centre-sign-error", -h),
        ("quad-centre-coefficient-not-divided", Fraction(b, -2)),
        ("quad-centre-not-halved", Fraction(b, -a)),
        ("quad-centre-linear-coefficient-read", b),
        ("quad-centre-halved-twice", Fraction(b, -4 * a)),
    ]
    working = [
        f"Completing the square in {var}: the {var} terms become {a}({_minus(var, h)})² plus a constant.",
        f"The centre has {var}-coordinate {h}.",
    ]
    return dict(text=phrase(rng, _CENTRE_STEMS, eq=text, var=var), working=working, **value_fields(h, wrong, minimum=4))


# -- medium / exam: semi-axes of a horizontal trace -------------------------------------------------------------


@cache
def _triples():
    out = []
    for r in range(5, 26):
        for p in range(3, r):
            q = isqrt(r * r - p * p)
            if q >= 3 and q * q + p * p == r * r:
                out.append((p, q, r))
    return tuple(out)


def _axis_pool(kind):
    """(P, k, m) with the cut w = k of a surface whose odd variable has parameter P: m = P·sqrt(factor) is an integer."""
    out = []
    for p, q, r in _triples():
        if kind == ELL:
            out.append((r, p, q))  # P² − k² = m²
        elif kind == H1:
            out.append((p, q, r))  # P² + k² = m²
        else:
            out.append((p, r, q))  # k² − P² = m²
    return out


def _semi_axes(rng, kind):
    """(P, k, m, A, B): axes A, B (for u, v) with A m / P and B m / P whole, A ≠ B."""
    for _ in range(80):
        P, k, m = rng.choice(_axis_pool(kind))
        good = [a for a in range(2, 21) if (a * m) % P == 0]
        if len(good) >= 2:
            A, B = rng.sample(good, 2)
            return P, k, m, A, B
    raise Redraw("no semi-axes")


def _kind_surface(rng, kind, P, A, B):
    """Surface of type ``kind`` with semi-axis parameters A, B (roles u, v) and P (role w), in a random orientation."""
    return _std(rng, kind, semi=(A, B, P), flip=False, cleared=False)


_AXIS_STEMS = (
    "The surface {eq} is cut by the plane {plane} in an ellipse. Find the semi-axis of that ellipse in the {var} direction.",
    "The cross-section of {eq} by the plane {plane} is an ellipse. How long is its semi-axis along the {var}-axis?",
    "Find the length of the semi-axis, parallel to the {var}-axis, of the ellipse in which {eq} is cut by the plane {plane}.",
)


def _axis_candidates(kind, A, B, P, k, m):
    value = Fraction(A * m, P)
    shift = {ELL: P - k, H1: P + k, H2: k - P}[kind]
    out = [
        ("quad-trace-axis-ignores-cut", A),
        ("quad-denominator-as-semi-axis", A * A),
        ("quad-trace-axis-not-rooted", value * value),
        ("quad-trace-axis-factor-unrooted", Fraction(A * m * m, P * P)),
        ("quad-trace-axis-k-not-squared", Fraction(A * shift, P)),
        ("quad-trace-axis-wrong-direction", Fraction(B * m, P)),
    ]
    other = {ELL: P * P + k * k, H1: P * P - k * k, H2: P * P + k * k}[kind]
    if other > 0 and isqrt(other) ** 2 == other:
        out.append(("quad-trace-axis-sign-flipped", Fraction(A * isqrt(other), P)))
    return out


def _check_axis(s: Surface, var, k, ask, value):
    """Independent check: substitute the plane and read the semi-axis from the conic's coefficients."""
    expr, a, b, label = _trace(s.f, var, k)
    check(label in (ELLIPSE, CIRCLE), f"the cut is not an ellipse: {label}")
    poly = sympy.Poly(expr, a, b)
    coeff = {a: poly.coeff_monomial(a * a), b: poly.coeff_monomial(b * b)}
    const = poly.coeff_monomial(1)
    check(
        poly.coeff_monomial(a) == 0 and poly.coeff_monomial(b) == 0 and poly.coeff_monomial(a * b) == 0,
        "trace is not centred",
    )
    squared = -const / coeff[ask]
    check(sympy.sqrt(squared) == value, "the substituted equation gives a different semi-axis")


@redrawn
def _medium_axis(rng):
    kind = rng.choice([ELL, ELL, H1, H2])
    P, k, m, A, B = _semi_axes(rng, kind)
    s = _kind_surface(rng, kind, P, A, B)
    u, v, w = s.roles
    sign_k = rng.choice([1, -1])
    ask_a = rng.random() < 0.5
    ask, var = (u, "first") if ask_a else (v, "second")
    value = Fraction((A if ask_a else B) * m, P)
    check(value.denominator == 1, "semi-axis not whole")
    _check_axis(s, w, sign_k * k, ask, int(value))
    wrong = _axis_candidates(kind, A if ask_a else B, B if ask_a else A, P, k, m)
    plane = f"{w} = {sign_k * k}".replace("-", "−")
    working = [
        f"Putting {plane} in the equation gives {show(sympy.expand(s.f.subs(w, sign_k * k)))} = 0.",
        f"Dividing by the constant, the {ask}² term is over {int(value) ** 2}, so the semi-axis in the {ask} direction is {int(value)}.",
    ]
    return dict(
        text=phrase(rng, _AXIS_STEMS, eq=s.text, plane=plane, var=ask),
        working=working,
        **value_fields(value, wrong, minimum=4),
    )


# -- hard -------------------------------------------------------------------------------------------------------

_OBLIQUE_STEMS = (
    "The surface {eq} is cut by the plane {plane}. What curve is the cross-section?",
    "What curve is the intersection of {eq} with the plane {plane}?",
    "Describe the curve of intersection of the surface {eq} and the plane {plane}.",
)


def _plane_text(p, w, q, x, r) -> str:
    """The plane p·w = q·x + r as the learner reads it: ``2z = −3x + 4``."""
    left = (str(p) if p != 1 else "") + str(w)
    slope = ("" if abs(q) == 1 else str(abs(q))) + str(x)
    right = ("−" if q < 0 else "") + slope + (" + " if r > 0 else " − ") + str(abs(r))
    return f"{left} = {right}"


@redrawn
def _hard_oblique(rng):
    cls = rng.choice(["ellipse", "parabola", "hyperbola"])
    p = rng.randint(2, 5) if cls != "parabola" else rng.randint(1, 3)
    if cls == "parabola":
        q = p * rng.choice([1, -1])
    elif cls == "ellipse":
        q = rng.choice([x for x in range(-p + 1, p) if x != 0])
    else:
        q = rng.choice([x for x in range(-8, 9) if abs(x) > p])
    r = rng.choice([x for x in range(-6, 7) if x != 0])
    u, v, w = rng.sample(VARS, 3)
    d = rng.choice([0, 0, 1, 4, 9])
    surface = u**2 + v**2 - w**2 - d
    check(classify(surface) == (CONE if d == 0 else H1), "the classifier disagrees")
    slope_var = rng.choice([u, v])
    # Independent check: substitute the plane and apply the determinant test for a conic.
    expr = sympy.expand(surface.subs(w, (q * slope_var + r) / sympy.Integer(p)) * p**2)
    label = conic_type(expr, u, v)
    expected = {"ellipse": ELLIPSE, "parabola": PARABOLA, "hyperbola": HYPERBOLA}[cls]
    check(label == expected, f"the slope rule says {expected} but the determinant test says {label}")
    wrong = _conic_wrong(rng, expected, minimum=4)
    eq = f"{show(u**2 + v**2 - w**2)} = {d}"
    reading = {
        "ellipse": "the plane is flatter than the generators of the surface (|slope| < 1), so it cuts every generator: an ellipse.",
        "parabola": "the plane is parallel to a generator of the surface (|slope| = 1): a parabola.",
        "hyperbola": "the plane is steeper than the generators of the surface (|slope| > 1), so it cuts both halves of the asymptotic cone: a hyperbola.",
    }[cls]
    working = [
        f"The generators of the cone {show(u**2 + v**2 - w**2)} = 0 have slope 1, so compare the slope |{q}/{p}| of the plane in the {slope_var} direction with 1.",
        f"The slope is {abs(Fraction(q, p))}: {reading}",
    ]
    return _choice(
        phrase(rng, _OBLIQUE_STEMS, eq=eq, plane=_plane_text(p, w, q, slope_var, r)), expected, wrong, working
    )


_RADIUS_STEMS = (
    "The paraboloid {para} is cut by the plane {plane} in a circle. Find the radius of the circle.",
    "The paraboloid {para} and the plane {plane} intersect in a circle. What is its radius?",
    "Find the radius of the circle in which the paraboloid {para} is cut by the plane {plane}.",
)


@redrawn
def _hard_radius(rng):
    u, v, w = rng.sample(VARS, 3)
    a = rng.choice([x for x in range(-6, 7, 2) if x != 0])
    b = rng.choice([x for x in range(-6, 7, 2) if x != 0])
    r = rng.randint(2, 9)
    c = r * r - (a * a + b * b) // 4
    if c == 0:
        raise Redraw("constant is zero")
    para = u**2 + v**2
    plane_rhs = a * u + b * v + c
    expr = sympy.expand(para - plane_rhs)
    check(conic_type(expr, u, v) == CIRCLE, "not a circle")
    poly = sympy.Poly(expr, u, v)
    d, e, f0 = poly.coeff_monomial(u), poly.coeff_monomial(v), poly.coeff_monomial(1)
    r2 = (d / 2) ** 2 + (e / 2) ** 2 - f0
    check(r2 == r * r, "the completed square gives a different radius")
    wrong = [
        ("quad-radius-not-rooted", r * r),
        ("quad-radius-from-constant", c if c > 0 else None),
        ("quad-radius-slope-not-halved", _int_root(c + a * a + b * b)),
        ("quad-radius-completing-sign", _int_root(c - (a * a + b * b) // 4)),
        ("quad-radius-centre-as-radius", abs(a) // 2),
        ("quad-radius-centre-as-radius", abs(b) // 2),
        ("quad-radius-constant-only", _int_root(c)),
    ]
    plane = f"{w} = {show(plane_rhs)}"
    working = [
        f"Substitute: {u}² + {v}² = {show(plane_rhs)}.",
        f"Complete the square: ({_minus(u, a // 2)})² + ({_minus(v, b // 2)})² = {_plus(c, (a * a + b * b) // 4)} = {r * r}.",
        f"The radius is √{r * r} = {r}.",
    ]
    return dict(
        text=phrase(rng, _RADIUS_STEMS, plane=plane, para=f"{w} = {show(para)}"),
        working=working,
        **value_fields(r, wrong, minimum=4),
    )


def _int_root(n):
    return isqrt(n) if n > 0 and isqrt(n) ** 2 == n else None


_INTERCEPT_STEMS = (
    "Let f(x, y, z) = {f}. The level surface f = c passes through P = ({p}) and meets the {axis}-axis at {at}. Find s.",
    "The level surface of f(x, y, z) = {f} through the point P = ({p}) crosses the {axis}-axis at {at}. Find s.",
)


@redrawn
def _hard_intercept(rng):
    coeffs = [rng.choice([1, 2, 3, 4, 5]) for _ in range(3)]
    if rng.random() < 0.5:
        coeffs[rng.randrange(3)] *= -1
    if len(set(coeffs)) == 1:
        raise Redraw("equal coefficients")
    i = rng.randrange(3)
    s_val = rng.randint(2, 6)
    c = coeffs[i] * s_val * s_val
    j, k = [x for x in range(3) if x != i]
    solutions = []
    for pj in range(1, 8):
        for pk in range(1, 8):
            rest = c - coeffs[j] * pj * pj - coeffs[k] * pk * pk
            if rest % coeffs[i] == 0 and rest // coeffs[i] > 0:
                pi = isqrt(rest // coeffs[i])
                if pi * pi == rest // coeffs[i] and pi >= 1 and pi != s_val:
                    solutions.append((pi, pj, pk))
    if not solutions:
        raise Redraw("no lattice point")
    pi, pj, pk = rng.choice(solutions)
    point = [0, 0, 0]
    point[i], point[j], point[k] = pi, pj, pk
    signs = [rng.choice([1, -1]) for _ in range(3)]
    point = [p * s for p, s in zip(point, signs)]
    f = sum(co * v**2 for co, v in zip(coeffs, VARS))
    check(f.subs(dict(zip(VARS, point))) == c, "the point is not on the level surface")
    roots = sympy.solve(coeffs[i] * sympy.Symbol("t", real=True) ** 2 - c, sympy.Symbol("t", real=True))
    check(sorted(roots) == [-s_val, s_val], "solving along the axis gives a different intercept")
    wrong = [
        ("quad-intercept-coordinate-of-point", abs(point[i])),
        ("quad-intercept-level-value", c),
        ("quad-intercept-coefficient-forgotten", _int_root(c)),
        ("quad-intercept-root-forgotten", Fraction(c, coeffs[i])),
        ("quad-intercept-coefficient-inverted", _int_root(c * coeffs[i])),
    ]
    axis = "xyz"[i]
    at = "(" + ", ".join("±s" if t == i else "0" for t in range(3)) + ")"
    working = [
        f"f(P) = {show(sympy.Integer(c))}, so the level surface is {show(f)} = {c}.",
        f"On the {axis}-axis only {axis} is non-zero: {coeffs[i]}{axis}² = {c}, so {axis}² = {c // coeffs[i]} and s = {s_val}.",
    ]
    return dict(
        text=phrase(
            rng, _INTERCEPT_STEMS, f=show(f), p=", ".join(str(p).replace("-", "−") for p in point), axis=axis, at=at
        ),
        working=working,
        **value_fields(s_val, wrong, minimum=4),
    )


_CROSS_STEMS = (
    "Which quadric surface is {eq}?",
    "Name the quadric surface {eq}.",
    "What type of surface is {eq}?",
)


@redrawn
def _hard_cross(rng):
    if rng.random() < 0.35:
        return _hard_cross_paraboloid(rng)
    a, c2 = rng.randint(1, 3), rng.randint(1, 3)
    b = isqrt(a * c2) + rng.randint(1, 2)  # b² > ac, so the 2×2 block is indefinite although a and c are both positive
    sign = rng.choice([1, -1])
    a, c2, b = sign * a, sign * c2, sign * rng.choice([1, -1]) * b
    gamma = rng.choice([1, 2, 3]) * rng.choice([1, -1])
    rhs = rng.choice([0, 1, 2, 3, 4, 6])
    u, v, w = rng.sample(VARS, 3)
    f = a * u**2 + 2 * b * u * v + c2 * v**2 + gamma * w**2 - rhs
    key = classify(f)
    det = a * c2 - b * b
    check(det < 0, "block should be indefinite")
    pos = 1 + (1 if gamma > 0 else 0)
    check(key == central_name(pos, 3 - pos, rhs), "the determinant reading differs from the classifier")
    naive = central_from_signs([a, c2, gamma], rhs)
    if naive == key:
        raise Redraw("the cross term does not change the answer")
    text = equation(f + rhs, sympy.Integer(rhs))
    working = [
        f"The {u}{v} term matters: the block of {u}² and {v}² has determinant {paren_negative(a)}·{paren_negative(c2)} − {paren_negative(b)}² = {det}, which is negative, so it has one positive and one negative direction.",
        f"Together with the {w}² term and the right side {rhs}, that gives a {key}.",
    ]
    return _choice(
        phrase(rng, _CROSS_STEMS, eq=text),
        key,
        _confusion_wrong(rng, key, extra=[("quad-cross-term-ignored", naive)], minimum=4),
        working,
    )


def _hard_cross_paraboloid(rng):
    """w = a u² + b u v + c v² with a and c the same sign: the diagonal says elliptic, b² − 4ac says otherwise."""
    want = rng.choice([HP, HP, PC])
    a, c2 = rng.randint(1, 4), rng.randint(1, 4)
    if want == PC:
        a = c2 = rng.randint(1, 4)
        b = 2 * a * rng.choice([1, -1])
        if rng.random() < 0.4:
            a, c2 = (1, 4) if rng.random() < 0.5 else (4, 1)
            b = 4 * rng.choice([1, -1])
    else:
        b = rng.choice([x for x in range(-9, 10) if x * x > 4 * a * c2])
    if rng.random() < 0.4:
        a, b, c2 = -a, -b, -c2
    u, v, w = rng.sample(VARS, 3)
    form = a * u**2 + b * u * v + c2 * v**2
    key = classify(w - form)
    check(key == want, "the discriminant reading differs from the classifier")
    check((b * b - 4 * a * c2 > 0) == (want == HP), "the discriminant sign does not match the type")
    naive = EP
    text = equation(w, form)
    working = [
        f"The quadratic form {show(form)} has discriminant b² − 4ac = {b * b - 4 * a * c2}.",
        {
            HP: "A positive discriminant means the form is indefinite, whatever the signs of the squares: a hyperbolic paraboloid.",
            PC: "A zero discriminant means the form is a perfect square, so the surface depends on one combination of the variables only: a parabolic cylinder.",
        }[key],
    ]
    return _choice(
        phrase(rng, _CROSS_STEMS, eq=text),
        key,
        _confusion_wrong(rng, key, extra=[("quad-cross-term-ignored", naive)], minimum=4),
        working,
    )


# -- generator --------------------------------------------------------------------------------------------------


@generator(
    "w7c_quadric",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7c_quadric(rng, tier="easy"):
    """Quadric surfaces, level sets and traces; the tiers are described in the module docstring."""
    if tier == "easy":
        return rng.choice([_easy_name, _easy_name, _easy_level, _easy_trace])(rng)
    if tier == "medium":
        return rng.choice([_medium_name, _medium_centre, _medium_axis, _medium_degenerate])(rng)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice([_hard_oblique, _hard_radius, _hard_intercept, _hard_cross])(rng)


@redrawn
def _medium_degenerate(rng):
    """A trace that is degenerate or empty: through the vertex or centre, or missing the surface."""
    for _ in range(60):
        name = rng.choice([ELL, H2, CONE, HP, H1])
        s = _std(rng, name)
        var = rng.choice(VARS)
        k = rng.choice([0, 0, 1, -1, 2, -2, 4, 5, -5, 7])
        _expr, _a, _b, label = _trace(s.f, var, k)
        if label in (POINT, EMPTY, LINES):
            return _trace_question(rng, s, var, k, minimum=4)
    raise Redraw("no degenerate trace")


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_w7c_quadric", level=19, presets={"exam": dict(tier="exam")})
@redrawn
def exd_x_w7c_quadric(rng, tier="exam"):
    """Exam chain on x²/a² + y²/b² ± z²/c² = 1 (or the two-sheet form) and a plane z = k: the surface, the two semi-axes of
    the cross-section, and its area as a multiple of π."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice([ELL, H1, H2])
    P, k, m, A, B = _semi_axes(rng, kind)
    s = _kind_surface(rng, kind, P, A, B)
    u, v, w = s.roles
    kv = k * rng.choice([1, -1])
    a_val, b_val = Fraction(A * m, P), Fraction(B * m, P)
    check(a_val.denominator == 1 and b_val.denominator == 1, "semi-axes not whole")
    a_val, b_val = int(a_val), int(b_val)
    _check_axis(s, w, kv, u, a_val)
    _check_axis(s, w, kv, v, b_val)
    if str(v) < str(u):  # ask for the x semi-axis before the y one, whichever way round the roles came out
        u, v, a_val, b_val, A, B = v, u, b_val, a_val, B, A
    area = a_val * b_val
    plane = f"{w} = {kv}".replace("-", "−")
    key = classify(s.f)
    check(key == kind, "the classifier disagrees")
    wrong_a = _axis_candidates(kind, A, B, P, k, m)
    wrong_b = _axis_candidates(kind, B, A, P, k, m)
    name_wrong = _confusion_wrong(rng, kind)
    return dict(
        text=f"A surface S has equation {s.text}. S is cut by the plane {plane} in an ellipse E.",
        parts=[
            choice_part(
                "What type of quadric surface is S?",
                kind,
                2,
                _name_working(s),
                [(t, n) for t, n in name_wrong],
            ),
            value_part(
                f"Find the semi-axis of E in the {u} direction.",
                a_val,
                2,
                [
                    f"With {plane}, the equation gives {show(sympy.expand(s.f.subs(w, kv)))} = 0, so the {u}-semi-axis is {a_val}."
                ],
                wrong_a,
                sensible_range=(0, None),
            ),
            value_part(
                f"Find the semi-axis of E in the {v} direction.",
                b_val,
                2,
                [f"In the same way the {v}-semi-axis is {b_val}."],
                wrong_b,
                sensible_range=(0, None),
            ),
            value_part(
                "The area of E is kπ. Find k.",
                area,
                3,
                [f"The area of an ellipse with semi-axes p and q is πpq, so k = {a_val} × {b_val} = {area}."],
                [
                    ("quad-area-uses-sum", a_val + b_val),
                    ("quad-area-halved", Fraction(a_val * b_val, 2)),
                    ("quad-area-uses-squares", a_val * a_val + b_val * b_val),
                    ("quad-area-perimeter-style", 2 * (a_val + b_val)),
                    ("quad-area-uses-denominators", A * A * B * B),
                ],
                follow_through="part_b*part_c",
                sensible_range=(0, None),
            ),
        ],
    )
