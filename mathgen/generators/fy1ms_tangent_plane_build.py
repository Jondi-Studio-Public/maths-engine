"""Generates `fy1ms_tangent_plane` questions and the exam chain `exd_x_fy1ms_tangent_plane`.

Skill: `ucv1-tangent-plane-linear-approximation` (multivariable-and-vector-calculus). The tangent plane to a surface
z = f(x, y) at a point, and the linear approximation it gives.

Tiers (topic `fy1ms_tangent_plane`; every answer is a number):

* easy   -- estimate f near (x0, y0) with the tangent plane, for a polynomial f and a small step (h, k);
* medium -- the constant term C of the tangent plane written z = Ax + By + C, for a polynomial or a monomial-style f
            that needs the power rule in both variables;
* hard   -- one of: a linear estimate for f = k * sqrt(x^2 + y^2) near a point with whole-number radius; or the
            z-axis intercept of the tangent plane to the surface a x^2 + b y^2 + c z^2 = k (a surface given
            implicitly, so the plane comes from the gradient of F, not from partial derivatives of z).

Exam (topic `exd_x_fy1ms_tangent_plane`) is a four-part chain on a hillside model z = f(x, y):
(a) the partial derivative in x at P (1 mark), (b) the partial derivative in y at P (1 mark), (c) the equation of the
tangent plane (2 marks, follow-through from (a) and (b)), (d) the estimated height at a nearby point (2 marks,
follow-through from (a) and (b)).

Construction: the point and the polynomial are drawn first and every value (f, f_x, f_y) is computed twice, from the
terms by hand and with `sympy.diff`; the two must agree. Distractors are the usual slips: the power-rule multiplier
dropped, the partials swapped, the shift of the plane forgotten or added instead of subtracted, the base value
f(x0, y0) left out, the exact value of f given instead of the approximation.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy import Rational, symbols

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, paren_negative
from ._fy1ms_common import (
    Fm,
    build_until,
    clear_of_tie,
    dp_text,
    fm_expr,
    num_dist,
    number_part,
    number_question,
    plain_expr,
    say,
    typed_expr,
)

x, y, z = symbols("x y z")

# A term is (coefficient, power of x, power of y).
Term = tuple[int, int, int]


def _expr(terms: list[Term]):
    return sum((c * x**i * y**j for c, i, j in terms), sympy.Integer(0))


def _value(terms: list[Term], px: int, py: int) -> int:
    return sum(c * px**i * py**j for c, i, j in terms)


def _fx(terms: list[Term], px: int, py: int, *, multiplier: bool = True) -> int:
    return sum((c * (i if multiplier else 1)) * px ** (i - 1) * py**j for c, i, j in terms if i > 0)


def _fy(terms: list[Term], px: int, py: int, *, multiplier: bool = True) -> int:
    return sum((c * (j if multiplier else 1)) * px**i * py ** (j - 1) for c, i, j in terms if j > 0)


def _draw_terms(rng: random.Random, family: str) -> list[Term]:
    nz = (-3, -2, -1, 1, 2, 3)
    if family == "quad":
        a, b, c, d, e = (rng.choice(nz) for _ in range(5))
        return [(a, 2, 0), (b, 1, 1), (c, 0, 2), (d, 1, 0), (e, 0, 1)]
    if family == "mono":
        p, q = rng.choice([(2, 1), (1, 2), (2, 2), (3, 1), (1, 3)])
        k = rng.choice((-2, -1, 1, 2, 3))
        extra = rng.choice((-3, -2, -1, 1, 2, 3))
        return [(k, p, q), (extra, 0, 1)] if rng.random() < 0.5 else [(k, p, q), (extra, 1, 0)]
    # cubic with a mixed term
    a, b, c = (rng.choice(nz) for _ in range(3))
    return [(a, 3, 0), (b, 1, 2), (c, 0, 1)]


def _draw_surface(rng: random.Random, families=("quad", "mono", "cubic"), *, positive: bool = False):
    """(terms, x0, y0, f0, fx, fy) with a usable, non-degenerate tangent plane."""
    for _ in attempts(2000, "a tangent plane with distinct non-zero slopes"):
        terms = _draw_terms(rng, rng.choice(families))
        px, py = rng.choice((-3, -2, -1, 1, 2, 3, 4)), rng.choice((-3, -2, -1, 1, 2, 3, 4))
        if positive:
            px, py = abs(px), abs(py)
        f0, fx, fy = _value(terms, px, py), _fx(terms, px, py), _fy(terms, px, py)
        if fx == 0 or fy == 0 or fx == fy or fx == -fy or f0 == 0:
            continue
        if abs(f0) > 150 or abs(fx) > 60 or abs(fy) > 60:
            continue
        if positive and not (10 <= f0 <= 150):
            continue
        # a second route: SymPy differentiation of the same polynomial
        f = _expr(terms)
        check(f.subs({x: px, y: py}) == f0, "f(x0, y0) disagrees with SymPy")
        check(sympy.diff(f, x).subs({x: px, y: py}) == fx, "f_x disagrees with SymPy")
        check(sympy.diff(f, y).subs({x: px, y: py}) == fy, "f_y disagrees with SymPy")
        return terms, px, py, f0, fx, fy
    raise AssertionError("unreachable")  # pragma: no cover


def _f_fm(terms: list[Term]) -> Fm:
    return fm_expr(_expr(terms))


def _step(rng: random.Random) -> tuple[Fraction, Fraction, int]:
    """A step (h, k) in tenths or hundredths, and the decimal places a linear estimate with whole-number slopes has."""
    dp = rng.choice((1, 1, 2))
    unit = Fraction(1, 10**dp)
    h = unit * rng.choice((-3, -2, -1, 1, 2, 3))
    k = unit * rng.choice((-3, -2, -1, 1, 2, 3))
    return h, k, dp


def _dec(value: Fraction) -> str:
    return str(float(value))


def _plane_shift(f0: int, fx: int, fy: int, px: int, py: int) -> int:
    """C in z = fx x + fy y + C."""
    return f0 - fx * px - fy * py


# ---------------------------------------------------------------------------
# Easy: estimate f near the point with the tangent plane


def _easy(rng: random.Random) -> dict:
    terms, px, py, f0, fx, fy = _draw_surface(rng, ("quad", "mono"))
    h, k, dp = _step(rng)
    est = f0 + fx * h + fy * k
    f = _expr(terms)
    exact = sum(Fraction(c) * (px + h) ** i * (py + k) ** j for c, i, j in terms)
    # second route: evaluate the plane through SymPy
    plane = f0 + fx * (x - px) + fy * (y - py)
    sym = plane.subs({x: Rational(px + h), y: Rational(py + k)})
    check(Fraction(int(sym.p), int(sym.q)) == est, "tangent-plane estimate disagrees with SymPy")
    x1, y1 = px + h, py + k
    cands = [
        num_dist(fx * h + fy * k, dp, "linear-approx-omits-base-value"),
        num_dist(f0 + fy * h + fx * k, dp, "tangent-plane-partials-swapped"),
        num_dist(exact, dp, "linear-approx-exact-value-given"),
        num_dist(f0 + fx * h, dp, "linear-approx-one-variable-only"),
        num_dist(f0 + fx * px * h + fy * py * k, dp, "linear-approx-slope-times-coordinate"),
    ]
    text, latex = say(
        "Let ",
        Fm("f(x, y) = " + plain_expr(f)),
        ". Use the tangent plane to the surface z = f(x, y) at the point ",
        Fm(f"({px}, {py})"),
        " to estimate ",
        Fm(f"f({_dec(x1)}, {_dec(y1)})"),
        ". Give your answer correct to ",
        f"{dp} decimal place{'s' if dp != 1 else ''}.",
    )
    latex = latex.replace("{f(x, y) = " + plain_expr(f) + "}", "{f(x, y) = " + sympy.latex(f) + "}")
    return number_question(
        text,
        latex,
        est,
        dp,
        cands,
        working=[
            f"f({px}, {py}) = {f0}, f_x = {fx} and f_y = {fy} at the point.",
            f"L(x, y) = {f0} + {paren_negative(fx)}(x - {paren_negative(px)}) + {paren_negative(fy)}(y - {paren_negative(py)}).",
            f"L({_dec(x1)}, {_dec(y1)}) = {dp_text(est, dp)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Medium: the constant term of the tangent plane


def _medium(rng: random.Random) -> dict:
    terms, px, py, f0, fx, fy = _draw_surface(rng)
    c = _plane_shift(f0, fx, fy, px, py)
    # second route: expand the point form with SymPy and read the constant term
    plane = sympy.expand(f0 + fx * (x - px) + fy * (y - py))
    check(plane.subs({x: 0, y: 0}) == c, "constant term disagrees with SymPy")
    check(plane.coeff(x) == fx and plane.coeff(y) == fy, "plane slopes disagree with SymPy")
    cands = [
        num_dist(f0, 0, "tangent-plane-shift-forgotten"),
        num_dist(f0 + fx * px + fy * py, 0, "tangent-plane-shift-sign"),
        num_dist(f0 - fx * px, 0, "tangent-plane-shift-one-variable"),
        num_dist(f0 - fx - fy, 0, "tangent-plane-shift-uses-slopes-only"),
        num_dist(-c, 0, "tangent-plane-constant-negated"),
    ]
    text, latex = say(
        "The surface ",
        Fm("z = " + plain_expr(_expr(terms)), "z = " + sympy.latex(_expr(terms))),
        " has a tangent plane at the point where ",
        Fm(f"x = {px}"),
        " and ",
        Fm(f"y = {py}"),
        ". Write this plane as ",
        Fm("z = Ax + By + C"),
        " and find the value of ",
        Fm("C"),
        ".",
    )
    return number_question(
        text,
        latex,
        c,
        0,
        cands,
        working=[
            f"At ({px}, {py}): z = {f0}, A = f_x = {fx}, B = f_y = {fy}.",
            f"z = {f0} + {paren_negative(fx)}(x - {paren_negative(px)}) + {paren_negative(fy)}(y - {paren_negative(py)}), so C = {f0} - {paren_negative(fx * px)} - {paren_negative(fy * py)} = {c}.",
        ],
    )


# ---------------------------------------------------------------------------
# Hard: a square-root surface, or a surface given implicitly

_RADIAL_POINTS = [
    (3, 4),
    (4, 3),
    (-3, 4),
    (3, -4),
    (-4, 3),
    (4, -3),
    (6, 8),
    (8, 6),
    (-6, 8),
    (6, -8),
    (-8, 6),
    (8, -6),
]


def _hard_radial(rng: random.Random) -> dict:
    px, py = rng.choice(_RADIAL_POINTS)
    r = int(round((px * px + py * py) ** 0.5))
    check(r * r == px * px + py * py, "not a whole-number radius")
    scale = rng.choice((1, 1, 2, 3))
    hh = Fraction(rng.choice((-4, -3, -2, 2, 3, 4)), 100)
    kk = Fraction(rng.choice((-4, -3, -2, 2, 3, 4)), 100)
    slope_x, slope_y = Fraction(scale * px, r), Fraction(scale * py, r)
    est = scale * r + slope_x * hh + slope_y * kk
    f = scale * sympy.sqrt(x**2 + y**2)
    # second route: SymPy differentiates the surface
    check(sympy.diff(f, x).subs({x: px, y: py}) == Rational(slope_x.numerator, slope_x.denominator), "f_x disagrees")
    check(sympy.diff(f, y).subs({x: px, y: py}) == Rational(slope_y.numerator, slope_y.denominator), "f_y disagrees")
    dp = 3
    true_value = scale * ((px + float(hh)) ** 2 + (py + float(kk)) ** 2) ** 0.5
    cands = [
        num_dist(slope_x * hh + slope_y * kk, dp, "linear-approx-omits-base-value"),
        num_dist(scale * r + slope_y * hh + slope_x * kk, dp, "tangent-plane-partials-swapped"),
        num_dist(scale * r + hh + kk, dp, "radial-slope-taken-as-one"),
        num_dist(
            scale * r + Fraction(scale * px, r * r) * hh + Fraction(scale * py, r * r) * kk,
            dp,
            "radial-slope-denominator-squared",
        ),
        num_dist(Fraction(true_value).limit_denominator(10**9), dp, "linear-approx-exact-value-given"),
    ]
    coef = "" if scale == 1 else f"{scale}"
    text, latex = say(
        "Let ",
        Fm(f"f(x, y) = {coef}√(x² + y²)", f"f(x, y) = {coef}\\sqrt{{x^2 + y^2}}"),
        ". Use the linear approximation of f at ",
        Fm(f"({px}, {py})"),
        " to estimate ",
        Fm(f"f({float(px + hh):g}, {float(py + kk):g})"),
        ". Give your answer correct to 3 decimal places.",
    )
    return number_question(
        text,
        latex,
        est,
        dp,
        cands,
        working=[
            f"f({px}, {py}) = {scale * r}; f_x = {coef}x/√(x² + y²) = {slope_x}, f_y = {coef}y/√(x² + y²) = {slope_y}.",
            f"L = {scale * r} + {paren_negative(slope_x)}({float(hh):g}) + {paren_negative(slope_y)}({float(kk):g}) = {dp_text(est, dp)}.",
        ],
    )


def _hard_implicit(rng: random.Random) -> dict:
    for _ in attempts(2000, "an implicit tangent plane with a usable intercept"):
        a, b, c = rng.choice((1, 1, 2)), rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        px, py, pz = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(3))
        total = a * px**2 + b * py**2 + c * pz**2
        if total > 60:
            continue
        zero = c * pz
        intercept = Fraction(total, zero)
        if intercept.denominator == 1:
            continue
        if clear_of_tie(intercept, 3) and abs(intercept) < 40:
            break
    # second route: SymPy builds the plane from the gradient of F
    big_f = a * x**2 + b * y**2 + c * z**2 - total
    grad = [sympy.diff(big_f, v).subs({x: px, y: py, z: pz}) for v in (x, y, z)]
    plane = sum(g * (v - p) for g, v, p in zip(grad, (x, y, z), (px, py, pz)))
    z_int = sympy.solve(plane.subs({x: 0, y: 0}), z)[0]
    check(Fraction(int(z_int.p), int(z_int.q)) == intercept, "intercept disagrees with SymPy")
    dp = 3
    x_intercept = Fraction(total, a * px)
    cands = [
        num_dist(Fraction(total, pz), dp, "implicit-plane-coefficients-dropped"),
        num_dist(Fraction(total, 2 * zero), dp, "implicit-plane-constant-halved"),
        num_dist(x_intercept, dp, "tangent-plane-wrong-axis-intercept"),
        num_dist(-intercept, dp, "tangent-plane-constant-negated"),
    ]
    surface = join_surface(a, b, c)
    text, latex = say(
        "The point ",
        Fm(f"P({px}, {py}, {pz})"),
        " lies on the surface ",
        Fm(f"{surface} = {total}", f"{_surface_tex(a, b, c)} = {total}"),
        ". The tangent plane to the surface at P meets the z-axis at the point (0, 0, ",
        Fm("t"),
        "). Find ",
        Fm("t"),
        ", correct to 3 decimal places.",
    )
    return number_question(
        text,
        latex,
        intercept,
        dp,
        cands,
        working=[
            f"The gradient of F = {surface} is ({2 * a * px}, {2 * b * py}, {2 * c * pz}) at P.",
            f"The plane is {join_terms([(a * px, 'x'), (b * py, 'y'), (c * pz, 'z')])} = {total}; put x = 0 and y = 0 to get t = {dp_text(intercept, dp)}.",
        ],
    )


def join_surface(a: int, b: int, c: int) -> str:
    def term(coef, var):
        return f"{'' if coef == 1 else coef}{var}²"

    return f"{term(a, 'x')} + {term(b, 'y')} + {term(c, 'z')}"


def _surface_tex(a: int, b: int, c: int) -> str:
    def term(coef, var):
        return f"{'' if coef == 1 else coef}{var}^2"

    return f"{term(a, 'x')} + {term(b, 'y')} + {term(c, 'z')}"


@generator(
    "fy1ms_tangent_plane",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_tangent_plane(rng: random.Random, tier: str = "easy") -> dict:
    """Tangent plane and linear approximation: an estimate (easy), the plane's constant (medium), a radial or implicit surface (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy tangent-plane question")
    if tier == "medium":
        return build_until(rng, _medium, "a medium tangent-plane question")
    if tier == "hard":
        return build_until(
            rng, lambda r: _hard_radial(r) if r.random() < 0.55 else _hard_implicit(r), "a hard tangent-plane question"
        )
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain


def _plane_expr_text(a, b, c) -> str:
    return "z = " + typed_expr(a * x + b * y + c)


@generator("exd_x_fy1ms_tangent_plane", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_tangent_plane(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the two partial derivatives at P, the tangent plane, a height estimate (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a tangent-plane exam question")


def _exam_chain(rng: random.Random) -> dict:
    terms, px, py, f0, fx, fy = _draw_surface(rng, ("quad", "mono", "cubic"), positive=True)
    f = _expr(terms)
    h, k, dp = _step(rng)
    est = f0 + fx * h + fy * k
    check(est > 0, "estimated height is not positive")
    exact = sum(Fraction(c0) * (px + h) ** i * (py + k) ** j for c0, i, j in terms)
    c = _plane_shift(f0, fx, fy, px, py)
    # distractors
    fx_wrong_mult = _fx(terms, px, py, multiplier=False)
    fy_wrong_mult = _fy(terms, px, py, multiplier=False)
    swapped_point_fx = _fx(terms, py, px)
    swapped_point_fy = _fy(terms, py, px)
    a_cands = [
        num_dist(fy, 0, "tangent-plane-partials-swapped"),
        num_dist(fx_wrong_mult, 0, "partial-power-rule-multiplier-dropped"),
        num_dist(swapped_point_fx, 0, "tangent-plane-point-coordinates-swapped"),
        num_dist(-fx, 0, "partial-sign-slip"),
    ]
    b_cands = [
        num_dist(fx, 0, "tangent-plane-partials-swapped"),
        num_dist(fy_wrong_mult, 0, "partial-power-rule-multiplier-dropped"),
        num_dist(swapped_point_fy, 0, "tangent-plane-point-coordinates-swapped"),
        num_dist(-fy, 0, "partial-sign-slip"),
    ]
    plane_true = fx * x + fy * y + c
    c_wrong = [
        ("tangent-plane-shift-forgotten", fx * x + fy * y + f0),
        ("tangent-plane-shift-sign", fx * x + fy * y + f0 + fx * px + fy * py),
        ("tangent-plane-partials-swapped", fy * x + fx * y + (f0 - fy * px - fx * py)),
        ("linear-approx-omits-base-value", fx * (x - px) + fy * (y - py)),
    ]
    c_dist = []
    for tag, e in c_wrong:
        if sympy.expand(e - plane_true) != 0:
            c_dist.append(
                Distractor(
                    _plane_expr_text(
                        *(sympy.expand(e).coeff(x), sympy.expand(e).coeff(y), sympy.expand(e).subs({x: 0, y: 0}))
                    ),
                    tag,
                    None,
                )
            )
    check(len(c_dist) >= 2, "too few plane distractors")
    d_cands = [
        num_dist(fx * h + fy * k, dp, "linear-approx-omits-base-value"),
        num_dist(f0 + fy * h + fx * k, dp, "tangent-plane-partials-swapped"),
        num_dist(exact, dp, "linear-approx-exact-value-given"),
        num_dist(f0 + fx * h, dp, "linear-approx-one-variable-only"),
    ]

    def frac_str(v: Fraction) -> str:
        return f"({v.numerator}/{v.denominator})"

    stem, stem_latex = say(
        "A model of a hillside gives the height z metres at the point ",
        Fm("(x, y)", "(x, y)"),
        ", where x and y are in hundreds of metres east and north of a survey peg, as ",
        Fm("z = " + plain_expr(f), "z = " + sympy.latex(f)),
        ". The point ",
        Fm(f"P({px}, {py}, {f0})"),
        " is on the hillside.",
    )
    x1, y1 = px + h, py + k
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            number_part(
                "Find ∂z/∂x at P.",
                fx,
                0,
                a_cands,
                marks=1,
                prompt_latex=r"\text{Find }\frac{\partial z}{\partial x}\text{ at P.}",
                working=[f"∂z/∂x at x = {px}, y = {py} is {fx}."],
            ),
            number_part(
                "Find ∂z/∂y at P.",
                fy,
                0,
                b_cands,
                marks=1,
                prompt_latex=r"\text{Find }\frac{\partial z}{\partial y}\text{ at P.}",
                working=[f"∂z/∂y at x = {px}, y = {py} is {fy}."],
            ),
            dict(
                prompt="Hence find the equation of the tangent plane to the hillside at P, in the form z = Ax + By + C.",
                answer_kind="expression",
                answer=str(plane_true),
                answer_text=_plane_expr_text(fx, fy, c),
                answer_latex=sympy.latex(plane_true).join(["z = ", ""]),
                marks=2,
                working=[
                    f"z - {f0} = {paren_negative(fx)}(x - {paren_negative(px)}) + {paren_negative(fy)}(y - {paren_negative(py)}), so {_plane_expr_text(fx, fy, c)}."
                ],
                distractors=c_dist,
                follow_through=f"part_a*(x - {paren_negative(px)}) + part_b*(y - {paren_negative(py)}) + {f0}",
            ),
            number_part(
                f"Hence use the tangent plane to estimate the height of the hillside at the point ({float(x1):g}, {float(y1):g}). Give your answer correct to {dp} decimal place{'s' if dp != 1 else ''}.",
                est,
                dp,
                d_cands,
                marks=2,
                follow_through=f"{f0} + part_a*{frac_str(h)} + part_b*{frac_str(k)}",
                sensible_range=(0, None),
                working=[
                    f"z ≈ {f0} + {paren_negative(fx)}({float(h):g}) + {paren_negative(fy)}({float(k):g}) = {dp_text(est, dp)} metres."
                ],
            ),
        ],
    )
