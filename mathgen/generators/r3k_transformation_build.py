"""Generates `r3k_transformation` questions (easy, medium, hard) and the exam chain `exd_x_r3k_transformation`.

``uc2-transformation-of-random-variable`` (``probability``, answer ``expression``, exact values only): the density of
Y = g(X) from the density of X.

* easy: Y = aX + b for a polynomial density of X on (0, L), with a negative as often as positive (so the Jacobian is
  1/|a| and the support flips).
* medium: a monotone non-linear map: Y = X^2 or Y = sqrt(X) or Y = 1/X for a polynomial density, or Y = e^(-X) or Y = e^X
  for an exponential X.
* hard: a map that is not one-to-one, Y = X^2 or Y = |X| for a polynomial density on (-a, b) (add the two branches
  where both apply), or the density of the sum of two independent uniform variables on one of the three pieces of its
  support (rising, flat, falling).
* exam (``exd_x_r3k_transformation``): Y = X^2 for a power density on (0, L), or Y = e^(-X) for an exponential X:
  (a) the distribution function F_Y(y), (b) F_Y(y0) (follows through from (a)), (c) the density f_Y(y), (d) P(y0 < Y <
  y1) (follows through from (a) and (b)).

Each density is first built with the change-of-variable formula f_X(h(y)) |h'(y)| summed over the branches, then
checked by the distribution-function method (differentiate F_X(h(y)) or 1 - F_X(h(y)) or F_X(sqrt(y)) - F_X(-sqrt(y)))
and by numerical quadrature of the density over its support. The convolution is checked by quadrature of the convolution
integral itself.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath
import sympy

from ..core import attempts, check, generator
from ._r3k_kit import (
    clean_working,
    redraw,
    stored,
    second_route,
    expr_distractors,
    expr_fields,
    frac_distractors,
    frac_fields,
    need,
    pint,
    pretty,
    fr,
)

x = sympy.Symbol("x", real=True)
y = sympy.Symbol("y", positive=True)
s = sympy.Symbol("s", positive=True)
t = sympy.Symbol("t", real=True)
_SUP = {"2": "²", "3": "³"}
_OPENINGS = (
    "The continuous random variable X has density function",
    "A continuous random variable X has probability density function",
    "Let X be a continuous random variable with density function",
)


def utext(expr) -> str:
    text = sympy.sstr(expr, order="lex")
    for k, v in _SUP.items():
        text = text.replace("**" + k, v)
    return text.replace("*", "")


def _power_density(n: int, lo, hi):
    """k x^n on (lo, hi) with k making it integrate to 1."""
    k = 1 / pint(x**n, (x, lo, hi))
    return k * x**n, k


def _density_text(k, n: int) -> str:
    body = "x" if n == 1 else f"x{_SUP[str(n)]}" if n in (2, 3) else f"x^{n}"
    if k == 1:
        return body
    return f"{k.p}{body}" if k.q == 1 else f"({k.p}/{k.q}){body}"


def _jacobian_density(fx, branches):
    """f_Y(y) = sum over the inverse branches h of f_X(h(y)) |h'(y)|."""
    total = 0
    for h in branches:
        total += fx.subs(x, h) * sympy.Abs(sympy.diff(h, y))
    return sympy.cancel(total)


def _cdf_route(fx, lo, hi, case, branches):
    """d/dy of the distribution function of Y built from F_X (the second route to f_Y)."""
    tt = sympy.Symbol("tt", real=True)
    FX = lambda u: sympy.integrate(fx.subs(x, tt), (tt, lo, u))  # noqa: E731
    if case == "two-sided":
        F = FX(branches[0]) - FX(branches[1])  # branches = (sqrt, -sqrt) for y within both sides
    elif case == "decreasing":
        F = 1 - FX(branches[0])
    else:
        F = FX(branches[0])
    return sympy.simplify(sympy.diff(F, y)), F


def _shifted(var: str, b: int) -> str:
    """`Y - 3`, `Y + 2` or just `Y`: the subtraction of b written without a doubled sign."""
    return var if b == 0 else f"{var} - {b}" if b > 0 else f"{var} + {-b}"


def _bracket(a: int) -> str:
    return f"({a})" if a < 0 else str(a)


def _rationals(values):
    """The values that are exact rationals, as Fractions (a follow-through value that is not rational cannot collide)."""
    out = []
    for v in values:
        v = sympy.nsimplify(v) if isinstance(v, sympy.Basic) else v
        if getattr(v, "is_Rational", False):
            out.append(fr(v))
        elif isinstance(v, (int, Fraction)):
            out.append(Fraction(v))
    return out


def _at(expr, point):
    """`expr` (any free symbol) evaluated at `point`."""
    return expr.subs({symbol: point for symbol in expr.free_symbols})


def _numeric_total(fy, lo, hi):
    f = sympy.lambdify(y, fy, "mpmath")
    return mpmath.quad(f, [mpmath.mpf(float(lo)), mpmath.inf if hi == sympy.oo else mpmath.mpf(float(hi))])


# -- easy: a linear map -------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a linear transformation of a power density"):
        n = rng.randint(1, 3)
        lim = rng.randint(1, 3)
        a = rng.choice((2, 3, 4, -2, -3, -4))
        b = rng.randint(-2, 5)
        fx, k = _power_density(n, 0, lim)
        if k.q > 27:
            continue
        break
    h = (y - b) / a
    fy = _jacobian_density(fx, [h])
    lo, hi = sorted((b, a * lim + b))
    if second_route():
        # second route: distribution function, differentiated
        cdf_form = "decreasing" if a < 0 else "increasing"
        fy2, _ = _cdf_route(fx, 0, lim, cdf_form, [h])
        check(sympy.simplify(fy - fy2) == 0, "the distribution function method disagrees")
        check(
            sympy.simplify(pint(sympy.expand(fy), (y, lo, hi)) - 1) == 0, "f_Y does not integrate to 1 over its support"
        )
    cands = [
        (fx.subs(x, h), "r3k-transf-forgot-jacobian"),
        (fx.subs(x, h) * abs(a), "r3k-transf-jacobian-inverted"),
        (fx.subs(x, h) / a if a < 0 else None, "r3k-transf-negative-slope-sign"),
        (fx.subs(x, y / a) / abs(a), "r3k-transf-forgot-the-shift"),
        (fx.subs(x, a * y + b) / abs(a), "r3k-transf-substituted-forward"),
    ]
    dist = need(expr_distractors(fy, cands, symbols=(y,)))
    g_text = f"{a}X + {b}" if b > 0 else (f"{a}X - {-b}" if b < 0 else f"{a}X")
    text = (
        f"The continuous random variable X has density function f_X(x) = {_density_text(k, n)} for 0 < x < {lim}, and 0 "
        f"otherwise. Let Y = {g_text}. Find the density function f_Y(y) of Y for {lo} < y < {hi}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Y = {g_text} is one-to-one, with X = ({_shifted('Y', b)})/{_bracket(a)} and |dx/dy| = 1/{abs(a)}.",
            f"f_Y(y) = f_X(({_shifted('y', b)})/{_bracket(a)}) / {abs(a)} = {pretty(fy)}.",
        ],
        **expr_fields(fy),
    )


# -- medium: a monotone non-linear map -----------------------------------------------------------------------------


def _medium_power(rng):
    kind = rng.choice(("square", "root", "reciprocal"))
    for _ in attempts(500, "a monotone map of a power density"):
        n = rng.randint(2, 3) if kind == "square" else rng.randint(1, 3)
        if kind == "square":
            lo, hi = 0, rng.randint(1, 3)
            g_txt, h = "X²", sympy.sqrt(y)
            y_lo, y_hi = 0, hi**2
        elif kind == "root":
            lo, hi = 0, rng.choice((1, 4, 9))
            g_txt, h = "√X", y**2
            y_lo, y_hi = 0, sympy.sqrt(hi)
        else:
            lo, hi = 1, rng.randint(2, 3)
            g_txt, h = "1/X", 1 / y
            y_lo, y_hi = sympy.Rational(1, hi), 1
        fx, k = _power_density(n, lo, hi)
        if k.q > 400 or k.p > 400:
            continue
        break
    fy = _jacobian_density(fx, [h])
    if second_route():
        case = "decreasing" if kind == "reciprocal" else "increasing"
        fy2, _ = _cdf_route(fx, lo, hi, case, [h])
        check(sympy.simplify(fy - fy2) == 0, "the distribution function method disagrees")
        check(abs(_numeric_total(fy, y_lo, y_hi) - 1) < 1e-10, "f_Y does not integrate to 1 over its support")
    gy = {"square": y**2, "root": sympy.sqrt(y), "reciprocal": 1 / y}[kind]
    gprime = sympy.diff(gy, y)
    cands = [
        (fx.subs(x, h), "r3k-transf-forgot-jacobian"),
        (fx.subs(x, h) / sympy.Abs(sympy.diff(h, y)), "r3k-transf-jacobian-inverted"),
        (fx.subs(x, gy) * sympy.Abs(gprime), "r3k-transf-substituted-forward"),
        (fx.subs(x, y) * sympy.Abs(sympy.diff(h, y)), "r3k-transf-forgot-to-invert"),
    ]
    dist = need(expr_distractors(fy, cands, symbols=(y,)))
    text = (
        f"The continuous random variable X has density function f_X(x) = {_density_text(k, n)} for {lo} < x < {hi}, and 0 "
        f"otherwise. Let Y = {g_txt}. Find the density function f_Y(y) of Y for {y_lo} < y < {y_hi}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Y = {g_txt} is one-to-one on the support of X, with X = {pretty(h)}.",
            f"f_Y(y) = f_X(h(y)) |h'(y)| = {pretty(fy)}.",
        ],
        **expr_fields(fy),
    )


def _medium_exponential(rng):
    lam = rng.choice((2, 3, 4))
    kind = rng.choice(("neg", "pos"))
    fx = lam * sympy.exp(-lam * x)
    if kind == "neg":
        h, g_txt, rng_txt = -sympy.log(y), "e^(-X)", "0 < y < 1"
        y_lo, y_hi = 0, 1
        fy = _jacobian_density(fx, [h])
        gy = sympy.exp(-y)
    else:
        h, g_txt, rng_txt = sympy.log(y), "e^X", "y > 1"
        y_lo, y_hi = 1, sympy.oo
        fy = _jacobian_density(fx, [h])
        gy = sympy.exp(y)
    if second_route():
        fy2, _ = _cdf_route(fx, 0, sympy.oo, "decreasing" if kind == "neg" else "increasing", [h])
        check(sympy.simplify(fy - fy2) == 0, "the distribution function method disagrees")
        check(abs(_numeric_total(fy, y_lo, y_hi) - 1) < 1e-10, "f_Y does not integrate to 1")
    cands = [
        (fx.subs(x, h), "r3k-transf-forgot-jacobian"),
        (fx.subs(x, h) / sympy.Abs(sympy.diff(h, y)), "r3k-transf-jacobian-inverted"),
        (fx.subs(x, gy) * sympy.Abs(sympy.diff(gy, y)), "r3k-transf-substituted-forward"),
        (-fy, "r3k-transf-negative-slope-sign"),
    ]
    dist = need(expr_distractors(fy, cands, symbols=(y,)))
    text = (
        f"The continuous random variable X has density function f_X(x) = {lam}e^(-{lam}x) for x > 0, and 0 otherwise. "
        f"Let Y = {g_txt}. Find the density function f_Y(y) of Y for {rng_txt}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Y = {g_txt} is one-to-one for x > 0, with X = {pretty(h)}.",
            f"f_Y(y) = f_X(h(y)) |h'(y)| = {pretty(fy)}.",
        ],
        **expr_fields(fy),
    )


# -- hard: two branches, and a convolution of uniforms -------------------------------------------------------------

_SIDES = ((1, 2), (1, 3), (2, 3), (1, 4))


def _hard_branches(rng):
    a, b = rng.choice(_SIDES)
    shape = rng.choice(("square_x2", "square_lin", "square_flat", "abs_lin", "abs_lin"))
    if shape.endswith("x2"):
        raw = x**2
    elif shape.endswith("lin"):
        raw = x + a
    else:
        raw = sympy.Integer(1)
    k = 1 / pint(raw, (x, -a, b))
    fx = k * raw
    if shape.startswith("square"):
        g_txt, plus, minus, y_hi = "X²", sympy.sqrt(y), -sympy.sqrt(y), a**2
    else:
        g_txt, plus, minus, y_hi = "|X|", y, -y, a
    fy = _jacobian_density(fx, [plus, minus])
    # second route: F_Y(y) = F_X(sqrt y) - F_X(-sqrt y) for 0 < y < a^2 (both sides inside the support)
    if second_route():
        fy2, _ = _cdf_route(fx, -a, b, "two-sided", [plus, minus])
        check(sympy.simplify(fy - fy2) == 0, "the distribution function method disagrees")
    # the full density over its whole support integrates to 1: two branches below y_hi, one branch above it
    one = fx.subs(x, plus) * sympy.Abs(sympy.diff(plus, y))
    y_top = b**2 if shape.startswith("square") else b
    if second_route():
        total = mpmath.quad(sympy.lambdify(y, fy, "mpmath"), [0, y_hi]) + mpmath.quad(
            sympy.lambdify(y, one, "mpmath"), [y_hi, y_top]
        )
        check(abs(total - 1) < 1e-6, "f_Y does not integrate to 1")
    cands = [
        (one, "r3k-transf-one-branch-only"),
        (sympy.simplify(fx.subs(x, plus) + fx.subs(x, minus)), "r3k-transf-forgot-jacobian"),
        (sympy.simplify(2 * one), "r3k-transf-doubled-one-branch"),
        (fx.subs(x, minus) * sympy.Abs(sympy.diff(minus, y)), "r3k-transf-one-branch-only"),
        (sympy.simplify(fy / 2), "r3k-transf-averaged-the-branches"),
        (
            sympy.simplify(
                fx.subs(x, plus) * sympy.Abs(sympy.diff(plus, y)) - fx.subs(x, minus) * sympy.Abs(sympy.diff(minus, y))
            ),
            "r3k-transf-subtracted-the-branches",
        ),
    ]
    dist = need(expr_distractors(fy, cands, symbols=(y,)))
    if shape.endswith("x2"):
        dens = "x²"
    elif shape.endswith("lin"):
        dens = f"(x + {a})"
    else:
        dens = ""
    lead = "" if k == 1 else (f"{k.p}" if k.q == 1 else f"({k.p}/{k.q})")
    dens = (lead + dens) if dens else (lead or "1")
    text = (
        f"The continuous random variable X has density function f_X(x) = {dens} for -{a} < x < {b}, and 0 otherwise. Let "
        f"Y = {g_txt}. Find the density function f_Y(y) of Y for 0 < y < {y_hi}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"For 0 < y < {y_hi}, Y = {g_txt} has two inverse branches, x = {pretty(plus)} and x = {pretty(minus)}, both inside the support of X.",
            f"f_Y(y) = f_X({pretty(plus)}) |h1'(y)| + f_X({pretty(minus)}) |h2'(y)| = {pretty(fy)}.",
        ],
        **expr_fields(fy),
    )


def _hard_convolution(rng):
    for _ in attempts(100, "two uniform widths"):
        a = rng.randint(1, 4)
        b = rng.randint(a + 1, 6)
        break
    piece = rng.choice(("rising", "flat", "falling"))
    if piece == "rising":
        fy, lo, hi = s / (a * b), 0, a
    elif piece == "flat":
        fy, lo, hi = sympy.Rational(1, b), a, b
    else:
        fy, lo, hi = (a + b - s) / (a * b), b, a + b
    # second route: numerical quadrature of the convolution integral at a sample point of the piece
    if second_route():
        point = mpmath.mpf(lo) + (mpmath.mpf(hi) - lo) * mpmath.mpf("0.37")

        def integrand(u):
            return (1 / mpmath.mpf(a) if 0 < u < a else 0) * (1 / mpmath.mpf(b) if 0 < point - u < b else 0)

        breaks = sorted({0, a, point - b if point - b > 0 else 0, point if point < a else a})
        breaks = [mpmath.mpf(v) for v in sorted(set(breaks)) if v <= a]
        conv = mpmath.quad(
            integrand,
            sorted(
                {mpmath.mpf(0), mpmath.mpf(a)}
                | {v for v in breaks if 0 <= v <= a}
                | ({point} if 0 <= point <= a else set())
                | ({point - b} if 0 <= point - b <= a else set())
            ),
        )
        exact = sympy.lambdify(s, fy, "mpmath")(point)
        check(abs(conv - exact) < 1e-10, "the convolution integral disagrees")
    rising, flat, falling = s / (a * b), sympy.Rational(1, b), (a + b - s) / (a * b)
    cands = [
        (rising if piece != "rising" else None, "r3k-transf-convolution-wrong-piece"),
        (flat if piece != "flat" else None, "r3k-transf-convolution-wrong-piece"),
        (falling if piece != "falling" else None, "r3k-transf-convolution-wrong-piece"),
        (sympy.Rational(1, a * b), "r3k-transf-convolution-product-of-densities"),
        (sympy.Rational(1, a + b), "r3k-transf-convolution-uniform-on-the-sum"),
    ]
    dist = need(expr_distractors(fy, cands, symbols=(s,)))
    text = (
        f"X and Y are independent continuous random variables, X uniform on (0, {a}) and Y uniform on (0, {b}). Find the "
        f"density function of S = X + Y for {lo} < s < {hi}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "f_S(s) = the integral of f_X(u) f_Y(s - u) du over the u for which both densities are non-zero.",
            f"On {lo} < s < {hi} the overlap gives f_S(s) = {pretty(fy)}.",
        ],
        **expr_fields(fy),
    )


@generator(
    "r3k_transformation",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_transformation(rng, tier="easy"):
    """Density of a function of a random variable; see the module docstring for the tiers."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return (_medium_power if rng.random() < 0.6 else _medium_exponential)(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return (_hard_branches if rng.random() < 0.6 else _hard_convolution)(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3k_transformation", level=19, presets={"exam": dict(tier="exam")})
@redraw
@clean_working
def exd_x_r3k_transformation(rng, tier="exam"):
    """F_Y(y), F_Y(y0), f_Y(y) and P(y0 < Y < y1) for Y = X^2 (power density) or Y = e^(-X) (exponential)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    if rng.random() < 0.55:
        n = rng.randint(1, 3)
        lim = rng.randint(3, 4)
        fx, k = _power_density(n, 0, lim)
        r0 = rng.randint(1, lim - 2)
        r1 = rng.randint(r0 + 1, lim - 1)
        y0, y1 = r0**2, r1**2
        y_hi = lim**2
        h = sympy.sqrt(y)
        cdf = sympy.expand(pint(fx.subs(x, t), (t, 0, h)))  # F_X(h(y)), integrated term by term
        density = f"{_density_text(k, n)} for 0 < x < {lim}"
        g_txt, support = "X²", f"0 < y < {y_hi}"
        fy = _jacobian_density(fx, [h])
        wrong_cdf = [
            (sympy.expand(pint(fx.subs(x, t), (t, 0, y))), "r3k-transf-cdf-of-x-not-y"),
            (sympy.expand(1 - cdf), "r3k-transf-cdf-upper-tail"),
            (sympy.expand(pint(fx.subs(x, t), (t, 0, y**2))), "r3k-transf-cdf-inverse-direction"),
        ]
        cdf_note = f"Y ≤ y exactly when X ≤ √y, so F_Y(y) = F_X(√y) = {pretty(cdf)}."
        wrong_b = fr(pint(fx, (x, 0, y0)))
        wrong_p = fr(pint(fx, (x, 0, y1)) - pint(fx, (x, 0, y0)))
    else:
        lam = rng.choice((2, 3, 4))
        fx = lam * sympy.exp(-lam * x)
        h = -sympy.log(y)
        cdf = y**lam  # 1 - F_X(-ln y) = e^(-lam (-ln y))
        fy = _jacobian_density(fx, [h])
        y0 = sympy.Rational(1, 2)
        y1 = sympy.Rational(3, 4)
        y_hi = 1
        density = f"{lam}e^(-{lam}x) for x > 0"
        g_txt, support = "e^(-X)", "0 < y < 1"
        wrong_cdf = [
            (1 - y**lam, "r3k-transf-cdf-upper-tail"),
            (1 - sympy.exp(-lam * y), "r3k-transf-cdf-of-x-not-y"),
            (sympy.exp(-lam * y), "r3k-transf-cdf-upper-tail"),
        ]
        cdf_note = f"Y ≤ y exactly when X ≥ -ln y, so F_Y(y) = 1 - F_X(-ln y) = {pretty(cdf)}."
        wrong_b = fr(sympy.Rational(1, 2) ** (lam - 1))
        wrong_p = fr(sympy.Rational(3, 4) ** (lam - 1) - sympy.Rational(1, 2) ** (lam - 1))
    # second route: F_Y is the integral of f_Y (differentiate to recover f_Y), and F_Y(top of support) = 1
    if second_route():
        check(sympy.simplify(sympy.diff(cdf, y) - fy) == 0, "d/dy of F_Y disagrees with the change-of-variable density")
        check(sympy.simplify(cdf.subs(y, y_hi)) == 1, "F_Y at the top of the support is not 1")
    b_val = cdf.subs(y, y0)
    p_val = cdf.subs(y, y1) - b_val
    fy_cands = [
        (fx.subs(x, h), "r3k-transf-forgot-jacobian"),
        (fx.subs(x, h) / sympy.Abs(sympy.diff(h, y)), "r3k-transf-jacobian-inverted"),
        (cdf, "r3k-transf-density-is-the-cdf"),
    ]
    a_dist = need(expr_distractors(cdf, wrong_cdf, symbols=(y,)))
    c_dist = need(expr_distractors(fy, fy_cands, symbols=(y,)))
    b_dist = need(
        frac_distractors(
            fr(b_val),
            [
                (fr(1 - b_val), "r3k-transf-cdf-upper-tail"),
                (wrong_b, "r3k-transf-cdf-of-x-not-y"),
                (fr(cdf.subs(y, y1)), "r3k-transf-value-at-the-wrong-point"),
                (fr(fy.subs(y, y0)), "r3k-transf-density-is-the-cdf"),
            ],
            positive=True,
            at_most=1,
            avoid=_rationals(_at(stored(d), y0) for d in a_dist),
        )
    )
    # a reply right by follow-through from a wrong (a) or (b) must never be one of (b)'s or (d)'s own distractors
    carried_d = _rationals(_at(stored(d), y1) - b_val for d in a_dist) + _rationals(
        cdf.subs(y, y1) - stored(d) for d in b_dist
    )
    d_dist = need(
        frac_distractors(
            fr(p_val),
            [
                (fr(cdf.subs(y, y1)), "r3k-transf-forgot-to-subtract"),
                (fr(cdf.subs(y, y1) + b_val), "r3k-transf-added-instead-of-subtracted"),
                (fr(1 - cdf.subs(y, y1)), "r3k-transf-cdf-upper-tail"),
                (fr(1 - b_val), "r3k-transf-upper-tail-only"),
                (wrong_p, "r3k-transf-cdf-of-x-not-y"),
                (fr(fy.subs(y, y1) - fy.subs(y, y0)), "r3k-transf-density-is-the-cdf"),
            ],
            positive=True,
            at_most=1,
            avoid=carried_d,
        )
    )
    y0_text = str(y0) if y0 == int(y0) else f"{y0.p}/{y0.q}"
    y1_text = str(y1) if y1 == int(y1) else f"{y1.p}/{y1.q}"
    ft_b = f"Subs(part_a, (y,), ({y0_text},))"
    ft_d = f"Subs(part_a, (y,), ({y1_text},)) - part_b"
    return dict(
        text=(
            f"{rng.choice(_OPENINGS)} f_X(x) = {density}, and 0 otherwise. Let Y = {g_txt}, "
            f"which takes values in {support}."
        ),
        parts=[
            dict(
                prompt="Find the distribution function F_Y(y) = P(Y ≤ y) of Y for %s." % support,
                marks=3,
                working=[cdf_note],
                distractors=a_dist,
                **expr_fields(cdf),
            ),
            dict(
                prompt=f"Hence find P(Y ≤ {y0_text}).",
                marks=1,
                follow_through=ft_b,
                working=[f"F_Y({y0_text}) = {b_val}."],
                distractors=b_dist,
                **frac_fields(fr(b_val)),
            ),
            dict(
                prompt="Find the density function f_Y(y) of Y for %s." % support,
                marks=2,
                working=[f"Differentiate: f_Y(y) = {pretty(fy)}."],
                distractors=c_dist,
                **expr_fields(fy),
            ),
            dict(
                prompt=f"Find P({y0_text} < Y < {y1_text}).",
                marks=2,
                follow_through=ft_d,
                working=[f"P = F_Y({y1_text}) - F_Y({y0_text}) = {p_val}."],
                distractors=d_dist,
                **frac_fields(fr(p_val)),
            ),
        ],
    )
