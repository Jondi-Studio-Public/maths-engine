"""Generates `r3k_conditional_density` questions (easy, medium, hard) and the exam chain `exd_x_r3k_conditional_density`.

``uc2-conditional-density-expectation`` (``probability``, answer ``expression``, exact values only): the conditional
density of one continuous random variable given the other, and conditional means and variances.

* easy: f(y | x) (or f(x | y)) for a density given in full on a triangle, or on a rectangle where it factorises (so the
  conditional density is the marginal one and X, Y are independent).
* medium: E(Y | X = x) (or E(X | Y = y)) for a density on a triangle, a polynomial in the conditioning value.
* hard: E(Y | X = x) for a density c(x + y)-type on a rectangle (a rational function of x), or Var(Y | X = x) on a
  triangle.
* exam (``exd_x_r3k_conditional_density``): a density on a rectangle: (a) the marginal f_X(x), (b) f(y | x) (follows
  through from (a)), (c) E(Y | X = x), (d) E(Y | X = x0) at a stated x0 (follows through from (c)).

Every conditional quantity is built from the joint density and the marginal, then checked a second way: the conditional
density must integrate to 1, and the conditional mean is compared with numerical quadrature of y f(x, y) / f_X(x) at a
sample point.
"""

from __future__ import annotations

import mpmath
import sympy

from ..core import attempts, check, generator
from ._r3k_kit import clean_working, second_route, expr_distractors, expr_fields, need, pint, pretty, rational
from .r3k_joint_density_build import utext

x, y = sympy.symbols("x y")


def _coef(c: sympy.Rational, g) -> str:
    """The density `c g` as written in a stem: `2y`, `(3/7)(x + y)`, `6x²y`."""
    body = utext(g)
    summed = isinstance(sympy.expand(g), sympy.Add)
    lead = "" if c == 1 else (str(c) if c.q == 1 else f"({c.p}/{c.q})")
    if summed:
        return f"{lead}({body})" if lead else f"({body})"
    return f"{lead}{body}"


# monomial densities on the triangle 0 < y < x < L, and sum densities on a rectangle
_MONO = (y, x, x * y, y**2, x**2, x * y**2, x**2 * y, x + y, x + 2 * y, 2 * x + y)
_SUMS = (x + y, x + 2 * y, 2 * x + y, x + y**2, x**2 + y)
_SEPARABLE = (x * y, x * y**2, x**2 * y, x**2 * y**2, x, y**2)


def _tri_const(g, lim):
    return 1 / pint(pint(g, (y, 0, x)), (x, 0, lim))


def _tri_x_marginal(c, g):
    return c * pint(g, (y, 0, x))


def _tri_y_marginal(c, g, lim):
    return c * pint(g, (x, y, lim))


def _rect_const(g, a, b):
    return 1 / pint(pint(g, (x, 0, a)), (y, 0, b))


def _check_conditional(cond, var, lo, hi):
    if second_route():
        check(rational(pint(cond, (var, lo, hi)) - 1) == 0, "the conditional density does not integrate to 1")


def _stem(density, region, given):
    return (
        f"The continuous random variables X and Y have joint density function f(x, y) = {density} for {region}, and "
        f"f(x, y) = 0 otherwise. {given}"
    )


# -- easy: the conditional density --------------------------------------------------------------------------------


def _easy_triangle(rng):
    for _ in attempts(500, "a triangle density with a readable conditional"):
        g = rng.choice(_MONO)
        lim = rng.randint(1, 3)
        c = _tri_const(g, lim)
        if c.q > 20 or c.p > 100:
            continue
        which = rng.choice(("y_given_x", "x_given_y"))
        fx, fy = _tri_x_marginal(c, g), _tri_y_marginal(c, g, lim)
        joint = c * g
        cond = rational(joint / (fx if which == "y_given_x" else fy))
        if cond.is_number:
            continue
        break
    given_var = x if which == "y_given_x" else y
    lo, hi = (0, x) if which == "y_given_x" else (y, lim)
    _check_conditional(cond, y if which == "y_given_x" else x, lo, hi)
    if second_route():
        f = sympy.lambdify((x, y), joint, "mpmath")
        pt = mpmath.mpf(lim) * mpmath.mpf("0.61")
        if which == "y_given_x":
            marg_num = mpmath.quad(lambda v: f(pt, v), [0, pt])
            value = f(pt, pt / 3) / marg_num
            exact = sympy.lambdify((x, y), cond, "mpmath")(pt, pt / 3)
        else:
            marg_num = mpmath.quad(lambda v: f(v, pt), [pt, lim])
            value = f((pt + lim) / 2, pt) / marg_num
            exact = sympy.lambdify((x, y), cond, "mpmath")((pt + lim) / 2, pt)
        check(abs(value - exact) < 1e-12, "numerical conditional density disagrees")
    wrong_marg = fy if which == "y_given_x" else fx
    cands = [
        (joint, "r3k-cond-gave-the-joint-density"),
        (rational(joint / wrong_marg), "r3k-cond-divided-by-the-other-marginal"),
        (
            rational(joint / (c * pint(g, (y if which == "y_given_x" else x, 0, lim)))),
            "r3k-cond-marginal-over-full-range",
        ),
        (rational(1 / (hi - lo)), "r3k-cond-assumed-uniform"),
    ]
    dist = need(expr_distractors(cond, cands, symbols=(x, y)))
    var = y if which == "y_given_x" else x
    given = (
        f"Find the conditional density of {var.name.upper()} given {given_var.name.upper()} = {given_var}, as an expression "
        f"in {var} and {given_var} valid on the range of {var}."
    )
    region = f"0 < y < x < {lim}"
    return dict(
        text=_stem(_coef(c, g), region, given),
        distractors=dist,
        working=[
            f"Find the marginal density of {given_var.name.upper()} by integrating the joint density over {var}.",
            f"Divide: f({var}|{given_var}) = f(x, y)/f_{given_var.name.upper()}({given_var}) = {pretty(cond)}.",
        ],
        **expr_fields(cond),
    )


def _easy_separable(rng):
    for _ in attempts(500, "a factorising rectangle density"):
        g = rng.choice(_SEPARABLE)
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        c = _rect_const(g, a, b)
        if c.q > 5 or c.p > 200:
            continue
        which = rng.choice(("y_given_x", "x_given_y"))
        fx = c * pint(g, (y, 0, b))
        fy = c * pint(g, (x, 0, a))
        joint = c * g
        cond = rational(joint / (fx if which == "y_given_x" else fy))
        if cond.is_number or not (cond.free_symbols <= ({y} if which == "y_given_x" else {x})):
            continue
        break
    var = y if which == "y_given_x" else x
    given_var = x if which == "y_given_x" else y
    marg = fy if which == "y_given_x" else fx
    check(rational(cond - marg) == 0, "X and Y should be independent")
    check(rational(pint(cond, (var, 0, b if which == "y_given_x" else a)) - 1) == 0, "cond does not integrate to 1")
    other_marg = fx if which == "y_given_x" else fy
    cands = [
        (joint, "r3k-cond-gave-the-joint-density"),
        (other_marg, "r3k-cond-divided-by-the-other-marginal"),
        (rational(joint * c), "r3k-cond-multiplied-by-c-twice"),
        (g, "r3k-cond-forgot-the-constant"),
    ]
    dist = need(expr_distractors(cond, cands, symbols=(x, y)))
    c_txt = c
    region = f"0 < x < {a} and 0 < y < {b}"
    given = (
        f"Find the conditional density of {var.name.upper()} given {given_var.name.upper()} = {given_var}, as an expression "
        f"in {var} (and {given_var} if it depends on it)."
    )
    return dict(
        text=_stem(_coef(c_txt, g), region, given),
        distractors=dist,
        working=[
            f"The marginal density of {given_var.name.upper()} is found by integrating out {var}; dividing the joint density by it leaves {pretty(cond)}.",
            f"The result does not involve {given_var}, so X and Y are independent.",
        ],
        **expr_fields(cond),
    )


# -- medium: a conditional mean on a triangle ----------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a triangle density with a readable conditional mean"):
        g = rng.choice(_MONO)
        lim = rng.randint(1, 3)
        c = _tri_const(g, lim)
        if c.q > 20 or c.p > 100:
            continue
        which = rng.choice(("y_given_x", "x_given_y"))
        joint = c * g
        if which == "y_given_x":
            fx = _tri_x_marginal(c, g)
            mean = rational(pint(y * joint, (y, 0, x)) / fx)
            var, given_var = y, x
            lo, hi = 0, x
        else:
            fy = _tri_y_marginal(c, g, lim)
            mean = rational(pint(x * joint, (x, y, lim)) / fy)
            var, given_var = x, y
            lo, hi = y, lim
        if mean.is_number or sympy.Poly(sympy.fraction(sympy.together(mean))[1], given_var).degree() > 0:
            continue
        break
    # second route: the conditional density integrates to 1 and the mean is numerically quadrature of v f / marginal
    if second_route():
        f = sympy.lambdify((x, y), joint, "mpmath")
        pt = mpmath.mpf(lim) * mpmath.mpf("0.7")
        if which == "y_given_x":
            marg = mpmath.quad(lambda v: f(pt, v), [0, pt])
            num = mpmath.quad(lambda v: v * f(pt, v), [0, pt]) / marg
        else:
            marg = mpmath.quad(lambda v: f(v, pt), [pt, lim])
            num = mpmath.quad(lambda v: v * f(v, pt), [pt, lim]) / marg
        check(abs(num - sympy.lambdify(given_var, mean, "mpmath")(pt)) < 1e-12, "numerical conditional mean disagrees")
    unconditional = (
        pint(pint(var * joint, (y, 0, x)), (x, 0, lim))
        if which == "y_given_x"
        else pint(pint(var * joint, (x, y, lim)), (y, 0, lim))
    )
    forgot_div = pint(var * joint, (var, lo, hi))
    cands = [
        (forgot_div, "r3k-cond-mean-forgot-to-divide"),
        (unconditional, "r3k-cond-mean-unconditional"),
        (rational((lo + hi) / 2), "r3k-cond-mean-assumed-uniform"),
        (
            rational(
                pint(var * joint, (var, 0, lim))
                / (_tri_x_marginal(c, g) if which == "y_given_x" else _tri_y_marginal(c, g, lim))
            ),
            "r3k-cond-mean-wrong-limits",
        ),
    ]
    dist = need(expr_distractors(mean, cands, symbols=(given_var,)))
    region = f"0 < y < x < {lim}"
    given = f"Find E({var.name.upper()} | {given_var.name.upper()} = {given_var}), as an expression in {given_var}."
    return dict(
        text=_stem(_coef(c, g), region, given),
        distractors=dist,
        working=[
            f"The conditional density of {var.name.upper()} given {given_var.name.upper()} = {given_var} is f(x, y)/f_{given_var.name.upper()}({given_var}) over {lo} < {var} < {hi}.",
            f"E({var.name.upper()} | {given_var.name.upper()} = {given_var}) = {pretty(mean)}.",
        ],
        **expr_fields(mean),
    )


# -- hard: a rational conditional mean, or a conditional variance ---------------------------------------------------


def _hard_rect_mean(rng):
    for _ in attempts(500, "a rectangle sum density"):
        g = rng.choice(_SUMS)
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        c = _rect_const(g, a, b)
        if c.q > 40 or c.p > 100:
            continue
        joint = c * g
        fx = pint(joint, (y, 0, b))
        mean = rational(pint(y * joint, (y, 0, b)) / fx)
        if sympy.Poly(sympy.fraction(sympy.together(mean))[1], x).degree() < 1:
            continue
        break
    if second_route():
        f = sympy.lambdify((x, y), joint, "mpmath")
        pt = mpmath.mpf(a) * mpmath.mpf("0.43")
        marg = mpmath.quad(lambda v: f(pt, v), [0, b])
        num = mpmath.quad(lambda v: v * f(pt, v), [0, b]) / marg
        check(abs(num - sympy.lambdify(x, mean, "mpmath")(pt)) < 1e-12, "numerical conditional mean disagrees")
    unconditional = pint(pint(y * joint, (y, 0, b)), (x, 0, a))
    cands = [
        (pint(y * joint, (y, 0, b)), "r3k-cond-mean-forgot-to-divide"),
        (unconditional, "r3k-cond-mean-unconditional"),
        (sympy.Rational(b, 2), "r3k-cond-mean-assumed-uniform"),
        (rational(pint(y * joint, (y, 0, a)) / pint(joint, (y, 0, a))), "r3k-cond-mean-wrong-limits"),
    ]
    dist = need(expr_distractors(mean, cands, symbols=(x,)))
    region = f"0 < x < {a} and 0 < y < {b}"
    return dict(
        text=_stem(_coef(c, g), region, "Find E(Y | X = x), as an expression in x."),
        distractors=dist,
        working=[
            f"f_X(x) = {pretty(rational(fx))}, so the conditional density of Y is f(x, y)/f_X(x) on 0 < y < {b}.",
            f"E(Y | X = x) = {pretty(mean)}.",
        ],
        **expr_fields(mean),
    )


def _hard_tri_variance(rng):
    for _ in attempts(500, "a triangle density with a readable conditional variance"):
        g = rng.choice(_MONO)
        lim = rng.randint(1, 3)
        c = _tri_const(g, lim)
        if c.q > 20 or c.p > 100:
            continue
        joint = c * g
        fx = _tri_x_marginal(c, g)
        m1 = pint(y * joint, (y, 0, x)) / fx
        m2 = pint(y**2 * joint, (y, 0, x)) / fx
        var = rational(m2 - m1**2)
        if var.is_number or sympy.Poly(sympy.fraction(sympy.together(var))[1], x).degree() > 0:
            continue
        break
    if second_route():
        f = sympy.lambdify((x, y), joint, "mpmath")
        pt = mpmath.mpf(lim) * mpmath.mpf("0.55")
        marg = mpmath.quad(lambda v: f(pt, v), [0, pt])
        mu = mpmath.quad(lambda v: v * f(pt, v), [0, pt]) / marg
        num = mpmath.quad(lambda v: (v - mu) ** 2 * f(pt, v), [0, pt]) / marg
        check(abs(num - sympy.lambdify(x, var, "mpmath")(pt)) < 1e-12, "numerical conditional variance disagrees")
    cands = [
        (rational(m2), "r3k-cond-variance-second-moment-only"),
        (rational(m1**2), "r3k-cond-variance-mean-squared"),
        (rational(m2 + m1**2), "r3k-cond-variance-wrong-sign"),
        (rational(x**2 / 12), "r3k-cond-variance-assumed-uniform"),
    ]
    dist = need(expr_distractors(var, cands, symbols=(x,)))
    return dict(
        text=_stem(_coef(c, g), f"0 < y < x < {lim}", "Find Var(Y | X = x), as an expression in x."),
        distractors=dist,
        working=[
            "Find the conditional density of Y given X = x, then E(Y | X = x) and E(Y² | X = x).",
            f"Var(Y | X = x) = E(Y² | X = x) - E(Y | X = x)² = {pretty(var)}.",
        ],
        **expr_fields(var),
    )


@generator(
    "r3k_conditional_density",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_conditional_density(rng, tier="easy"):
    """Conditional densities, means and variances; see the module docstring for the tiers."""
    if tier == "easy":
        return (_easy_triangle if rng.random() < 0.6 else _easy_separable)(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return (_hard_rect_mean if rng.random() < 0.6 else _hard_tri_variance)(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3k_conditional_density", level=19, presets={"exam": dict(tier="exam")})
@clean_working
def exd_x_r3k_conditional_density(rng, tier="exam"):
    """f_X(x), f(y | x), E(Y | X = x) and E(Y | X = x0) for a sum density on a rectangle."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam rectangle density"):
        g = rng.choice(_SUMS)
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        c = _rect_const(g, a, b)
        if c.q > 40 or c.p > 100:
            continue
        joint = c * g
        if not joint.has(y):  # the conditional density would be the constant 1/b: not an interesting question
            continue
        fx = rational(pint(joint, (y, 0, b)))
        mean = rational(pint(y * joint, (y, 0, b)) / fx)
        if sympy.Poly(sympy.fraction(sympy.together(mean))[1], x).degree() < 1:
            continue
        x0 = sympy.Rational(a, 2) if rng.random() < 0.5 else sympy.Integer(a)
        value = mean.subs(x, x0)
        if value.q > 60:
            continue
        break
    cond = rational(joint / fx)
    _check_conditional(cond, y, 0, b)
    if second_route():
        check(pint(fx, (x, 0, a)) == 1, "f_X does not integrate to 1")
    if second_route():
        f = sympy.lambdify((x, y), joint, "mpmath")
        pt = mpmath.mpf(x0.p) / x0.q
        num = mpmath.quad(lambda v: v * f(pt, v), [0, b]) / mpmath.quad(lambda v: f(pt, v), [0, b])
        check(abs(num - mpmath.mpf(value.p) / value.q) < 1e-12, "numerical E(Y | X = x0) disagrees")
    a_cands = [
        (pint(c * g, (x, 0, a)), "r3k-cond-marginal-of-the-other-variable"),
        (pint(g, (y, 0, b)), "r3k-cond-marginal-forgot-c"),
        (joint, "r3k-cond-gave-the-joint-density"),
    ]
    b_cands = [
        (joint, "r3k-cond-gave-the-joint-density"),
        (rational(joint / pint(c * g, (x, 0, a))), "r3k-cond-divided-by-the-other-marginal"),
        (g, "r3k-cond-forgot-the-constant"),
    ]
    c_cands = [
        (pint(y * joint, (y, 0, b)), "r3k-cond-mean-forgot-to-divide"),
        (pint(pint(y * joint, (y, 0, b)), (x, 0, a)), "r3k-cond-mean-unconditional"),
        (sympy.Rational(b, 2), "r3k-cond-mean-assumed-uniform"),
        (rational(pint(y**2 * joint, (y, 0, b)) / fx), "r3k-cond-mean-second-moment"),
    ]
    d_cands = [
        (pint(pint(y * joint, (y, 0, b)), (x, 0, a)), "r3k-cond-mean-unconditional"),
        (pint(y * joint, (y, 0, b)).subs(x, x0), "r3k-cond-mean-forgot-to-divide"),
        (sympy.Rational(b, 2), "r3k-cond-mean-assumed-uniform"),
        (rational(pint(y**2 * joint, (y, 0, b)) / fx).subs(x, x0), "r3k-cond-mean-second-moment"),
    ]
    fx_dist = need(expr_distractors(fx, a_cands, symbols=(x,), limit=2))
    cond_dist = need(expr_distractors(cond, b_cands, symbols=(x, y), limit=2))
    mean_dist = need(expr_distractors(mean, c_cands, symbols=(x,), limit=2))
    # a reply right by follow-through from a wrong (c) must never meet a (d) distractor
    carried = [sympy.sympify(d.value).subs(x, x0) for d in mean_dist]
    d_cands = [(v, t) for v, t in d_cands if not any(sympy.sympify(v) == w for w in carried)]
    val_dist = need(expr_distractors(value, d_cands, limit=2))
    x0_text = f"{x0.p}/{x0.q}" if x0.q != 1 else str(x0.p)
    return dict(
        text=(
            f"The continuous random variables X and Y have joint density function f(x, y) = {_coef(c, g)} for "
            f"0 < x < {a} and 0 < y < {b}, and f(x, y) = 0 otherwise."
        ),
        parts=[
            dict(
                prompt="Find the marginal density f_X(x) of X for 0 < x < %d." % a,
                marks=2,
                working=[f"Integrate out y from 0 to {b}: f_X(x) = {pretty(fx)}."],
                distractors=fx_dist,
                **expr_fields(fx),
            ),
            dict(
                prompt="Find the conditional density f(y|x) of Y given X = x.",
                marks=2,
                working=[f"f(y|x) = f(x, y)/f_X(x) = {pretty(cond)}."],
                distractors=cond_dist,
                **expr_fields(cond),
            ),
            dict(
                prompt="Find E(Y | X = x) as an expression in x.",
                marks=3,
                working=[f"Integrate y f(y|x) from 0 to {b}: E(Y | X = x) = {pretty(mean)}."],
                distractors=mean_dist,
                **expr_fields(mean),
            ),
            dict(
                prompt=f"Hence find E(Y | X = {x0_text}).",
                marks=1,
                follow_through=f"Subs(part_c, (x,), ({x0_text},))",
                working=[f"Substitute x = {x0_text}: {value}."],
                distractors=val_dist,
                **expr_fields(value),
            ),
        ],
    )
