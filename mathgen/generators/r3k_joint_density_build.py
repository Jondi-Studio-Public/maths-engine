"""Generates `r3k_joint_density` questions (easy, medium, hard) and the exam chain `exd_x_r3k_joint_density`.

``uc2-joint-continuous-density`` (``probability``, answer ``expression``, exact values only): a joint density of two
continuous random variables that vanishes outside a region of the plane.

* easy: the normalising constant c of f(x, y) = c g(x, y) on a rectangle, where g is a sum such as x + 2y or x^2 + y (so
  the double integral does not split into a product of two single integrals).
* medium: a probability P(Y > X) or P(X + Y < L) for a density on a square, with c given: the region of integration has
  a limit that depends on the other variable.
* hard: the marginal density f_X(x) or f_Y(y) of f(x, y) = c g(x, y) on a triangle, where c must be found first and the
  inner limit depends on the outer variable.
* exam (``exd_x_r3k_joint_density``): a density c g on a triangle: (a) c, (b) a marginal density, (c) a probability
  P(X <= L/2) or P(Y >= L/2), (d) E(Y); (b), (c) and (d) follow through from c.

Every double integral is done in SymPy in the stated order and again with the order of integration reversed, and
the marginal is also checked against numerical quadrature of the joint density at a sample point.
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
    need,
    pint,
    pretty,
    rational,
)

x, y = sympy.symbols("x y")
_SUP = {"2": "²", "3": "³"}


def utext(expr) -> str:
    """A polynomial in x, y written in a stem: `x²y`, `2x + y`."""
    text = sympy.sstr(sympy.expand(expr), order="lex")
    for k, v in _SUP.items():
        text = text.replace("**" + k, v)
    return text.replace("*", "")


# g(x, y): non-separable sums for the rectangle questions, and a wider set for the triangles
_SUMS = (x + y, x + 2 * y, 2 * x + y, x + y**2, x**2 + y, x**2 + y**2, x + 3 * y, 3 * x + 2 * y)
_ANY = _SUMS + (x * y, x**2 * y, x * y**2, x, y, x**2, y**2)
_ASYM = (x + 2 * y, 2 * x + y, x + y**2, x**2 + y, x + 3 * y, 3 * x + 2 * y, x * y**2, x**2 * y, x, y)


def _g_text(g, coefficient: str = "c") -> str:
    """The density `coefficient × g` as written in a stem: `c(x + y)`, `c·xy²`, `3x²`, `(2/27)y`, `(3/28)(x² + y)`."""
    body = utext(g)
    summed = isinstance(sympy.expand(g), sympy.Add)
    if coefficient == "c":
        return f"c({body})" if summed else f"c·{body}"
    lead = coefficient if "/" not in coefficient else f"({coefficient})"
    return f"{lead}({body})" if summed else f"{lead}{body}"


def _exact(value) -> sympy.Rational:
    return sympy.nsimplify(value)


# -- regions ------------------------------------------------------------------------------------------------------
# "lower": 0 < y < x < L     "upper": 0 < x < y < L


def _tri_double(g, kind, lim, x_first: bool):
    """The integral of g over the triangle, inner variable first in the order stated by `x_first`."""
    if kind == "lower":
        if x_first:  # y outer from 0 to L, x inner from y to L
            return pint(pint(g, (x, y, lim)), (y, 0, lim))
        return pint(pint(g, (y, 0, x)), (x, 0, lim))
    if x_first:  # y outer from 0 to L, x inner from 0 to y
        return pint(pint(g, (x, 0, y)), (y, 0, lim))
    return pint(pint(g, (y, x, lim)), (x, 0, lim))


def _tri_marginal(g, kind, lim, which):
    """The unnormalised marginal of g over the triangle, as an expression in its own variable."""
    if which == "x":
        return pint(g, (y, 0, x)) if kind == "lower" else pint(g, (y, x, lim))
    return pint(g, (x, y, lim)) if kind == "lower" else pint(g, (x, 0, y))


def _region_text(kind, lim) -> str:
    return f"0 < y < x < {lim}" if kind == "lower" else f"0 < x < y < {lim}"


# -- easy: the constant on a rectangle ----------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a rectangle density with a readable constant"):
        g = rng.choice(_SUMS)
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        total = pint(pint(g, (x, 0, a)), (y, 0, b))
        if total.q > 20 or total.p > 60 or total == 1:
            continue
        break
    c = 1 / total
    other = pint(pint(g, (y, 0, b)), (x, 0, a))
    check(other == total, "the two orders of integration disagree")
    check(pint(pint(c * g, (x, 0, a)), (y, 0, b)) == 1, "c g does not integrate to 1")
    box = pint(pint(g, (x, 0, max(a, b))), (y, 0, max(a, b)))
    cands = [
        (total, "r3k-jointden-forgot-reciprocal"),
        (1 / box if box != total else None, "r3k-jointden-wrong-region-limits"),
        (1 / pint(pint(g, (x, 0, a)), (y, 0, a)) if a != b else None, "r3k-jointden-wrong-region-limits"),
        (1 / (total + 1), "r3k-jointden-integral-plus-one"),
        (
            1 / (a * b * g.subs({x: sympy.Rational(a, 2), y: sympy.Rational(b, 2)})),
            "r3k-jointden-midpoint-for-integral",
        ),
    ]
    dist = need(expr_distractors(c, cands))
    text = (
        f"The continuous random variables X and Y have joint density function f(x, y) = {_g_text(g)} for 0 < x < {a} "
        f"and 0 < y < {b}, and f(x, y) = 0 otherwise. Find the constant c."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[f"The density integrates to 1 over the rectangle, so c × {total} = 1.", f"c = {c}."],
        **expr_fields(c),
    )


# -- medium: a probability over a sub-region of a square -----------------------------------------------------------


def _event_probability(c, g, event, lim):
    if event == "above":  # P(Y > X): x outer 0..L, y from x to L
        first = pint(pint(c * g, (y, x, lim)), (x, 0, lim))
        second = pint(pint(c * g, (x, 0, y)), (y, 0, lim))
    else:  # P(X + Y < L)
        first = pint(pint(c * g, (y, 0, lim - x)), (x, 0, lim))
        second = pint(pint(c * g, (x, 0, lim - y)), (y, 0, lim))
    check(first == second, "the two orders of integration disagree")
    return first


def _medium(rng):
    for _ in attempts(500, "a square density with a readable probability"):
        event = rng.choice(("above", "above", "sum"))
        g = rng.choice(_ASYM if event == "above" else _ANY)
        lim = rng.randint(1, 3)
        total = pint(pint(g, (x, 0, lim)), (y, 0, lim))
        c = 1 / total
        if c.q > 40 or c.p > 12:
            continue
        correct = _event_probability(c, g, event, lim)
        if correct in (0, 1, sympy.Rational(1, 2)) or correct.q > 400:
            continue
        break
    if second_route():
        check(pint(pint(c * g, (x, 0, lim)), (y, 0, lim)) == 1, "c g does not integrate to 1")
        # numerical second route (quadrature of the indicator via mpmath)
        f = sympy.lambdify((x, y), c * g, "mpmath")
        if event == "above":
            num = mpmath.quad(lambda xv: mpmath.quad(lambda yv: f(xv, yv), [xv, lim]), [0, lim])
        else:
            num = mpmath.quad(lambda xv: mpmath.quad(lambda yv: f(xv, yv), [0, lim - xv]), [0, lim])
        check(abs(num - mpmath.mpf(correct.p) / correct.q) < 1e-12, "numerical quadrature disagrees")
    half = sympy.Rational(lim, 2)
    box = pint(pint(c * g, (x, 0, half)), (y, 0, half))
    if event == "above":
        cands = [
            (1 - correct, "r3k-jointden-complement-region"),
            (sympy.Rational(1, 2), "r3k-jointden-assumed-symmetry"),
            (box, "r3k-jointden-event-as-box"),
        ]
        event_text = "Y > X"
    else:
        cands = [
            (1 - correct, "r3k-jointden-complement-region"),
            (box, "r3k-jointden-event-as-box"),
            (correct / 2, "r3k-jointden-halved-the-region"),
        ]
        event_text = f"X + Y < {lim}"
    dist = need(expr_distractors(correct, cands), 2)
    ctext = f"{c.p}/{c.q}" if c.q != 1 else str(c.p)
    text = (
        f"The continuous random variables X and Y have joint density function f(x, y) = {_g_text(g, ctext)} for 0 < x < {lim} "
        f"and 0 < y < {lim}, and f(x, y) = 0 otherwise. Find P({event_text})."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "Integrate the density over the part of the square where the event holds; one limit depends on the other variable.",
            f"The probability is {correct}.",
        ],
        **expr_fields(correct),
    )


# -- hard: a marginal density on a triangle ------------------------------------------------------------------------


def _draw_triangle(rng, *, need_const=True):
    for _ in attempts(800, "a triangle density with a readable constant"):
        kind = rng.choice(("lower", "upper"))
        g = rng.choice(_ANY)
        lim = rng.randint(1, 3)
        total = _tri_double(g, kind, lim, False)
        if total == 0:
            continue
        c = 1 / total
        if c.q > 30 or c.p > 120:
            continue
        which = rng.choice(("x", "y"))
        marg = c * _tri_marginal(g, kind, lim, which)
        var = x if which == "x" else y
        if sympy.expand(marg).is_number or sympy.Poly(marg, var).degree() < 1:
            continue
        if any(abs(t.p) > 200 or t.q > 200 for t in sympy.Poly(marg, var).all_coeffs()):
            continue
        break
    return kind, g, lim, total, c, which, var, marg


def _marginal_second_route(c, g, kind, lim, which, marg, var):
    if second_route():
        check(_tri_double(g, kind, lim, True) == _tri_double(g, kind, lim, False), "the two orders disagree")
        lo, hi = (0, lim)
        check(pint(marg, (var, lo, hi)) == 1, "the marginal does not integrate to 1")
        f = sympy.lambdify((x, y), c * g, "mpmath")
        point = mpmath.mpf(lim) * mpmath.mpf("0.37")
        if which == "x":
            a_, b_ = (0, point) if kind == "lower" else (point, lim)
            num = mpmath.quad(lambda yv: f(point, yv), [a_, b_])
        else:
            a_, b_ = (point, lim) if kind == "lower" else (0, point)
            num = mpmath.quad(lambda xv: f(xv, point), [a_, b_])
        exact = sympy.lambdify(var, marg, "mpmath")(point)
        check(abs(num - exact) < 1e-12, "numerical quadrature disagrees with the marginal")


def _marginal_candidates(c, g, kind, lim, which, marg, var):
    other = "y" if which == "x" else "x"
    full = c * pint(g, (y if which == "x" else x, 0, lim))
    wrong_var = c * _tri_marginal(g, kind, lim, other)
    wrong_var = wrong_var.subs(sympy.Symbol(other), var)
    unnorm = _tri_marginal(g, kind, lim, which)
    cands = [
        (full, "r3k-jointden-marginal-wrong-limits"),
        (wrong_var, "r3k-jointden-marginal-other-variable"),
        (unnorm, "r3k-jointden-marginal-forgot-c"),
    ]
    anti = sympy.Poly(g, y if which == "x" else x).integrate().as_expr()
    upper_only = (anti.subs(y if which == "x" else x, lim)) if (kind == "upper") == (which == "x") else None
    if upper_only is not None:
        cands.append((c * upper_only, "r3k-jointden-marginal-lower-limit-dropped"))
    return cands


def _hard(rng):
    kind, g, lim, total, c, which, var, marg = _draw_triangle(rng)
    _marginal_second_route(c, g, kind, lim, which, marg, var)
    dist = need(expr_distractors(marg, _marginal_candidates(c, g, kind, lim, which, marg, var), symbols=(var,)))
    region = _region_text(kind, lim)
    text = (
        f"The continuous random variables X and Y have joint density function f(x, y) = {_g_text(g)} for {region}, and "
        f"f(x, y) = 0 otherwise, where c is a constant. Find the marginal density f_{which.upper()}({which}) of "
        f"{which.upper()} for 0 < {which} < {lim}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"The density integrates to 1 over the triangle, so c = 1/{total}.",
            f"Integrate out {'y' if which == 'x' else 'x'} over the part of the triangle at a fixed {which}.",
            f"f_{which.upper()}({which}) = {pretty(marg)}.",
        ],
        **expr_fields(marg),
    )


@generator(
    "r3k_joint_density",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_joint_density(rng, tier="easy"):
    """Joint continuous density: the constant, a probability over a region, a marginal; see the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3k_joint_density", level=19, presets={"exam": dict(tier="exam")})
@redraw
@clean_working
def exd_x_r3k_joint_density(rng, tier="exam"):
    """c, a marginal density, a probability and E(Y) for a density on a triangle, the last three following through from c."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(800, "an exam triangle density"):
        kind, g, lim, total, c, which, var, marg = _draw_triangle(rng)
        if lim % 2:  # keep the half-way point whole
            continue
        event = rng.choice(("x_le", "y_ge"))
        half = lim // 2
        # the event region as a pair of nested integrals, both orders
        if event == "x_le":
            if kind == "lower":
                k1 = pint(pint(g, (y, 0, x)), (x, 0, half))
                k2 = pint(pint(g, (x, y, half)), (y, 0, half))
            else:
                k1 = pint(pint(g, (y, x, lim)), (x, 0, half))
                k2 = pint(pint(g, (x, 0, y)), (y, 0, half)) + pint(pint(g, (x, 0, half)), (y, half, lim))
        else:
            if kind == "lower":
                k1 = pint(pint(g, (x, y, lim)), (y, half, lim))
                k2 = pint(pint(g, (y, half, x)), (x, half, lim))
            else:
                k1 = pint(pint(g, (x, 0, y)), (y, half, lim))
                k2 = pint(pint(g, (y, half, lim)), (x, 0, half)) + pint(pint(g, (y, x, lim)), (x, half, lim))
        check(k1 == k2, "the two orders of integration disagree for the event")
        prob = c * k1
        ey_inner = _tri_double(y * g, kind, lim, False)
        ey_check = _tri_double(y * g, kind, lim, True)
        check(ey_inner == ey_check, "E(Y) differs between orders")
        ey = c * ey_inner
        if prob in (0, 1) or prob.q > 300 or ey.q > 300:
            continue
        break
    _marginal_second_route(c, g, kind, lim, which, marg, var)
    check(_tri_double(c * g, kind, lim, True) == 1, "c g does not integrate to 1 over the triangle")
    event_text = f"X ≤ {half}" if event == "x_le" else f"Y ≥ {half}"
    k_c = k1
    m_c = ey_inner
    ey_cands = [
        (ey_inner, "r3k-jointden-forgot-reciprocal"),
        (_tri_double(x * g, kind, lim, False) * c, "r3k-jointden-ex-for-ey"),
        (sympy.Rational(lim, 2), "r3k-jointden-mean-from-the-midpoint"),
    ]
    p_cands = [
        (1 - prob, "r3k-jointden-complement-region"),
        (k_c, "r3k-jointden-forgot-reciprocal"),
        (prob / 2, "r3k-jointden-halved-the-region"),
        (sympy.Rational(1, 2), "r3k-jointden-assumed-symmetry"),
    ]
    c_cands = [
        (total, "r3k-jointden-forgot-reciprocal"),
        (
            1 / _tri_double(g, "upper" if kind == "lower" else "lower", lim, False)
            if _tri_double(g, "upper" if kind == "lower" else "lower", lim, False) != total
            else None,
            "r3k-jointden-swapped-triangle",
        ),
        (1 / pint(pint(g, (x, 0, lim)), (y, 0, lim)), "r3k-jointden-wrong-region-limits"),
    ]
    b_cands = _marginal_candidates(c, g, kind, lim, which, marg, var)
    c_dist = need(expr_distractors(c, c_cands))
    # a reply right by follow-through from a wrong c must never be one of the later parts' own distractors
    wrong_c = [stored(d) for d in c_dist]
    b_dist = need(
        expr_distractors(marg, b_cands, symbols=(var,), avoid=[w * _tri_marginal(g, kind, lim, which) for w in wrong_c])
    )
    p_dist = need(expr_distractors(prob, p_cands, avoid=[w * k_c for w in wrong_c]))
    e_dist = need(expr_distractors(ey, ey_cands, avoid=[w * m_c for w in wrong_c]))
    name = which.upper()
    cs = sympy.Symbol("part_a")
    ft_marg = sympy.sstr(cs * _tri_marginal(g, kind, lim, which)).replace("**", "^")
    text = (
        f"The continuous random variables X and Y have joint density function f(x, y) = {_g_text(g)} for "
        f"{_region_text(kind, lim)}, and f(x, y) = 0 otherwise, where c is a constant."
    )
    return dict(
        text=text,
        parts=[
            dict(
                prompt="Find the value of c.",
                marks=2,
                working=[f"The density integrates to 1 over the triangle, so c × {total} = 1.", f"c = {c}."],
                distractors=c_dist,
                **expr_fields(c),
            ),
            dict(
                prompt=f"Find the marginal density f_{name}({which}) of {name} for 0 < {which} < {lim}.",
                marks=3,
                follow_through=ft_marg,
                working=[f"Integrate out {'y' if which == 'x' else 'x'}: f_{name}({which}) = {pretty(marg)}."],
                distractors=b_dist,
                **expr_fields(marg),
            ),
            dict(
                prompt=f"Find P({event_text}).",
                marks=3,
                sensible_range=None,
                follow_through=f"part_a*({sympy.sstr(k_c)})",
                working=[f"P({event_text}) = c × {k_c} = {prob}."],
                distractors=p_dist,
                **expr_fields(prob),
            ),
            dict(
                prompt="Find E(Y).",
                marks=2,
                follow_through=f"part_a*({sympy.sstr(m_c)})",
                working=[f"E(Y) = c × {m_c} = {ey}."],
                distractors=e_dist,
                **expr_fields(ey),
            ),
        ],
    )
