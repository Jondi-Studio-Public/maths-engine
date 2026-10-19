"""Exam chains for the three substitution skills of `w6_substitution_build`.

* ``exd_x_w6_subst_rewrite_x`` (``integrate-substitution-rewrite-x``): a definite integral ``∫ x f(x + a) dx`` from 0 to b
  with u = x + a: (a) x in terms of u, (b) the integrand as a function of u, (c) the upper limit of u, (d) the exact value.
* ``exd_x_w6_subst_exp_trig`` (``integrate-substitution-exponential-trigonometric``): either
  ``∫ c sec^k(x) tan^n(x) dx`` from 0 to an angle with u = tan x, or ``∫ c e^(2x) √(e^x + a) dx`` with u = e^x + a:
  (a) the upper limit of u, (b) the integrand in u, (c) the exact value.
* ``exd_x_w6_subst_definite`` (``definite-integral-substitution-changed-limits``): ``√(a² - x²)`` or ``x²/√(a² - x²)`` from
  0 to a sin α with x = a sin θ: (a) the upper limit of θ in degrees, (b) the constant k in ``k ∫ (1 ± cos 2θ) dθ``,
  (c) the exact value, following through from (a) and (b).

Every chain is built from the u-integral backwards and checked against Simpson's rule on the original integral.
"""

from __future__ import annotations

import sympy
from sympy import Rational as R
from sympy import Symbol, cos, exp, pi, sec, sin, sqrt, tan

from ..answers import Distractor
from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    Redraw,
    canon,
    check_definite,
    exact_fields,
    int_fields,
    integral_math,
    integral_text,
    key_fields,
    part,
    pick_expr,
    pick_ints,
    retry,
    shown,
    tex,
    x,
)

u = Symbol("u")


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _same(a, b) -> bool:
    return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0


def pick_in_u(key, cands, count: int = 3, minimum: int = 2, rng=None) -> list[Distractor]:
    """Wrong expressions in u: different from the key and from each other (an exact symbolic test)."""
    kept: list[tuple] = []
    cands = list(cands) if rng is None else rng.sample(list(cands), len(list(cands)))
    for expr, tag in cands:
        if expr is None:
            continue
        expr = sympy.sympify(expr)
        if _same(expr, key) or any(_same(expr, k) for k, _ in kept):
            continue
        kept.append((expr, tag))
        if len(kept) == count:
            break
    if len(kept) < minimum:
        raise Redraw("fewer than two distinct wrong expressions in u")
    out = []
    for expr, tag in kept:
        text = canon(expr)
        out.append(Distractor(text, tag, text))
    return out


def _nc(cands):
    """Wrong exact values carry no constant of integration."""
    return [Cand(c.expr, c.tag, const=False) for c in cands if c is not None]


def _stem(rng, patterns, f, lo, hi, hint):
    ts = dict(I=integral_text(f, lo, hi), H=hint[0])
    ms = dict(I=integral_math(f, lo, hi), H=hint[1])
    pattern = patterns[rng.randrange(len(patterns))]
    return pattern.format(**ts), latex_from_inline(pattern.format(**ms))


def _u_hint(u_expr):
    return f"u = {shown(u_expr)}", rf"\(u = {tex(u_expr)}\)"


# =====================================================================================================================
# rewrite x
# =====================================================================================================================

_REWRITE_STEMS = (
    "Let I be the value of {I}. Use the substitution {H}.",
    "Consider the integral I = {I}, and use the substitution {H}.",
)


@generator("exd_x_w6_subst_rewrite_x", level=15, presets={"exam": dict(tier="exam")})
@retry
def exd_x_w6_subst_rewrite_x(rng, tier="exam"):
    """x in terms of u, the integrand in u, the new upper limit and the exact value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    kind = _pick(rng, ("sqrt", "invsqrt", "linear", "inverse"))
    c = _pick(rng, (1, 1, 2, 3, 4))
    cf = "" if c == 1 else str(c)
    if kind in ("sqrt", "invsqrt"):
        for _ in attempts(100, "perfect-square limits"):
            s1, s2 = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3, 4, 5, 6))
            a = rng.randint(1, s1 * s1)
            if s2 > s1:
                break
        hi = s2 * s2 - a
        p = R(1, 2) if kind == "sqrt" else R(-1, 2)
        text = f"{cf}x√(x + {a})" if kind == "sqrt" else f"{cf}x/√(x + {a})"
        lower_u = s1 * s1
        lo = lower_u - a
    else:
        a = _pick(rng, (1, 2, 3, 4))
        n = _pick(rng, (2, 3, 4)) if kind == "linear" else _pick(rng, (3, 4))
        p = n if kind == "linear" else -n
        hi = _pick(rng, (1, 2, 3, 4)) if kind == "linear" else _pick(rng, (1, 2, 3, 5))
        text = f"{cf}x(x + {a})^{n}" if kind == "linear" else f"{cf}x/(x + {a})^{n}"
        lo = 0
        lower_u = a
    upper_u = hi + a
    f = Itg(c * x * (x + a) ** p, text=text)
    g = sympy.expand(c * (u - a) * u**p)
    G = lambda t: sympy.integrate(g, u).subs(u, t)  # noqa: E731
    value = sympy.nsimplify(G(upper_u) - G(lower_u))
    check_definite(value, f.value, lo, hi)
    stem, stem_latex = _stem(rng, _REWRITE_STEMS, f, lo, hi, _u_hint(x + a))
    wrong_x = lambda t: sympy.integrate(sympy.expand(c * u * u**p), u).subs(u, t)  # noqa: E731
    pa = part(
        "Write x in terms of u.",
        "Write x in terms of u.",
        1,
        [f"u = x + {a}, so x = u - {a}."],
        key_fields(u - a, const=False),
        pick_in_u(
            u - a,
            [
                (u + a, "w6-subst-shift-sign-wrong"),
                (u, "w6-subst-x-replaced-by-u"),
                (a - u, "w6-subst-x-from-u-reversed"),
            ],
            rng=rng,
        ),
    )
    pb = part(
        "Show that I can be written as the integral of g(u) du between the limits of u, and find g(u).",
        "Show that \\(I\\) can be written as the integral of \\(g(u)\\,\\mathrm{d}u\\) between the limits of \\(u\\), and find \\(g(u)\\).",
        2,
        [f"x = u - {a} and dx = du, so the integrand becomes {cf}(u - {a}) times a power of u."],
        key_fields(g, const=False),
        pick_in_u(
            g,
            [
                (sympy.expand(c * (u + a) * u**p), "w6-subst-shift-sign-wrong"),
                (sympy.expand(c * u * u**p), "w6-subst-x-replaced-by-u"),
                (sympy.expand(c * (u - a)), "w6-subst-power-of-u-dropped"),
                (sympy.expand(c * (u - a) * u ** (p + 1)), "w6-subst-exponent-raised-twice"),
            ],
            rng=rng,
        ),
    )
    pc = part(
        "Find the value of u at the upper limit of the integral.",
        "Find the value of \\(u\\) at the upper limit of the integral.",
        1,
        [f"When x = {hi}, u = {hi} + {a} = {upper_u}."],
        int_fields(upper_u),
        pick_ints(
            upper_u,
            [
                (hi, "w6-subst-limits-not-changed"),
                (hi - a, "w6-subst-limit-shift-sign-wrong"),
                (lower_u, "w6-subst-lower-limit-for-upper"),
            ],
            rng=rng,
        ),
        sensible_range=(0, None),
    )
    pd = part(
        "Hence find the exact value of I.",
        "Hence find the exact value of \\(I\\).",
        3,
        [f"I is the integral of g(u) from u = {lower_u} to u = {upper_u}, which is {value}."],
        exact_fields(value),
        pick_expr(
            value,
            False,
            _nc(
                [
                    Cand(G(hi) - G(lo), "w6-subst-limits-not-changed"),
                    Cand(G(upper_u), "w6-subst-lower-limit-dropped"),
                    Cand(-value, "w6-subst-limits-swapped"),
                    Cand(wrong_x(upper_u) - wrong_x(lower_u), "w6-subst-x-replaced-by-u"),
                ]
            ),
            rng=rng,
        ),
    )
    return dict(text=stem, latex=stem_latex, parts=[pa, pb, pc, pd])


# =====================================================================================================================
# exponential and trigonometric
# =====================================================================================================================

_ET_STEMS = (
    "Let I be the value of {I}. Use the substitution {H}.",
    "Consider the integral I = {I}, and use the substitution {H}.",
)


def _exp_chain(rng):
    c = _pick(rng, (1, 2, 3, 4))
    cf = "" if c == 1 else str(c)
    for _ in attempts(100, "perfect-square limits"):
        s1, s2 = _pick(rng, (2, 3)), _pick(rng, (3, 4, 5))
        if s2 > s1:
            break
    a = s1 * s1 - 1
    b = s2 * s2 - a
    f = Itg(c * exp(2 * x) * sqrt(exp(x) + a), text=f"{cf}e^(2x)√(e^x + {a})")
    g = sympy.expand(c * (u - a) * sqrt(u))
    G = lambda t: sympy.integrate(g, u).subs(u, t)  # noqa: E731
    u1, u2 = s1 * s1, s2 * s2
    value = sympy.nsimplify(G(u2) - G(u1))
    hi = sympy.log(b)
    check_definite(value, f.value, 0, hi)
    stem, stem_latex = _stem(rng, _ET_STEMS, f, 0, hi, _u_hint(exp(x) + a))
    pa = part(
        "Find the value of u at the upper limit of the integral.",
        "Find the value of \\(u\\) at the upper limit of the integral.",
        1,
        [f"When x = ln {b}, u = e^(ln {b}) + {a} = {u2}."],
        int_fields(u2),
        pick_ints(
            u2,
            [
                (b, "w6-subst-limit-shift-forgotten"),
                (b - a, "w6-subst-limit-shift-sign-wrong"),
                (u1, "w6-subst-lower-limit-for-upper"),
            ],
            rng=rng,
        ),
        sensible_range=(0, None),
    )
    pb = part(
        "Show that I can be written as the integral of g(u) du between the limits of u, and find g(u).",
        "Show that \\(I\\) can be written as the integral of \\(g(u)\\,\\mathrm{d}u\\) between the limits of \\(u\\), and find \\(g(u)\\).",
        2,
        [f"e^x = u - {a} and du = e^x dx, so e^(2x) dx = (u - {a}) du."],
        key_fields(g, const=False),
        pick_in_u(
            g,
            [
                (sympy.expand(c * (u + a) * sqrt(u)), "w6-subst-shift-sign-wrong"),
                (sympy.expand(c * u * sqrt(u)), "w6-subst-x-replaced-by-u"),
                (sympy.expand(c * sqrt(u)), "w6-subst-odd-power-not-split"),
                (sympy.expand(c * (u - a) * u), "w6-subst-root-dropped"),
            ],
            rng=rng,
        ),
    )
    pc = part(
        "Hence find the exact value of I.",
        "Hence find the exact value of \\(I\\).",
        3,
        [f"I is the integral of g(u) from u = {u1} to u = {u2}, which is {value}."],
        exact_fields(value),
        pick_expr(
            value,
            False,
            _nc(
                [
                    Cand(G(b) - G(1), "w6-subst-limit-shift-forgotten"),
                    Cand(G(u2), "w6-subst-lower-limit-dropped"),
                    Cand(-value, "w6-subst-limits-swapped"),
                    Cand(
                        sympy.integrate(sympy.expand(c * u * sqrt(u)), u).subs(u, u2)
                        - sympy.integrate(sympy.expand(c * u * sqrt(u)), u).subs(u, u1),
                        "w6-subst-x-replaced-by-u",
                    ),
                ]
            ),
            rng=rng,
        ),
    )
    return dict(text=stem, latex=stem_latex, parts=[pa, pb, pc])


def _tan_chain(rng):
    c = _pick(rng, (1, 1, 2, 3, 4))
    cf = "" if c == 1 else str(c)
    k = _pick(rng, (4, 6))
    n = _pick(rng, (1, 2, 3))
    alpha, u_hi = _pick(rng, ((pi / 4, sympy.Integer(1)), (pi / 3, sqrt(3)), (pi / 6, sqrt(3) / 3)))
    tail = "tan(x)" if n == 1 else f"tan^{n}(x)"
    f = Itg(c * sec(x) ** k * tan(x) ** n, text=f"{cf}sec^{k}(x){tail}")
    g = sympy.expand(c * u**n * (1 + u**2) ** ((k - 2) // 2))
    G = lambda t: sympy.integrate(g, u).subs(u, t)  # noqa: E731
    value = sympy.simplify(G(u_hi))
    check_definite(value, f.value, 0, alpha)
    stem, stem_latex = _stem(rng, _ET_STEMS, f, 0, alpha, _u_hint(tan(x)))
    wrong_g = [
        (sympy.expand(c * u**n * (1 - u**2) ** ((k - 2) // 2)), "w6-subst-identity-sign"),
        (sympy.expand(c * u**n), "w6-subst-identity-term-dropped"),
        (sympy.expand(c * u**n * (1 + u**4) ** ((k - 2) // 2)), "w6-subst-square-binomial-expanded-wrong"),
        (sympy.expand(c * u**n * (1 + u**2)), "w6-subst-identity-power-not-raised")
        if k == 6
        else (sympy.expand(c * u ** (n + 1)), "w6-subst-identity-term-dropped"),
    ]
    pa = part(
        "Find the value of u at the upper limit of the integral.",
        "Find the value of \\(u\\) at the upper limit of the integral.",
        1,
        [f"When x = {shown(alpha)}, u = tan({shown(alpha)}) = {shown(u_hi)}."],
        exact_fields(u_hi),
        pick_expr(
            u_hi,
            False,
            _nc(
                [
                    Cand(alpha, "w6-subst-limits-not-changed"),
                    Cand(sec(alpha), "w6-subst-wrong-inner-function"),
                    Cand(sin(alpha), "w6-subst-wrong-inner-function"),
                    Cand(1 / u_hi if u_hi != 1 else sympy.Integer(0), "w6-subst-wrong-inner-function"),
                ]
            ),
            rng=rng,
        ),
    )
    pb = part(
        "Show that I can be written as the integral of g(u) du between the limits of u, and find g(u).",
        "Show that \\(I\\) can be written as the integral of \\(g(u)\\,\\mathrm{d}u\\) between the limits of \\(u\\), and find \\(g(u)\\).",
        2,
        [
            f"Write sec^{k} x = sec^2 x (1 + tan^2 x)"
            + ("^2" if k == 6 else "")
            + ", so with u = tan x and du = sec^2 x dx the integrand is a polynomial in u."
        ],
        key_fields(g, const=False),
        pick_in_u(g, wrong_g, rng=rng),
    )
    pc = part(
        "Hence find the exact value of I.",
        "Hence find the exact value of \\(I\\).",
        3,
        [f"I is the integral of g(u) from 0 to {shown(u_hi)}, which is {shown(value)}."],
        exact_fields(value),
        pick_expr(
            value,
            False,
            _nc(
                [
                    Cand(G(alpha) - G(0), "w6-subst-limits-not-changed"),
                    Cand(c * ((u_hi ** (n + 1)) / (n + 1)), "w6-subst-identity-term-dropped"),
                    Cand(-value, "w6-subst-limits-swapped"),
                    Cand(sympy.integrate(wrong_g[0][0], u).subs(u, u_hi), "w6-subst-identity-sign"),
                ]
            ),
            rng=rng,
        ),
    )
    return dict(text=stem, latex=stem_latex, parts=[pa, pb, pc])


@generator("exd_x_w6_subst_exp_trig", level=15, presets={"exam": dict(tier="exam")})
@retry
def exd_x_w6_subst_exp_trig(rng, tier="exam"):
    """The new upper limit, the integrand in u and the exact value, for an exponential or a trigonometric integral."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _pick(rng, (_exp_chain, _tan_chain, _tan_chain))(rng)


# =====================================================================================================================
# trigonometric substitution
# =====================================================================================================================

_TRIG_STEMS = (
    "Let I be the value of {I}. Use the substitution {H}.",
    "Consider the integral I = {I}, and use the substitution {H}.",
)


@generator("exd_x_w6_subst_definite", level=15, presets={"exam": dict(tier="exam")})
@retry
def exd_x_w6_subst_definite(rng, tier="exam"):
    """The upper limit of θ, the constant in front of the integral of 1 ± cos 2θ, and the exact value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a = _pick(rng, (2, 4, 6, 8))
    degrees = _pick(rng, (30, 45, 60))
    alpha = pi * degrees / 180
    hi = sympy.simplify(a * sin(alpha))
    even_root = rng.random() < 0.5
    root = sqrt(a * a - x * x)
    if even_root:
        f = Itg(root, text=f"√({a * a} - x^2)")
        sign = 1
    else:
        f = Itg(x**2 / root, text=f"x^2/√({a * a} - x^2)")
        sign = -1
    k = R(a * a, 2)
    value = sympy.simplify(k * (alpha + sign * sin(alpha) * cos(alpha)))
    check_definite(value, f.value, 0, hi)
    hint = (f"x = {a}sin(θ)", rf"\(x = {a}\sin\theta\)")
    stem, stem_latex = _stem(rng, _TRIG_STEMS, f, 0, hi, hint)
    form = "cos 2θ"
    pa = part(
        "Find the upper limit of θ, in degrees, for the integral in θ.",
        "Find the upper limit of \\(\\theta\\), in degrees, for the integral in \\(\\theta\\).",
        1,
        [f"When x = {shown(hi)}, sin θ = {shown(hi)}/{a}, so θ = {degrees}°."],
        int_fields(degrees),
        pick_ints(
            degrees,
            [
                (90 - degrees, "w6-subst-complementary-angle"),
                (180 - degrees if degrees != 90 else None, "w6-subst-supplementary-angle"),
                (2 * degrees, "w6-subst-double-angle-limit"),
                (degrees // 2, "w6-subst-half-angle-limit"),
            ],
            rng=rng,
        ),
        sensible_range=(0, 90),
    )
    pb = part(
        f"Show that I can be written as k times the integral of (1 {'+' if sign == 1 else '-'} {form}) dθ from 0 to the upper limit, and find k.",
        f"Show that \\(I\\) can be written as \\(k\\) times the integral of \\(1 {'+' if sign == 1 else '-'} \\cos 2\\theta\\) with respect to \\(\\theta\\) from \\(0\\) to the upper limit, and find \\(k\\).",
        2,
        [
            f"dx = {a} cos θ dθ and √({a * a} - x^2) = {a} cos θ, so the integrand becomes {a * a} {'cos' if sign == 1 else 'sin'}^2 θ.",
            f"Using {'cos' if sign == 1 else 'sin'}^2 θ = (1 {'+' if sign == 1 else '-'} cos 2θ)/2 gives k = {a * a}/2 = {k}.",
        ],
        int_fields(k),
        pick_ints(
            k,
            [
                (a * a, "w6-subst-double-angle-factor-dropped"),
                (a, "w6-subst-amplitude-not-squared"),
                (a * a // 4, "w6-subst-double-angle-factor-doubled"),
                (a // 2, "w6-subst-amplitude-halved"),
            ],
            rng=rng,
        ),
        sensible_range=(0, None),
    )
    angle = "part_a*pi/180"
    ft = f"part_b*({angle} {'+' if sign == 1 else '-'} sin(2*{angle})/2)"
    pc = part(
        "Hence find the exact value of I.",
        "Hence find the exact value of \\(I\\).",
        3,
        [f"I = {k}[θ {'+' if sign == 1 else '-'} (sin 2θ)/2] from 0 to {shown(alpha)}, which is {shown(value)}."],
        exact_fields(value),
        pick_expr(
            value,
            False,
            _nc(
                [
                    Cand(k * (alpha - sign * sin(alpha) * cos(alpha)), "w6-subst-identity-sign"),
                    Cand(k * alpha, "w6-subst-identity-term-dropped"),
                    Cand(k * (alpha + sign * sin(2 * alpha)), "w6-subst-double-angle-factor-dropped"),
                    Cand(k * (hi + sign * sin(alpha) * cos(alpha)), "w6-subst-limits-not-changed"),
                ]
            ),
            rng=rng,
        ),
        follow_through=ft,
    )
    return dict(text=stem, latex=stem_latex, parts=[pa, pb, pc])
