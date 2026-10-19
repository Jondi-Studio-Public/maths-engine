"""Generates `r3b_integrate_f_prime_over_f` questions (easy, medium, hard) and the exam chain
`r3b_exam_integrate_f_prime_over_f`.

Skill ``integrate-f-prime-over-f`` (``integration``, answer ``expression``): integrals that are recognised by inspection
as ``f'/f`` (giving ``ln f``) or ``f' e^f`` (giving ``e^f``), with no substitution steps written out.

* easy: the numerator is exactly ``f'``: ``2x/(x^2 + c)``, ``k e^(kx)/(e^(kx) + c)`` or ``g' e^g``, where ``f`` or ``g``
  is positive for every x, so no absolute value is needed.
* medium: the numerator is a constant multiple of ``f'`` (``x/(x^2 + c)``, ``a x^3 e^(x^4)``, ``sin x/(c + cos x)``,
  ``cos x e^(sin x)``), so the answer carries a fraction or a minus sign.
* hard: ``f`` can be negative, so the answer needs ``ln(|f|)``: a quadratic or cubic with real roots whose derivative
  is hidden in the numerator, ``tan(kx)`` and ``cot(kx)``, and ``1/(x ln x)``.
* exam (``r3b_exam_integrate_f_prime_over_f``): the gradient function ``a(x + h)/((x + h)^2 + 1)`` with an unknown
  ``a``; (a) ``a`` from a stated gradient, (b) the curve through the point where ``f = 1``, (c) its exact height at
  another x, (d) the gradient at a third x.

The hard tier and the exam ask for a logarithm of an absolute value as ``ln(|u|)``, which the grader reads where it
does not read ``ln|x^2 - 4|``; the stem says so.
"""

from __future__ import annotations

from sympy import Abs, E, Integer, Rational, cos, exp, log, sin, tan

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    Redraw,
    check_antiderivative,
    frac_fields,
    integral_math,
    integral_text,
    key_fields,
    exact_fields,
    part,
    plus,
    pick_expr,
    pick_fractions,
    retry,
    say,
    shown,
    tex,
    x,
)

_FIND = ("Find {I}.", "Determine {I}.", "Find the indefinite integral {I}.")
_BAR_NOTE = " Write the logarithm of an absolute value as ln(|u|)."


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _ln(u):
    return log(Abs(u))


def _quot(num, den) -> Itg:
    """num/den printed as a fraction, whatever SymPy would simplify it to."""
    return Itg(num / den, text=f"({shown(num)})/({shown(den)})", latex=rf"\frac{{{tex(num)}}}{{{tex(den)}}}")


def _result(rng, f: Itg, F, cands, working, suffix: str = ""):
    check_antiderivative(F, f.value)
    text, latex = say(rng, _FIND, {"I": integral_text(f)}, {"I": integral_math(f)}, suffix=suffix)
    return dict(
        text=text, latex=latex, working=working, distractors=pick_expr(F, True, cands, rng=rng), **key_fields(F)
    )


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy_log(rng):
    kind = rng.choice(["power", "exp"])
    if kind == "power":
        n = rng.choice([2, 4])
        c = rng.randint(1, 9)
        f_ = x**n + c
        num = n * x ** (n - 1)
    else:
        k = rng.choice([1, 2, 3])
        c = rng.randint(1, 9)
        f_ = exp(k * x) + c
        num = k * exp(k * x)
    itg = _quot(num, f_)
    F = log(f_)
    cands = [
        Cand(log(num), "r3b-fprime-log-of-numerator"),
        Cand(num / f_ * x, "r3b-fprime-treated-as-constant"),
        Cand(f_ / num, "r3b-fprime-quotient-inverted"),
        Cand(-num / f_**2, "r3b-fprime-differentiated"),
    ]
    return (
        itg,
        F,
        cands,
        [
            f"The numerator is the derivative of the denominator, so the integral is ln of the denominator ({shown(f_)} is always positive)."
        ],
    )


def _easy_exp(rng):
    kind = rng.choice(["x2", "x3", "sin", "x4"])
    g = {"x2": x**2, "x3": x**3, "sin": sin(x), "x4": x**4}[kind]
    gp = g.diff(x)
    itg = Itg(
        gp * exp(g),
        text=f"{shown(gp)}e^({shown(g)})" if gp != 1 else f"e^({shown(g)})",
        latex=rf"{tex(gp)}\, e^{{{tex(g)}}}",
    )
    F = exp(g)
    cands = [
        Cand(exp(g) / gp, "r3b-fprime-exp-divided-by-derivative"),
        Cand(exp(g + 1) / (g + 1), "r3b-exp-integral-power-rule"),
        Cand(gp * exp(g) * x, "r3b-fprime-treated-as-constant"),
        Cand(gp * exp(g), "r3b-fprime-not-integrated"),
    ]
    return (
        itg,
        F,
        cands,
        [f"The factor {shown(gp)} is the derivative of {shown(g)}, so the integral is e^({shown(g)})."],
    )


def _easy(rng):
    f, F, cands, working = rng.choice([_easy_log, _easy_exp])(rng)
    return _result(rng, f, F, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium_log(rng):
    kind = rng.choice(["power", "cos", "sin", "exp"])
    a = _nonzero(rng, -9, 9)
    c = rng.randint(2, 9)
    if kind == "power":
        n = rng.choice([2, 4])
        f_, num, mult = x**n + c, a * x ** (n - 1), Rational(a, n)
    elif kind == "cos":  # a sin x / (c + cos x), and c + cos x > 0 because c >= 2
        f_, num, mult = c + cos(x), a * sin(x), Integer(-a)
    elif kind == "sin":
        f_, num, mult = c + sin(x), a * cos(x), Integer(a)
    else:
        k = rng.choice([2, 3])
        f_, num, mult = exp(k * x) + c, a * exp(k * x), Rational(a, k)
    check((num - mult * f_.diff(x)).simplify() == 0, "the numerator is not the stated multiple of f'")
    F = mult * log(f_)
    cands = [
        Cand(a * log(f_), "r3b-fprime-multiple-not-adjusted"),
        Cand(log(f_) / mult, "r3b-fprime-multiple-inverted"),
        Cand(-mult * log(f_), "r3b-fprime-sign-lost"),
        Cand(log(num), "r3b-fprime-log-of-numerator"),
    ]
    working = [
        f"The numerator is {shown(mult)} times the derivative of the denominator, so the integral is {shown(mult)}ln({shown(f_)})."
    ]
    return _quot(num, f_), F, cands, working


def _medium_exp(rng):
    kind = rng.choice(["x4", "x3", "sin", "cos"])
    a = _nonzero(rng, -9, 9)
    g, num, mult = {
        "x4": (x**4, a * x**3, Rational(a, 4)),
        "x3": (x**3, a * x**2, Rational(a, 3)),
        "sin": (sin(x), a * cos(x), Integer(a)),
        "cos": (cos(x), a * sin(x), Integer(-a)),
    }[kind]
    check((num - mult * g.diff(x)).simplify() == 0, "the factor is not the stated multiple of g'")
    itg = Itg(num * exp(g), text=f"{shown(num)}e^({shown(g)})", latex=rf"{tex(num)}\, e^{{{tex(g)}}}")
    F = mult * exp(g)
    cands = [
        Cand(a * exp(g), "r3b-fprime-multiple-not-adjusted"),
        Cand(exp(g) / mult, "r3b-fprime-multiple-inverted"),
        Cand(-mult * exp(g), "r3b-fprime-sign-lost"),
        Cand(num * exp(g), "r3b-fprime-not-integrated"),
    ]
    return (
        itg,
        F,
        cands,
        [
            f"{shown(num)} is {shown(mult)} times {shown(g.diff(x))}, the derivative of {shown(g)}; the integral is {shown(mult)}e^({shown(g)})."
        ],
    )


def _medium(rng):
    f, F, cands, working = rng.choice([_medium_log, _medium_exp])(rng)
    return _result(rng, f, F, cands, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_quadratic(rng):
    for _ in attempts(200, "a quadratic with real roots and a non-trivial multiple"):
        r1, r2 = rng.sample(range(-5, 6), 2)
        b, c = -(r1 + r2), r1 * r2
        m = _nonzero(rng, -4, 4)
        if b != 0 and c != 0 and r1 != 0 and r2 != 0:
            break
    f_ = x**2 + b * x + c
    fp = 2 * x + b
    num = m * fp
    # the numerator is printed expanded, so the multiple has to be seen: (2m)x + m b
    itg = _quot(num.expand(), f_)
    F = m * _ln(f_)
    cands = [
        Cand(2 * m * _ln(f_), "r3b-fprime-multiple-not-adjusted"),
        Cand(m * log(f_), "r3b-recip-abs-dropped"),
        Cand(Rational(m, 2) * _ln(f_), "r3b-fprime-multiple-inverted"),
        Cand(_ln(num.expand()), "r3b-fprime-log-of-numerator"),
    ]
    return itg, F, cands, [f"The derivative of the denominator is {shown(fp)}; the numerator is {m} times that."]


def _hard_cubic(rng):
    a = rng.choice([1, 2, 3, -1, -2])
    c = _nonzero(rng, -6, 6)
    m = rng.choice([1, 2, 3, -1, -2])
    f_ = x**3 + a * x + c
    fp = 3 * x**2 + a
    num = (m * fp).expand()
    itg = _quot(num, f_)
    F = m * _ln(f_)
    cands = [
        Cand(m * log(f_), "r3b-recip-abs-dropped"),
        Cand(3 * m * _ln(f_), "r3b-fprime-multiple-not-adjusted"),
        Cand(_ln(num), "r3b-fprime-log-of-numerator"),
        Cand(Rational(m, 3) * _ln(f_), "r3b-fprime-multiple-inverted"),
    ]
    return itg, F, cands, [f"The numerator is {m} times {shown(fp)}, the derivative of the denominator."]


def _coef(a) -> str:
    return "" if a == 1 else "-" if a == -1 else str(a)


def _hard_trig(rng):
    k = rng.choice([1, 2, 3])
    which = rng.choice(["tan", "cot"])
    a = _nonzero(rng, -6, 6)
    kx = "x" if k == 1 else f"{k}x"
    c = _coef(a)
    if which == "tan":
        itg = Itg(a * tan(k * x), text=f"{c}tan({kx})", latex=rf"{c}\tan {kx}")
        F = -Rational(a, k) * _ln(cos(k * x))
        cands = [
            Cand(Rational(a, k) * _ln(cos(k * x)), "r3b-fprime-sign-lost"),
            Cand(Rational(a, k) * _ln(sin(k * x)), "r3b-fprime-trig-ratio-confused"),
            Cand(-a * _ln(cos(k * x)), "r3b-fprime-multiple-not-adjusted"),
            Cand(-Rational(a, k) * log(cos(k * x)), "r3b-recip-abs-dropped"),
        ]
        working = [f"tan({k}x) = sin({k}x)/cos({k}x), and the derivative of cos({k}x) is -{k}sin({k}x)."]
    else:
        itg = Itg(
            a * cos(k * x) / sin(k * x), text=f"{c}cos({kx})/sin({kx})", latex=rf"\frac{{{c}\cos {kx}}}{{\sin {kx}}}"
        )
        F = Rational(a, k) * _ln(sin(k * x))
        cands = [
            Cand(-Rational(a, k) * _ln(sin(k * x)), "r3b-fprime-sign-lost"),
            Cand(Rational(a, k) * _ln(cos(k * x)), "r3b-fprime-trig-ratio-confused"),
            Cand(a * _ln(sin(k * x)), "r3b-fprime-multiple-not-adjusted"),
            Cand(Rational(a, k) * log(sin(k * x)), "r3b-recip-abs-dropped"),
        ]
        working = [f"The derivative of sin({k}x) is {k}cos({k}x)."]
    return itg, F, cands, working


def _hard_log_log(rng):
    a = _nonzero(rng, -8, 8)
    itg = Itg(a / (x * log(x)), text=f"{a}/(x ln(x))", latex=rf"\frac{{{a}}}{{x \ln x}}")
    F = a * _ln(log(x))
    cands = [
        Cand(a * log(log(x)), "r3b-recip-abs-dropped"),
        Cand(a * _ln(x) * _ln(log(x)), "r3b-fprime-log-of-product"),
        Cand(a * _ln(x), "r3b-fprime-inner-function-ignored"),
        Cand(-a / (x * log(x)) ** 2, "r3b-fprime-differentiated"),
    ]
    return itg, F, cands, ["The derivative of ln x is 1/x, so 1/(x ln x) is f'/f with f = ln x."]


def _hard(rng):
    f, F, cands, working = rng.choice([_hard_quadratic, _hard_cubic, _hard_trig, _hard_log_log])(rng)
    return _result(rng, f, F, cands, working, suffix=_BAR_NOTE)


@generator(
    "r3b_integrate_f_prime_over_f",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_integrate_f_prime_over_f(rng, tier="easy"):
    """Integrals recognised as f'/f or f' e^f; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_integrate_f_prime_over_f", level=15, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_integrate_f_prime_over_f(rng, tier="exam"):
    """Exam chain: an unknown multiple in the numerator, the curve, an exact height and a gradient."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "exam constants"):
        h = rng.choice([1, 2, 3])
        a = _nonzero(rng, -8, 8)
        x0 = rng.choice([-h + 1, -h + 2, -h - 1, -h + 3])
        s2 = rng.choice([1, 2, 3, -1, -2])
        x3 = rng.choice([-h + 1, -h - 2, -h + 2, -h + 4])
        y0 = _nonzero(rng, -9, 9)
        if len({x0 + h, s2, x3 + h}) == 3:
            break
    fx = x**2 + 2 * h * x + h**2 + 1
    f_prime = a * (x + h) / fx
    F = Rational(a, 2) * log(fx) + y0
    check_antiderivative(F, f_prime)
    n0 = (x0 + h) ** 2 + 1
    g = f_prime.subs(x, x0)
    x2 = -h + s2
    n2 = s2**2 + 1
    height = F.subs(x, x2)
    n3 = (x3 + h) ** 2 + 1
    grad3 = f_prime.subs(x, x3)
    check(F.subs(x, -h) == y0 and Rational(a * (x0 + h), n0) == g, "the stated conditions do not hold")
    check(height == Rational(a, 2) * log(n2) + y0, "the exact height differs from the logarithm of the stated value")
    den = f"x^2 + {2 * h}x + {h * h + 1}"
    stem = (
        f"A curve y = f(x) has gradient function f'(x) = a(x + {h})/({den}), where a is a constant. "
        f"The gradient of the curve at x = {x0} is {g}, and the curve passes through the point ({-h}, {y0})."
    )
    stem_latex = (
        rf"A curve \(y = f(x)\) has gradient function \(f'(x) = \frac{{a(x + {h})}}{{{den}}}\), where \(a\) is a constant. "
        rf"The gradient of the curve at \(x = {x0}\) is \({tex(g)}\), and the curve passes through the point \(({-h}, {y0})\)."
    )
    ratio = Rational(n0, x0 + h)
    pa = part(
        "Find the value of a.",
        "Find the value of \\(a\\).",
        1,
        [f"f'({x0}) = a({x0 + h})/{n0} = {g}, so a = {a}."],
        frac_fields(a),
        pick_fractions(
            a,
            [
                (g * n0, "r3b-chain-factor-not-divided-out"),
                (g * ratio * 2, "r3b-fprime-multiple-not-adjusted"),
                (g * n0 / x0 if x0 else None, "r3b-chain-substitution-slip"),
                (g, "r3b-chain-denominator-ignored"),
            ],
            rng=rng,
        ),
    )
    pb = part(
        "Find f(x).",
        "Find \\(f(x)\\).",
        3,
        [
            f"The numerator is (a/2) times the derivative of the denominator, so f(x) = (a/2) ln({den}) + c.",
            f"At x = {-h} the denominator is 1 and ln 1 = 0, so c = {y0}.",
        ],
        key_fields(F, const=False),
        pick_expr(
            F,
            False,
            [
                Cand(a * log(fx) + y0, "r3b-fprime-multiple-not-adjusted", const=False),
                Cand(Rational(a, 2) * log(fx), "r3b-chain-point-not-used", const=False),
                Cand(Rational(a, 2) * log(fx) - y0, "r3b-chain-constant-sign", const=False),
                Cand(Rational(a, 2) * log(a * (x + h)) + y0, "r3b-fprime-log-of-numerator", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/2*log(x**2+{2 * h}*x+{h * h + 1})+({y0})",
    )
    pc = part(
        f"Find the exact value of f({x2}).",
        f"Find the exact value of \\(f({x2})\\).",
        2,
        [f"At x = {x2} the denominator is {n2}, so f({x2}) = ({Rational(a, 2)}) ln {n2} {plus(y0)}."],
        exact_fields(height),
        pick_expr(
            height,
            False,
            [
                Cand(Rational(a, 2) * log(n2), "r3b-chain-constant-dropped", const=False),
                Cand(a * log(n2) + y0, "r3b-fprime-multiple-not-adjusted", const=False),
                Cand(
                    Rational(a, 2) * log(s2**2) + y0 if s2**2 > 1 else None, "r3b-chain-substitution-slip", const=False
                ),
                Cand(Rational(a, 2) * log(n2 + 1) + y0, "r3b-chain-substitution-slip", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/2*log({n2})+({y0})",
    )
    pd = part(
        f"Find the gradient of the curve at x = {x3}.",
        f"Find the gradient of the curve at \\(x = {x3}\\).",
        2,
        [f"f'({x3}) = {a}({x3 + h})/{n3} = {grad3}."],
        frac_fields(grad3),
        pick_fractions(
            grad3,
            [
                (Rational(a * (x3 + h), (x3 + h) ** 2), "r3b-chain-denominator-constant-dropped"),
                (Rational(a * x3, n3), "r3b-chain-numerator-shift-dropped"),
                (Rational(a * (x3 + h), n3 - 1) if n3 > 2 else None, "r3b-chain-denominator-constant-dropped"),
                (-grad3, "r3b-chain-sign-slip"),
            ],
            rng=rng,
        ),
        follow_through=f"part_a*({x3 + h})/{n3}",
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
