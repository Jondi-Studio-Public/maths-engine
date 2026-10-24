"""Generates `r3b_integrate_reciprocal` questions (easy, medium, hard) and the exam chain
`r3b_exam_integrate_reciprocal`.

Skill ``integrate-reciprocal-linear`` (``integration``, answer ``expression``): antiderivatives that are logarithms
of a linear expression, typed with the absolute value and the constant of integration.

* easy: ``a/x`` or ``1/(x + b)``, where the answer is ``a ln|x| + C`` or ``ln|x + b| + C``.
* medium: ``a/(px + q)`` with ``p`` a whole number other than 1 (negative as often as positive), so the coefficient
  of ``x`` divides the answer and, when negative, changes its sign.
* hard: the integrand is not yet a sum of reciprocals: ``(px + c)/x``, ``(x^2 + c)/x``, ``(px + c)/(x + b)`` which
  needs a division first, or a sum of two reciprocals with a power term.
* exam (``r3b_exam_integrate_reciprocal``): the gradient function ``k/(px + q)`` with an unknown ``k``; (a) ``k`` from
  a stated gradient, (b) the curve through a point where ``px + q = 1``, (c) the exact height at another x, (d) the x
  where the gradient takes a stated value.

Every antiderivative is differentiated back and compared with the integrand at five points.
"""

from __future__ import annotations

from sympy import Abs, Rational, log

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    check_antiderivative,
    exact_fields,
    frac_fields,
    integral_math,
    integral_text,
    key_fields,
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

_FIND = (
    "Find {I}.",
    "Determine {I}.",
    "Find the indefinite integral {I}.",
)


def _cx(c) -> str:
    """A coefficient as written before x: 1 is left out and -1 is a bare minus sign."""
    return "" if c == 1 else "-" if c == -1 else str(c)


def _ln(u):
    return log(Abs(u))


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _result(rng, f: Itg, F, cands, working):
    check_antiderivative(F, f.value)
    text, latex = say(rng, _FIND, {"I": integral_text(f)}, {"I": integral_math(f)})
    return dict(
        text=text, latex=latex, working=working, distractors=pick_expr(F, True, cands, rng=rng), **key_fields(F)
    )


def _frac_text(num, den: str) -> str:
    return f"{num}/{den}"


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    if rng.random() < 0.6:
        a = _nonzero(rng, -9, 9)
        f = Itg(a / x, text=f"{a}/x", latex=rf"\frac{{{a}}}{{x}}")
        F = a * _ln(x)
        cands = [
            Cand(a * log(x), "r3b-recip-abs-dropped"),
            Cand(_ln(a * x), "r3b-recip-coefficient-inside-log"),
            Cand(_ln(x), "r3b-recip-coefficient-dropped"),
            Cand(-a / x**2, "r3b-recip-differentiated"),
        ]
        working = [f"The integral of 1/x is ln|x|, so {a}/x integrates to {a}ln|x|."]
    else:
        b = _nonzero(rng, -9, 9)
        f = Itg(
            1 / (x + b),
            text=f"1/(x {'+' if b > 0 else '-'} {abs(b)})",
            latex=rf"\frac{{1}}{{x {'+' if b > 0 else '-'} {abs(b)}}}",
        )
        F = _ln(x + b)
        cands = [
            Cand(log(x + b), "r3b-recip-abs-dropped"),
            Cand(_ln(x), "r3b-recip-shift-dropped"),
            Cand(-1 / (x + b) ** 2, "r3b-recip-differentiated"),
            Cand(_ln(x) + b, "r3b-recip-shift-added-outside"),
        ]
        working = [
            f"The denominator has derivative 1, so 1/(x {'+' if b > 0 else '-'} {abs(b)}) integrates to ln|x {'+' if b > 0 else '-'} {abs(b)}|."
        ]
    return _result(rng, f, F, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _linear_text(p, q) -> str:
    if p < 0 and q > 0:
        return f"{q} - {'' if p == -1 else abs(p)}x"
    lead = "x" if p == 1 else "-x" if p == -1 else f"{p}x"
    return f"{lead} {'+' if q > 0 else '-'} {abs(q)}"


def _medium(rng):
    p = rng.choice([2, 3, 4, 5, -2, -3, -4, -5])
    q = _nonzero(rng, -9, 9)
    a = _nonzero(rng, -9, 9)
    lin = p * x + q
    ltxt = _linear_text(p, q)
    f = Itg(a / lin, text=f"{a}/({ltxt})", latex=rf"\frac{{{a}}}{{{ltxt}}}")
    F = Rational(a, p) * _ln(lin)
    cands = [
        Cand(a * _ln(lin), "r3b-recip-not-divided"),
        Cand(a * p * _ln(lin), "r3b-recip-multiplied-by-p"),
        Cand(Rational(a, p) * _ln(p * x), "r3b-recip-shift-dropped"),
        Cand(Rational(a, p) * log(lin), "r3b-recip-abs-dropped"),
        Cand(Rational(a, abs(p)) * _ln(lin), "r3b-recip-sign-of-p-lost") if p < 0 else None,
    ]
    working = [
        f"The derivative of {ltxt} is {p}, so divide by {p}: {a}/({ltxt}) integrates to ({Rational(a, p)}) ln|{ltxt}|."
    ]
    return _result(rng, f, F, cands, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_over_x(rng):
    if rng.random() < 0.5:
        p, c = _nonzero(rng, -5, 5), _nonzero(rng, -9, 9)
        f = Itg(
            (p * x + c) / x,
            text=f"({_cx(p)}x {'+' if c > 0 else '-'} {abs(c)})/x",
            latex=rf"\frac{{{_cx(p)}x {'+' if c > 0 else '-'} {abs(c)}}}{{x}}",
        )
        F = p * x + c * _ln(x)
        cands = [
            Cand((p + c) * x, "r3b-recip-term-cancelled-wrongly"),
            Cand(c * _ln(x), "r3b-recip-divided-term-dropped"),
            Cand(p * x + c * x * _ln(x), "r3b-recip-log-multiplied-by-x"),
            Cand(p * x + c * log(x), "r3b-recip-abs-dropped"),
        ]
        working = [
            f"Divide each term by x: ({_cx(p)}x {'+' if c > 0 else '-'} {abs(c)})/x = {p} {'+' if c > 0 else '-'} {abs(c)}/x."
        ]
    else:
        n = rng.choice([2, 3])
        c = _nonzero(rng, -9, 9)
        f = Itg(
            (x**n + c) / x,
            text=f"(x^{n} {'+' if c > 0 else '-'} {abs(c)})/x",
            latex=rf"\frac{{x^{{{n}}} {'+' if c > 0 else '-'} {abs(c)}}}{{x}}",
        )
        F = x**n / n + c * _ln(x)
        cands = [
            Cand(x**n / n + c * x, "r3b-recip-term-cancelled-wrongly"),
            Cand(x**n + c * _ln(x), "r3b-recip-power-not-divided"),
            Cand(x ** (n + 1) / (n + 1) + c * _ln(x), "r3b-recip-power-not-reduced"),
            Cand(c * _ln(x), "r3b-recip-divided-term-dropped"),
        ]
        working = [
            f"Divide each term by x: (x^{n} {'+' if c > 0 else '-'} {abs(c)})/x = {'x' if n == 2 else f'x^{n - 1}'} {'+' if c > 0 else '-'} {abs(c)}/x."
        ]
    return f, F, cands, working


def _hard_improper(rng):
    b = _nonzero(rng, -6, 6)
    p = rng.choice([1, 2, 3, -1, -2])
    for _ in attempts(100, "an improper fraction with a non-zero remainder"):
        c = _nonzero(rng, -7, 7)
        if c - p * b != 0:
            break
    r = c - p * b
    num_text, den_text = _linear_text(p, c), _linear_text(1, b)
    f = Itg(
        (p * x + c) / (x + b),
        text=f"({num_text})/({den_text})",
        latex=rf"\frac{{{num_text}}}{{{den_text}}}",
    )
    F = p * x + r * _ln(x + b)
    cands = [
        Cand(p * x + c * _ln(x + b), "r3b-recip-remainder-not-found"),
        Cand(p * x * _ln(x + b) + c * _ln(x + b), "r3b-recip-numerator-treated-as-constant"),
        Cand(r * _ln(x + b), "r3b-recip-divided-term-dropped"),
        Cand(p * x - r * _ln(x + b), "r3b-recip-remainder-sign"),
    ]
    sign = "+" if r > 0 else "-"
    working = [
        f"Divide first: {num_text} = {p}({den_text}) {sign} {abs(r)}, so the integrand is {p} {sign} {abs(r)}/({den_text}).",
    ]
    return f, F, cands, working


def _hard_sum(rng):
    a = _nonzero(rng, -6, 6)
    b = _nonzero(rng, -6, 6)
    p = rng.choice([2, 3, 4])
    q = _nonzero(rng, -5, 5)
    c = _nonzero(rng, -5, 5)
    lin = p * x + q
    ltxt = _linear_text(p, q)
    f = Itg(
        a / x + b / lin + c * x,
        text=f"{a}/x {'+' if b > 0 else '-'} {abs(b)}/({ltxt}) {'+' if c > 0 else '-'} {_cx(abs(c))}x",
        latex=rf"\frac{{{a}}}{{x}} {'+' if b > 0 else '-'} \frac{{{abs(b)}}}{{{ltxt}}} {'+' if c > 0 else '-'} {_cx(abs(c))}x",
    )
    F = a * _ln(x) + Rational(b, p) * _ln(lin) + Rational(c, 2) * x**2
    cands = [
        Cand(a * _ln(x) + b * _ln(lin) + Rational(c, 2) * x**2, "r3b-recip-not-divided"),
        Cand(a * _ln(x) + Rational(b, p) * _ln(lin) + c * x, "r3b-recip-linear-term-not-integrated"),
        Cand(a * _ln(x) + Rational(b, p) * _ln(lin) + c * x**2, "r3b-recip-power-not-divided"),
        Cand(a * _ln(x) + Rational(b, abs(p)) * _ln(lin) + Rational(c, 2) * x**2, "r3b-recip-sign-of-p-lost")
        if p < 0
        else None,
    ]
    return f, F, cands, [f"Integrate term by term: the middle term divides by {p}."]


def _hard(rng):
    f, F, cands, working = rng.choice([_hard_over_x, _hard_improper, _hard_sum])(rng)
    return _result(rng, f, F, cands, working)


@generator(
    "r3b_integrate_reciprocal",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_integrate_reciprocal(rng, tier="easy"):
    """Logarithmic antiderivatives of reciprocal linear expressions; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_integrate_reciprocal", level=14, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_integrate_reciprocal(rng, tier="exam"):
    """Exam chain: an unknown numerator, the curve through a point, an exact height and where a gradient is reached."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "exam constants"):
        p = rng.choice([2, 3, 4, 5])
        x0 = rng.choice([0, -1, -1, -2])
        q = 1 - p * x0
        x1, x2, x3 = rng.sample(range(x0 + 1, x0 + 7), 3)
        n1, n2, n3 = p * x1 + q, p * x2 + q, p * x3 + q
        g = rng.choice([1, 2, 3, -1, -2, -3])
        k = g * n1
        y0 = _nonzero(rng, -9, 9)
        if abs(k) <= 30 and len({n1, n2, n3}) == 3 and n2 >= 2 and n3 >= 2 and k % p:
            break
    g3 = Rational(k, n3)
    lin = p * x + q
    f_prime = k / lin
    F = Rational(k, p) * _ln(lin) + y0
    check_antiderivative(F, f_prime)
    check(F.subs(x, x0) == y0 and f_prime.subs(x, x1) == g and f_prime.subs(x, x3) == g3, "the conditions do not hold")
    height = F.subs(x, x2)
    ltxt = _linear_text(p, q)
    wording = rng.randrange(3)
    stem = [
        f"A curve y = f(x) has gradient function f'(x) = k/({ltxt}), where k is a constant. "
        f"The gradient of the curve at x = {x1} is {g}, and the curve passes through the point ({x0}, {y0}).",
        f"The gradient function of a curve y = f(x) is f'(x) = k/({ltxt}), where k is a constant. "
        f"The gradient of the curve at x = {x1} is {g}, and the curve passes through the point ({x0}, {y0}).",
        f"A curve is such that f'(x) = k/({ltxt}), where k is a constant. "
        f"The gradient of the curve at x = {x1} is {g}, and the curve passes through the point ({x0}, {y0}).",
    ][wording]
    stem_latex = [
        rf"A curve \(y = f(x)\) has gradient function \(f'(x) = \frac{{k}}{{{ltxt}}}\), "
        rf"where \(k\) is a constant. The gradient of the curve at \(x = {x1}\) is \({g}\), "
        rf"and the curve passes through the point \(({x0}, {y0})\).",
        rf"The gradient function of a curve \(y = f(x)\) is \(f'(x) = \frac{{k}}{{{ltxt}}}\), "
        rf"where \(k\) is a constant. The gradient of the curve at \(x = {x1}\) is \({g}\), "
        rf"and the curve passes through the point \(({x0}, {y0})\).",
        rf"A curve is such that \(f'(x) = \frac{{k}}{{{ltxt}}}\), "
        rf"where \(k\) is a constant. The gradient of the curve at \(x = {x1}\) is \({g}\), "
        rf"and the curve passes through the point \(({x0}, {y0})\).",
    ][wording]
    pa = part(
        "Find the value of k.",
        "Find the value of \\(k\\).",
        1,
        [f"f'({x1}) = k/{n1} = {g}, so k = {k}."],
        frac_fields(k),
        pick_fractions(
            k,
            [
                (Rational(g, n1), "r3b-chain-divided-instead-of-multiplied"),
                (g * x1 or None, "r3b-chain-substitution-ignores-constant"),
                (g * p * x1 or None, "r3b-chain-substitution-ignores-constant"),
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
            f"f(x) = ({k}/{p}) ln|{ltxt}| + c.",
            f"At x = {x0}, {ltxt} = 1 and ln 1 = 0, so c = {y0}. For this x, {ltxt} is positive.",
        ],
        key_fields(F, const=False),
        pick_expr(
            F,
            False,
            [
                Cand(Rational(k, p) * _ln(lin), "r3b-chain-point-not-used", const=False),
                Cand(Rational(k, p) * log(lin) + y0, "r3b-recip-abs-dropped", const=False),
                Cand(k * _ln(lin) + y0, "r3b-recip-not-divided", const=False),
                Cand(Rational(k, p) * _ln(x) + y0, "r3b-recip-shift-dropped", const=False),
                Cand(Rational(k, p) * _ln(lin) - y0, "r3b-chain-constant-sign", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/{p}*log(Abs({p}*x+({q})))+({y0})",
    )
    pc = part(
        f"Find the exact value of f({x2}).",
        f"Find the exact value of \\(f({x2})\\).",
        2,
        [f"{ltxt} = {n2} at x = {x2}, so f({x2}) = ({Rational(k, p)}) ln {n2} {plus(y0)}."],
        exact_fields(height),
        pick_expr(
            height,
            False,
            [
                Cand(Rational(k, p) * log(n2), "r3b-chain-constant-dropped", const=False),
                Cand(
                    Rational(k, p) * log(x2) + y0 if x2 > 0 else None,
                    "r3b-chain-substituted-into-wrong-part",
                    const=False,
                ),
                Cand(k * log(n2) + y0, "r3b-recip-not-divided", const=False),
                Cand(Rational(k, p) * log(n2 + 1) + y0, "r3b-chain-substitution-slip", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/{p}*log({n2})+({y0})",
    )
    x_ans = Rational(n3 - q, p)
    pd = part(
        f"Find the value of x at which the gradient of the curve is {g3}.",
        f"Find the value of \\(x\\) at which the gradient of the curve is \\({tex(g3)}\\).",
        2,
        [f"k/({ltxt}) = {g3} gives {ltxt} = {n3}, so x = {x_ans}."],
        frac_fields(x_ans),
        pick_fractions(
            x_ans,
            [
                (n3 - q, "r3b-chain-coefficient-of-x-ignored"),
                (Rational(n3 + q, p), "r3b-chain-constant-sign"),
                (Rational(n3, p), "r3b-chain-constant-ignored"),
                (-x_ans, "r3b-chain-sign-slip"),
            ],
            rng=rng,
        ),
        follow_through=f"(part_a/({g3}) - ({q}))/{p}",
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
