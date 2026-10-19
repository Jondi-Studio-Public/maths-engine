"""Generates `r3b_differentiate_ln` questions (easy, medium, hard) and the exam chain `r3b_exam_differentiate_ln`.

Skill ``differentiate-ln-composite`` (``differentiation``, answer ``expression``): derivatives of the natural logarithm of
a function of x, where the argument is never just x.

* easy: ``ln(ax + b)``, ``ln(x^2 + c)`` and ``ln(x^n)``, where the derivative is ``f'(x)/f(x)`` (and ``n/x`` for a
  power, which is simpler than the quotient SymPy would first give).
* medium: the argument is a trigonometric, exponential or polynomial function whose derivative is not a constant:
  ``ln(sin kx)``, ``ln(cos kx)``, ``ln(e^(kx) + c)`` and ``ln(x^2 + bx + c)``.
* hard: the argument is a product, a quotient or a root, which is simplified with the laws of logarithms before it is
  differentiated; or the gradient at a stated x (an exact fraction) of such a function.
* exam (``r3b_exam_differentiate_ln``): ``f(x) = ln(x^2 + k)`` with an unknown ``k`` fixed by a stated gradient; (a) ``k``,
  (b) ``f'(x)``, (c) the gradient at another x, (d) the exact difference ``f(p) - f(q)``, which needs the laws of logs.

The key is checked against SymPy's own derivative of the logarithm, compared at five points.
"""

from __future__ import annotations

from sympy import Rational, cos, exp, log, sin, sqrt, tan

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Redraw,
    exact_fields,
    frac_fields,
    key_fields,
    part,
    pick_expr,
    pick_fractions,
    retry,
    same_function,
    say,
    shown,
    tex,
    x,
)

_STEMS = (
    "Differentiate ln({s}) with respect to x.",
    "Find dy/dx if y = ln({s}).",
    "Find f'(x) if f(x) = ln({s}).",
)
_STEMS_AT = (
    "Find the gradient of the curve y = ln({s}) at the point where x = {x0}.",
    "Find f'({x0}) if f(x) = ln({s}).",
)


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _coef(a) -> str:
    return "" if a == 1 else "-" if a == -1 else str(a)


def _linear(a, b) -> str:
    return f"{_coef(a)}x {'+' if b > 0 else '-'} {abs(b)}"


def _build(rng, inside, text, latex, key, cands, working, *, x0=None):
    """The question dict for d/dx ln(inside), checked against SymPy's derivative of the logarithm."""
    check(same_function(key, log(inside).diff(x)) if x0 is None else True, "key differs from SymPy's derivative")
    if x0 is None:
        stem, stem_latex = say(rng, _STEMS, {"s": text}, {"s": rf"\({latex}\)"})
        return dict(
            text=stem,
            latex=stem_latex,
            working=working,
            distractors=pick_expr(key, False, cands, rng=rng),
            **key_fields(key, const=False),
        )
    stem, stem_latex = say(rng, _STEMS_AT, {"s": text}, {"s": rf"\({latex}\)"}, {"x0": x0})
    value = Rational(key.subs(x, x0))
    check(value == Rational(log(inside).diff(x).subs(x, x0)), "gradient differs from SymPy's derivative")
    return dict(
        text=stem,
        latex=stem_latex,
        working=working,
        distractors=pick_expr(
            value,
            False,
            [Cand(Rational(c.expr.subs(x, x0)) if c.expr is not None else None, c.tag, False) for c in cands],
            rng=rng,
        ),
        **key_fields(value, const=False),
    )


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    kind = rng.choice(["linear", "quad", "power"])
    if kind == "linear":
        a, b = _nonzero(rng, -7, 7), _nonzero(rng, -9, 9)
        if abs(a) == 1:
            a *= rng.choice([2, 3])
        f_ = a * x + b
        key = a / f_
        text = latex = _linear(a, b)
        cands = [
            Cand(1 / f_, "r3b-lnd-chain-factor-missing", False),
            Cand(1 / Rational(a), "r3b-lnd-derivative-of-inside-only", False),
            Cand(f_ / a, "r3b-lnd-quotient-inverted", False),
            Cand(a * log(f_), "r3b-lnd-log-kept", False),
        ]
        working = [f"d/dx ln(f(x)) = f'(x)/f(x), with f(x) = {text} and f'(x) = {a}."]
    elif kind == "quad":
        c = rng.randint(1, 9)
        p = rng.choice([1, 1, 2, 3])
        f_ = p * x**2 + c
        key = 2 * p * x / f_
        text, latex = f"{_coef(p)}x^2 + {c}", f"{_coef(p)}x^2 + {c}"
        cands = [
            Cand(1 / f_, "r3b-lnd-chain-factor-missing", False),
            Cand(2 * p * x, "r3b-lnd-derivative-of-inside-only", False),
            Cand(f_ / (2 * p * x), "r3b-lnd-quotient-inverted", False),
            Cand(2 * p * x * log(f_), "r3b-lnd-log-kept", False),
        ]
        working = [f"f(x) = {text}, f'(x) = {2 * p}x, so the derivative is f'(x)/f(x)."]
    else:
        n = rng.choice([2, 3, 4, 5])
        f_ = x**n
        key = n / x
        text, latex = f"x^{n}", f"x^{{{n}}}"
        cands = [
            Cand(1 / x**n, "r3b-lnd-chain-factor-missing", False),
            Cand(1 / x, "r3b-lnd-power-coefficient-dropped", False),
            Cand(n * x ** (n - 1), "r3b-lnd-derivative-of-inside-only", False),
            Cand(x / n, "r3b-lnd-quotient-inverted", False),
        ]
        working = [f"ln(x^{n}) = {n}ln x, so the derivative is {n}/x. (Or f'/f = {n}x^{n - 1}/x^{n}.)"]
    return _build(rng, f_, text, latex, key, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    kind = rng.choice(["sin", "cos", "exp", "quad"])
    k = rng.choice([1, 2, 3, 4])
    kx = "x" if k == 1 else f"{k}x"
    if kind == "sin":
        f_, key = sin(k * x), k * cos(k * x) / sin(k * x)
        text, latex = f"sin({kx})", rf"\sin {kx}"
        cands = [
            Cand(cos(k * x) / sin(k * x), "r3b-lnd-chain-factor-missing", False),
            Cand(k * sin(k * x) / cos(k * x), "r3b-lnd-trig-ratio-inverted", False),
            Cand(1 / sin(k * x), "r3b-lnd-chain-factor-missing", False),
            Cand(-k * cos(k * x) / sin(k * x), "r3b-lnd-sign-slip", False),
        ]
    elif kind == "cos":
        f_, key = cos(k * x), -k * sin(k * x) / cos(k * x)
        text, latex = f"cos({kx})", rf"\cos {kx}"
        cands = [
            Cand(k * sin(k * x) / cos(k * x), "r3b-lnd-sign-slip", False),
            Cand(-sin(k * x) / cos(k * x), "r3b-lnd-chain-factor-missing", False),
            Cand(-k * cos(k * x) / sin(k * x), "r3b-lnd-trig-ratio-inverted", False),
            Cand(1 / cos(k * x), "r3b-lnd-chain-factor-missing", False),
        ]
    elif kind == "exp":
        c = rng.randint(1, 8)
        f_, key = exp(k * x) + c, k * exp(k * x) / (exp(k * x) + c)
        text, latex = (f"e^x + {c}" if k == 1 else f"e^({kx}) + {c}"), rf"e^{{{kx}}} + {c}"
        cands = [
            Cand(1 / f_, "r3b-lnd-chain-factor-missing", False),
            Cand(k * exp(k * x) / exp(k * x), "r3b-lnd-constant-dropped", False),
            Cand(k, "r3b-lnd-exponentials-cancelled", False),
            Cand(exp(k * x) / (exp(k * x) + c), "r3b-lnd-chain-factor-missing", False),
        ]
    else:
        b = _nonzero(rng, -6, 6)
        c = rng.randint(b * b // 4 + 1, b * b // 4 + 9)
        f_, key = x**2 + b * x + c, (2 * x + b) / (x**2 + b * x + c)
        bx = "x" if abs(b) == 1 else f"{abs(b)}x"
        text = latex = f"x^2 {'+' if b > 0 else '-'} {bx} + {c}"
        cands = [
            Cand(1 / f_, "r3b-lnd-chain-factor-missing", False),
            Cand(2 * x / f_, "r3b-lnd-linear-term-derivative-dropped", False),
            Cand((2 * x + b) / (2 * x + b + c), "r3b-lnd-denominator-differentiated", False),
            Cand(f_ / (2 * x + b), "r3b-lnd-quotient-inverted", False),
        ]
    return _build(rng, f_, text, latex, key, cands, [f"d/dx ln(f(x)) = f'(x)/f(x), with f(x) = {text}."])


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard(rng):
    kind = rng.choice(["product", "quotient", "root", "gradient"])
    if kind == "product":
        a = _nonzero(rng, -5, 5)
        n = rng.choice([2, 3])
        b, c = rng.choice([2, 3, -2, 1]), _nonzero(rng, -5, 5)
        u, v = x + a, b * x + c
        inside = u**n * v
        key = n / u + b / v
        text = f"(x {'+' if a > 0 else '-'} {abs(a)})^{n}({_linear(b, c)})"
        latex = rf"(x {'+' if a > 0 else '-'} {abs(a)})^{{{n}}}({_linear(b, c)})"
        cands = [
            Cand(1 / u + b / v, "r3b-lnd-power-coefficient-dropped", False),
            Cand(n / u - b / v, "r3b-lnd-sum-as-difference", False),
            Cand((n / u) * (b / v), "r3b-lnd-logarithmic-derivatives-multiplied", False),
            Cand(1 / (u**n * v), "r3b-lnd-chain-factor-missing", False),
        ]
        working = [
            f"ln(uv) = ln u + ln v, and ln(u^{n}) = {n}ln u, so the derivative is {n}/(x {'+' if a > 0 else '-'} {abs(a)}) {'+' if b > 0 else '-'} {abs(b)}/({_linear(b, c)})."
        ]
    elif kind == "quotient":
        c = rng.randint(1, 9)
        p, q = rng.choice([1, 2, 3]), _nonzero(rng, -6, 6)
        u, v = p * x + q, x**2 + c
        inside = u / v
        key = p / u - 2 * x / v
        text = f"({_linear(p, q)})/(x^2 + {c})"
        latex = rf"\frac{{{_linear(p, q)}}}{{x^2 + {c}}}"
        cands = [
            Cand(p / u + 2 * x / v, "r3b-lnd-difference-as-sum", False),
            Cand(1 / u - 1 / v, "r3b-lnd-chain-factor-missing", False),
            Cand(p / u * 2 * x / v, "r3b-lnd-logarithmic-derivatives-multiplied", False),
            Cand((p * v - 2 * x * u) / v**2, "r3b-lnd-quotient-rule-not-divided-by-quotient", False),
        ]
        working = [f"ln(u/v) = ln u - ln v, so the derivative is {p}/({_linear(p, q)}) - 2x/(x^2 + {c})."]
    elif kind == "root":
        c = rng.randint(1, 9)
        p = rng.choice([1, 2, 3])
        inside = sqrt(p * x**2 + c)
        key = p * x / (p * x**2 + c)
        text = f"sqrt({'' if p == 1 else p}x^2 + {c})"
        latex = rf"\sqrt{{{_coef(p)}x^2 + {c}}}"
        cands = [
            Cand(2 * p * x / (p * x**2 + c), "r3b-lnd-root-halving-dropped", False),
            Cand(1 / (p * x**2 + c), "r3b-lnd-chain-factor-missing", False),
            Cand(p * x / (p * x**2 + c) ** Rational(3, 2), "r3b-lnd-root-power-misapplied", False),
            Cand(2 * p * x / (p * x**2 + c) * 2, "r3b-lnd-root-halving-dropped", False),
        ]
        working = [f"ln(sqrt(f)) = (1/2) ln f, so the derivative is (1/2) f'/f = {p}x/({_coef(p)}x^2 + {c})."]
    else:
        a = rng.choice([1, 2, 3, -1, -2])
        n = rng.choice([2, 3])
        b, c = rng.choice([2, 3, 4]), rng.choice([1, 3, 5, -1, -3])
        u, v = x + a, b * x + c
        inside = u**n * v
        for _ in attempts(100, "a stated x where neither factor vanishes"):
            x0 = rng.choice([0, 1, 2, 3, -1, 4])
            if u.subs(x, x0) != 0 and v.subs(x, x0) != 0 and not (n / u + b / v).subs(x, x0).is_integer:
                break  # a fractional value, so the key does not stand out from the fractional slips
        key = n / u + b / v
        text = f"(x {'+' if a > 0 else '-'} {abs(a)})^{n}({_linear(b, c)})"
        latex = rf"(x {'+' if a > 0 else '-'} {abs(a)})^{{{n}}}({_linear(b, c)})"
        cands = [
            Cand(1 / u + b / v, "r3b-lnd-power-coefficient-dropped", False),
            Cand(n / u - b / v, "r3b-lnd-sum-as-difference", False),
            Cand((n / u) * (b / v), "r3b-lnd-logarithmic-derivatives-multiplied", False),
            Cand(1 / (u**n * v), "r3b-lnd-chain-factor-missing", False),
        ]
        working = [
            f"Use ln(uv) = ln u + ln v first: the derivative is {n}/(x {'+' if a > 0 else '-'} {abs(a)}) {'+' if b > 0 else '-'} {abs(b)}/({_linear(b, c)}); then put x = {x0}."
        ]
        return _build(rng, inside, text, latex, key, cands, working, x0=x0)
    return _build(rng, inside, text, latex, key, cands, working)


@generator(
    "r3b_differentiate_ln",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_differentiate_ln(rng, tier="easy"):
    """Derivatives of ln of a function of x; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_differentiate_ln", level=14, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_differentiate_ln(rng, tier="exam"):
    """Exam chain: an unknown constant inside a logarithm, the derivative, a gradient and an exact difference."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "exam constants"):
        k = rng.randint(1, 12)
        x1, x2, p, q = rng.sample([1, 2, 3, 4, 5, 6, -1, -2, -3], 4)
        if x1 > 0 and x2 != x1 and p**2 != q**2 and (p**2 + k) != (q**2 + k):
            break
    fx = x**2 + k
    g = Rational(2 * x1, x1**2 + k)
    grad2 = Rational(2 * x2, x2**2 + k)
    diff_ln = log(Rational(p**2 + k, q**2 + k))
    check(same_function(2 * x / fx, log(fx).diff(x)), "f' differs from SymPy's derivative")
    wording = rng.randrange(3)
    stem = [
        "Let f(x) = ln(x^2 + k), where k is a positive constant.",
        "The curve y = f(x) is given by f(x) = ln(x^2 + k), where k is a positive constant.",
        "A function is defined by f(x) = ln(x^2 + k) for a positive constant k.",
    ][wording]
    stem += f" The gradient of the curve y = f(x) at x = {x1} is {g}."
    stem_latex = [
        r"Let \(f(x) = \ln(x^2 + k)\), where \(k\) is a positive constant.",
        r"The curve \(y = f(x)\) is given by \(f(x) = \ln(x^2 + k)\), where \(k\) is a positive constant.",
        r"A function is defined by \(f(x) = \ln(x^2 + k)\) for a positive constant \(k\).",
    ][wording]
    stem_latex += rf" The gradient of the curve \(y = f(x)\) at \(x = {x1}\) is \({tex(g)}\)."
    pa = part(
        "Find the value of k.",
        "Find the value of \\(k\\).",
        2,
        [f"f'(x) = 2x/(x^2 + k), so f'({x1}) = {2 * x1}/({x1 * x1} + k) = {g}, giving k = {k}."],
        frac_fields(k),
        pick_fractions(
            k,
            [
                (v, t)
                for v, t in [
                    (Rational(2 * x1, 1) / g, "r3b-chain-square-term-forgotten"),
                    (Rational(2, 1) / g - x1, "r3b-chain-square-taken-as-linear"),
                    (Rational(1, 1) / g - x1**2, "r3b-chain-numerator-two-dropped"),
                    (x1**2 + k, "r3b-chain-denominator-for-constant"),
                ]
                if v > 0
            ],
            rng=rng,
        ),
    )
    pb = part(
        "Hence find f'(x).",
        "Hence find \\(f'(x)\\).",
        2,
        ["Differentiate ln(x^2 + k) as f'/f with the value of k found in (a)."],
        key_fields(2 * x / (x**2 + k), const=False),
        pick_expr(
            2 * x / (x**2 + k),
            False,
            [
                Cand(1 / (x**2 + k), "r3b-lnd-chain-factor-missing", False),
                Cand(2 * x, "r3b-lnd-derivative-of-inside-only", False),
                Cand(2 * x / (x**2), "r3b-lnd-constant-dropped", False),
                Cand((x**2 + k) / (2 * x), "r3b-lnd-quotient-inverted", False),
            ],
            rng=rng,
        ),
        follow_through="2*x/(x**2+part_a)",
    )
    pc = part(
        f"Find the gradient of the curve at x = {x2}.",
        f"Find the gradient of the curve at \\(x = {x2}\\).",
        1,
        [f"f'({x2}) = {2 * x2}/({x2 * x2} + {k}) = {grad2}."],
        frac_fields(grad2),
        pick_fractions(
            grad2,
            [
                (Rational(1, x2**2 + k), "r3b-lnd-chain-factor-missing"),
                (Rational(2 * x2, x2**2), "r3b-lnd-constant-dropped"),
                (Rational(x2**2 + k, 2 * x2), "r3b-lnd-quotient-inverted"),
                (Rational(2 * x2, k), "r3b-chain-square-term-forgotten"),
            ],
            rng=rng,
        ),
        follow_through=f"{2 * x2}/({x2 * x2}+part_a)",
    )
    pd = part(
        f"Find the exact value of f({p}) - f({q}), as a single logarithm.",
        f"Find the exact value of \\(f({p}) - f({q})\\), as a single logarithm.",
        3,
        [f"f({p}) - f({q}) = ln({p * p} + {k}) - ln({q * q} + {k}) = ln(({p * p + k})/({q * q + k}))."],
        exact_fields(diff_ln),
        pick_expr(
            diff_ln,
            False,
            [
                Cand(log(Rational((p**2 + k) * (q**2 + k))), "r3b-lnd-difference-as-sum", False),
                Cand(log(Rational(p**2 - q**2, 1)) if p**2 > q**2 else None, "r3b-lnd-log-of-difference", False),
                Cand(log(Rational(p**2, q**2)), "r3b-chain-constant-dropped", False),
                Cand(log(Rational(q**2 + k, p**2 + k)), "r3b-lnd-quotient-inverted", False),
            ],
            rng=rng,
        ),
        follow_through=f"log(({p * p}+part_a)/({q * q}+part_a))",
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
