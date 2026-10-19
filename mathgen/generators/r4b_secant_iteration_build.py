"""Secant iteration: `r4b_secant_iteration` (easy, medium, hard) and the exam chain `exd_x_r4b_secant_iteration`.

``uc2-secant-iteration`` (``numerical-and-computational-mathematics``):

* easy: one secant step for ``f(x) = x² − N`` from two stated starting values, ``x₂`` to 4 decimal places.
* medium: the equation is given as ``x³ = ax + b`` (not as ``f(x) = 0``), so the learner first rearranges it; one
  secant step from two stated starting values, ``x₂`` to 4 decimal places.
* hard: the same kind of equation, but two steps (``x₃``) from starting values that do not bracket the root, to
  4 decimal places.
* exam: ``f(x) = x² − N`` in four parts: ``x₂`` (fraction), ``f(x₂)`` (fraction), ``x₃`` built from the learner's
  ``x₂`` and ``f(x₂)``, and the absolute error of ``x₃`` as an estimate of √N, to 6 decimal places.

Every step is computed in exact `Fraction`s from ``x_{n+1} = x_n − f(x_n)(x_n − x_{n−1})/(f(x_n) − f(x_{n−1}))`` and
checked a second way: the secant line through the two points is built with SymPy and its zero is taken.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from ..core import attempts, check, generator, round_half_up
from ._r4b_kit import (
    X,
    Redraw,
    dec,
    fraction_part,
    fr,
    margin_ok,
    number_part,
    poly_eval,
    poly_prompt,
    poly_sympy,
    sub,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_NEWTON = "num2-secant-newton-step-confused"
M_SIGN = "num2-secant-correction-sign"
M_NO_F = "num2-secant-f-factor-dropped"
M_ANCHOR = "num2-secant-anchored-at-older-point"
M_WINDOW = "num2-secant-window-not-advanced"
M_EARLY = "num2-secant-stopped-early"
M_RESID = "num2-secant-residual-for-error"
M_SIGNED = "num2-secant-error-sign"
M_WRONG_ITERATE = "num2-secant-error-of-wrong-iterate"
M_AVERAGE = "num2-secant-averaged-starting-values"
M_REARRANGE = "num2-secant-wrong-rearrangement"


def step(f, a: Fraction, b: Fraction) -> Fraction:
    """One secant step from x_{n-1} = a and x_n = b."""
    fa, fb = f(a), f(b)
    check(fb != fa, "the secant line is horizontal")
    return b - fb * (b - a) / (fb - fa)


def step_slips(f, df, a, b):
    """The wrong next values a learner gets from the starting pair (a, b)."""
    fa, fb = f(a), f(b)
    out = []
    if df(b) != 0:
        out.append((M_NEWTON, b - fb / df(b)))
    out.append((M_SIGN, b + fb * (b - a) / (fb - fa)))
    out.append((M_NO_F, b - (b - a) / (fb - fa)))
    out.append((M_ANCHOR, a - fa * (b - a) / (fb - fa)))
    out.append((M_AVERAGE, (a + b) / 2))
    return out


def sympy_step(coeffs, a, b):
    """x-intercept of the chord through (a, f(a)) and (b, f(b)), from the two-point form (a·f(b) − b·f(a))/(f(b) − f(a))."""
    fa, fb = poly_eval(coeffs, a), poly_eval(coeffs, b)
    return (a * fb - b * fa) / (fb - fa)


def _real_root(coeffs) -> float:
    """The largest real root: Newton's method from beyond the Cauchy bound converges monotonically down to it."""
    c = [float(v) for v in coeffs]
    x = 1 + max(abs(v) for v in c[:-1]) / abs(c[-1])
    for _ in range(200):
        fx = sum(v * x**k for k, v in enumerate(c))
        dfx = sum(k * v * x ** (k - 1) for k, v in enumerate(c) if k)
        nxt = x - fx / dfx
        if abs(nxt - x) < 1e-13:
            return nxt
        x = nxt
    return x


def _sqrt_exact(n) -> Fraction:
    """√n to 40 decimal places, as an exact Fraction, from integer square roots."""
    return Fraction(isqrt(int(n) * 10**80), 10**40)


def _eq(a, b):
    return f"x³ = {'' if a == 1 else a}x + {b}"


def _fmt_f(coeffs):
    return poly_prompt(coeffs)


def _pair(rng, lo, hi):
    for _ in attempts(200, "two distinct starting values"):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a != b:
            return a, b
    raise AssertionError("unreachable")


def _equation_cubic(rng, tier):
    """x³ = a x + b with a stated pair of starting values; returns f coefficients (x³ − a x − b)."""
    for _ in attempts(500, "a cubic equation with a usable secant step"):
        a = rng.randint(1, 8)
        b = rng.randint(2, 20)
        coeffs = [-b, -a, 0, 1]
        x0, x1 = _pair(rng, 0, 4 if tier == "medium" else 6)
        if tier == "hard":
            # both starting values to the right of the largest root, so the pair does not bracket it
            r = _real_root(coeffs)
            if min(x0, x1) < r + 0.3 or min(x0, x1) < 1:
                continue
        f = lambda v, c=coeffs: poly_eval(c, v)  # noqa: E731
        if f(Fraction(x0)) == f(Fraction(x1)):
            continue
        return coeffs, a, b, x0, x1
    raise AssertionError("unreachable")


@generator(
    "r4b_secant_iteration",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_secant_iteration(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a secant question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        for _ in attempts(500, "a square-root equation"):
            n = rng.randint(2, 40)
            if isqrt(n) ** 2 == n:
                continue
            x0, x1 = _pair(rng, 1, 7)
            if x0 + x1 == 0 or (x0 * x1 + n) % (x0 + x1) == 0:
                continue  # an integer x₂ would be guessable
            if not margin_ok(Fraction(x0 * x1 + n, x0 + x1), 4, 0.12):
                continue
            break
        coeffs = [-n, 0, 1]
        f = lambda v, c=coeffs: poly_eval(c, v)  # noqa: E731
        df = lambda v: 2 * v  # noqa: E731
        a, b = Fraction(x0), Fraction(x1)
        x2 = step(f, a, b)
        check(x2 == sympy_step(coeffs, a, b), "the chord through the two points disagrees")
        check(x2 == Fraction(x0 * x1 + n, x0 + x1), "closed form disagrees")
        fx = _fmt_f(coeffs)
        text = say_marked(
            rng.choice(
                [
                    f"The secant method is applied to «f(x) = {fx}» with starting values «x{sub(0)} = {x0}» "
                    f"and «x{sub(1)} = {x1}». Find «x{sub(2)}» to 4 decimal places.",
                    f"Take «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}» as the starting values of the secant method for "
                    f"«f(x) = {fx}». Find «x{sub(2)}» to 4 decimal places.",
                    f"Apply one step of the secant method to «f(x) = {fx}» from «x{sub(0)} = {x0}» and "
                    f"«x{sub(1)} = {x1}». What is «x{sub(2)}», to 4 decimal places?",
                ]
            )
        )
        part = number_part(
            "",
            x2,
            4,
            step_slips(f, df, a, b),
            1,
            working=[
                f"f({x0}) = {f(a)} and f({x1}) = {f(b)}.",
                f"x{sub(2)} = {x1} − f({x1})({x1} − {x0})/(f({x1}) − f({x0})) = {x2}.",
            ],
        )
        return _as_question(text, part)
    if tier == "medium":
        for _ in attempts(200, "a medium secant draw"):
            coeffs, ca, cb, x0, x1 = _equation_cubic(rng, "medium")
            f = lambda v, c=coeffs: poly_eval(c, v)  # noqa: E731
            a, b = Fraction(x0), Fraction(x1)
            x2 = step(f, a, b)
            if x2.denominator > 1 and abs(x2) <= 12 and margin_ok(x2, 4, 0.12):
                break
        df = lambda v, a_=ca: 3 * v * v - a_  # noqa: E731
        check(x2 == sympy_step(coeffs, a, b), "the chord through the two points disagrees")
        eq = _eq(ca, cb)
        text = say_marked(
            rng.choice(
                [
                    f"The equation «{eq}» is to be solved by the secant method with starting values «x{sub(0)} = {x0}» "
                    f"and «x{sub(1)} = {x1}». Find «x{sub(2)}» to 4 decimal places.",
                    f"Use the secant method on «{eq}» with «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}» as the starting "
                    f"values. Find «x{sub(2)}» to 4 decimal places.",
                    f"Starting from «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}», apply one step of the secant method to "
                    f"the equation «{eq}». What is «x{sub(2)}», to 4 decimal places?",
                ]
            )
        )
        slips = step_slips(f, df, a, b)
        # solving g(x) = x³ - (a x + b) with the sides swapped changes nothing, but dropping the constant does
        g = lambda v: v**3 - ca * v  # noqa: E731
        if g(a) != g(b):
            slips.append((M_REARRANGE, step(g, a, b)))
        part = number_part(
            "",
            x2,
            4,
            slips,
            1,
            working=[
                f"Take f(x) = x³ − {'' if ca == 1 else ca}x − {cb}, so f({x0}) = {f(a)} and f({x1}) = {f(b)}.",
                f"x{sub(2)} = {x1} − f({x1})({x1} − {x0})/(f({x1}) − f({x0})) = {x2}.",
            ],
        )
        return _as_question(text, part)
    # hard: two steps, 4 decimal places
    for _ in attempts(500, "a two-step secant question"):
        coeffs, ca, cb, x0, x1 = _equation_cubic(rng, "hard")
        f = lambda v, c=coeffs: poly_eval(c, v)  # noqa: E731
        a, b = Fraction(x0), Fraction(x1)
        x2 = step(f, a, b)
        if f(x2) == f(b) or f(x2) == 0:
            continue
        x3 = step(f, b, x2)
        if not margin_ok(x3, 4, 0.12) or abs(x3) > 50:
            continue
        root = _real_root(coeffs)
        if abs(float(x3) - root) >= min(abs(float(a) - root), abs(float(b) - root)) or float(x2) <= 0:
            continue  # keep the iteration visibly closing in on the root
        break
    check(x3 == sympy_step(coeffs, b, x2), "the second chord disagrees")
    eq = _eq(ca, cb)
    text = say_marked(
        rng.choice(
            [
                f"The equation «{eq}» is solved by the secant method with starting values «x{sub(0)} = {x0}» and "
                f"«x{sub(1)} = {x1}». Find «x{sub(3)}» to 4 decimal places.",
                f"Use the secant method on «{eq}» with «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}» as the starting "
                f"values. Find «x{sub(3)}», the second new estimate after the starting values, to 4 decimal places.",
                f"Starting from «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}», apply the secant method twice to the "
                f"equation «{eq}». What is «x{sub(3)}», to 4 decimal places?",
            ]
        )
    )
    cands = [
        (M_WINDOW, x2 - f(x2) * (x2 - a) / (f(x2) - f(a)) if f(x2) != f(a) else None),
        (M_EARLY, x2),
        (M_SIGN, x2 + f(x2) * (x2 - b) / (f(x2) - f(b))),
        (M_NO_F, x2 - (x2 - b) / (f(x2) - f(b))),
        (M_ANCHOR, b - f(b) * (x2 - b) / (f(x2) - f(b))),
    ]
    part = number_part(
        "",
        x3,
        4,
        cands,
        1,
        working=[
            f"f(x) = x³ − {'' if ca == 1 else ca}x − {cb}; the first step gives x{sub(2)} = {x2}.",
            f"x{sub(3)} = x{sub(2)} − f(x{sub(2)})(x{sub(2)} − x{sub(1)})/(f(x{sub(2)}) − f(x{sub(1)})) ≈ {dec(x3, 4)}.",
        ],
    )
    return _as_question(text, part)


def _as_question(text_pair, part):
    text, latex = text_pair
    return dict(
        text=text,
        latex=latex,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind=part["answer_kind"],
        distractors=_need3(part["distractors"]),
        working=part["working"],
    )


def _need3(options):
    if len(options) < 3:
        raise Redraw
    return options


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_secant_iteration", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4b_secant_iteration(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam secant question")


def _exam(rng):
    for _ in attempts(500, "an exam secant draw"):
        n = rng.randint(2, 40)
        if isqrt(n) ** 2 == n:
            continue
        r = isqrt(n)
        x0 = rng.randint(max(1, r - 1), r)
        x1 = rng.randint(r + 1, r + 2)
        coeffs = [-n, 0, 1]
        f = lambda v, c=coeffs: poly_eval(c, v)  # noqa: E731
        a, b = Fraction(x0), Fraction(x1)
        x2 = step(f, a, b)
        fx2 = f(x2)
        x3 = step(f, b, x2)
        root = _sqrt_exact(n)
        err_f = abs(x3 - root)
        if fx2 == 0 or x3 == x2 or not margin_ok(err_f, 6, 0.12) or err_f < Fraction(1, 10**4):
            continue
        break
    check(x2 == sympy_step(coeffs, a, b) and x3 == sympy_step(coeffs, b, x2), "a chord disagrees")
    df = lambda v: 2 * v  # noqa: E731
    stem, latex = say_marked(
        rng.choice(
            [
                f"The secant method is applied to «f(x) = x² − {n}» with starting values «x{sub(0)} = {x0}» and "
                f"«x{sub(1)} = {x1}» to approximate «√{n}».",
                f"To approximate «√{n}», the secant method is used on «f(x) = x² − {n}» from «x{sub(0)} = {x0}» and "
                f"«x{sub(1)} = {x1}».",
                f"Starting from «x{sub(0)} = {x0}» and «x{sub(1)} = {x1}», the secant method is applied to "
                f"«f(x) = x² − {n}» to approximate «√{n}».",
            ]
        )
    )
    f1 = f(b)
    p_a = fraction_part(
        f"Find «x{sub(2)}» as an exact fraction.",
        x2,
        step_slips(f, df, a, b),
        2,
        working=[f"x{sub(2)} = {x1} − f({x1})({x1} − {x0})/(f({x1}) − f({x0})) = {x2}."],
        value_range=(0, None),
    )
    p_b = fraction_part(
        f"Find «f(x{sub(2)})» as an exact fraction.",
        fx2,
        [
            ("num2-secant-evaluates-x-not-f", x2),
            ("num2-secant-residual-previous-iterate", f(b)),
            ("num2-secant-residual-sign", -fx2),
            ("num2-secant-square-dropped", x2 - n),
        ],
        1,
        ft=f"part_a**2 - {n}",
        working=[f"f(x{sub(2)}) = ({x2})² − {n} = {fx2}."],
    )
    ft_c = f"part_a - part_b*(part_a - {x1})/(part_b - {f1})"
    c_cands = [
        (M_WINDOW, x2 - fx2 * (x2 - a) / (fx2 - f(a)) if fx2 != f(a) else None),
        (M_EARLY, x2),
        (M_SIGN, x2 + fx2 * (x2 - b) / (fx2 - f1)),
        (M_NO_F, x2 - (x2 - b) / (fx2 - f1)),
        (M_ANCHOR, b - f1 * (x2 - b) / (fx2 - f1)),
    ]
    p_c = fraction_part(
        f"Using your «x{sub(2)}» and «f(x{sub(2)})», find «x{sub(3)}» as an exact fraction.",
        x3,
        c_cands,
        2,
        ft=ft_c,
        working=[f"x{sub(3)} = x{sub(2)} − f(x{sub(2)})(x{sub(2)} − x{sub(1)})/(f(x{sub(2)}) − f(x{sub(1)})) = {x3}."],
        value_range=(0, None),
    )
    err_x2 = abs(x2 - root)
    d_cands = [
        (M_SIGNED, -err_f),
        (M_WRONG_ITERATE, err_x2),
        (M_RESID, abs(f(x3))),
        ("num2-secant-error-against-x2", abs(x3 - x2)),
    ]
    p_d = number_part(
        f"Using your «x{sub(3)}», find the absolute error of «x{sub(3)}» as an estimate of «√{n}», to 6 decimal places.",
        err_f,
        6,
        d_cands,
        2,
        ft=f"Abs(part_c - sqrt({n}))",
        working=[f"√{n} ≈ {dec(root, 6)}, so |x{sub(3)} − √{n}| ≈ {dec(err_f, 6)}."],
        value_range=(0, 5),
    )
    for p in (p_a, p_b, p_c, p_d):
        p["prompt"], p["prompt_latex"] = say_marked(p["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
