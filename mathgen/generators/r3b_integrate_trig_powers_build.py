"""Generates `r3b_integrate_trig_powers` questions (easy, medium, hard) and the exam chain
`r3b_exam_integrate_trig_powers`.

Skill ``integrate-trig-powers`` (``integration``, answer ``expression``): integrals of powers of sine and cosine, by the
double-angle identity for even powers and by substitution for an odd power; indefinite (with ``+ C``) and exact definite.

* easy: ``c sin^2(kx)`` or ``c cos^2(kx)``, which needs ``sin^2 = (1 - cos 2x)/2`` or ``cos^2 = (1 + cos 2x)/2``.
* medium: ``sin^3``, ``cos^3``, ``sin^5`` or ``cos^5`` of ``kx``: split off one factor and substitute.
* hard: one of ``sin^m x cos^n x`` with one odd power, ``sin^4`` or ``cos^4`` of ``kx`` by the double angle twice, or the
  exact value of a definite integral of a power.
* exam (``r3b_exam_integrate_trig_powers``): over 0 to pi/2, (a) an odd power of sine, (b) an even power of sine (a multiple
  of pi), (c) a combination of the two, which follows from (a) and (b), and (d) a product with an odd power.

Every antiderivative is differentiated back and every exact value is compared with Simpson's rule.
"""

from __future__ import annotations

from sympy import Rational, cos, pi, sin

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    check_antiderivative,
    check_definite,
    exact_fields,
    frac_fields,
    integral_math,
    integral_text,
    key_fields,
    part,
    pick_expr,
    pick_fractions,
    retry,
    say,
    x,
)

_FIND = (
    "Find {I}.",
    "Determine {I}.",
    "Find the indefinite integral {I}.",
)
_EXACT = (
    "Find the exact value of {I}.",
    "Evaluate {I}, giving an exact answer.",
)

_FN = {"sin": sin, "cos": cos}


def _arg(k) -> str:
    return "x" if k == 1 else f"{k}x"


def _power(fn: str, n: int, k=1):
    """(plain text, LaTeX) of fn^n(kx)."""
    arg = _arg(k)
    if n == 1:
        return f"{fn}({arg})", rf"\{fn}({arg})"
    return f"{fn}^{n}({arg})", rf"\{fn}^{{{n}}}({arg})"


def _itg(value, *pieces) -> Itg:
    """An integrand from pieces that are `(text, latex)` pairs, a coefficient being a plain integer or None."""
    text = "".join(t for t, _ in pieces)
    latex = "".join(m for _, m in pieces)
    return Itg(value, text=text, latex=latex)


def _coef(c) -> tuple[str, str]:
    return ("", "") if c == 1 else (f"{c} ", f"{c}")


def _fn_pow(fn: str, n: int, k=1):
    return _FN[fn](k * x) ** n


def _indef(rng, f: Itg, F, cands, working):
    check_antiderivative(F, f.value)
    text, latex = say(rng, _FIND, {"I": integral_text(f)}, {"I": integral_math(f)})
    return dict(
        text=text, latex=latex, working=working, distractors=pick_expr(F, True, cands, rng=rng), **key_fields(F)
    )


def _no_constant(cands):
    """Wrong exact values carry no constant of integration."""
    return [Cand(c.expr, c.tag, const=False) for c in cands if c is not None]


def _definite(rng, f: Itg, value, hi, cands, working, lo=0):
    check_definite(value, f.value, lo, hi)
    cands = _no_constant(cands)
    text, latex = say(rng, _EXACT, {"I": integral_text(f, lo, hi)}, {"I": integral_math(f, lo, hi)})
    return dict(
        text=text,
        latex=latex,
        working=working,
        distractors=pick_expr(value, False, cands, rng=rng),
        **exact_fields(value),
    )


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    fn = rng.choice(["sin", "cos"])
    k = rng.choice([1, 1, 2, 3])
    c = rng.choice([1, 2, 4, 6])
    sign = -1 if fn == "sin" else 1
    f = _itg(c * _fn_pow(fn, 2, k), _coef(c), _power(fn, 2, k))
    F = c * (x / 2 + sign * sin(2 * k * x) / (4 * k))
    wrong_sign = c * (x / 2 - sign * sin(2 * k * x) / (4 * k))
    cands = [
        Cand(wrong_sign, "r3b-trig-wrong-double-angle-sign"),
        Cand(c * (x / 2 + sign * sin(2 * k * x) / 4), "r3b-trig-inner-derivative-ignored"),
        Cand(c * (x + sign * sin(2 * k * x) / (2 * k)), "r3b-trig-half-angle-factor-dropped"),
        Cand(c * _fn_pow(fn, 3, k) / (3 * k), "r3b-trig-power-rule-on-trig"),
        Cand(c * (x / 2 + sign * _FN[fn](2 * k * x) / (4 * k)), "r3b-trig-integrated-as-differentiated"),
    ]
    ident = "(1 - cos 2x)/2" if fn == "sin" else "(1 + cos 2x)/2"
    working = [
        f"Use the double-angle identity {fn}^2 A = {ident.replace('x', 'A')} with A = {_arg(k)}.",
        f"Integrate: the term in cos 2A integrates to sin(2A)/{2 * k}, so the result is {c}[x/2 {'-' if fn == 'sin' else '+'} sin({_arg(2 * k)})/{4 * k}] + C.",
    ]
    return _indef(rng, f, F, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _odd_power_antiderivative(fn: str, n: int, k):
    """Antiderivative of fn^n(kx) for odd n by splitting off one factor: the other function, as a polynomial in u."""
    u_fn = "cos" if fn == "sin" else "sin"
    sign = -1 if fn == "sin" else 1
    m = (n - 1) // 2
    u = _FN[u_fn](k * x)
    # (1 - u^2)^m expanded: sum_j C(m, j) (-1)^j u^(2j); integrates to sign * sum ... u^(2j+1)/(2j+1)
    from math import comb

    terms = sum(comb(m, j) * (-1) ** j * u ** (2 * j + 1) / (2 * j + 1) for j in range(m + 1))
    return sign * terms / k, u_fn


def _medium(rng):
    fn = rng.choice(["sin", "cos"])
    n = rng.choice([3, 3, 5])
    k = rng.choice([1, 1, 2, 3])
    f = _itg(_fn_pow(fn, n, k), _power(fn, n, k))
    F, u_fn = _odd_power_antiderivative(fn, n, k)
    u = _FN[u_fn](k * x)
    sign = -1 if fn == "sin" else 1
    if n == 3:
        cands = [
            Cand(sign * (u - u**3 / 3) / k, "r3b-trig-substitution-sign"),
            Cand(sign * (u - u**3) / k, "r3b-trig-substituted-power-not-divided"),
            Cand(sign * (u + u**3 / 3) / k, "r3b-trig-identity-sign"),
            Cand(sign * (u**3 / 3 - u), "r3b-trig-inner-derivative-ignored") if k > 1 else None,
            Cand(_fn_pow(fn, 4, k) / (4 * k), "r3b-trig-power-rule-on-trig"),
            Cand(-sign * (u - u**3 / 3) / k, "r3b-trig-substitution-sign"),
        ]
    else:
        cands = [
            Cand(sign * (u - u**5 / 5) / k, "r3b-trig-square-binomial-expanded-wrong"),
            Cand(sign * (u - u**3 / 3 + u**5 / 5) / k, "r3b-trig-binomial-cross-term-dropped"),
            Cand(sign * (u - 2 * u**3 + u**5 / 5) / k, "r3b-trig-substituted-power-not-divided"),
            Cand(-sign * (u - 2 * u**3 / 3 + u**5 / 5) / k, "r3b-trig-substitution-sign"),
            Cand(_fn_pow(fn, 6, k) / (6 * k), "r3b-trig-power-rule-on-trig"),
        ]
    other = "cos" if fn == "sin" else "sin"
    working = [
        f"Write {fn}^{n}({_arg(k)}) = {fn}({_arg(k)})(1 - {other}^2({_arg(k)}))" + ("" if n == 3 else "^2"),
        f"Substitute u = {other}({_arg(k)}), du = {'-' if fn == 'sin' else ''}{k if k != 1 else ''}{fn}({_arg(k)}) dx, and integrate the polynomial in u.",
    ]
    return _indef(rng, f, F, cands, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_mixed(rng):
    """sin^m x cos^n x with one odd power."""
    odd_on = rng.choice(["sin", "cos"])
    even = rng.choice([2, 4])
    if odd_on == "sin":
        fn_odd, fn_even = "sin", "cos"
    else:
        fn_odd, fn_even = "cos", "sin"
    sign = -1 if odd_on == "sin" else 1
    u = _FN[fn_even](x)
    F = sign * (u ** (even + 1) / (even + 1) - u ** (even + 3) / (even + 3))
    value = _FN[odd_on](x) ** 3 * _FN[fn_even](x) ** even
    f = Itg(
        value,
        text=f"{_power('sin', 3 if odd_on == 'sin' else even)[0]} {_power('cos', 3 if odd_on == 'cos' else even)[0]}",
        latex=_power("sin", 3 if odd_on == "sin" else even)[1] + _power("cos", 3 if odd_on == "cos" else even)[1],
    )
    # keep the printed order sin then cos; the value is the same product
    cands = [
        Cand(sign * (u ** (even + 1) / (even + 1) + u ** (even + 3) / (even + 3)), "r3b-trig-identity-sign"),
        Cand(
            sign * (u ** (even + 1) / (even + 3) - u ** (even + 3) / (even + 1)),
            "r3b-trig-exponent-denominator-mismatch",
        ),
        Cand(sign * u ** (even + 1) / (even + 1), "r3b-trig-substitution-term-dropped"),
        Cand(-sign * (u ** (even + 1) / (even + 1) - u ** (even + 3) / (even + 3)), "r3b-trig-substitution-sign"),
        Cand(sin(x) ** 4 * cos(x) ** (even + 1) / (4 * (even + 1)), "r3b-trig-power-rule-on-trig"),
    ]
    working = [
        f"Split off one {fn_odd}: {fn_odd}^3 x = {fn_odd} x (1 - {fn_even}^2 x).",
        f"Substitute u = {fn_even} x and integrate u^{even}(1 - u^2) = u^{even} - u^{even + 2}.",
    ]
    return _indef(rng, f, F, cands, working)


def _hard_fourth(rng):
    fn = rng.choice(["sin", "cos"])
    k = rng.choice([1, 1, 2])
    sign = -1 if fn == "sin" else 1
    f = _itg(_fn_pow(fn, 4, k), _power(fn, 4, k))
    F = 3 * x / 8 + sign * sin(2 * k * x) / (4 * k) + sin(4 * k * x) / (32 * k)
    cands = [
        Cand(x / 4 + sign * sin(2 * k * x) / (4 * k), "r3b-trig-square-of-double-angle-dropped"),
        Cand(
            3 * x / 8 + sign * sin(2 * k * x) / (4 * k) - sin(4 * k * x) / (32 * k), "r3b-trig-wrong-double-angle-sign"
        ),
        Cand(
            3 * x / 8 + sign * sin(2 * k * x) / (2 * k) + sin(4 * k * x) / (16 * k),
            "r3b-trig-half-angle-factor-dropped",
        ),
        Cand(_fn_pow(fn, 5, k) / (5 * k), "r3b-trig-power-rule-on-trig"),
        Cand(
            3 * x / 8 - sign * sin(2 * k * x) / (4 * k) + sin(4 * k * x) / (32 * k), "r3b-trig-wrong-double-angle-sign"
        ),
    ]
    working = [
        f"Write {fn}^4 A = (({'1 - ' if fn == 'sin' else '1 + '}cos 2A)/2)^2 = 1/4 {'-' if fn == 'sin' else '+'} cos(2A)/2 + cos^2(2A)/4, then cos^2(2A) = (1 + cos 4A)/2.",
        f"So {fn}^4 A = 3/8 {'-' if fn == 'sin' else '+'} cos(2A)/2 + cos(4A)/8 with A = {_arg(k)}. Integrate term by term.",
    ]
    return _indef(rng, f, F, cands, working)


def _hard_definite(rng):
    """Exact value of a definite integral of a power of sine or cosine."""
    fn = rng.choice(["sin", "cos"])
    n = rng.choice([2, 3, 4, 5])
    k = rng.choice([1, 2])
    hi = pi / (2 * k)
    f = _itg(_fn_pow(fn, n, k), _power(fn, n, k))
    wallis = {2: pi / 4, 3: Rational(2, 3), 4: 3 * pi / 16, 5: Rational(8, 15)}[n]
    value = wallis / k
    cands = [
        Cand(wallis, "r3b-trig-inner-derivative-ignored") if k > 1 else None,
        Cand(
            value * 2 if n % 2 == 0 else value * Rational(3, 2),
            "r3b-trig-half-angle-factor-dropped" if n % 2 == 0 else "r3b-trig-substitution-term-dropped",
        ),
        Cand(Rational(1, (n + 1) * k), "r3b-trig-power-rule-on-trig"),
        Cand(-value, "r3b-trig-sign-slip"),
        Cand(value / 2, "r3b-trig-half-angle-factor-doubled")
        if n % 2 == 0
        else Cand(value * 2, "r3b-trig-substitution-term-dropped"),
    ]
    working = [
        "Use the double-angle identity for an even power, or split off one factor and substitute for an odd power.",
        f"Evaluate the antiderivative between 0 and {'pi/' + str(2 * k) if k > 1 else 'pi/2'}.",
    ]
    return _definite(rng, f, value, hi, cands, working)


def _hard(rng):
    return rng.choice([_hard_mixed, _hard_fourth, _hard_definite])(rng)


@generator(
    "r3b_integrate_trig_powers",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_integrate_trig_powers(rng, tier="easy"):
    """Integrals of powers of sine and cosine; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------

_WALLIS = {2: pi / 4, 3: Rational(2, 3), 4: 3 * pi / 16, 5: Rational(8, 15)}


@generator("r3b_exam_integrate_trig_powers", level=14, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_integrate_trig_powers(rng, tier="exam"):
    """Exam chain over 0 to pi/2: an odd power of sine, an even power, a combination of them, a product with an odd power."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice([3, 5])
    m = rng.choice([2, 4])
    p, q = rng.choice([2, 3, 4, 5, 6]), rng.choice([2, 3, 4, 6, 8])
    e = rng.choice([2, 4])
    hi = pi / 2
    i_n, i_m = _WALLIS[n], _WALLIS[m]
    combo = p * i_n + q * i_m
    prod = Rational(1, e + 1) - Rational(1, e + 3)
    s_n, s_m = _power("sin", n), _power("sin", m)
    f_n = Itg(sin(x) ** n, text=s_n[0], latex=s_n[1])
    f_m = Itg(sin(x) ** m, text=s_m[0], latex=s_m[1])
    f_c = Itg(
        p * sin(x) ** n + q * sin(x) ** m,
        text=f"{p} {s_n[0]} + {q} {s_m[0]}",
        latex=f"{p}{s_n[1]} + {q}{s_m[1]}",
    )
    f_p = Itg(
        sin(x) ** 3 * cos(x) ** e,
        text=f"{_power('sin', 3)[0]} {_power('cos', e)[0]}",
        latex=_power("sin", 3)[1] + _power("cos", e)[1],
    )
    for f, v in ((f_n, i_n), (f_m, i_m), (f_c, combo), (f_p, prod)):
        check_definite(v, f.value, 0, hi)
    stem = rng.choice(
        [
            "Evaluate each integral exactly. Give each answer as a fraction or as a multiple of pi.",
            "Each of these definite integrals has an exact value. Write each answer as a fraction or as a multiple of pi.",
            "Work out the exact value of each integral below, giving each answer as a fraction or as a multiple of pi.",
        ]
    )
    lo_hi = (0, hi)

    def prompt(f):
        return "Find the exact value of " + integral_text(f, *lo_hi) + ".", "Find the exact value of " + integral_math(
            f, *lo_hi
        ) + "."

    pa_p, pa_l = prompt(f_n)
    pa = part(
        pa_p,
        pa_l,
        2,
        [
            f"Write sin^{n} x = sin x (1 - cos^2 x)"
            + (" and expand the square" if n == 5 else "")
            + ", then substitute u = cos x."
        ],
        frac_fields(i_n),
        pick_fractions(
            i_n,
            [
                (Rational(1, n + 1), "r3b-trig-power-rule-on-trig"),
                (-i_n, "r3b-trig-sign-slip"),
                (1, "r3b-trig-substitution-term-dropped"),
                (Rational(1, n), "r3b-trig-substitution-term-dropped"),
            ],
            rng=rng,
        ),
    )
    pb_p, pb_l = prompt(f_m)
    pb = part(
        pb_p,
        pb_l,
        2,
        [
            f"Use the double-angle identity {'twice ' if m == 4 else ''}to remove the power, then integrate term by term."
        ],
        exact_fields(i_m),
        pick_expr(
            i_m,
            False,
            _no_constant(
                [
                    Cand(i_m * 2, "r3b-trig-half-angle-factor-dropped"),
                    Cand(i_m / 2, "r3b-trig-half-angle-factor-doubled"),
                    Cand(Rational(1, m + 1), "r3b-trig-power-rule-on-trig"),
                    Cand(pi / 2, "r3b-trig-squared-taken-as-one"),
                ]
            ),
            rng=rng,
        ),
    )
    pc_p, pc_l = prompt(f_c)
    pc = part(
        pc_p,
        pc_l,
        1,
        ["Integrate each term separately, using the results of parts (a) and (b)."],
        exact_fields(combo),
        pick_expr(
            combo,
            False,
            _no_constant(
                [
                    Cand(p * Rational(1, n + 1) + q * i_m, "r3b-trig-power-rule-on-trig"),
                    Cand(i_n + i_m, "r3b-trig-coefficients-ignored"),
                    Cand(p * i_n - q * i_m, "r3b-trig-sign-slip"),
                    Cand(p * i_n + q * i_m / 2, "r3b-trig-half-angle-factor-doubled"),
                    Cand(p * i_n + q * i_m * 2, "r3b-trig-half-angle-factor-dropped"),
                ]
            ),
            rng=rng,
        ),
        follow_through=f"{p}*part_a + {q}*part_b",
    )
    pd_p, pd_l = prompt(f_p)
    pd = part(
        pd_p,
        pd_l,
        3,
        [
            f"Write sin^3 x = sin x (1 - cos^2 x) and substitute u = cos x: the integral becomes the integral from 0 to 1 of u^{e}(1 - u^2) du."
        ],
        frac_fields(prod),
        pick_fractions(
            prod,
            [
                (Rational(1, e + 1), "r3b-trig-substitution-term-dropped"),
                (Rational(1, e + 3), "r3b-trig-substitution-term-dropped"),
                (-prod, "r3b-trig-substitution-sign"),
                (Rational(1, (e + 1) * 4), "r3b-trig-power-rule-on-trig"),
                (Rational(1, e + 1) + Rational(1, e + 3), "r3b-trig-identity-sign"),
            ],
            rng=rng,
        ),
    )
    return dict(text=stem, latex=latex_from_inline(stem), parts=[pa, pb, pc, pd])
