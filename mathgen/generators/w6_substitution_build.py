"""Integration by substitution beyond a hidden `du`: the leftover `x` that must be rewritten, `u` a power, a root, an
exponential or a trigonometric function, definite integrals whose limits change, and trigonometric substitution.

Three skills, each with easy, medium and hard topics (and an exam chain in `w6_substitution_exams_build`):

* ``integrate-substitution-rewrite-x`` (topics ``w6_subst_rewrite_x``): the substitution leaves an `x` in the integrand
  that has to be written in `u`.
  - easy: ``c x (x + a)^n`` with ``u = x + a`` (the hint is given), so ``x = u - a``;
  - medium: ``x√(x + a)``, ``x/√(x + a)``, ``x(bx + a)^n`` (an extra ``1/b`` per factor of ``dx``) and ``x/(x + a)^n``;
  - hard (no hint): ``x³√(x² + a)``, ``x⁵/(x² + a)^n``, ``x³(x² + a)^n`` and the root substitutions
    ``(1 + √x)^n/√x`` and ``1/(√x (1 + √x)^n)``.
* ``integrate-substitution-exponential-trigonometric`` (``w6_subst_exp_trig``):
  - easy: ``e^x (e^x + a)^n``, ``e^x/(e^x + a)^n``, ``cos x (a + sin x)^n``, ``sin x/(a + cos x)^n``, ``sec² x tan^n x``
    and ``cos x e^(sin x)``, with the substitution given;
  - medium: ``e^(2x)(e^x + a)^n``, ``e^(2x)√(e^x + a)``, ``e^(2x)/(e^x + a)^n`` (the leftover ``e^x`` is ``u - a``) and
    ``sin x cos x (a + sin² x)^n`` (``du = 2 sin x cos x dx``);
  - hard (no hint): ``sec⁴ x``, ``sec⁶ x``, ``sec⁴ x tan^n x``, ``sin³ x/cos⁴ x``, ``cos³ x/sin² x`` and ``e^(3x)/(e^x + a)⁴``
    (odd and even powers beyond the squares).
* ``definite-integral-substitution-changed-limits`` (``w6_subst_definite``): an exact value found by changing the limits.
  - easy: ``c x^(m-1)(x^m + k)^n`` with ``u = x^m + k``;
  - medium: ``x√(x + a)``, ``x/√(x + a)``, ``x(x + a)^n`` and ``x/(x + a)^n`` with ``u = x + a``;
  - hard: exponential and trigonometric integrands (no hint), and ``√(a² - x²)`` or ``x²/√(a² - x²)`` with ``x = a sin θ``.

Every antiderivative is built by integrating the transformed integrand in `u` with SymPy, then differentiated back at
sample points; every exact value is compared with Simpson's rule on the original integral.
"""

from __future__ import annotations

import sympy
from sympy import Rational as R
from sympy import Symbol, cos, exp, pi, sec, sin, sqrt, tan

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    check_antiderivative,
    check_definite,
    exact_fields,
    integral_math,
    integral_text,
    key_fields,
    pick_expr,
    retry,
    say,
    shown,
    tex,
    x,
)

u = Symbol("u")

_FIND = ("Find {I}.", "Determine {I}.", "Find the indefinite integral {I}.")
_FIND_HINT = (
    "Find {I}. Use the substitution {H}.",
    "Determine {I} using the substitution {H}.",
    "Use the substitution {H} to find {I}.",
)
_EXACT = ("Find the exact value of {I}.", "Evaluate {I}, giving an exact answer.")
_EXACT_HINT = (
    "Use the substitution {H} to find the exact value of {I}.",
    "Find the exact value of {I} using the substitution {H}.",
)


# -- small helpers -------------------------------------------------------------------------------------------------


def _xp(p: int) -> str:
    """``x``, ``x^2``, ... (``1`` for a zero power) as a factor after a coefficient."""
    return "" if p == 0 else "x" if p == 1 else f"x^{p}"


def _up(n: int) -> str:
    """``u`` for the first power, ``u^3`` otherwise."""
    return "u" if n == 1 else f"u^{n}"


def _ratio_times(c: int, m: int) -> str:
    """``(3/2) times `` for c/m, ``3 times `` for a whole ratio and nothing for 1."""
    r = sympy.Rational(c, m)
    if r == 1:
        return ""
    return f"{r} times " if r.q == 1 else f"({r.p}/{r.q}) times "


def _cf(c) -> str:
    return "" if c == 1 else str(c)


def _sh(a) -> str:
    """``x + 3`` or ``x - 2``."""
    return f"x + {a}" if a > 0 else f"x - {-a}"


def _plus(a, body: str) -> str:
    """``a + body`` written ``body + a`` style: ``e^x + 3`` or ``e^x - 2``."""
    return f"{body} + {a}" if a > 0 else f"{body} - {-a}"


def _F(g, u_expr):
    """The antiderivative of g(u) du with u replaced by `u_expr`, integrated term by term so that it keeps the shape of
    a sum of powers (SymPy's own integrate may fold such a sum into one fraction)."""
    total = 0
    for term in sympy.Add.make_args(sympy.expand(g)):
        coef, power = term.as_independent(u)[0], term.as_independent(u)[1]
        exponent = (
            0 if power == 1 else sympy.Poly(power, u).degree() if power.is_polynomial(u) else power.as_base_exp()[1]
        )
        check(exponent != -1, "a term of u^-1 would integrate to a logarithm")
        total += coef * u ** (exponent + 1) / (exponent + 1)
    return total.subs(u, u_expr)


def _hint(u_expr, name: str = "u"):
    return f"{name} = {shown(u_expr)}", rf"\({name} = {tex(u_expr)}\)"


def _power_base(F):
    """The one expression b (not x itself) such that the antiderivative F is a sum of numeric multiples of powers of b,
    else None. Two antiderivatives that differ by a constant are both right but the grader compares them exactly, so a
    stem names this form (a sum of powers of b, not expanded) to make the expected answer the only natural one."""
    base = None
    for term in sympy.Add.make_args(sympy.sympify(F)):
        rest = term.as_independent(x)[1]
        if rest == 1:
            continue
        b, e = rest.as_base_exp()
        if not e.is_Rational or b == x or b.is_number:
            return None
        if base is None:
            base = b
        elif base != b:
            return None
    return base


def _indef(rng, f: Itg, F, cands, working, hint=None):
    check_antiderivative(F, f.value)
    ts = dict(I=integral_text(f))
    ms = dict(I=integral_math(f))
    if hint is None:
        text, latex = say(rng, _FIND, ts, ms)
    else:
        ts["H"], ms["H"] = hint
        text, latex = say(rng, _FIND_HINT, ts, ms)
    base = _power_base(F)
    if base is not None:
        name = shown(base).replace("sqrt(x)", "√x")
        name = f"({name})" if base.is_Add else name
        text += f" Give your answer as a sum of powers of {name}, not expanded, plus C."
        latex += latex_from_inline(
            rf" Give your answer as a sum of powers of \({tex(base)}\), not expanded, plus \(C\)."
        )
    return dict(
        text=text, latex=latex, working=working, distractors=pick_expr(F, True, cands, rng=rng), **key_fields(F)
    )


def _definite(rng, f: Itg, value, lo, hi, cands, working, hint=None):
    check_definite(value, f.value, lo, hi)
    cands = [Cand(c.expr, c.tag, const=False) for c in cands if c is not None]
    ts = dict(I=integral_text(f, lo, hi))
    ms = dict(I=integral_math(f, lo, hi))
    if hint is None:
        text, latex = say(rng, _EXACT, ts, ms)
    else:
        ts["H"], ms["H"] = hint
        text, latex = say(rng, _EXACT_HINT, ts, ms)
    return dict(
        text=text,
        latex=latex,
        working=working,
        distractors=pick_expr(value, False, cands, rng=rng),
        **exact_fields(value),
    )


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


_A_SIGNED = (-4, -3, -2, -1, 1, 2, 3, 4, 5)
_COEF = (1, 1, 2, 3, 4, 6)


# =====================================================================================================================
# rewrite x
# =====================================================================================================================


def _rx_easy(rng):
    a, n, c = _pick(rng, _A_SIGNED), _pick(rng, (2, 3, 4, 5)), _pick(rng, _COEF)
    v = x + a
    f = Itg(c * x * v**n, text=f"{_cf(c)}x({_sh(a)})^{n}")
    F = _F(c * (u - a) * u**n, v)
    cands = [
        Cand(c * x * v ** (n + 1) / (n + 1), "w6-subst-leftover-x-kept"),
        Cand(_F(c * u * u**n, v), "w6-subst-x-replaced-by-u"),
        Cand(_F(c * (u + a) * u**n, v), "w6-subst-shift-sign-wrong"),
        Cand(c * (v ** (n + 2) / (n + 2) - a * v ** (n + 1) / (n + 2)), "w6-subst-exponent-denominator-mismatch"),
    ]
    working = [
        f"Let u = {_sh(a)}, so x = u {'-' if a > 0 else '+'} {abs(a)} and du = dx.",
        f"The integrand becomes {_cf(c)}(u {'-' if a > 0 else '+'} {abs(a)}){_up(n)}, which expands and integrates term by term. Replace u by {_sh(a)}.",
    ]
    return _indef(rng, f, F, cands, working, _hint(v))


def _rx_medium(rng):
    kind = _pick(rng, ("sqrt", "invsqrt", "linear", "inverse"))
    c = _pick(rng, _COEF)
    if kind in ("sqrt", "invsqrt"):
        a = _pick(rng, (-2, -1, 1, 2, 3, 4, 5, 6))
        v = x + a
        p = R(1, 2) if kind == "sqrt" else R(-1, 2)
        f = Itg(
            c * x * v**p,
            text=f"{_cf(c)}x√({_sh(a)})" if kind == "sqrt" else f"{_cf(c)}x/√({_sh(a)})",
        )
        F = _F(c * (u - a) * u**p, v)
        cands = [
            Cand(c * x * v ** (p + 1) / (p + 1), "w6-subst-leftover-x-kept"),
            Cand(_F(c * u * u**p, v), "w6-subst-x-replaced-by-u"),
            Cand(_F(c * (u + a) * u**p, v), "w6-subst-shift-sign-wrong"),
            Cand(c * (v ** (p + 2) - a * v ** (p + 1)) / (p + 2), "w6-subst-exponent-denominator-mismatch"),
        ]
        hint = _hint(v)
        step = f"x = u {'-' if a > 0 else '+'} {abs(a)}"
    elif kind == "linear":
        b, a, n = _pick(rng, (2, 3)), _pick(rng, (-4, -3, -2, -1, 1, 2, 3, 4)), _pick(rng, (2, 3, 4))
        v = b * x + a
        f = Itg(c * x * v**n, text=f"{_cf(c)}x({b}x {'+' if a > 0 else '-'} {abs(a)})^{n}")
        g = R(c, b * b) * (u - a) * u**n
        F = _F(g, v)
        cands = [
            Cand(_F(R(c, b) * (u - a) * u**n, v), "w6-subst-du-factor-forgotten"),
            Cand(_F(R(c, b * b) * u * u**n, v), "w6-subst-x-replaced-by-u"),
            Cand(_F(R(c, b * b) * (u + a) * u**n, v), "w6-subst-shift-sign-wrong"),
            Cand(c * x * v ** (n + 1) / ((n + 1) * b), "w6-subst-leftover-x-kept"),
        ]
        hint = _hint(v)
        step = f"x = (u {'-' if a > 0 else '+'} {abs(a)})/{b}"
    else:
        a, n = _pick(rng, (-3, -2, -1, 1, 2, 3, 4, 5)), _pick(rng, (3, 4))
        v = x + a
        f = Itg(c * x / v**n, text=f"{_cf(c)}x/({_sh(a)})^{n}")
        F = _F(c * (u - a) * u ** (-n), v)
        cands = [
            Cand(c * x * v ** (1 - n) / (1 - n), "w6-subst-leftover-x-kept"),
            Cand(_F(c * u * u ** (-n), v), "w6-subst-x-replaced-by-u"),
            Cand(_F(c * (u + a) * u ** (-n), v), "w6-subst-shift-sign-wrong"),
            Cand(c * (v ** (2 - n) - a * v ** (1 - n)) / (1 - n), "w6-subst-exponent-denominator-mismatch"),
        ]
        hint = _hint(v)
        step = f"x = u {'-' if a > 0 else '+'} {abs(a)}"
    working = [
        f"Let u = {shown(v)}, so {step} and the factor of dx becomes a factor of du.",
        "Rewrite the whole integrand in u, split it into powers of u, integrate, then replace u by the bracket.",
    ]
    return _indef(rng, f, F, cands, working, hint)


def _rx_hard(rng):
    kind = _pick(rng, ("cube_root", "fifth", "cube_power", "root_a", "root_b"))
    c = _pick(rng, _COEF)
    if kind in ("cube_root", "fifth", "cube_power"):
        a = _pick(rng, (1, 2, 3, 4, 5, 6, 8, 9))
        v = x**2 + a
        if kind == "cube_root":
            f = Itg(c * x**3 * sqrt(v), text=f"{_cf(c)}x^3√(x^2 + {a})")
            g, g_plus = c * (u - a) * sqrt(u) / 2, c * (u + a) * sqrt(u) / 2
            g_wrong_x, g_dropped = c * u * sqrt(u) / 2, c * sqrt(u) / 2
        elif kind == "fifth":
            n = _pick(rng, (4, 5))
            f = Itg(c * x**5 / v**n, text=f"{_cf(c)}x^5/(x^2 + {a})^{n}")
            g, g_plus = c * (u - a) ** 2 * u ** (-n) / 2, c * (u + a) ** 2 * u ** (-n) / 2
            g_wrong_x, g_dropped = c * u**2 * u ** (-n) / 2, c * u * u ** (-n) / 2
        else:
            n = _pick(rng, (2, 3, 4))
            f = Itg(c * x**3 * v**n, text=f"{_cf(c)}x^3(x^2 + {a})^{n}")
            g, g_plus = c * (u - a) * u**n / 2, c * (u + a) * u**n / 2
            g_wrong_x, g_dropped = c * u * u**n / 2, c * u**n / 2
        F = _F(g, v)
        cands = [
            Cand(_F(g_wrong_x, v), "w6-subst-x-replaced-by-u"),
            Cand(_F(g_dropped, v), "w6-subst-odd-power-not-split"),
            Cand(_F(2 * g, v), "w6-subst-du-factor-forgotten"),
            Cand(_F(g_plus, v), "w6-subst-shift-sign-wrong"),
        ]
        if kind == "fifth":
            cands.append(Cand(_F(c * (u * u - a * a) * u ** (-n) / 2, v), "w6-subst-square-binomial-expanded-wrong"))
        working = [
            f"Let u = x^2 + {a}, so du = 2x dx and x^2 = u - {a}.",
            "Split off x dx = du/2 and write the remaining even power of x in u. Integrate and replace u.",
        ]
    else:
        v = 1 + sqrt(x)
        if kind == "root_a":
            n = _pick(rng, (2, 3, 4))
            f = Itg(c * v**n / sqrt(x), text=f"{_cf(c)}(1 + √x)^{n}/√x")
            g = 2 * c * u**n
        else:
            n = _pick(rng, (2, 3))
            f = Itg(c / (sqrt(x) * v**n), text=f"{c}/(√x(1 + √x)^{n})")
            g = 2 * c * u ** (-n)
        F = _F(g, v)
        cands = [
            Cand(_F(g / 2, v), "w6-subst-du-factor-forgotten"),
            Cand(_F(2 * g, v), "w6-subst-du-factor-doubled"),
            Cand(_F(g, sqrt(x)), "w6-subst-wrong-inner-function"),
            Cand(
                c * v ** (n + 1) / (n + 1) if kind == "root_a" else c * v ** (1 - n) / (1 - n),
                "w6-subst-power-rule-on-bracket",
            ),
        ]
        working = [
            "Let u = 1 + √x, so du = dx/(2√x) and dx/√x = 2 du.",
            "The integral becomes 2 times the integral of a power of u. Integrate and replace u.",
        ]
    return _indef(rng, f, F, cands, working)


@generator(
    "w6_subst_rewrite_x",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def w6_subst_rewrite_x(rng, tier="easy"):
    """Substitution where the leftover x has to be rewritten in u; the tiers are in the module docstring."""
    if tier == "easy":
        return _rx_easy(rng)
    if tier == "medium":
        return _rx_medium(rng)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return _rx_hard(rng)


# =====================================================================================================================
# exponential and trigonometric substitutions
# =====================================================================================================================

_SIN, _COS, _TAN = sin(x), cos(x), tan(x)
_EX = exp(x)


def _et_easy(rng):
    kind = _pick(rng, ("exp_pow", "exp_inv", "sin_pow", "cos_inv", "sec_tan", "exp_trig"))
    c = _pick(rng, _COEF)
    if kind == "exp_pow":
        a, n = _pick(rng, (-1, 1, 2, 3, 4)), _pick(rng, (2, 3, 4, 5))
        v = _EX + a
        f = Itg(c * _EX * v**n, text=f"{_cf(c)}e^x({_plus(a, 'e^x')})^{n}")
        F = c * v ** (n + 1) / (n + 1)
        cands = [
            Cand(c * v ** (n + 1), "w6-subst-power-not-divided"),
            Cand(c * _EX * v ** (n + 1) / (n + 1), "w6-subst-leftover-x-kept"),
            Cand(c * v**n / n, "w6-subst-exponent-not-raised"),
            Cand(c * v ** (n + 1) / (n + 1) / _EX, "w6-subst-inner-derivative-divided"),
        ]
        hint = _hint(v)
        wk = ["du = e^x dx, so the integrand is c u^n du."]
    elif kind == "exp_inv":
        a, n = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3, 4))
        v = _EX + a
        f = Itg(c * _EX / v**n, text=f"{_cf(c)}e^x/({_plus(a, 'e^x')})^{n}")
        F = c * v ** (1 - n) / (1 - n)
        cands = [
            Cand(-F, "w6-subst-negative-exponent-sign"),
            Cand(c * v ** (1 - n), "w6-subst-power-not-divided"),
            Cand(c * sympy.log(v), "w6-subst-log-for-power"),
            Cand(c * _EX * v ** (1 - n) / (1 - n), "w6-subst-leftover-x-kept"),
        ]
        hint = _hint(v)
        wk = ["du = e^x dx, so the integrand is c u^(-n) du."]
    elif kind == "sin_pow":
        a, n = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3, 4))
        v = a + _SIN
        f = Itg(c * _COS * v**n, text=f"{_cf(c)}cos(x)({a} + sin(x))^{n}")
        F = c * v ** (n + 1) / (n + 1)
        cands = [
            Cand(c * v ** (n + 1), "w6-subst-power-not-divided"),
            Cand(-F, "w6-subst-sign-of-du-wrong"),
            Cand(c * _COS * v ** (n + 1) / (n + 1), "w6-subst-leftover-x-kept"),
            Cand(c * v**n / n, "w6-subst-exponent-not-raised"),
        ]
        hint = _hint(v)
        wk = ["du = cos x dx."]
    elif kind == "cos_inv":
        a, n = _pick(rng, (2, 3)), _pick(rng, (2, 3))
        v = a + _COS
        f = Itg(c * _SIN / v**n, text=f"{_cf(c)}sin(x)/({a} + cos(x))^{n}")
        F = c * v ** (1 - n) / (n - 1)
        cands = [
            Cand(-F, "w6-subst-minus-from-du-dropped"),
            Cand(c * v ** (1 - n), "w6-subst-power-not-divided"),
            Cand(c * sympy.log(v), "w6-subst-log-for-power"),
            Cand(c * _SIN * v ** (1 - n) / (1 - n), "w6-subst-leftover-x-kept"),
        ]
        hint = _hint(v)
        wk = [
            f"du = -sin x dx, so sin x dx = -du and the integral is {'minus ' if c == 1 else f'-{c} times '}the integral of u^(-{n}) du."
        ]
    elif kind == "sec_tan":
        n = _pick(rng, (2, 3, 4, 5))
        f = Itg(c * sec(x) ** 2 * _TAN**n, text=f"{_cf(c)}sec^2(x)tan^{n}(x)")
        F = c * _TAN ** (n + 1) / (n + 1)
        cands = [
            Cand(c * _TAN ** (n + 1), "w6-subst-power-not-divided"),
            Cand(c * sec(x) ** 2 * _TAN ** (n + 1) / (n + 1), "w6-subst-leftover-x-kept"),
            Cand(c * sec(x) ** (n + 1) / (n + 1), "w6-subst-wrong-inner-function"),
            Cand(c * _TAN**n / n, "w6-subst-exponent-not-raised"),
        ]
        hint = _hint(_TAN)
        wk = ["du = sec^2 x dx."]
    else:
        which = _pick(rng, ("sin", "cos"))
        if which == "sin":
            f = Itg(c * _COS * exp(_SIN), text=f"{_cf(c)}cos(x)e^(sin(x))")
            F = c * exp(_SIN)
            cands = [
                Cand(-F, "w6-subst-sign-of-du-wrong"),
                Cand(c * _SIN * exp(_SIN), "w6-subst-leftover-x-kept"),
                Cand(c * exp(_SIN) / _COS, "w6-subst-inner-derivative-divided"),
                Cand(c * exp(_COS), "w6-subst-wrong-inner-function"),
            ]
            hint = _hint(_SIN)
            wk = [f"du = cos x dx, so the integral is {f'{c} times ' if c != 1 else ''}the integral of e^u du."]
        else:
            f = Itg(c * _SIN * exp(_COS), text=f"{_cf(c)}sin(x)e^(cos(x))")
            F = -c * exp(_COS)
            cands = [
                Cand(-F, "w6-subst-minus-from-du-dropped"),
                Cand(c * _COS * exp(_COS), "w6-subst-leftover-x-kept"),
                Cand(c * exp(_COS) / _SIN, "w6-subst-inner-derivative-divided"),
                Cand(-c * exp(_SIN), "w6-subst-wrong-inner-function"),
            ]
            hint = _hint(_COS)
            wk = ["du = -sin x dx, so sin x dx = -du."]
    working = [f"Let {hint[0]}. " + wk[0], "Integrate in u, then replace u."]
    return _indef(rng, f, F, cands, working, hint)


def _et_medium(rng):
    kind = _pick(rng, ("exp_pow", "exp_sqrt", "exp_inv", "sin_sq"))
    c = _pick(rng, _COEF)
    if kind in ("exp_pow", "exp_sqrt", "exp_inv"):
        v = None
        if kind == "exp_pow":
            a, n = _pick(rng, (-1, 1, 2, 3)), _pick(rng, (2, 3, 4))
            v = _EX + a
            f = Itg(c * exp(2 * x) * v**n, text=f"{_cf(c)}e^(2x)({_plus(a, 'e^x')})^{n}")
            g, p = c * (u - a) * u**n, n
        elif kind == "exp_sqrt":
            a = _pick(rng, (1, 2, 3, 5, 8))
            v = _EX + a
            f = Itg(c * exp(2 * x) * sqrt(v), text=f"{_cf(c)}e^(2x)√({_plus(a, 'e^x')})")
            g, p = c * (u - a) * sqrt(u), R(1, 2)
        else:
            a, n = _pick(rng, (1, 2, 3)), _pick(rng, (3, 4))
            v = _EX + a
            f = Itg(c * exp(2 * x) / v**n, text=f"{_cf(c)}e^(2x)/({_plus(a, 'e^x')})^{n}")
            g, p = c * (u - a) * u ** (-n), -n
        F = _F(g, v)
        cands = [
            Cand(_F(c * u * u**p, v), "w6-subst-x-replaced-by-u"),
            Cand(_F(c * (u + a) * u**p, v), "w6-subst-shift-sign-wrong"),
            Cand(c * exp(2 * x) * v ** (p + 1) / (p + 1), "w6-subst-leftover-x-kept"),
            Cand(c * (v ** (p + 2) - a * v ** (p + 1)) / (p + 2), "w6-subst-exponent-denominator-mismatch"),
        ]
        working = [
            f"Let u = {shown(v)}, so du = e^x dx and e^x = u {'-' if a > 0 else '+'} {abs(a)}.",
            f"Write e^(2x) dx = e^x · e^x dx = (u {'-' if a > 0 else '+'} {abs(a)}) du, integrate term by term and replace u.",
        ]
    else:
        a, n = _pick(rng, (1, 2, 3, 4)), _pick(rng, (2, 3))
        v = a + _SIN**2
        f = Itg(c * _SIN * _COS * v**n, text=f"{_cf(c)}sin(x)cos(x)({a} + sin^2(x))^{n}")
        F = R(c, 2 * (n + 1)) * v ** (n + 1)
        cands = [
            Cand(2 * F, "w6-subst-du-factor-forgotten"),
            Cand(c * v ** (n + 1) / (n + 1) / 4, "w6-subst-du-factor-doubled"),
            Cand(c * _SIN * _COS * v ** (n + 1) / (2 * (n + 1)), "w6-subst-leftover-x-kept"),
            Cand(R(c, 2) * v**n / n, "w6-subst-exponent-not-raised"),
        ]
        working = [
            f"Let u = {a} + sin^2 x, so du = 2 sin x cos x dx and sin x cos x dx = du/2.",
            "Integrate (c/2) u^n and replace u.",
        ]
    return _indef(rng, f, F, cands, working, _hint(v))


def _et_hard(rng):
    kind = _pick(rng, ("sec4", "sec6", "sec4tan", "sin3", "cos3", "exp3"))
    c = _pick(rng, (1, 1, 2, 3, 4))
    if kind == "sec4":
        f = Itg(c * sec(x) ** 4, text=f"{_cf(c)}sec^4(x)")
        F = c * (_TAN + _TAN**3 / 3)
        cands = [
            Cand(c * sec(x) ** 5 / 5, "w6-subst-power-rule-on-trig"),
            Cand(c * (_TAN + _TAN**3), "w6-subst-power-not-divided"),
            Cand(c * _TAN**3 / 3, "w6-subst-identity-term-dropped"),
            Cand(c * (_TAN - _TAN**3 / 3), "w6-subst-identity-sign"),
        ]
        wk = ["Write sec^4 x = sec^2 x (1 + tan^2 x) and let u = tan x, du = sec^2 x dx. Then integrate 1 + u^2."]
    elif kind == "sec6":
        f = Itg(c * sec(x) ** 6, text=f"{_cf(c)}sec^6(x)")
        F = c * (_TAN + 2 * _TAN**3 / 3 + _TAN**5 / 5)
        cands = [
            Cand(c * sec(x) ** 7 / 7, "w6-subst-power-rule-on-trig"),
            Cand(c * (_TAN + _TAN**3 + _TAN**5 / 5), "w6-subst-square-binomial-expanded-wrong"),
            Cand(c * (_TAN + _TAN**3 / 3 + _TAN**5 / 5), "w6-subst-identity-term-dropped"),
            Cand(c * (_TAN - 2 * _TAN**3 / 3 + _TAN**5 / 5), "w6-subst-identity-sign"),
        ]
        wk = ["Write sec^6 x = sec^2 x (1 + tan^2 x)^2 and let u = tan x. Then integrate (1 + u^2)^2 = 1 + 2u^2 + u^4."]
    elif kind == "sec4tan":
        n = _pick(rng, (1, 2, 3))
        f = Itg(c * sec(x) ** 4 * _TAN**n, text=f"{_cf(c)}sec^4(x)tan{'' if n == 1 else f'^{n}'}(x)")
        F = c * (_TAN ** (n + 1) / (n + 1) + _TAN ** (n + 3) / (n + 3))
        cands = [
            Cand(c * _TAN ** (n + 1) / (n + 1), "w6-subst-identity-term-dropped"),
            Cand(c * (_TAN ** (n + 1) / (n + 1) - _TAN ** (n + 3) / (n + 3)), "w6-subst-identity-sign"),
            Cand(c * (_TAN ** (n + 1) + _TAN ** (n + 3)), "w6-subst-power-not-divided"),
            Cand(c * (_TAN ** (n + 1) / (n + 3) + _TAN ** (n + 3) / (n + 1)), "w6-subst-exponent-denominator-mismatch"),
        ]
        wk = ["Write sec^4 x = sec^2 x (1 + tan^2 x), let u = tan x, and integrate u^n (1 + u^2)."]
    elif kind == "sin3":
        f = Itg(c * _SIN**3 / _COS**4, text=f"{_cf(c)}sin^3(x)/cos^4(x)")
        F = c * (_COS ** (-3) / 3 - 1 / _COS)
        cands = [
            Cand(-F, "w6-subst-minus-from-du-dropped"),
            Cand(c * (_COS ** (-3) / 3 + 1 / _COS), "w6-subst-identity-sign"),
            Cand(c * _COS ** (-3) / 3, "w6-subst-identity-term-dropped"),
            Cand(c * _SIN**4 / (4 * _COS**4), "w6-subst-power-rule-on-trig"),
        ]
        wk = ["Write sin^3 x = sin x (1 - cos^2 x), let u = cos x, du = -sin x dx, and integrate -(1 - u^2)/u^4."]
    elif kind == "cos3":
        f = Itg(c * _COS**3 / _SIN**2, text=f"{_cf(c)}cos^3(x)/sin^2(x)")
        F = c * (-1 / _SIN - _SIN)
        cands = [
            Cand(c * (1 / _SIN + _SIN), "w6-subst-negative-exponent-sign"),
            Cand(c * (-1 / _SIN + _SIN), "w6-subst-identity-sign"),
            Cand(-c / _SIN, "w6-subst-identity-term-dropped"),
            Cand(c * _COS**4 / (4 * _SIN**2), "w6-subst-power-rule-on-trig"),
        ]
        wk = ["Write cos^3 x = cos x (1 - sin^2 x), let u = sin x, du = cos x dx, and integrate (1 - u^2)/u^2."]
    else:
        a = _pick(rng, (1, 2, 3))
        v = _EX + a
        f = Itg(c * exp(3 * x) / v**4, text=f"{_cf(c)}e^(3x)/({_plus(a, 'e^x')})^4")
        F = _F(c * (u - a) ** 2 * u ** (-4), v)
        cands = [
            Cand(_F(c * (u * u - a * a) * u ** (-4), v), "w6-subst-square-binomial-expanded-wrong"),
            Cand(_F(c * u**2 * u ** (-4), v), "w6-subst-x-replaced-by-u"),
            Cand(_F(c * (u + a) ** 2 * u ** (-4), v), "w6-subst-shift-sign-wrong"),
            Cand(c * exp(3 * x) * v ** (-3) / (-3), "w6-subst-leftover-x-kept"),
        ]
        wk = [
            f"Let u = {shown(v)}, so e^x = u - {a} and e^(3x) dx = (u - {a})^2 du. Expand, integrate term by term and replace u."
        ]
    return _indef(rng, f, F, cands, wk)


@generator(
    "w6_subst_exp_trig",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def w6_subst_exp_trig(rng, tier="easy"):
    """Exponential and trigonometric substitutions; the tiers are in the module docstring."""
    if tier == "easy":
        return _et_easy(rng)
    if tier == "medium":
        return _et_medium(rng)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return _et_hard(rng)


# =====================================================================================================================
# definite integrals and changed limits
# =====================================================================================================================


def _limit_text(v) -> str:
    return shown(sympy.sympify(v))


def _sd_easy(rng):
    m = _pick(rng, (2, 2, 3, 4))
    k, n, c = _pick(rng, (1, 2, 3, 4, 5)), _pick(rng, (2, 3, 4)), _pick(rng, (1, 2, 3, 4, 6))
    lo = _pick(rng, (0, 0, 1))
    hi = lo + _pick(rng, (1, 2))
    body = x ** (m - 1)
    v = x**m + k
    xp = "" if m == 2 else f"^{m - 1}"
    ftext = f"{_cf(c)}x{xp}(x^{m} + {k})^{n}"
    f = Itg(c * body * v**n, text=ftext)
    u1, u2 = lo**m + k, hi**m + k
    G = lambda t: R(c, m) * R(t) ** (n + 1) / (n + 1)  # noqa: E731
    value = G(u2) - G(u1)
    cands = [
        Cand(G(hi) - G(lo), "w6-subst-limits-not-changed"),
        Cand(G(u2), "w6-subst-lower-limit-dropped"),
        Cand(m * (G(u2) - G(u1)), "w6-subst-du-factor-forgotten"),
        Cand(-value, "w6-subst-limits-swapped"),
    ]
    working = [
        f"Let u = x^{m} + {k}, so du = {m}{_xp(m - 1)} dx. When x = {lo}, u = {u1}; when x = {hi}, u = {u2}.",
        f"The integral becomes {_ratio_times(c, m)}the integral of {_up(n)} du from {u1} to {u2}, which is {shown(value)}.",
    ]
    return _definite(rng, f, value, lo, hi, cands, working, _hint(v))


_SQ = (1, 4, 9, 16, 25)


def _sd_medium(rng):
    kind = _pick(rng, ("sqrt", "invsqrt", "linear", "inverse"))
    c = _pick(rng, (1, 1, 2, 3, 6))
    if kind in ("sqrt", "invsqrt"):
        for _ in attempts(200, "perfect-square u limits"):
            s1, s2 = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3, 4, 5))
            a = rng.randint(1, s1 * s1)
            if s2 > s1:
                break
        lo, hi = s1 * s1 - a, s2 * s2 - a
        p = R(1, 2) if kind == "sqrt" else R(-1, 2)
        v = x + a
        f = Itg(
            c * x * v**p,
            text=f"{_cf(c)}x√({_sh(a)})" if kind == "sqrt" else f"{_cf(c)}x/√({_sh(a)})",
        )
        G = lambda t: sympy.integrate(c * (u - a) * u**p, u).subs(u, t)  # noqa: E731
        u1, u2 = s1 * s1, s2 * s2
        wrong_x = lambda t: sympy.integrate(c * u * u**p, u).subs(u, t)  # noqa: E731
    elif kind == "linear":
        n = _pick(rng, (2, 3, 4))
        a = _pick(rng, (1, 2, 3, 4))
        lo = 0
        hi = _pick(rng, (1, 2, 3))
        v = x + a
        f = Itg(c * x * v**n, text=f"{_cf(c)}x({_sh(a)})^{n}")
        u1, u2 = lo + a, hi + a
        G = lambda t: sympy.integrate(c * (u - a) * u**n, u).subs(u, t)  # noqa: E731
        wrong_x = lambda t: sympy.integrate(c * u * u**n, u).subs(u, t)  # noqa: E731
    else:
        n = _pick(rng, (3, 4))
        a = _pick(rng, (1, 2, 3))
        lo = 0
        hi = _pick(rng, (1, 2, 3, 5))
        v = x + a
        f = Itg(c * x / v**n, text=f"{_cf(c)}x/({_sh(a)})^{n}")
        u1, u2 = lo + a, hi + a
        G = lambda t: sympy.integrate(c * (u - a) * u ** (-n), u).subs(u, t)  # noqa: E731
        wrong_x = lambda t: sympy.integrate(c * u * u ** (-n), u).subs(u, t)  # noqa: E731
    value = sympy.simplify(G(u2) - G(u1))
    cands = [
        Cand(G(hi) - G(lo), "w6-subst-limits-not-changed"),
        Cand(wrong_x(u2) - wrong_x(u1), "w6-subst-x-replaced-by-u"),
        Cand(G(u2), "w6-subst-lower-limit-dropped"),
        Cand(-value, "w6-subst-limits-swapped"),
    ]
    working = [
        f"Let u = {_sh(a)}, so x = u - {a}. When x = {lo}, u = {u1}; when x = {hi}, u = {u2}.",
        f"Write the integrand as {_cf(c)}(u - {a}) times a power of u, integrate term by term and evaluate from u = {u1} to u = {u2}: {shown(value)}.",
    ]
    return _definite(rng, f, value, lo, hi, cands, working, _hint(v))


def _sd_hard(rng):
    kind = _pick(rng, ("exp_inv", "exp_sqrt", "sin_inv", "sec4", "trig_sub", "trig_sub_sq"))
    c = _pick(rng, (1, 1, 2, 3, 4))
    if kind == "exp_inv":
        a, n, b = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3)), _pick(rng, (2, 3, 4, 5, 7))
        v = _EX + a
        f = Itg(c * _EX / v**n, text=f"{_cf(c)}e^x/({_plus(a, 'e^x')})^{n}")
        G = lambda t: R(c) * R(t) ** (1 - n) / (1 - n)  # noqa: E731
        u1, u2 = 1 + a, b + a
        lo, hi = 0, sympy.log(b)
        value = G(u2) - G(u1)
        cands = [
            Cand(G(b) - G(1), "w6-subst-limit-shift-forgotten"),
            Cand(G(u2), "w6-subst-lower-limit-dropped"),
            Cand(-value, "w6-subst-limits-swapped"),
            Cand(c * (sympy.log(u2) - sympy.log(u1)), "w6-subst-log-for-power"),
        ]
        working = [
            f"Let u = e^x + {a}, so du = e^x dx. When x = 0, u = {u1}; when x = ln {b}, u = {u2}.",
            f"The integral is {f'{c} times ' if c != 1 else ''}the integral of u^(-{n}) du from {u1} to {u2}.",
        ]
        return _definite(rng, f, value, lo, hi, cands, working)
    if kind == "exp_sqrt":
        for _ in attempts(100, "perfect-square limits"):
            s1, s2 = _pick(rng, (2, 3)), _pick(rng, (3, 4, 5))
            if s2 > s1:
                break
        a = s1 * s1 - 1
        b = s2 * s2 - a
        f = Itg(c * exp(2 * x) * sqrt(_EX + a), text=f"{_cf(c)}e^(2x)√({_plus(a, 'e^x')})")
        G = lambda t: sympy.integrate(c * (u - a) * sqrt(u), u).subs(u, t)  # noqa: E731
        u1, u2 = s1 * s1, s2 * s2
        lo, hi = 0, sympy.log(b)
        value = sympy.simplify(G(u2) - G(u1))
        cands = [
            Cand(G(b) - G(1), "w6-subst-limit-shift-forgotten"),
            Cand(G(u2), "w6-subst-lower-limit-dropped"),
            Cand(-value, "w6-subst-limits-swapped"),
            Cand(
                sympy.integrate(c * u * sqrt(u), u).subs(u, u2) - sympy.integrate(c * u * sqrt(u), u).subs(u, u1),
                "w6-subst-x-replaced-by-u",
            ),
        ]
        working = [
            f"Let u = e^x + {a}, so e^x = u - {a} and e^(2x) dx = (u - {a}) du. When x = 0, u = {u1}; when x = ln {b}, u = {u2}.",
            f"Evaluate the integral of {_cf(c)}(u - {a})√u du from {u1} to {u2}.",
        ]
        return _definite(rng, f, value, lo, hi, cands, working)
    if kind == "sin_inv":
        a, n = _pick(rng, (1, 2, 3)), _pick(rng, (2, 3))
        f = Itg(c * _COS / (a + _SIN) ** n, text=f"{_cf(c)}cos(x)/({a} + sin(x))^{n}")
        G = lambda t: R(c) * R(t) ** (1 - n) / (1 - n)  # noqa: E731
        u1, u2 = a, a + 1
        lo, hi = 0, pi / 2
        value = G(u2) - G(u1)
        cands = [
            Cand(c * (R(u2) ** (1 - n) - R(u1) ** (1 - n)), "w6-subst-power-not-divided"),
            Cand(G(u2), "w6-subst-lower-limit-dropped"),
            Cand(-value, "w6-subst-limits-swapped"),
            Cand(c * (sympy.log(u2) - sympy.log(u1)), "w6-subst-log-for-power"),
        ]
        working = [
            f"Let u = {a} + sin x, so du = cos x dx. When x = 0, u = {u1}; when x = pi/2, u = {u2}.",
            f"The integral is {f'{c} times ' if c != 1 else ''}the integral of u^(-{n}) du from {u1} to {u2}.",
        ]
        return _definite(rng, f, value, lo, hi, cands, working)
    if kind == "sec4":
        n = _pick(rng, (0, 1, 2, 3))
        f = Itg(
            c * sec(x) ** 4 * _TAN**n,
            text=f"{_cf(c)}sec^4(x)" + ("" if n == 0 else f"tan{'' if n == 1 else f'^{n}'}(x)"),
        )
        G = lambda t: c * (sympy.S(t) ** (n + 1) / (n + 1) + sympy.S(t) ** (n + 3) / (n + 3))  # noqa: E731
        lo, hi = 0, pi / 4
        value = G(1) - G(0)
        cands = [
            Cand(G(pi / 4) - G(0), "w6-subst-limits-not-changed"),
            Cand(c * R(1, n + 1), "w6-subst-identity-term-dropped"),
            Cand(c * (R(1, n + 1) - R(1, n + 3)), "w6-subst-identity-sign"),
            Cand(-value, "w6-subst-limits-swapped"),
        ]
        working = [
            "Write sec^4 x = sec^2 x (1 + tan^2 x) and let u = tan x, du = sec^2 x dx. When x = 0, u = 0; when x = pi/4, u = 1.",
            f"Evaluate the integral of {_cf(c)}{_up(n)}(1 + u^2) du from 0 to 1.",
        ]
        return _definite(rng, f, value, lo, hi, cands, working)
    # trigonometric substitution x = a sin(theta)
    a = _pick(rng, (1, 2, 3, 4, 5, 6))
    alpha = _pick(rng, (pi / 6, pi / 4, pi / 3))
    hi = a * sin(alpha)
    root = sqrt(a * a - x * x)
    if kind == "trig_sub":
        f = Itg(root, text=f"√({a * a} - x^2)")
        sign = 1
    else:
        f = Itg(x**2 / root, text=f"x^2/√({a * a} - x^2)")
        sign = -1
    value = sympy.simplify(R(a * a, 2) * (alpha + sign * sin(alpha) * cos(alpha)))
    cands = [
        Cand(R(a * a, 2) * (alpha - sign * sin(alpha) * cos(alpha)), "w6-subst-identity-sign"),
        Cand(R(a * a, 2) * alpha, "w6-subst-identity-term-dropped"),
        Cand(a * a * (alpha + sign * sin(alpha) * cos(alpha)), "w6-subst-double-angle-factor-dropped"),
        Cand(a * (alpha + sign * sin(alpha) * cos(alpha)), "w6-subst-amplitude-not-squared"),
        Cand(R(a * a, 2) * (hi + sign * sin(alpha) * cos(alpha)), "w6-subst-limits-not-changed"),
    ]
    coef = "" if a == 1 else str(a)
    hint = (f"x = {coef}sin(θ)", rf"\(x = {coef}\sin\theta\)")
    fn = "cos" if kind == "trig_sub" else "sin"
    sq = "1 + cos 2θ" if kind == "trig_sub" else "1 - cos 2θ"
    lead = "" if a == 1 else f"{a * a} times "
    working = [
        f"Let x = {coef} sin θ, so dx = {coef} cos θ dθ and √({a * a} - x^2) = {coef} cos θ. When x = 0, θ = 0; when x = {_limit_text(hi)}, θ = {_limit_text(alpha)}.".replace(
            "= sin", "= sin"
        ).replace("  ", " "),
        f"The integral is {lead}the integral of {fn}^2 θ, which is ({a * a}/2) times the integral of ({sq}).".replace(
            "is the", "is the"
        ),
    ]
    return _definite(rng, f, value, 0, hi, cands, working, hint)


@generator(
    "w6_subst_definite",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def w6_subst_definite(rng, tier="easy"):
    """Definite integrals by substitution with changed limits; the tiers are in the module docstring."""
    if tier == "easy":
        return _sd_easy(rng)
    if tier == "medium":
        return _sd_medium(rng)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return _sd_hard(rng)
