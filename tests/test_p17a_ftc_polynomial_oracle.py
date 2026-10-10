"""Re-derives each rendered question's numeric answer from `question.text` alone, using
`sympy.integrate`/`sympy.solve` -- never importing anything from
`mathgen.generators.p17a_ftc_polynomial_build`. `N_SEEDS` is overridable via `P17A_ORACLE_SEEDS` (default 250
per tier, so >= 1000 questions over the four tiers).
"""

from __future__ import annotations

import os
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest
from sympy import Eq, Poly, Rational, Symbol, integrate, solve, sympify

from mathgen import make

TOPIC = "p17a_ftc_inverse"
N_SEEDS = int(os.environ.get("P17A_ORACLE_SEEDS", "250"))

_x = Symbol("x")
_c = Symbol("c_oracle")


def _poly_expr(poly_text: str):
    """'-8x' / '6x - 3' / '2x^2 - 4x + 7' / '-x^2 + 5' -> a sympy expr in `_x`."""
    py = poly_text.strip().replace("^", "**")
    py = re.sub(r"(\d)(x)", r"\1*\2", py)
    return sympify(py, locals={"x": _x})


def _coeff_c_expr(poly_text: str):
    """'c x^2' / 'c x^2 + 5x' / 'c x^2 - x' -> a sympy expr in `_x`, `_c`."""
    m = re.match(r"^c x\^2(?:\s*([+-])\s*(\d*)x)?$", poly_text.strip())
    assert m, poly_text
    sign, digits = m.group(1), m.group(2)
    if sign is None:
        coeff = 0
    else:
        mag = int(digits) if digits else 1
        coeff = mag if sign == "+" else -mag
    return _c * _x**2 + coeff * _x


def _round2(value) -> Fraction:
    dec = Decimal(str(value.evalf(40) if hasattr(value, "evalf") else value))
    return Fraction(str(dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))


# ---------------------------------------------------------------------------
# easy: unknown coefficient c
# ---------------------------------------------------------------------------

_EASY_POLY_RE = re.compile(r"of \((.+?)\) dx")
_EASY_BOUNDS_RE = re.compile(r"from (?:x = )?(-?\d+) to (?:x = )?(-?\d+)")
_EASY_V_RE = re.compile(r"(?:equals|is) (-?\d+(?:/\d+)?)")


def _oracle_easy(text: str) -> str:
    poly_m = _EASY_POLY_RE.search(text)
    bounds_m = _EASY_BOUNDS_RE.search(text)
    v_m = _EASY_V_RE.search(text)
    assert poly_m and bounds_m and v_m, text
    poly_text = poly_m.group(1)
    a, b = int(bounds_m.group(1)), int(bounds_m.group(2))
    V = Rational(v_m.group(1))
    expr = _coeff_c_expr(poly_text)
    F = integrate(expr, (_x, a, b))
    solved = solve(Eq(F, V), _c)
    assert len(solved) == 1, (text, solved)
    c = Rational(solved[0])
    return str(c.p) if c.q == 1 else f"{c.p}/{c.q}"


@pytest.mark.parametrize("tier", ["easy"])
def test_easy_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_easy(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


# ---------------------------------------------------------------------------
# limit tiers: unknown k
# ---------------------------------------------------------------------------

_LIMIT_POLY_RE = re.compile(r"of \((.+?)\) dx")
_LIMIT_LOWER_RE = re.compile(r"from (?:x = )?(-?\d+) to (?:x = )?k")
_LIMIT_RHS_RE = re.compile(r"(?:equals|is) (.+?),? (?:where|and|with) k > (-?\d+)", re.S)
_MULT_RE = re.compile(
    r"(-?\d+(?:/\d+)?) times the (?:definite integral|integral)(?: of the same function)? "
    r"from (?:x = )?(-?\d+) to (?:x = )?(-?\d+)"
)


def _oracle_limit(text: str) -> str:
    poly_m = _LIMIT_POLY_RE.search(text)
    lower_m = _LIMIT_LOWER_RE.search(text)
    rhs_m = _LIMIT_RHS_RE.search(text)
    assert poly_m and lower_m and rhs_m, text
    poly_text = poly_m.group(1)
    L = int(lower_m.group(1))
    rhs, domain_lo = rhs_m.group(1), int(rhs_m.group(2))
    expr = _poly_expr(poly_text)
    _k = Symbol("k_oracle", real=True)

    mult_m = _MULT_RE.search(rhs)
    if mult_m:
        mult_text, a2, b2 = mult_m.groups()
        mult = Rational(mult_text)
        a2, b2 = int(a2), int(b2)
        W = integrate(expr, (_x, a2, b2))
        target = mult * W
    else:
        target = Rational(rhs.strip().rstrip("."))

    F_k = integrate(expr, (_x, L, _k))
    diff = sympify(F_k - target)
    # A cubic with three real roots forces `solve`'s Cardano route through complex intermediates, so
    # `.is_real` is `None` even for a genuinely real root; `Poly(...).real_roots` isolates the real
    # roots directly (exact, never numeric-only) regardless of that.
    roots = Poly(diff, _k, domain="QQ").real_roots()
    in_domain = [r for r in roots if r.evalf(40) > domain_lo]
    assert len(in_domain) == 1, (text, roots, in_domain)
    return str(_fmt2(_round2(in_domain[0])))


def _fmt2(value: Fraction) -> str:
    scaled = value * 100
    assert scaled.denominator == 1, value
    n = scaled.numerator
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, frac = divmod(n, 100)
    return f"{sign}{whole}.{frac:02d}"


@pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
def test_limit_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_limit(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


# ---------------------------------------------------------------------------
# red-first: the oracle must actually be sensitive to a wrong answer
# ---------------------------------------------------------------------------


def test_easy_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    expected = _oracle_easy(q.text)
    corrupted = q.distractors[0]
    assert corrupted != expected
    with pytest.raises(AssertionError):
        assert corrupted == expected


def test_limit_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="hard")
    expected = _oracle_limit(q.text)
    corrupted = q.distractors[0]
    assert corrupted != expected
    with pytest.raises(AssertionError):
        assert corrupted == expected


# In lowest terms the displayed multiplier `r`'s denominator must never equal `|I_ref|` (the
# reference integral's own value) -- that lets a learner read the RHS off the multiplier's
# denominator without ever evaluating the reference integral. `r`, the reference limits and the
# integrand are parsed from `question.text` alone; `I_ref` is computed here with sympy, never read
# from the generator.

_RATIO_SEEDS = 300


def _oracle_exam_ratio(text: str) -> tuple[Fraction, int]:
    """-> (r in lowest terms, I_ref) parsed/re-derived from `question.text`
    alone, independent of the generator's own multiplier/`W` computation."""
    poly_m = _LIMIT_POLY_RE.search(text)
    rhs_m = _LIMIT_RHS_RE.search(text)
    assert poly_m and rhs_m, text
    poly_text = poly_m.group(1)
    rhs = rhs_m.group(1)
    mult_m = _MULT_RE.search(rhs)
    assert mult_m, text
    mult_text, a2, b2 = mult_m.groups()
    r = Fraction(mult_text) if "/" in mult_text else Fraction(int(mult_text), 1)
    a2, b2 = int(a2), int(b2)
    expr = _poly_expr(poly_text)
    I_ref = integrate(expr, (_x, a2, b2))
    I_ref = int(I_ref)
    return r, I_ref


@pytest.mark.parametrize("tier", ["exam"])
def test_exam_ratio_never_reveals_reference_integral(tier):
    checked = 0
    violations = []
    for seed in range(_RATIO_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        r, I_ref = _oracle_exam_ratio(q.text)
        if r.denominator == abs(I_ref):
            violations.append((seed, r, I_ref, q.text))
        assert r.denominator <= 4, (seed, r, I_ref, q.text)
        checked += 1
    assert checked == _RATIO_SEEDS
    assert not violations, (len(violations), violations[:5])
