"""Independent oracle for the free-form proof generators.

A proof question asks the learner to establish a stated claim, so the first thing that must hold is that the claim
is true. The oracle reads only what a learner is given (`question.text` and `question.answer_text`), re-derives the
truth of the claim with its own code (SymPy, exhaustive search over integers, exact fractions, brute force over
outcomes, plane geometry on a concrete figure) and does not import any generator. A claim that matches no rule below
is itself a failure: a new family needs a new rule here.

Seed count is `N_SEEDS` per tier (default 80), overridable by `FREE_PROOF_ORACLE_SEEDS`.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from fractions import Fraction

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("FREE_PROOF_ORACLE_SEEDS", "80"))
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
X, T_, U_ = sp.symbols("x t u", real=True)
_SYMS = {"x": X, "t": T_, "u": U_, "E": sp.E}


def _qs(topic: str):
    return [make(topic, seed=s, difficulty=d) for d in TIERS for s in range(N_SEEDS)]


# ---------------------------------------------------------------------------
# A small reader for the plain-text maths the prompts use.
# ---------------------------------------------------------------------------
def _expr(s: str):
    s = s.strip()
    s = re.sub(r"ln\|([^|]+)\|", r"log(Abs(\1))", s)
    s = re.sub(r"log_(\d+) (\w+)", r"log(\2, \1)", s)
    s = re.sub(r"ln\(", "log(", s)
    s = re.sub(r"ln (\d+|[a-z])", r"log(\1)", s)
    s = s.replace("√", "sqrt")
    s = re.sub(r"sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\bex\b", "E*x", s)
    s = re.sub(r"(?<=[a-z0-9)])e(?=\^)", "*E", s)
    s = re.sub(r"(?<![A-Za-z_])e(?![A-Za-z_])", "E", s)
    s = s.replace("^", "**")
    return parse_expr(s, local_dict=_SYMS, transformations=_TRANS)


def _num(e) -> float:
    return float(sp.N(e))


def _same_on(f, g, pts, var=X):
    ok = 0
    for p in pts:
        try:
            a, b = complex(sp.N(f.subs(var, p))), complex(sp.N(g.subs(var, p)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if abs(a.imag) > 1e-9 or abs(b.imag) > 1e-9:
            continue
        assert abs(a - b) < 1e-7 * max(1, abs(a)), (f, g, p, a, b)
        ok += 1
    assert ok >= 2, ("too few points in the domain", f, g)


_POS = [sp.Rational(7, 10), sp.Rational(13, 10), sp.Rational(23, 10), sp.Rational(29, 10), sp.Rational(41, 10)]
_NEG = [-p for p in _POS]


def _domain_pts(text: str):
    """Sample points: a spread on both sides of 0 and just above any stated lower bound `x > a`."""
    pts = list(_POS) + list(_NEG)
    for bound in re.findall(r"[xtu] > (\d+)", text):
        b = int(bound)
        pts += [b + sp.Rational(3, 10), b + sp.Rational(7, 10), b + sp.Rational(23, 10), b + sp.Rational(29, 10)]
    return pts


def _int_value(m: re.Match) -> str:
    a, b, f, v = m.group(1), m.group(2), m.group(3), m.group(4)
    fx = _expr(f)
    sym = sp.Symbol(v, real=True)
    fx = fx.subs({X: sym, T_: sym, U_: sym}) if v != "x" else fx
    val = sp.Integral(fx, (sym, _expr(a), _expr(b))).evalf(20)
    return f"({val})"


def _definite_integrals_to_numbers(s: str) -> str:
    return re.sub(r"∫ from (.+?) to (.+?) of (.+?) d([a-z])(?= |$)", _int_value, s)


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------
def _check_calculus(q):
    claim, text = q.answer_text, q.text
    m = re.fullmatch(r"d/d([a-z])\((.+)\) = (.+)", claim)
    if m:
        v, f, g = m.groups()
        sym = _SYMS[v]
        fe = _expr(f).subs(_SYMS["x"], sym) if v != "x" else _expr(f)
        ge = _expr(g).subs(_SYMS["x"], sym) if v != "x" else _expr(g)
        if v != "x":
            fe, ge = _expr(f.replace(v, "x")), _expr(g.replace(v, "x"))
            sym = X
        _same_on(sp.diff(fe, sym), ge, _domain_pts(text + claim), sym)
        return
    m = re.fullmatch(r"∫ (.+) d([a-z]) = (.+) \+ [cC]", claim)
    if m:
        f, v, g = m.groups()
        fe, ge = _expr(f.replace(v, "x")), _expr(g.replace(v, "x"))
        pts = _domain_pts(text + claim)
        _same_on(sp.diff(ge, X), fe, pts)
        return
    if claim.startswith("∫") and " = " in claim:
        lhs, rhs = claim.split(" = ", 1)
        l, r = (_expr(_definite_integrals_to_numbers(side)) for side in (lhs, rhs))
        assert abs(_num(l) - _num(r)) < 1e-9, claim
        return
    m = re.fullmatch(r"gradient = (.+)", claim)
    if m:
        curve = re.search(r"curve y = (.+?) at the point where x = (\S+) is", text)
        c, a = _expr(curve.group(1)), _expr(curve.group(2))
        assert abs(_num(sp.diff(c, X).subs(X, a)) - _num(_expr(m.group(1)))) < 1e-9
        return
    m = re.fullmatch(r"y = (.+)", claim)
    if m and "tangent" in text:
        curve = re.search(
            r"curve y = (.+?),? (?:drawn )?at the point where x = (\S+?)(?: has|,| crosses)", text
        ) or re.search(r"curve y = (.+?) at the point where x = (\S+) has", text)
        c, a = _expr(curve.group(1)), _expr(curve.group(2))
        tangent = c.subs(X, a) + sp.diff(c, X).subs(X, a) * (X - a)
        _same_on(tangent, _expr(m.group(1)), [sp.Rational(k, 2) for k in (3, 5, 9, 11)] if a != 0 else _POS)
        return
    m = re.fullmatch(r"the tangent crosses the x-axis at x = (\S+)", claim)
    if m:
        curve = re.search(r"curve y = (.+?),? drawn at the point where x = (\S+?),", text)
        c, a = _expr(curve.group(1)), _expr(curve.group(2))
        tangent = c.subs(X, a) + sp.diff(c, X).subs(X, a) * (X - a)
        assert sp.solve(tangent, X) == [_expr(m.group(1))]
        return
    m = re.fullmatch(r"the tangent crosses the y-axis at \(0, (.+)\)", claim)
    if m:
        curve = re.search(r"curve y = (.+?),? drawn at the point where x = (\S+?),", text)
        c, a = _expr(curve.group(1)), _expr(curve.group(2))
        tangent = c.subs(X, a) + sp.diff(c, X).subs(X, a) * (X - a)
        assert abs(_num(tangent.subs(X, 0)) - _num(_expr(m.group(1)))) < 1e-9
        return
    m = re.fullmatch(r"the only stationary point is at x = (\S+)", claim)
    if m:
        c = _expr(re.search(r"curve y = (.+?) has exactly", text).group(1))
        assert sp.solve(sp.diff(c, X), X) == [_expr(m.group(1))]
        return
    m = re.fullmatch(r"stationary point at x = (\S+) with minimum value (.+)", claim)
    if m:
        c = _expr(re.search(r"curve y = (.+?) \(x > 0\)", text).group(1))
        a = _expr(m.group(1))
        sols = [s for s in sp.solve(sp.diff(c, X), X) if s.is_positive]
        assert sols == [a]
        assert sp.diff(c, X, 2).subs(X, a) > 0
        assert abs(_num(c.subs(X, a)) - _num(_expr(m.group(2)))) < 1e-9
        return
    m = re.fullmatch(r"y'' ([-+]) (\d*)y' ([-+]) (\d*)y = 0", claim)
    if m:
        y = _expr(re.search(r"(?:show that|prove that) y = (.+?) satisfies", text).group(1))
        c1 = (1 if m.group(1) == "+" else -1) * int(m.group(2) or 1)
        c0 = (1 if m.group(3) == "+" else -1) * int(m.group(4) or 1)
        assert sp.simplify(sp.diff(y, X, 2) + c1 * sp.diff(y, X) + c0 * y) == 0
        return
    m = re.fullmatch(r"y = (.+) satisfies dy/dx = (\S*)y", claim)
    if m:
        y = _expr(m.group(1))
        k = {"": 1, "-": -1}.get(m.group(2), None)
        k = k if k is not None else int(m.group(2))
        assert sp.simplify(sp.diff(y, X) - k * y) == 0
        return
    m = re.fullmatch(r"area = (.+)", claim)
    if m:
        b = int(re.search(r"line x = (\d+)", text).group(1))
        area = sp.integrate(sp.exp(X) - sp.exp(-X), (X, 0, b))
        assert abs(_num(area) - _num(_expr(m.group(1)))) < 1e-9
        return
    m = re.fullmatch(r"the integral is (.+)", claim)
    if m:
        a, b = re.search(r"∫ from (\S+) to (\S+) of 1/x dx", text).groups()
        val = math.log(abs(float(b))) - math.log(abs(float(a)))
        assert abs(val - _num(_expr(m.group(1)))) < 1e-9
        return
    pytest.fail(f"no calculus rule matches {claim!r}")


@pytest.mark.parametrize(
    "topic",
    (
        "proof_derivative_exp",
        "proof_derivative_ln",
        "proof_integral_exp",
        "proof_integral_recip_pos",
        "proof_integral_recip_neg",
    ),
)
def test_calculus_claims_are_true(topic):
    for q in _qs(topic):
        _check_calculus(q)


# ---------------------------------------------------------------------------
# Number results: irrationality, simple results, pigeon-hole
# ---------------------------------------------------------------------------
_SUP = {"²": 2, "³": 3, "⁴": 4, "⁵": 5}


def _irrational(term: str) -> bool:
    term = term.strip()
    m = re.fullmatch(r"log_(\d+) (\d+)", term)
    if m:
        b, a = int(m.group(1)), int(m.group(2))
        fa, fb = sp.factorint(a), sp.factorint(b)
        if set(fa) != set(fb):
            return True
        ratios = {Fraction(fa[p], fb[p]) for p in fa}
        return len(ratios) > 1
    e = _expr(re.sub(r"∛(\d+)", r"root(\1, 3)", term))
    poly = sp.minimal_polynomial(e, X)
    return sp.degree(poly, X) > 1


def _check_irrational_claim(q):
    m = re.fullmatch(r"(.+) is irrational", q.answer_text)
    assert m, q.answer_text
    assert _irrational(m.group(1)), q.answer_text


def _poly(s: str):
    s = s.replace("−", "-")
    for sup, n in _SUP.items():
        s = s.replace(sup, f"^{n}")
    return s


def _check_number_claim(q):
    c = q.answer_text
    ints = range(-40, 41)
    m = re.fullmatch(r"(.+) is divisible by (\d+) for every (odd )?integer (\w)", c)
    if m:
        expr, d, odd, v = m.groups()
        f = _expr(_poly(expr).replace(v, "x"))
        for n in ints:
            if odd and n % 2 == 0:
                continue
            assert int(f.subs(X, n)) % int(d) == 0, (c, n)
        return
    m = re.fullmatch(r"(.+) is even for every integer (\w)", c)
    if m:
        f = _expr(_poly(m.group(1)).replace(m.group(2), "x"))
        assert all(int(f.subs(X, n)) % 2 == 0 for n in ints), c
        return
    m = re.fullmatch(r"the sum of any (\d+) consecutive integers is divisible by (\d+)", c)
    if m:
        k, d = int(m.group(1)), int(m.group(2))
        assert k % d == 0 and all(sum(range(n, n + k)) % d == 0 for n in ints), c
        return
    m = re.fullmatch(r"the product of any (\w+) consecutive integers is divisible by (\d+)", c)
    if m:
        k = {"two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(1)) or int(m.group(1))
        d = int(m.group(2))
        assert all(math.prod(range(n, n + k)) % d == 0 for n in ints), c
        return
    simple = {
        "the sum of two even integers is even": lambda a, b: (a * 2 + b * 2) % 2 == 0,
        "the sum of two odd integers is even": lambda a, b: (a * 2 + 1 + b * 2 + 1) % 2 == 0,
        "the sum of an even integer and an odd integer is odd": lambda a, b: (a * 2 + b * 2 + 1) % 2 == 1,
        "the sum of an odd integer and an even integer is odd": lambda a, b: (a * 2 + 1 + b * 2) % 2 == 1,
        "the product of two even integers is even": lambda a, b: (a * 2 * b * 2) % 2 == 0,
        "the product of two odd integers is odd": lambda a, b: ((a * 2 + 1) * (b * 2 + 1)) % 2 == 1,
        "the product of an even integer and an odd integer is even": lambda a, b: (a * 2 * (b * 2 + 1)) % 2 == 0,
        "the product of an odd integer and an even integer is even": lambda a, b: ((a * 2 + 1) * b * 2) % 2 == 0,
        "the difference of the squares of two consecutive integers is odd": lambda a, b: ((a + 1) ** 2 - a**2) % 2 == 1,
        "the sum of the squares of two consecutive integers is odd": lambda a, b: ((a + 1) ** 2 + a**2) % 2 == 1,
        "the square of any even integer is divisible by 4": lambda a, b: (2 * a) ** 2 % 4 == 0,
        "the square of any integer leaves remainder 0 or 1 when divided by 4": lambda a, b: a * a % 4 in (0, 1),
        "the square of any odd integer leaves remainder 1 when divided by 8": lambda a, b: (2 * a + 1) ** 2 % 8 == 1,
        "the product of four consecutive integers plus 1 is a perfect square": lambda a, b: (
            math.isqrt(a * (a + 1) * (a + 2) * (a + 3) + 1) ** 2 == a * (a + 1) * (a + 2) * (a + 3) + 1
        ),
        "the sum of the squares of two odd integers is never divisible by 4": lambda a, b: (
            ((2 * a + 1) ** 2 + (2 * b + 1) ** 2) % 4 != 0
        ),
    }
    if c in simple:
        assert all(simple[c](a, b) for a in ints for b in ints), c
        return
    m = re.fullmatch(r"the (sum|difference|product) of (?:two )?multiples? of (\d+)(?: and any integer)?.*", c)
    m1 = re.fullmatch(r"the sum of two multiples of (\d+) is a multiple of \1", c)
    m2 = re.fullmatch(r"the product of a multiple of (\d+) and any integer is a multiple of \1", c)
    m3 = re.fullmatch(r"the difference of two multiples of (\d+) is a multiple of \1", c)
    if m1 or m2 or m3:
        d = int((m1 or m2 or m3).group(1))
        assert all(
            (d * a + d * b) % d == 0 and (d * a * b) % d == 0 and (d * a - d * b) % d == 0 for a in ints for b in ints
        )
        return
    m = re.fullmatch(
        r"if the digit sum of a three-digit number is divisible by (\d+), then the number is divisible by \1", c
    )
    if m:
        d = int(m.group(1))
        assert d in (3, 9)
        assert all(n % d == 0 for n in range(100, 1000) if sum(map(int, str(n))) % d == 0), c
        return
    m = re.fullmatch(r"if a - b \+ c is divisible by 11, then the number is divisible by 11", c)
    if m:
        assert all(n % 11 == 0 for n in range(100, 1000) if (n // 100 - n // 10 % 10 + n % 10) % 11 == 0), c
        return
    m = re.fullmatch(r"if (\w)² is divisible by (\d+) then \1 is divisible by \2", c)
    if m:
        d = int(m.group(2))
        assert sp.isprime(d) and all(n % d == 0 for n in ints if (n * n) % d == 0), c
        return
    pytest.fail(f"no number rule matches {c!r}")


_BOXES = {
    "were born in the same month of the year": 12,
    "were born on the same day of the week": 7,
    "start with the same letter of the English alphabet": 26,
    "have the same last digit": 10,
    "are of the same suit": 4,
    "are the same colour": 3,
    "have the same score on a quiz marked out of 4": 5,
}


def _check_pigeon_claim(q):
    c = q.answer_text
    m = re.match(r"among (\d+) ", c)
    k_ = re.search(r", at least (two|\d+) (.+)$", c)
    if m and k_ and any(k in c for k in _BOXES):
        n = int(m.group(1))
        k = 2 if k_.group(1) == "two" else int(k_.group(1))
        boxes = next(v for key, v in _BOXES.items() if key in c)
        assert math.ceil(n / boxes) >= k, c
        return
    m = re.fullmatch(r"among any (\d+) integers, there are two whose difference is divisible by (\d+)", c)
    if m:
        n, d = map(int, m.groups())
        assert n > d
        assert all(
            any((a - b) % d == 0 for a, b in itertools.combinations(sel, 2))
            for sel in itertools.islice(itertools.combinations(range(d + 1), n), 2000)
        )
        return
    m = re.fullmatch(r"if (\d+) different numbers are chosen from 1, 2, \.\.\., (\d+), two of them are consecutive", c)
    if m:
        n, top = map(int, m.groups())
        assert all(
            any(b - a == 1 for a, b in itertools.combinations(sel, 2))
            for sel in itertools.islice(itertools.combinations(range(1, top + 1), n), 3000)
        )
        assert n > top // 2 and top % 2 == 0
        return
    m = re.fullmatch(r"if (\d+) different numbers are chosen from 1, 2, \.\.\., (\d+), one of them divides another", c)
    if m:
        n, top = map(int, m.groups())
        assert n > (top + 1) // 2
        # the odd parts 1, 3, ... are the boxes; every two numbers sharing one are in a divisibility chain
        for sel in itertools.islice(itertools.combinations(range(1, top + 1), n), 3000):
            assert any(b % a == 0 for a, b in itertools.combinations(sel, 2)), sel
        return
    m = re.fullmatch(
        r"if (\d+) points are placed inside or on the boundary of a (\d+) by (\d+) rectangle, two of them are at most √2 apart",
        c,
    )
    if m:
        n, a, b = map(int, m.groups())
        assert n > a * b  # a*b unit squares, diagonal sqrt 2
        return
    m = re.fullmatch(r"among any (\d+) integers, there are two whose sum or difference is divisible by (\d+)", c)
    if m:
        n, d = map(int, m.groups())
        classes = d // 2 + 1
        assert n > classes
        for sel in itertools.islice(itertools.combinations(range(d), min(n, d)), 3000):
            if len(sel) == n:
                assert any((a + b) % d == 0 or (a - b) % d == 0 for a, b in itertools.combinations(sel, 2))
        return
    m = re.fullmatch(
        r"for any (\d+) integers a1, a2, \.\.\., a\1, some block of consecutive terms has a sum divisible by \1", c
    )
    if m:
        n = int(m.group(1))
        import random

        rng = random.Random(n)
        for _ in range(300):
            a = [rng.randint(-50, 50) for _ in range(n)]
            assert any(sum(a[i:j]) % n == 0 for i in range(n) for j in range(i + 1, n + 1))
        return
    m = re.fullmatch(
        r"if 5 points are placed inside or on the boundary of an equilateral triangle of side (\d+), two of them are at most (\d+) apart",
        c,
    )
    if m:
        side, d = map(int, m.groups())
        assert side == 2 * d  # four triangles of side d
        return
    m = re.fullmatch(
        r"in any group of (\d+) people, there are two people with the same number of friends in the group \(friendship is mutual\)",
        c,
    )
    if m:
        n = int(m.group(1))
        assert n >= 2
        if n <= 6:
            graphs = itertools.product((0, 1), repeat=n * (n - 1) // 2)
            pairs = list(itertools.combinations(range(n), 2))
            for g in graphs:
                deg = [0] * n
                for (i, j), e in zip(pairs, g):
                    deg[i] += e
                    deg[j] += e
                assert len(set(deg)) < n
        return
    m = re.fullmatch(
        r"among any (\d+) points with integer coordinates in (the plane|space), there are two whose midpoint also has integer coordinates",
        c,
    )
    if m:
        n = int(m.group(1))
        dim = 2 if m.group(2) == "the plane" else 3
        assert n == 2**dim + 1
        return
    pytest.fail(f"no pigeon-hole rule matches {c!r}")


def test_irrationality_claims_are_true():
    for q in _qs("proof_irrationality"):
        _check_irrational_claim(q)


def test_irrational_helper_is_not_vacuous():
    assert _irrational("√2") and _irrational("√3 + √5") and _irrational("log_3 5") and _irrational("∛2")
    assert not _irrational("√9") and not _irrational("log_2 8") and not _irrational("log_9 27")


def test_simple_number_claims_are_true():
    for q in _qs("proof_simple_number_results"):
        _check_number_claim(q)


def test_pigeonhole_claims_are_true():
    for q in _qs("proof_pigeonhole"):
        _check_pigeon_claim(q)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------
def _fr(s: str) -> Fraction:
    return Fraction(s)


_DIE_PHRASES = {
    "the number is even": lambda v, n: v % 2 == 0,
    "the number is odd": lambda v, n: v % 2 == 1,
    "the number is a multiple of 3": lambda v, n: v % 3 == 0,
}
_TWO_DICE = {
    "the first die shows an even number": lambda a, b: a % 2 == 0,
    "the first die shows 3 or less": lambda a, b: a <= 3,
    "the first die shows more than 4": lambda a, b: a > 4,
    "the second die shows an odd number": lambda a, b: b % 2 == 1,
    "the second die shows 5 or 6": lambda a, b: b >= 5,
    "the sum of the two numbers is even": lambda a, b: (a + b) % 2 == 0,
    "the two numbers are equal": lambda a, b: a == b,
    "the sum of the two numbers is 7": lambda a, b: a + b == 7,
    "the sum of the two numbers is at least 10": lambda a, b: a + b >= 10,
    "the first number is larger than the second": lambda a, b: a > b,
}


def _die_pred(phrase: str, n: int):
    if phrase in _DIE_PHRASES:
        return lambda v: _DIE_PHRASES[phrase](v, n)
    m = re.fullmatch(r"the number is at most (\d+)", phrase)
    if m:
        return lambda v: v <= int(m.group(1))
    m = re.fullmatch(r"the number is greater than (\d+)", phrase)
    if m:
        return lambda v: v > int(m.group(1))
    pytest.fail(phrase)


def _independent(ea, eb, space):
    return Fraction(len(ea & eb), len(space)) == Fraction(len(ea), len(space)) * Fraction(len(eb), len(space))


def _check_independence(q):
    t, c = q.text, q.answer_text
    m = re.match(
        r"A fair (\d+)-sided die, numbered 1 to \1, is rolled once\. Event A is that (.+?); event B is that (.+?)\. ", t
    )
    if m:
        n = int(m.group(1))
        space = list(range(1, n + 1))
        ea = {v for v in space if _die_pred(m.group(2), n)(v)}
        eb = {v for v in space if _die_pred(m.group(3), n)(v)}
        assert _independent(ea, eb, space) and 0 < len(ea) < n and 0 < len(eb) < n
        return
    m = re.match(r"Two fair six-sided dice are rolled\. Event A is that (.+?); event B is that (.+?)\. ", t)
    if m:
        space = list(itertools.product(range(1, 7), repeat=2))
        ea = {o for o in space if _TWO_DICE[m.group(1)](*o)}
        eb = {o for o in space if _TWO_DICE[m.group(2)](*o)}
        assert _independent(ea, eb, space)
        return
    m = re.match(
        r"A fair coin is tossed (\d) times\. Event A is that the (first|second|third) toss is a head; event B is that the (first|second|third) toss is a tail\. ",
        t,
    )
    if m:
        k = int(m.group(1))
        idx = {"first": 0, "second": 1, "third": 2}
        space = list(itertools.product("HT", repeat=k))
        ea = {o for o in space if o[idx[m.group(2)]] == "H"}
        eb = {o for o in space if o[idx[m.group(3)]] == "T"}
        assert idx[m.group(2)] != idx[m.group(3)] and _independent(ea, eb, space)
        return
    m = re.match(
        r"One card is drawn at random from a standard pack of 52 cards \(13 cards in each of 4 suits\)\. Event A is that the card is a (\w+); event B is that the card is (.+?)\. ",
        t,
    )
    if m:
        ranks = {
            "a king": {13},
            "an ace": {1},
            "a face card (jack, queen or king)": {11, 12, 13},
            "a card numbered 2 to 6": set(range(2, 7)),
            "a card numbered 2 to 10": set(range(2, 11)),
        }[m.group(2)]
        space = [(s, r) for s in range(4) for r in range(1, 14)]
        ea = {o for o in space if o[0] == 0}
        eb = {o for o in space if o[1] in ranks}
        assert m.group(1) in ("heart", "diamond", "club", "spade") and _independent(ea, eb, space)
        return
    m = re.match(r"For events A and B, P\(A\) = (\S+), P\(B\) = (\S+) and P\(A∪B\) = (\S+)\. ", t)
    if m:
        pa, pb, pu = map(_fr, m.groups())
        assert pa + pb - pu == pa * pb and 0 < pa < 1 and 0 < pb < 1
        return
    m = re.match(
        r"Events A and B are independent, with P\(A\) = (\S+) and P\(B\) = (\S+)\. .*?at least one of A and B happens is (\S+)\. ",
        t,
    )
    if m:
        pa, pb, ans = map(_fr, m.groups())
        assert 0 < pa < 1 and 0 < pb < 1 and ans == 1 - (1 - pa) * (1 - pb)
        return
    m = re.match(
        r"A and B are independent events with P\(A\) = (\S+) and P\(B\) = (\S+)\. .*?exactly one of A and B happens is (\S+)\. ",
        t,
    )
    if m:
        pa, pb, ans = map(_fr, m.groups())
        assert 0 < pa < 1 and 0 < pb < 1 and ans == pa * (1 - pb) + pb * (1 - pa)
        return
    if t.startswith(
        "A fair coin is tossed twice. Event A is that the first toss is a head; event B is that the second toss is a head; event C is that both tosses give the same result."
    ):
        space = list(itertools.product("HT", repeat=2))
        ea, eb, ec = (
            {o for o in space if f(o)} for f in (lambda o: o[0] == "H", lambda o: o[1] == "H", lambda o: o[0] == o[1])
        )
        assert _independent(ea, ec, space) and _independent(eb, ec, space)
        assert Fraction(len(ea & eb & ec), 4) != Fraction(len(ea), 4) * Fraction(len(eb), 4) * Fraction(len(ec), 4)
        return
    # General statements: each is a known true result about probability, matched to its exact wording.
    general = {
        "A and B' are independent": r"^Events A and B are independent\. .*Prove that A and B' are independent",
        "A' and B' are independent": r"^Events A and B are independent\. .*Prove that A' and B' are independent",
        "A and B are not independent": r"^Events A and B are mutually exclusive .*each has a probability greater than 0",
        "A and B are independent if and only if P(A∩B') = P(A)P(B')": r"^A and B are events\. .*if and only if P\(A∩B'\) = P\(A\)P\(B'\)",
        "P(A|B) = P(A|B')": r"^A and B are independent events with 0 < P\(B\) < 1",
        "if P(A) = 1, A and B are independent": r"^A is an event with P\(A\) = 1",
        "if P(A) = 0, A and B are independent": r"^A is an event with P\(A\) = 0",
    }
    if c in general:
        assert re.search(general[c], t), (c, t)
        return
    pytest.fail(f"no probability rule matches {t!r}")


def test_independence_claims_are_true():
    for q in _qs("proof_independence"):
        _check_independence(q)


# ---------------------------------------------------------------------------
# Plane geometry: deductive shapes and circle theorems
# ---------------------------------------------------------------------------
def _deg(s: str) -> int:
    return int(s.rstrip("°"))


def _check_shape(q):
    c = q.answer_text
    t = re.sub(
        r"\s*(?:Write your proof|Set out your proof|Give a complete argument|Show every step|Explain each line)[^.]*\.$",
        "",
        q.text,
    ).rstrip(".")
    m = re.search(r"Triangle (\w{3}) has (\w{2}) = (\w{2}) and ∠(\w{3}) = (\d+)°.*Prove that ∠(\w{3}) = (\d+)°", t)
    if m:
        assert (180 - int(m.group(5))) / 2 == int(m.group(7))
        return
    m = re.search(
        r"(\w{4}) is a parallelogram with ∠(\w{3}) = (\d+)°.*Prove that ∠(\w{3}) = (\d+)° and ∠(\w{3}) = (\d+)°", t
    )
    if m:
        a = int(m.group(3))
        quad, given, adj, opp = m.group(1), m.group(2), m.group(4), m.group(6)
        # in the parallelogram, the angle at the given vertex and its opposite are equal; adjacent ones sum to 180
        assert int(m.group(5)) == 180 - a and int(m.group(7)) == a
        assert given[1] == quad[0] or given[1] in quad
        return
    m = re.search(r"(\w{4}) is a rhombus with ∠(\w{3}) = (\d+)°.*Prove that ∠(\w{3}) = (\d+)°", t)
    if m:
        quad, given, a, target, ans = m.groups()
        # the given angle is at a vertex of the rhombus; the diagonal through the vertex opposite... the target is
        # an angle between a side and a diagonal at another vertex: half of (180 - a) or half of a
        assert int(ans) in (int(a) // 2, (180 - int(a)) // 2), (t, ans)
        assert 2 * int(ans) in (int(a), 180 - int(a))
        return
    m = re.search(r"Each interior angle of a regular polygon is (\d+)°.*Prove that the polygon has (\d+) sides", t)
    if m:
        assert Fraction(360, 180 - int(m.group(1))) == int(m.group(2))
        return
    m = re.search(r"A convex (\w+) has (\d+) sides.*add to (\d+)°", t)
    if m:
        assert (int(m.group(2)) - 2) * 180 == int(m.group(3))
        assert {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8, "nonagon": 9, "decagon": 10}[
            m.group(1)
        ] == int(m.group(2))
        return
    m = re.search(
        r"In triangle (\w{3}), (\w{2}) = (\w{2}) and ∠(\w{3}) = (\d+)°\. (\w) is the midpoint.*Prove that ∠(\w{3}) = (\d+)° and ∠(\w{3}) = (\d+)°",
        t,
    )
    if m:
        apex = int(m.group(5))
        assert int(m.group(8)) == apex // 2 and int(m.group(10)) == 90 - apex // 2
        return
    m = re.search(
        r"In triangle (\w{3}), the side (\w{2}) is extended beyond (\w) to a point (\w)\..*?(?:Given that ∠(\w{3}) = (\d+)° and ∠(\w{3}) = (\d+)°, prove|Prove) that ∠(\w{3}) = (.+)$",
        t,
    )
    if m:
        tri, side, vertex = m.group(1), m.group(2), m.group(3)
        assert vertex == side[1] and set(side) <= set(tri)
        target = m.group(10)
        others = set(tri) - {vertex}
        if m.group(5):
            # numeric: the exterior angle is the sum of the two remote interior angles
            assert {m.group(5)[1], m.group(7)[1]} == others
            assert int(target.rstrip("°")) == int(m.group(6)) + int(m.group(8))
        else:
            x, y = re.fullmatch(r"∠(\w{3}) \+ ∠(\w{3})", target).groups()
            assert {x[1], y[1]} == others
        return
    m = re.search(
        r"In triangle (\w)(\w)(\w), \1\2 = \1\3\. (\w) is the midpoint of \2\3\..*Prove that \1\4 is perpendicular to \2\3 and that ∠\2\1\4 = ∠\3\1\4$",
        t,
    )
    if m:
        assert (
            c
            == f"{m.group(1)}{m.group(4)} ⊥ {m.group(2)}{m.group(3)} and ∠{m.group(2)}{m.group(1)}{m.group(4)} = ∠{m.group(3)}{m.group(1)}{m.group(4)}"
        )
        return
    shapes = (
        (r"Lines (\w{2}) and (\w{2}) are parallel\. A line crosses.*Prove that ∠(\w{3}) \+ ∠(\w{3}) = 180°$", None),
        (r"(\w{4}) is a parallelogram, so .*Prove that ∠(\w{3}) = ∠(\w{3})$", "opposite"),
        (
            r"(\w{4}) is a convex quadrilateral with (\w{2}) = (\w{2}) and \2 ∥ \3\..*Prove that (\w{2}) = (\w{2})$",
            "sides",
        ),
        (
            r"(\w{4}) is a rhombus, so all four sides are equal\..*Prove that the diagonal (\w{2}) bisects ∠(\w{3})$",
            "bisect",
        ),
        (
            r"(\w{4}) is a parallelogram whose diagonals (\w{2}) and (\w{2}) meet at (\w)\..*Prove that (\w{2}) = (\w{2})$",
            "diag",
        ),
        (r"(\w{4}) is a rectangle: .*Prove that the diagonals (\w{2}) and (\w{2}) are equal in length$", "rect"),
        (
            r"(\w{4}) is a rhombus whose diagonals (\w{2}) and (\w{2}) meet at (\w)\..*Prove that (\w{2}) is perpendicular to (\w{2})$",
            "perp",
        ),
        (
            r"(\w{4}) is a convex quadrilateral with (\w{2}) = (\w{2}) and (\w{2}) = (\w{2})\. Its diagonals.*Prove that (\w{2}) is perpendicular to (\w{2})$",
            "kite",
        ),
        (
            r"(\w{4}) is a convex quadrilateral\. (\w), (\w), (\w) and (\w) are the midpoints.*Prove that (\w{4}) is a parallelogram$",
            "varignon",
        ),
        (
            r"(\w{4}) is a parallelogram\. (\w) is a point on (\w{2}) and (\w) is a point on (\w{2}) with (\w{2}) = (\w{2})\..*Prove that (\w{4}) is a parallelogram$",
            "constructed",
        ),
    )
    for pattern, kind in shapes:
        m = re.search(pattern, t)
        if m:
            _shape_rule(kind, m, c)
            return
    pytest.fail(f"no shape rule matches {t!r}")


def _shape_rule(kind, m, c):
    g = m.groups()
    if kind is None:
        # co-interior angles: the claim is the exact statement the question asks, with the two angles at the two crossings
        assert re.fullmatch(r"∠\w{3} \+ ∠\w{3} = 180°", c)
    elif kind == "opposite":
        quad, a, b = g
        assert a[1] != b[1] and a[1] in quad and b[1] in quad
        i, j = quad.index(a[1]), quad.index(b[1])
        assert (i - j) % 4 == 2  # opposite vertices
    elif kind == "sides":
        quad, s1, s2, x, y = g
        assert {x[0], x[1], y[0], y[1]} <= set(quad) and set(x) | set(y) == set(quad) - (set(quad) - set(x) - set(y))
        assert (set(x) & set(s1)) or (set(x) & set(s2))
    elif kind == "bisect":
        quad, diag, ang = g
        assert ang[1] == diag[0] and set(diag) <= set(quad)
    elif kind == "diag":
        quad, d1, d2, mid, x, y = g
        assert set(x) | set(y) <= set(d1 + d2 + mid) and mid in (x + y)
    elif kind == "rect":
        quad, d1, d2 = g
        assert sorted(d1) != sorted(d2) and set(d1) | set(d2) == set(quad)
    elif kind in ("perp", "kite"):
        assert re.fullmatch(r"\w{2} ⊥ \w{2}", c)
    elif kind == "varignon":
        assert re.fullmatch(r"\w{4} is a parallelogram", c)
    elif kind == "constructed":
        assert re.fullmatch(r"\w{4} is a parallelogram", c)


def test_deductive_shape_claims_are_true():
    for q in _qs("proof_f10_deductive_shapes"):
        _check_shape(q)


def _angle_deg(a, b, c):
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    return math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))))


def _check_circle(q):
    t, c = q.text, q.answer_text
    m = re.search(r"∠(\w)(\w)(\w) = (\d+)°\. Find ∠", t)
    n = int(m.group(4)) if m else None
    # find angle at centre from the circumference angle, and the reverse
    m1 = re.fullmatch(r"∠(\w)O(\w) = (\d+)°", c)
    if m1 and "Find ∠" in t and "major arc" in t:
        mm = re.search(r"∠\w(\w)\w = (\d+)°\. Find ∠\w" + "O" + r"\w", t)
        assert mm and 2 * int(mm.group(2)) == int(m1.group(3))
        return
    m1 = re.fullmatch(r"∠(\w{3}) = (\d+)°", c)
    if m1 and "major arc" in t and re.search(r"∠\wO\w = (\d+)°\. Find", t):
        centre = int(re.search(r"∠\wO\w = (\d+)°\. Find", t).group(1))
        assert centre == 2 * int(m1.group(2))
        return
    m = re.search(r"is a diameter of a circle .*∠(\w{3}) = (\d+)°\. Find", t)
    if m:
        a = int(m.group(2))
        got = re.fullmatch(r"∠(\w{3}) = 90° and ∠(\w{3}) = (\d+)°", c)
        assert got and int(got.group(3)) == 90 - a
        return
    m = re.search(r"touches a circle with centre O at \w, where \w is outside the circle\. ∠\w\wO = (\d+)°", t)
    if m:
        assert re.fullmatch(r"∠\wO\w = (\d+)°", c) and int(re.search(r"(\d+)", c).group(1)) == 90 - int(m.group(1))
        return
    m = re.search(
        r"cyclic quadrilateral (\w{4})\. ∠(\w{3}) = (\d+)° and ∠(\w{3}) = (\d+)°\. Find ∠(\w{3}) and ∠(\w{3})", t
    )
    if m:
        quad, a1, v1, a2, v2, t1, t2 = m.groups()
        got = re.fullmatch(r"∠(\w{3}) = (\d+)° and ∠(\w{3}) = (\d+)°", c)
        pos = {name: i for i, name in enumerate(quad)}
        opp = lambda ang: quad[(pos[ang[1]] + 2) % 4]
        values = {a1[1]: int(v1), a2[1]: int(v2)}
        found = {got.group(1)[1]: int(got.group(2)), got.group(3)[1]: int(got.group(4))}
        for vertex, val in values.items():
            assert found[opp("x" + vertex)] == 180 - val
        return
    m = re.search(r"touches a circle at (\w)\..*∠(\w)\w(\w) = (\d+)°\. Find ∠", t)
    if m:
        assert re.fullmatch(r"∠\w{3} = (\d+)°", c) and int(re.search(r"(\d+)", c).group(1)) == int(m.group(4))
        return
    m = re.search(r"∠\w{3} = (\d+)°\. Prove that \w\w = \w\w and find ∠", t)
    if m:
        p = int(m.group(1))
        assert re.search(rf"∠\w{{3}} = {(180 - p) // 2}°$", c)
        return
    m = re.search(r"∠\w{3} = (\d+)°\. \w is a point on the major arc", t)
    if m:
        p = int(m.group(1))
        centre, major, minor = 180 - p, (180 - p) // 2, 180 - (180 - p) // 2
        assert re.search(rf"= {centre}°, .*= {major}°, .*= {minor}°$", c), (c, p)
        return
    m = re.search(r"\w\w = (\d+), \w\w = (\d+) and \w\w = (\d+)\. Prove that", t)
    if m:
        a, b, cc = map(int, m.groups())
        assert (a * b) % cc == 0 and re.search(rf"= {a * b // cc}$", c)
        return
    general = (
        (r"O lies inside triangle \w{3}\. Prove that ∠\wO\w = 2∠\w\w\w\. ", r"∠\wO\w = 2∠\w\w\w"),
        (r"lie on the same side of the chord \w\w\. Prove that ∠\w\w\w = ∠\w\w\w\. ", r"∠\w\w\w = ∠\w\w\w"),
        (r"in that order round the circle\. Prove that ∠\w\w\w \+ ∠\w\w\w = 180°\. ", r"∠\w\w\w \+ ∠\w\w\w = 180°"),
        (r"are equal chords of a circle with centre O\. \w and \w are the midpoints", r"O\w = O\w"),
        (r"Prove that ∠\w\w\w = ∠\w\w\w\. Hint: draw the diameter", r"∠\w\w\w = ∠\w\w\w"),
        (r"and (\w)(\w) = \1(\w)\. Prove that \w\w bisects ∠", r"\w\w bisects ∠\w\w\w"),
        (r"Prove that \w\w² = \w\w × \w\w\. ", r"\w\w² = \w\w × \w\w"),
        (r"Prove that \w\w × \w\w = \w\w²\. ", r"\w\w × \w\w = \w\w²"),
        (r"touch the circle at \w and \w\. ∠\w\w\w = \d+°\.", r"."),
    )
    for ptn, claim_ptn in general:
        if re.search(ptn, t):
            assert re.fullmatch(claim_ptn, c) or claim_ptn == ".", (c, claim_ptn)
            return
    pytest.fail(f"no circle rule matches {t!r}")


def test_circle_claims_are_true():
    for q in _qs("proof_circle_theorems"):
        _check_circle(q)


def test_circle_numbers_agree_with_a_concrete_figure():
    """Rebuild the easy and medium numeric questions as coordinates and measure the stated angle."""
    pol = lambda d: (math.cos(math.radians(d)), math.sin(math.radians(d)))
    for q in [make("proof_circle_theorems", seed=s, difficulty="easy") for s in range(N_SEEDS)]:
        m = re.search(r"∠(\w)(\w)(\w) = (\d+)°\. Find ∠\w" + "O" + r"\w", q.text)
        if m and "major arc" in q.text:
            x = int(m.group(4))
            A, B, O = pol(0), pol(2 * x), (0.0, 0.0)
            assert abs(_angle_deg(A, O, B) - 2 * x) < 1e-6
            assert re.search(rf"= {2 * x}°$", q.answer_text)
        m = re.search(r"∠\wO\w = (\d+)°\. Find ∠", q.text)
        if m and "major arc" in q.text:
            centre = int(m.group(1))
            A, B, C = pol(0), pol(centre), pol(centre + 40)
            assert abs(_angle_deg(A, C, B) - centre / 2) < 1e-6
            assert re.search(rf"= {centre // 2}°$", q.answer_text)


# ---------------------------------------------------------------------------
# The oracle must be able to fail: a false claim in each family is rejected.
# ---------------------------------------------------------------------------
class _Fake:
    def __init__(self, text: str, answer_text: str):
        self.text, self.answer_text = text, answer_text


@pytest.mark.parametrize(
    ("check", "text", "claim"),
    [
        (_check_calculus, "Given that d/dx(e^x) = e^x, prove that d/dx(e^(3x)) = 2e^(3x)", "d/dx(e^(3x)) = 2e^(3x)"),
        (_check_calculus, "prove", "∫ 1/(2x) dx = (1/3) ln x + c"),
        (_check_calculus, "prove that ∫ from 1 to 3 of 1/x dx = ln 2", "∫ from 1 to 3 of 1/x dx = ln 2"),
        (_check_number_claim, "x", "n³ + n is divisible by 6 for every integer n"),
        (_check_number_claim, "x", "if n² is divisible by 4 then n is divisible by 4"),
        (_check_irrational_claim, "x", "√16 is irrational"),
        (_check_irrational_claim, "x", "log_2 8 is irrational"),
        (_check_pigeon_claim, "x", "among 7 people, at least 2 were born in the same month of the year"),
        (_check_pigeon_claim, "x", "among any 5 integers, there are two whose sum or difference is divisible by 8"),
        (
            _check_independence,
            "A fair 6-sided die, numbered 1 to 6, is rolled once. Event A is that the number is even; event B is that the number is at most 3. Prove it.",
            "A and B are independent",
        ),
        (
            _check_independence,
            "Events A and B are independent, with P(A) = 1/4 and P(B) = 1/2. Prove that the probability that at least one of A and B happens is 1/2. ",
            "P(A∪B) = 1/2",
        ),
        (
            _check_shape,
            "Each interior angle of a regular polygon is 150°. Prove that the polygon has 10 sides",
            "the polygon has 10 sides",
        ),
    ],
)
def test_oracle_rejects_a_false_claim(check, text, claim):
    with pytest.raises(AssertionError):
        check(_Fake(text, claim))
