"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from
`mathgen.generators.p12c_trig_apply_implicit_diff_build` -- every model
(``a``, ``b``, ``c`` for skill 1; the implicit curve's coefficients for
skill 2) and every stated time/target/bound is parsed back out of the
rendered prose with regular expressions, then re-derived with
`sympy.diff`/`sympy.solve`/`sympy.idiff`, never by calling into the
generator's own construction.
"""

import re

import pytest
from sympy import Eq, Rational, cos, diff, idiff, nsolve, pi, sin, symbols, sympify

from mathgen import make
from mathgen.core import symbolically_equal

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC_1 = "p12c_trig_deriv_apply"
TOPIC_2 = "p12c_implicit_diff_gradient"

t, x, y = symbols("t x y")


# =====================================================================
# Skill 1: trigonometric-derivative practical applications
# =====================================================================

_MODEL_RE = re.compile(
    r"modelled by \w+\(t\) = (?:(?P<a>\d+)·)?sin\((?:(?P<b>\d+)·)?t\)"
    r"(?: (?P<sign>[+-]) (?P<c>\d+))?,"
)


def _parse_model(text: str):
    m = _MODEL_RE.search(text)
    assert m is not None, f"could not parse model from: {text!r}"
    a = int(m.group("a")) if m.group("a") else 1
    b = int(m.group("b")) if m.group("b") else 1
    if m.group("c") is None:
        c = 0
    else:
        c = int(m.group("c")) if m.group("sign") == "+" else -int(m.group("c"))
    return a, b, c


_PI_TEXT_RE = re.compile(r"(?P<num>\d*)π(?:/(?P<den>\d+))?")


def _parse_pi_time(token: str) -> Rational:
    """ "π" -> 1, "π/8" -> 1/8, "2π/3" -> 2/3 (the coefficient
    of pi -- the token IS the full time value, already t = angle/b)."""
    m = _PI_TEXT_RE.fullmatch(token)
    assert m is not None, f"could not parse a pi-fraction time from: {token!r}"
    num = int(m.group("num")) if m.group("num") else 1
    den = int(m.group("den")) if m.group("den") else 1
    return Rational(num, den)


def _dHdt(a, b, c):
    H = a * sin(b * t) + c
    return diff(H, t)


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        a, b, c = _parse_model(q.text)
        dHdt = _dHdt(a, b, c)
        rendered = float(q.answer_text)

        if tier == "easy":
            oracle_val = float(dHdt.subs(t, 0))
        elif tier == "medium":
            m = re.search(r"at t = (?P<time>\S+) \w+, in", q.text)
            assert m is not None, f"seed={seed}: could not find medium time in: {q.text!r}"
            t1 = _parse_pi_time(m.group("time")) * pi
            oracle_val = float(dHdt.subs(t, t1))
        elif tier == "hard":
            m = re.search(r"equals (?P<target>-?\d+(?:\.\d+)?) .+?\. Give your answer", q.text)
            assert m is not None, f"seed={seed}: could not find hard target in: {q.text!r}"
            bound_m = re.search(r"For 0 ≤ t ≤ (?P<bound>-?\d+\.\d+)", q.text)
            assert bound_m is not None
            target = float(bound_m and m.group("target"))
            bound = float(bound_m.group("bound"))
            # Solve dH/dt(t) = target numerically in [0, bound]; cos(b*t) is
            # monotonic there by construction, so the root is unique.
            sol = nsolve(Eq(dHdt, target), t, bound / 2)
            oracle_val = round(float(sol), 2)
        else:  # exam
            times = re.findall(r"at t = (\S+) \w+ with the rate at t = (\S+) \w+\.", q.text)
            assert times, f"seed={seed}: could not find exam times in: {q.text!r}"
            tok1, tok2 = times[0]
            t1 = _parse_pi_time(tok1) * pi
            t2 = _parse_pi_time(tok2) * pi
            rate1 = float(dHdt.subs(t, t1))
            rate2 = float(dHdt.subs(t, t2))
            oracle_val = max(rate1, rate2)

        assert abs(oracle_val - rendered) < 0.02, (
            f"seed={seed}/{tier}: oracle {oracle_val} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        rendered = float(q.answer_text)
        for d in q.distractors:
            assert abs(float(d) - rendered) > 0.005, (
                f"seed={seed}/{tier}: distractor {d!r} equals the rendered answer {rendered}"
            )


# =====================================================================
# Skill 2: implicit differentiation
# =====================================================================

_EQ_RE = re.compile(r"(?:implicitly by|equation) (?P<eq>[^.]+? = [^.]+?)\. ")


def _parse_equation(text: str):
    m = _EQ_RE.search(text)
    assert m is not None, f"could not parse implicit equation from: {text!r}"
    eq_text = m.group("eq").replace("·", "*")
    lhs_text, rhs_text = eq_text.split("=")
    lhs = sympify(lhs_text.strip().replace("^", "**"), locals={"x": x, "y": y})
    rhs = sympify(rhs_text.strip(), locals={"x": x, "y": y})
    return lhs, rhs


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        lhs, rhs = _parse_equation(q.text)
        # sympy.idiff, a route independent of the generator's own
        # term-by-term construction.
        oracle_yprime = idiff(lhs - rhs, y, x)
        rendered = sympify(q.answer_text.replace("^", "**"), locals={"x": x, "y": y})
        assert symbolically_equal(oracle_yprime, rendered), (
            f"seed={seed}/{tier}: oracle {oracle_yprime} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        lhs, rhs = _parse_equation(q.text)
        oracle_yprime = idiff(lhs - rhs, y, x)
        for d in q.distractors:
            d_expr = sympify(d.replace("^", "**"), locals={"x": x, "y": y})
            assert not symbolically_equal(oracle_yprime, d_expr), (
                f"seed={seed}/{tier}: distractor {d!r} equals the oracle answer {oracle_yprime}"
            )
