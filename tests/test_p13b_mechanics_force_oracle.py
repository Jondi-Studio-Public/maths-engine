"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p13b_mechanics_force_build` -- every
number (mass, velocity, force, friction, displacement, force constant) is
parsed back out of the rendered prose with regular expressions, then
re-derived from first physics principles (signed momentum, force
combination, F=ma, momentum conservation, SUVAT, and -- for the
non-constant-force tiers -- v*dv/dx = a(x) solved with `sympy.solve`), never
by calling into the generator's own construction.
"""

import re
from fractions import Fraction

import pytest
from sympy import Rational, solve, symbols

from mathgen import make

SEEDS = range(125)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC_A = "p13b_momentum_force_reaction"
TOPIC_B = "p13b_constant_nonconstant_force"

x_sym, v_sym = symbols("x v")


# =====================================================================
# Skill A: momentum, resultant force, Newton's second and third laws
# =====================================================================


def _first_int(pattern: str, text: str, group: int = 1) -> int:
    m = re.search(pattern, text)
    assert m is not None, f"pattern {pattern!r} not found in: {text!r}"
    return int(m.group(group))


def _all_ints(pattern: str, text: str) -> list:
    return [int(g) for g in re.findall(pattern, text)]


@pytest.mark.parametrize("tier", TIERS)
def test_skill_a_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=tier)
        rendered = int(q.answer_text)

        if tier == "easy":
            m = _first_int(r"(\d+) kg", q.text)
            v = _first_int(r"(\d+) m/s", q.text)
            direction = re.search(r"\b(right|left)\b", q.text).group(1)
            oracle = m * v if direction == "right" else -m * v

        elif tier == "medium":
            f1, f2 = _all_ints(r"(\d+) N", q.text)[:2]
            oracle = f1 - f2

        elif tier == "hard":
            m = _first_int(r"(\d+) kg", q.text)
            f1, f2, friction = _all_ints(r"(\d+) N", q.text)[:3]
            net_applied = f1 - f2
            sign = 1 if net_applied > 0 else -1
            net_force = net_applied - sign * friction
            assert net_force % m == 0, f"seed={seed}: net force {net_force} not divisible by mass {m}"
            oracle = net_force // m

        else:  # exam
            mA, mB = _all_ints(r"(\d+) kg", q.text)[:2]
            vA, absVB = _all_ints(r"(\d+) m/s", q.text)[:2]
            vB = -absVB
            num = mA * vA + mB * vB
            den = mA + mB
            assert num % den == 0, f"seed={seed}: {num} not divisible by {den}"
            oracle = num // den

        assert oracle == rendered, f"seed={seed}/{tier}: oracle {oracle} != rendered answer {rendered}\ntext: {q.text}"
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_a_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=tier)
        rendered = int(q.answer_text)
        for d in q.distractors:
            assert int(d) != rendered, f"seed={seed}/{tier}: distractor {d!r} equals the rendered answer {rendered}"


# =====================================================================
# Skill B: constant vs non-constant force
# =====================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_skill_b_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=tier)
        rendered = int(q.answer_text)
        m = _first_int(r"(\d+) kg", q.text)

        if tier == "easy":
            F = _first_int(r"force of (\d+) N", q.text)
            s = _first_int(r"travell?(?:ed|ing) (\d+) m\b", q.text)
            a = Fraction(F, m)
            v_squared = 2 * a * s
            v = int(round(float(v_squared) ** 0.5))
            assert v * v == v_squared, f"seed={seed}: 2as={v_squared} not a perfect square"
            oracle = v

        elif tier == "medium":
            F = _first_int(r"force of (\d+) N", q.text)
            u = _first_int(r"(\d+) m/s", q.text)
            m_further = re.search(r"further (\d+) m\b", q.text) or re.search(r"(\d+) more metres", q.text)
            assert m_further is not None, f"seed={seed}: could not find displacement in: {q.text!r}"
            s = int(m_further.group(1))
            a = Fraction(F, m)
            v_squared = u * u + 2 * a * s
            v = int(round(float(v_squared) ** 0.5))
            assert v * v == v_squared, f"seed={seed}: u^2+2as={v_squared} not a perfect square"
            oracle = v

        elif tier == "hard":
            k = _first_int(r"F\(x\) = (\d+)", q.text)
            s = _first_int(r"x = (\d+) m", q.text)
            # Independent route: sympy.solve on v**2 - (k/m)*x**2 == 0 (from
            # integrating v*dv/dx = (k/m)*x with v(0) = 0), evaluated at x = s.
            solved = solve(v_sym**2 - Rational(k, m) * x_sym**2, v_sym)
            positive = [sol for sol in solved if sol.subs(x_sym, s).is_positive]
            assert len(positive) == 1, f"seed={seed}: expected one positive root, got {positive}"
            v_expr = positive[0].subs(x_sym, s)
            assert v_expr.is_Integer, f"seed={seed}: hard oracle not an exact integer: {v_expr}"
            oracle = int(v_expr)

        else:  # exam
            u = _first_int(r"(\d+) m/s", q.text)
            k = _first_int(r"F\(x\) = (\d+)", q.text)
            v_target = _first_int(r"reaches (\d+) m/s", q.text)
            # Independent route: sympy.solve on
            # v_target**2 - u**2 - (k/m)*x**2 == 0.
            solved = solve(v_target**2 - u**2 - Rational(k, m) * x_sym**2, x_sym)
            positive = [sol for sol in solved if sol.is_positive]
            assert len(positive) == 1, f"seed={seed}: expected one positive root, got {positive}"
            x_expr = positive[0]
            assert x_expr.is_Integer, f"seed={seed}: exam oracle not an exact integer: {x_expr}"
            oracle = int(x_expr)

        assert oracle == rendered, f"seed={seed}/{tier}: oracle {oracle} != rendered answer {rendered}\ntext: {q.text}"
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_b_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=tier)
        rendered = int(q.answer_text)
        for d in q.distractors:
            assert int(d) != rendered, f"seed={seed}/{tier}: distractor {d!r} equals the rendered answer {rendered}"
