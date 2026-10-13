"""Tests the exam chain for integration by parts: the cyclic integrals of e^(ax) sin(bx) and e^(ax) cos(bx)."""

from __future__ import annotations

import re

import pytest
import sympy
from mpmath import mp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_r2c_by_parts_cyclic"
SKILL_ID = "senior-integrate-by-parts"
HARD_TOPIC = "integrate_by_parts"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_STEM = re.compile(
    r"I = ∫ from x = 0 to x = (\S+) of e\^(?:x|\((\d+)x\))·(sin|cos)\((?:x|(\d+)x)\) dx and "
    r"J = ∫ from x = 0 to x = \S+ of e\^(?:x|\(\d+x\))·(sin|cos)\("
)


def _parse(text: str):
    return parse_expr(text.replace("^", "**").replace("π", "pi"), transformations=_TRANSFORMS)


def oracle(q):
    """a, b, the limit, which function is I; then the exact k, c and numerical I, J from the integrands."""
    found = _STEM.search(q.stem)
    limit_text, a, f_name, b = found.group(1), int(found.group(2) or 1), found.group(3), int(found.group(4) or 1)
    limit = sympy.N(_parse(limit_text), 30)
    funs = {"sin": mp.sin, "cos": mp.cos}
    other = "cos" if f_name == "sin" else "sin"
    with mp.workdps(30):
        edges = [mp.mpf(str(limit)) * i / 8 for i in range(9)]
        i_val = mp.quad(lambda t: mp.exp(a * t) * funs[f_name](b * t), edges)
        j_val = mp.quad(lambda t: mp.exp(a * t) * funs[other](b * t), edges)
    k = sympy.Rational(-b, a) if f_name == "sin" else sympy.Rational(b, a)
    c = 1 + sympy.Rational(b * b, a * a)
    return k, c, i_val, j_val


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, HARD_TOPIC)


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "expression", "expression"]
        assert [p.marks for p in q.parts] == [2, 2, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_both_functions_occur(questions):
    assert {_STEM.search(q.stem).group(3) for q in questions[:60]} == {"sin", "cos"}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        k, c, i_val, j_val = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        with mp.workdps(30):
            got_i = mp.mpf(str(sympy.N(_parse(c_t), 30)))
            got_j = mp.mpf(str(sympy.N(_parse(d_t), 30)))
            ok = (
                sympy.Rational(a_t) == k
                and sympy.Rational(b_t) == c
                and abs(got_i - i_val) < 1e-10 * (1 + abs(i_val))
                and abs(got_j - j_val) < 1e-10 * (1 + abs(j_val))
            )
        if not ok:
            bad.append((q.stem, canonical(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_k():
    """A wrong first-step coefficient k, carried into c, I and J, still earns the later marks."""

    def carried(q):
        k = sympy.Rational(q.parts[0].answer_text)
        found = _STEM.search(q.stem)
        a, b = int(found.group(2) or 1), int(found.group(4) or 1)
        k2 = -k
        for wrong in (k + sympy.Rational(1, 3), k + sympy.Rational(2, 3), k * 3):
            c_wrong = 1 - wrong * k2
            if wrong != 0 and c_wrong not in (0, wrong):
                break
        limit = sympy.N(_parse(found.group(1)), 30)
        i_formula = q.parts[2].follow_through.replace("part_a", f"({wrong})").replace("part_b", f"({c_wrong})")
        i_wrong = _parse(i_formula)
        j_formula = q.parts[3].follow_through.replace("part_c", f"({i_wrong})")
        j_wrong = _parse(j_formula)
        assert limit > 0 and a > 0 and b > 0
        return "a", str(wrong), {"b": str(c_wrong), "c": str(i_wrong), "d": str(j_wrong)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
