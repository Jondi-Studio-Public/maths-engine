"""Tests the easy, medium and hard tiers of `limit-degree-comparison` (topic `w6_limit_degree`).

The oracle reads the expression, the direction and the question type back out of the TEXT and finds the limit with
SymPy, never with the generator's own leading-term rule.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tests import w7b_support as sup

TOPIC = "w6_limit_degree"
EXAM = "exd_x_r3d_limit_growth"
SKILL = sup.skill_of("limit-degree-comparison")
X = sympy.Symbol("x")
P = sympy.Symbol("p")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(text: str):
    text = text.replace("√", "sqrt").replace("−", "-")
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"x": X, "p": P})


def _expression(text: str) -> tuple[str, int]:
    """The function text and the direction (+1 or -1) of a 'limit ... as x -> ...' question."""
    m = re.search(r"limit as x → (-?∞) of (.+?)\. Give", text)
    if m:
        arrow, func = m.groups()
    else:
        func, arrow = re.search(r"limit of (.+?) as x → (-?∞)", text).groups()
    return func, (-1 if arrow.startswith("-") else 1)


def _answer_value(text: str):
    return {"∞": sympy.oo, "-∞": -sympy.oo}.get(text) if text in ("∞", "-∞") else sympy.Rational(text)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "expression")
    ids = {m.id for m in SKILL.misconceptions}
    assert {"w6-limit-direction-sign-ignored", "w6-limit-root-of-x-squared-taken-as-x"} <= ids
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard")]
    assert steps == sorted(steps) and len(set(steps)) == 3


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_at_least_twelve_distinct_answers_in_sixty_seeds(tier):
    assert len(sup.answer_spread(TOPIC, tier, 60)) >= 12


def test_the_limit_questions_match_sympy():
    seen = {"easy": 0, "medium": 0, "hard": 0}
    for tier in sup.TIERS:
        for seed in range(40):
            q = sup.single(TOPIC, tier, seed)
            if "variable" in q.text or "whole number" in q.text:
                continue
            func, direction = _expression(q.text)
            value = sympy.limit(_parse(func), X, direction * sympy.oo)
            assert value == _answer_value(q.answer_text), (tier, seed, q.text, q.answer_text, value)
            seen[tier] += 1
    assert all(n >= 15 for n in seen.values()), seen


def test_the_power_questions_match_sympy():
    checked = 0
    for seed in range(120):
        q = sup.single(TOPIC, "easy", seed)
        if "p is a whole number" not in q.text:
            continue
        m = re.search(r"limit of \((.+?)\) / \((.+?)\) as x", q.text) or re.search(
            r"\((.+?)\) / \((.+?)\) has no", q.text
        )
        num, den = _parse(m.group(1)), _parse(m.group(2))
        key = int(q.answer_text)

        def lim(p):
            return sympy.limit(num.subs(P, p) / den, X, sympy.oo)

        if "largest" in q.text:
            assert lim(key) == 0 and lim(key + 1) != 0, (seed, q.text, key)
        else:
            assert lim(key) in (sympy.oo, -sympy.oo) and lim(key - 1) not in (sympy.oo, -sympy.oo), (seed, q.text, key)
        checked += 1
    assert checked >= 20


def test_hard_covers_factored_and_root_forms_and_both_directions():
    texts = [sup.single(TOPIC, "hard", s).text for s in range(80)]
    assert any("√" in t for t in texts) and any("(x^2 +" in t for t in texts)
    assert any("→ -∞" in t for t in texts)


def test_medium_covers_sums_and_minus_infinity():
    texts = [sup.single(TOPIC, "medium", s).text for s in range(80)]
    assert any(") + (" in t for t in texts) and any("→ -∞" in t for t in texts)


def test_no_python_notation_or_double_negatives_in_text():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single(TOPIC, tier, seed)
            assert "**" not in q.text and "+ -" not in q.text and "- -" not in q.text, q.text
