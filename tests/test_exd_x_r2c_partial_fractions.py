"""Tests the exam chain for partial fractions: the whole-number part, A, B and the exact definite integral."""

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

TOPIC = "exd_x_r2c_partial_fractions"
SKILL_ID = "senior-use-partial-fractions-where-necessary-for-integration-in"
HARD_TOPIC = "use_partial_fractions_where_necessary_for_integration_in"
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def oracle(q):
    """k, A, B and the integral, re-derived from the rendered fraction, the stated form and the limits."""
    found = re.search(r"f\(x\) = \((.+?)\)/\((.+?)\)(?:\.| is to| ,|,)", q.stem)
    f = _parse(found.group(1)) / _parse(found.group(2))
    form = re.search(r"k \+ A/(.+?) \+ B/(.+?), where", q.stem)
    den_a, den_b = _parse(form.group(1)), _parse(form.group(2))
    k, a, b = sympy.symbols("k a b")
    points = [sympy.Rational(7, 2), sympy.Rational(11, 3), sympy.Rational(-13, 5)]
    solved = sympy.solve([k + a / den_a.subs(x, t) + b / den_b.subs(x, t) - f.subs(x, t) for t in points], [k, a, b])
    lo, hi = (int(v) for v in re.search(r"from x = (-?\d+) to x = (-?\d+)", q.parts[3].prompt).groups())
    with mp.workdps(30):
        integral = mp.quad(sympy.lambdify(x, f, "mpmath"), [lo, hi])
    return solved[k], solved[a], solved[b], integral


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
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "expression"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 3]
        assert all(name in q.parts[3].follow_through for name in ("part_a", "part_b", "part_c"))
        assert q.parts[2].follow_through


def test_both_shapes_occur(questions):
    kinds = {")^2" in q.stem.split("can be")[1] or "^2," in q.stem for q in questions[:80]}
    assert kinds == {True, False}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        k, a, b, integral = oracle(q)
        texts = [p.answer_text for p in q.parts]
        got = mp.mpf(str(sympy.N(_parse(texts[3]), 30)))
        if not ([sympy.Rational(t) for t in texts[:3]] == [k, a, b] and abs(got - integral) < mp.mpf("1e-9")):
            bad.append((q.stem, canonical(q), (k, a, b)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def _value(text: str) -> sympy.Expr:
    return _parse(text)


def test_follow_through_from_wrong_k():
    """A wrong k carried through (c) and (d): the later parts earn their marks."""

    def carried(q):
        a = _value(q.parts[1].answer_text)
        for shift in (5, 6, 7, 8):  # a wrong k whose next answer is neither zero nor a repeat of an earlier reply
            k_wrong = int(q.parts[0].answer_text) + shift
            b_wrong = _parse(q.parts[2].follow_through.replace("part_a", f"({k_wrong})").replace("part_b", f"({a})"))
            if b_wrong not in (0, a, k_wrong):
                break
        d_follow = q.parts[3].follow_through
        d_wrong = _parse(
            d_follow.replace("part_a", f"({k_wrong})").replace("part_b", f"({a})").replace("part_c", f"({b_wrong})")
        )
        return "a", str(k_wrong), {"c": str(b_wrong), "d": str(sympy.expand_log(d_wrong, force=True))}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
