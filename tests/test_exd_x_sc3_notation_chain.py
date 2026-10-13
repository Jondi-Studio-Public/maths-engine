"""Exam tier of `algebraic-notation`: two readings of nearly the same notation, then a comparison."""

from __future__ import annotations

import re

import exd_x_sc3_support as sup
from sympy import Rational, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    split_symbols,
    standard_transformations,
)

SKILL = "algebraic-notation"
TOPIC = "exd_x_sc3_notation_chain"
SEEDS = range(300)
_TRANSFORMS = standard_transformations + (split_symbols, implicit_multiplication_application)
a_, b_ = symbols("a b")


def _value(text: str, env: dict):
    """The value of notation such as '3ab²' or '(2 - a)²' at `env`, read by SymPy's implicit-multiplication rules."""
    return parse_expr(text.replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"a": a_, "b": b_}).subs(env)


def oracle(q) -> dict[str, str]:
    env = {a_: int(re.search(r"a = (-?\d+)", q.stem).group(1))}
    if m := re.search(r"b = (-?\d+)", q.stem):
        env[b_] = int(m.group(1))
    first = _value(re.fullmatch(r"Find the value of (.+)\.", q.parts[0].prompt).group(1), env)
    second = _value(re.fullmatch(r"Find the value of (.+)\.", q.parts[1].prompt).group(1), env)
    third_prompt = q.parts[2].prompt
    if m := re.fullmatch(r"How many times as large as (.+) is (.+)\?", third_prompt):
        third = _value(m.group(2), env) / _value(m.group(1), env)
    else:
        third = _value(re.fullmatch(r"Hence find the value of (.+)\.", third_prompt).group(1), env)
    assert all(v == Rational(int(v)) for v in (first, second, third))
    return {"a": str(first), "b": str(second), "c": str(third)}


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert oracle(q) == sup.canonical(q), (seed, q.text)


def test_the_two_readings_differ_and_all_forms_occur():
    forms = set()
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert q.parts[0].answer_text != q.parts[1].answer_text
        forms.add(re.sub(r"[-\d]+", "#", q.parts[0].prompt))
    assert len(forms) >= 3


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_answer_space():
    sup.assert_answer_space(TOPIC, minimum=100)


def test_follow_through_from_wrong_evaluations():
    checked = {"ratio": 0, "difference": 0}
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        first, second = (int(p.answer_text) for p in q.parts[:2])
        ratio = q.parts[2].follow_through == "part_b/part_a"
        divisors = [d * sign for d in range(2, abs(second) + 1) if second % d == 0 for sign in (1, -1)]
        # A slip in (a): the comparison made from the slipped value is still marked right.
        for wrong in [first + 11, first - 7, first * 2 + 1, *divisors]:
            if wrong in (0, first):
                continue
            if ratio:
                if second % wrong:
                    continue
                reply = str(second // wrong)
            else:
                reply = str(second - wrong)
            if reply in (str(wrong), str(second), "0", q.parts[2].answer_text):
                continue
            result = sup.grade(sup.skill(SKILL), q, {"a": str(wrong), "b": str(second), "c": reply})
            assert result.parts["a"].marks == 0
            assert result.parts["c"].follow_through is True and result.parts["c"].marks == 2, (q.text, wrong, reply)
            checked["ratio" if ratio else "difference"] += 1
            break
    assert min(checked.values()) >= 30, checked
