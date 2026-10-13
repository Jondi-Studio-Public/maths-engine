"""The multi-part exam tier `exd_x_r2b_exp_model_chain` of
senior-use-exponential-functions-and-their-derivatives-to-solve: N(t) = A e^(kt) with k unknown.

The oracle reads A, the initial rate (and whether N is increasing or decreasing) and the target out of the rendered
text and re-derives every key with SymPy; it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_exp_model_chain"
SKILL_ID = "senior-use-exponential-functions-and-their-derivatives-to-solve"
SEEDS = range(300)


def read(q):
    big_a = int(re.search(r"N\(t\) = (\d+)e", q.stem).group(1))
    rate = sympy.Rational(re.search(r"(?:rate of|at) (\d+(?:\.\d+)?) \w+ per", q.stem).group(1))
    sign = -1 if "decreasing" in q.stem else 1
    target = int(re.search(r"N = (\d+)", q.parts[1].prompt).group(1))
    return big_a, sign * rate, target


def oracle(q):
    big_a, r0, target = read(q)
    k = r0 / big_a  # N'(0) = kA
    t = sympy.log(sympy.Rational(target, big_a)) / k
    return {
        "a": sup.dp(k, 2),
        "b": sup.dp(k * target, 2),
        "c": sup.dp(t, 2),
        "d": sup.dp((target - big_a) / sympy.Float(sup.dp(t, 2), 30), 2),
    }


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["number"] * 4, [2, 1, 2, 2], [0, 1, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        assert oracle(q) == {p.label: p.answer_text for p in q.parts}, (seed, q.stem)


def test_both_directions_are_set():
    signs = {read(sup.exam_question(TOPIC, s))[1] > 0 for s in range(40)}
    assert signs == {True, False}


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_k():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        big_a, _, target = read(q)
        wrong = q.parts[0].distractors[0]
        k = sup.to_rational(wrong)
        t = sympy.log(sympy.Rational(target, big_a)) / k
        if t <= 0:
            continue
        t_text = sup.dp(t, 2)
        avg = (target - big_a) / sympy.Float(t_text, 30)
        reply = {"a": wrong, "b": sup.dp(k * target, 2), "c": t_text, "d": sup.dp(avg, 2)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        if result.parts["b"].marks == 1:
            checked += 1
            assert result.parts["b"].follow_through
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, reply)
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 30
