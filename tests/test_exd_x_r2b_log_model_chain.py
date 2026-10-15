"""The multi-part exam tier `exd_x_r2b_log_model_chain` of
senior-use-logarithmic-functions-and-their-derivatives-to-solve: V(x) = a ln(x) + b with a unknown.

The oracle reads b, x0 and the two rates out of the rendered text and re-derives every key with SymPy; it never
imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_log_model_chain"
SKILL_ID = "senior-use-logarithmic-functions-and-their-derivatives-to-solve"
SEEDS = range(300)


def read(q):
    b_sign, b_abs = re.search(r"a ln\(x\) ([+-]) (\d+)", q.stem).groups()
    b = int(b_abs) * (1 if b_sign == "+" else -1)
    x0 = int(re.search(r"[Ww]hen x = (\d+)", q.stem).group(1))
    r0 = sympy.Rational(str(re.search(r"is (\d+(?:\.\d+)?) points? per", q.stem).group(1)))
    r1 = sympy.Rational(str(re.search(r"is (\d+(?:\.\d+)?) points? per", q.parts[1].prompt).group(1)))
    return b, x0, r0, r1


def oracle(q):
    b, x0, r0, r1 = read(q)
    a = r0 * x0  # V'(x) = a / x
    x1 = a / r1
    v1 = a * sympy.log(x1) + b
    v0 = a * sympy.log(x0) + b
    return {
        "a": str(a),
        "b": str(x1),
        "c": sup.dp(v1, 2),
        "d": sup.dp((v1 - v0) / (x1 - x0), 3),
    }


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "fraction", "number", "number"], [2, 1, 2, 2], [0, 1, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        assert oracle(q) == {p.label: p.answer_text for p in q.parts}, (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_a():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        b, x0, _, r1 = read(q)
        wrong = q.parts[0].distractors[0]
        a = sup.to_rational(wrong)
        x1 = a / r1
        if x1 == x0:
            continue
        v1, v0 = a * sympy.log(x1) + b, a * sympy.log(x0) + b
        avg = (sympy.Float(sup.dp(v1, 2), 30) - v0) / (x1 - x0)
        reply = {"a": wrong, "b": str(x1), "c": sup.dp(v1, 2), "d": sup.dp(avg, 3)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        if result.parts["b"].marks == 1:
            checked += 1
            assert result.parts["b"].follow_through
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, reply)
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (seed, reply, q.stem)
    assert checked >= 30
