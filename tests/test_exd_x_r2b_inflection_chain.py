"""The multi-part exam tier `exd_x_r2b_inflection_chain` of
senior-understand-the-concepts-of-concavity-and-points-of: a cubic with an unknown coefficient and a stated point
of inflection.

The oracle reads the cubic, the inflection and the test point out of the rendered text and re-derives every key with
SymPy; it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_inflection_chain"
SKILL_ID = "senior-understand-the-concepts-of-concavity-and-points-of"
SEEDS = range(300)
x, k = sympy.symbols("x k")


def read(q):
    poly_text = re.search(r"f\(x\) = (.+?)(?:, where k| and k)", q.stem).group(1)
    p = int(re.search(r"x = (-?\d+)\.$", q.stem).group(1))
    x0 = int(re.search(r"Find f''\((-?\d+)\)", q.parts[1].prompt).group(1))
    return sup.parse_poly(poly_text, x=x, k=k), p, x0


def oracle(q):
    poly, p, x0 = read(q)
    k_val = sympy.solve(sympy.diff(poly, x, 2).subs(x, p), k)[0]
    f = poly.subs(k, k_val)
    second = sympy.diff(f, x, 2).subs(x, x0)
    shape = "concave up" if second > 0 else "concave down"
    slope, y0 = sympy.diff(f, x).subs(x, p), f.subs(x, p)
    tangent = sympy.expand(slope * (x - p) + y0)
    return {"a": str(k_val), "b": f"{second}, {shape}", "c": str(y0), "tangent": tangent}


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "ordered_list", "fraction", "expression"], [2, 2, 1, 2], [0, 0, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        want = oracle(q)
        got = {p.label: p.answer_text for p in q.parts}
        assert (want["a"], want["b"], want["c"]) == (got["a"], got["b"], got["c"]), (seed, q.stem)
        stated = sup.parse_poly(got["d"].removeprefix("y = "), x=x)
        assert sympy.expand(stated - want["tangent"]) == 0, (seed, got["d"])


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_k():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        poly, p, _ = read(q)
        wrong = q.parts[0].distractors[0]
        wk = sup.to_rational(wrong)
        f = poly.subs(k, wk)
        y0, slope = f.subs(x, p), sympy.diff(f, x).subs(x, p)
        tangent = sympy.expand(slope * (x - p) + y0)
        if 0 in (y0, slope) or wk in (y0, slope) or y0 == slope:
            continue
        checked += 1
        reply = {"a": wrong, "b": q.parts[1].answer_text, "c": str(y0), "d": "y = " + str(tangent).replace("**", "^")}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through, (seed, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 40
