"""The multi-part exam tier `exd_x_r2b_total_change_chain` of
senior-calculate-total-change-by-integrating-instantaneous-or-marginal: a rate with an unknown constant.

The oracle reads the rate, the first period and its total out of the rendered text and re-derives every key with
SymPy's definite integral; it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_total_change_chain"
SKILL_ID = "senior-calculate-total-change-by-integrating-instantaneous-or-marginal"
SEEDS = range(300)
t, k = sympy.symbols("t k")


def read(q):
    rate = sup.parse_poly(re.search(r"r\(t\) = (kt(?:\^\d)? \+ \d+)", q.stem).group(1), t=t, k=k)
    a = int(re.search(r"first (\d+) ", q.stem).group(1))
    total = int(re.search(r"(?:total is|is) (\d+) [\w-]+\.$", q.stem).group(1))
    lo, hi = (int(v) for v in re.search(r"t = (\d+) and t = (\d+)", q.parts[1].prompt).groups())
    return rate, a, total, hi


def oracle(q):
    rate, a, total, hi = read(q)
    k_val = sympy.solve(sympy.integrate(rate, (t, 0, a)) - total, k)[0]
    r = rate.subs(k, k_val)
    amount = sympy.integrate(r, (t, a, hi))
    avg = amount / (hi - a)
    return k_val, amount, avg, r.subs(t, hi) - avg


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "number", "number", "number"], [2, 2, 1, 1], [0, 1, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        k_val, amount, avg, gap = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert sympy.Rational(a) == k_val, (seed, a)
        assert (b, c, d) == (sup.dp(amount, 2), sup.dp(avg, 2), sup.dp(gap, 2)), (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_k():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        rate, a, _, hi = read(q)
        wrong = q.parts[0].distractors[0]
        wk = sympy.Rational(wrong)
        r = rate.subs(k, wk)
        amount_text = sup.dp(sympy.integrate(r, (t, a, hi)), 2)
        avg_text = sup.dp(sympy.Rational(amount_text) / (hi - a), 2)
        gap_text = sup.dp(r.subs(t, hi) - sympy.Rational(avg_text), 2)
        canon = [q.parts[i].answer_text for i in (1, 2, 3)]
        if [amount_text, avg_text, gap_text] == canon or wk in (sympy.Rational(amount_text), sympy.Rational(avg_text)):
            continue
        if any(sympy.Rational(v) == 0 for v in (amount_text, avg_text, gap_text)):
            continue
        checked += 1
        result = grade(sk, q, {"a": wrong, "b": amount_text, "c": avg_text, "d": gap_text})
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 2), ("c", 1), ("d", 1)):
            assert result.parts[label].marks == marks and result.parts[label].follow_through, (seed, label)
    assert checked >= 30
