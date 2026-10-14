"""Exam tier of `apply-a-rule-to-inputs`: a real formula at two inputs, then the change between the results."""

from __future__ import annotations

import re
from fractions import Fraction

import exd_x_sc3_support as sup

SKILL = "apply-a-rule-to-inputs"
TOPIC = "exd_x_sc3_apply_rule_chain"
SEEDS = range(300)


def _inputs(q) -> tuple[int, int]:
    out = []
    for p in q.parts[:2]:
        m = re.search(r"speed is (\d+) m/s|there are (\d+) teams|after (\d+) seconds?|after (\d+) weeks", p.prompt)
        out.append(int(next(g for g in m.groups() if g)))
    return out[0], out[1]


def _rule(q):
    stem = q.stem
    if "kinetic energy" in stem:
        m = int(re.search(r"mass of (\d+) kg", stem).group(1))
        return lambda v: Fraction(m * v * v, 2)
    if "round-robin" in stem:
        return lambda n: Fraction(n * (n - 1), 2)
    if "cricket ball" in stem:
        u = int(re.search(r"at (\d+) metres per second", stem).group(1))
        return lambda t: Fraction(u * t - 5 * t * t)
    a, d = (int(g) for g in re.search(r"Here a = (\d+) and d = (\d+)", stem).groups())
    return lambda n: Fraction(n, 2) * (2 * a + (n - 1) * d)


def oracle(q) -> dict[str, str]:
    rule = _rule(q)
    x1, x2 = _inputs(q)
    r1, r2 = rule(x1), rule(x2)
    assert r1.denominator == r2.denominator == 1
    return {"a": str(int(r1)), "b": str(int(r2)), "c": str(int(r2 - r1))}


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert oracle(q) == sup.canonical(q), (seed, q.text)


def test_all_four_formulas_occur():
    stems = " ".join(sup.exam(TOPIC, s).stem for s in range(60))
    for word in ("kinetic energy", "round-robin", "cricket ball", "saves"):
        assert word in stems


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_follow_through_from_a_wrong_first_evaluation():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        a, b = (int(p.answer_text) for p in q.parts[:2])
        wrong_a = next(a + o for o in (17, 19, 23) if b - (a + o) not in (a + o, b, 0))
        result = sup.grade(sup.skill(SKILL), q, {"a": str(wrong_a), "b": str(b), "c": str(b - wrong_a)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 2
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (q.text, a, b)


def test_follow_through_from_a_wrong_second_evaluation():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        a, b = (int(p.answer_text) for p in q.parts[:2])
        wrong_b = next(b - o for o in (13, 11, 7) if (b - o) - a not in (a, b - o, 0))
        result = sup.grade(sup.skill(SKILL), q, {"a": str(a), "b": str(wrong_b), "c": str(wrong_b - a)})
        assert result.parts["b"].marks == 0 and result.parts["c"].follow_through is True, (q.text, a, b)
