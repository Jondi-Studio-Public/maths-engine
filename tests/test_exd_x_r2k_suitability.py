"""Tests for the exam sources of the Bernoulli and binomial suitability skills.

Every answer, and the final yes or no, is re-derived with exact fractions from the numbers in the rendered stem.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_suitability  # noqa: F401  (registers the topics)
from tests import exd_x_r2k_kit as kit

BERNOULLI = "senior-identify-contexts-suitable-for-modelling-by-bernoulli-random"
BINOMIAL = "senior-identify-contexts-suitable-for-modelling-by-binomial-random"
_POP = re.compile(r"(\d+) [a-z ]+?(?:,| contains| holds)[^.]*?(\d+) (?:of which|that are)")


def _numbers(q):
    stem = q.stem
    first = stem.split(". ")[0]
    nums = [int(v) for v in re.findall(r"\d+", first)]
    return nums[0], nums[1], stem


def bernoulli_oracle(q) -> dict[str, str]:
    big_n, big_k, stem = _numbers(q)
    m, j = (
        int(v) for v in re.search(r"first (\d+) [a-z ]+? are examined and set aside, and (\d+) of them", stem).groups()
    )
    t = int(re.search(r"falls by less than (\d+)%", stem).group(1))
    a = Fraction(big_k, big_n)
    b = Fraction(big_k - j, big_n - m)
    fall = (a - b) / a * 100
    return {
        "a": kit.frac_text(a),
        "b": kit.frac_text(b),
        "c": kit.fixed(fall, 1),
        "d": "yes" if fall < t else "no",
    }


def binomial_oracle(q) -> dict[str, str]:
    big_n, big_k, stem = _numbers(q)
    n = int(re.search(r"A sample of (\d+) ", stem).group(1))
    t = int(re.search(r"by less than (\d+)% of the exact", stem).group(1))
    a = Fraction(big_k, big_n) ** n
    b = Fraction(1)
    for i in range(n):
        b *= Fraction(big_k - i, big_n - i)
    over = (a - b) / b * 100
    return {
        "a": kit.frac_text(a),
        "b": kit.frac_text(b),
        "c": kit.fixed(over, 1),
        "d": "yes" if over < t else "no",
    }


def test_wiring_and_shapes():
    for skill_id, topic in (
        (BERNOULLI, "exd_x_r2k_bernoulli_suitability"),
        (BINOMIAL, "exd_x_r2k_binomial_suitability"),
    ):
        kit.check_wiring(skill_id, topic)
        kit.check_shapes(skill_id)
        kit.check_no_defects(skill_id)
        kit.check_determinism(skill_id)


def test_bernoulli_oracle_over_300_seeds():
    bad = [s for s in range(300) if bernoulli_oracle(kit.exam_q(BERNOULLI, s)) != kit.canon(kit.exam_q(BERNOULLI, s))]
    assert bad == []


def test_binomial_oracle_over_300_seeds():
    bad = [s for s in range(300) if binomial_oracle(kit.exam_q(BINOMIAL, s)) != kit.canon(kit.exam_q(BINOMIAL, s))]
    assert bad == []


def test_both_decisions_occur():
    for skill_id in (BERNOULLI, BINOMIAL):
        decisions = [kit.canon(kit.exam_q(skill_id, s))["d"] for s in range(120)]
        assert 30 <= decisions.count("yes") <= 90, (skill_id, decisions.count("yes"))


def test_grading_and_tags():
    for skill_id in (BERNOULLI, BINOMIAL):
        kit.check_grading(skill_id, min_parts=4)
        kit.check_tags(skill_id)


def test_answer_space():
    for skill_id in (BERNOULLI, BINOMIAL):
        kit.check_answer_space(skill_id)


def test_bernoulli_follow_through_from_a_wrong_start():
    def chain(q):
        c = kit.canon(q)
        a_wrong = Fraction(c["a"]) / 2
        b = Fraction(c["b"])
        fall = (a_wrong - b) / a_wrong * 100
        if fall <= 0:
            a_wrong = Fraction(c["a"]) * 2
            fall = (a_wrong - b) / a_wrong * 100
        return "a", {**c, "a": kit.frac_text(a_wrong), "c": kit.fixed(fall, 1)}

    sk = kit.skill(BERNOULLI)
    checked = 0
    for seed in range(100):
        q = kit.exam_q(BERNOULLI, seed)
        label, replies = chain(q)
        a_wrong = Fraction(replies["a"])
        fall = (a_wrong - Fraction(q.parts[1].answer_text)) / a_wrong * 100
        if not 0 < fall < 100 or not 0 < a_wrong < 1:
            continue
        result = kit.grade(sk, q, replies)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through
        checked += 1
    assert checked >= 50


def test_binomial_follow_through_from_a_wrong_exact_probability():
    sk = kit.skill(BINOMIAL)
    checked = 0
    for seed in range(100):
        q = kit.exam_q(BINOMIAL, seed)
        c = kit.canon(q)
        b_wrong = Fraction(c["b"]) / 2
        over = (Fraction(c["a"]) - b_wrong) / b_wrong * 100
        replies = {**c, "b": kit.frac_text(b_wrong), "c": kit.fixed(over, 1)}
        result = kit.grade(sk, q, replies)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through
        checked += 1
    assert checked == 100
