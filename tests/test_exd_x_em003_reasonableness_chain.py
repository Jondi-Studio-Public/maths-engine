"""Tests the exam chain for ACMEM003: estimate by 1 significant figure, judge a stated answer, exact value, percentage error."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import round_half_up
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

TOPIC = "exd_x_em003_reasonableness_chain"
SKILL_ID = "senior-ascertain-the-reasonableness-of-answers-to-arithmetic-calculations"

REASONABLE = "Reasonable: the answer is about the size of the estimate"
LARGE = "Not reasonable: the answer is about 10 times too large"
SMALL = "Not reasonable: the answer is about 10 times too small"


def round_one_figure(n: int) -> int:
    """n to 1 significant figure, written out digit by digit rather than with the generator's helper."""
    digits = str(n)
    lead, rest = int(digits[0]), int(digits[1:]) if len(digits) > 1 else 0
    up = Fraction(rest, 10 ** (len(digits) - 1)) >= Fraction(1, 2)
    return (lead + up) * 10 ** (len(digits) - 1)


def stated_answer(q) -> int:
    """The calculator answer in the prompt of (b)."""
    prompt = [p for p in q.parts if p.label == "b"][0].prompt
    return int(re.search(r"\$?(\d[\d,]*)", prompt).group(1).replace(",", ""))


def oracle(q):
    """(a) to (d) from the two numbers after "worked out as" and the number in the prompt of (b)."""
    a, b = (int(x) for x in re.search(r"worked out as (\d+) × (\d+)\.", q.text).groups())
    estimate = round_one_figure(a) * round_one_figure(b)
    exact = a * b
    stated = stated_answer(q)
    ratio = Fraction(stated, estimate)
    if ratio > 5:
        verdict = LARGE
    elif ratio < Fraction(1, 5):
        verdict = SMALL
    else:
        verdict = REASONABLE
    percent = round_half_up(Fraction(abs(estimate - exact) * 100, exact))
    return estimate, verdict, exact, percent


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_wired():
    check_wired(SKILL_ID, TOPIC)


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "mcq", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(len(p.distractors) == 3 for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        estimate, verdict, exact, percent = oracle(q)
        a, b, c, d = canonical(q).values()
        if (Fraction(a), b, Fraction(c), Fraction(d)) != (estimate, verdict, exact, percent):
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_all_three_verdicts_occur_and_the_stated_answer_is_never_the_exact_one(questions):
    seen = set()
    for q in questions[:300]:
        _, verdict, exact, _ = oracle(q)
        seen.add(verdict)
        assert stated_answer(q) != exact
    assert seen == {REASONABLE, LARGE, SMALL}


def test_the_estimate_is_a_real_estimate(questions):
    """Between 5% and 40% from the exact value, a whole number of dollars, and a percentage that is not a tie."""
    for q in questions[:300]:
        estimate, _, exact, _ = oracle(q)
        gap = Fraction(abs(estimate - exact), exact)
        assert Fraction(5, 100) <= gap <= Fraction(40, 100), q.text
        assert abs(gap * 100 % 1 - Fraction(1, 2)) > Fraction(1, 20), q.text


def test_distractors_have_one_error_each(questions):
    for q in questions[:120]:
        for p in q.parts:
            assert p.answer_text not in p.distractors
            assert len(set(p.distractors)) == 3
            if p.answer_kind == "number":
                assert all(re.fullmatch(r"\d+", d) for d in p.distractors), (p.label, p.distractors)


def test_plausible_magnitudes(questions):
    for q in questions[:300]:
        estimate, _, exact, _ = oracle(q)
        assert 150 <= exact <= 60000, q.text


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_estimate():
    """A wrong estimate carries into (d): the learner's percentage is |exact - estimate| / exact of their own estimate."""

    def carried(q):
        estimate, _, exact, percent = oracle(q)
        for factor in (2, 3, 4, 5):
            wrong = estimate * factor
            carried_percent = Fraction(abs(exact - wrong) * 100, exact)
            if (
                abs(carried_percent % 1 - Fraction(1, 2)) > Fraction(1, 20)
                and round_half_up(carried_percent) != percent
            ):
                return "a", str(wrong), {"d": str(round_half_up(carried_percent))}
        raise AssertionError(q.text)

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_answer_space():
    check_answer_space(TOPIC)


def test_prompt_shapes():
    check_shapes(TOPIC, "reasonableness_verdict")


def test_unknown_tier_is_refused():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="easy")
