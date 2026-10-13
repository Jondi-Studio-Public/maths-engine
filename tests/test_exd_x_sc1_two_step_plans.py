"""Exam source of `two-step-equations`: two charging plans compared through the same quantity."""

from __future__ import annotations

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "two-step-equations"
TOPIC = "exd_x_sc1_two_step_plans"


def facts(q):
    amounts = sup.money(q.stem)
    assert len(amounts) == 5
    return amounts  # fee A, rate A, total A, fee B, total B


def oracle(q):
    fee_a, rate_a, total_a, fee_b, total_b = facts(q)
    assert (total_a - fee_a) % rate_a == 0
    n = (total_a - fee_a) // rate_a
    assert (total_b - fee_b) % n == 0
    return {"a": n, "b": (total_b - fee_b) // n}


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.marks for p in q.parts] == [3, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        assert {p.label: int(p.answer_text) for p in q.parts} == oracle(q), (seed, q.text)


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_count():
    """A wrong count of units, used correctly, gives the learner's own price and still earns (b)."""
    checked = 0
    for seed in range(150):
        q = sup.exam(SKILL_ID, seed)
        _fa, _ra, _ta, fee_b, total_b = facts(q)
        truth = oracle(q)
        bad = truth["a"] + 1
        own = (total_b - fee_b) / bad
        text = f"{own:.4f}".rstrip("0").rstrip(".")
        result = grade(sup.skill(SKILL_ID), q, {"a": str(bad), "b": text})
        assert result.parts["a"].marks == 0
        # the price is graded to the whole number the question's own answer is, so only an exact one earns it
        if own == int(own) and int(own) != bad:
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through, (seed, bad, text)
            checked += 1
    assert checked >= 5


def test_answer_space_and_determinism():
    sup.check_answer_space(SKILL_ID)
    sup.check_deterministic(SKILL_ID)
