"""Tests the steady-state skill for first-order linear recurrence relations (ACMGM077).

Covers ``mathgen/generators/pv0a_steady_state_build.py`` (topic
``pv0a_recurrence_steady_state``).
"""

from dataclasses import replace

import functools
import re
from fractions import Fraction

import mathgen.generators.pv0a_steady_state_build  # noqa: F401 -- registers the topic
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, round_half_up

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
TOPIC = "pv0a_recurrence_steady_state"
SKILL_ID = "senior-use-first-order-linear-recurrence-relations-to-model"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if src.difficulty == "exam":
        # The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
        src = replace(src, topic="pv0a_recurrence_steady_state", params={"variant": "exam"})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Determinism and basic shape
# =============================================================================


def test_deterministic_across_seeds():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            b = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.text == b.text
            assert a.answer_text == b.answer_text
            assert a.distractors == b.distractors


def test_distractors_distinct_nonempty_never_equal_answer():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert len(q.distractors) >= 1, (difficulty, seed)
            assert len(set(q.distractors)) == len(q.distractors), (difficulty, seed)
            assert q.answer_text not in q.distractors, (difficulty, seed)


def test_at_least_20_distinct_questions_per_tier_over_60_seeds():
    for difficulty in DIFFICULTIES:
        texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
        assert len(texts) >= 20, (difficulty, len(texts))


def test_at_least_two_distractors_on_85_percent_of_draws():
    for difficulty in DIFFICULTIES:
        counts = [len(make(TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200)]
        at_least_two = sum(1 for c in counts if c >= 2)
        assert at_least_two / 200 >= 0.85, (difficulty, at_least_two)


def test_skill_loads_with_expected_shape():
    skill = _skills_dict()[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == set(DIFFICULTIES)
    assert next(s for s in skill.sources if s.difficulty == "exam").stages == ["exam"]
    assert next(s for s in skill.sources if s.difficulty == "easy").stages == [0, 1, 2]
    assert next(s for s in skill.sources if s.difficulty == "medium").stages == [3, 4]
    assert next(s for s in skill.sources if s.difficulty == "hard").stages == [5, 6, "maintenance"]


# =============================================================================
# Real-grader accept/reject
# =============================================================================


def test_number_grader_rejects_a_deliberately_wrong_answer():
    grader = GRADERS["number"]
    for difficulty in DIFFICULTIES:
        q = _make_from_sources(difficulty, 0)
        if "." in q.answer_text:
            whole, frac = q.answer_text.lstrip("-").split(".")
            wrong = str(int(whole) + 97) + "." + frac
        else:
            wrong = str(int(q.answer_text) + 97)
        assert not grader(q, wrong).correct, (difficulty, q.text, wrong)


def test_easy_and_medium_and_exam_answers_carry_the_context_decimal_precision():
    # Every context is either 0dp (fish/members) or 1dp (mg/kg/g); an easy-tier answer's decimal
    # count must match whichever the question states.
    for difficulty in ("easy", "medium", "exam"):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if "to 1 decimal place" in q.text:
                assert "." in q.answer_text and len(q.answer_text.split(".")[1]) == 1, (seed, q.text)
            else:
                assert "." not in q.answer_text, (seed, q.text)


def test_integer_answer_has_no_decimal_point():
    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if "to the nearest whole number" in q.text:
                assert "." not in q.answer_text, (difficulty, seed, q.answer_text)
                assert isinstance(q.answer, int), (difficulty, seed, q.answer)


# =============================================================================
# Hard-tier boundary/plausibility property, proven RED first
# =============================================================================

_HARD_AFTER_RE = re.compile(r"u\(n\+1\) = (\d(?:\.\d)?)u\(n\) \+ (\d+)")
_HARD_BEFORE_RE = re.compile(r"u\(n\+1\) = (\d(?:\.\d)?)\(u\(n\) \+ (\d+)\)")


def _given_points_own_fixed_value(text: str) -> Fraction:
    """The fixed point of whichever recurrence FORM is literally stated in the text -- i.e. the
    value the recurrence's own u(n) settles to, NOT necessarily what the question asks for."""
    m_before = _HARD_BEFORE_RE.search(text)
    if m_before:
        a, b = Fraction(m_before.group(1)), Fraction(m_before.group(2))
        return a * b / (1 - a)  # pre-event fixed point of u(n+1) = a(u(n) + b)
    m_after = _HARD_AFTER_RE.search(text)
    a, b = Fraction(m_after.group(1)), Fraction(m_after.group(2))
    return b / (1 - a)


def test_hard_tier_asks_about_the_other_point_not_the_given_one():
    """recurrence's own u(n) was given at, rather than genuinely the OTHER point. Proven red first by
    temporarily replacing `_hard`'s `answer_value` with `given_value` (the point NOT asked for) and
    rerunning: the very first seed already failed --
        AssertionError: (0, '32.0', '32')
    confirming this test only passes because the
    generator genuinely answers the other point, not the stated one."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        d = 0 if "to the nearest whole number" in q.text else 1
        given_rounded = round_half_up(_given_points_own_fixed_value(q.text), d)
        answer_value = Fraction(q.answer_text)
        assert given_rounded != answer_value, (seed, q.answer_text, str(given_rounded))


def test_check_fires_on_a_corrupted_steady_state_claim():
    from mathgen.generators.pv0a_steady_state_build import _verify_steady_state

    fired = False
    try:
        _verify_steady_state(Fraction(1, 2), Fraction(10), Fraction(100))  # true L is 20, not 100
    except VerificationError:
        fired = True
    assert fired


def test_check_accepts_the_true_steady_state():
    from mathgen.generators.pv0a_steady_state_build import _verify_steady_state

    _verify_steady_state(Fraction(1, 2), Fraction(10), Fraction(20))  # b/(1-a) = 10/0.5 = 20
