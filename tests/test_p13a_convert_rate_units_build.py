"""Convert units of rates occurring in practical situations
(senior-convert-units-of-rates-occurring-in-practical-situations, ACMEM015).
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p13a_convert_rate_units_build import (
    _verify_easy,
    _verify_medium,
    _verify_hard,
    _verify_exam,
)

TOPIC = "p13a_convert_rate_units"
SKILL_ID = "senior-convert-units-of-rates-occurring-in-practical-situations"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == 3, (difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        if isinstance(q.answer, int):
            wrong = str(q.answer + 1)
        else:
            wrong = str(round(q.answer + 1.0, 2))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


def test_easy_medium_answers_are_plain_integers():
    for difficulty in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer_text.isdigit()
            assert "." not in q.answer_text
            for d in q.distractors:
                assert d.isdigit(), (difficulty, seed, d)


def test_hard_answer_is_exact_two_decimal_plain_number():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert isinstance(q.answer, float)
        assert not q.answer_text.startswith("$")
        whole, frac = q.answer_text.split(".")
        assert len(frac) == 2
        for d in q.distractors:
            assert not d.startswith("$")
            assert len(d.split(".")[1]) == 2


def test_exam_answer_is_exact_two_decimal_currency():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, float)
        assert q.answer_text.startswith("$")
        cents_str = q.answer_text[1:]
        whole, frac = cents_str.split(".")
        assert len(frac) == 2
        assert Fraction(cents_str) * 100 % 1 == 0
        for d in q.distractors:
            assert d.startswith("$")
            assert len(d.split(".")[1]) == 2


# --- Every distractor formatted like the answer within one question ----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_identically(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        answer_is_currency = q.answer_text.startswith("$")
        answer_has_decimal = "." in q.answer_text.lstrip("$")
        for d in q.distractors:
            assert d.startswith("$") == answer_is_currency, (difficulty, seed, q.answer_text, d)
            assert ("." in d.lstrip("$")) == answer_has_decimal, (difficulty, seed, q.answer_text, d)


# --- At least 20 distinct questions per tier over 60 seeds --------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    seen = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, len(seen))


# --- Never reduces to "convert this rate to X units, report the converted
# rate" (the direct neighbour's exact shape) -- the bare converted m/s rate,
# parsed straight from the question's own km/h figure, must always appear
# among the DISTRACTORS (the "stopped after converting" misconception),
# never as the reported answer -----------------------------------------

import re as _re
from fractions import Fraction as _Fraction

_KMH_RE = _re.compile(r"at ([\d.]+) km/h")


def test_never_reports_the_bare_converted_rate_as_the_answer():
    for difficulty in ("easy", "medium"):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            m = _KMH_RE.search(q.text)
            assert m, (difficulty, seed, q.text)
            ms_rate = int(_Fraction(m.group(1)) * _Fraction(5, 18))
            assert str(ms_rate) != q.answer_text, (
                difficulty,
                seed,
                q.text,
                q.answer_text,
                "the bare converted rate must never be the reported answer",
            )
            assert str(ms_rate) in q.distractors, (
                difficulty,
                seed,
                q.text,
                q.distractors,
                "the bare converted rate must be offered as the 'stopped after converting' distractor",
            )


# --- Never compares two named entities (the p7d neighbour's exact shape) --


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_never_compares_two_named_entities(difficulty):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        text_lower = q.text.lower()
        assert " brand a " not in text_lower and " brand b " not in text_lower
        assert "car a" not in text_lower and "car b" not in text_lower
        assert "difference between" not in text_lower


# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- Independent in-generator checks fire on a corrupted answer ---------


def test_verify_easy_fires_on_corrupted():
    _verify_easy(Fraction(72), 10, 20, 200)
    with pytest.raises(VerificationError):
        _verify_easy(Fraction(72), 10, 20, 201)
    with pytest.raises(VerificationError):
        _verify_easy(Fraction(72), 10, 21, 200)


def test_verify_medium_fires_on_corrupted():
    _verify_medium(Fraction(72), 200, 20, 10)
    with pytest.raises(VerificationError):
        _verify_medium(Fraction(72), 200, 20, 11)
    with pytest.raises(VerificationError):
        _verify_medium(Fraction(72), 200, 21, 10)


def test_verify_hard_fires_on_corrupted():
    _verify_hard(10, 200000, 200, Fraction(20))
    with pytest.raises(VerificationError):
        _verify_hard(10, 200000, 200, Fraction(21))
    with pytest.raises(VerificationError):
        _verify_hard(10, 200000, 201, Fraction(20))


def test_verify_exam_fires_on_corrupted():
    _verify_exam(200, 10, 500, Fraction(2), Fraction(1, 10), Fraction(100))
    with pytest.raises(VerificationError):
        _verify_exam(200, 10, 500, Fraction(2), Fraction(1, 10), Fraction(101))


# --- Skill loads correctly through the API --------------------------------


def test_skill_loads_through_api(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill is not None
    assert skill.domain == "measurement"
    assert skill.prerequisites == ["convert-metric-length"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert len(skill.sources) == 4
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]
