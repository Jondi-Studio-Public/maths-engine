"""See mathgen/generators/p13a_bracketed_number_equations_build.py for the shapes and misconception design."""

from collections import Counter
from pathlib import Path

import pytest

from sympy import Eq

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p13a_bracketed_number_equations_build import _QSYM, _verify_blank

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p13a_bracketed_number"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_not_the_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        assert all(d for d in q.distractors)
        assert len(q.distractors) == len(set(q.distractors))
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert int(d) != int(q.answer_text)  # exact-integer skill: any nonzero gap is enough


@pytest.mark.parametrize("difficulty", TIERS)
def test_never_a_letter_variable_in_text_or_answer(difficulty):
    """The whole boundary against equations-with-brackets: the unknown is
    always ``?``, never a letter, and the answer is a bare number."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "?" in q.text
        assert " x " not in q.answer_text
        assert "x =" not in q.text and "x=" not in q.text
        int(q.answer_text)  # bare integer, never "x = <n>"
        for d in q.distractors:
            int(d)


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        deliberately_wrong = str(int(q.answer_text) + 1000)
        assert not grade_number(q, deliberately_wrong).correct


# --- answer kind --------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_plain_integer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)
        assert "." not in q.answer_text


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions in 60 seeds"


# --- 200-seed answer-concentration instrument --------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_stays_under_25_percent(difficulty):
    counts = Counter(make(TOPIC, seed=seed, difficulty=difficulty).answer for seed in range(200))
    top_value, top_count = counts.most_common(1)[0]
    assert top_count / 200 <= 0.25, (
        f"{difficulty}: answer {top_value} appears in {top_count}/200 draws, over the 25% cap"
    )


# --- shape-specific: never zero-coefficient / degenerate ---------------


def test_hard_never_draws_a_zero_bracket_value():
    """hard's sign-error distractor collapses onto the answer iff q+b == 0;
    the generator must never emit that degenerate draw."""
    import re

    pattern = re.compile(r"(-?\d+) - (\d+) × \(\? \+ (\d+)\)")
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        match = pattern.search(q.text)
        assert match, q.text
        _d, a, b = match.groups()
        answer = int(q.answer_text)
        assert answer + int(b) != 0


def test_no_standalone_letter_x_in_rendered_text_or_working():
    """The whole boundary against equations-with-brackets depends on the
    unknown never being a letter -- but the multiplication sign itself was
    rendered as the literal letter "x" (e.g. "5 x (? + 4) = -60"), which a
    learner could read as 5 times x times (? + 4). Checked at 200 seeds per
    tier, against both ``q.text`` and every line of ``q.working``, with a
    word-boundary regex so a standalone letter "x" is caught but nothing
    else (this skill never renders a variable named ``x``)."""
    import re

    pattern = re.compile(r"\bx\b")
    for difficulty in TIERS:
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert not pattern.search(q.text), f"{difficulty} seed={seed}: standalone letter 'x' in text: {q.text!r}"
            for line in q.working:
                assert not pattern.search(line), (
                    f"{difficulty} seed={seed}: standalone letter 'x' in working line: {line!r}"
                )


# --- in-generator independent check, exercised directly --------------------


def test_verify_blank_fires_on_corrupted_answer():
    equation = Eq(5 * (_QSYM + 4), 40)  # ? = 4
    _verify_blank(equation, 4, "must not raise for the true root")  # must not raise
    with pytest.raises(VerificationError):
        _verify_blank(equation, 5, "corrupted expected value")


# --- skill contract -------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-bracketed-number-equations"]

    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert sorted(skill.prerequisites) == ["order-of-operations", "two-step-equations"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
    assert [source.params for source in skill.sources] == [
        {"shape": "isolate"},
        {"shape": "outer_add"},
        {"shape": "leading_subtract"},
    ]
    assert [source.stages for source in skill.sources] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, "maintenance"],
    ]
    assert len(skill.misconceptions) == 3
