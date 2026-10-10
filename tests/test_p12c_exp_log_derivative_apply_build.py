"""Exponential and logarithmic derivative applications.

Tests for senior-use-exponential-functions-and-their-derivatives-to-solve
(ACMMM101, topic p12c_exp_deriv_apply) and
senior-use-logarithmic-functions-and-their-derivatives-to-solve
(ACMMM163, topic p12c_log_deriv_apply).
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

EXP_TOPIC = "p12c_exp_deriv_apply"
LOG_TOPIC = "p12c_log_deriv_apply"
EXP_SKILL_ID = "senior-use-exponential-functions-and-their-derivatives-to-solve"
LOG_SKILL_ID = "senior-use-logarithmic-functions-and-their-derivatives-to-solve"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
EXP_MODES = {"easy": "rate_at_zero", "medium": "rate_at_t", "hard": "time_for_rate", "exam": "rate_at_value"}
LOG_MODES = {"easy": "rate_at_x", "medium": "rate_at_resolved_x", "hard": "x_for_rate", "exam": "value_at_rate"}
SEEDS = range(40)
TOL = stated_precision_tolerance(2)  # Fraction(1, 200)

PARAMS = [(EXP_TOPIC, EXP_SKILL_ID, EXP_MODES), (LOG_TOPIC, LOG_SKILL_ID, LOG_MODES)]


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    """Generate through the skill's own `sources` entry, params-driven, not a
    bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- Distractor gap exceeds the stated-precision tolerance, every seed -----


@pytest.mark.parametrize("topic,skill_id,modes", PARAMS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_gap_exceeds_stated_precision_tolerance(skills_dict, topic, skill_id, modes, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                assert gap > TOL, (difficulty, seed, q.answer_text, q.distractors, gap)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic,skill_id,modes", PARAMS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, skill_id, modes, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        wrong = str(round(float(q.answer) + 1.0, 2))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer states exactly 2 decimal places, matching the instruction -----


@pytest.mark.parametrize("topic,skill_id,modes", PARAMS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_state_exactly_two_decimal_places(skills_dict, topic, skill_id, modes, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        for text in [q.answer_text] + list(q.distractors):
            assert "." in text, (difficulty, seed, text)
            assert len(text.split(".")[1]) == 2, (difficulty, seed, text)


# --- Instruction phrase present in both text and latex ---------------------


@pytest.mark.parametrize("topic,skill_id,modes", PARAMS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_instruction_phrase_in_both_text_and_latex(skills_dict, topic, skill_id, modes, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        assert "decimal place" in q.text.lower(), (difficulty, seed, q.text)
        assert "decimal place" in q.latex.lower(), (difficulty, seed, q.latex)


# --- Boundary: never asks the learner to find the model itself -------------

_FORBIDDEN_PHRASES = ["particular solution", "find k", "find n0", "initial condition"]


@pytest.mark.parametrize("topic,skill_id,modes", PARAMS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_never_asks_to_find_the_model(skills_dict, topic, skill_id, modes, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        lower = q.text.lower()
        for phrase in _FORBIDDEN_PHRASES:
            assert phrase not in lower, (difficulty, seed, phrase, q.text)


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


@pytest.mark.parametrize("topic,modes", [(EXP_TOPIC, EXP_MODES), (LOG_TOPIC, LOG_MODES)])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, modes, difficulty):
    mode = modes[difficulty]
    seen = {make(topic, seed=s, difficulty=difficulty, mode=mode).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, mode, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


@pytest.mark.parametrize("topic", [EXP_TOPIC, LOG_TOPIC])
def test_raises_value_error_on_unknown_mode(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, mode="bogus")


# --- The skills' own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


@pytest.mark.parametrize("skill_id", [EXP_SKILL_ID, LOG_SKILL_ID])
def test_skill_sources_and_stages(skills_dict, skill_id):
    skill = skills_dict[skill_id]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []


# --- Hard/exam bounds: plausibility, measured not assumed -------------------


def test_exp_hard_time_bounds_plausible():
    ts = [float(make(EXP_TOPIC, seed=s, difficulty="hard", mode="time_for_rate").answer) for s in range(60)]
    assert all(0 < t < 200 for t in ts), ts
    denom = len(ts)
    ok = sum(1 for t in ts if 0 < t < 200)
    assert ok == denom, f"{ok}/{denom} within bounds"


def test_exp_exam_rate_nonzero():
    for seed in range(60):
        q = make(EXP_TOPIC, seed=seed, difficulty="exam", mode="rate_at_value")
        assert q.answer != 0, (seed, q.text)


def test_log_hard_x_bounds_plausible():
    xs = [float(make(LOG_TOPIC, seed=s, difficulty="hard", mode="x_for_rate").answer) for s in range(60)]
    denom = len(xs)
    ok = sum(1 for x in xs if 0 < x < 1000)
    assert ok == denom, f"{ok}/{denom} within bounds"


def test_log_exam_x_and_value_bounds_plausible():
    for seed in range(60):
        q = make(LOG_TOPIC, seed=seed, difficulty="exam", mode="value_at_rate")
        assert q.answer is not None, (seed, q.text)


# The learner sees the text verbatim (latex = text). Proved red before the prompt
# switched to a middle dot.
def test_no_prompt_uses_the_letter_x_as_a_multiplication_sign():
    import re
    from mathgen.skills import load_skills
    from mathgen import make

    _, skills = load_skills()
    ids = {
        "senior-use-exponential-functions-and-their-derivatives-to-solve",
        "senior-use-logarithmic-functions-and-their-derivatives-to-solve",
    }
    bad = re.compile(r"\d x (?:e\^|ln\()")
    for skill in skills:
        if skill.id not in ids:
            continue
        for src in skill.sources:
            for seed in range(40):
                q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
                assert not bad.search(q.text), q.text
