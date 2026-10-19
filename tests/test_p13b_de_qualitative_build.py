"""Tests the two qualitative differential-equation skills built by `p13b_de_qualitative_build`: slope-field
examination and formulation.

Both skills built by ``mathgen/generators/p13b_de_qualitative_build.py``:

- senior-examine-slope-direction-or-gradient-fields-of-a (topic
  ``p13b_slope_field_examine``, answer: mcq, all four tiers)
- senior-formulate-differential-equations-including-the-logistic-equation-that
  (topic ``p13b_formulate_de``, answer: mcq, all four tiers)

See the generator module's docstring for why every tier of Skill A is ``mcq``.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check
from mathgen.generators.p13b_de_qualitative_build import (
    CLASSIFY_FORMS,
    EASY_FORMS,
    MEDIUM_FORMS,
    TARGET_NAMES,
    _slope_classify_question,
    _slope_match_question,
    _slope_value_question,
)

SLOPE_TOPIC = "p13b_slope_field_examine"
FORMULATE_TOPIC = "p13b_formulate_de"
SLOPE_SKILL = "senior-examine-slope-direction-or-gradient-fields-of-a"
FORMULATE_SKILL = "senior-formulate-differential-equations-including-the-logistic-equation-that"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)


# --- Determinism -----------------------------------------------------------


@pytest.mark.parametrize("topic", [SLOPE_TOPIC, FORMULATE_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_determinism(topic, tier):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=tier)
        q2 = make(topic, seed=seed, difficulty=tier)
        assert q1.to_json() == q2.to_json()


# --- Distractor hygiene ------------------------------------------------------


@pytest.mark.parametrize("topic", [SLOPE_TOPIC, FORMULATE_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"{topic} {tier} seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"{topic} {tier} seed={seed}: dup distractors"
        assert len(q.distractors) == 3, f"{topic} {tier} seed={seed}: expected 3 distractors (4 options)"


# --- Real grader: mcq is graded by STRING INDEX into mcq_options ------------


@pytest.mark.parametrize("topic", [SLOPE_TOPIC, FORMULATE_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_index_rejects_distractor_indices(topic, tier):
    grader = GRADERS["mcq"]
    assert grader is grade_mcq
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        result = grader(q, correct_index)
        assert result.correct, f"{topic} {tier} seed={seed}: correct index rejected"
        for dist in q.distractors:
            dist_index = options.index(dist)
            result = grader(q, dist_index)
            assert not result.correct, f"{topic} {tier} seed={seed}: distractor index {dist_index} accepted"
        out_of_range = len(options)
        result = grader(q, out_of_range)
        assert not result.correct, f"{topic} {tier} seed={seed}: out-of-range index accepted"


# --- Answer kind / formatted alike ------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_slope_options_formatted_alike_per_tier(tier):
    """Every tier's 4 options share one shape: integers (easy/medium),
    'Point X' labels (hard), or 'dy/dx = ...' equation strings (exam)."""
    for seed in SEEDS:
        q = make(SLOPE_TOPIC, seed=seed, difficulty=tier)
        options = mcq_options(q)
        if tier in ("easy", "medium"):
            for opt in options:
                assert re.fullmatch(r"-?\d+", opt), f"{tier} seed={seed}: {opt!r} not integer-shaped"
        elif tier == "hard":
            for opt in options:
                assert re.fullmatch(r"Point [A-D]", opt), f"{tier} seed={seed}: {opt!r} not a point label"
        else:
            for opt in options:
                assert opt.startswith("dy/dx = "), f"{tier} seed={seed}: {opt!r} not an ODE option"


@pytest.mark.parametrize("tier", TIERS)
def test_formulate_options_are_all_equation_shaped(tier):
    for seed in SEEDS:
        q = make(FORMULATE_TOPIC, seed=seed, difficulty=tier)
        options = mcq_options(q)
        for opt in options:
            assert re.match(r"^d[A-Za-z]/d[a-z] = ", opt), f"{tier} seed={seed}: {opt!r} not equation-shaped"
        # Never a solved y(t)/y(x) -- the boundary against solve-ode-exponential.
        assert "Find" not in q.text and "find y" not in q.text.lower()


# --- Never drifts into solve-ode-exponential's territory --------------------


@pytest.mark.parametrize("tier", TIERS)
def test_formulate_never_asks_to_solve(tier):
    for seed in SEEDS:
        q = make(FORMULATE_TOPIC, seed=seed, difficulty=tier)
        assert "models this situation?" in q.text
        assert not re.search(r"\bFind\b", q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_slope_never_renders_a_diagram_reference(tier):
    for seed in SEEDS:
        q = make(SLOPE_TOPIC, seed=seed, difficulty=tier)
        for banned in ("shown below", "pictured", "diagram", "graph below", "plotted"):
            assert banned not in q.text.lower(), f"{tier} seed={seed}: text references a visual: {q.text!r}"


# --- Independent-check tripwire ----------------------------------------------


def test_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(1 == 2, "should fail")
    check(1 == 1, "should not fail")


def test_slope_value_question_uses_the_independent_sympy_route():
    """A deliberately corrupted point-evaluation must be caught: patch a
    form's own lambda to disagree with its sympy text and confirm the
    build-time check raises."""
    bad_forms = [("x + y", "x + y", lambda x, y: x + y + 1)]  # lambda now wrong
    import random

    rng = random.Random(0)
    with pytest.raises((VerificationError, AssertionError)):
        for _ in range(50):
            _slope_value_question(rng, bad_forms)


# --- At least 20 distinct rendered questions per tier over 60 seeds --------


@pytest.mark.parametrize("topic", [SLOPE_TOPIC, FORMULATE_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic} {tier}: only {len(texts)} distinct questions over 60 seeds"


# --- 200-seed answer-concentration measurement, over the LABEL distribution -


@pytest.mark.parametrize("topic", [SLOPE_TOPIC, FORMULATE_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_label_concentration(topic, tier):
    """mcq option TEXT is randomised per question (drawn letters, drawn values), so the measure pointed at
    'answers' for an mcq skill is the correct-option LABEL (its position in mcq_options) -- the one thing
    every question of a given (topic, tier) shares a common domain for."""
    counts = Counter()
    for seed in range(200):
        q = make(topic, seed=seed, difficulty=tier)
        options = mcq_options(q)
        counts[options.index(q.answer_text)] += 1
    top_share = counts.most_common(1)[0][1] / 200
    assert top_share <= 0.40, f"{topic} {tier}: label concentration {top_share:.1%} ({counts})"


def test_400_seed_key_label_share_slope_hard():
    """`_slope_classify_question`'s `wrong_idx = rng.randrange(4)` is already a uniform draw over the 4 labels by
    construction; this is a MEASUREMENT, not a fix --."""
    counts = Counter(make(SLOPE_TOPIC, seed=seed, difficulty="hard").answer_text for seed in range(400))
    total = sum(counts.values())
    assert total == 400
    for label, n in counts.items():
        share = n / total
        assert 0.19 <= share <= 0.31, f"{label}: {n}/{total} = {share:.1%} outside 25% +/- 6 (counts={dict(counts)})"


@pytest.mark.parametrize("tier", ("hard", "exam"))
def test_400_seed_answer_value_concentration_slope(tier):
    """The slope skill's hard/exam tiers have a genuinely small answer
    universe (4 points / the forms with a valid unique feature subset) --
    measure the actual VALUE distribution, not just the option slot."""
    counts = Counter(make(SLOPE_TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(400))
    top_share = counts.most_common(1)[0][1] / 400
    assert top_share <= 0.40, f"{tier}: {dict(counts)}"


# --- Build-time invariants fire on a corrupted candidate set ----------------


def test_easy_formulate_invariant_fires_on_corrupted_candidates():
    candidates = {
        "key": lambda Q: 2 * Q,
        "decoy_time": lambda Q: 2 * 5,
        "decoy_const": lambda Q: 2,
        "decoy_square": lambda Q: 2 * Q * Q,
    }
    q1, q2 = 3, 6
    satisfying = [name for name, fn in candidates.items() if fn(q1) != 0 and fn(q2) / fn(q1) == 2]
    assert satisfying == ["key"]
    # Corrupt: make decoy_square also linear -- now the invariant is no longer unique.
    candidates["decoy_square"] = lambda Q: 2 * Q
    satisfying = [name for name, fn in candidates.items() if fn(q1) != 0 and fn(q2) / fn(q1) == 2]
    assert set(satisfying) == {"key", "decoy_square"}
    with pytest.raises(VerificationError):
        check(satisfying == ["key"], "invariant should no longer be unique")


# --- Skill loading ------------------------------------------------------


def test_slope_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SLOPE_SKILL), None)
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert len(skill.sources) == 4
    stages = [set(s.stages) for s in skill.sources]
    assert stages == [{0, 1, 2}, {3, 4}, {5, 6, "maintenance"}, {"exam"}]
    assert skill.sources[3].difficulty == "exam"


def test_formulate_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == FORMULATE_SKILL), None)
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert len(skill.sources) == 4
    assert skill.sources[3].difficulty == "exam"
    assert set(skill.sources[3].stages) == {"exam"}


# --- Sanity on the shared pools/helpers exercised above ---------------------


def test_form_pools_nonempty():
    assert len(EASY_FORMS) >= 4
    assert len(MEDIUM_FORMS) >= 4
    assert len(CLASSIFY_FORMS) == len(EASY_FORMS) + len(MEDIUM_FORMS)
    assert len(TARGET_NAMES) >= 3


def test_slope_classify_and_match_are_deterministic_via_rng():
    import random

    r1, r2 = random.Random(42), random.Random(42)
    d1 = _slope_classify_question(r1)
    d2 = _slope_classify_question(r2)
    assert d1 == d2
    r3, r4 = random.Random(7), random.Random(7)
    m1 = _slope_match_question(r3)
    m2 = _slope_match_question(r4)
    assert m1 == m2
