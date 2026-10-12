"""Energy conservation check and wave-parameter relation.

Tests for ``ucv1-energy-conservation-check`` (topic
``p14f_energy_conservation_check``) and ``ucv1-wave-parameter-relation``
(topic ``p14f_wave_parameter_relation``).
"""

from dataclasses import replace

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

SKILLS = {
    "p14f_energy_conservation_check": "ucv1-energy-conservation-check",
    "p14f_wave_parameter_relation": "ucv1-wave-parameter-relation",
}

# ucv1-energy-conservation-check states its answer to 1 d.p. at every tier;
# ucv1-wave-parameter-relation's canonical answers are always plain integers
# (no decimal point), graded EXACTLY.
DP_TOL = {
    "ucv1-energy-conservation-check": stated_precision_tolerance(1),
    "ucv1-wave-parameter-relation": Fraction(0),
}


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    """Generate through the skill's own `sources` entry, params-driven, not a
    bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if src.difficulty == "exam" and skill_id == "ucv1-energy-conservation-check":
        # The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
        src = replace(
            src, topic="p14f_energy_conservation_check", params={"variant": "inverse", "mk_hi": 8, "xv_hi": 14}
        )
    if src.difficulty == "exam" and skill_id == "ucv1-wave-parameter-relation":
        # The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
        src = replace(src, topic="p14f_wave_parameter_relation", params={"variant": "medium_change", "hi": 20})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene -------------------------------------------------


@pytest.mark.parametrize("topic,skill_id", SKILLS.items())
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_gap_exceeds_stated_precision_tolerance(skills_dict, topic, skill_id, difficulty):
    tol = DP_TOL[skill_id]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                assert gap > tol, (skill_id, difficulty, seed, q.answer_text, q.distractors, gap)


# --- The real grader accepts the canonical and rejects every distractor -----


@pytest.mark.parametrize("topic,skill_id", SKILLS.items())
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_answer_rejects_distractors(skills_dict, topic, skill_id, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        assert grade_number(q, q.answer_text).correct is True, (skill_id, difficulty, seed, q.text)
        for d in q.distractors:
            assert grade_number(q, d).correct is False, (skill_id, difficulty, seed, q.text, d)


@pytest.mark.parametrize("topic,skill_id", SKILLS.items())
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_a_deliberately_wrong_answer_is_rejected(skills_dict, topic, skill_id, difficulty):
    tol = DP_TOL[skill_id]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        wrong = Fraction(q.answer_text) + tol + Fraction(1, 10) + 1
        if "." in q.answer_text:
            wrong_text = str(float(wrong))
        else:
            wrong_text = str(int(wrong))
        assert grade_number(q, wrong_text).correct is False, (skill_id, difficulty, seed, q.text, wrong_text)


# --- Answer shape: precision, formatting -----------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_energy_answer_always_states_1dp(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-energy-conservation-check", difficulty, seed)
        assert "." in q.answer_text, (difficulty, seed, q.answer_text)
        dp = len(q.answer_text.split(".")[1])
        assert dp == 1, (difficulty, seed, q.answer_text)
        for d in q.distractors:
            assert "." in d and len(d.split(".")[1]) == 1, (difficulty, seed, d)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_wave_answer_is_always_a_plain_integer(skills_dict, difficulty):
    # hard and exam are the exceptions: the answer is an exact integer
    # VALUE, but it is DISPLAYED to 2 decimal places (matching every
    # distractor's shape) so the correct option is never the only "clean"
    # integer-looking one on offer
    # (test_answer_is_not_the_only_option_of_its_shape, mathgen suite).
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", difficulty, seed)
        if difficulty in ("hard", "exam"):
            assert q.answer_text.endswith(".00"), (difficulty, seed, q.answer_text)
            assert float(q.answer_text) == int(float(q.answer_text))
        else:
            assert "." not in q.answer_text, (difficulty, seed, q.answer_text)
            assert q.answer_text == str(int(q.answer_text))


# --- Skill-specific instruction phrase checks -------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_energy_prompt_states_sign_convention_and_precision(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-energy-conservation-check", difficulty, seed)
        assert "minus" in q.text and "state 1" in q.text and "state 2" in q.text, (difficulty, seed, q.text)
        assert "decimal place" in q.text, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_wave_formula_never_printed_except_exam(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", difficulty, seed)
        assert "fλ" not in q.text and "f * λ" not in q.text and "v = f" not in q.text, (difficulty, seed, q.text)


def test_wave_exam_names_frequency_as_the_invariant(skills_dict):
    """It now states that frequency is unchanged across a medium boundary and gives two speed/wavelength states."""
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", "exam", seed)
        assert "frequency" in q.text.lower() and "unchanged" in q.text.lower(), (seed, q.text)
        assert "Hz" not in q.text, (seed, q.text, "frequency is never a printed number on this tier")


# --- At least 20 distinct rendered questions per tier over 60 seeds --------

# --- Independent in-generator checks fire on a corrupted answer ------------


def test_energy_independent_check_fires_on_corruption(monkeypatch):
    """Corrupt only the sympy.Rational independent-check route (never the
    Fraction-based construction/resample loop) so `check_equivalent` must be
    the thing that raises."""
    import mathgen.generators.p14f_energy_wave_build as gen_mod
    from mathgen.core import VerificationError

    real_rational = gen_mod.sympy.Rational
    monkeypatch.setattr(gen_mod.sympy, "Rational", lambda *a, **k: real_rational(*a, **k) + 1)
    with pytest.raises(VerificationError):
        make("p14f_energy_conservation_check", seed=0, difficulty="easy")


def test_wave_independent_check_fires_on_corruption(monkeypatch):
    import mathgen.generators.p14f_energy_wave_build as gen_mod
    from mathgen.core import VerificationError

    real_solve = gen_mod.sympy.solve
    monkeypatch.setattr(gen_mod.sympy, "solve", lambda *a, **k: [x + 1 for x in real_solve(*a, **k)])
    with pytest.raises(VerificationError):
        make("p14f_wave_parameter_relation", seed=0, difficulty="easy")


# --- Skill loading ------------------------------------------------------


def test_energy_skill_loads_with_expected_shape(skills_dict):
    skill = skills_dict["ucv1-energy-conservation-check"]
    assert skill.prerequisites == ["simple-harmonic-angular-frequency"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    difficulties_covered = {src.difficulty for src in skill.sources}
    assert difficulties_covered == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


def test_wave_skill_loads_with_expected_shape(skills_dict):
    skill = skills_dict["ucv1-wave-parameter-relation"]
    assert skill.prerequisites == ["wave-frequency-from-speed-wavelength"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    difficulties_covered = {src.difficulty for src in skill.sources}
    assert difficulties_covered == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


# Proved RED against the earlier ``verify_frequency`` variant before the fix.


def test_wave_exam_tier_never_prints_two_candidate_values(skills_dict):
    """The exam prompt must never hand the learner two candidate numeric
    answers to choose between -- that is a printed 50% guess, and it was
    exactly the earlier defect (two 'f = N Hz' tokens in the text)."""
    import re

    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", "exam", seed)
        # No pattern of the form "<label> = <number> ... or ... <label> = <number>"
        # naming two candidate values for the SAME unknown.
        assert not re.search(r"=\s*\d+(?:\.\d+)?\s*\w*\s+or\s+\w*\s*=\s*\d+", q.text), (seed, q.text)


def test_wave_exam_tier_requires_two_chained_relations(skills_dict):
    """The exam unknown must not be recoverable from a single v = f*lambda
    substitution the way solve_v (easy) or the shipping neighbour already
    is -- it must require deriving an intermediate quantity (frequency, the
    invariant across two media) before the final relation. Structurally:
    the prompt must name TWO distinct speeds (two media/states), not one."""
    import re

    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", "exam", seed)
        speeds = re.findall(r"v\d?\s*=\s*\d+(?:\.\d+)?\s*m/s", q.text)
        assert len(speeds) >= 2, (seed, q.text, "exam tier must name two speeds/media")


def test_wave_exam_tier_not_neighbour_kernel(skills_dict):
    """The exam tier must not be answerable by the single-substitution
    kernel of ``wave-frequency-from-speed-wavelength`` (f = v / lambda from
    one v, one lambda) -- it must need the SECOND wavelength/medium's data
    too. A learner using only the first v1, lambda1 pair must not reach the
    printed answer."""
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, "ucv1-wave-parameter-relation", "exam", seed)
        # The answer must not equal a same-medium computation from the
        # FIRST v/lambda pair alone (that would be the neighbour's own
        # kernel in disguise). Concretely: the working must reference a
        # second medium/state explicitly.
        assert (
            "medium 2" in q.text.lower()
            or "new" in q.text.lower()
            or "enters" in q.text.lower()
            or "second" in q.text.lower()
        ), (seed, q.text)
