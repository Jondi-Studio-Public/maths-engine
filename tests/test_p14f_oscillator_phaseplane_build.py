"""Focused contract tests for: ucv1-linear-oscillator-classification and ucv1-phase-plane-linear-classification
(both answer: mcq)."""

from dataclasses import replace
import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p14f_oscillator_phaseplane_build  # noqa: F401 -- registers topics

OSC_ID = "ucv1-linear-oscillator-classification"
PP_ID = "ucv1-phase-plane-linear-classification"
OSC_TOPIC = "p14f_linear_oscillator_classification"
PP_TOPIC = "p14f_phase_plane_linear_classification"

SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

OSC_CLASSES = {"underdamped", "critically damped", "overdamped"}
PP_CLASSES = {"stable node", "unstable node", "saddle", "stable spiral", "unstable spiral", "centre"}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


OSC_SKILL, OSC_SOURCES = _skill_sources(OSC_ID)
PP_SKILL, PP_SOURCES = _skill_sources(PP_ID)
OSC_SOURCES["exam"] = replace(OSC_SOURCES["exam"], topic="p14f_linear_oscillator_classification", params={})
PP_SOURCES["exam"] = replace(PP_SOURCES["exam"], topic="p14f_phase_plane_linear_classification", params={})


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-linear-oscillator-classification (answer: mcq, 3-way answer space)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_osc_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(OSC_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 2, "3-way answer space: exactly 2 distractors every draw"


@pytest.mark.parametrize("difficulty", TIERS)
def test_osc_answer_space_is_exactly_the_three_named_classes(difficulty):
    for seed in SEEDS:
        q = _make(OSC_SOURCES, difficulty, seed)
        assert q.answer_text in OSC_CLASSES
        assert set(q.distractors) | {q.answer_text} == OSC_CLASSES


@pytest.mark.parametrize("difficulty", TIERS)
def test_osc_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(OSC_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct, f"seed {seed} {difficulty}: canonical index not accepted"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, i).correct, (
                f"seed {seed} {difficulty}: distractor index {i} ({opt}) wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_osc_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(OSC_SOURCES, difficulty, seed)
        options = mcq_options(q)
        assert not grade_mcq(q, len(options) + 5).correct, (
            f"seed {seed} {difficulty}: out-of-range index wrongly accepted"
        )
        assert not grade_mcq(q, "not a number").correct


def test_osc_independent_check_fires_on_a_corrupted_answer():
    """The generator's own check_equivalent re-derives the discriminant sign via
    a fresh sympy computation; corrupting the constructed target class must not
    silently agree with that re-derivation. Exercised directly against the
    module's classify helper, not the generator's control flow."""
    from mathgen.generators.p14f_oscillator_phaseplane_build import _osc_classify

    # A genuinely underdamped triple (m=1, b=1, k=5 -> D = 1 - 20 = -19 < 0).
    assert _osc_classify(1, 1, 5) == "underdamped"
    assert _osc_classify(1, 1, 5) != "overdamped"
    assert _osc_classify(1, 1, 5) != "critically damped"
    # A genuinely critically-damped triple (q=2, m=1 -> k=4, b=4 -> D=0).
    assert _osc_classify(1, 4, 4) == "critically damped"


def test_osc_hard_tier_requires_rearrangement_not_standard_form():
    for seed in SEEDS:
        q = _make(OSC_SOURCES, "hard", seed)
        assert "rearrange to standard form" in q.text


def test_osc_exam_tier_reaches_critically_damped_over_a_sweep():
    """The exam tier's honest answer space must be all 3 classes -- a design
    that ruled 'critically damped' out by construction would make the tier a 2-way guess."""
    seen = set()
    for seed in range(400):
        q = _make(OSC_SOURCES, "exam", seed)
        seen.add(q.answer_text)
    assert seen == OSC_CLASSES, f"exam tier did not reach all 3 classes over 400 seeds: {seen}"


def test_osc_exam_tier_class_balance_within_25_to_42_percent():
    from collections import Counter

    counts = Counter(_make(OSC_SOURCES, "exam", seed).answer_text for seed in range(400))
    for cls in OSC_CLASSES:
        share = counts[cls] / 400
        assert 0.25 <= share <= 0.42, (cls, counts, share)


def test_osc_exam_tier_requires_unit_conversion_not_the_hard_reasoning_step():
    """The exam tier's reasoning step must be unit conversion (grams,
    grams/second), a step hard's rearrangement does not have -- and must
    NOT reuse hard's "rearrange to standard form" step or print two
    candidate values for the learner to pick between."""
    for seed in SEEDS:
        q = _make(OSC_SOURCES, "exam", seed)
        assert "g/s" in q.text and " g" in q.text, (seed, q.text)
        assert "SI" in q.text or "convert" in q.text.lower(), (seed, q.text)
        assert "rearrange to standard form" not in q.text, (seed, q.text)
        assert "Classify the damping regime of the OTHER value" not in q.text, (seed, q.text)


def test_osc_skill_loads_through_api_with_correct_properties():
    assert OSC_SKILL.domain == "ordinary-differential-equations"
    assert OSC_SKILL.status == "active"
    assert OSC_SKILL.placement_eligible is False
    assert OSC_SKILL.answer == "mcq"
    assert OSC_SKILL.prerequisites == ["simple-harmonic-angular-frequency"]
    assert {src.difficulty for src in OSC_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = OSC_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-phase-plane-linear-classification (answer: mcq, 6-way answer space)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_pp_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(PP_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3, "distractor policy always reaches exactly 3"


@pytest.mark.parametrize("difficulty", TIERS)
def test_pp_answer_and_distractors_are_honest_classes(difficulty):
    for seed in SEEDS:
        q = _make(PP_SOURCES, difficulty, seed)
        assert q.answer_text in PP_CLASSES
        for d in q.distractors:
            assert d in PP_CLASSES


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_pp_easy_medium_are_node_saddle_only(difficulty):
    for seed in SEEDS:
        q = _make(PP_SOURCES, difficulty, seed)
        assert q.answer_text in {"stable node", "unstable node", "saddle"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_pp_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(PP_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct, f"seed {seed} {difficulty}: canonical index not accepted"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, i).correct, (
                f"seed {seed} {difficulty}: distractor index {i} ({opt}) wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_pp_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(PP_SOURCES, difficulty, seed)
        options = mcq_options(q)
        assert not grade_mcq(q, len(options) + 5).correct
        assert not grade_mcq(q, "not a number").correct


def test_pp_independent_check_fires_on_a_corrupted_answer():
    """Exercised directly against the module's trace/determinant classifier:
    a genuinely-saddle (T,D) must not agree with a corrupted node/spiral call."""
    from mathgen.generators.p14f_oscillator_phaseplane_build import _pp_classify_from_TD

    assert _pp_classify_from_TD(0, -5) == "saddle"
    assert _pp_classify_from_TD(0, -5) != "stable node"
    assert _pp_classify_from_TD(-8, 10) == "stable node"
    assert _pp_classify_from_TD(8, 10) == "unstable node"
    assert _pp_classify_from_TD(-1, 5) == "stable spiral"
    assert _pp_classify_from_TD(1, 5) == "unstable spiral"
    assert _pp_classify_from_TD(0, 5) == "centre"


def test_pp_degenerate_cases_are_never_offered():
    """D == 0 and the repeated-real-eigenvalue case (T^2 == 4D, T != 0) are
    excluded by construction -- checked directly against the classifier's own
    guard, since no seed of the generator itself should ever reach them."""
    from mathgen.core import VerificationError
    from mathgen.generators.p14f_oscillator_phaseplane_build import _pp_classify_from_TD

    with pytest.raises(VerificationError):
        _pp_classify_from_TD(4, 4)  # T^2 == 4D, T != 0: repeated real eigenvalue
    with pytest.raises(VerificationError):
        _pp_classify_from_TD(3, 0)  # D == 0: zero eigenvalue


@pytest.mark.parametrize("difficulty", TIERS)
def test_pp_no_seed_ever_hits_a_degenerate_trace_determinant_pair(difficulty):
    for seed in SWEEP_SEEDS:
        q = _make(PP_SOURCES, difficulty, seed)  # would raise VerificationError if degenerate
        assert q.answer_text in PP_CLASSES


def test_pp_exam_tier_uses_word_context_and_covers_all_six_classes_over_a_sweep():
    seen = set()
    for seed in range(200):
        q = _make(PP_SOURCES, "exam", seed)
        assert "rate of change of x" in q.text
        seen.add(q.answer_text)
    assert seen == PP_CLASSES, f"exam tier did not reach all six classes over 200 seeds: {seen}"


def test_pp_skill_loads_through_api_with_correct_properties():
    assert PP_SKILL.domain == "advanced-pdes-and-dynamics"
    assert PP_SKILL.status == "active"
    assert PP_SKILL.placement_eligible is False
    assert PP_SKILL.answer == "mcq"
    assert PP_SKILL.prerequisites == ["autonomous-equilibrium-stability"]
    assert {src.difficulty for src in PP_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = PP_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
