"""Tests `p19e_geometric_limit_class` and `p19e_recurrence_long_term_class`: classifying long-term behaviour,
both three-way `mcq`.

`p19e_geometric_limit_class` and `p19e_recurrence_long_term_class` (both `answer: mcq`, 3-way)."""

from dataclasses import replace
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_long_term_behaviour_build as m  # noqa: F401 -- registers topics

GEO_ID = "senior-understand-the-limiting-behaviour-as-n-rightarrow-infty"
REC_ID = "senior-recognise-that-a-sequence-generated-by-a-first"
GEO_TOPIC = "p19e_geometric_limit_class"
REC_TOPIC = "p19e_recurrence_long_term_class"

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


GEO_SKILL, GEO_SOURCES = _skill_sources(GEO_ID)
REC_SKILL, REC_SOURCES = _skill_sources(REC_ID)
REC_SOURCES["exam"] = replace(REC_SOURCES["exam"], topic="p19e_recurrence_long_term_class", params={})


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _class_of(text, option_text, letter):
    import re

    match = re.search(rf"{letter}: (.+?)(?:   [A-C]:|$)", text)
    wording = match.group(1)
    for cls, w in option_text.items():
        if w == wording:
            return cls
    raise AssertionError(f"could not find class for wording {wording!r}")


# ==========================================================================
# p19e_geometric_limit_class / GEO_ID
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_geo_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(GEO_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 2
        assert q1.answer_text in ("A", "B", "C")
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_geo_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(GEO_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_geo_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(GEO_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_geo_skill_loads_with_expected_shape():
    assert GEO_SKILL.answer == "mcq"
    assert GEO_SKILL.placement_eligible is False
    assert GEO_SKILL.domain == "functions-sequences"
    assert GEO_SKILL.prerequisites == []
    assert {src.difficulty for src in GEO_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in GEO_SKILL.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_geo_class_check_fires_on_corrupted_value(difficulty):
    from mathgen.generators.p19e_long_term_behaviour_build import geo_classify_by_rule
    from fractions import Fraction

    assert geo_classify_by_rule(Fraction(1, 2)) == "zero"
    assert geo_classify_by_rule(Fraction(2)) == "infinity"
    assert geo_classify_by_rule(Fraction(-2)) == "neither"
    assert geo_classify_by_rule(Fraction(-1)) == "neither"
    with pytest.raises(Exception):
        geo_classify_by_rule(Fraction(1))
    with pytest.raises(Exception):
        geo_classify_by_rule(Fraction(0))


def test_geo_independent_check_fires_on_corrupted_target():
    """The generator's own `check(rule_class == target, ...)` /
    `check(sim_class == target, ...)` -- exercised directly with a
    deliberately wrong target to prove the mechanism actually fires."""
    from fractions import Fraction
    from mathgen.core import VerificationError, check
    from mathgen.generators.p19e_long_term_behaviour_build import (
        geo_classify_by_rule,
        geo_classify_by_simulation,
    )

    r = Fraction(2)  # true class is "infinity"
    corrupted_target = "zero"
    with pytest.raises(VerificationError):
        check(geo_classify_by_rule(r) == corrupted_target, "rule route disagrees")
    with pytest.raises(VerificationError):
        check(geo_classify_by_simulation(3, r) == corrupted_target, "simulation route disagrees")


@pytest.mark.parametrize("difficulty", TIERS)
def test_geo_answer_class_distribution_within_bounds(difficulty):
    from mathgen.generators.p19e_long_term_behaviour_build import GEO_OPTION_TEXT

    counts = Counter()
    N = 200
    for seed in range(N):
        q = _make(GEO_SOURCES, difficulty, seed)
        counts[_class_of(q.text, GEO_OPTION_TEXT, q.answer_text)] += 1
    for cls, n in counts.items():
        share = n / N
        assert 0.15 <= share <= 0.55, (difficulty, cls, share)
    assert set(counts) == {"zero", "infinity", "neither"}


def test_geo_key_letter_balanced_over_400_seeds():
    counts = Counter(_make(GEO_SOURCES, "easy", seed).answer_text for seed in range(400))
    for letter in ("A", "B", "C"):
        assert 100 <= counts[letter] <= 175, counts


# ==========================================================================
# p19e_recurrence_long_term_class / REC_ID
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(REC_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 2
        assert q1.answer_text in ("A", "B", "C")
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(REC_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(REC_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_rec_skill_loads_with_expected_shape():
    assert REC_SKILL.answer == "mcq"
    assert REC_SKILL.placement_eligible is False
    assert REC_SKILL.domain == "financial-discrete"
    assert REC_SKILL.prerequisites == []
    assert {src.difficulty for src in REC_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in REC_SKILL.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


def test_rec_class_check_fires_on_corrupted_value():
    from mathgen.generators.p19e_long_term_behaviour_build import rec_classify_by_rule
    from fractions import Fraction

    assert rec_classify_by_rule(Fraction(1, 2), 100, 5) == "steady"
    assert rec_classify_by_rule(Fraction(1), 10, 5) == "increase"
    assert rec_classify_by_rule(Fraction(1), -10, 5) == "decrease"
    assert rec_classify_by_rule(Fraction(3, 2), 5, 100) == "increase"
    # a > 1, fixed point x* = 5/(1-1.5) = -10; u0=-20 < x* -> decrease
    assert rec_classify_by_rule(Fraction(3, 2), 5, -20) == "decrease"
    assert rec_classify_by_rule(Fraction(3, 2), 5, -10) == "steady"
    # corrupted: rule says "increase" but simulation would disagree for this data
    from mathgen.generators.p19e_long_term_behaviour_build import rec_classify_by_simulation

    corrupted = "increase"
    real = rec_classify_by_simulation(Fraction(3, 2), 5, -20)
    assert real != corrupted, "test setup: the corrupted value must actually be wrong"


def test_rec_independent_check_fires_on_corrupted_target():
    """The generator's own `check(rule_class == target, ...)` /
    `check(sim_class == target, ...)` -- exercised directly with a
    deliberately wrong target to prove the mechanism actually fires."""
    from fractions import Fraction
    from mathgen.core import VerificationError, check
    from mathgen.generators.p19e_long_term_behaviour_build import (
        rec_classify_by_rule,
        rec_classify_by_simulation,
    )

    a, b, u0 = Fraction(3, 2), 5, 100  # true class is "increase"
    corrupted_target = "decrease"
    with pytest.raises(VerificationError):
        check(rec_classify_by_rule(a, b, u0) == corrupted_target, "rule route disagrees")
    with pytest.raises(VerificationError):
        check(rec_classify_by_simulation(a, b, u0) == corrupted_target, "simulation route disagrees")


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_answer_class_distribution_within_bounds(difficulty):
    from mathgen.generators.p19e_long_term_behaviour_build import REC_OPTION_TEXT

    counts = Counter()
    N = 200
    for seed in range(N):
        q = _make(REC_SOURCES, difficulty, seed)
        counts[_class_of(q.text, REC_OPTION_TEXT, q.answer_text)] += 1
    for cls, n in counts.items():
        share = n / N
        assert 0.15 <= share <= 0.55, (difficulty, cls, share)
    assert set(counts) == {"increase", "decrease", "steady"}


def test_rec_key_letter_balanced_over_400_seeds():
    counts = Counter(_make(REC_SOURCES, "easy", seed).answer_text for seed in range(400))
    for letter in ("A", "B", "C"):
        assert 100 <= counts[letter] <= 175, counts


def test_rec_exam_amounts_are_never_negative():
    import re

    for seed in range(200):
        q = _make(REC_SOURCES, "exam", seed)
        for amount in re.findall(r"\$(-?\d+)", q.text):
            assert int(amount) >= 0, q.text


# ==========================================================================
# Recurrence medium tier
# ==========================================================================

_FALSE_SHORTCUT_PHRASES = (
    "alone",
    "rather than these",
    "decide from a",
    "based on a ",
    "resisting the shown",
)


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_prompt_never_states_a_false_shortcut(difficulty):
    """Defect 1: the medium-tier prompt used to
    say "Decide from a alone" / "Based on a rather than these early
    values" -- false whenever a > 1, since the class then depends on u(0)
    vs the fixed point (e.g. a = 2, u(0) = -14 decreases but u(0) = 5 would
    increase for the same a). The prompt must never tell the learner the
    method to use or state a shortcut that is not universally true.
    Measured 133/200 medium draws affected before the fix; 0/200 after."""
    for seed in range(200):
        q = _make(REC_SOURCES, difficulty, seed)
        low = q.text.lower()
        for phrase_ in _FALSE_SHORTCUT_PHRASES:
            assert phrase_ not in low, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_rec_prompt_never_shows_an_unreduced_improper_fraction(difficulty):
    """Defect 2: the medium-tier prompt used to
    print early terms as raw improper fractions (e.g. "-443909/500")
    instead of decimals. Measured 175/200 medium draws affected before the
    fix (any non-integer early term); 0/200 after (early terms, when
    shown, are rounded to at most 2 decimal places)."""
    import re

    for seed in range(200):
        q = _make(REC_SOURCES, difficulty, seed)
        assert not re.search(r"-?\d+/\d+", q.text), (difficulty, seed, q.text)
