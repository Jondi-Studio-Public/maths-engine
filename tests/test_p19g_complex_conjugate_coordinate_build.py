"""The independent oracle below parses ONLY `question.text` (never anything
from `mathgen.generators.p19g_complex_conjugate_coordinate_build`) via
regex/string-split, and negates the SECOND parsed coordinate itself -- a
route that never reuses the generator's own `Fraction`-based negation call.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]
TOPIC = "p19g_complex_conjugate_coordinate"
SKILL_ID = "senior-understand-and-use-location-of-complex-conjugates-in"

_POINT_RE = re.compile(r"\(\s*(-?\d+(?:/\d+)?)\s*,\s*(-?\d+(?:/\d+)?)\s*\)")
_RIGHT_LEFT_RE = re.compile(r"(\d+) units (?:to the )?(right|left)")
_ABOVE_BELOW_RE = re.compile(r"(\d+) units (above|below) the real axis")
_A_PLUS_BI_RE = re.compile(r"\d+i\b")


# ===========================================================================
# Independent oracle: parses ONLY the rendered text
# ===========================================================================


def _oracle_given_point(text: str, tier: str) -> tuple[Fraction, Fraction]:
    """Re-derive the KNOWN point (gx, gy) the question actually states --
    the plotted z for easy/medium/hard, the plotted conjugate w for exam --
    whose reflection (gx, -gy) is the graded answer. Parses ONLY `text`."""
    if tier == "hard":
        rmatch = _RIGHT_LEFT_RE.search(text)
        amatch = _ABOVE_BELOW_RE.search(text)
        assert rmatch and amatch, text
        absx, xdir = rmatch.groups()
        absy, ydir = amatch.groups()
        gx = Fraction(absx) if xdir == "right" else -Fraction(absx)
        gy = Fraction(absy) if ydir == "above" else -Fraction(absy)
        return gx, gy
    match = _POINT_RE.search(text)
    assert match, text
    return Fraction(match.group(1)), Fraction(match.group(2))


def _oracle_answer(text: str, tier: str) -> tuple[Fraction, Fraction]:
    gx, gy = _oracle_given_point(text, tier)
    return gx, -gy  # negated independently, here -- never the generator's own call


def _parse_pair_text(pair_text: str) -> tuple[Fraction, Fraction]:
    match = _POINT_RE.fullmatch(pair_text)
    assert match, pair_text
    return Fraction(match.group(1)), Fraction(match.group(2))


# ===========================================================================
# Determinism, shape, distractor hygiene, every tier
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_non_empty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.distractors, f"seed {seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)), q.distractors
        assert len(q.distractors) == 3, f"seed {seed}: {len(q.distractors)} distractors"


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
    assert len(texts) >= 20, f"[{difficulty}]: only {len(texts)} distinct questions"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike_bracket_pair(difficulty):
    """No option identifiable by shape: the answer and every distractor are
    all an unlabelled "(comp, comp)" pair, each component a plain integer or
    a simple fraction -- never a decimal, never a labelled "x = ..." form."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for opt in [q.answer_text, *q.distractors]:
            assert _POINT_RE.fullmatch(opt), opt
            assert "." not in opt, opt
            assert "x" not in opt and "y" not in opt, opt


@pytest.mark.parametrize("difficulty", TIERS)
def test_never_uses_a_plus_bi_notation(difficulty):
    """Never write a complex number as "a + bi" anywhere -- that is the
    shipping neighbour `complex-conjugate`'s territory, never this skill's."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for field in [q.text, q.answer_text, *q.distractors]:
            assert not _A_PLUS_BI_RE.search(field), field


# ===========================================================================
# The contract's own load-bearing exclusions: y != 0, and (construction
# discipline) x != 0, on every seed, every tier, directly
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_y_is_never_zero(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        _gx, gy = _oracle_given_point(q.text, difficulty)
        assert gy != 0, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_x_is_never_zero(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        gx, _gy = _oracle_given_point(q.text, difficulty)
        assert gx != 0, (difficulty, seed, q.text)


# ===========================================================================
# The three misconceptions are pairwise distinct from the answer and from
# each other, every seed, every tier -- verified with check_pairwise_distinct
# directly, on values re-derived from the RENDERED TEXT (not the generator's
# own internal gx/gy)
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_misconceptions_pairwise_distinct_from_rendered_text(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        gx, gy = _oracle_given_point(q.text, difficulty)
        answer = (gx, -gy)
        negate_both = (-gx, -gy)
        negate_neither = (gx, gy)
        negate_x_only = (-gx, gy)
        # Must not raise -- all four are genuinely different values.
        check_pairwise_distinct(
            answer,
            negate_both,
            negate_neither,
            negate_x_only,
            msg=f"[{difficulty} seed {seed}] a misconception collides",
        )


def test_check_pairwise_distinct_fires_on_a_corrupted_case():
    """The check itself is a real check, not decorative -- proved on a
    deliberately colliding case, not by mutating the generator module."""
    with pytest.raises(VerificationError):
        check_pairwise_distinct((3, -5), (3, -5))


# ===========================================================================
# Independent oracle: re-derive the answer from the RENDERED TEXT, over
# 300 seeds/tier, via a regex/string-split route -- the generator's own
# `Fraction`-based negation is never called here
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_matches_generator_over_300_seeds_per_tier(difficulty):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        oracle_x, oracle_y = _oracle_answer(q.text, difficulty)
        gen_x, gen_y = _parse_pair_text(q.answer_text)
        assert oracle_x == gen_x and oracle_y == gen_y, (difficulty, seed, q.text, q.answer_text)
        check_equivalent((oracle_x, oracle_y), (gen_x, gen_y), "oracle/generator answer mismatch")
        # Also cross-check the JSON-serialised `answer` dict, not just the text.
        assert Fraction(str(q.answer["x"])) == oracle_x
        assert Fraction(str(q.answer["y"])) == oracle_y


# ===========================================================================
# Real-grader accept/reject, and the pair-tolerance-exactness rule
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        result = GRADERS["pair"](q, q.answer_text)
        assert result.correct, (difficulty, seed)
        for distractor in q.distractors:
            result = GRADERS["pair"](q, distractor)
            assert not result.correct, (difficulty, seed, distractor)


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_value_0_05_off_on_either_component(difficulty):
    """Every part of a pair answer is an integer or an exact fraction,
    and a value 0.05 out is rejected by construction -- proved directly
    against the real grader, not merely argued."""
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        gx, gy = _parse_pair_text(q.answer_text)
        off_x = f"({float(gx) + 0.05:g}, {gy})"
        off_y = f"({gx}, {float(gy) + 0.05:g})"
        assert not GRADERS["pair"](q, off_x).correct, (difficulty, seed, off_x)
        assert not GRADERS["pair"](q, off_y).correct, (difficulty, seed, off_y)


# ===========================================================================
# Answer-space contract: positive/negative y split and answer concentration,
# both at 200 seeds/tier
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_positive_and_negative_y_both_represented_over_200_seeds(difficulty):
    positive = negative = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        _gx, gy = _oracle_given_point(q.text, difficulty)
        if gy > 0:
            positive += 1
        else:
            negative += 1
    assert positive > 0 and negative > 0, (difficulty, positive, negative)
    # Neither sign should be a rare fluke -- both comfortably represented.
    assert positive >= 40 and negative >= 40, (difficulty, positive, negative)


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_under_15_percent_over_200_seeds(difficulty):
    answers = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        answers[q.answer_text] += 1
    top = max(answers.values()) / 200
    assert top < 0.15, (difficulty, answers.most_common(3))


# ===========================================================================
# Difficulty-progression-specific shape checks
# ===========================================================================


@pytest.mark.parametrize("difficulty", ["easy"])
def test_easy_x_is_always_positive(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        gx, _gy = _oracle_given_point(q.text, difficulty)
        assert gx > 0, (seed, q.text)
        assert "/" not in q.text  # easy stays plain integers, never a fraction
        assert "." not in q.answer_text


@pytest.mark.parametrize("difficulty", ["medium"])
def test_medium_x_is_always_negative(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        gx, _gy = _oracle_given_point(q.text, difficulty)
        assert gx < 0, (seed, q.text)


@pytest.mark.parametrize("difficulty", ["medium"])
def test_medium_includes_fractional_coordinates_over_60_seeds(difficulty):
    saw_fraction = False
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        if "/" in q.text:
            saw_fraction = True
            break
    assert saw_fraction


@pytest.mark.parametrize("difficulty", ["hard"])
def test_hard_tier_states_the_point_indirectly_in_words(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not _POINT_RE.search(q.text), (seed, q.text)
        assert "units" in q.text and "real axis" in q.text, q.text


@pytest.mark.parametrize("difficulty", ["exam"])
def test_exam_tier_gives_the_conjugates_point_and_asks_for_the_original(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "conjugate" in q.text and " z" in q.text and " w" in q.text, q.text
        assert "representing z" in q.text, q.text


def test_measure_tiers_shapes_are_genuinely_distinct_hard_vs_lower():
    """Hard's worded-description shape never collides with easy/medium's
    bare-coordinate shape, over the same seed range -- a coarse proxy for
    the tier-escalation instrument run separately by `measure_tiers.py`."""
    hard_texts = {make(TOPIC, seed=s, difficulty="hard").text for s in SEEDS}
    easy_texts = {make(TOPIC, seed=s, difficulty="easy").text for s in SEEDS}
    assert not (hard_texts & easy_texts)
    for t in hard_texts:
        assert not _POINT_RE.search(t)
    for t in easy_texts:
        assert _POINT_RE.search(t)


# ===========================================================================
# Skill contract loading tests
# ===========================================================================


def test_skill_loads_with_correct_answer_kind_and_prerequisites():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]
    assert skill.answer == "pair"
    assert skill.prerequisites == []


def test_skill_is_active_placement_ineligible_and_general_band():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert getattr(skill, "learner_band", "general") == "general"


def test_skill_has_sources_covering_all_four_tiers():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    difficulties = {src.difficulty for src in by_id[SKILL_ID].sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in by_id[SKILL_ID].sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


def test_check_equivalent_fires_on_a_corrupted_claim():
    """`check_equivalent` inside the generator module is a real check, not
    decorative -- proved on a deliberately wrong claim."""
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "deliberately wrong")
