"""Tests for rectangle-area-perimeter-three-units.

The checks for `mathgen/generators/exd_e5_rectangle_area_perimeter_three_units_build.py` are:

1. An INDEPENDENT ORACLE (`_independent_answer_text` below) that parses
   `question.text` alone -- the two mm side lengths, via the literal
   "`{L} mm by {W} mm`" substring every one of the 12 templates embeds --
   and re-derives perimeter_mm/area_cm2/area_m2 from scratch with plain
   `Fraction` arithmetic. It does NOT import anything from
   `mathgen.generators.exd_e5_rectangle_area_perimeter_three_units_build`
   -- not a helper, not a constant (confirmed by this file's own import
   list: only `re`, `math`, `fractions`, `mathgen.make`, `mathgen.core`,
   `mathgen.text_quality`, `mathgen.grading`, `mathgen.skills` and -- in the
   corruption-firing tests ONLY -- the generator module itself, to
   monkeypatch it).
2. A firing test: corrupting one part (perimeter, then area_cm2, then
   area_m2) to return a wrong value and confirming the GENERATOR's own
   in-body independent recomputation (`_final`, run every time `make()` is
   called) raises `VerificationError`.

In the SHIPPED code, `_final` recomputes all three parts independently from `L`/`W`/`area_mm2` (never from a
part's own already-returned value) and raises via `check`/`check_equivalent` on any disagreement. The test
that DOES distinguish the two is the corruption-firing family below: it monkeypatches
`_part_perimeter`/`_part_area_cm2`/`_part_area_m2` (module-level functions, not closures, exactly so this
monkeypatch is possible -- see the generator module's own docstring) to return a wrong-by-1 value and asserts
`make` raises `VerificationError`. A `_final` that TRUSTED the part's own value instead of rechecking it
against `L`/`W`/`area_mm2` would emit the corrupted value in `answer_text` with no exception at all, and this
test would fail to see one raised.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_rectangle_area_perimeter_three_units"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)


def _all_questions(tier, seeds=SEEDS):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# -- by-hand text-quality, variety and distractor checks --------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns_over_a_seed_sweep(tier):
    for q in _all_questions(tier, seeds=range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_per_tier(tier):
    shapes = {prompt_shape(q) for q in _all_questions(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} distinct prompt shapes over 60 seeds: {shapes}"


def test_tiers_are_masking_safe_shapes_never_collide_across_tiers():
    """This skill has no structural (odd/even-style) tier lever -- every
    tier draws sides from the same range and escalates ENTIRELY through
    added template wording (the generator module's own docstring). Checked
    directly here, pairwise, rather than merely assumed from the wording
    design: `scripts/measure_tiers.py`'s own per-tier escalation
    classification is a coarser instrument than this pairwise check."""
    shapes = {tier: {prompt_shape(q) for q in _all_questions(tier)} for tier in TIERS}
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1 :]:
            overlap = shapes[a] & shapes[b]
            assert overlap == set(), f"{a} and {b} share shape(s): {overlap}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism_same_seed_and_tier_is_byte_identical(tier):
    for seed in (0, 1, 2, 3, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


# the independent oracle ---------------------------
#
# Parses the rendered `question.text` -- the two mm side lengths, via the literal "{L} mm by {W} mm"
# substring every template embeds -- and recomputes perimeter_mm/area_cm2/area_m2 from scratch with
# plain `Fraction` arithmetic. Shares no code, constant or helper with
# `mathgen.generators.exd_e5_rectangle_area_perimeter_three_units_build` -- confirmed by this file's
# own import list at the top of the module.

_SIDES_RE = re.compile(r"(\d+)\s*mm\s+by\s+(\d+)\s*mm")
_ANSWER_RE = re.compile(r"^perimeter_mm = (\d+), area_cm2 = (\d+), area_m2 = (\d+)$")


def _parse_sides(question_text: str) -> tuple[int, int]:
    m = _SIDES_RE.search(question_text)
    assert m is not None, f"oracle: no '<L> mm by <W> mm' found in {question_text!r}"
    return int(m.group(1)), int(m.group(2))


def _round_half_up_int(value: Fraction) -> int:
    """Independent half-up rounding to the nearest whole number, NOT
    imported from `mathgen.core.round_half_up` -- the "+infinity" tie-break,
    correct for the always-positive areas this oracle ever rounds."""
    return math.floor(value + Fraction(1, 2))


def _independent_answer_text(question_text: str) -> str:
    """Re-derive `answer_text` from the rendered prompt alone -- the
    oracle. Never reads `question.answer`/`question.working`."""
    L, W = _parse_sides(question_text)
    perimeter_mm = 2 * (L + W)
    area_mm2 = L * W
    area_cm2 = _round_half_up_int(Fraction(area_mm2, 100))
    area_m2 = _round_half_up_int(Fraction(area_mm2, 1_000_000))
    return f"perimeter_mm = {perimeter_mm}, area_cm2 = {area_cm2}, area_m2 = {area_m2}"


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _all_questions(tier):
        oracle_answer = _independent_answer_text(q.text)
        if oracle_answer != q.answer_text:
            mismatches.append((q.seed, q.text, oracle_answer, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_answer_shape_is_recognised(tier):
    for q in _all_questions(tier, seeds=range(30)):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_every_part_value_appears_in_working(tier):
    """Every part's own value appears somewhere in `working`, so a learner
    can check their intermediate steps -- not only the final answer."""
    for q in _all_questions(tier, seeds=range(20)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        perimeter_text, cm2_text, m2_text = m.groups()
        assert len(q.working) == 3, q.working
        assert perimeter_text in q.working[0], q.working[0]
        assert cm2_text in q.working[1], q.working[1]
        assert m2_text in q.working[2], q.working[2]


# -- the independent-verification-firing tests --------------------------------
#
# These tests (only) import the generator module directly, to monkeypatch a
# module-level part function into returning a corrupted value -- exercises
# the GENERATOR's own in-body independent recomputation (`_final`, run on
# every real call to `make()`), not the oracle above.

import mathgen.generators.exd_e5_rectangle_area_perimeter_three_units_build as _gen  # noqa: E402


def test_corrupted_perimeter_part_raises_verification_error(monkeypatch):
    original = _gen._part_perimeter

    def _corrupted(rng, values, *, L, W, perimeter_mm):
        value, working = original(rng, values, L=L, W=W, perimeter_mm=perimeter_mm)
        return value + 1, working  # wrong-by-1 perimeter

    monkeypatch.setattr(_gen, "_part_perimeter", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_perimeter_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="hard")  # no exception


def test_corrupted_area_cm2_part_raises_verification_error(monkeypatch):
    original = _gen._part_area_cm2

    def _corrupted(rng, values, *, L, W, area_mm2, cm2_rounded):
        value, working = original(rng, values, L=L, W=W, area_mm2=area_mm2, cm2_rounded=cm2_rounded)
        return value + 1, working  # wrong-by-1 area_cm2

    monkeypatch.setattr(_gen, "_part_area_cm2", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


def test_corrupted_area_cm2_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="easy")  # no exception


def test_corrupted_area_m2_part_raises_verification_error(monkeypatch):
    original = _gen._part_area_m2

    def _corrupted(rng, values, *, L, W, area_mm2, m2_rounded):
        value, working = original(rng, values, L=L, W=W, area_mm2=area_mm2, m2_rounded=m2_rounded)
        return value + 1, working  # wrong-by-1 area_m2

    monkeypatch.setattr(_gen, "_part_area_m2", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="exam")


def test_corrupted_area_m2_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="exam")  # no exception


# -- the magnitude constraint: area_m2 >= 1 raw, every seed/tier ------------
#
# Proven here as an EXPLICIT, UNCONDITIONAL assertion over every seed/tier tested, not merely
# asserted by the side-length range's own construction.


@pytest.mark.parametrize("tier", TIERS)
def test_raw_area_m2_never_below_one_square_metre(tier):
    for q in _all_questions(tier):
        L, W = _parse_sides(q.text)
        raw_area_m2 = Fraction(L * W, 1_000_000)
        assert raw_area_m2 >= 1, (
            f"{tier} seed {q.seed}: raw area_m2 {raw_area_m2} < 1 m^2 -- inside the "
            f"0.051 pair-tolerance danger zone (L={L}, W={W})"
        )


# -- every distractor's gap explicitly, every seed/tier/part ----------------


@pytest.mark.parametrize("tier", TIERS)
def test_mm2_to_cm2_divide_by_ten_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        L, W = _parse_sides(q.text)
        area_mm2 = L * W
        true_cm2 = _round_half_up_int(Fraction(area_mm2, 100))
        wrong_cm2 = _round_half_up_int(Fraction(area_mm2, 10))
        gap = abs(true_cm2 - wrong_cm2)
        assert gap > 0.5, f"{tier} seed {q.seed}: divide-by-10 cm2 gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_perimeter_area_formula_swap_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        L, W = _parse_sides(q.text)
        true_perimeter = 2 * (L + W)
        wrong_perimeter = L * W  # uses the area formula for perimeter
        gap = abs(true_perimeter - wrong_perimeter)
        assert gap > 0.5, f"{tier} seed {q.seed}: perimeter/area formula-swap gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_mm2_to_m2_linear_factor_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        L, W = _parse_sides(q.text)
        area_mm2 = L * W
        true_m2 = _round_half_up_int(Fraction(area_mm2, 1_000_000))
        wrong_m2 = _round_half_up_int(Fraction(area_mm2, 1000))
        gap = abs(true_m2 - wrong_m2)
        assert gap > 0.5, f"{tier} seed {q.seed}: mm2->m2 linear-factor gap {gap} <= 0.5"
        assert gap > 0.051


# -- the space-grouped-thousands formatting hazard --------------------------
#
# `perimeter_mm` can run into the tens of thousands at this skill's
# metre-scale side-length range. Confirmed here directly against the REAL
# `mathgen.grading.grade_pair` on the largest-magnitude seed found over a wide
# sweep, and that no numeric token in `answer_text` contains an internal
# space (which `mathgen.grading`'s labelled-pair parser would misread as
# several separate values).

_INTERNAL_SPACE_IN_NUMBER_RE = re.compile(r"\d \d")


@pytest.mark.parametrize("tier", TIERS)
def test_no_space_grouped_numbers_in_answer_text(tier):
    for q in _all_questions(tier):
        assert not _INTERNAL_SPACE_IN_NUMBER_RE.search(q.answer_text), (
            f"{tier} seed {q.seed}: space-grouped digits in {q.answer_text!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_largest_magnitude_seed_grades_correctly_against_real_grader(tier):
    questions = _all_questions(tier, seeds=range(200))
    largest = max(questions, key=lambda q: int(_ANSWER_RE.match(q.answer_text).group(1)))
    result = grade_pair(largest, largest.answer_text)
    assert result.correct, f"{tier}: largest-magnitude seed {largest.seed} rejected: {largest.answer_text!r}"
    perimeter_token = _ANSWER_RE.match(largest.answer_text).group(1)
    assert perimeter_token.isdigit(), perimeter_token  # plain digits, no separator at all


# -- real grader accept/reject, INCLUDING row 12's label-order battery ------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_order(tier):
    for q in _all_questions(tier, seeds=range(10)):
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"{tier} seed {q.seed}: canonical order rejected: {q.answer_text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_reordered_labels(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        perimeter_text, cm2_text, m2_text = m.groups()
        reordered = f"area_m2 = {m2_text}, perimeter_mm = {perimeter_text}, area_cm2 = {cm2_text}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed {q.seed}: reordered labels rejected: {reordered!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_wrong_value_under_a_correct_label(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        perimeter_text, cm2_text, m2_text = m.groups()
        wrong_m2 = int(m2_text) + 1000  # far outside any 0.051 tolerance
        given = f"perimeter_mm = {perimeter_text}, area_cm2 = {cm2_text}, area_m2 = {wrong_m2}"
        result = grade_pair(q, given)
        assert not result.correct, f"{tier} seed {q.seed}: wrong area_m2 value wrongly accepted: {given!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_correct_values_under_swapped_labels(tier):
    """area_cm2 and area_m2 VALUES swapped between their own labels -- same
    value set, wrong pairing. area_cm2 is always exactly 100x area_m2 at
    this skill's side-length range, so the two never coincide -- a genuine
    test on every seed, not merely on a lucky one."""
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        perimeter_text, cm2_text, m2_text = m.groups()
        assert cm2_text != m2_text, f"{tier} seed {q.seed}: area_cm2 == area_m2, cannot demonstrate a swap"
        swapped = f"perimeter_mm = {perimeter_text}, area_cm2 = {m2_text}, area_m2 = {cm2_text}"
        result = grade_pair(q, swapped)
        assert not result.correct, f"{tier} seed {q.seed}: swapped area_cm2/area_m2 wrongly accepted: {swapped!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_dropped_label(tier):
    """A three-label canonical against a learner answer with only two
    labels falls through to the positional path (row 12's 'not measured'
    note) -- still correctly rejected here since the positional value list
    lengths differ."""
    for q in _all_questions(tier, seeds=range(5)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        perimeter_text, cm2_text, _m2_text = m.groups()
        dropped = f"perimeter_mm = {perimeter_text}, area_cm2 = {cm2_text}"
        result = grade_pair(q, dropped)
        assert not result.correct, f"{tier} seed {q.seed}: dropped label wrongly accepted: {dropped!r}"


# -- distractor hygiene -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all_questions(tier, seeds=range(40)):
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"


# -- at least 20 distinct rendered questions per tier over 60 seeds ---------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all_questions(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"


# -- skill loads --------------------------------------------------------------


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "rectangle-area-perimeter-three-units")
        assert skill.domain == "measurement"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {
            "rectangle-area",
            "rectangle-perimeter",
            "convert-metric-length",
            "senior-use-metric-units-of-area-their-abbreviations-conversions",
        }
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
