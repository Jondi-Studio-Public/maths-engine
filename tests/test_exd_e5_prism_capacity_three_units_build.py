"""Tests `exd_e5_prism_capacity_three_units`: a prism's capacity in three units, checked against an independent
oracle.

Tests for prism-capacity-three-units.

The checks for `mathgen/generators/exd_e5_prism_capacity_three_units_build.py` are:

1. An INDEPENDENT ORACLE (`_independent_answer_text` below) that parses
   `question.text` alone -- the rendered "L m [by|long,] W cm [by|wide and] H
   mm [high]" dimension triple -- and recomputes the m3/cm3/L capacity from
   scratch with plain `Fraction` arithmetic. It does NOT import anything from
   `mathgen.generators.exd_e5_prism_capacity_three_units_build` -- not a
   helper, not a constant (confirmed by this file's own import list: only
   `re`, `fractions`, `mathgen.make`, `mathgen.core` (`VerificationError`
   only), `mathgen.text_quality`, `mathgen.grading`, `mathgen.skills`, and -- in the
   corruption-firing tests ONLY -- the generator module itself, to
   monkeypatch it).
2. A firing test per part: corrupting one part (m3, cm3, then L) to return a
   wrong value and confirming the GENERATOR's own in-body independent
   recomputation (`_final`, run every time `make()` is called, via the
   mm-pivot route) raises `VerificationError`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_prism_capacity_three_units"
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
    """hazard: unit words and digit runs are invisible to `prompt_shape`), so
    the only thing that can separate one tier's shapes from another's is
    each tier's own added template wording -- confirmed directly here, not
    assumed from the template design alone."""
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
# Parses the rendered `question.text` -- the length (m), width (cm) and height (mm), in whichever of
# this module's own template shapes rendered them -- and recomputes m3/cm3/L from scratch with plain
# `Fraction` arithmetic. Shares no code, constant or helper with
# `mathgen.generators.exd_e5_prism_capacity_three_units_build` -- confirmed by this file's own
# import list at the top of the module.

_DIMS_RE_BY = re.compile(r"(\d+) m by (\d+) cm by (\d+) mm")
_DIMS_RE_LONG = re.compile(r"(\d+) m long, (\d+) cm wide and (\d+) mm high")
_ANSWER_RE = re.compile(r"^m3 = (\d+(?:\.\d+)?), cm3 = (\d+), L = (\d+)$")


def _parse_dims(question_text: str) -> tuple[int, int, int]:
    m = _DIMS_RE_BY.search(question_text) or _DIMS_RE_LONG.search(question_text)
    assert m is not None, f"oracle: dimension triple not recognised: {question_text!r}"
    length_m, width_cm, height_mm = (int(g) for g in m.groups())
    return length_m, width_cm, height_mm


def _fmt_exact(value: Fraction, max_places: int) -> str:
    """Shortest terminating decimal string for `value` within `max_places`
    -- an independent re-implementation of `exact_decimal_string`'s
    contract, not an import of it."""
    for places in range(0, max_places + 1):
        scaled = value * (10**places)
        if scaled.denominator == 1:
            n = scaled.numerator
            if places == 0:
                return str(n)
            digits = str(n).zfill(places + 1)
            return f"{digits[:-places]}.{digits[-places:]}"
    raise AssertionError(f"oracle: {value} does not terminate within {max_places} places")


def _independent_answer_text(question_text: str) -> str:
    """Re-derive `answer_text` from the rendered prompt alone -- the oracle.
    Never reads `question.answer`/`question.working`. Converts every
    dimension to cm (a pivot unit), multiplies, then derives m3/L from that
    single cm3 value -- the most direct, independent route a human would
    take, deliberately not mirroring the generator's own mm-pivot
    cross-check."""
    length_m, width_cm, height_mm = _parse_dims(question_text)
    length_cm = Fraction(length_m) * 100
    width_cm_f = Fraction(width_cm)
    height_cm = Fraction(height_mm) / 10
    cm3 = length_cm * width_cm_f * height_cm
    m3 = cm3 / 1_000_000
    L = cm3 / 1_000
    assert cm3.denominator == 1, f"oracle: cm3 {cm3} is not a whole number"
    assert L.denominator == 1, f"oracle: L {L} is not a whole number"
    m3_text = _fmt_exact(m3, 2)
    return f"m3 = {m3_text}, cm3 = {int(cm3)}, L = {int(L)}"


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
        m3_text, cm3_text, L_text = m.groups()
        assert len(q.working) == 3, q.working
        assert cm3_text in q.working[0], q.working[0]
        assert m3_text in q.working[1], q.working[1]
        assert L_text in q.working[2], q.working[2]


# -- the independent-verification-firing tests --------------------------------
#
# These tests (only) import the generator module directly, to monkeypatch a
# module-level part function into returning a corrupted value -- exercises
# the GENERATOR's own in-body independent recomputation (`_final`, run on
# every real call to `make()`, via the mm-pivot route), not the oracle above.

import mathgen.generators.exd_e5_prism_capacity_three_units_build as _gen  # noqa: E402


def test_corrupted_m3_part_raises_verification_error(monkeypatch):
    original = _gen._part_m3

    def _corrupted(rng, values, *, m3_val, cm3_val):
        value, working = original(rng, values, m3_val=m3_val, cm3_val=cm3_val)
        return value + 1, working  # wrong-by-1 m3 value

    monkeypatch.setattr(_gen, "_part_m3", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_m3_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="hard")  # no exception


def test_corrupted_cm3_part_raises_verification_error(monkeypatch):
    original = _gen._part_cm3

    def _corrupted(rng, values, *, cm3_val, length_m, width_cm, height_mm):
        value, working = original(
            rng,
            values,
            cm3_val=cm3_val,
            length_m=length_m,
            width_cm=width_cm,
            height_mm=height_mm,
        )
        return value + 1000, working  # wrong-by-1000 cm3 value

    monkeypatch.setattr(_gen, "_part_cm3", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


def test_corrupted_cm3_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="easy")  # no exception


def test_corrupted_L_part_raises_verification_error(monkeypatch):
    original = _gen._part_L

    def _corrupted(rng, values, *, L_val, cm3_val):
        value, working = original(rng, values, L_val=L_val, cm3_val=cm3_val)
        return value + 1, working  # wrong-by-1 L value

    monkeypatch.setattr(_gen, "_part_L", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="medium")


def test_corrupted_L_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="medium")  # no exception


# -- the magnitude constraint: volume_m3 >= 1 every seed/tier ----------------
#
# `_PAIR_TOLERANCE = 0.051` (`mathgen/grading/typed.py:421`) is a FIXED absolute
# tolerance, independent of stated precision. This is a named
# load-bearing test, not incidental coverage: an unconditional assertion
# over every seed at every tier, not a spot check.


@pytest.mark.parametrize("tier", TIERS)
def test_volume_m3_is_at_least_1_every_seed(tier):
    checked = 0
    for q in _all_questions(tier):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        m3_text, _cm3_text, _L_text = m.groups()
        m3_value = float(m3_text)
        assert m3_value >= 1, f"{tier} seed {q.seed}: volume_m3 = {m3_value} < 1"
        checked += 1
    assert checked == 60


def test_volume_m3_floor_is_the_measured_3_0_worst_case():
    """The construction's own worst-case draw (length=2, width_cm=150,
    height_mm=1000) gives volume_m3 = 2*15*10/100 = 3.0 -- the module
    docstring's own claimed floor, reproduced here directly rather than
    merely asserted in prose. Confirms the floor is comfortably clear of the
    >= 1 requirement, not a near-miss."""
    length_m, width_cm, height_mm = 2, 150, 1000
    length_cm = Fraction(length_m) * 100
    height_cm = Fraction(height_mm) / 10
    cm3 = length_cm * Fraction(width_cm) * height_cm
    m3 = cm3 / 1_000_000
    assert m3 == Fraction(3, 1)
    assert m3 >= 1


# -- every distractor's gap > 0.5, every seed/tier/part, unconditionally ----


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_divisor_m3_distractor_clears_0_5_by_construction(tier):
    """The /1000-instead-of-/1000000 m3 misconception must clear the true m3 value by MORE than 0.5 on every
    seed it is offered on."""
    checked = 0
    for q in _all_questions(tier, seeds=range(40)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        m3_text, cm3_text, _L_text = m.groups()
        true_m3 = float(m3_text)
        cm3 = int(cm3_text)
        wrong_m3 = cm3 / 1000
        gap = abs(wrong_m3 - true_m3)
        assert gap > 0.5, f"{tier} seed {q.seed}: wrong-divisor m3 gap {gap} <= 0.5"
        assert gap > 0.051
        checked += 1
    assert checked == 40


@pytest.mark.parametrize("tier", TIERS)
def test_unconverted_raw_product_cm3_distractor_clears_0_5_by_construction(tier):
    """The unconverted-mixed-units cm3 misconception (raw metre/cm/mm
    numbers multiplied with no unit reconciliation) must clear the true cm3
    value by MORE than 0.5 on every seed -- proved here as always exactly
    true cm3 / 10 by this skill's fixed width/height (multiple of
    10/multiple of 100) construction."""
    checked = 0
    for q in _all_questions(tier, seeds=range(40)):
        length_m, width_cm, height_mm = _parse_dims(q.text)
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        _m3_text, cm3_text, _L_text = m.groups()
        true_cm3 = int(cm3_text)
        raw_product = length_m * width_cm * height_mm
        assert raw_product == true_cm3 // 10, (
            f"{tier} seed {q.seed}: raw product {raw_product} is not exactly true_cm3/10 ({true_cm3 // 10})"
        )
        gap = abs(raw_product - true_cm3)
        assert gap > 0.5, f"{tier} seed {q.seed}: unconverted-raw-product gap {gap} <= 0.5"
        assert gap > 0.051
        checked += 1
    assert checked == 40


@pytest.mark.parametrize("tier", TIERS)
def test_confuses_L_and_cm3_distractor_clears_0_5_by_construction(tier):
    """cm3 mistakenly reported under the L label must clear the true L value
    by MORE than 0.5 -- always true since L is always cm3 / 1000, so the two
    numeric values coincide only if cm3 == 0, which never happens."""
    checked = 0
    for q in _all_questions(tier, seeds=range(40)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        _m3_text, cm3_text, L_text = m.groups()
        cm3 = int(cm3_text)
        true_L = int(L_text)
        gap = abs(cm3 - true_L)
        assert gap > 0.5, f"{tier} seed {q.seed}: confuses-L-cm3 gap {gap} <= 0.5"
        assert gap > 0.051
        checked += 1
    assert checked == 40


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all_questions(tier, seeds=range(40)):
        assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"


# -- the formatting hazard: no space-grouped thousands, largest-magnitude ---
#
# `volume_cm3` runs into the millions. Confirm directly against the REAL
# `mathgen.grading.grade_pair`, on the largest-magnitude seed found over a wide
# sweep, that the rendered `answer_text` grades correctly -- not by
# inspection of the rendering code alone.


def test_no_space_grouped_thousands_on_largest_magnitude_seed():
    largest_q = None
    largest_cm3 = -1
    for tier in TIERS:
        for q in _all_questions(tier, seeds=range(100)):
            m = _ANSWER_RE.match(q.answer_text)
            assert m is not None
            cm3 = int(m.group(2))
            if cm3 > largest_cm3:
                largest_cm3 = cm3
                largest_q = q
    assert largest_q is not None
    assert largest_cm3 >= 1_000_000, f"largest cm3 found was only {largest_cm3}"
    # Never a literal space between digits of any number in answer_text.
    for token in re.findall(r"\d[\d ]*\d|\d", largest_q.answer_text):
        assert " " not in token, f"space-grouped numeral found: {token!r} in {largest_q.answer_text!r}"
    result = grade_pair(largest_q, largest_q.answer_text)
    assert result.correct, (
        f"real grader rejected the canonical answer on the largest-magnitude seed: {largest_q.answer_text!r}"
    )


# -- real grader accept/reject, INCLUDING the label-order battery -----------


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
        m3_text, cm3_text, L_text = m.groups()
        reordered = f"L = {L_text}, m3 = {m3_text}, cm3 = {cm3_text}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed {q.seed}: reordered labels rejected: {reordered!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_wrong_value_under_a_correct_label(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        m3_text, cm3_text, L_text = m.groups()
        wrong_L = int(L_text) + 1_000_000  # far outside any 0.051 tolerance
        given = f"m3 = {m3_text}, cm3 = {cm3_text}, L = {wrong_L}"
        result = grade_pair(q, given)
        assert not result.correct, f"{tier} seed {q.seed}: wrong L value wrongly accepted: {given!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_correct_values_under_swapped_labels(tier):
    """cm3 and L VALUES swapped between their own labels -- same value set,
    wrong pairing. Always a genuine test here: cm3 and L never coincide
    numerically (L is always cm3 / 1000, and cm3 is never 0)."""
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        m3_text, cm3_text, L_text = m.groups()
        assert cm3_text != L_text
        swapped = f"m3 = {m3_text}, cm3 = {L_text}, L = {cm3_text}"
        result = grade_pair(q, swapped)
        assert not result.correct, f"{tier} seed {q.seed}: swapped cm3/L wrongly accepted: {swapped!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_dropped_label(tier):
    """A three-label canonical against a learner answer with only two labels
    falls through to the positional path (row 12's 'not measured' note) --
    still correctly rejected here since the positional value list lengths
    differ."""
    for q in _all_questions(tier, seeds=range(5)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        m3_text, cm3_text, _L_text = m.groups()
        dropped = f"m3 = {m3_text}, cm3 = {cm3_text}"
        result = grade_pair(q, dropped)
        assert not result.correct, f"{tier} seed {q.seed}: dropped label wrongly accepted: {dropped!r}"


# -- at least 20 distinct rendered questions per tier over 60 seeds ---------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all_questions(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"


# -- skill loads --------------------------------------------------------------


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "prism-capacity-three-units")
        assert skill.domain == "measurement"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {
            "rectangular-prism-volume",
            "convert-metric-length",
            "convert-metric-capacity",
        }
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
