"""Tests `exd_e5_mean_median_mode_one_set`: one data set with graded mean, median and mode, checked against an
independent oracle.

Tests for mean-median-mode-one-set.

The checks for `mathgen/generators/exd_e5_mean_median_mode_one_set_build.py` are:

1. An INDEPENDENT ORACLE (`_independent_answer_text` below) that parses
   `question.text` alone -- the rendered comma-separated data-set list --
   and re-derives mean/median/mode from scratch with plain `Fraction`
   arithmetic and `collections.Counter`. It does NOT import anything from
   `mathgen.generators.exd_e5_mean_median_mode_one_set_build` -- not a
   helper, not a constant (confirmed by this file's own import list: only
   `re`, `fractions`, `collections`, `mathgen.make`, `mathgen.core`,
   `mathgen.text_quality`, `mathgen.grading`, `mathgen.skills`, and -- in the
   corruption-firing tests ONLY -- the generator module itself, to
   monkeypatch it).
2. A firing test: corrupting one part (mode, then mean) to return a wrong
   value and confirming the GENERATOR's own in-body independent
   recomputation (`_final`, run every time `make()` is called) raises
   `VerificationError`.

In the SHIPPED code, `_final` recomputes mean/median/mode independently from `data` (never from a part's own
already-returned value) and raises via `check`/`check_equivalent` on any disagreement. The test that DOES
distinguish the two is the corruption-firing family below: it monkeypatches `_part_mode`/`_part_mean`
(module-level functions, not closures, exactly so this monkeypatch is possible -- see the generator module's
own docstring) to return a wrong-by-a-fixed-amount value and asserts `make` raises `VerificationError`. A
`_final` that TRUSTED the part's own value instead of rechecking it against `data` would emit the corrupted
value in `answer_text` with no exception at all, and this test would fail to see one raised.

## `test_mode_computed_as_largest_value_is_rejected_when_it_differs`

Directly answers "which test would fail if the
mode were computed by 'largest value' instead of 'most frequent value' on a
seed where they differ": it patches `_part_mode` to return `max(data)`
whenever that differs from the true mode value, and confirms
`VerificationError` raises via `_final`'s own independent mode
re-verification (`check(len(winners) == 1 and winners[0] == mode_v, ...)`,
`itertools.groupby`-based, never `max()`). Had a *real*, uncorrupted
generator made this mistake, the independent ORACLE below would also have
caught it on the next question it examined, since the oracle's own mode
computation (`collections.Counter`) never takes `max()` either -- but THIS
test is the one that demonstrates the failure directly, on a seed
engineered to expose it, rather than waiting on a coincidence.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_mean_median_mode_one_set"
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
    """The odd/even list-length lever (this skill's PRIMARY tier lever,
    per the generator module's own docstring) plus each tier's own added
    wording must produce shapes that never collide with ANY other tier's
    shapes, confirmed directly (not assumed from the odd/even design
    alone) -- `scripts/measure_tiers.py`'s own per-tier escalation
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
# Parses the rendered `question.text` -- the comma-separated data-set list, wherever it sits in that
# tier's own wording -- and recomputes mean/ median/mode from scratch with plain `Fraction`
# arithmetic and `collections.Counter`. Shares no code, constant or helper with
# `mathgen.generators.exd_e5_mean_median_mode_one_set_build` -- confirmed by this file's own import
# list at the top of the module.

# At least 8 and at most 12 comma-separated integers -- matches this
# skill's own "8-12 values" contract and is robust to every tier's
# differently-worded surrounding sentence (a "{count} values" mention
# elsewhere in the text is never itself part of a >= 8-long comma run).
_VALUES_RE = re.compile(r"\d+(?:,\s*\d+){7,11}")
_ANSWER_RE = re.compile(r"^mean = (-?\d+(?:\.\d+)?), median = (-?\d+(?:\.\d+)?), mode = (-?\d+)$")


def _parse_data_set(question_text: str) -> list[int]:
    m = _VALUES_RE.search(question_text)
    assert m is not None, f"oracle: no 8-12 value comma list found in {question_text!r}"
    return [int(tok) for tok in m.group(0).split(",")]


def _round_half_up(value: Fraction, dp: int) -> Fraction:
    """Independent half-up rounding, NOT imported from
    `mathgen.core.round_half_up` -- the "+infinity" tie-break, correct for
    the always-positive means this oracle ever rounds."""
    scale = 10**dp
    import math

    return Fraction(math.floor(value * scale + Fraction(1, 2)), scale)


def _fmt(value: Fraction, dp: int) -> str:
    scaled = value * (10**dp)
    assert scaled.denominator == 1, f"oracle: {value} is not exact at {dp} dp"
    n = int(scaled)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if dp == 0:
        return f"{sign}{n}"
    digits = str(n).zfill(dp + 1)
    return f"{sign}{digits[:-dp]}.{digits[-dp:]}"


def _independent_answer_text(question_text: str) -> str:
    """Re-derive `answer_text` from the rendered prompt alone -- the
    oracle. Never reads `question.answer`/`question.working`."""
    data = _parse_data_set(question_text)
    n = len(data)
    assert 8 <= n <= 12, f"oracle: parsed {n} values, expected 8-12: {data}"

    mean_frac = _round_half_up(Fraction(sum(data), n), 1)
    mean_text = _fmt(mean_frac, 1)

    ordered = sorted(data)
    if n % 2:
        median_frac = Fraction(ordered[n // 2])
    else:
        median_frac = Fraction(ordered[n // 2 - 1] + ordered[n // 2], 2)
    median_dp = 0 if median_frac.denominator == 1 else 1
    median_text = _fmt(median_frac, median_dp)

    counts = Counter(data)
    max_count = max(counts.values())
    winners = [v for v, c in counts.items() if c == max_count]
    assert len(winners) == 1, f"oracle: mode is not unique: {counts}"
    mode_text = str(winners[0])

    return f"mean = {mean_text}, median = {median_text}, mode = {mode_text}"


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
        mean_text, median_text, mode_text = m.groups()
        assert len(q.working) == 3, q.working
        assert mean_text in q.working[0], q.working[0]
        assert median_text in q.working[1], q.working[1]
        assert mode_text in q.working[2], q.working[2]


# -- the independent-verification-firing tests --------------------------------
#
# These tests (only) import the generator module directly, to monkeypatch a
# module-level part function into returning a corrupted value -- exercises
# the GENERATOR's own in-body independent recomputation (`_final`, run on
# every real call to `make()`), not the oracle above.

import mathgen.generators.exd_e5_mean_median_mode_one_set_build as _gen  # noqa: E402


def test_corrupted_mode_part_raises_verification_error(monkeypatch):
    original = _gen._part_mode

    def _corrupted(rng, values, *, mode_value, data):
        value, working = original(rng, values, mode_value=mode_value, data=data)
        return value + 1, working  # wrong-by-1 mode value

    monkeypatch.setattr(_gen, "_part_mode", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_mode_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="hard")  # no exception


def test_corrupted_mean_part_raises_verification_error(monkeypatch):
    original = _gen._part_mean

    def _corrupted(rng, values, *, rounded_mean, data, count):
        value, working = original(rng, values, rounded_mean=rounded_mean, data=data, count=count)
        return value + 1, working  # wrong-by-1 mean value

    monkeypatch.setattr(_gen, "_part_mean", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


def test_corrupted_mean_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="easy")  # no exception


def test_mode_computed_as_largest_value_is_rejected_when_it_differs(monkeypatch):
    """Patches `_part_mode` to return `max(data)` whenever that is not already the true mode; leaves the seed
    uncorrupted (delegates to the original) on the rare seed where they coincide, so the corruption is only
    ever a genuine one."""
    original = _gen._part_mode
    triggered = False

    def _corrupted(rng, values, *, mode_value, data):
        value, working = original(rng, values, mode_value=mode_value, data=data)
        largest = max(data)
        if largest == mode_value:
            return value, working  # coincidence on this seed -- leave uncorrupted
        nonlocal triggered
        triggered = True
        return largest, working

    monkeypatch.setattr(_gen, "_part_mode", _corrupted)
    found_failure = False
    for seed in range(30):
        try:
            make(TOPIC, seed=seed, difficulty="hard")
        except VerificationError:
            found_failure = True
            break
    assert triggered, "no seed in range(30) had max(data) != mode_value -- cannot demonstrate"
    assert found_failure, "the largest-value mode bug was not rejected by any seed in range(30)"


# -- the 0.051 hazard: numeric gaps, computed and asserted, not assumed -----


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_divisor_mean_distractor_clears_0_051_by_construction(tier):
    """The mean-wrong-divisor misconception (total / (n-1)) must clear the
    true mean by MORE than 0.051 (`mathgen/grading/typed.py::_PAIR_TOLERANCE`,
    FIXED, absolute, independent of stated precision) on every seed it is
    offered on -- this module's own `_MIN_MEAN_GAP = 1` retry-loop
    guarantee, checked here directly against the rendered question rather
    than assumed from the generator's own construction."""
    checked = 0
    for q in _all_questions(tier, seeds=range(40)):
        data = _parse_data_set(q.text)
        n = len(data)
        true_mean = Fraction(sum(data), n)
        wrong_mean = Fraction(sum(data), n - 1)
        gap = abs(float(true_mean) - float(wrong_mean))
        assert gap > 0.051, f"{tier} seed {q.seed}: wrong-divisor gap {gap} <= 0.051"
        checked += 1
    assert checked == 40


@pytest.mark.parametrize("tier", TIERS)
def test_unsorted_median_distractor_clears_0_051_when_offered(tier):
    """The median-does-not-sort misconception, whenever it is genuinely
    different from the true median, must clear 0.051 -- proved here as a
    GENERAL property of this skill's value grid (every candidate is either
    a plain integer or the average of two integers, so two distinct
    candidates differ by >= 0.5), not merely observed on a sample."""
    found_a_genuine_case = False
    for q in _all_questions(tier, seeds=range(60)):
        data_rendered = _parse_data_set(q.text)  # as PRESENTED (unsorted)
        n = len(data_rendered)
        if n % 2:
            unsorted_v = Fraction(data_rendered[n // 2])
        else:
            unsorted_v = Fraction(data_rendered[n // 2 - 1] + data_rendered[n // 2], 2)
        ordered = sorted(data_rendered)
        if n % 2:
            true_median = Fraction(ordered[n // 2])
        else:
            true_median = Fraction(ordered[n // 2 - 1] + ordered[n // 2], 2)
        if unsorted_v == true_median:
            continue  # honestly-dropped coincidental collision
        found_a_genuine_case = True
        gap = abs(float(unsorted_v) - float(true_median))
        assert gap >= 0.5, f"{tier} seed {q.seed}: unsorted-median gap {gap} < 0.5"
        assert gap > 0.051
    assert found_a_genuine_case, f"{tier}: unsorted-median misconception never genuinely differed over 60 seeds"


@pytest.mark.parametrize("tier", TIERS)
def test_mode_largest_value_distractor_clears_0_051_when_offered(tier):
    found_a_genuine_case = False
    for q in _all_questions(tier, seeds=range(60)):
        data = _parse_data_set(q.text)
        counts = Counter(data)
        max_count = max(counts.values())
        true_mode = [v for v, c in counts.items() if c == max_count][0]
        largest = max(data)
        if largest == true_mode:
            continue
        found_a_genuine_case = True
        gap = abs(largest - true_mode)
        assert gap >= 1, f"{tier} seed {q.seed}: mode-largest gap {gap} < 1"
        assert gap > 0.051
    assert found_a_genuine_case, f"{tier}: mode-largest-value misconception never genuinely differed over 60 seeds"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all_questions(tier, seeds=range(40)):
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"


# -- mode uniqueness, over a real seed sweep (construction proof) -----------


@pytest.mark.parametrize("tier", TIERS)
def test_mode_is_unambiguous_over_60_seeds(tier):
    """Exactly one value's frequency strictly exceeds every other's, on
    every question -- construction proof, not an assumption."""
    for q in _all_questions(tier):
        data = _parse_data_set(q.text)
        counts = Counter(data)
        max_count = max(counts.values())
        winners = [v for v, c in counts.items() if c == max_count]
        assert len(winners) == 1, f"{tier} seed {q.seed}: mode tie {counts}"


# -- odd/even median escalation, over a real seed sweep ---------------------


def test_easy_and_medium_draw_odd_counts_only():
    for tier in ("easy", "medium"):
        for q in _all_questions(tier, seeds=range(40)):
            assert len(_parse_data_set(q.text)) % 2 == 1, f"{tier}: even count in {q.text!r}"


def test_hard_and_exam_draw_even_counts_only():
    for tier in ("hard", "exam"):
        for q in _all_questions(tier, seeds=range(40)):
            assert len(_parse_data_set(q.text)) % 2 == 0, f"{tier}: odd count in {q.text!r}"


# -- real grader accept/reject, INCLUDING the label-order battery ------
#
# Four of the battery's six cases are 3-label-scoped and apply directly; the other two cases are a
# 4-label demonstration (labels a/b/c/d) that is inherently inapplicable to a skill whose canonical
# answer always carries exactly 3 labels -- not reproduced here for that reason (already generic,
# not a claim about THIS skill's own labels).


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
        mean_text, median_text, mode_text = m.groups()
        reordered = f"mode = {mode_text}, mean = {mean_text}, median = {median_text}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed {q.seed}: reordered labels rejected: {reordered!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_wrong_value_under_a_correct_label(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        mean_text, median_text, mode_text = m.groups()
        wrong_mode = int(mode_text) + 1000  # far outside any 0.051 tolerance
        given = f"mean = {mean_text}, median = {median_text}, mode = {wrong_mode}"
        result = grade_pair(q, given)
        assert not result.correct, f"{tier} seed {q.seed}: wrong mode value wrongly accepted: {given!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_correct_values_under_swapped_labels(tier):
    """mean and median VALUES swapped between their own labels -- same
    value set, wrong pairing. Only a genuine test when mean != median on
    that seed (guaranteed to find one within the first 10 seeds of every
    tier, checked here rather than assumed)."""
    found_a_genuine_case = False
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        mean_text, median_text, mode_text = m.groups()
        if mean_text == median_text:
            continue
        found_a_genuine_case = True
        swapped = f"mean = {median_text}, median = {mean_text}, mode = {mode_text}"
        result = grade_pair(q, swapped)
        assert not result.correct, f"{tier} seed {q.seed}: swapped mean/median wrongly accepted: {swapped!r}"
    assert found_a_genuine_case, f"{tier}: mean never differed from median over 10 seeds -- cannot demonstrate"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_dropped_label(tier):
    """A three-label canonical against a learner answer with only two
    labels falls through to the positional path (row 12's 'not measured'
    note) -- still correctly rejected here since the positional value list
    lengths differ."""
    for q in _all_questions(tier, seeds=range(5)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        mean_text, median_text, _mode_text = m.groups()
        dropped = f"mean = {mean_text}, median = {median_text}"
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
        skill = next(s for s in skills if s.id == "mean-median-mode-one-set")
        assert skill.domain == "statistics-probability"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"order-numbers-to-10", "whole-number-division-facts"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
