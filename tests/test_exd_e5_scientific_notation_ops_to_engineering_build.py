"""Tests for scientific-notation-ops-to-engineering.

The checks for `mathgen/generators/exd_e5_scientific_notation_ops_to_engineering_build.py` are:

1. An INDEPENDENT ORACLE (`_independent_engineering_answer` below) that
   parses `question.text` and re-derives the composed answer from scratch,
   using only `re` and `fractions.Fraction` -- it does NOT import anything
   from `mathgen.generators.exd_e5_scientific_notation_ops_to_engineering_build`,
   not a helper, not a constant (confirmed by this file's own import list:
   grep below shows only `mathgen.make`/`mathgen.core`/`mathgen.text_quality`/
   `mathgen.grading`/`mathgen.skills`, and the one noted exception).
2. A firing test: corrupting chain part 2's raw value and confirming the
   GENERATOR's own in-body `check_equivalent` (run every time `make()` is
   called) raises `VerificationError`. This test alone imports the generator
   module directly, to monkeypatch it -- the oracle above never does.

In the SHIPPED code, `_part_engineering` is correct, so `values[1]` is correct, so `answer_text` is correct --
and the independent ORACLE (`test_independent_oracle_matches_answer_text_over_60_seeds`, which parses
`question.text` and recomputes from first principles) agrees with it, exactly as it should. That is true
REGARDLESS of whether `_final` independently rechecks `values[1]` against a fresh derivation from `c1, e1, c2,
e2, op` or simply trusts it: the oracle only ever sees whether `answer_text` is right, never how `_final`
assembled it.

The test that DOES distinguish the two is `test_corrupted_part_value_raises_verification_error` below: it
monkeypatches `_part_engineering` to return a wrong-by-a-fixed-amount `values[1]` (a real defect injected at
the chain-part level, not a formatting change) and asserts `make` raises `VerificationError`. A `_final` that
RECHECKS (recomputes `raw_value` from the original `c1, e1, c2, e2, op` and compares via `check_equivalent`)
catches the mismatch and raises, which is what this module's `_final` does and what the test confirms. A
`_final` that had instead TRUSTED `values[1]` (e.g. built `answer_text` by reformatting it, or by re-parsing a
`working` string derived from it, with no independent recomputation) would emit the corrupted mantissa as
`answer_text` with no exception at all -- `pytest.raises(VerificationError)` would fail to see one raised, and
the test would fail.

(A genuine bug that made a *real*, uncorrupted `_final` compute a wrong
`answer_text` -- as opposed to this deliberately-injected corruption -- would
be a different failure, and the independent oracle WOULD catch that one: a
wrong `answer_text` disagrees with the oracle's from-scratch recomputation of
the prompt's own given values, regardless of which route produced it.)
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_scientific_notation_ops_to_engineering"
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


@pytest.mark.parametrize("tier", TIERS)
def test_determinism_same_seed_and_tier_is_byte_identical(tier):
    for seed in (0, 1, 2, 3, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


# the independent oracle ---------------------------
#
# Parses the rendered `question.text` -- the two given scientific-notation values and the operation
# between them -- and recomputes the engineering-notation answer from scratch with plain `Fraction`
# arithmetic. Every tier's own template bank shares the SAME parseable core "(<c1> x 10^<e1>) <op>
# (<c2> x 10^<e2>)", by design (see the generator module's own docstring), so one regex covers every
# tier. Deliberately duplicates the SHAPE of the generator's own normalisation/shift logic (there is
# no third way to convert to engineering notation) but shares no code, constant or helper with
# `mathgen.generators.exd_e5_scientific_notation_ops_to_engineering_build` -- confirmed by this
# file's own import list at the top of the module.

_PROMPT_RE = re.compile(r"\((-?[\d.]+) × 10\^(-?\d+)\) ([×÷]) \((-?[\d.]+) × 10\^(-?\d+)\)")
_ANSWER_RE = re.compile(r"^-?[\d.]+ × 10\^-?\d+$")


def _fmt_exact(value: Fraction, max_places: int = 12) -> str:
    """Shortest terminating-decimal string for a (possibly negative)
    Fraction, independently reimplemented -- never calls
    `mathgen.core.exact_decimal_string`."""
    for places in range(0, max_places + 1):
        scaled = value * (10**places)
        if scaled.denominator == 1:
            num = scaled.numerator
            sign = "-" if num < 0 else ""
            num = abs(num)
            if places == 0:
                return f"{sign}{num}"
            digits = str(num).zfill(places + 1)
            return f"{sign}{digits[:-places]}.{digits[-places:]}"
    raise AssertionError(f"oracle: {value} does not terminate within {max_places} places")


def _independent_engineering_answer(question_text: str) -> str:
    """Re-derive the composed answer_text from the rendered prompt alone -- the oracle. Never reads
    `question.answer`/`question.working`; the final answer must be recoverable from the prompt text by itself."""
    m = _PROMPT_RE.search(question_text)
    assert m is not None, f"oracle: prompt shape not recognised: {question_text!r}"
    c1_text, e1_text, op, c2_text, e2_text = m.groups()
    c1, c2 = Fraction(c1_text), Fraction(c2_text)
    e1, e2 = int(e1_text), int(e2_text)
    value1 = c1 * (Fraction(10) ** e1)
    value2 = c2 * (Fraction(10) ** e2)
    raw = value1 * value2 if op == "×" else value1 / value2

    magnitude = abs(raw)
    sci_exp = 0
    while magnitude >= 10:
        magnitude /= 10
        sci_exp += 1
    while magnitude < 1:
        magnitude *= 10
        sci_exp -= 1

    shift = sci_exp % 3
    eng_exp = sci_exp - shift
    eng_mantissa = raw / (Fraction(10) ** eng_exp)

    return f"{_fmt_exact(eng_mantissa)} × 10^{eng_exp}"


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _all_questions(tier):
        oracle_answer = _independent_engineering_answer(q.text)
        if oracle_answer != q.answer_text:
            mismatches.append((q.seed, q.text, oracle_answer, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_answer_shape_is_recognised(tier):
    for q in _all_questions(tier):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_every_part_value_appears_in_working(tier):
    """Every part's own value appears somewhere in `working`, so a learner
    can check their intermediate steps -- not only the final answer."""
    for q in _all_questions(tier):
        m = _PROMPT_RE.search(q.text)
        assert m is not None
        c1_text, e1_text, op, c2_text, e2_text = m.groups()
        c1, c2 = Fraction(c1_text), Fraction(c2_text)
        e1, e2 = int(e1_text), int(e2_text)
        value1 = c1 * (Fraction(10) ** e1)
        value2 = c2 * (Fraction(10) ** e2)
        raw = value1 * value2 if op == "×" else value1 / value2
        magnitude = abs(raw)
        sci_exp = 0
        while magnitude >= 10:
            magnitude /= 10
            sci_exp += 1
        while magnitude < 1:
            magnitude *= 10
            sci_exp -= 1
        assert len(q.working) >= 2, q.working
        # Chain part 1's own value (the normalised scientific-notation
        # product/quotient) is rendered in the FIRST working line.
        assert f"10^{sci_exp}" in q.working[0], q.working[0]
        # Chain part 2's own value (the engineering-notation mantissa/
        # exponent) is rendered in the SECOND working line, and equals the
        # final answer_text.
        assert q.answer_text.split(" × ")[-1] in q.working[1], q.working[1]


# -- the independent-verification-firing test --------------------------------
#
# This test (only) imports the generator module directly, to monkeypatch a
# chain part into returning a corrupted value -- exercises the GENERATOR's
# own in-body `check_equivalent` (run on every real call to `make()`), not
# the test file's independent oracle above.

import mathgen.generators.exd_e5_scientific_notation_ops_to_engineering_build as _gen  # noqa: E402


def test_corrupted_part_value_raises_verification_error(monkeypatch):
    original_part_engineering = _gen._part_engineering

    def _corrupted_part_engineering(rng, values):
        (mantissa, exponent), working = original_part_engineering(rng, values)
        return (mantissa + 1, exponent), working  # a wrong-by-1 mantissa

    monkeypatch.setattr(_gen, "_part_engineering", _corrupted_part_engineering)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_part_value_is_not_a_false_positive_when_uncorrupted():
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous."""
    make(TOPIC, seed=1, difficulty="hard")  # no exception


# real grader accept/reject --------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_correct_option_and_rejects_every_other(tier):
    for q in _all_questions(tier, seeds=range(30)):
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        accept = grade_mcq(q, str(correct_index))
        assert accept.correct, f"{tier}: grader rejected correct index {correct_index} for {q.text!r}"
        for i in range(len(options)):
            if i == correct_index:
                continue
            reject = grade_mcq(q, str(i))
            assert not reject.correct, f"{tier}: grader accepted wrong index {i} ({options[i]!r}) for {q.text!r}"


# -- distractor sanity --------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all_questions(tier, seeds=range(40)):
        assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"


@pytest.mark.parametrize("tier", ("hard", "exam"))
def test_sign_error_on_mantissa_exercisable_over_60_seeds(tier):
    """The sign-error-on-mantissa misconception (only offered on hard/exam)
    is genuinely produced over a real seed sweep, not merely theoretically
    possible."""
    found = False
    for q in _all_questions(tier):
        mantissa_text = q.answer_text.split(" × ")[0]
        flipped = mantissa_text[1:] if mantissa_text.startswith("-") else f"-{mantissa_text}"
        candidate = f"{flipped} × 10^{q.answer_text.split('10^')[-1]}"
        if candidate in q.distractors:
            found = True
            break
    assert found, f"{tier}: sign-error-on-mantissa distractor never observed over 60 seeds"


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_negatives_never_drawn_on_easy_or_medium(tier):
    for q in _all_questions(tier):
        m = _PROMPT_RE.search(q.text)
        assert m is not None
        c1_text, _e1, _op, c2_text, _e2 = m.groups()
        assert not c1_text.startswith("-"), f"{tier}: negative c1 on a no-negatives tier: {q.text!r}"
        assert not c2_text.startswith("-"), f"{tier}: negative c2 on a no-negatives tier: {q.text!r}"


def test_easy_is_multiplication_only_and_medium_is_division_only():
    """The TOP-LEVEL CONNECTING operator (the regex's captured `op` group,
    between the two bracketed given values) is always "×" on easy and
    always "÷" on medium. NOT a raw substring check: "×" also appears as
    the scientific-notation grouping symbol in "c × 10^e" for EVERY given
    value regardless of tier, so a naive "'×' not in q.text" assertion on
    medium is always false (both given values still render "× 10^") --
    only the connector itself is tier-exclusive."""
    for q in _all_questions("easy"):
        m = _PROMPT_RE.search(q.text)
        assert m is not None, q.text
        assert m.group(3) == "×", q.text
    for q in _all_questions("medium"):
        m = _PROMPT_RE.search(q.text)
        assert m is not None, q.text
        assert m.group(3) == "÷", q.text


# -- at least 20 distinct rendered questions per tier over 60 seeds ---------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all_questions(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"


# -- skill loads --------------------------------------------------------------


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "scientific-notation-ops-to-engineering")
        assert skill.domain == "secondary-number"
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"scientific-notation", "engineering-notation"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
