"""Independent oracle for significant-figures / engineering-notation.

Derives every expected answer purely from ``question.text`` (and, for the
MCQ skill, the shape of ``question.answer_text``) -- imports nothing from
``mathgen.generators.p8a_notation_build``. Generates through each skill's own
``sources`` (topic + params + difficulty), never a bare
``make(topic, seed, difficulty=tier)``.
"""

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import yaml

import mathgen

SIG_FIGS_SKILL = "skills/secondary-number/significant-figures.yaml"
ENG_NOTATION_SKILL = "skills/secondary-number/engineering-notation.yaml"


def _load_sources(path):
    with open(path, encoding="utf-8") as f:
        skill = yaml.safe_load(f)
    return skill["sources"]


# the exam-tier source uses a different topic and question shape
SIG_FIGS_SOURCES = [src for src in _load_sources(SIG_FIGS_SKILL) if src["difficulty"] != "exam"]
ENG_NOTATION_SOURCES = _load_sources(ENG_NOTATION_SKILL)


# --- significant_figures -------------------------------------------------


_SIG_FIGS_RE = re.compile(r"Round ([\d.]+) to (\d+) significant figures?\.")


def _oracle_round_significant(value: Decimal, sf: int) -> Decimal:
    """A from-scratch significant-figure rounding, independent of the
    generator's own two routes: find the place value of the first nonzero
    digit by walking the string once, then round via ``Decimal.quantize``."""
    text = format(value, "f")
    int_part, _, frac_part = text.partition(".")
    digits = int_part + frac_part
    point_at = len(int_part)
    first_nonzero = next(i for i, ch in enumerate(digits) if ch != "0")
    msd_place = point_at - 1 - first_nonzero
    quantize_exp = msd_place - sf + 1
    return value.quantize(Decimal(1).scaleb(quantize_exp), rounding=ROUND_HALF_UP)


@pytest.mark.parametrize("source_index", range(len(SIG_FIGS_SOURCES)))
@pytest.mark.parametrize("seed", range(60))
def test_significant_figures_oracle(source_index, seed):
    spec = SIG_FIGS_SOURCES[source_index]
    question = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))
    match = _SIG_FIGS_RE.match(question.text)
    assert match, f"unrecognised question shape: {question.text!r}"
    value = Decimal(match.group(1))
    sf = int(match.group(2))
    expected = _oracle_round_significant(value, sf)
    actual = Decimal(question.answer_text)
    assert expected == actual, (
        f"source {source_index} seed {seed}: expected {expected}, got {actual}\nQuestion: {question.text}"
    )
    for distractor_text in question.distractors:
        distractor = Decimal(distractor_text)
        assert distractor != expected, (
            f"source {source_index} seed {seed}: distractor {distractor} matches expected "
            f"{expected}\nQuestion: {question.text}"
        )


# --- engineering_notation --------------------------------------------------


_ENG_QUESTION_RE = re.compile(r"Write ([\d.]+) . 10\^(-?\d+) in engineering notation")
_ENG_OPTION_RE = re.compile(r"^([\d.]+) . 10\^(-?\d+)(?: \(([\d.]+) (\w+)\))?$")

# exam: "engineering form of a SUM OR DIFFERENCE of two scientific-notation values with far-apart
# exponents" -- a genuinely different question shape from the three regroup-only variants above (the
# original _ENG_QUESTION_RE never matches it, by design: it is a different operation, not the same
# shape with new numbers), so it needs its own from-scratch oracle route rather than an extension of
# the regroup-only one.
_ENG_SUM_QUESTION_RE = re.compile(r"([\d.]+) . 10\^(-?\d+) ([+-]) ([\d.]+) . 10\^(-?\d+)")


@pytest.mark.parametrize("seed", range(60))
def test_engineering_notation_sum_difference_oracle(seed):
    """Independent oracle for the `exam` source specifically: re-derives the
    combined value from the two scientific-notation operands stated in the
    question text via exact `Fraction` arithmetic (never `Decimal`, and never
    importing anything from `p8a_notation_build.py`), regroups to engineering
    form, and rounds the mantissa to 4 significant figures independently of
    the generator's own `_round_significant`/`_digit_string_round` routes."""
    from fractions import Fraction

    exam_sources = [s for s in ENG_NOTATION_SOURCES if s.get("difficulty") == "exam"]
    assert exam_sources, "expected an exam source on engineering-notation"
    spec = exam_sources[0]
    question = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))
    qmatch = _ENG_SUM_QUESTION_RE.search(question.text)
    assert qmatch, f"unrecognised sum/difference question shape: {question.text!r}"
    a, exp_a, op, b, exp_b = qmatch.groups()
    value = Fraction(Decimal(a)) * Fraction(10) ** int(exp_a)
    other = Fraction(Decimal(b)) * Fraction(10) ** int(exp_b)
    value = value + other if op == "+" else value - other

    # Regroup to engineering form (multiple of 3) from the exact Fraction sum.
    eng_exp = 0
    magnitude = abs(value)
    while magnitude >= Decimal(1000):
        magnitude /= 1000
        eng_exp += 3
    while magnitude < 1:
        magnitude *= 1000
        eng_exp -= 3
    # Refine within the chosen decade by powers of ten (Fraction has no
    # `.adjusted()`; walk by single powers of ten instead of assuming a
    # decade boundary lands exactly on a multiple of 3).
    exact_mantissa = value / (Fraction(10) ** eng_exp)
    while exact_mantissa >= 1000:
        eng_exp += 3
        exact_mantissa = value / (Fraction(10) ** eng_exp)
    while exact_mantissa < 1:
        eng_exp -= 3
        exact_mantissa = value / (Fraction(10) ** eng_exp)

    raw_mantissa = Decimal(exact_mantissa.numerator) / Decimal(exact_mantissa.denominator)
    expected_mantissa = _oracle_round_significant(raw_mantissa, 4)
    expected_prefix = _SI_PREFIXES[eng_exp]

    amatch = _ENG_OPTION_RE.match(question.answer_text)
    assert amatch, f"unrecognised answer option shape: {question.answer_text!r}"
    actual_mantissa = Decimal(amatch.group(1))
    actual_exp = int(amatch.group(2))
    actual_prefix = amatch.group(4)
    assert actual_exp == eng_exp, f"seed {seed}: exponent {actual_exp} != {eng_exp}\nQuestion: {question.text}"
    assert actual_mantissa == expected_mantissa, (
        f"seed {seed}: mantissa {actual_mantissa} != {expected_mantissa}\nQuestion: {question.text}"
    )
    assert actual_prefix == expected_prefix, (
        f"seed {seed}: prefix {actual_prefix!r} != {expected_prefix!r}\nQuestion: {question.text}"
    )
    for distractor_text in question.distractors:
        dmatch = _ENG_OPTION_RE.match(distractor_text)
        assert dmatch, f"unrecognised distractor shape: {distractor_text!r}"
        d_mantissa = Decimal(dmatch.group(1))
        d_exp = int(dmatch.group(2))
        d_prefix = dmatch.group(4)
        same = d_mantissa == expected_mantissa and d_exp == eng_exp and d_prefix == expected_prefix
        assert not same, (
            f"seed {seed}: distractor {distractor_text!r} matches the expected "
            f"(mantissa, exponent, prefix)\nQuestion: {question.text}"
        )


_SI_PREFIXES = {
    -12: "pico",
    -9: "nano",
    -6: "micro",
    -3: "milli",
    3: "kilo",
    6: "mega",
    9: "giga",
    12: "tera",
}


def _oracle_engineering_form(coefficient: Decimal, sci_exp: int):
    """floor(sci_exp / 3) * 3 directly from the given scientific form --
    the independent route."""
    eng_exp = 3 * (sci_exp // 3)
    shift = sci_exp - eng_exp
    mantissa = coefficient * (Decimal(10) ** shift)
    return mantissa, eng_exp


# The `exam` source is a genuinely different question shape (a sum/difference of two
# scientific-notation values, not a bare regroup) -- covered by its own
# `test_engineering_notation_sum_difference_oracle` above, never by this regroup-only regex.
# Excluded here by difficulty, not by a hardcoded index, so this sweep does not silently stop
# covering a future non-exam source added at the same list position.
_REGROUP_ONLY_SOURCE_INDICES = [i for i, s in enumerate(ENG_NOTATION_SOURCES) if s.get("difficulty") != "exam"]


@pytest.mark.parametrize("source_index", _REGROUP_ONLY_SOURCE_INDICES)
@pytest.mark.parametrize("seed", range(60))
def test_engineering_notation_oracle(source_index, seed):
    spec = ENG_NOTATION_SOURCES[source_index]
    question = mathgen.make(spec["topic"], seed=seed, difficulty=spec["difficulty"], **(spec["params"] or {}))
    qmatch = _ENG_QUESTION_RE.match(question.text)
    assert qmatch, f"unrecognised question shape: {question.text!r}"
    coefficient = Decimal(qmatch.group(1))
    sci_exp = int(qmatch.group(2))
    assert sci_exp % 3 != 0, f"question's own exponent is already a multiple of 3: {question.text!r}"

    expected_mantissa, expected_exp = _oracle_engineering_form(coefficient, sci_exp)
    assert Decimal(1) <= expected_mantissa < Decimal(1000)
    assert expected_exp % 3 == 0
    expected_prefix = _SI_PREFIXES[expected_exp]

    amatch = _ENG_OPTION_RE.match(question.answer_text)
    assert amatch, f"unrecognised answer option shape: {question.answer_text!r}"
    actual_mantissa = Decimal(amatch.group(1))
    actual_exp = int(amatch.group(2))
    actual_prefix = amatch.group(4)
    assert actual_mantissa == expected_mantissa, (
        f"source {source_index} seed {seed}: mantissa {actual_mantissa} != {expected_mantissa}\n"
        f"Question: {question.text}"
    )
    assert actual_exp == expected_exp, (
        f"source {source_index} seed {seed}: exponent {actual_exp} != {expected_exp}\nQuestion: {question.text}"
    )
    assert actual_prefix == expected_prefix, (
        f"source {source_index} seed {seed}: prefix {actual_prefix!r} != {expected_prefix!r}\nQuestion: {question.text}"
    )

    # every distractor option must fail at least one of (mantissa, exponent, prefix)
    for distractor_text in question.distractors:
        dmatch = _ENG_OPTION_RE.match(distractor_text)
        assert dmatch, f"unrecognised distractor shape: {distractor_text!r}"
        d_mantissa = Decimal(dmatch.group(1))
        d_exp = int(dmatch.group(2))
        d_prefix = dmatch.group(4)
        same = d_mantissa == expected_mantissa and d_exp == expected_exp and d_prefix == expected_prefix
        assert not same, (
            f"source {source_index} seed {seed}: distractor {distractor_text!r} matches the "
            f"expected (mantissa, exponent, prefix)\nQuestion: {question.text}"
        )
