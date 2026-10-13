"""Tests for
`mathgen.core.compose()`'s worked example,
`mathgen.generators.exd_compose_example.exd_compose_scientific_to_engineering`.

`exd_compose_example.py` is NOT a `_build` module, so none of the
`_build`-scoped sweeps in `tests/test_mathgen_*.py`
(`test_no_bad_text_patterns`, `test_prompt_variety`, `test_pinned_hashes`,
...) ever see this topic. This file asserts, by hand, the properties those
sweeps would otherwise give for free -- no bad text patterns, DCM-001 prompt
variety, determinism -- plus EXD-004's own two required proofs:

1. An INDEPENDENT ORACLE (`_independent_engineering_answer` below) that
   parses `question.text` and re-derives the composed answer from scratch,
   using only `fractions.Fraction` and a regex -- it does not import
   anything from `mathgen.generators.exd_compose_example`, not a helper, not
   a constant.
2. A firing test: corrupting one chain part's raw value and confirming the
   GENERATOR's own in-body `check_equivalent` (a *different*, in-generator
   check, run every time `make()` is called -- not the oracle above) raises
   `VerificationError`. This test alone imports the generator module
   directly, to monkeypatch it; the oracle above never does.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import Question, VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_compose_scientific_to_engineering"
_SEEDS = range(1, 41)  # 40 seeds, per the task's "30-40 seeds" instruction


# -- by-hand DCM/MSB checks (no BUILD_TOPICS sweep ever runs these here) ----


def test_no_bad_text_patterns_over_a_seed_sweep():
    for seed in range(1, 31):  # 30 seeds
        q = make(TOPIC, seed=seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects}"


def test_prompt_variety_dcm_001():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_determinism_same_seed_is_byte_identical():
    for seed in (1, 2, 3, 100, 12345):
        first = make(TOPIC, seed=seed).to_json()
        second = make(TOPIC, seed=seed).to_json()
        assert first == second


# -- the independent oracle (EXD-004's real proof) ---------------------------
#
# Parses the rendered `question.text` -- the two given scientific-notation
# values and the operation between them -- and recomputes the engineering-
# notation answer from scratch with plain `Fraction` arithmetic. Deliberately
# duplicates the SHAPE of the generator's own normalisation/shift logic
# (there is no third way to convert to engineering notation) but shares no
# code, constant or helper with `mathgen.generators.exd_compose_example` --
# confirmed by this file's own import list (see the module docstring above).

_PROMPT_RE = re.compile(r"\(([\d.]+) × 10\^(-?\d+)\) ([×÷]) \(([\d.]+) × 10\^(-?\d+)\)")
_ANSWER_RE = re.compile(r"^([\d.]+) × 10\^(-?\d+)$")


def _independent_engineering_answer(question_text: str) -> str:
    """Re-derive `exd_compose_scientific_to_engineering`'s answer_text from
    the rendered prompt alone -- the oracle. Never reads `question.answer`/
    `question.working`; the final answer must be recoverable from the
    prompt text by itself (EXD-004)."""
    m = _PROMPT_RE.search(question_text)
    assert m is not None, f"oracle: prompt shape not recognised: {question_text!r}"
    c1_text, e1_text, op, c2_text, e2_text = m.groups()
    c1, c2 = Fraction(c1_text), Fraction(c2_text)
    e1, e2 = int(e1_text), int(e2_text)
    value1 = c1 * (Fraction(10) ** e1)
    value2 = c2 * (Fraction(10) ** e2)
    raw = value1 * value2 if op == "×" else value1 / value2

    # Normalise to scientific form [1, 10) x 10^sci_exp.
    mantissa = raw
    sci_exp = 0
    while mantissa >= 10:
        mantissa /= 10
        sci_exp += 1
    while mantissa < 1:
        mantissa *= 10
        sci_exp -= 1

    # Engineering form: exponent is the nearest lower multiple of 3.
    shift = sci_exp % 3
    eng_exp = sci_exp - shift
    eng_mantissa = raw / (Fraction(10) ** eng_exp)

    for places in range(0, 12):
        scaled = eng_mantissa * (10**places)
        if scaled.denominator == 1:
            if places == 0:
                mantissa_str = str(scaled.numerator)
            else:
                digits = str(scaled.numerator).zfill(places + 1)
                mantissa_str = f"{digits[:-places]}.{digits[-places:]}"
            break
    else:
        raise AssertionError(f"oracle: mantissa {eng_mantissa} does not terminate within 12 places")
    return f"{mantissa_str} × 10^{eng_exp}"


def test_independent_oracle_matches_answer_text_over_40_seeds():
    mismatches = []
    for seed in _SEEDS:
        q = make(TOPIC, seed=seed)
        oracle_answer = _independent_engineering_answer(q.text)
        if oracle_answer != q.answer_text:
            mismatches.append((seed, q.text, oracle_answer, q.answer_text))
    assert mismatches == [], mismatches


def test_independent_oracle_answer_shape_is_recognised():
    """Every oracle-recomputed answer, and the generator's own
    `answer_text`, matches the same `<mantissa> × 10^<exponent>` shape --
    confirms the oracle is not accidentally comparing two strings that both
    happen to be unparsed nonsense."""
    for seed in _SEEDS:
        q = make(TOPIC, seed=seed)
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


def test_every_part_value_appears_in_working():
    """EXD-004's third clause: every part's own value appears somewhere in
    `working`, so a learner can check their intermediate steps -- not only
    the final answer."""
    for seed in _SEEDS:
        q = make(TOPIC, seed=seed)
        m = _PROMPT_RE.search(q.text)
        assert m is not None
        c1_text, e1_text, op, c2_text, e2_text = m.groups()
        c1, c2 = Fraction(c1_text), Fraction(c2_text)
        e1, e2 = int(e1_text), int(e2_text)
        value1 = c1 * (Fraction(10) ** e1)
        value2 = c2 * (Fraction(10) ** e2)
        raw = value1 * value2 if op == "×" else value1 / value2
        mantissa = raw
        sci_exp = 0
        while mantissa >= 10:
            mantissa /= 10
            sci_exp += 1
        while mantissa < 1:
            mantissa *= 10
            sci_exp -= 1
        # Chain part 1's own value (the normalised scientific-notation
        # product/quotient) is rendered in the FIRST working line.
        assert len(q.working) >= 2, q.working
        assert f"10^{sci_exp}" in q.working[0], q.working[0]
        # Chain part 2's own value (the engineering-notation mantissa/
        # exponent) is rendered in the SECOND working line, and equals the
        # final answer_text.
        assert q.answer_text.split(" × ")[-1] in q.working[1], q.working[1]


# -- the independent-verification-firing test --------------------------------
#
# This test (only) imports the generator module directly, to monkeypatch a
# chain part into returning a corrupted value -- the "checked FIRING on a
# corrupted answer, exercised directly" proof that an independent oracle needs. It is a different check from the
# oracle above: this exercises the GENERATOR's own in-body
# `check_equivalent` (run on every real call to `make()`), not the test
# file's independent oracle.

import mathgen.generators.exd_compose_example as _exd_compose_example  # noqa: E402


def test_corrupted_part_value_raises_verification_error(monkeypatch):
    original_part_engineering = _exd_compose_example._part_engineering

    def _corrupted_part_engineering(rng, values):
        (mantissa, exponent), working = original_part_engineering(rng, values)
        return (mantissa + 1, exponent), working  # a wrong-by-1 mantissa

    monkeypatch.setattr(_exd_compose_example, "_part_engineering", _corrupted_part_engineering)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1)


def test_corrupted_part_value_is_not_a_false_positive_when_uncorrupted():
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous (every seed fails
    regardless of corruption)."""
    make(TOPIC, seed=1)  # no exception


# -- the multi-part grading ruling (measured, not extended) ------------------
#
# Measures whether `mathgen.grading.typed.grade_pair` already handles a
# THREE-label dict answer correctly, before deciding whether it needs extending.

from mathgen.grading import typed as _grading  # noqa: E402


def _labelled_question(answer: dict, answer_text: str) -> Question:
    return Question(
        topic="scratch-three-label",
        level=1,
        seed=0,
        text="scratch",
        latex="scratch",
        answer=answer,
        answer_text=answer_text,
    )


_THREE_LABEL_Q = _labelled_question(
    {"mean": 851, "median": 850, "mode": 849},
    "mean = 851, median = 850, mode = 849",
)


def test_grade_pair_accepts_three_label_dict_in_canonical_order():
    result = _grading.grade_pair(_THREE_LABEL_Q, "mean = 851, median = 850, mode = 849")
    assert result.correct is True


def test_grade_pair_accepts_three_label_dict_in_any_order():
    result = _grading.grade_pair(_THREE_LABEL_Q, "mode = 849, mean = 851, median = 850")
    assert result.correct is True


def test_grade_pair_rejects_wrong_value_under_a_correct_label():
    result = _grading.grade_pair(_THREE_LABEL_Q, "mean = 851, median = 850, mode = 1")
    assert result.correct is False


def test_grade_pair_rejects_correct_value_under_the_wrong_label():
    # "mean" and "median" values swapped between labels -- same value set,
    # wrong pairing.
    result = _grading.grade_pair(_THREE_LABEL_Q, "mean = 850, median = 851, mode = 849")
    assert result.correct is False


def test_grade_pair_handles_a_four_label_dict_too():
    q = _labelled_question(
        {"a": 1, "b": 2, "c": 3, "d": 4},
        "a = 1, b = 2, c = 3, d = 4",
    )
    assert _grading.grade_pair(q, "d = 4, c = 3, b = 2, a = 1").correct is True
    assert _grading.grade_pair(q, "a = 1, b = 2, c = 3, d = 5").correct is False
