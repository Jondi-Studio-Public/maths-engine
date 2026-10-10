"""Independent oracle for `p9e_fraction_percent_quantity_strategy` and `p9e_terminating_recurring_decimal`.

Independent oracle for two F-10 number skills.

Reads only `question.text`, `question.answer_text` and `question.distractors` -- never anything from
`mathgen.generators.p9e_fraction_percent_recurring_build` (no helper, no constant, no table). Each answer is
re-derived from scratch by a route the generator does not use internally.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

FP_TOPIC = "p9e_fraction_percent_quantity_strategy"
TR_TOPIC = "p9e_terminating_recurring_decimal"

# =============================================================================
# Oracle 1: p9e_fraction_percent_quantity_strategy
# =============================================================================

_FP_RE = re.compile(r"^What is (.+) of (\d+)\?$")


def _fp_value_as_fraction(value_str: str) -> Fraction:
    """Parse whichever of the three registers the question shows -- a
    fraction ("3/8"), a decimal ("0.375") or a percentage ("37.5%") -- into
    an exact `Fraction`, independently of any pre-built table."""
    if value_str.endswith("%"):
        return Fraction(value_str[:-1]) / 100
    if "/" in value_str:
        num_str, den_str = value_str.split("/")
        return Fraction(int(num_str), int(den_str))
    return Fraction(value_str)


def _fp_oracle_answer(text: str) -> int:
    match = _FP_RE.match(text)
    if not match:
        raise ValueError(f"unparseable p9e_fraction_percent_quantity_strategy question: {text!r}")
    value_str, quantity_str = match.groups()
    value = _fp_value_as_fraction(value_str)
    quantity = int(quantity_str)
    result = value * quantity
    if result.denominator != 1:
        raise ValueError(f"non-integer result {result} for {text!r}")
    return result.numerator


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fp_oracle_recomputes_every_seed(difficulty):
    challenged = []
    for seed in range(60):
        q = make(FP_TOPIC, seed=seed, difficulty=difficulty)
        oracle_answer = _fp_oracle_answer(q.text)
        if str(oracle_answer) != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, oracle_answer))
    assert not challenged, challenged


def test_fp_oracle_rejects_a_corrupted_answer():
    q = make(FP_TOPIC, seed=0, difficulty="easy")
    oracle_answer = _fp_oracle_answer(q.text)
    assert str(oracle_answer) == q.answer_text
    assert str(oracle_answer + 1) != q.answer_text


# =============================================================================
# Oracle 2: p9e_terminating_recurring_decimal
# =============================================================================

_TR_RE = re.compile(r"^Write (0\.[0-9]+(?:\.\.\.)?) as a fraction in lowest terms\.$")


def _recover_lead_and_block(digits: str) -> tuple[str, str]:
    """Given the digits shown before "..." (construction always shows
    ``lead + block*3``, lead length 0-2, block length 1-2), recover
    (lead, block) by trying the SHORTEST candidate block length first --
    the same "simplest explanation" a human reader would apply. This is a
    property of the question's own stated display convention (three full
    repeats before the ellipsis), not of the generator's internals."""
    n = len(digits)
    for q in (1, 2):
        p = n - 3 * q
        if p < 0 or p > 2:
            continue
        block = digits[p : p + q]
        if digits[p:] == block * 3:
            return digits[:p], block
    raise ValueError(f"cannot recover a repeating pattern from {digits!r}")


def _tr_oracle_value(shown: str) -> Fraction:
    """Re-derive the exact fraction from the shown decimal notation alone,
    via plain geometric-series summation -- never the generator's own
    closed-form subtraction route."""
    body = shown[2:]  # strip the leading "0."
    if body.endswith("..."):
        digits = body[:-3]
        lead, block = _recover_lead_and_block(digits)
        p, q = len(lead), len(block)
        tail = Fraction(int(block), 10**q - 1)
        if p == 0:
            return tail
        return Fraction(int(lead), 10**p) + tail / (10**p)
    n = len(body)
    return Fraction(int(body), 10**n)


def _tr_oracle_answer_text(shown: str) -> str:
    value = _tr_oracle_value(shown)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tr_oracle_recomputes_every_seed(difficulty):
    challenged = []
    for seed in range(60):
        q = make(TR_TOPIC, seed=seed, difficulty=difficulty)
        match = _TR_RE.match(q.text)
        if not match:
            challenged.append((seed, q.text, "unparseable"))
            continue
        shown = match.group(1)
        oracle_text = _tr_oracle_answer_text(shown)
        if oracle_text != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, oracle_text))
    assert not challenged, challenged


def test_tr_oracle_rejects_a_corrupted_answer():
    q = make(TR_TOPIC, seed=0, difficulty="hard")
    match = _TR_RE.match(q.text)
    shown = match.group(1)
    oracle_text = _tr_oracle_answer_text(shown)
    assert oracle_text == q.answer_text
    corrupted = Fraction(oracle_text) + Fraction(1, 1000)
    assert str(corrupted) != q.answer_text


def test_tr_oracle_worked_examples_from_the_task_file():
    """Worked examples, re-derived independently."""
    assert _tr_oracle_value("0.777...") == Fraction(7, 9)
    assert _tr_oracle_value("0.454545...") == Fraction(5, 11)
    assert _tr_oracle_value("0.41666...") == Fraction(5, 12)
    assert _tr_oracle_value("0.1666...") == Fraction(1, 6)
    assert _tr_oracle_value("0.125") == Fraction(1, 8)
    assert _tr_oracle_value("0.0625") == Fraction(1, 16)
    assert _tr_oracle_value("0.9375") == Fraction(15, 16)
