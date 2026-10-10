"""It imports NOTHING from ``mathgen.generators.t4_degrees_radians_decimal_build``."""

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import pytest

from mathgen import make

TOPIC1 = "t4_degrees_to_radians_decimal"
TOPIC2 = "t4_radians_to_degrees_decimal"

SEEDS = range(40)


def _round_half_up(value: float, places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    return Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP)


_TOPIC1_EASY_RE = re.compile(r"Convert (\d+) degrees? to radians")
_TOPIC1_MEDIUM_RE = re.compile(r"Convert (-?[\d.]+) degrees to radians")
_TOPIC1_HARD_RE = re.compile(r"Convert (\d+)°(\d+)′ to radians")

_TOPIC2_EASY_RE = re.compile(r"Convert ([\d.]+) rad to degrees")
_TOPIC2_MEDIUM_RE = re.compile(r"Convert (-?[\d.]+) rad to degrees")
_TOPIC2_HARD_RE = re.compile(r"Convert (\d*)pi(?:/(\d+))? rad to degrees")


def _assert_matches(answer_text: str, expected: Decimal, places: int, where: str) -> None:
    got = Decimal(answer_text)
    assert got == expected, f"{where}: answer_text {answer_text!r} != recomputed {expected}"
    # confirm the stated precision genuinely matches (regex-count digits)
    m = re.match(r"^-?\d+\.(\d+)$", answer_text)
    assert m and len(m.group(1)) == places, f"{where}: {answer_text!r} does not state {places}dp"


@pytest.mark.parametrize("seed", SEEDS)
def test_topic1_easy_forward_whole(seed):
    q = make(TOPIC1, seed=seed, variant="forward_whole")
    m = _TOPIC1_EASY_RE.search(q.text)
    assert m, q.text
    D = int(m.group(1))
    expected = _round_half_up(math.radians(D), 2)
    _assert_matches(q.answer_text, expected, 2, f"topic1/easy seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_topic1_medium_forward_signed(seed):
    q = make(TOPIC1, seed=seed, variant="forward_signed")
    m = _TOPIC1_MEDIUM_RE.search(q.text)
    assert m, q.text
    D = float(m.group(1))
    expected = _round_half_up(math.radians(D), 3)
    _assert_matches(q.answer_text, expected, 3, f"topic1/medium seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_topic1_hard_dms_forward(seed):
    q = make(TOPIC1, seed=seed, variant="dms_forward")
    m = _TOPIC1_HARD_RE.search(q.text)
    assert m, q.text
    D, M = int(m.group(1)), int(m.group(2))
    total_degrees = D + Fraction(M, 60)
    expected = _round_half_up(math.radians(float(total_degrees)), 4)
    _assert_matches(q.answer_text, expected, 4, f"topic1/hard seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_topic2_easy_forward_basic(seed):
    q = make(TOPIC2, seed=seed, variant="forward_basic")
    m = _TOPIC2_EASY_RE.search(q.text)
    assert m, q.text
    R = float(m.group(1))
    expected = _round_half_up(math.degrees(R), 1)
    _assert_matches(q.answer_text, expected, 1, f"topic2/easy seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_topic2_medium_forward_signed(seed):
    q = make(TOPIC2, seed=seed, variant="forward_signed")
    m = _TOPIC2_MEDIUM_RE.search(q.text)
    assert m, q.text
    R = float(m.group(1))
    expected = _round_half_up(math.degrees(R), 2)
    _assert_matches(q.answer_text, expected, 2, f"topic2/medium seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_topic2_hard_pi_fraction_input(seed):
    q = make(TOPIC2, seed=seed, variant="pi_fraction_input")
    m = _TOPIC2_HARD_RE.search(q.text)
    assert m, q.text
    k = int(m.group(1)) if m.group(1) else 1
    denom = int(m.group(2)) if m.group(2) else 1
    expected = _round_half_up(math.degrees(k * math.pi / denom), 2)
    _assert_matches(q.answer_text, expected, 2, f"topic2/hard seed={seed}")
