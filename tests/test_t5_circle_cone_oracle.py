"""Independent oracle for T5 circle-reverse-from-area + cone-volume.

Re-derives every answer by parsing ``question.text``/``answer_text`` alone,
using ``decimal.Decimal`` at high precision -- a genuinely different
algorithm/library from the generator's exact-``Fraction`` Newton's-method
square root. Never imports anything from
``mathgen.generators.t5_circle_cone_build``.
"""

import re
from decimal import Decimal, ROUND_HALF_UP, getcontext

import pytest

import mathgen

getcontext().prec = 50

CIRCLE_TOPIC = "circle_reverse_from_area"
CONE_TOPIC = "cone_volume"
SEEDS = range(50)

PI_DEC = Decimal("3.14159")

_AREA_RE = re.compile(r"area of ([\d.]+) cm²")
_CONE_EASY_RE = re.compile(r"radius of (\d+) cm and a height of (\d+) cm")
_CONE_MEDIUM_RE = re.compile(r"diameter of (\d+) cm and a height of (\d+) cm")
_CONE_HARD_RE = re.compile(r"radius of (\d+) cm and a slant height of (\d+) cm")


def _round1(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# circle_reverse_from_area
# ---------------------------------------------------------------------------


def oracle_circle_easy(text: str) -> str:
    area = Decimal(_AREA_RE.search(text).group(1))
    r = (area / PI_DEC).sqrt()
    return _round1(r)


def oracle_circle_medium(text: str) -> str:
    area = Decimal(_AREA_RE.search(text).group(1))
    r_exact = (area / PI_DEC).sqrt()
    # Route A: round the radius to 1dp first, then double.
    r_rounded = Decimal(_round1(r_exact))
    route_a = _round1(r_rounded * 2)
    # Route B: double the exact radius, round once.
    route_b = _round1(r_exact * 2)
    assert route_a == route_b, f"oracle routes disagree: A={route_a} B={route_b}"
    return route_b


def oracle_circle_hard(text: str) -> str:
    area = Decimal(_AREA_RE.search(text).group(1))
    r_exact = (area / PI_DEC).sqrt()
    # Route A: round the radius to 1dp first, then C = 2*pi*r, rounded again.
    r_rounded = Decimal(_round1(r_exact))
    route_a = _round1(2 * PI_DEC * r_rounded)
    # Route B: 2*sqrt(pi*area) directly, rounded once.
    route_b = _round1(2 * (PI_DEC * area).sqrt())
    assert route_a == route_b, f"oracle routes disagree: A={route_a} B={route_b}"
    return route_b


@pytest.mark.parametrize("seed", SEEDS)
def test_circle_easy_oracle(seed):
    q = mathgen.make(CIRCLE_TOPIC, seed=seed, difficulty="easy")
    assert oracle_circle_easy(q.text) == q.answer_text


@pytest.mark.parametrize("seed", SEEDS)
def test_circle_medium_oracle(seed):
    q = mathgen.make(CIRCLE_TOPIC, seed=seed, difficulty="medium")
    assert oracle_circle_medium(q.text) == q.answer_text


@pytest.mark.parametrize("seed", SEEDS)
def test_circle_hard_oracle(seed):
    q = mathgen.make(CIRCLE_TOPIC, seed=seed, difficulty="hard")
    assert oracle_circle_hard(q.text) == q.answer_text


# ---------------------------------------------------------------------------
# cone_volume
# ---------------------------------------------------------------------------


def _oracle_cone_volume(r, h) -> str:
    v = Decimal(1) / Decimal(3) * PI_DEC * Decimal(r) ** 2 * Decimal(h)
    return _round1(v)


def oracle_cone_easy(text: str) -> str:
    r, h = (int(x) for x in _CONE_EASY_RE.search(text).groups())
    return _oracle_cone_volume(r, h)


def oracle_cone_medium(text: str) -> str:
    d, h = (int(x) for x in _CONE_MEDIUM_RE.search(text).groups())
    r = Decimal(d) / 2
    return _oracle_cone_volume(r, h)


def oracle_cone_hard(text: str) -> str:
    r, l = (int(x) for x in _CONE_HARD_RE.search(text).groups())
    h = (Decimal(l) ** 2 - Decimal(r) ** 2).sqrt()
    return _oracle_cone_volume(r, h)


@pytest.mark.parametrize("seed", SEEDS)
def test_cone_easy_oracle(seed):
    q = mathgen.make(CONE_TOPIC, seed=seed, difficulty="easy")
    assert oracle_cone_easy(q.text) == q.answer_text


@pytest.mark.parametrize("seed", SEEDS)
def test_cone_medium_oracle(seed):
    q = mathgen.make(CONE_TOPIC, seed=seed, difficulty="medium")
    assert oracle_cone_medium(q.text) == q.answer_text


@pytest.mark.parametrize("seed", SEEDS)
def test_cone_hard_oracle(seed):
    q = mathgen.make(CONE_TOPIC, seed=seed, difficulty="hard")
    assert oracle_cone_hard(q.text) == q.answer_text
