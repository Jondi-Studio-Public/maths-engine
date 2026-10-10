"""It reads only ``question.text`` and ``question.answer_text``, recomputes the DMS pair from scratch with its
own rounding, and checks the two agree.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make

# Registers the topic without exposing any of its internals to this file.
import mathgen.generators.t4_dms_build  # noqa: F401

TOPIC = "t4_radians_to_dms"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)

_DECIMAL_R_RE = re.compile(r"(-?\d+\.\d+) rad")
_FRACTION_R_RE = re.compile(r"(\d+)/(\d+) rad")
_ANSWER_RE = re.compile(r"^(\d+)°(\d{2})′$")


def _radian_value(text: str, tier: str) -> float:
    if tier == "hard":
        m = _FRACTION_R_RE.search(text)
        assert m, f"no k/m fraction found in {text!r}"
        return int(m.group(1)) / int(m.group(2))
    m = _DECIMAL_R_RE.search(text)
    assert m, f"no decimal radian value found in {text!r}"
    return float(m.group(1))


def _oracle_dms(R: float) -> tuple[int, int]:
    """Own, from-scratch computation: floor/round, no calls into the
    generator module at all."""
    degrees_decimal = R * 180 / math.pi
    degrees = math.floor(degrees_decimal)
    minutes = round((degrees_decimal - degrees) * 60)
    if minutes == 60:
        degrees += 1
        minutes = 0
    return degrees, minutes


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_recomputes_the_dms_pair_from_the_question_alone(tier):
    disagreements = []
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        R = _radian_value(q.text, tier)
        m = _ANSWER_RE.match(q.answer_text)
        assert m, f"{tier} seed={seed}: unparseable answer_text {q.answer_text!r}"
        rendered_degrees, rendered_minutes = int(m.group(1)), int(m.group(2))
        oracle_degrees, oracle_minutes = _oracle_dms(R)
        if (oracle_degrees, oracle_minutes) != (rendered_degrees, rendered_minutes):
            disagreements.append(
                (tier, seed, R, (rendered_degrees, rendered_minutes), (oracle_degrees, oracle_minutes))
            )
    assert not disagreements, (
        "oracle disagreed with the rendered answer on "
        f"{len(disagreements)} of {len(SEEDS)} seeds for tier {tier!r}: {disagreements}"
    )
