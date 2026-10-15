"""Tests for the multi-part exam source of senior-interpret-timetables-such-as-bus-train-and-ferry: wait, journey,
time to spare and whole-trip speed from a six-service timetable. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_timetable_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_timetable_chain"
SKILL = "senior-interpret-timetables-such-as-bus-train-and-ferry"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 1, 2, 2]
FT_LABELS = {"c", "d"}


def _minutes(clock: str) -> int:
    h, m = clock.split(":")
    return 60 * int(h) + int(m)


def _one(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _facts(q):
    arrive_stop, deadline = (
        _minutes(c)
        for c in re.search(r"reaches [^.]* at (\d\d:\d\d) and must be at .* by (\d\d:\d\d)\.", q.stem).groups()
    )
    distance = int(re.search(r"The route is (\d+) km", q.stem).group(1))
    services = [
        (_minutes(d), _minutes(a)) for d, a in re.findall(r"the (\d\d:\d\d) \w+ arrives at (\d\d:\d\d)", q.stem)
    ]
    assert len(services) == 6
    ok = [s for s in services if s[0] >= arrive_stop and s[1] <= deadline]
    dep, arr = max(ok)
    wait, journey = dep - arrive_stop, arr - dep
    return arrive_stop, deadline, distance, services, wait, journey, deadline - arr


def oracle(q):
    arrive_stop, deadline, distance, services, wait, journey, spare = _facts(q)
    return {
        "a": str(wait),
        "b": str(journey),
        "c": str(spare),
        "d": _one(Fraction(60 * distance, wait + journey)),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_oracle_agrees_with_every_part():
    CHECKS["oracle_agrees"]()


def test_composed_reply_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_are_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_the_latest_departure_overall_is_a_trap():
    """The latest-leaving service arrives after the deadline on every draw, so ignoring the deadline is wrong."""
    for s in range(120):
        arrive_stop, deadline, distance, services, wait, journey, spare = _facts(exam(TOPIC, s))
        assert max(services)[1] > deadline
        assert wait >= 0 and spare >= 3


def test_wrong_wait_carries_into_spare_time_and_speed():
    def chain(q):
        arrive_stop, deadline, distance, services, wait, journey, spare = _facts(q)
        wrong_a = wait + 6
        replies = {
            "a": str(wrong_a),
            "b": str(journey),
            "c": str(deadline - arrive_stop - wrong_a - journey),
            "d": _one(Fraction(60 * distance, wrong_a + journey)),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_journey_carries_into_spare_time_and_speed():
    def chain(q):
        arrive_stop, deadline, distance, services, wait, journey, spare = _facts(q)
        wrong_b = journey + 9
        replies = {
            "a": str(wait),
            "b": str(wrong_b),
            "c": str(deadline - arrive_stop - wait - wrong_b),
            "d": _one(Fraction(60 * distance, wait + wrong_b)),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)
