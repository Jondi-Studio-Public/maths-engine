"""Exam chain of `calculate-overtime`: overtime hours, tiered overtime pay, gross pay, average hourly pay."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_overtime_week"
SKILL = S.load_skill("calculate-overtime")
_LOADING = {
    "time and a quarter": Fraction(5, 4),
    "time and a half": Fraction(3, 2),
    "double time": Fraction(2),
    "double time and a half": Fraction(5, 2),
}


def _read(q):
    m = re.search(
        r"paid \$(\d+\.\d\d) per hour for the first (\d+) hours of a week\. Hours beyond \d+ are overtime: the first "
        r"(\d+) overtime hours are paid at (.+?) and any further overtime hours at (.+?)\. Last week, \w+ worked "
        r"(\d+) hours\.",
        q.stem,
    )
    rate, ordinary, first_hours, l1, l2, worked = m.groups()
    return Fraction(rate), int(ordinary), int(first_hours), _LOADING[l1], _LOADING[l2], int(worked)


def _two(x: Fraction) -> str:
    scaled = x * 100
    cents = int(scaled + Fraction(1, 2)) if scaled >= 0 else None
    return f"{cents // 100}.{cents % 100:02d}"


def oracle(q):
    rate, ordinary, first_hours, m1, m2, worked = _read(q)
    pay = Fraction(0)
    for hour in range(1, worked + 1):  # hour-by-hour, independent of the closed form
        pay += rate * (1 if hour <= ordinary else m1 if hour <= ordinary + first_hours else m2)
    overtime = worked - ordinary
    over_pay = pay - rate * ordinary
    return {"a": str(overtime), "b": _two(over_pay), "c": _two(pay), "d": _two(pay / worked)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_hours(questions):
    def chain(q):
        rate, ordinary, first_hours, m1, m2, worked = _read(q)
        wrong = int(q.parts[0].distractors[0])  # e.g. all hours worked, taken as the overtime hours
        over = rate * (m1 * min(wrong, first_hours) + m2 * max(wrong - first_hours, 0))
        gross = rate * ordinary + over
        return {"a": str(wrong), "b": _two(over), "c": _two(gross), "d": _two(Fraction(_two(gross)) / worked)}, {
            "b",
            "c",
            "d",
        }

    S.check_follow_through(SKILL, questions, chain)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99", "b": "1.11", "c": "2.22", "d": "3.33"})


def test_deterministic():
    S.check_deterministic(TOPIC)
