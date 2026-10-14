"""Tests the multi-part exam source of `senior-convert-between-grams-and-milligrams`.

The oracle reads mass, rate, doses, tablet strength (grams) and box size from the rendered stem.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_dose_tablets"
SKILL_ID = "senior-convert-between-grams-and-milligrams"
SKILL = skill_for(SKILL_ID)
_STEM = re.compile(
    r"mass of (\d+) kg .* at (\d+) mg per kilogram .* over (\d+) doses\. Each tablet contains ([\d.]+) g .* boxes of (\d+)\."
)


def parse(q):
    mass, rate, doses, grams, box = _STEM.search(q.stem).groups()
    return int(mass), int(rate), int(doses), Fraction(grams), int(box)


def oracle(q):
    mass, rate, doses, grams, box = parse(q)
    daily = mass * rate
    per_dose = Fraction(daily, doses)
    tablets = per_dose / (grams * 1000)
    assert tablets.denominator == 1
    return {
        "a": str(daily),
        "b": str(per_dose),
        "c": str(tablets),
        "d": str(box // (int(tablets) * doses)),
    }


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=40)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_daily_dose():
    """A daily dose that is two tablets per dose too high, used correctly in (b), (c) and (d)."""
    done = 0
    for q in exam_questions(TOPIC, 80):
        mass, rate, doses, grams, box = parse(q)
        strength = int(grams * 1000)
        wrong = mass * rate + 2 * doses * strength
        b = wrong // doses
        c = b // strength
        d = box // (c * doses)
        if d == 0 or d in (c, b):
            continue
        got = grade(SKILL, q, {"a": str(wrong), "b": str(b), "c": str(c), "d": str(d)})
        assert got.parts["a"].marks == 0
        for label in "bcd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )
        done += 1
    assert done >= 10


def test_follow_through_from_a_wrong_tablet_count():
    done = 0
    for q in exam_questions(TOPIC, 60):
        mass, rate, doses, grams, box = parse(q)
        daily = mass * rate
        strength = int(grams * 1000)
        tablets = daily // doses // strength
        wrong = tablets + 1
        d = box // (wrong * doses)
        if d == 0 or d == wrong:
            continue  # a repeat of the earlier reply is not followed through
        got = grade(SKILL, q, {"a": str(daily), "b": str(daily // doses), "c": str(wrong), "d": str(d)})
        assert got.parts["c"].marks == 0
        if d != box // (tablets * doses):
            assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
            done += 1
    assert done >= 5


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
