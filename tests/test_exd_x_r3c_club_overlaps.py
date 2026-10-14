"""Tests the multi-part exam source of `senior-determine-and-use-the-formulas-for-finding-the`.

The oracle reads the eight numbers from the rendered stem and solves for the seven Venn regions with a linear
system, which uses none of the inclusion-exclusion formulas the parts are built on.
"""

from __future__ import annotations

import re

import sympy

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

TOPIC = "exd_x_r3c_club_overlaps"
SKILL_ID = "senior-determine-and-use-the-formulas-for-finding-the"
SKILL = skill_for(SKILL_ID)


def parse(q):
    n, none, sa, sb, sc, ab, ac, bc = (int(v) for v in re.findall(r"\d+", q.stem))
    return n, none, sa, sb, sc, ab, ac, bc


def regions(q):
    n, none, sa, sb, sc, ab, ac, bc = parse(q)
    oa, ob, oc, tab, tac, tbc, t = sympy.symbols("oa ob oc tab tac tbc t")
    sol = sympy.solve(
        [
            oa + tab + tac + t - sa,
            ob + tab + tbc + t - sb,
            oc + tac + tbc + t - sc,
            tab + t - ab,
            tac + t - ac,
            tbc + t - bc,
            oa + ob + oc + tab + tac + tbc + t - (n - none),
        ],
        [oa, ob, oc, tab, tac, tbc, t],
        dict=True,
    )[0]
    return {str(k): int(v) for k, v in sol.items()}


def oracle(q):
    n, none, *_ = parse(q)
    r = regions(q)
    return {
        "a": str(n - none),
        "b": str(r["t"]),
        "c": str(r["tab"] + r["tac"] + r["tbc"]),
        "d": str(r["oa"] + r["ob"] + r["oc"]),
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


def test_follow_through_from_a_wrong_union():
    """A union that is two too low, used correctly in (b), (c) and (d)."""
    done = 0
    for q in exam_questions(TOPIC, 60):
        n, none, sa, sb, sc, ab, ac, bc = parse(q)
        wrong = n - none - 2
        b = wrong - (sa + sb + sc) + (ab + ac + bc)
        c = ab + ac + bc - 3 * b
        d = wrong - b - c
        if min(b, c, d) < 1 or len({wrong, b, c, d}) < 4:
            continue  # a non-positive count is outside the sensible range; repeats are not followed through
        got = grade(SKILL, q, {"a": str(wrong), "b": str(b), "c": str(c), "d": str(d)})
        assert got.parts["a"].marks == 0
        for label in "bcd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )
        done += 1
    assert done >= 10


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [1, 2, 1, 1]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
