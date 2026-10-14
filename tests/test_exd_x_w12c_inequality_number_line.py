"""Tests the exam chain for inequalities on a number line: the shaded interval, a second inequality, count and sum.

The oracle reads the shaded interval from the figure, parses the second inequality from the prompt, and finds the
integers satisfying both by brute force from the two statements themselves, never from the solved bounds or the key.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

import pytest

from mathgen import figures, make
from mathgen.grading import grade as grade_kind
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_w12c_inequality_number_line"
SKILL_ID = "inequality-solutions-number-line"
HARD_TOPIC = "fg1_inequality_number_line"
HARD_KWARGS = {"tier": "hard"}
_SECOND = re.compile(r"satisfy (-?\d*)x (?:([+-]) (\d+) )?(<|≤|>|≥) (-?\d+)\.")
_OPS = {"<": lambda l, r: l < r, "≤": lambda l, r: l <= r, ">": lambda l, r: l > r, "≥": lambda l, r: l >= r}


def _second(q):
    m = _SECOND.search(q.parts[1].prompt)
    assert m, q.parts[1].prompt
    coef, sign, const, op, rhs = m.groups()
    p = -1 if coef == "-" else (int(coef) if coef else 1)
    c = (int(const) * (1 if sign == "+" else -1)) if const else 0
    return p, c, op, int(rhs)


def oracle(q):
    """The shaded interval's statement, the second inequality's solution probe, the integers satisfying both."""
    iv = q.figure["intervals"][0]
    a, b = int(iv["from"]), int(iv["to"])
    open_s, open_e = bool(iv.get("open_start")), bool(iv.get("open_end"))
    p, c, op, rhs = _second(q)
    shown = [i for i in range(-15, 16) if (i > a if open_s else i >= a) and (i < b if open_e else i <= b)]
    both = [i for i in shown if _OPS[op](p * i + c, rhs)]
    interval = f"{a} {'<' if open_s else '≤'} x {'<' if open_e else '≤'} {b}"
    return interval, (p, c, op, rhs), shown, both


def _solved_agrees(text: str, statement) -> bool:
    """The solved inequality "x op u" holds at exactly the points where the original statement holds."""
    p, c, op, rhs = statement
    m = re.fullmatch(r"x (<|≤|>|≥) (-?\d+(?:\.\d+)?)", text)
    if not m:
        return False
    u = F(m.group(2))
    return all(_OPS[op](p * pt + c, rhs) == _OPS[m.group(1)](pt, u) for pt in (F(k, 4) for k in range(-80, 81)))


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["inequality", "inequality", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(p.sensible_range is not None for p in q.parts if p.answer_kind == "number")


def test_figure_is_a_shaded_interval_and_hides_the_inequality(questions):
    for q in questions[:100]:
        assert q.figure is not None and figures.validate(q.figure) == []
        assert len(q.figure["intervals"]) == 1
        figures.assert_no_leak(q.figure, [q.parts[0].answer_text])


def test_both_directions_and_strictness_occur(questions):
    uppers = sum(oracle(q)[3][0] == oracle(q)[2][0] for q in questions[:200])  # the second inequality cuts the top
    assert 40 < uppers < 160
    assert {"<" in q.parts[0].answer_text.split("x")[0] for q in questions[:100]} == {True, False}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        interval, statement, shown, both = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        ok = (
            a_t == interval
            and _solved_agrees(b_t, statement)
            and int(c_t) == len(both)
            and int(d_t) == sum(both)
            and 3 <= len(both) < len(shown)
            and both == list(range(both[0], both[-1] + 1))
        )
        if not ok:
            bad.append((q.parts[1].prompt, canonical(q)))
    assert bad == [], bad[:2]


def test_inequality_parts_are_graded_as_sets():
    """The typed answer is the solution set: an equivalent way of writing it is accepted by the real grader."""
    q = exam(TOPIC, 3)
    interval, _statement, _shown, _both = oracle(q)
    a, b = re.findall(r"-?\d+", interval)[:2]
    flipped = q.parts[0].answer_text
    # Writing the double inequality in the other order is the same set.
    m = re.fullmatch(r"(-?\d+) (<|≤) x (<|≤) (-?\d+)", flipped)
    assert m
    swap = {"<": ">", "≤": "≥"}
    reply = f"{m.group(4)} {swap[m.group(3)]} x {swap[m.group(2)]} {m.group(1)}"
    assert grade_kind("inequality", q.parts[0], reply).correct


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_count():
    """A wrong count carries through the sum of the run, which keeps the end the number line fixes."""

    def carried(q):
        _interval, _statement, shown, both = oracle(q)
        n = len(both)
        for wrong in (n + 2, n + 3, n - 2, n + 1):
            if wrong < 2:
                continue
            if both[0] == shown[0]:  # the second inequality cut the top: the run keeps its lowest integer
                total = wrong * both[0] + wrong * (wrong - 1) // 2
            else:
                total = wrong * both[-1] - wrong * (wrong - 1) // 2
            if total not in (wrong, 0, 1, -1, sum(both)):  # not a repeat of another reply
                return "c", str(wrong), {"d": str(total)}
        raise AssertionError("no usable wrong count")

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
