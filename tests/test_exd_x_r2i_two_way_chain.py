"""Tests for the multi-part exam source of senior-interpret-information-presented-in-two-way-tables: complete a
table from its total, then compare two row percentages. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_two_way_chain  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_two_way_chain"
SKILL = "senior-interpret-information-presented-in-two-way-tables"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]
FT_LABELS = {"c", "d"}


def _one(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _table(q):
    n = int(re.search(r"survey of (\d+)", q.stem).group(1))
    rows = re.search(r"Rows: ([^.]*)\.", q.stem).group(1).split(", ")
    cols = re.search(r"Columns: ([^.]*)\.", q.stem).group(1).split(", ")
    cell = {}
    for r in rows:
        for c in cols:
            m = re.search(rf"{re.escape(r)}/{re.escape(c)}: (\d+|not recorded)", q.stem)
            cell[(r, c)] = None if m.group(1) == "not recorded" else int(m.group(1))
    return n, rows, cols, cell


def _facts(q):
    n, rows, cols, cell = _table(q)
    (missing,) = [k for k, v in cell.items() if v is None]
    assert missing == (rows[1], cols[1])
    c22 = n - sum(v for v in cell.values() if v is not None)
    c11, c12, c21 = cell[(rows[0], cols[0])], cell[(rows[0], cols[1])], cell[(rows[1], cols[0])]
    p1 = Fraction(100 * c11, c11 + c12)
    p2 = Fraction(100 * c21, c21 + c22)
    return n, c11, c12, c21, c22, p1, p2


def oracle(q):
    n, c11, c12, c21, c22, p1, p2 = _facts(q)
    shown1, shown2 = Fraction(_one(p1)), Fraction(_one(p2))
    return {"a": str(c22), "b": _one(p1), "c": _one(p2), "d": _one(abs(shown1 - shown2))}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_source_wired():
    CHECKS["wired"]()


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


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_prompts_name_the_groups_and_the_response():
    for s in range(60):
        q = exam(TOPIC, s)
        n, rows, cols, cell = _table(q)
        assert f"in the {rows[0]} group" in q.parts[1].prompt
        assert f"in the {rows[1]} group" in q.parts[2].prompt
        assert f"in the {rows[1]} group whose response was {cols[1]}" in q.parts[0].prompt


def test_wrong_missing_cell_carries_into_the_second_percentage():
    def chain(q):
        n, c11, c12, c21, c22, p1, p2 = _facts(q)
        wrong_a = c22 + 5
        wrong_c = Fraction(_one(Fraction(100 * c21, c21 + wrong_a)))
        replies = {
            "a": str(wrong_a),
            "b": _one(p1),
            "c": _one(wrong_c),
            "d": _one(abs(Fraction(_one(p1)) - wrong_c)),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_first_percentage_carries_into_the_difference():
    def chain(q):
        n, c11, c12, c21, c22, p1, p2 = _facts(q)
        wrong_b = Fraction(_one(p1)) + 3
        replies = {
            "a": str(c22),
            "b": _one(wrong_b),
            "c": _one(p2),
            "d": _one(abs(wrong_b - Fraction(_one(p2)))),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)
