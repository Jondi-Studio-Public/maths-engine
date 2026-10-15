"""Exam-tier checks for `f10-grid-reference-pathway`: finishing column, finishing row, shortest way back in km."""

import re
from fractions import Fraction

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
    decimal_text,
    text_of,
)

TOPIC = "exd_x_sc2_grid_hike"
SKILL = "f10-grid-reference-pathway"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _facts(q):
    text = text_of(q)
    cols = LETTERS.index(re.search(r"columns A to ([A-Z])", text).group(1)) + 1
    rows = int(re.search(r"rows 1 to (\d+)", text).group(1))
    north_first = "northern edge" in text
    side = Fraction(re.search(r"grid square is ([\d.]+) km", text).group(1))
    start = re.search(r"starts in square ([A-Z])(\d+)", text)
    sc, sr = LETTERS.index(start.group(1)) + 1, int(start.group(2))
    legs = [
        (Fraction(d) / side, w)
        for d, w in re.findall(r"([\d.]+) km (north|south|east|west)", text.split("starts in")[1])
    ]
    return cols, rows, north_first, side, sc, sr, legs


def _finish(q):
    cols, rows, north_first, side, sc, sr, legs = _facts(q)
    col, row = sc, sr
    for squares, word in legs:
        assert squares.denominator == 1
        n = int(squares)
        if word == "east":
            col += n
        elif word == "west":
            col -= n
        elif word == "north":
            row += -n if north_first else n
        else:
            row += n if north_first else -n
        assert 1 <= col <= cols and 1 <= row <= rows
    return col, row


def oracle(q):
    cols, rows, north_first, side, sc, sr, legs = _facts(q)
    col, row = _finish(q)
    return {"a": str(col), "b": str(row), "c": decimal_text((abs(col - sc) + abs(row - sr)) * side, 1)}


def wrong_chain(q):
    cols, rows, north_first, side, sc, sr, legs = _facts(q)
    col, row = _finish(q)
    wrong_col = col + 1 if col < cols else col - 1
    wrong_row = row  # only (a) is wrong; (b) is right, (c) follows from the learner's own column
    return {
        "a": str(wrong_col),
        "b": str(wrong_row),
        "c": decimal_text((abs(wrong_col - sc) + abs(wrong_row - sr)) * side, 1),
    }


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("f10_grid_reference_pathway", seed=s, tier="hard"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_column():
    """(a) wrong, (b) right, (c) worked from the learner's column: (c) earns its marks by follow-through."""
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_walk_stays_on_the_map_and_changes_both_coordinates():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        cols, rows, north_first, side, sc, sr, legs = _facts(q)
        col, row = _finish(q)
        assert col != sc and row != sr


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "901", "b": "902", "c": "903.5"})
