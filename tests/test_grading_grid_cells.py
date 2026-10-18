"""The ``grid_cells`` answer kind (`mathgen.grading.grid`): reading a reply and the four match modes."""

from __future__ import annotations

import itertools

import pytest

from mathgen.answers import ANSWER_KINDS
from mathgen.core import Question
from mathgen.grading import GRADERS, grade
from mathgen.grading import grid

L_SHAPE = [[0, 0], [1, 0], [2, 0], [0, 1]]
CROSS_NET = [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [1, 3]]


def _question(cells, match="exact", size=(6, 6)) -> Question:
    answer = grid.cells_answer(cells, size, match)
    return Question(
        topic="t", level=0, seed=1, text="Shade the cells.", latex="", answer=answer,
        answer_text=grid.cell_text(cells), answer_latex="", working=[], difficulty="easy", distractors=[],
        answer_kind="grid_cells",
    )  # fmt: skip


def _ok(question, reply) -> bool:
    return grade("grid_cells", question, reply).correct


def test_the_kind_is_registered_in_both_lists():
    assert "grid_cells" in GRADERS and "grid_cells" in ANSWER_KINDS


@pytest.mark.parametrize(
    "reply",
    [
        "(0, 0) (1, 0) (2, 0) (0, 1)",
        "(0,1)(2,0)(1,0)(0,0)",
        "(0, 0), (1, 0), (2, 0) and (0, 1)",
        "(0, 0) (1, 0) (2, 0) (0, 1) (0, 0)",  # a cell twice counts once
        "[[0, 0], [1, 0], [2, 0], [0, 1]]",
        '{"cells": [[0, 1], [0, 0], [2, 0], [1, 0]]}',
        "  \n(0, 0)\n(1, 0)\n(2, 0)\n(0, 1)  ",
        "[(0, 0), (1, 0), (2, 0), (0, 1)]",  # the typed pairs in list brackets
        "{(0,0),(1,0),(2,0),(0,1)}",  # ... or in set braces
        "[(0, 0) (1, 0) (2, 0) (0, 1)]",
        " [ (0, 0), (1, 0), (2, 0), (0, 1) ] ",
    ],
)
def test_exact_accepts_every_spelling(reply):
    assert _ok(_question(L_SHAPE), reply)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "(0, 0) (1, 0) (2, 0)",  # one cell short
        "(0, 0) (1, 0) (2, 0) (0, 1) (1, 1)",  # one cell over
        "(1, 0) (2, 0) (3, 0) (1, 1)",  # moved
        "(0, 0) (1, 0) (2, 0) (0, 1) (6, 0)",  # off the grid
        "(0, 0) (1, 0) (2, 0) (0, 1) (-1, 0)",
        "(0, 0) (1, 0) (2, 0) (0.0, 1)",
        "(0, 0) (1, 0) (2, 0) (1/1, 1)",
        "(0, 0) (1, 0) (2, 0) (0, 1) x",
        "the cells (0, 0) (1, 0) (2, 0) (0, 1)",
        "[[0, 0], [1, 0], [2, 0], [0, 1.0]]",
        "[[0, 0], [1, 0], [2, 0], [0, true]]",
        '{"cells": [[0, 0], [1, 0], [2, 0], [0, 1]], "extra": 1}',
        '{"cells": "(0, 0)"}',
        "[[0, 0], [1, 0], [2, 0], [0, 1]",  # broken JSON
        "[" * 5000 + "]" * 5000,
        "(0, 0) " * 401,
        "[(0, 0), (1, 0), (2, 0), (0, 1)",  # unbalanced brackets
        "(0, 0), (1, 0), (2, 0), (0, 1)]",
        "[(0, 0), (1, 0), (2, 0), (0, 1)}",  # mismatched brackets
        "[(0, 0), (1, 0), (2, 0), (0, 1), x]",  # filler other than separators, inside the brackets
        "[(0, 0), (1, 0), (2, 0), (0, 1)] x",
        "x [(0, 0), (1, 0), (2, 0), (0, 1)]",
        "[[(0, 0), (1, 0), (2, 0), (0, 1)]]",  # only one outer pair is allowed
        "[(0, 0), (1, 0), (2, 0), (0, 1), (6, 0)]",  # off the grid
        "[(0, 0), (1, 0), (2, 0)]",  # one short
        "[(0, 0), (1, 0), (2, 0), (0, 1)] " * 2 + "[",
        "[(0, 0) " * 401 + "]",
    ],
)
def test_exact_refuses_everything_else(reply):
    assert not _ok(_question(L_SHAPE), reply)


def test_translate_allows_a_move_and_nothing_else():
    q = _question(L_SHAPE, "translate")
    assert _ok(q, "(3, 4) (4, 4) (5, 4) (3, 5)")
    assert not _ok(q, "(3, 4) (4, 4) (5, 4) (5, 5)")  # flipped
    assert not _ok(q, "(4, 4) (5, 4) (6, 4) (4, 5)")  # runs off the 6 by 6 grid


def test_congruent_allows_turns_and_flips():
    q = _question(L_SHAPE, "congruent")
    assert _ok(q, "(3, 4) (4, 4) (5, 4) (5, 5)")  # flipped
    assert _ok(q, "(0, 0) (0, 1) (0, 2) (1, 2)")  # turned
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (1, 1)")  # a T
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (0, 1) (0, 2)")


def test_cube_net_accepts_any_net():
    q = _question(CROSS_NET, "cube_net")
    assert _ok(q, grid.cell_text(CROSS_NET))
    assert _ok(q, "(0, 0) (1, 0) (1, 1) (2, 1) (2, 2) (3, 2)")  # the staircase net
    assert _ok(q, "(1, 0) (0, 1) (1, 1) (2, 1) (3, 1) (2, 2)")  # a 1-4-1 net, flaps on different cells


def test_cube_net_refuses_non_nets():
    q = _question(CROSS_NET, "cube_net")
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (3, 0) (4, 0) (5, 0)")  # a row of six
    assert not _ok(q, "(0, 0) (1, 0) (0, 1) (1, 1) (2, 1) (3, 1)")  # a 2 by 2 block
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (3, 0) (0, 1) (3, 1)")  # a U: both flaps fold onto the same face
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (3, 0) (1, 1)")  # five cells
    assert not _ok(q, "(0, 0) (1, 0) (2, 0) (4, 0) (1, 1) (1, 2)")  # not joined


def _free_hexominoes():
    shapes = {frozenset([(0, 0)])}
    for _ in range(5):
        grown = set()
        for shape in shapes:
            for (x, y), (dx, dy) in itertools.product(shape, ((1, 0), (-1, 0), (0, 1), (0, -1))):
                cell = (x + dx, y + dy)
                if cell not in shape:
                    grown.add(min(grid.orientations(shape | {cell}), key=sorted))
        shapes = grown
    return shapes


def test_exactly_eleven_of_the_thirty_five_hexominoes_are_cube_nets():
    shapes = _free_hexominoes()
    assert len(shapes) == 35
    assert sum(grid.is_cube_net(s) for s in shapes) == 11


def test_a_net_is_a_net_in_every_orientation():
    for shape in _free_hexominoes():
        verdicts = {grid.is_cube_net(o) for o in grid.orientations(shape)}
        assert len(verdicts) == 1


def test_cells_answer_refuses_cells_off_the_grid():
    with pytest.raises(ValueError):
        grid.cells_answer([[6, 0]], (6, 6))
    with pytest.raises(ValueError):
        grid.cells_answer([[0, 0]], (6, 6), "rotate")


def test_a_malformed_answer_grades_nothing_correct():
    q = _question(L_SHAPE)
    q.answer = {"cells": L_SHAPE, "grid": [6, 6], "match": "anything"}
    assert not _ok(q, grid.cell_text(L_SHAPE))
    q.answer = L_SHAPE
    assert not _ok(q, grid.cell_text(L_SHAPE))


def test_the_reply_is_reported_sorted():
    result = grade("grid_cells", _question(L_SHAPE), "(0, 1) (2, 0) (0, 0) (1, 0)")
    assert result.given_normalised == "(0, 0) (1, 0) (2, 0) (0, 1)"
