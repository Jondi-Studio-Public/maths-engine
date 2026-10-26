"""Round 4 fixes in the interactive, grid, matrix and chain graders and the two generators behind them.

A2 numeric bar labels, A4 grid references on the grid map, C3 any column basis, D16 en dash in a matrix entry and a
full stop on a proof chain.
"""

from __future__ import annotations

import itertools

import pytest
import sympy

from mathgen import engine
from mathgen.answers import ANSWER_KINDS
from mathgen.core import Question
from mathgen.grading import GRADERS, grade, grid
from mathgen.grading.matrix import column_basis_answer, is_column_basis, read_column_numbers

SKILLS = {s.id: s for s in engine.load_skills()[1]}


def _make(skill_id: str, source: int, seed: int):
    skill = SKILLS[skill_id]
    return skill, engine.make_with_id(skill, seed, source=skill.sources[source])[0]


def _text(distractor) -> str:
    return distractor if isinstance(distractor, str) else distractor.text


def _ok(skill_id: str, source: int, seed: int, reply) -> bool:
    skill, question = _make(skill_id, source, seed)
    return engine.grade(skill, question, reply).correct


# -- A2: a numeric bar label is compared by value --------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, correct",
    [
        ("1.8 = 4, 1.9 = 2, 2 = 1, 2.1 = 3, 2.2 = 4", True),
        ("1.8 = 4, 1.9 = 2, 2.0 = 1, 2.1 = 3, 2.2 = 4", True),
        ("2.20 = 4, 2 = 1, 1.8 = 4, 1.9 = 2, 2.1 = 3", True),
        ("1.8 = 4, 1.9 = 2, 2 = 1, 2.1 = 3, 2.2 = 5", False),
        ("1.8 = 4, 1.9 = 2, 2 = 1, 2.1 = 3, 2.3 = 4", False),
        ("1.8 = 4, 1.9 = 2, 2 = 1, 2.0 = 3, 2.2 = 4", False),
        ("1.8 = 4, 1.9 = 2, 2 = 1, 2.1 = 3, 2.1 = 4", False),
    ],
)
def test_dot_plot_numeric_labels_match_by_value(reply, correct):
    assert _ok("construct-dot-plots-from-data", 1, 4, reply) is correct


# -- A4: grid references on the grid map -----------------------------------------------------------------------------


def test_grid_map_shows_the_key_as_a_grid_reference():
    _, question = _make("place-features-on-a-grid-map", 0, 3)
    assert question.answer_text == "D4"
    assert question.answer["cells"] == [[3, 3]]
    assert all("(" not in _text(d) for d in question.distractors)


@pytest.mark.parametrize("reply", ["D4", "d4", " d4. ", "D 4", "D04", "(3, 3)", "[[3, 3]]", '{"cells": [[3, 3]]}'])
def test_grid_map_accepts_the_reference_and_the_cell(reply):
    assert _ok("place-features-on-a-grid-map", 0, 3, reply)


@pytest.mark.parametrize("reply", ["E4", "D5", "4D", "D", "4", "D4 D5", "D4 (3, 3)", "(3, 3) D4", "D40", "Z4", ""])
def test_grid_map_refuses_other_squares_and_mixed_forms(reply):
    assert not _ok("place-features-on-a-grid-map", 0, 3, reply)


def test_grid_map_regions_take_several_references():
    skill, question = _make("place-features-on-a-grid-map", 2, 1)
    cells = [tuple(c) for c in question.answer["cells"]]
    assert question.answer_text == grid.ref_text(cells)
    reply = ", ".join(grid.ref_text([c]).lower() for c in cells)
    assert engine.grade(skill, question, reply).correct
    assert not engine.grade(skill, question, grid.ref_text(cells[:-1])).correct


@pytest.mark.parametrize("source", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", range(8))
def test_grid_map_key_and_every_distractor_round_trip(source, seed):
    _, question = _make("place-features-on-a-grid-map", source, seed)
    for part in question.parts or [question]:
        if part.answer_kind != "grid_cells":
            continue
        assert grade("grid_cells", part, part.answer_text).correct
        for d in part.distractors:
            assert not grade("grid_cells", part, _text(d)).correct


def test_references_are_off_unless_the_question_asks_for_them():
    answer = grid.cells_answer([[3, 3]], (7, 6))
    assert "refs" not in answer
    q = Question(
        topic="t", level=0, seed=1, text="Shade the cell.", latex="", answer=answer, answer_text="(3, 3)",
        answer_latex="", working=[], difficulty="easy", distractors=[], answer_kind="grid_cells",
    )  # fmt: skip
    assert grade("grid_cells", q, "(3, 3)").correct
    assert not grade("grid_cells", q, "D4").correct


def test_ref_text_sorts_by_row_then_column_and_refuses_columns_past_z():
    assert grid.ref_text([(1, 1), (0, 0), (1, 0)]) == "A1 B1 B2"
    with pytest.raises(ValueError):
        grid.ref_text([(26, 0)])


# -- C3: any independent columns are a basis -------------------------------------------------------------------------

COLUMN_SPACE = "ucv1-column-space-basis-dimension"


def _reply_c(skill, question, text) -> bool:
    graded = engine.grade(skill, question, {"a": "2", "b": "2", "c": text, "d": "(0, 0)"})
    return graded.parts["c"].correct


@pytest.mark.parametrize(
    "reply, correct",
    [
        ("1, 3", True),
        ("3, 1", True),
        ("1, 4", True),
        ("2, 3", True),
        ("3, 4", True),
        ("columns 1 and 4", True),
        ("c2, c3", True),
        ("{2, 3}", True),
        ("1, 3.", True),
        ("1, 2", False),
        ("1", False),
        ("1, 3, 4", False),
        ("1, 2, 3, 4", False),
        ("1, 1", False),
        ("0, 3", False),
        ("1, 5", False),
        ("13", False),
        ("1, x", False),
        ("", False),
    ],
)
def test_seed_3_basis_replies(reply, correct):
    skill, question = _make(COLUMN_SPACE, 3, 3)
    assert question.answer_text.endswith("(c) 1, 3; (d) (-1, -2)")
    assert _reply_c(skill, question, reply) is correct


@pytest.mark.parametrize("seed", range(20))
def test_every_independent_pair_is_accepted_and_no_dependent_pair_is(seed):
    skill, question = _make(COLUMN_SPACE, 3, seed)
    part = question.parts[2]
    rows = part.answer["matrix"]
    for i, j in itertools.combinations(range(1, 5), 2):
        independent = sympy.Matrix(rows)[:, [i - 1, j - 1]].rank() == 2
        assert _reply_c(skill, question, f"{i}, {j}") is independent
    assert part.answer["rank"] == 2 and part.answer["pivots"] == sorted(map(int, part.answer_text.split(", ")))


@pytest.mark.parametrize("seed", range(20))
def test_no_distractor_of_the_basis_part_is_a_basis(seed):
    skill, question = _make(COLUMN_SPACE, 3, seed)
    part = question.parts[2]
    assert len(part.distractors) >= 2
    for d in part.distractors:
        assert not is_column_basis(part.answer, read_column_numbers(_text(d))), _text(d)
        assert not _reply_c(skill, question, _text(d))


def test_the_basis_part_has_its_own_answer_kind():
    assert "column_basis" in GRADERS and "column_basis" in ANSWER_KINDS
    _, question = _make(COLUMN_SPACE, 3, 3)
    assert [p.answer_kind for p in question.parts] == ["number", "number", "column_basis", "matrix"]


def test_column_basis_answer_refuses_pivots_that_are_not_a_basis():
    rows = [[1, 2, 0], [2, 4, 1], [3, 6, 1]]
    assert column_basis_answer(rows, [1, 3])["rank"] == 2
    with pytest.raises(ValueError):
        column_basis_answer(rows, [1, 2])
    with pytest.raises(ValueError):
        column_basis_answer(rows, [1])


def test_basis_with_fractional_entries_is_checked_exactly():
    rows = [["1/3", 1, 2], [1, 3, 6]]
    answer = column_basis_answer(rows, [1])
    assert is_column_basis(answer, [2]) and is_column_basis(answer, [3])


@pytest.mark.parametrize("text", ["1, 3", "1 3", "1;3", "columns 1, 3", "Col 1 and col 3", "[1, 3]", "c1 c3"])
def test_column_numbers_are_read(text):
    assert read_column_numbers(text) == [1, 3]


@pytest.mark.parametrize("text", ["", "one, three", "1.5, 3", "-1, 3", "1, 3 because", "1+3", "x1, 3", "1" * 300])
def test_column_numbers_are_refused(text):
    assert read_column_numbers(text) is None


# -- D16: an en dash is a minus in a matrix entry; a full stop closes a proof chain ----------------------------------


def test_matrix_en_dash_is_a_minus():
    reply = "[[2√(5)/5, √(5)/5], [–√(5)/5, 2√(5)/5]]"
    assert _ok("uc2-orthogonal-diagonalisation", 0, 21, reply)
    assert not _ok("uc2-orthogonal-diagonalisation", 0, 21, "[[2√(5)/5, √(5)/5], [√(5)/5, 2√(5)/5]]")


def test_matrix_en_dash_in_a_plain_matrix():
    q = Question(
        topic="t", level=0, seed=1, text="Give A.", latex="", answer=[[1, -2], [-3, 4]], answer_text="[[1, -2], [-3, 4]]",
        answer_latex="", working=[], difficulty="easy", distractors=[], answer_kind="matrix",
    )  # fmt: skip
    assert grade("matrix", q, "[[1, –2], [–3, 4]]").correct
    assert not grade("matrix", q, "[[1, 2], [–3, 4]]").correct


def test_step_chain_accepts_a_closing_full_stop():
    skill, question = _make("proof-parity-steps", 0, 11)
    key = question.answer_text
    assert engine.grade(skill, question, key).correct
    assert engine.grade(skill, question, key + ".").correct
    assert engine.grade(skill, question, key.replace(" = ", "\n") + ".").correct


def test_step_chain_still_refuses_a_wrong_last_line_with_a_full_stop():
    skill, question = _make("proof-parity-steps", 0, 11)
    first = question.answer_text.split(" = ")[0]
    assert not engine.grade(skill, question, first + " = " + first + " + 1.").correct
