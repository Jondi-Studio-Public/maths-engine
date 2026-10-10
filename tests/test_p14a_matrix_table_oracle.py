"""Independent oracle for the `p14a_matrix_table` questions: it re-derives each answer from the rendered question
text.

Parses only ``question.text`` / ``question.answer_text`` / ``question.
distractors`` and re-derives the answer from scratch by reading the rendered
table straight out of the prompt text. Does NOT import
``mathgen.generators.p14a_matrix_table_build`` -- if this file ever needs a
name from that module, the check is not independent.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make

TOPIC = "p14a_matrix_table"

# The table is always introduced by the literal "below: " connector and
# closed by ". Find" or ". How many", regardless of which of the 3 intro
# phrasings a given seed drew (every template in the generator shares this
# connector on purpose, precisely so an independent parser can locate the
# table without depending on the surrounding prose).
_TABLE_RE = re.compile(r"below: (?P<table>.*?)\.\s*(?P<query>(?:Find|How many).*)$", re.S)

STORE_NAMES = ("Riverside", "Hillcrest", "Oakville", "Bayview", "Meadowbrook", "Fairfield")
PRODUCT_NAMES = ("Bats", "Balls", "Nets", "Hoops", "Cones", "Bibs")
TOWN_NAMES = ("Milbrook", "Ashford", "Denton", "Rosewood", "Carrington", "Fairview")
NEIGHBOURHOOD_NAMES = (
    "Riverside Estate",
    "Hillcrest Heights",
    "Oakville Green",
    "Bayview Point",
    "Meadowbrook Fields",
    "Fairfield Corner",
    "Kingsford Reserve",
)


def _parse_table(table_text: str):
    rows = table_text.split("; ")
    header = [cell.strip() for cell in rows[0].split(" | ")]
    col_labels = header[1:]
    row_labels = []
    matrix = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row.split(" | ")]
        row_labels.append(cells[0])
        matrix.append([int(c) for c in cells[1:]])
    return row_labels, col_labels, matrix


def _find_label(pool, text: str) -> str | None:
    for label in pool:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return None


def _expected_easy(text: str) -> int:
    m = _TABLE_RE.search(text)
    assert m, f"could not locate table in {text!r}"
    row_labels, col_labels, matrix = _parse_table(m.group("table"))
    query = m.group("query")
    col = _find_label(col_labels, query)
    row = _find_label(row_labels, query)
    assert col is not None and row is not None, f"could not find row/col label in {query!r}"
    return matrix[row_labels.index(row)][col_labels.index(col)]


def _expected_medium(text: str) -> int:
    m = _TABLE_RE.search(text)
    assert m, f"could not locate table in {text!r}"
    row_labels, col_labels, matrix = _parse_table(m.group("table"))
    query = m.group("query")
    row = _find_label(row_labels, query)
    if row is not None:
        return sum(matrix[row_labels.index(row)])
    col = _find_label(col_labels, query)
    assert col is not None, f"could not find row/col label in {query!r}"
    j = col_labels.index(col)
    return sum(r[j] for r in matrix)


def _expected_adjacency(text: str, node_pool) -> int:
    m = _TABLE_RE.search(text)
    assert m, f"could not locate table in {text!r}"
    labels, _col_labels, matrix = _parse_table(m.group("table"))
    query = m.group("query")
    node = _find_label(node_pool, query)
    assert node is not None, f"could not find node label in {query!r}"
    idx = labels.index(node)
    if "leav" in query:
        return sum(matrix[idx])
    assert "arriv" in query, f"expected 'leaving/leave' or 'arriving/arrive' in {query!r}"
    return sum(row[idx] for row in matrix)


def _expected_hard(text: str) -> int:
    return _expected_adjacency(text, TOWN_NAMES)


def _expected_exam(text: str) -> int:
    return _expected_adjacency(text, NEIGHBOURHOOD_NAMES)


_EXPECTED = {
    "easy": _expected_easy,
    "medium": _expected_medium,
    "hard": _expected_hard,
    "exam": _expected_exam,
}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
@pytest.mark.parametrize("seed", range(60))
def test_matrix_table_oracle(seed: int, difficulty: str):
    q = make(TOPIC, seed=seed, difficulty=difficulty)
    expected = _EXPECTED[difficulty](q.text)
    given = int(q.answer_text)
    assert expected == given, (
        f"seed {seed}/{difficulty}: expected {expected}, answer_text says {q.answer_text}. Question: {q.text}"
    )
    for d in q.distractors:
        assert int(d) != expected, (
            f"seed {seed}/{difficulty}: distractor {d} equals the independently-derived "
            f"expected value {expected}. Question: {q.text}"
        )
