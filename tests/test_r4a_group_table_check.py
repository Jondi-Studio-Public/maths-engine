"""Tests `uc2-group-table-check`: the Cayley table is read back out of the question TEXT and every blank is re-derived by
searching for fillings that make a group (Latin square, identity, associativity); the tier rules (Latin propagation alone
for easy, not for medium and hard), the typed table grader on every key and distractor, and the exam chain."""

from __future__ import annotations

import itertools
import re

import mathgen.generators.r4a_group_table_check_build  # noqa: F401  (registers the topics)
from tests import r4a_support as sup

TOPIC = "r4a_group_table_check"
SKILL_ID = "uc2-group-table-check"


def _grid(text):
    """(headers, rows) from "Table (∘ — a — b): a — x — y; b — ? — z."; blanks are None."""
    match = re.search(r"Table \(([^)]*)\): (.*?)\.(?: |$)", text)
    headers = [h.strip() for h in match.group(1).split("—")][1:]
    rows = []
    for chunk in match.group(2).split("; "):
        cells = [c.strip() for c in chunk.split("—")]
        rows.append((cells[0], [None if c == "?" else c for c in cells[1:]]))
    return headers, rows


def _matrix(headers, rows):
    assert [r[0] for r in rows] == headers
    return [list(r[1]) for r in rows]


def _is_group(table, headers):
    m = len(headers)
    idx = {h: i for i, h in enumerate(headers)}
    t = [[idx[c] for c in row] for row in table]
    if any(sorted(row) != list(range(m)) for row in t) or any(sorted(col) != list(range(m)) for col in zip(*t)):
        return False
    if not any(all(t[e][j] == j and t[j][e] == j for j in range(m)) for e in range(m)):
        return False
    return all(t[t[a][b]][c] == t[a][t[b][c]] for a in range(m) for b in range(m) for c in range(m))


def _group_fillings(headers, matrix):
    blanks = [(r, c) for r, row in enumerate(matrix) for c, v in enumerate(row) if v is None]
    options = []
    for r, c in blanks:
        used = {v for v in matrix[r] if v} | {matrix[i][c] for i in range(len(matrix)) if matrix[i][c]}
        options.append([h for h in headers if h not in used])
    found = []
    for choice in itertools.product(*options):
        trial = [list(row) for row in matrix]
        for (r, c), v in zip(blanks, choice):
            trial[r][c] = v
        if _is_group(trial, headers):
            found.append({f"{r},{c + 1}": v for (r, c), v in zip(blanks, choice)})
    return found


def _latin_solves(headers, matrix):
    t = [list(row) for row in matrix]
    changed = True
    while changed:
        changed = False
        for r, row in enumerate(t):
            for c, v in enumerate(row):
                if v is None:
                    used = {x for x in t[r] if x} | {t[i][c] for i in range(len(t)) if t[i][c]}
                    left = [h for h in headers if h not in used]
                    if len(left) == 1:
                        t[r][c] = left[0]
                        changed = True
    return all(v is not None for row in t for v in row)


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "table")


def test_each_tier_blanks_have_exactly_the_stated_group_completion():
    for tier in sup.TIERS:
        for seed in range(40):
            question = sup.q(TOPIC, tier, seed)
            headers, rows = _grid(question.text)
            matrix = _matrix(headers, rows)
            fillings = _group_fillings(headers, matrix)
            assert len(fillings) == 1, (tier, seed)
            expected = {k: str(v) for k, v in question.answer["blanks"].items()}
            assert fillings[0] == expected, (tier, seed)


def test_tier_rules():
    for seed in range(60):
        headers, rows = _grid(sup.q(TOPIC, "easy", seed).text)
        matrix = _matrix(headers, rows)
        assert "e" in headers and _latin_solves(headers, matrix)
        headers, rows = _grid(sup.q(TOPIC, "medium", seed).text)
        matrix = _matrix(headers, rows)
        assert "e" in headers and not _latin_solves(headers, matrix)
        assert sum(v is None for row in matrix for v in row) == 4
        question = sup.q(TOPIC, "hard", seed)
        headers, rows = _grid(question.text)
        matrix = _matrix(headers, rows)
        assert "e" not in headers and not _latin_solves(headers, matrix)
        assert sum(v is None for row in matrix for v in row) == 6
        full = [list(row) for row in matrix]
        for key, value in question.answer["blanks"].items():
            r, c = map(int, key.split(","))
            full[r][c - 1] = value
        assert any(full[a][b] != full[b][a] for a in range(8) for b in range(8)), "hard tables are non-abelian"


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "table", seeds=40)


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC, seeds=25)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 40, "medium": 40, "hard": 40})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (3, 2, 2, 1), ("table", "number", "number", "number"), (0, 0, 0, 1))


def test_exam_keys_match_an_independent_search():
    for seed in range(60):
        question = sup.exam(TOPIC, seed)
        headers, rows = _grid(question.stem)
        matrix = _matrix(headers, rows)
        fillings = _group_fillings(headers, matrix)
        a, b, c, d = question.parts
        assert fillings == [{k: str(v) for k, v in a.answer["blanks"].items()}]
        full = [list(row) for row in matrix]
        for key, value in fillings[0].items():
            r, col = map(int, key.split(","))
            full[r][col - 1] = value
        ident = next(h for i, h in enumerate(headers) if all(full[i][j] == headers[j] for j in range(8)))

        def power_order(x):
            k, y = 1, x
            while y != ident:
                y = full[headers.index(y)][headers.index(x)]
                k += 1
            return k

        assert int(b.answer) == sum(1 for h in headers if full[headers.index(h)][headers.index(h)] == ident)
        element = re.search(r"order of the element (\w+)", c.prompt).group(1)
        assert int(c.answer) == power_order(element)
        assert int(d.answer) == sum(1 for h in headers if power_order(h) == 2) == int(b.answer) - 1


def test_exam_follow_through_for_the_elements_of_order_two():
    def replies(question):
        return {p.label: p.answer_text for p in question.parts} | {"b": str(int(question.parts[1].answer) + 2)}

    def expected(question, sent):
        return str(int(sent["b"]) - 1)

    sup.check_follow_through(SKILL_ID, TOPIC, "d", replies, expected)
