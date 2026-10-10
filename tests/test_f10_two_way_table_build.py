"""Generator-specific proofs for `f10-two-way-table-relationship` (`answer: table`).

The independent oracle (it reads only the question text and solves with SymPy) is
`tests/test_f10_two_way_table_oracle.py`.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table, table_prompt
from mathgen.skills import load_skills
import mathgen.generators.f10_two_way_table_build as gen

SKILL_ID = "f10-two-way-table-relationship"
TOPIC = "f10_two_way_table"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)


def _skill():
    _domains, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


SKILL = _skill()
SOURCES = {src.difficulty: src for src in SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _filled(q):
    """The complete grid (integers) from the masked rows plus the blanks."""
    rows = [list(r) for r in q.answer["rows"]]
    for key, value in q.answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    return [r[1:] for r in rows]


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "table"
    assert [s.difficulty for s in SKILL.sources] == TIERS
    assert "AC9M10ST04" in SKILL.tags


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism(difficulty):
    for seed in range(15):
        assert _make(difficulty, seed).to_json() == _make(difficulty, seed).to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_completed_table_adds_up_both_ways(difficulty):
    for seed in SEEDS:
        grid = _filled(_make(difficulty, seed))
        n_rows, n_cols = len(grid) - 1, len(grid[0]) - 1
        for i in range(n_rows):
            assert sum(grid[i][:n_cols]) == grid[i][n_cols]
        for j in range(n_cols):
            assert sum(grid[i][j] for i in range(n_rows)) == grid[n_rows][j]
        assert sum(grid[n_rows][:n_cols]) == grid[n_rows][n_cols]
        assert all(v >= 0 for row in grid for v in row)


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_each_form_and_rejects_every_distractor(difficulty):
    miscon = {m.id for m in SKILL.misconceptions}
    for seed in SEEDS:
        q = _make(difficulty, seed)
        blanks = q.answer["blanks"]
        keys = sorted(blanks, key=lambda k: tuple(map(int, k.split(","))))
        assert grade_table(q, q.answer_text).correct
        assert grade_table(q, json.dumps({k: str(blanks[k]) for k in keys})).correct
        assert grade_table(q, ", ".join(str(blanks[k]) for k in keys)).correct
        assert 2 <= len(q.distractors) <= 3
        for d in q.distractors:
            assert not grade_table(q, d).correct, (difficulty, seed, d)
            assert q.option_notes[d] in miscon


@pytest.mark.parametrize("difficulty", TIERS)
def test_changing_or_dropping_any_one_blank_is_rejected(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        blanks = q.answer["blanks"]
        for key in blanks:
            bad = {k: str(v) for k, v in blanks.items()}
            bad[key] = str(blanks[key] + 1)
            assert not grade_table(q, json.dumps(bad)).correct
            assert not grade_table(q, json.dumps({k: v for k, v in bad.items() if k != key})).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_masked_prompt_hides_every_blank_and_blank_labels_are_distinct(difficulty):
    for seed in range(40):
        q = _make(difficulty, seed)
        prompt = table_prompt(q.answer)
        for key in q.answer["blanks"]:
            r, c = map(int, key.split(","))
            assert prompt["rows"][r][c] is None
        assert len(set(q.answer_text.split(", "))) == len(q.answer["blanks"])


@pytest.mark.parametrize("difficulty", TIERS)
def test_blank_positions_and_answers_vary(difficulty):
    qs = [make(TOPIC, seed=s, difficulty=difficulty) for s in range(200)]
    patterns = Counter(frozenset(q.answer["blanks"]) for q in qs)
    assert len(patterns) >= 10 and max(patterns.values()) <= 0.2 * 200
    assert len(Counter(q.answer_text for q in qs)) >= 150


def test_medium_blanks_can_be_filled_by_single_unknown_lines_and_hard_cannot():
    for seed in range(60):
        q = _make("medium", seed)
        grid = _filled(q)
        unknown = {(int(k.split(",")[0]), int(k.split(",")[1]) - 1) for k in q.answer["blanks"]}
        assert gen._propagate(len(grid) - 1, len(grid[0]) - 1, unknown) == set()
    for seed in range(60):
        q = _make("hard", seed)
        grid = _filled(q)
        unknown = {(int(k.split(",")[0]), int(k.split(",")[1]) - 1) for k in q.answer["blanks"]}
        assert gen._propagate(3, 3, unknown), "hard tier must need the stated condition"
        _rank, solution = gen._solve(3, 3, grid, unknown)
        assert solution is None, "without the condition the hard table must be underdetermined"


def test_solver_finds_unique_flags_underdetermined_and_inconsistent():
    grid = gen._full_grid([[2, 3], [4, 5]])
    _rank, solution = gen._solve(2, 2, grid, {(0, 0), (0, 1)})
    assert solution == {(0, 0): 2, (0, 1): 3}
    # all four interior cells unknown with only the totals given: many tables fit
    assert gen._solve(2, 2, grid, {(0, 0), (0, 1), (1, 0), (1, 1)})[1] is None
    assert gen._solve(2, 2, grid, {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)})[1] is None
    bad = [row[:] for row in grid]
    bad[0][2] = 99
    assert gen._solve(2, 2, bad, {(0, 0)})[1] is None  # inconsistent totals


def test_text_is_one_line_and_names_the_table():
    for tier in TIERS:
        for seed in range(60):
            q = _make(tier, seed)
            assert "\n" not in q.text and "  " not in q.text
            assert "Table (" in q.text and "?" in q.text


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.f10_two_way_table(__import__("random").Random(0), tier="exam")
