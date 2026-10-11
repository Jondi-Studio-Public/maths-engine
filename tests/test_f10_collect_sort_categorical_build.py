"""Generator-specific proofs for `f10-collect-sort-compare-categorical-data` (`answer: table`).

The independent oracle (it reads only the question text) is `tests/test_f10_collect_sort_categorical_oracle.py`.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table, table_prompt
from mathgen.skills import load_skills
import mathgen.generators.f10_collect_sort_categorical_build as gen

SKILL_ID = "f10-collect-sort-compare-categorical-data"
TOPIC = "f10_collect_sort_categorical"
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


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "table"
    assert [s.difficulty for s in SKILL.sources] == TIERS
    assert {"AC9MFST01", "AC9M1ST01"} <= set(SKILL.tags)


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism(difficulty):
    for seed in range(15):
        a, b = _make(difficulty, seed), _make(difficulty, seed)
        assert a.to_json() == b.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_json_and_labelled_and_bare_forms(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert grade_table(q, q.answer_text).correct
        blanks = q.answer["blanks"]
        keys = sorted(blanks, key=lambda k: tuple(map(int, k.split(","))))
        assert grade_table(q, json.dumps({k: str(blanks[k]) for k in keys})).correct
        assert grade_table(q, ", ".join(str(blanks[k]) for k in keys)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_is_rejected_by_the_real_grader_and_tagged(difficulty):
    miscon = {m.id for m in SKILL.misconceptions}
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert 2 <= len(q.distractors) <= 3, (difficulty, seed)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade_table(q, d).correct, (difficulty, seed, d)
            assert q.option_notes[d] in miscon


@pytest.mark.parametrize("difficulty", TIERS)
def test_one_wrong_blank_or_a_missing_blank_is_rejected(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        blanks = q.answer["blanks"]
        for key in blanks:
            bad = {k: str(v) for k, v in blanks.items()}
            bad[key] = str(blanks[key] + 1)
            assert not grade_table(q, json.dumps(bad)).correct
            partial = {k: str(v) for k, v in blanks.items() if k != key}
            assert not grade_table(q, json.dumps(partial)).correct
        assert not grade_table(q, "").correct
        assert not grade_table(q, "not a table").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_masked_prompt_never_carries_a_blank_value(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        prompt = table_prompt(q.answer)
        for key in q.answer["blanks"]:
            r, c = map(int, key.split(","))
            assert prompt["rows"][r][c] is None
            assert q.answer["rows"][r][c] is None
        # medium and hard always keep a given number; an easy list question may blank every count
        numeric = [cell for row in q.answer["rows"] for cell in row[1:] if isinstance(cell, int)]
        assert numeric or difficulty == "easy", (difficulty, seed)


@pytest.mark.parametrize("difficulty", TIERS)
def test_blank_positions_vary(difficulty):
    patterns = Counter(frozenset(make(TOPIC, seed=s, difficulty=difficulty).answer["blanks"]) for s in range(200))
    assert len(patterns) >= 8
    assert max(patterns.values()) <= 0.4 * 200


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_pool_is_varied(difficulty):
    pool = Counter(_make(difficulty, s).answer_text for s in range(200))
    assert len(pool) >= 100
    assert max(pool.values()) <= 0.2 * 200


@pytest.mark.parametrize("difficulty", TIERS)
def test_text_is_one_line_clean_and_states_the_table(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert "\n" not in q.text and "  " not in q.text, (difficulty, seed, q.text)
        assert "Table (" in q.text and "?" in q.text
        assert not re.search(r"\b(\w+)\s+\1\b", q.text, re.I), (difficulty, seed, q.text)


def test_easy_has_both_list_and_tally_variants():
    kinds = Counter("Tally" in make(TOPIC, seed=s, difficulty="easy").answer["columns"] for s in range(60))
    assert kinds[True] >= 15 and kinds[False] >= 15


def test_medium_comparison_column_is_top_minus_count():
    for seed in range(40):
        q = _make("medium", seed)
        rows = q.answer["rows"]
        values = [q.answer["blanks"].get(f"{r},1", row[1]) for r, row in enumerate(rows)]
        top = max(values)
        assert values.count(top) == 1
        for r, row in enumerate(rows):
            assert q.answer["blanks"].get(f"{r},2", row[2]) == top - values[r]


def test_hard_total_and_derived_rows_are_always_blank_for_derived():
    for seed in range(60):
        q = _make("hard", seed)
        k = len(q.answer["rows"]) - 2
        assert f"{k + 1},1" in q.answer["blanks"]
        total = q.answer["blanks"].get(f"{k},1", q.answer["rows"][k][1])
        freqs = [q.answer["blanks"].get(f"{r},1", q.answer["rows"][r][1]) for r in range(k)]
        assert sum(freqs) == total


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.f10_collect_sort_categorical(__import__("random").Random(0), tier="exam")
