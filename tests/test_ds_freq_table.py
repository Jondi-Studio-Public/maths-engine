"""Proofs for `f10-frequency-table-discrete-data` (Year 3, AC9M3ST01, `answer: table`).

Generator-side checks (every distractor is a wrong table the real grader rejects, blanks vary, tiers escalate) and an
independent oracle that reads only `question.text`: it recounts the supplied data, rebuilds each table and compares
with `answer_text`.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table, table_prompt
from mathgen.skills import load_skills
import mathgen.generators.ds_freq_table_build as gen

SKILL_ID = "f10-frequency-table-discrete-data"
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
    assert tuple(SKILL.year_level) == ("3",)
    assert [s.difficulty for s in SKILL.sources] == TIERS
    assert all(s.topic.startswith("ds_freq_table_") for s in SKILL.sources)
    assert "AC9M3ST01" in SKILL.tags


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism(difficulty):
    for seed in range(15):
        assert _make(difficulty, seed).to_json() == _make(difficulty, seed).to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_json_and_bare_forms(difficulty):
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


@pytest.mark.parametrize("difficulty", TIERS)
def test_masked_prompt_never_carries_a_blank_value(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        prompt = table_prompt(q.answer)
        for key in q.answer["blanks"]:
            r, c = map(int, key.split(","))
            assert prompt["rows"][r][c] is None
            assert q.answer["rows"][r][c] is None
        assert any(isinstance(cell, int) for row in q.answer["rows"] for cell in row[1:]) or difficulty == "easy"


@pytest.mark.parametrize("difficulty", TIERS)
def test_blank_positions_and_answers_vary(difficulty):
    patterns = Counter(frozenset(_make(difficulty, s).answer["blanks"]) for s in range(200))
    assert len(patterns) >= 8 and max(patterns.values()) <= 0.4 * 200
    pool = Counter(_make(difficulty, s).answer_text for s in range(200))
    assert len(pool) >= 100 and max(pool.values()) <= 0.2 * 200


@pytest.mark.parametrize("difficulty", TIERS)
def test_text_is_one_line_clean(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert "\n" not in q.text and "  " not in q.text
        assert "Table (" in q.text and "?" in q.text
        assert not re.search(r"\b(\w+)\s+\1\b", q.text, re.I), q.text


def test_unknown_tier_is_an_error():
    import random

    for fn in (gen.ds_freq_table_easy, gen.ds_freq_table_derived, gen.ds_freq_table_cumulative):
        with pytest.raises(ValueError):
            fn(random.Random(0), tier="exam")


# ---------------------------------------------------------------------------------------------------------------
# Oracle: from the text alone
# ---------------------------------------------------------------------------------------------------------------

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")


def _table(text):
    m = _TABLE_RE.search(text)
    assert m, text
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return m.group("head").split(" — "), rows


def _tally(s):
    return 0 if s == "none" else s.count("|") - 4 * s.count("/") + 5 * s.count("/")


def _observations(text, head, rows):
    """The observations, from the tally column, the named facts or the bare list."""
    if "Tally" in head:
        out = []
        for r in rows[:-1] if rows[-1][0] != "Total" else rows:
            pass
        return None
    body = text.split("Fill in each ?")[0]
    nums = re.findall(r"(?:size |has |borrowed |ate |: )(\d+)(?= |;|\.|,)", body)
    listed = re.search(r"(?:were|recorded): ((?:\d+, )*\d+)\.", body)
    if listed:
        return [int(x) for x in listed.group(1).split(", ")]
    return [int(x) for x in nums]


def _cell(r, c, fills):
    return fills.get((r[0], c))


def _solve(text, tier):
    head, rows = _table(text)
    cc = head.index("Count")
    values = [r for r in rows if re.fullmatch(r"\d+", r[0])]
    k = len(values)
    vs = [int(r[0]) for r in values]
    if "Tally" in head:
        fs = [_tally(r[1]) for r in values]
    elif "counted in all" in text and "Running total" in head and "table below is only part" in text:
        n = int(re.search(r"(\d+) \w+ were counted in all", text).group(1))
        known = sum(int(r[cc]) for r in values if r[cc] != "?")
        fs = [int(r[cc]) if r[cc] != "?" else n - known for r in values]
    else:
        obs = _observations(text, head, rows)
        c = Counter(obs)
        fs = [c[v] for v in vs]
    out = {}
    for i, r in enumerate(values):
        if r[cc] == "?":
            out[(r[0], "Count")] = fs[i]
        if len(r) > cc + 1 and r[cc + 1] == "?":
            h = head[cc + 1]
            out[(r[0], h)] = sum(fs[: i + 1]) if h == "Running total" else sum(fs[i:])
    for r in rows:
        if r[0] == "Total" and r[cc] == "?":
            out[("Total", "Count")] = sum(fs)
        m = re.fullmatch(r"(At least|Fewer than|More than) (\d+)|(\d+) or fewer", r[0])
        if m and r[cc] == "?":
            if m.group(1):
                t = int(m.group(2))
                keep = {"At least": lambda v: v >= t, "Fewer than": lambda v: v < t, "More than": lambda v: v > t}[
                    m.group(1)
                ]
            else:
                t = int(m.group(3))
                keep = lambda v: v <= t  # noqa: E731
            out[(r[0], "Count")] = sum(f for v, f in zip(vs, fs) if keep(v))
    return out


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_matches_answer(difficulty):
    for seed in range(120):
        q = _make(difficulty, seed)
        want = _solve(q.text, difficulty)
        expected = ", ".join(f"{a} {b} = {v}" for (a, b), v in want.items())
        assert len(want) == len(q.answer["blanks"]), (difficulty, seed, q.text)
        assert sorted(expected.split(", ")) == sorted(q.answer_text.split(", ")), (difficulty, seed, q.text, expected)


def test_hard_covers_all_three_shapes_and_medium_all_forms():
    shapes = Counter()
    for s in range(120):
        head = _table(_make("hard", s).text)[0]
        shapes[head[-1] if head[-1] != "Count" else "x"] += 1
        if "Running total" in head and "part finished" in _make("hard", s).text:
            shapes["partial"] += 1
    assert shapes["This many or more"] >= 15 and shapes["Running total"] >= 15 and shapes["partial"] >= 15
    forms = Counter(_table(_make("medium", s).text)[1][-1][0].split()[0] for s in range(120))
    assert len(forms) >= 4


def test_medium_and_hard_values_include_a_zero_count_sometimes():
    for tier in ("easy", "medium", "hard"):
        zeros = sum(
            any(
                (row[-1] == 0 or row[1] == 0) and re.fullmatch(r"\d+", row[0])
                for row in _make(tier, s).answer["rows"]
                if isinstance(row[1], int)
            )
            for s in range(80)
        )
        zeros += sum(
            1
            for s in range(80)
            for key, v in _make(tier, s).answer["blanks"].items()
            if v == 0 and int(key.split(",")[0]) < len(_make(tier, s).answer["rows"]) - 1
        )
        assert zeros >= 5, tier
