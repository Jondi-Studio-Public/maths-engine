"""Contract tests for f10-grid-reference-pathway (topic f10_grid_reference_pathway, answer kind pair)."""

import re

import pytest

from mathgen import make
from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills

TOPIC = "f10_grid_reference_pathway"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)
LETTERS = "ABCDEFGHIJ"
MOVE_RE = re.compile(r"(\d+) squares? (right|left|up|down|north|south|east|west)")
START_RE = re.compile(r"starts in square ([A-J])(\d+)")
FROM_TO_RE = re.compile(r"from square ([A-J])(\d+) to square ([A-J])(\d+)")
DIRS = {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1), "east": (1, 0), "west": (-1, 0)}


def _solve(q, difficulty):
    """Re-derive the answer from the question text alone."""
    top_down = "numbered from the top" in q.text
    m = START_RE.search(q.text)
    col, row = LETTERS.index(m.group(1)) + 1, int(m.group(2))
    for count, word in MOVE_RE.findall(q.text):
        n = int(count)
        if word in ("right", "east"):
            col += n
        elif word in ("left", "west"):
            col -= n
        elif word in ("up", "north"):
            row += -n if top_down else n
        else:
            row += n if top_down else -n
    return col, row


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_and_distractors_clear_the_real_grader(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.to_json() == make(TOPIC, seed=seed, difficulty=difficulty).to_json()
        assert isinstance(q.answer, dict) and len(q.answer) == 2
        assert grade_pair(q, q.answer_text).correct
        # a labelled pair may be typed in either order
        reordered = ", ".join(f"{k} = {v}" for k, v in reversed(list(q.answer.items())))
        assert grade_pair(q, reordered).correct
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grade_pair(q, d).correct, (q.text, d)
        assert set(q.option_notes) == set(q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_matches_an_independent_reading_of_the_text(difficulty):
    forward = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        if FROM_TO_RE.search(q.text):
            a, r0, b, r1 = FROM_TO_RE.search(q.text).groups()
            across = abs(LETTERS.index(b) - LETTERS.index(a))
            vertical = abs(int(r1) - int(r0))
            assert sorted(q.answer.values()) == sorted([across, vertical]), q.text
            continue
        forward += 1
        col, row = _solve(q, difficulty)
        assert (q.answer["column"], q.answer["row"]) == (col, row), q.text
    assert forward > 0


def test_every_step_stays_on_the_grid():
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if FROM_TO_RE.search(q.text):
                continue
            top_down = "numbered from the top" in q.text
            cols = LETTERS.index(re.search(r"columns A to ([A-J])", q.text).group(1)) + 1
            rows = int(re.search(r"row (\d+) is the (?:bottom|top) row\.", q.text).group(1))
            m = START_RE.search(q.text)
            col, row = LETTERS.index(m.group(1)) + 1, int(m.group(2))
            for count, word in MOVE_RE.findall(q.text):
                n = int(count)
                if word in ("right", "east"):
                    col += n
                elif word in ("left", "west"):
                    col -= n
                elif word in ("up", "north"):
                    row += -n if top_down else n
                else:
                    row += n if top_down else -n
                assert 1 <= col <= cols and 1 <= row <= rows, q.text


def test_hard_tier_numbers_rows_from_the_top_and_mixes_both_question_shapes():
    kinds = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "numbered from the top" in q.text
        kinds.add("inverse" if FROM_TO_RE.search(q.text) else "forward")
    assert kinds == {"inverse", "forward"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_pool_is_varied(difficulty):
    from collections import Counter

    pool = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(200))
    assert len(pool) >= 15
    assert max(pool.values()) / 200 <= 0.25
    texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(texts) >= 40


def test_no_singular_plural_slip():
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "1 squares" not in q.text
            assert not any("1 squares" in step for step in q.working)


def test_skill_is_active_pair_and_every_note_is_declared():
    _, skills = load_skills()
    skill = next(x for x in skills if x.id == "f10-grid-reference-pathway")
    assert skill.status == "active" and skill.answer == "pair"
    ids = {m["id"] if isinstance(m, dict) else m.id for m in skill.misconceptions}
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert set(q.option_notes.values()) <= ids
