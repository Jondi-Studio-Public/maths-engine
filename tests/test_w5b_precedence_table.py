"""Tests `critical-path-from-a-precedence-table` (topic `w5b_precedence_table`) and its exam chain.

The oracle reads the table back out of the question TEXT. The critical time is recomputed by memoised recursion over
predecessors (no forward/backward pass, no route list); the dummy count comes from the solver, which
`test_w5b_dummy_tables.py` certifies against hand-worked cases.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade
from mathgen.generators._w5b_dummy_kit import fewest_dummies
from tests import exd_x_r3b_support as S

TOPIC = "w5b_precedence_table"
EXAM = "exd_x_w5b_precedence_table"
SKILL_ID = "critical-path-from-a-precedence-table"
_SKILL = S.skill(SKILL_ID)
_ROW = re.compile(r"Activity ([A-Z]): duration (\d+) days, (?:no predecessors|predecessors: ([A-Z, ]+))\.")


def _table(text):
    dur, preds = {}, {}
    for name, d, p in _ROW.findall(text):
        dur[name] = int(d)
        preds[name] = [s.strip() for s in p.split(",")] if p else []
    return dur, preds


def _completion(dur, preds):
    memo = {}

    def finish(a):
        if a not in memo:
            memo[a] = dur[a] + max((finish(p) for p in preds[a]), default=0)
        return memo[a]

    return max(finish(a) for a in dur)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _kind(q):
    text = q.text
    if "dummy" in text and (
        "How many dummy" in text or "smallest number of dummy" in text or "how many dummies" in text
    ):
        return "dummies"
    if "arrows" in text:
        return "arrows"
    return "time"


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    S.check_wiring(EXAM, SKILL_ID)


def test_easy_counts_dummies_of_five_or_six_activities():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        dur, preds = _table(q.text)
        assert _kind(q) == "dummies", q.text
        assert len(preds) in (5, 6)
        assert int(q.answer_text) == fewest_dummies(preds), q.text
        seen[int(q.answer_text)] += 1
    assert set(seen) >= {0, 1, 2}, seen


def test_medium_is_the_critical_time_of_a_table_that_needs_a_dummy():
    for seed in range(120):
        q = _q("medium", seed)
        dur, preds = _table(q.text)
        assert _kind(q) == "time"
        assert len(preds) in (6, 7)
        assert fewest_dummies(preds) >= 1, q.text
        assert int(q.answer_text) == _completion(dur, preds), q.text


def test_hard_has_three_shapes_and_each_key_is_right():
    kinds = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        dur, preds = _table(q.text)
        kind = _kind(q)
        kinds[kind] += 1
        if kind == "dummies":
            assert len(preds) in (7, 8)
            assert int(q.answer_text) == fewest_dummies(preds) >= 2, q.text
        elif kind == "arrows":
            assert int(q.answer_text) == len(preds) + fewest_dummies(preds), q.text
        else:
            assert len(preds) in (7, 8)
            assert int(q.answer_text) == _completion(dur, preds), q.text
    assert set(kinds) == {"dummies", "arrows", "time"}, kinds


def test_hard_time_has_a_rival_within_two_days():
    checked = 0
    for seed in range(300):
        q = _q("hard", seed)
        if _kind(q) != "time":
            continue
        dur, preds = _table(q.text)
        finish = _completion(dur, preds)
        assert 1 <= finish - _rival(dur, preds, finish) <= 2, q.text
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_every_distractor_is_wrong_and_the_key_is_right(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (tier, seed, d)
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_distractors_are_tagged_integers(tier):
    from mathgen.misconceptions import CATALOGUE

    for seed in range(60):
        q = _q(tier, seed)
        for d in q.distractors:
            assert re.fullmatch(r"\d+", d)
            assert q.option_notes.get(d) in CATALOGUE, (tier, seed, d)


def test_tiers_differ_in_shape():
    shapes = {t: {_kind(_q(t, s)) for s in range(60)} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] == {"dummies"} and shapes["medium"] == {"time"} and len(shapes["hard"]) == 3


# -- exam chain ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(EXAM, 250)


def test_exam_structure_and_shapes():
    S.check_structure(EXAM, "abcd", [2, 1, 2, 2])
    S.check_shapes(EXAM, SKILL_ID)


def test_exam_keys_match_the_oracle(questions):
    for q in questions:
        dur, preds = _table(q.stem)
        dummies = fewest_dummies(preds)
        assert 7 <= len(preds) <= 8 and 2 <= dummies <= 4, q.stem
        assert int(q.parts[0].answer_text) == dummies, q.stem
        assert int(q.parts[1].answer_text) == len(preds) + dummies, q.stem
        finish = _completion(dur, preds)
        assert int(q.parts[2].answer_text) == finish, q.stem
        y, cut = re.search(r"Activity ([A-Z]) is shortened by (\d+) days", q.parts[3].prompt).groups()
        new_finish = _completion({**dur, y: dur[y] - int(cut)}, preds)
        assert int(q.parts[3].answer_text) == new_finish, q.stem
        assert 0 < finish - new_finish < int(cut)


def test_exam_grading_tags_and_space():
    S.check_grading(EXAM, SKILL_ID, 120)
    S.check_tags(EXAM)
    S.check_answer_space(EXAM)
    S.check_clean_and_deterministic(EXAM)


def test_follow_through_from_a_wrong_dummy_count_and_wrong_time(questions):
    checked = 0
    for q in questions[:120]:
        dur, preds = _table(q.stem)
        wrong_a = int(q.parts[0].answer_text) + 1
        saving = int(q.parts[2].answer_text) - int(q.parts[3].answer_text)
        wrong_c = int(q.parts[2].answer_text) + 3
        reply = {
            "a": str(wrong_a),
            "b": str(wrong_a + len(preds)),
            "c": str(wrong_c),
            "d": str(wrong_c - saving),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (q.stem, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked >= 100


def _rival(dur, preds, finish):
    """The length of the longest route other than the critical one, by depth-first search over successors."""
    succ = {a: [b for b in preds if a in preds[b]] for a in preds}
    lengths = []

    def walk(a, total):
        total += dur[a]
        if not succ[a]:
            lengths.append(total)
        for b in succ[a]:
            walk(b, total)

    for a in preds:
        if not preds[a]:
            walk(a, 0)
    assert max(lengths) == finish
    lengths.remove(finish)
    return max(lengths)
