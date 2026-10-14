"""Tests for the multi-part exam source of `integer-arithmetic`.

The oracle reads the signed start and the two repeated changes from the rendered stem (each change is a product
of two counts, so the order they appear in does not matter) and recomputes the three answers.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_integer_changes"
SKILL_ID = "integer-arithmetic"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _read(q):
    text = re.sub(r"\d+ pm", "", q.stem).replace("-$", "-").replace("$", "")
    s, a1, a2, b1, b2 = (int(v) for v in re.findall(r"-?\d+", text))
    return s, a1 * a2, b1 * b2


def oracle(q) -> dict[str, str]:
    s, fall, rise = _read(q)
    after = s - fall
    end = after + rise
    return {"a": str(after), "b": str(end), "c": str(end - s)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_prompt_shapes_new_against_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(_HARD.topic, seed=s, difficulty="hard", **dict(_HARD.params or {}))) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_oracle_agrees_over_300_seeds(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_negative_starts_occur(questions):
    assert any(_read(q)[0] < 0 for q in questions) and any(_read(q)[0] > 0 for q in questions)


def test_key_accepted_and_every_distractor_rejected(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)


def test_tagged_distractors_carry_values(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)


def test_follow_through_from_a_wrong_first_value(questions):
    checked = 0
    for q in questions[:150]:
        s, _, rise = _read(q)
        for d in q.parts[0].distractors:
            wrong = int(d)
            end, change = wrong + rise, wrong + rise - s
            if (
                0 in (wrong, end, change)
                or len({wrong, end, change, int(q.parts[1].answer_text), int(q.parts[2].answer_text)}) < 5
            ):
                continue
            result = grade(_SKILL, q, {"a": d, "b": str(end), "c": str(change)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, d)
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, d)
            checked += 1
            break
    assert checked >= 100
