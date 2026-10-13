"""Exam source of `ucv1-isomorphism-invariant`: order, element order, exponent and index in Z_m x Z_n."""

from __future__ import annotations

import math
import re

SKILL_ID = "ucv1-isomorphism-invariant"
TOPIC = "exd_x_r3i_product_group_orders"
LABELS = ["a", "b", "c", "d"]

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_prompt_variety_and_new_shapes():
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(hard_src.topic, seed=s, **(hard_src.params or {}))) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_answer_space(questions):
    tuples = Counter(tuple(_canonical(q).values()) for q in questions)
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / len(questions) <= 0.05


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()


def oracle(q):
    m, n = (int(v) for v in re.search(r"Z_(\d+) x Z_(\d+)", q.stem).groups())
    p, qq = (int(v) for v in re.search(r"\((\d+), (\d+)\)", q.parts[1].prompt).groups())
    k, x, y = 1, p % m, qq % n
    while (x, y) != (0, 0):
        x, y, k = (x + p) % m, (y + qq) % n, k + 1
    top = max(math.lcm(a, b) for a in range(1, m + 1) for b in range(1, n + 1) if m % a == 0 and n % b == 0)
    return {"a": str(m * n), "b": str(k), "c": str(top), "d": str(m * n // top)}


def test_claim_is_decided_by_the_last_part(questions):
    # A cyclic G would make (c) repeat (a), so every draw refutes the claim: the index in (d) always exceeds 1.
    assert all(int(q.parts[3].answer_text) > 1 for q in questions)
    assert all(q.parts[0].answer_text != q.parts[2].answer_text for q in questions)


def test_follow_through_from_wrong_group_order(questions):
    checked = 0
    for q in questions[:200]:
        top = int(q.parts[2].answer_text)
        wa = top * (int(q.parts[3].answer_text) + 1)  # a wrong |G| that is still a multiple of the key to (c)
        reply = {**_canonical(q), "a": str(wa), "d": str(wa // top)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
        checked += 1
    assert checked > 10
