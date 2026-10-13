"""Exam tier of `f10-rational-percentage-estimation`: round a price, estimate a non-benchmark discount, estimate the sale price."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_sale_price_estimate"
SKILL_ID = "f10-rational-percentage-estimation"
HARD = ("rational_percentage_estimation", {"tier": "hard"})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    price = int(re.search(r"priced at \$(\d+)", q.stem).group(1))
    pct = int(re.search(r"with (\d+)% off", q.stem).group(1))
    rounded = (price + 10) // 20 * 20
    discount = rounded * pct // 100
    assert rounded * pct % 100 == 0
    return {"a": str(rounded), "b": str(discount), "c": str(rounded - discount)}


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in range(300)]


def _canon(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC] and exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, **HARD[1])) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_the_percentage_is_never_a_benchmark(questions):
    for q in questions:
        assert int(re.search(r"with (\d+)% off", q.stem).group(1)) not in (10, 25, 50)


def test_key_matches_independent_derivation(questions):
    bad = [(q.seed, oracle(q), _canon(q)) for q in questions if oracle(q) != _canon(q)]
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 4
        for p in q.parts:
            assert len(p.distractors) >= 2
            for d in p.distractors:
                assert grade(_SKILL, q, {**_canon(q), p.label: d}).parts[p.label].marks == 0, (q.seed, p.label, d)


def test_follow_through_from_wrong_rounding(questions):
    for q in questions[:100]:
        pct = int(re.search(r"with (\d+)% off", q.stem).group(1))
        wrong = int(q.parts[0].distractors[0])
        if wrong * pct % 100:
            continue  # a wrong rounding that gives a fractional discount is not a clean chain
        disc = wrong * pct // 100
        if wrong - disc in (wrong, disc):
            continue
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(disc), "c": str(wrong - disc)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["c"].marks == 1


def test_follow_through_from_wrong_discount(questions):
    for q in questions[:100]:
        a = int(q.parts[0].answer_text)
        wrong = int(q.parts[1].distractors[0])
        if a - wrong in (a, wrong):  # a reply that only repeats an earlier value earns nothing
            continue
        result = grade(_SKILL, q, {"a": str(a), "b": str(wrong), "c": str(a - wrong)})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1


def test_answer_space(questions):
    counts = Counter(q.answer_text for q in questions)
    assert len(counts) >= 100 and counts.most_common(1)[0][1] <= 15
