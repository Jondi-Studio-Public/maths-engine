"""Checks for the multi-part exam source of `input-output-tables`: a rule inferred from a gapped table, then used.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with exact fractions over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong earlier answer used correctly in later parts still earns those marks.
"""

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc4_table_rule"
SKILL = next(s for s in load_skills()[1] if s.id == "input-output-tables")
SEEDS = range(300)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def _pairs(q):
    return [(int(x), int(y)) for x, y in re.findall(r"= (-?\d+) gives \w = (-?\d+)", q.stem)]


def oracle(q):
    pairs = _pairs(q)
    (x1, y1), (x2, y2) = pairs[0], pairs[1]
    rate = Fraction(y2 - y1, x2 - x1)
    start = y1 - rate * x1
    for x, y in pairs:  # every listed pair lies on the line
        assert y == rate * x + start
    x0 = int(re.search(r"input is (\d+)\.", q.parts[2].prompt).group(1))
    target = int(re.search(r"output of (-?\d+)", q.parts[3].prompt).group(1))
    return [str(rate), str(start), str(rate * x0 + start), str((target - start) / rate)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == [], seed


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("function_evaluation", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_shape_and_wiring():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    for seed in range(40):
        q = _q(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]


def test_deterministic():
    for seed in (0, 1, 7, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_keys_match_independent_derivation(questions):
    bad = [
        (s, q.text, oracle(q), _canonical(q))
        for s, q in zip(SEEDS, questions)
        if oracle(q) != list(_canonical(q).values())
    ]
    assert bad == [], bad[:2]


def test_grader_accepts_key_and_rejects_every_distractor(questions):
    bad = []
    for seed, q in zip(SEEDS, questions):
        canon = _canonical(q)
        result = grade(SKILL, q, canon)
        if not (result.correct and result.marks == result.max_marks):
            bad.append(("key rejected", seed, canon))
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                if grade(SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", seed, p.label, d))
    assert bad == [], bad[:3]


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            assert [d for d in p.distractors if p.option_notes.get(d)], (q.text, p.label)


def test_follow_through_from_a_wrong_rate(questions):
    checked = 0
    for q in questions[:150]:
        pairs = _pairs(q)
        x0 = int(re.search(r"input is (\d+)\.", q.parts[2].prompt).group(1))
        target = int(re.search(r"output of (-?\d+)", q.parts[3].prompt).group(1))
        wrong_rate = Fraction(q.parts[0].distractors[0])
        x1, y1 = pairs[0]
        start = y1 - wrong_rate * x1
        if 0 in (start, wrong_rate * x0 + start) or start == wrong_rate:
            # the grader withholds follow-through from a zero reply, or one that just repeats the earlier answer
            continue
        reply = {
            "a": q.parts[0].distractors[0],
            "b": str(float(start)).rstrip("0").rstrip("."),
            "c": str(float(wrong_rate * x0 + start)).rstrip("0").rstrip("."),
            "d": str(float((target - start) / wrong_rate)).rstrip("0").rstrip("."),
        }
        result = grade(SKILL, q, reply)
        assert result.parts["a"].marks == 0
        # (b) and (c) are exact in the learner's decimals; (d) can need more places than a typed reply gives.
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        checked += 1
    assert checked >= 100


def test_all_wrong_earns_nothing(questions):
    for q in questions[:40]:
        result = grade(SKILL, q, {"a": "9999", "b": "-8888", "c": "7777", "d": "6666"})
        assert result.marks == 0 and not result.correct
