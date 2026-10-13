"""Checks for the multi-part exam source of `senior-estimate-lengths`: pace a length, round up the units, cost them.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with exact fractions over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong length used correctly in the later parts still earns those marks.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc4_est_length"
SKILL = next(s for s in load_skills()[1] if s.id == "senior-estimate-lengths")
SEEDS = range(300)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def _facts(q):
    pace = int(re.search(r"about (\d+) cm", q.stem).group(1))
    paces = int(re.search(r"takes (\d+) paces", q.stem).group(1))
    unit = Fraction(re.search(r"(\d+(?:\.\d+)?) m (?:long|lengths)", q.stem).group(1))
    price = int(re.search(r"costs \$(\d+)", q.parts[2].prompt).group(1))
    return pace, paces, unit, price


def oracle(q):
    pace, paces, unit, price = _facts(q)
    length = Fraction(pace * paces, 100)
    units = math.ceil(length / unit)
    return [f"{float(length):.1f}", str(units), str(units * price)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == [], seed


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("estimate_length_reference", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_shape_and_wiring():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    for seed in range(40):
        q = _q(seed)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_deterministic():
    for seed in (0, 1, 7, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_the_count_always_needs_rounding_up(questions):
    for q in questions:
        pace, paces, unit, _ = _facts(q)
        units = Fraction(pace * paces, 100) / unit
        assert units.denominator != 1 and Fraction(1, 10) <= units % 1 <= Fraction(9, 10)


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
        if seed >= 150:
            break
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


def test_follow_through_from_a_wrong_length(questions):
    checked = 0
    for q in questions[:150]:
        _, _, unit, price = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        units = math.ceil(wrong / unit)
        reply = {"a": q.parts[0].distractors[0], "b": str(units), "c": str(units * price)}
        result = grade(SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, reply)
        assert result.parts["b"].follow_through is True
        checked += 1
    assert checked == 150


def test_all_wrong_earns_nothing(questions):
    for q in questions[:40]:
        result = grade(SKILL, q, {"a": "-9999", "b": "-5", "c": "-7"})
        assert result.marks == 0 and not result.correct
