"""Checks for the multi-part exam source of `senior-estimate-the-areas-of-different-shapes`: a paddock paced out.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with exact fractions over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong area used correctly in the later parts still earns those marks.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc4_est_area"
SKILL = next(s for s in load_skills()[1] if s.id == "senior-estimate-the-areas-of-different-shapes")
SEEDS = range(300)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def _facts(q):
    pace = Fraction(re.search(r"One pace is (\d+(?:\.\d+)?) m", q.stem).group(1))
    long_paces, wide_paces = (int(x) for x in re.search(r"(\d+) paces long and (\d+) paces wide", q.stem).groups())
    stock = int(re.search(r"(\d+) sheep per hectare", q.parts[2].prompt).group(1))
    return pace, long_paces, wide_paces, stock


def oracle(q):
    pace, long_paces, wide_paces, stock = _facts(q)
    area = pace * long_paces * pace * wide_paces
    hectares = area / 10000
    return [str(int(area)), f"{float(hectares):.2f}", str(math.floor(hectares * stock))]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == [], seed


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("estimate_area_reference", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_shape_and_wiring():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    for seed in range(40):
        q = _q(seed)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_deterministic():
    for seed in (0, 1, 7, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_the_paddock_is_longer_than_it_is_wide_and_the_sheep_count_needs_rounding(questions):
    for q in questions:
        pace, long_paces, wide_paces, stock = _facts(q)
        assert long_paces > wide_paces
        hectares = pace * long_paces * pace * wide_paces / 10000
        assert Fraction(1, 10) <= (hectares * stock) % 1 <= Fraction(9, 10)


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


def test_follow_through_from_a_wrong_area(questions):
    checked = 0
    for q in questions[:150]:
        _, _, _, stock = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        hectares = wrong / 10000
        shown = Fraction(math.floor(hectares * 100 + Fraction(1, 2)), 100)
        if shown == 0 or hectares * stock % 1 in (0,):
            continue
        reply = {
            "a": q.parts[0].distractors[0],
            "b": f"{float(shown):.2f}",
            "c": str(math.floor(shown * stock)),
        }
        result = grade(SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        checked += 1
    assert checked >= 100


def test_all_wrong_earns_nothing(questions):
    for q in questions[:40]:
        result = grade(SKILL, q, {"a": "-9999", "b": "-5", "c": "-7"})
        assert result.marks == 0 and not result.correct
