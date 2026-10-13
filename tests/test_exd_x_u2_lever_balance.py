"""Exam tier for moment-of-a-force: net moment of two forces on a pivoted beam, then the balancing distance.

(a) several prompt shapes, some absent from the practice tier; (b) each part's key re-derived from the RENDERED
PROMPT over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor; (d) a wrong net
moment used correctly in part (b) earns that part's marks.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_lever_balance"
SKILL_ID = "moment-of-a-force"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round(value: Fraction, places: int) -> str:
    d = Decimal(value.numerator) / Decimal(value.denominator)
    return str(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _printed(q):
    forces = [Fraction(v) for v in re.findall(r"force of (\d+) N", q.text)]
    dist = {side: Fraction(v) for v, side in re.findall(r"at ([\d.]+) m to the (left|right) of P", q.text)}
    sin = Fraction(1, 2) if "30°" in q.text else Fraction(1)
    return forces, dist, sin


def _oracle(q) -> dict[str, str]:
    (f1, f2, f3), dist, sin = _printed(q)
    net = f2 * dist["right"] * sin - f1 * dist["left"]  # clockwise positive
    return {"a": _round(net, 1), "b": _round(abs(net) / f3, 2)}


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("engineering", seed=s, kinds=["moment"])) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b"]
        assert [p.marks for p in q.parts] == [2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True]


def test_both_angle_variants_and_both_net_signs_occur(questions):
    assert any("30°" in q.text for q in questions) and any("30°" not in q.text for q in questions)
    nets = [Fraction(_oracle(q)["a"]) for q in questions]
    assert any(n < 0 for n in nets) and any(n > 0 for n in nets)


def test_deterministic():
    assert _q(5).to_dict() == _q(5).to_dict()


def test_key_matches_independent_derivation_over_300_seeds(questions):
    bad = [(q.text, q.answer_text) for q in questions if {p.label: p.answer_text for p in q.parts} != _oracle(q)]
    assert bad == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


def test_key_accepted_and_every_distractor_rejected(questions):
    bad = []
    for q in questions[:150]:
        full = grade(_SKILL, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks == 4):
            bad.append(("key rejected", q.text))
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert all(note in CATALOGUE for note in p.option_notes.values())
            for d in p.distractors:
                if grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", q.text, p.label, d))
    assert bad == []


def test_follow_through_from_a_wrong_net_moment(questions):
    for q in questions[:120]:
        (_, _, f3), _, _ = _printed(q)
        key = Fraction(_oracle(q)["a"])
        for offset in (Fraction(5), Fraction(10), Fraction(15)):
            wrong = key + offset
            if wrong != 0 and abs(wrong) / f3 != abs(key) / f3 and abs(wrong) != abs(key):
                break
        reply = {"a": _round(wrong, 1), "b": _round(abs(wrong) / f3, 2)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, reply)
        assert result.marks == 2 < result.max_marks and not result.correct


def test_all_wrong_earns_zero(questions):
    for q in questions[:30]:
        assert grade(_SKILL, q, {"a": "9999.9", "b": "77.77"}).marks == 0
