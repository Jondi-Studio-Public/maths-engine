"""Exam tier shared by stress-from-force-area, strain-from-extension and youngs-modulus: a tension test in four parts.

(a) the exam source shows several prompt shapes, none of them in the practice tier; (b) every part's key is
re-derived from the RENDERED PROMPT with independent exact arithmetic over 300 seeds; (c) the skill's grader
accepts the key and rejects every distractor; (d) a wrong earlier answer used correctly in a later part earns
that part's marks.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_bar_in_tension"
SKILL_IDS = ("stress-from-force-area", "strain-from-extension", "youngs-modulus")
PRACTICE = {"stress-from-force-area": "stress", "strain-from-extension": "strain", "youngs-modulus": "youngs_modulus"}
SEEDS = range(300)

_SKILLS = {s.id: s for s in load_skills()[1] if s.id in SKILL_IDS}


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round(value: Fraction, places: int) -> str:
    d = Decimal(value.numerator) / Decimal(value.denominator)
    return str(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the printed stem and the printed second-bar details."""
    width, thick, length = (
        int(v) for v in re.search(r"section (\d+) mm by (\d+) mm and an original length of (\d+) mm", q.stem).groups()
    )
    load = Fraction(re.search(r"load of ([\d.]+) kN", q.stem).group(1))
    ext = Fraction(re.search(r"extends by ([\d.]+) mm", q.stem).group(1))
    length2 = int(re.search(r"(\d+) mm long", q.text).group(1))
    load2 = Fraction(re.findall(r"([\d.]+) kN", q.text)[1])
    area = width * thick
    stress = load * 1000 / area
    strain = ext / length  # a pure number
    modulus_gpa = stress / strain / 1000
    ext2 = load2 * 1000 * length2 / (area * modulus_gpa * 1000)
    return {
        "a": _round(stress, 1),
        "b": str(int(strain * 10**6)),
        "c": _round(modulus_gpa, 1),
        "d": _round(ext2, 2),
    }


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


@pytest.mark.parametrize("skill_id", SKILL_IDS)
def test_source_wired(skill_id):
    exam = [s for s in _SKILLS[skill_id].sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


@pytest.mark.parametrize("skill_id", SKILL_IDS)
def test_prompt_variety_and_new_shapes(skill_id):
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("engineering", seed=s, kinds=[PRACTICE[skill_id]])) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_deterministic():
    assert _q(7).to_dict() == _q(7).to_dict()


def test_key_matches_independent_derivation_over_300_seeds(questions):
    bad = [
        (q.text, {p.label: p.answer_text for p in q.parts}, _oracle(q))
        for q in questions
        if {p.label: p.answer_text for p in q.parts} != _oracle(q)
    ]
    assert bad == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


@pytest.mark.parametrize("skill_id", SKILL_IDS)
def test_key_accepted_and_every_distractor_rejected(skill_id, questions):
    skill = _SKILLS[skill_id]
    bad = []
    for q in questions[:120]:
        full = grade(skill, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks == 7):
            bad.append(("key rejected", q.text))
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(skill, q, {**canon, p.label: d})
                if result.parts[p.label].marks != 0:
                    bad.append(("distractor accepted", q.text, p.label, d))
    assert bad == []


def test_tagged_distractors_carry_catalogue_ids(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:60]:
        for p in q.parts:
            assert set(p.option_notes) == set(p.distractors), (q.text, p.label)
            assert all(note in CATALOGUE for note in p.option_notes.values())


def test_follow_through_from_a_wrong_stress(questions):
    skill = _SKILLS["youngs-modulus"]
    for q in questions[:100]:
        key = _oracle(q)
        width, thick = (int(v) for v in re.search(r"section (\d+) mm by (\d+)", q.stem).groups())
        length2 = int(re.search(r"(\d+) mm long", q.text).group(1))
        load2 = Fraction(re.findall(r"([\d.]+) kN", q.text)[1])
        strain = Fraction(key["b"])
        for offset in (10, 20, 30):
            a_wrong = Fraction(key["a"]) + offset
            c_wrong = Fraction(_round(a_wrong * 1000 / strain, 1))
            # The engine withholds follow-through from a reply that merely repeats an earlier value.
            if c_wrong not in (a_wrong, strain):
                break
        d_wrong = _round(load2 * length2 / (width * thick * c_wrong), 2)
        reply = {"a": _round(a_wrong, 1), "b": key["b"], "c": _round(c_wrong, 1), "d": d_wrong}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through
        assert result.marks == 5 < result.max_marks and not result.correct


def test_follow_through_from_a_wrong_modulus(questions):
    skill = _SKILLS["youngs-modulus"]
    for q in questions[:100]:
        key = _oracle(q)
        c_wrong = Fraction(key["c"]) + 20
        width, thick = (int(v) for v in re.search(r"section (\d+) mm by (\d+)", q.stem).groups())
        length2 = int(re.search(r"(\d+) mm long", q.text).group(1))
        load2 = Fraction(re.findall(r"([\d.]+) kN", q.text)[1])
        d_wrong = _round(load2 * length2 / (width * thick * c_wrong), 2)
        result = grade(skill, q, {"a": key["a"], "b": key["b"], "c": _round(c_wrong, 1), "d": d_wrong})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through
        assert result.marks == 5


def test_all_wrong_earns_zero(questions):
    skill = _SKILLS["stress-from-force-area"]
    for q in questions[:30]:
        result = grade(skill, q, {"a": "9999.9", "b": "1", "c": "8888.8", "d": "77.77"})
        assert result.marks == 0
