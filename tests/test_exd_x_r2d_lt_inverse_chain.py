"""Tests the four-part exam chain on the inverse of a linear map (determinant, the original point, the original area)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_lt_inverse_chain"
SKILL_ID = "senior-define-and-use-inverses-of-linear-transformations-and"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard(seed):
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]  # the hardest practice tier
    return make(src.topic, seed=seed, **src.params)


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].params == {"tier": "exam"} and list(exam[0].stages) == ["exam"]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_FLAGS
        assert all(p.distractors and all(d in p.option_notes for d in p.distractors) for p in q.parts)


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees_with_the_rendered_prompt(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_full_marks_for_the_key(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_every_distractor_is_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_all_wrong_earns_nothing(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: f"9999{i}" for i, p in enumerate(q.parts)})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(tuple(_canonical(q).values()) for q in questions)
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / len(questions) <= 0.05, tuples.most_common(3)


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()


LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 1, 2]
FT_FLAGS = [False, True, True, True]


def _txt(x: Fraction) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _coeffs(expr: str):
    out = {"x": 0, "y": 0}
    for sign, digits, var in re.findall(r"([+-]?)\s*(\d*)([xy])", expr.replace(" ", "")):
        out[var] = (-1 if sign == "-" else 1) * (int(digits) if digits else 1)
    return out["x"], out["y"]


def _facts(q):
    row1, row2 = re.search(r"\(x, y\) (?:→|to) \(([^,]+), ([^)]+)\)", q.stem).groups()
    a, b = _coeffs(row1)
    c, d = _coeffs(row2)
    qx, qy = (int(v) for v in re.search(r"Q\((-?\d+), (-?\d+)\)", q.stem).groups())
    area = int(re.search(r"has area (\d+) square", q.parts[3].prompt).group(1))
    return a, b, c, d, qx, qy, area


def oracle(q):
    a, b, c, d, qx, qy, area = _facts(q)
    det = a * d - b * c
    # solve T(P) = Q as two linear equations by elimination, independent of the adjugate formula
    px = Fraction(d * qx - b * qy, det)
    py = Fraction(a * qy - c * qx, det)
    assert (a * px + b * py, c * px + d * py) == (qx, qy)
    return {"a": _txt(det), "b": _txt(px), "c": _txt(py), "d": _txt(Fraction(area, abs(det)))}


def test_follow_through_from_a_wrong_determinant(questions):
    for q in questions[:100]:
        a, b, c, d, qx, qy, area = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        reply = {
            "a": _txt(wrong),
            "b": _txt(Fraction(d * qx - b * qy) / wrong),
            "c": _txt(Fraction(a * qy - c * qx) / wrong),
            "d": _txt(Fraction(area) / abs(wrong)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            if reply[label] == reply["a"] or reply[label] == q.parts["abcd".index(label)].answer_text:
                continue
            assert result.parts[label].marks == MARKS["abcd".index(label)], (q.text, label, reply)
            assert result.parts[label].follow_through is True


def test_the_determinant_is_never_a_unit(questions):
    assert all(abs(int(q.parts[0].answer_text)) >= 2 for q in questions)


def test_both_signs_of_determinant_occur(questions):
    signs = {q.parts[0].answer_text.startswith("-") for q in questions}
    assert signs == {True, False}


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable(
        "exd_x_r2d_lt_inverse_chain", "senior-define-and-use-inverses-of-linear-transformations-and"
    )
