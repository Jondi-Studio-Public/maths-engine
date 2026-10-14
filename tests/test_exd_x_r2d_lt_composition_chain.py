"""Tests the four-part exam chain on a composite G o F with an unknown dilation factor (k, determinant, image area, a second image)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_lt_composition_chain"
SKILL_ID = "senior-define-and-use-composition-of-linear-transformations-and"
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
MARKS = [3, 2, 2, 2]
FT_FLAGS = [False, True, True, True]


def _txt(x: Fraction) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _matrix(q):
    """The matrix of F, rebuilt from the stem with exact fractions."""
    rot = re.search(r"cos θ = (\d+)/(\d+) and sin θ = (-?)(\d+)/(\d+)", q.stem)
    if rot:
        c, h, neg, s, h2 = rot.groups()
        cos, sin = Fraction(int(c), int(h)), Fraction(int(s), int(h2)) * (-1 if neg else 1)
        return [[cos, -sin], [sin, cos]], 1
    line = re.search(r"y = (-?)(\d*)x(?:/(\d+))?", q.stem)
    neg, num, den = line.groups()
    g = Fraction(int(num or 1) * (-1 if neg else 1), int(den or 1))
    # reflection matrix from the unit direction (1, g): 2 u u^T / |u|^2 - I
    n = 1 + g * g
    return [[2 / n - 1, 2 * g / n], [2 * g / n, 2 * g * g / n - 1]], -1


def _apply(mat, x, y):
    return mat[0][0] * x + mat[0][1] * y, mat[1][0] * x + mat[1][1] * y


def _facts(q):
    px, py = (int(v) for v in re.search(r"P\((-?\d+), (-?\d+)\)", q.stem).groups())
    p1x, p1y = (int(v) for v in re.search(r"P'\((-?\d+), (-?\d+)\)", q.stem).groups())
    qx, qy = (int(v) for v in re.search(r"Q\((-?\d+), (-?\d+)\)", q.parts[3].prompt).groups())
    m = int(re.search(r"\(kx, (-?\d+)y\)", q.stem).group(1))
    area = int(re.search(r"area (\d+) square", q.parts[2].prompt).group(1))
    return px, py, p1x, p1y, qx, qy, m, area


def oracle(q):
    mat, det_f = _matrix(q)
    px, py, p1x, p1y, qx, qy, m, area = _facts(q)
    fp = _apply(mat, px, py)
    k = Fraction(p1x) / fp[0]
    assert m * fp[1] == p1y
    det = k * m * det_f
    fq = _apply(mat, qx, qy)
    return {"a": _txt(k), "b": _txt(det), "c": _txt(area * abs(det)), "d": _txt(k * fq[0])}


def test_follow_through_from_a_wrong_k(questions):
    for q in questions[:100]:
        mat, det_f = _matrix(q)
        px, py, p1x, p1y, qx, qy, m, area = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        det = wrong * m * det_f
        fq = _apply(mat, qx, qy)
        reply = {"a": _txt(wrong), "b": _txt(det), "c": _txt(area * abs(det)), "d": _txt(wrong * fq[0])}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            value = Fraction(reply[label]) if "/" not in reply[label] else Fraction(reply[label])
            if value == wrong or reply[label] == q.parts["abcd".index(label)].answer_text:
                continue
            assert result.parts[label].marks == 2, (q.text, label, reply)
            assert result.parts[label].follow_through is True


def test_the_area_factor_is_the_absolute_determinant(questions):
    negatives = [q for q in questions if q.parts[1].answer_text.startswith("-")]
    assert len(negatives) >= 60
    for q in negatives[:40]:
        assert not q.parts[2].answer_text.startswith("-")
        assert q.parts[2].distractors  # the signed product is among the plausible wrong answers


def test_both_kinds_of_first_map_occur(questions):
    kinds = {"rotation" if "rotation" in q.stem else "reflection" for q in questions}
    assert kinds == {"rotation", "reflection"}


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable(
        "exd_x_r2d_lt_composition_chain", "senior-define-and-use-composition-of-linear-transformations-and"
    )
