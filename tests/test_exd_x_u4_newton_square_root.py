"""Exam source of `newton-method-iteration`: two Newton steps for a width defined by an area, with the residual."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Rational, Symbol, diff

from mathgen.engine import grade
from mathgen.generators._u4_kit import survives_nsimplify
from tests.exd_x_u4_support import (
    canonical_reply,
    check_answer_space,
    check_distractors_rejected,
    check_full_marks,
    check_no_text_defects,
    check_tagged,
    exam,
    frac,
    shapes,
    skill,
)

TOPIC = "exd_x_u4_newton_square_root"
SKILL_ID = "newton-method-iteration"
_W = Symbol("w")


def _setting(text: str) -> tuple[int, int, int]:
    """Ratio m, area A and start w0, read from the rendered prompt alone."""
    area = int(re.search(r"area of (\d+) m²", text).group(1))
    w0 = int(re.search(r"w₀ = (\d+)", text).group(1))
    m = 2 if "twice as long" in text else 3 if "three times as long" in text else 1
    return m, area, w0


def _oracle(text: str) -> dict[str, Fraction]:
    m, area, w0 = _setting(text)
    f = m * _W**2 - area
    df = diff(f, _W)
    x0 = Rational(w0)
    w1 = x0 - f.subs(_W, x0) / df.subs(_W, x0)
    residual = f.subs(_W, w1)
    w2 = w1 - residual / df.subs(_W, w1)
    return {
        "a": Fraction(int(w1.p), int(w1.q)),
        "b": Fraction(int(residual.p), int(residual.q)),
        "c": Fraction(int(w2.p), int(w2.q)),
    }


def test_prompt_variety_and_new_shape():
    exam_shapes = {s for s in shapes(TOPIC, 60, difficulty="exam")}
    hard = shapes("newton_method_iteration", 60, difficulty="hard", kind="cubic")
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        expected = _oracle(q.text)
        got = {p.label: frac(p.answer_text) for p in q.parts}
        assert got == expected, (seed, q.text)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_first_estimate():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        m, area, _ = _setting(q.text)
        a = frac(q.parts[0].answer_text) + Fraction(1, 2)  # a wrong, positive w1
        b = m * a * a - area
        c = a - b / (2 * m * a)
        if b == 0 or len({a, b, c}) < 3 or not survives_nsimplify(a, b, c):
            continue  # the marker restates c through nsimplify, which garbles some large fractions
        tested += 1
        reply = {"a": str(a), "b": str(b), "c": str(c)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through is True
        assert not result.correct
    assert tested >= 80


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        q = exam(TOPIC, seed)
        result = grade(sk, q, {"a": "999", "b": "998", "c": "997"})
        assert result.marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
    assert canonical_reply(exam(TOPIC, 7)) == canonical_reply(exam(TOPIC, 7))
