"""Exam chain of `ucv1-quotient-ring-operation`: reduce a product, find a root, evaluate at it."""

from __future__ import annotations

import re

import pytest
import sympy

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    exam_question,
)

TOPIC = "exd_x_r3f_quotient_ring_roots"
SKILL_ID = "ucv1-quotient-ring-operation"
X = sympy.Symbol("x")


def _poly_from_text(text: str, p: int):
    """Parse `5x^2 + 12x + 4` style text (implicit products) into a Poly over GF(p)."""
    terms = re.findall(r"([+-]?)\s*(\d*)(x(?:\^(\d+))?)?", text.replace(" ", ""))
    expr = 0
    for sign, coeff, var, power in terms:
        if not coeff and not var:
            continue
        c = int(coeff) if coeff else 1
        degree = (int(power) if power else 1) if var else 0
        expr += (-1 if sign == "-" else 1) * c * X**degree
    return sympy.Poly(expr, X, modulus=p)


def read_question(q):
    p = int(re.search(r"Z_(\d+)\[x\]", q.stem).group(1))
    m = _poly_from_text(re.search(r"m\(x\) = ([^,]*?)(?: in Z_|,| in the| and)", q.stem).group(1), p)
    a = _poly_from_text(re.search(r"a\(x\) = ([^,]*?) and b", q.stem).group(1), p)
    b = _poly_from_text(re.search(r"b\(x\) = ([^,.]*)", q.stem).group(1), p)
    return p, m, a, b


def oracle(q):
    p, m, a, b = read_question(q)
    reduced = (a * b).rem(m)
    roots = sorted(r for r in range(p) if m.eval(r) % p == 0)
    root = roots[0]
    return reduced, root, int(reduced.eval(root)) % p, roots


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [3, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["expression", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        p, m, a, b = read_question(q)
        reduced, root, value, roots = oracle(q)
        assert len(roots) == 2, q.text
        got = _poly_from_text(q.parts[0].answer_text.replace("*", ""), p)
        assert got == reduced, q.text
        assert int(q.parts[1].answer_text) == root
        assert int(q.parts[2].answer_text) == value
        # the value is also a(root) b(root) directly, since evaluating at a root of m respects the reduction
        assert value == (int(a.eval(root)) * int(b.eval(root))) % p, q.text
        assert value != 0


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC, min_options=2)


def test_answer_space():
    assert_answer_space(TOPIC)


def test_every_prime_occurs(questions):
    assert {read_question(q)[0] for q in questions} == {5, 7, 11, 13}


def _wrong_reduction_replies(q):
    p, m, a, b = read_question(q)
    _reduced, root, _value, _roots = oracle(q)
    for wrong_text in q.parts[0].distractors:
        wrong = _poly_from_text(wrong_text.replace("*", ""), p)
        value = int(wrong.eval(root)) % p
        if value not in (0, int(q.parts[2].answer_text)):
            return {"a": wrong_text, "b": str(root), "c": str(value)}
    return None


def test_follow_through_from_a_wrong_reduction():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_reduction_replies, ["a"], ["c"], seeds=range(300))


def _other_root_replies(q):
    p, m, a, b = read_question(q)
    reduced, root, _value, roots = oracle(q)
    other = roots[1]
    return {"a": q.parts[0].answer_text, "b": str(other), "c": str(int(reduced.eval(other)) % p)}


def test_follow_through_from_the_other_root():
    def usable(q):
        replies = _other_root_replies(q)
        return replies if int(replies["c"]) != 0 else None

    assert_follow_through(SKILL_ID, TOPIC, usable, ["b"], ["c"])


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "99*x + 98", "b": "97", "c": "96"})
