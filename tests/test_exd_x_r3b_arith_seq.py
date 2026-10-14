"""Tests the multi-part exam tier of `senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear`."""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_arith_seq"
SKILL_ID = "senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear"
_SKILL = S.skill(SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _facts(q):
    """(A, p, B, q, N, threshold, increasing) from the rendered text alone."""
    a, p, b, pos_q = (int(v) for v in re.findall(r"\d+", q.stem)[:4])
    n_total = int(re.search(r"first (\d+)", q.parts[2].prompt).group(1))
    threshold = int(re.search(r"(?:more|fewer) than \$?(\d+)", q.parts[3].prompt).group(1))
    return a, p, b, pos_q, n_total, threshold, "more than" in q.parts[3].prompt


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [1, 2, 2, 2])


def test_keys_match_independent_sequence_over_300_seeds(questions):
    for q in questions:
        a, p, b, pos_q, n_total, threshold, increasing = _facts(q)
        d, rem = divmod(b - a, pos_q - p)
        assert rem == 0 and int(q.parts[0].answer_text) == d, q.stem
        t1 = a - (p - 1) * d
        assert int(q.parts[1].answer_text) == t1, q.stem
        terms = [t1 + k * d for k in range(60)]
        assert int(q.parts[2].answer_text) == sum(terms[:n_total]), q.stem
        first = next(k + 1 for k, v in enumerate(terms) if (v > threshold if increasing else v < threshold))
        assert int(q.parts[3].answer_text) == first, q.stem
        assert (d > 0) == increasing and all(v > 0 for v in terms[:first])


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_common_difference(questions):
    checked = 0
    for q in questions[:100]:
        a, p, _, _, n_total, threshold, increasing = _facts(q)
        wrong_d = int(q.parts[0].distractors[0])
        if wrong_d == 0:
            continue
        t1 = a - (p - 1) * wrong_d
        total = n_total * (2 * t1 + (n_total - 1) * wrong_d) // 2
        first = (threshold - t1) // wrong_d + 2
        reply = {"a": str(wrong_d), "b": str(t1), "c": str(total), "d": str(first)}
        if t1 <= 0 or total <= 0 or first <= 0 or len(set(reply.values())) < 4:
            continue
        if n_total * (2 * t1 + (n_total - 1) * wrong_d) % 2:
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.stem, label, reply)
        checked += 1
    assert checked >= 40


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "9999", "b": "-3", "c": "11", "d": "-17"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
