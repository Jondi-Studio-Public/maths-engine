"""Tests `uc2-permutation-sign-and-order`: orders, parities and powers re-derived from the question text with SymPy's
permutation class, the typed grader on every key and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import re

from sympy.combinatorics import Permutation

import mathgen.generators.r4a_permutation_sign_and_order_build  # noqa: F401  (registers the topics)
from tests import r4a_support as sup

TOPIC = "r4a_permutation_sign_and_order"
SKILL_ID = "uc2-permutation-sign-and-order"
_IMAGES = re.compile(r"to ([\d, ]+) respectively|images ([\d, ]+) of 1")
_CYCLE = re.compile(r"\(([\d ]+)\)")


def _perm_from(text):
    match = _IMAGES.search(text)
    images = [int(v) for v in (match.group(1) or match.group(2)).split(",")]
    return Permutation([v - 1 for v in images], size=len(images))


def _perm_of_cycles(group_text, n):
    """A product of disjoint cycles, e.g. "(2 10 4)(7 8)"."""
    perm = Permutation(list(range(n)))
    for chunk in _CYCLE.findall(group_text):
        cycle = [int(v) - 1 for v in chunk.split()]
        perm = perm * Permutation([cycle], size=n)  # disjoint, so the order of the factors does not matter
    return perm


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_key_is_the_order():
    for seed in range(100):
        question = sup.q(TOPIC, "easy", seed)
        assert int(question.answer) == _perm_from(question.text).order(), (seed, question.text)


def test_medium_key_counts_the_even_permutations():
    for seed in range(100):
        question = sup.q(TOPIC, "medium", seed)
        n = int(re.search(r"S_(\d+)", question.text).group(1))
        listed = re.findall(r"\([a-e]\) ((?:\([\d ]+\))+)", question.text)
        assert len(listed) == 5
        evens = sum(1 for item in listed if _perm_of_cycles(item, n).is_even)
        assert int(question.answer) == evens, (seed, question.text)
        assert 1 <= evens <= 4


def test_hard_key_is_the_order_of_the_power():
    for seed in range(100):
        question = sup.q(TOPIC, "hard", seed)
        perm = _perm_from(question.text)
        k = int(re.search(r"σ\^(\d+)", question.text).group(1))
        assert int(question.answer) == (perm**k).order(), (seed, question.text)
        assert int(question.answer) < perm.order()


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 4, "medium": 4, "hard": 6})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 2, 2, 3), ("number",) * 4, (0, 0, 0, 1))


def test_exam_keys_match_sympy():
    for seed in range(150):
        question = sup.exam(TOPIC, seed)
        perm = _perm_from(question.stem)
        a, b, c, d = question.parts
        assert int(a.answer) == perm.order()
        assert int(b.answer) == perm.size - perm.cycles  # n minus the number of cycles, fixed points included
        k = int(re.search(r"σ\^(\d+)", c.prompt).group(1))
        assert int(c.answer) == (perm**k).order()
        many = int(re.search(r"σ\^(\d+), how many", d.prompt).group(1))
        assert int(d.answer) == sum(1 for j in range(1, many + 1) if (perm**j).is_even)
        assert (int(b.answer) % 2 == 0) == perm.is_even


def test_exam_follow_through_for_the_count_of_even_powers():
    def replies(question):
        wrong_b = int(question.parts[1].answer) + 1  # the other parity
        return {p.label: p.answer_text for p in question.parts} | {"b": str(wrong_b)}

    def expected(question, sent):
        many = int(re.search(r"σ\^(\d+), how many", question.parts[3].prompt).group(1))
        return str(many * (3 + (-1) ** int(sent["b"])) // 4)

    sup.check_follow_through(SKILL_ID, TOPIC, "d", replies, expected)
