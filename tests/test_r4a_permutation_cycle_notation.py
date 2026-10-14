"""Tests `uc2-permutation-cycle-notation`: keys re-derived from the question text with SymPy's permutation class and by
tracing points through the written cycles, the typed grader on every key and distractor, and the four-part exam chain."""

from __future__ import annotations

import re

import pytest
from sympy.combinatorics import Permutation

import mathgen.generators.r4a_permutation_cycle_notation_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4a_support as sup

TOPIC = "r4a_permutation_cycle_notation"
SKILL_ID = "uc2-permutation-cycle-notation"
_SKILL = sup.skill(SKILL_ID)
_IMAGES = re.compile(r"sends 1, 2, \.\.\., \d+ to ([\d, ]+) respectively")
_PQ = re.compile(r"p = \(([\d, ]+)\) and q = \(([\d, ]+)\)")
_CYCLES = re.compile(r"\((\d+(?: \d+)+)\)")


def _normal(cycles_zero_based):
    out = []
    for cycle in cycles_zero_based:
        cycle = [x + 1 for x in cycle]
        at = cycle.index(min(cycle))
        out.append(cycle[at:] + cycle[:at])
    return sorted(out)


def _from_images(images):
    return Permutation([v - 1 for v in images], size=len(images))


def _traced(cycles, n):
    images = []
    for start in range(1, n + 1):
        x = start
        for cycle in reversed(cycles):
            if x in cycle:
                x = cycle[(cycle.index(x) + 1) % len(cycle)]
        images.append(x)
    return images


def _written_cycles(text):
    return [[int(v) for v in m.split()] for m in _CYCLES.findall(text)]


def _n_of(text):
    return int(re.search(r"S_(\d+)", text).group(1))


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "set")


def test_easy_key_is_the_cycle_decomposition():
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        images = [int(v) for v in _IMAGES.search(question.text).group(1).split(",")]
        assert [list(c) for c in question.answer] == _normal(_from_images(images).cyclic_form), (seed, question.text)
        assert len(question.answer) >= 2


def test_medium_key_applies_q_first():
    for seed in range(80):
        question = sup.q(TOPIC, "medium", seed)
        p, q = (tuple(int(v) for v in g.split(",")) for g in _PQ.search(question.text).groups())
        product = [p[q[i] - 1] for i in range(len(p))]
        assert [list(c) for c in question.answer] == _normal(_from_images(product).cyclic_form), (seed, question.text)
        assert sorted(map(tuple, question.answer)) != sorted(
            map(tuple, _normal(_from_images([q[p[i] - 1] for i in range(len(p))]).cyclic_form))
        )


def test_hard_key_reads_the_product_right_to_left():
    for seed in range(120):
        question = sup.q(TOPIC, "hard", seed)
        cycles = _written_cycles(question.text)
        assert len(cycles) == 3
        n = _n_of(question.text) if "S_" in question.text else max(max(c) for c in cycles)
        images = _traced(cycles, n)
        assert [list(c) for c in question.answer] == _normal(_from_images(images).cyclic_form), (seed, question.text)
        left_to_right = _traced(list(reversed(cycles)), n)
        assert images != left_to_right


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "set")


def test_every_key_is_in_the_stated_normal_form():
    for tier in sup.TIERS:
        for seed in range(80):
            cycles = sup.q(TOPIC, tier, seed).answer
            assert cycles == sorted(cycles, key=lambda c: c[0])
            assert all(len(c) >= 2 and c[0] == min(c) and len(set(c)) == len(c) for c in cycles)
            flat = [x for c in cycles for x in c]
            assert len(flat) == len(set(flat))


def test_a_rotated_cycle_or_other_order_of_cycles_grades_by_the_stated_convention():
    question = sup.q(TOPIC, "easy", 3)
    reordered = ", ".join(reversed(["(" + ", ".join(map(str, c)) + ")" for c in question.answer]))
    assert grade(_SKILL, question, "{" + reordered + "}").correct  # cycles are a set, so their order is free
    first = question.answer[0]
    rotated = "(" + ", ".join(map(str, first[1:] + first[:1])) + ")"
    others = ", ".join("(" + ", ".join(map(str, c)) + ")" for c in question.answer[1:])
    assert not grade(_SKILL, question, "{" + rotated + ", " + others + "}").correct  # not in the stated normal form


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 40, "medium": 40, "hard": 40})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (3, 2, 2, 2), ("set", "number", "number", "number"), (0, 0, 1, 0))


def test_exam_keys_match_sympy():
    for seed in range(120):
        question = sup.exam(TOPIC, seed)
        cycles = _written_cycles(question.stem)
        n = _n_of(question.stem)
        perm = _from_images(_traced(cycles, n))
        a, b, c, d = question.parts
        assert [list(x) for x in a.answer] == _normal(perm.cyclic_form)
        assert int(b.answer) == perm.order()
        big = int(re.search(r"σ\^(\d+)", c.prompt).group(1))
        least = next(r for r in range(1, 200) if perm**r == perm**big)
        assert int(c.answer) == least
        x = int(re.search(r"image of (\d+) under", d.prompt).group(1))
        assert int(d.answer) == (perm**big)(x - 1) + 1


def test_exam_follow_through_from_a_wrong_order():
    def replies(question):
        b = question.parts[1]
        big = int(re.search(r"σ\^(\d+)", question.parts[2].prompt).group(1))
        wrong = int(b.answer) + 1
        if big % wrong in (0,) or big % wrong == int(question.parts[2].answer):
            return None
        return {p.label: p.answer_text for p in question.parts} | {"b": str(wrong)}

    def expected(question, sent):
        big = int(re.search(r"σ\^(\d+)", question.parts[2].prompt).group(1))
        return str(big % int(sent["b"]))

    sup.check_follow_through(SKILL_ID, TOPIC, "c", replies, expected)


@pytest.mark.parametrize("seed", range(5))
def test_exam_part_a_accepts_the_cycles_in_any_order(seed):
    question = sup.exam(TOPIC, seed)
    a = question.parts[0]
    text = "{" + ", ".join(reversed(["(" + ", ".join(map(str, c)) + ")" for c in a.answer])) + "}"
    replies = {p.label: p.answer_text for p in question.parts} | {"a": text}
    assert grade(_SKILL, question, replies).parts["a"].marks == 3
