"""Tests `uc2-orbit-stabiliser-count`: stabilisers and orbits are found by acting with every group element on the object
read back out of the question TEXT (SymPy powers for the cyclic group, explicit rotations and reflections for the n-gon),
the typed grader on every key and distractor, and the four-part exam chain."""

from __future__ import annotations

import re

from sympy.combinatorics import Permutation

import mathgen.generators.r4a_orbit_stabiliser_count_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4a_support as sup

TOPIC = "r4a_orbit_stabiliser_count"
SKILL_ID = "uc2-orbit-stabiliser-count"


def _dihedral(n):
    return [tuple((x + j) % n for x in range(n)) for j in range(n)] + [
        tuple((j - x) % n for x in range(n)) for j in range(n)
    ]


def _act(g, colouring):
    out = [None] * len(colouring)
    for v, c in enumerate(colouring):
        out[g[v]] = c
    return tuple(out)


def _black(text):
    return {int(v) for v in re.search(r"\{([\d, ]+)\}", text).group(1).split(",")}


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_key_is_the_stabiliser_order():
    for seed in range(100):
        question = sup.q(TOPIC, "easy", seed)
        n = (
            int(re.search(r"S_(\d+)", question.text).group(1))
            if "S_" in question.text
            else int(re.search(r"\{1, \.\.\., (\d+)\}", question.text).group(1))
        )
        cycles = [[int(v) - 1 for v in c.split()] for c in re.findall(r"\(([\d ]+)\)", question.text)]
        x = int(re.search(r"(?:stabiliser of|fix the point) (\d+)", question.text).group(1)) - 1
        perm = Permutation(cycles, size=n)
        fixing = sum(1 for j in range(perm.order()) if (perm**j)(x) == x)
        assert int(question.answer) == fixing, (seed, question.text)
        assert fixing > 1


def test_medium_key_is_the_orbit_size_of_a_colouring():
    for seed in range(100):
        question = sup.q(TOPIC, "medium", seed)
        n = int(re.search(r"regular (\d+)-gon", question.text).group(1))
        black = _black(question.text)
        colouring = tuple("B" if v in black else "W" for v in range(n))
        orbit = {_act(g, colouring) for g in _dihedral(n)}
        assert int(question.answer) == len(orbit), (seed, question.text)


def test_hard_key_is_the_orbit_size_of_a_three_colour_word():
    for seed in range(100):
        question = sup.q(TOPIC, "hard", seed)
        n = int(re.search(r"regular (\d+)-gon", question.text).group(1))
        word = re.search(r"\b([RGB]{%d})\b" % n, question.text).group(1)
        orbit = {_act(g, tuple(word)) for g in _dihedral(n)}
        assert int(question.answer) == len(orbit), (seed, question.text)
        assert len(set(word)) >= 2


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 6, "medium": 4, "hard": 4})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (3, 2, 2, 2), ("number",) * 4, (0, 1, 1, 0))


def test_exam_keys_by_acting_with_every_element():
    for seed in range(120):
        question = sup.exam(TOPIC, seed)
        n = int(re.search(r"regular (\d+)-gon", question.stem).group(1))
        black = _black(question.stem)
        colouring = tuple("B" if v in black else "W" for v in range(n))
        a, b, c, d = question.parts
        images = [_act(g, colouring) for g in _dihedral(n)]
        assert int(a.answer) == images.count(colouring)
        assert int(b.answer) == len(set(images))
        target_black = _black(c.prompt)
        target = tuple("B" if v in target_black else "W" for v in range(n))
        assert target in images
        assert int(c.answer) == images.count(target)
        q_size = int(re.search(r"has (\d+) different colourings", d.prompt).group(1))
        assert (2 * n) % q_size == 0 and int(d.answer) == 2 * n // q_size


def test_exam_follow_through_from_a_wrong_stabiliser():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(120):
        question = sup.exam(TOPIC, seed)
        n = int(re.search(r"regular (\d+)-gon", question.stem).group(1))
        true_a = int(question.parts[0].answer)
        for wrong in (true_a * 2, true_a // 2 if true_a % 2 == 0 else None):
            if wrong is None or wrong < 2 or (2 * n) % wrong:
                continue
            orbit = 2 * n // wrong
            sent = {p.label: p.answer_text for p in question.parts} | {
                "a": str(wrong),
                "b": str(orbit),
                "c": str(wrong),
            }
            if str(orbit) == question.parts[1].answer_text or orbit == wrong or orbit == 1:
                continue
            result = grade(s, question, sent)
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, sent)
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, sent)
            checked += 1
    assert checked >= 10, checked
