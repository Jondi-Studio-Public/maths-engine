"""Tests `uc2-symmetric-group-conjugacy`: classes, centralisers and element orders are counted by running through every
permutation of {1, ..., n} (n at most 9) and reading the permutations out of the question text."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import factorial, gcd, lcm

import mathgen.generators.r4c_symmetric_group_conjugacy_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_symmetric_group_conjugacy"
SKILL_ID = "uc2-symmetric-group-conjugacy"


def _cycle_type(images):
    seen, out = set(), []
    for start in range(len(images)):
        if start in seen:
            continue
        length, x = 0, start
        while x not in seen:
            seen.add(x)
            x = images[x]
            length += 1
        out.append(length)
    return tuple(sorted(out, reverse=True))


@lru_cache(maxsize=None)
def _type_counts(n):
    """How many permutations of n points have each cycle type, by running through all of them."""
    return Counter(_cycle_type(p) for p in permutations(range(n)))


def _sigma(text, n):
    images = list(range(n))
    for chunk in re.findall(r"\(([\d ]+)\)", re.search(r"σ = ((?:\([\d ]+\))+)", text).group(1)):
        cycle = [int(v) - 1 for v in chunk.split()]
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            images[a] = b
    return images


def _n(text):
    return int(re.search(r"S_(\d+)", text).group(1))


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_class_sizes_by_brute_force():
    for seed in range(60):
        question = sup.q(TOPIC, "easy", seed)
        n = _n(question.text)
        assert int(question.answer) == _type_counts(n)[_cycle_type(_sigma(question.text, n))], (seed, question.text)


def test_medium_centraliser_order_by_brute_force():
    for seed in range(40):
        question = sup.q(TOPIC, "medium", seed)
        n = _n(question.text)
        images = _sigma(question.text, n)
        if n == 9:  # a direct count over 362880 permutations is slow; use class size and orbit-stabiliser instead
            assert int(question.answer) == factorial(n) // _type_counts(n)[_cycle_type(images)]
            continue
        commuting = sum(1 for p in permutations(range(n)) if all(p[images[i]] == images[p[i]] for i in range(n)))
        assert int(question.answer) == commuting, (seed, question.text)


def test_hard_counts_elements_of_the_order():
    seen = set()
    for seed in range(80):
        question = sup.q(TOPIC, "hard", seed)
        n = _n(question.text)
        found = re.search(r"exactly (\d+)|is (\d+) the least", question.text)
        k = int(found.group(1) or found.group(2))
        seen.add((n, k))
        count = sum(c for t, c in _type_counts(n).items() if lcm(*t) == k)
        assert int(question.answer) == count, (seed, question.text)
    assert len(seen) >= 8


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    for tier, minimum, text_minimum in (("easy", 5, 40), ("medium", 6, 40), ("hard", 6, 8)):
        answers = {sup.q(TOPIC, tier, seed).answer_text for seed in range(80)}
        texts = {sup.q(TOPIC, tier, seed).text for seed in range(80)}
        assert len(answers) >= minimum and len(texts) >= text_minimum, (tier, len(answers), len(texts))


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(
        SKILL_ID, TOPIC, "abcd", (1, 3, 1, 3), ("number", "number", "number", "ordered_list"), (0, 0, 1, 0)
    )


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(60):
        question = sup.exam(TOPIC, seed)
        n = _n(question.stem)
        images = _sigma(question.stem, n)
        e = int(re.search(r"σ\^(\d+)", question.parts[3].prompt).group(1))
        pa, pb, pc, pd = question.parts
        t = _cycle_type(images)
        assert int(pa.answer) == lcm(*t)
        assert int(pb.answer) == _type_counts(n)[t]
        assert int(pc.answer) == factorial(n) // int(pb.answer)
        power = list(range(n))
        for _ in range(e):
            power = [images[power[i]] for i in range(n)]
        assert pd.answer == list(_cycle_type(power)), (seed, question.stem)
        assert pd.answer != list(t)


def test_exam_follow_through_for_the_centraliser():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        n = _n(question.stem)
        wrong = int(question.parts[1].answer) * 2
        if factorial(n) % wrong:
            continue
        sent = {p.label: p.answer_text for p in question.parts} | {"b": str(wrong), "c": str(factorial(n) // wrong)}
        result = grade(s, question, sent)
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, sent)
        checked += 1
    assert checked >= 10
