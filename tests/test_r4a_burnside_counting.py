"""Tests `uc2-burnside-counting`: every key is a direct count of orbits (all colourings listed and grouped under the
rotations, or the rotations and reflections, of the ring of beads) using the numbers read back out of the question TEXT,
the typed grader on every key and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import re
from math import gcd
from itertools import combinations, product

import mathgen.generators.r4a_burnside_counting_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4a_support as sup

TOPIC = "r4a_burnside_counting"
SKILL_ID = "uc2-burnside-counting"


def _symmetries(n, reflections):
    maps = [tuple((x + j) % n for x in range(n)) for j in range(n)]
    if reflections:
        maps += [tuple((j - x) % n for x in range(n)) for j in range(n)]
    return maps


def _orbits(colourings, maps):
    seen, count = set(), 0
    for c in colourings:
        if c in seen:
            continue
        count += 1
        for g in maps:
            image = [None] * len(c)
            for v, col in enumerate(c):
                image[g[v]] = col
            seen.add(tuple(image))
    return count


def _beads_and_colours(text):
    n = int(re.search(r"(\d+) beads", text).group(1))
    k = int(re.search(r"(\d+) (?:colours|possible colours)", text).group(1))
    return n, k


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_key_counts_necklaces():
    for seed in range(60):
        question = sup.q(TOPIC, "easy", seed)
        n, k = _beads_and_colours(question.text)
        assert int(question.answer) == _orbits(product(range(k), repeat=n), _symmetries(n, False)), (
            seed,
            question.text,
        )


def test_medium_key_counts_bracelets():
    for seed in range(60):
        question = sup.q(TOPIC, "medium", seed)
        n, k = _beads_and_colours(question.text)
        bracelets = _orbits(product(range(k), repeat=n), _symmetries(n, True))
        assert int(question.answer) == bracelets, (seed, question.text)
        assert bracelets < _orbits(product(range(k), repeat=n), _symmetries(n, False))


def test_hard_key_counts_arrangements_with_an_exact_number_of_gold_beads():
    seen = set()
    for seed in range(100):
        question = sup.q(TOPIC, "hard", seed)
        n = int(re.search(r"(\d+) beads", question.text).group(1))
        m = int(re.search(r"exactly (\d+)", question.text).group(1))
        reflections = "bracelet" in question.text
        colourings = [tuple(1 if v in chosen else 0 for v in range(n)) for chosen in combinations(range(n), m)]
        assert int(question.answer) == _orbits(colourings, _symmetries(n, reflections)), (seed, question.text)
        seen.add(reflections)
    assert seen == {True, False}


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 6, "medium": 6, "hard": 8}, text_fraction=0.3)


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 2, 3, 2), ("number",) * 4, (0, 0, 1, 1))


def test_exam_keys_by_direct_counting():
    for seed in range(100):
        question = sup.exam(TOPIC, seed)
        n, k = _beads_and_colours(question.stem)
        a, b, c, d = question.parts
        every = list(product(range(k), repeat=n))
        rotation = tuple((x + 2) % n for x in range(n))
        reflection = tuple((-x) % n for x in range(n))  # fixes beads 0 and n/2
        assert int(a.answer) == sum(1 for col in every if all(col[rotation[v]] == col[v] for v in range(n)))
        assert int(b.answer) == sum(1 for col in every if all(col[reflection[v]] == col[v] for v in range(n)))
        assert int(c.answer) == _orbits(every, _symmetries(n, True))
        assert int(d.answer) == _orbits(every, _symmetries(n, False))


def test_exam_follow_through_from_a_wrong_fixed_count():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        n, k = _beads_and_colours(question.stem)
        a, b = int(question.parts[0].answer), int(question.parts[1].answer)
        sent = {p.label: p.answer_text for p in question.parts} | {"a": str(a + n)}
        # the exact follow-through of the stated formula, worked out by hand from the symmetry counts
        rotations = [tuple((x + j) % n for x in range(n)) for j in range(n)]
        total_rot_const = (
            sum(k ** len(_cycles(g)) for g in rotations) - sum(1 for j in range(n) if gcd(j, n) == 2 and j) * a
        )
        times_a = sum(1 for j in range(1, n) if gcd(j, n) == gcd(2, n))
        necklaces = (total_rot_const + times_a * (a + n)) / n
        if necklaces != int(necklaces) or int(necklaces) == int(question.parts[3].answer):
            continue
        sent["d"] = str(int(necklaces))
        result = grade(s, question, sent)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, sent)
        checked += 1
    assert checked >= 5, checked


def _cycles(g):
    seen, out = set(), []
    for start in range(len(g)):
        if start in seen:
            continue
        length, v = 0, start
        while v not in seen:
            seen.add(v)
            v = g[v]
            length += 1
        out.append(length)
    return out
