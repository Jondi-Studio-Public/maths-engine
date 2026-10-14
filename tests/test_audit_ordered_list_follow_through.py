"""An ordered_list part can work out its answer from earlier parts, and can be the earlier part a later one uses."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import engine
from mathgen.grading import parts as P

_SKILLS = {s.id: s for s in engine.load_skills()[1]}
_DIST = _SKILLS["senior-use-relative-frequencies-obtained-from-data-to-obtain"]
_PERM = _SKILLS["permutation-composition"]


def _exam(skill, seed):
    index = next(i for i, s in enumerate(skill.sources) if s.difficulty == "exam")
    return engine.make_for_skill(skill, seed, source=index)


def _cycle_lengths(perm):
    seen, lengths = set(), []
    for start in range(1, len(perm) + 1):
        length, point = 0, start
        while point not in seen:
            seen.add(point)
            point, length = perm[point - 1], length + 1
        if length:
            lengths.append(length)
    return lengths


def _image(perm, point, power):
    for _ in range(power):
        point = perm[point - 1]
    return point


def _ints(text):
    return [int(v) for v in re.findall(r"\d+", text)]


def test_distribution_follows_through_from_a_wrong_unknown():
    credited = 0
    for seed in range(1, 31):
        q = _exam(_DIST, seed)
        a, b, c, d = q.parts
        # Reading the table back from the canonical distribution: item t is frequency t over the total.
        probs = [Fraction(v) for v in d.answer_text.split(",")]
        n = math.lcm(*(p.denominator for p in probs))
        freqs = [int(p * n) for p in probs]
        m, k = int(b.answer_text), int(c.answer_text)
        if m == k or m not in freqs or k not in freqs:
            continue
        i, j = freqs.index(m), freqs.index(k)
        if freqs.count(m) > 1 or freqs.count(k) > 1:
            continue
        # A learner who swaps the two unknowns and keeps their own working consistent.
        swapped = list(freqs)
        swapped[i], swapped[j] = k, m
        reply = {
            "a": a.answer_text,
            "b": str(k),
            "c": str(m),
            "d": ", ".join(str(Fraction(f, n)) for f in swapped),
        }
        result = engine.grade(_DIST, q, reply)
        assert result.parts["d"].correct, (seed, reply)
        assert not result.parts["b"].correct
        credited += 1
        # The same list is no follow-through when the learner's own unknowns do not give it.
        wrong = dict(reply, d=d.answer_text)
        assert not engine.grade(_DIST, q, wrong).parts["d"].follow_through
    assert credited >= 15


def test_distribution_junk_and_impossible_probabilities_earn_nothing():
    for seed in range(1, 11):
        q = _exam(_DIST, seed)
        labels = [p.label for p in q.parts]
        for junk in ("0", "1", "2", "-1"):
            result = engine.grade(_DIST, q, {label: junk for label in labels})
            assert result.parts["d"].marks == 0, (seed, junk)
        full = {p.label: p.answer_text for p in q.parts}
        big = dict(full, b="500", c="-469", d="1/60, 7/30, 25/3, -469/60, 1/30")
        assert engine.grade(_DIST, q, big).parts["d"].marks == 0


def test_permutation_follows_through_from_a_wrong_composite():
    credited = 0
    for seed in range(1, 41):
        q = _exam(_PERM, seed)
        a, b, c = q.parts
        distractors = [d for d in a.distractors if isinstance(d, str)] or [str(d) for d in a.distractors]
        for text in distractors:
            perm = _ints(text)
            if sorted(perm) != list(range(1, len(perm) + 1)) or len(perm) != len(a.answer):
                continue
            order = math.lcm(*_cycle_lengths(perm))
            point, power = _ints(c.prompt)[-2:]
            reply = {"a": text, "b": str(order), "c": str(_image(perm, point, power))}
            result = engine.grade(_PERM, q, reply)
            if order == int(b.answer_text) and reply["c"] == c.answer_text:
                continue  # the wrong composite happens to give the canonical numbers
            assert not result.parts["a"].correct
            assert result.parts["b"].marks == b.marks, (seed, reply)
            if order != int(b.answer_text):
                assert result.parts["b"].follow_through
            if reply["c"] != c.answer_text:
                assert result.parts["c"].marks == c.marks, (seed, reply)
            credited += 1
    assert credited >= 20


def test_permutation_junk_and_non_permutations_carry_nothing():
    for seed in range(1, 11):
        q = _exam(_PERM, seed)
        n = len(q.parts[0].answer)
        for junk in ("0", "1", "2", "-1"):
            result = engine.grade(_PERM, q, {"a": junk, "b": junk, "c": junk})
            for part in q.parts:
                if part.answer_text != junk:  # a junk reply that is the canonical answer is simply right
                    assert result.parts[part.label].marks == 0, (seed, junk, part.label)
        not_a_permutation = "(" + ", ".join(["1"] * n) + ")"
        result = engine.grade(_PERM, q, {"a": not_a_permutation, "b": "1", "c": "1"})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 0


def test_list_functions_and_reply_values():
    assert P.reply_value("ordered_list", "(2, 3, 1)") == P.sympy.Tuple(2, 3, 1)
    assert P.reply_value("ordered_list", "2, x") is None
    assert P._perm_order(P.sympy.Tuple(2, 3, 1, 5, 4)) == 6
    with pytest.raises(ValueError):
        P._perm_order(P.sympy.Tuple(1, 1))
    assert P._perm_image(P.sympy.Tuple(2, 3, 1), 1, 4) == 2
