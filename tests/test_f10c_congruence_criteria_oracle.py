"""Independent oracle for `f10-congruence-similarity-criteria`. It imports nothing from the generator: it parses
the rendered question text (side lengths, angles and the vertex correspondence) and decides, by its own rules,
which conclusion the stated facts justify, then compares with the keyed answer and the offered distractors."""

import itertools
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "f10c_congruence_criteria"
TIERS = ["easy", "medium", "hard"]


def _parse(text):
    names = re.findall(r"[Tt]riangle ([A-Z]{3})\b", text)
    assert len(names) >= 2, text
    n1, n2 = names[0], names[1]
    i1 = text.index(n1)
    i2 = text.index(n2, i1 + 3)
    end = text.index("orrespond", i2)
    seg1, seg2 = text[i1 + 3 : i2], text[i2 + 3 : end]
    out = []
    for name, seg in ((n1, seg1), (n2, seg2)):
        sides = {}
        for a, b, v in re.findall(r"\b([A-Z])([A-Z]) = (\d+) cm", seg):
            sides[tuple(sorted((name.index(a), name.index(b))))] = int(v)
        angles = {name.index(v): int(d) for v, d in re.findall(r"angle ([A-Z]) = (\d+) degrees", seg)}
        out.append((sides, angles))
    return out


def _complete(angles):
    angles = dict(angles)
    if len(angles) == 2:
        angles[3 - sum(angles)] = 180 - sum(angles.values())
    return angles


def _conclusions(text):
    (s1, a1), (s2, a2) = _parse(text)
    full1, full2 = _complete(a1), _complete(a2)
    if any(full1[v] != full2[v] for v in set(full1) & set(full2)):
        return set()
    shared = sorted(set(s1) & set(s2))
    ratios = {Fraction(s2[e], s1[e]) for e in shared}
    if len(ratios) > 1:
        return set()
    equal_sides = bool(ratios) and ratios == {Fraction(1)}
    both = sorted(v for v in a1 if v in a2 and a1[v] == a2[v])
    found = set()
    between = [v for e, f in itertools.combinations(shared, 2) for v in set(e) & set(f) if len(set(e) & set(f)) == 1]
    sim_aa = len([v for v in full1 if v in full2]) >= 2
    if sim_aa:
        found.add("Similar by AA")
    if len(shared) == 3:
        found.add("Similar by SSS (proportional sides)")
    if any(v in both for v in between):
        found.add("Similar by SAS (proportional sides)")
    if equal_sides:
        if len(shared) == 3:
            found.add("Congruent by SSS")
        if any(v in both for v in between):
            found.add("Congruent by SAS")
        for u, w in itertools.combinations(both, 2):
            if (u, w) in shared:
                found.add("Congruent by ASA")
            if [e for e in shared if e != (u, w)]:
                found.add("Congruent by AAS")
        for v in both:
            if a1[v] == 90:
                others = tuple(x for x in range(3) if x != v)
                if others in shared and any(v in e for e in shared):
                    found.add("Congruent by RHS")
    return found


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_the_only_test_met_and_distractors_are_not_met(difficulty):
    neither = "Neither can be concluded from this information"
    for seed in range(250):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        met = _conclusions(q.text)
        if q.answer_text == neither:
            assert not met, (q.text, met)
        else:
            assert q.answer_text in met, (q.text, q.answer_text, met)
            others = met - {q.answer_text}
            if q.answer_text.startswith("Congruent"):
                assert all(o.startswith("Similar") for o in others), (q.text, met)
            else:
                assert not others, (q.text, met)
        assert not (set(q.distractors) & met), (q.text, q.distractors, met)
        if q.answer_text.startswith("Congruent by") and q.answer_text != "Congruent by SSA":
            assert not any(d.startswith("Similar") for d in q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_false_tests_are_never_met_and_never_the_answer(difficulty):
    for seed in range(250):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text not in ("Congruent by SSA", "Congruent by AAA")
        assert "SSA" not in "".join(_conclusions(q.text))


def test_parser_reads_a_rendered_question():
    q = make(TOPIC, seed=3, difficulty="hard")
    (s1, a1), (s2, a2) = _parse(q.text)
    assert s1 or a1 and s2 or a2
