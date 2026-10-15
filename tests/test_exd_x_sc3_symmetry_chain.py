"""Exam tier of `f10-symmetry-construction`: sides from an angle fact, then the symmetry of the marked polygon."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import exd_x_sc3_support as sup

SKILL = "f10-symmetry-construction"
TOPIC = "exd_x_sc3_symmetry_chain"
SEEDS = range(300)
_ORDINAL = {
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _sides(stem: str) -> int:
    if m := re.search(r"Each interior angle of a regular polygon is (\d+)°", stem):
        return round(360 / (180 - int(m.group(1))))
    if m := re.search(r"Each exterior angle of a regular polygon is (\d+)°", stem):
        return round(360 / int(m.group(1)))
    m = re.search(r"add to (\d+)°", stem)
    return int(m.group(1)) // 180 + 2


def _dotted_symmetry(n: int, k: int) -> tuple[int, int]:
    """(reflections, rotations) of a regular n-gon that map the marked vertices {0, k, 2k, ...} onto themselves,
    found by trying every symmetry of the polygon."""
    marked = {i for i in range(n) if i % k == 0}
    rotations = sum(1 for s in range(n) if {(i + s) % n for i in marked} == marked)
    reflections = sum(1 for s in range(n) if {(s - i) % n for i in marked} == marked)
    return reflections, rotations


def oracle(q) -> dict[str, str]:
    k = _ORDINAL[re.search(r"every (\w+) vertex", q.stem).group(1)]
    n = _sides(q.stem)
    lines, order = _dotted_symmetry(n, k)
    assert lines == order
    return {"a": str(n), "b": str(lines), "c": sup.typed(Fraction(360, order), 1)}


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert oracle(q) == sup.canonical(q), (seed, q.text)


def test_all_three_angle_facts_occur():
    stems = " ".join(sup.exam(TOPIC, s).stem for s in range(60))
    assert "interior angle of a regular" in stems and "exterior angle" in stems and "add to" in stems


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_follow_through_from_a_wrong_side_count():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        k = _ORDINAL[re.search(r"every (\w+) vertex", q.stem).group(1)]
        n = int(q.parts[0].answer_text)
        wrong_n = n + k  # a whole number of dots again, so (b) is an exact whole number
        expect = {"b": str(wrong_n // k), "c": sup.typed(Fraction(360 * k, wrong_n), 1)}
        sup.assert_follow_through(SKILL, q, {"a": str(wrong_n)}, expect)
