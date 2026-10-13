"""Exam tier of `f10-quadrilateral-properties`: a three-part chain on one rhombus."""

from __future__ import annotations

import math
import re

import exd_x_sc3_support as sup

SKILL = "f10-quadrilateral-properties"
TOPIC = "exd_x_sc3_quadrilateral_chain"
SEEDS = range(300)

_ANGLES = re.compile(r"Angle (\w)(\w)(\w) = \((\d+)x \+ (\d+)\)° and angle \w\w\w = \((\d+)x - (\d+)\)°\.")
_LENGTHS = re.compile(r"perimeter of (\d+) cm\. Its diagonal (\w)(\w) is (\d+) cm long")


def _angle(p, v, q) -> float:
    a, b = (p[0] - v[0], p[1] - v[1]), (q[0] - v[0], q[1] - v[1])
    return math.degrees(math.acos((a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b))))


def oracle(q) -> dict[str, str]:
    m = _ANGLES.search(q.stem)
    if m:
        p, qq, r, s = (int(m.group(i)) for i in (4, 5, 6, 7))
        x = next(v for v in range(1, 200) if (p * v + qq) + (r * v - s) == 180)
        big = p * x + qq
        # Coordinates: rhombus ABCD with the angle at B equal to `big`; measure the angle between side AB and diagonal AC.
        bb = (0.0, 0.0)
        aa = (1.0, 0.0)
        cc = (math.cos(math.radians(big)), math.sin(math.radians(big)))
        dd = (aa[0] + cc[0], aa[1] + cc[1])
        assert abs(_angle(aa, bb, cc) - big) < 1e-9 and abs(_angle(bb, cc, dd) - (180 - big)) < 1e-9
        return {"a": str(x), "b": sup.typed(big, 1), "c": f"{_angle(bb, aa, cc):.1f}"}
    m = _LENGTHS.search(q.stem)
    assert m, q.stem
    perimeter, diagonal = int(m.group(1)), int(m.group(4))
    side = perimeter // 4
    half = diagonal // 2
    other = math.isqrt(side * side - half * half)
    assert other * other + half * half == side * side
    return {"a": str(side), "b": str(other), "c": str(2 * other)}


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert oracle(q) == sup.canonical(q), (seed, q.text)


def test_both_forms_occur():
    kinds = {bool(_ANGLES.search(sup.exam(TOPIC, s).stem)) for s in range(40)}
    assert kinds == {True, False}


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_follow_through_from_a_wrong_first_part():
    checked = 0
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        m = _ANGLES.search(q.stem)
        if m:
            p, qq = int(m.group(4)), int(m.group(5))
            wrong_x = int(q.parts[0].answer_text) + 1
            big = p * wrong_x + qq
            expect = {"b": sup.typed(big, 1), "c": sup.typed(90 - big / 2, 1)}
            sup.assert_follow_through(SKILL, q, {"a": str(wrong_x)}, expect)
        else:
            half = int(_LENGTHS.search(q.stem).group(4)) // 2
            wrong_side = int(q.parts[0].answer_text) + 1
            other = round(math.sqrt(wrong_side**2 - half**2))
            sup.assert_follow_through(SKILL, q, {"a": str(wrong_side)}, {"b": str(other), "c": str(2 * other)})
        checked += 1
    assert checked == len(SEEDS)
