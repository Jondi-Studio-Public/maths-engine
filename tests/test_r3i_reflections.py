"""Tests the reflections-compose-to-rotation skill: keys re-derived with complex numbers, grading, and the exam chain."""

from __future__ import annotations

import cmath
import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "reflections-compose-to-rotation"
TOPIC = "r3i_reflect_rotate"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
EXAM = "r3i_exam_reflect_rotate"


def _matrix(text):
    """Read '[[√(3)/2, -1/2], ...]' into a list of rows of floats."""
    return [
        [float(parse_expr(c.replace("√", "sqrt"), transformations=_TRANSFORMS)) for c in row] for row in _rows(text)
    ]


def _rows(text):
    return [re.split(r",\s*(?![^()]*\))", r.strip()) for r in re.findall(r"\[([^\[\]]+)\]", text)]


def _images(mirrors):
    """Columns of the single matrix equal to reflecting in each mirror (degrees) in turn."""
    cols = []
    for z in (1 + 0j, 1j):
        for deg in mirrors:
            z = cmath.exp(2j * math.radians(deg)) * z.conjugate()
        cols.append(z)
    return [[cols[0].real, cols[1].real], [cols[0].imag, cols[1].imag]]


def _close(a, b):
    return all(abs(x - y) < 1e-9 for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def _angles(text):
    return [int(v) for v in re.findall(r"(\d+)°", text.split("Find")[0])]


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "matrix", ["senior-define-and-use-composition-of-linear-transformations-and"])


def test_easy_keys_are_the_product_in_the_order_applied():
    shapes = set()
    for seed in range(120):
        q = sup.tier_question(TOPIC, "easy", seed)
        stem = q.text
        mats = {name: _matrix(m) for name, m in re.findall(r"([PQR]) = (\[\[.+?\]\])", stem)}
        last = stem.split("Find the matrix")[1]
        if "rotat" in last.split(" and ")[0] or "rotates" in last:
            first_is_rotation = last.index("rotates") < last.index("reflects")
        else:
            first_is_rotation = False
        if "Q" in mats:
            shapes.add("two mirrors")
            expected = _mul(mats["Q"], mats["P"])
        elif first_is_rotation:
            shapes.add("turn then mirror")
            expected = _mul(mats["P"], mats["R"])
        else:
            shapes.add("mirror then turn")
            expected = _mul(mats["R"], mats["P"])
        assert _close(_matrix(q.answer_text), expected), (seed, stem, q.answer_text)
    assert shapes == {"two mirrors", "turn then mirror", "mirror then turn"}


def test_medium_keys_match_the_complex_number_reflections():
    for seed in range(150):
        q = sup.tier_question(TOPIC, "medium", seed)
        a, b = _angles(q.text)
        assert _close(_matrix(q.answer_text), _images([a, b])), (seed, q.text, q.answer_text)


def test_hard_keys_match_the_complex_number_reflections():
    seen = set()
    for seed in range(200):
        q = sup.tier_question(TOPIC, "hard", seed)
        key = _matrix(q.answer_text)
        if "three lines" in q.text:
            seen.add("three")
            assert _close(key, _images(_angles(q.text)[:3])), (seed, q.text)
            assert abs(key[0][0] + key[1][1]) < 1e-9  # a reflection has trace 0
        else:
            seen.add("second mirror")
            a, phi = _angles(q.text)[0], _angles(q.text)[-1]
            # L2 is the reflection matrix with a + (L1 then L2) giving the stated rotation
            first = _images([a])
            product = _mul(key, first)  # reflect in L1, then L2
            expected = [
                [math.cos(math.radians(phi)), -math.sin(math.radians(phi))],
                [math.sin(math.radians(phi)), math.cos(math.radians(phi))],
            ]
            assert _close(product, expected), (seed, q.text, q.answer_text)
    assert seen == {"three", "second mirror"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    # a rotation by a multiple of 30 degrees has only eleven non-trivial values, so the medium tier is capped there
    distinct, top = {"easy": (6, 0.35), "medium": (10, 0.2), "hard": (10, 0.2)}[tier]
    sup.assert_answer_space(TOPIC, tier, distinct=distinct, top=top)


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM)
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(40)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(40)]
    assert all("The matrix of" in t for t in easy)
    assert not any("The matrix of" in t for t in medium)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "matrix", "number", "matrix"], [2, 2, 1, 3], [False, False, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)


def test_exam_keys_match_the_complex_number_reflections():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        a, b = _angles(q.text)
        phi = int(q.parts[0].answer_text)
        assert phi == (2 * (b - a)) % 360
        assert _close(_matrix(q.parts[1].answer_text), _images([a, b]))
        assert int(q.parts[2].answer_text) * phi == 360
        px, py = (int(v) for v in re.search(r"point \((-?\d+), (-?\d+)\)", q.parts[3].prompt).groups())
        image = _images([a, b])
        expected = [[image[0][0] * px + image[0][1] * py, image[1][0] * px + image[1][1] * py]]
        assert _close(expected, _matrix(q.parts[3].answer_text)), (seed, q.parts[3].answer_text)


def test_exam_follow_through_from_a_wrong_first_part():
    sk = sup.skill_object(SKILL)
    usable = 0
    for seed in range(30):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (45, 60, 90, 120, 180):
            if wrong == int(canon["a"]):
                continue
            replies = {**canon, "a": str(wrong), "c": str(360 // wrong)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            if result.parts["c"].marks == q.parts[2].marks:
                usable += 1
                break
    assert usable >= 20
