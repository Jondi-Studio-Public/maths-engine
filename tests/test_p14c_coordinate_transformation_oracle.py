"""Re-derives each rendered question's answer from `question.text` alone, via a fresh, hand-written affine-map
computation (never importing anything from `mathgen.generators.p14c_coordinate_transformation_build`).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14c_coordinate_transformation"

_POINT_RE = re.compile(r"point \((-?\d+), (-?\d+)\)")
_TRANSLATE_RE = re.compile(r"translate the result by \((-?\d+), (-?\d+)\)")
_IMAGE_RE = re.compile(r"image point \((-?\d+), (-?\d+)\)")
_DILATION_RE = re.compile(r"\(x, y\) -> \(([^,]+), ([^)]+)\)")
_ROTATION_RE = re.compile(r"(\d+) degrees counterclockwise")


def _int_pair(match: re.Match) -> tuple[int, int]:
    a, b = match.groups()
    return int(a), int(b)


def _parse_coef(term: str, var: str) -> int:
    term = term.strip()
    if term == var:
        return 1
    if term == f"-{var}":
        return -1
    return int(term[: -len(var)])


def _linear_from_text(text: str, x: int, y: int) -> tuple[int, int]:
    """Apply the linear part named in `text` to (x, y), by a rule written
    fresh here (never the generator's own matrix build)."""
    lower = text.lower()
    if "reflect it in the x-axis" in lower:
        return x, -y
    if "reflect it in the y-axis" in lower:
        return -x, y
    if "reflect it in the line y = x" in lower:
        return y, x
    if "reflect it in the line y = -x" in lower:
        return -y, -x
    rot = _ROTATION_RE.search(text)
    if rot is not None:
        theta = int(rot.group(1))
        if theta == 90:
            return -y, x
        if theta == 180:
            return -x, -y
        return y, -x  # 270
    dil = _DILATION_RE.search(text)
    if dil is not None:
        a = _parse_coef(dil.group(1), "x")
        b = _parse_coef(dil.group(2), "y")
        return a * x, b * y
    raise AssertionError(f"could not identify the linear part in: {text!r}")


def _linear_inverse_from_text(text: str, x: int, y: int) -> tuple[Fraction, Fraction]:
    """Apply the INVERSE of the linear part named in `text` to (x, y)."""
    lower = text.lower()
    if "reflect it in the x-axis" in lower:
        return Fraction(x), Fraction(-y)
    if "reflect it in the y-axis" in lower:
        return Fraction(-x), Fraction(y)
    if "reflect it in the line y = x" in lower:
        return Fraction(y), Fraction(x)
    if "reflect it in the line y = -x" in lower:
        return Fraction(-y), Fraction(-x)
    rot = _ROTATION_RE.search(text)
    if rot is not None:
        theta = int(rot.group(1))
        # Inverse rotation = rotate by (360 - theta).
        inv_theta = (360 - theta) % 360
        if inv_theta == 90:
            return Fraction(-y), Fraction(x)
        if inv_theta == 180:
            return Fraction(-x), Fraction(-y)
        if inv_theta == 270:
            return Fraction(y), Fraction(-x)
        return Fraction(x), Fraction(y)  # inv_theta == 0, unreachable (M never identity)
    dil = _DILATION_RE.search(text)
    if dil is not None:
        a = _parse_coef(dil.group(1), "x")
        b = _parse_coef(dil.group(2), "y")
        return Fraction(x, a), Fraction(y, b)
    raise AssertionError(f"could not identify the linear part in: {text!r}")


def _oracle_forward(text: str) -> tuple[int, int]:
    x0, y0 = _int_pair(_POINT_RE.search(text))
    tx, ty = _int_pair(_TRANSLATE_RE.search(text))
    lx, ly = _linear_from_text(text, x0, y0)
    return lx + tx, ly + ty


def _oracle_exam(text: str) -> tuple[Fraction, Fraction]:
    tx, ty = _int_pair(_TRANSLATE_RE.search(text))
    x1, y1 = _int_pair(_IMAGE_RE.search(text))
    sx, sy = x1 - tx, y1 - ty
    return _linear_inverse_from_text(text, sx, sy)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_recomputes_forward_tiers(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        x1, y1 = _oracle_forward(q.text)
        assert (x1, y1) == (q.answer["x"], q.answer["y"]), (
            f"{tier} seed={seed}: oracle ({x1}, {y1}) != canonical ({q.answer['x']}, {q.answer['y']}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam_tier():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        x0, y0 = _oracle_exam(q.text)
        assert (x0, y0) == (q.answer["x"], q.answer["y"]), (
            f"exam seed={seed}: oracle ({x0}, {y0}) != canonical ({q.answer['x']}, {q.answer['y']}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Proven red first: the oracle actually checks something ---------------


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    x1, y1 = _oracle_forward(q.text)
    assert (x1, y1) != (x1 + 1, y1)


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    x0, y0 = _oracle_exam(q.text)
    assert (x0, y0) != (x0 + 1, y0)
