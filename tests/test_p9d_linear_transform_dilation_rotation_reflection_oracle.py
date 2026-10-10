"""Independent oracle for linear_transform_dilation_rotation_reflection.

Never imports mathgen.generators.p9d_linear_transform_dilation_rotation_reflection_build
-- not a helper, not a constant, not a table. Everything below is re-derived
from `question.text` and `question.answer_text` alone, the two fields a
learner actually receives.
"""

import re

import pytest

from mathgen import make

TOPIC = "linear_transform_dilation_rotation_reflection"
_SEEDS = range(60)


def _parse_answer(answer_text):
    match = re.match(r"x = (-?\d+), y = (-?\d+)", answer_text)
    assert match, f"unparseable answer_text: {answer_text!r}"
    return int(match.group(1)), int(match.group(2))


def _derive_dilation(text):
    """Read the two coefficient terms and the point straight from the
    sentence ("Apply a dilation (x, y) -> (2x, -3y) to the point (4, -1).
    Find the image point."), then apply the per-axis rule."""
    m = re.search(r"\(x, y\) -> \((-?\d+)x, (-?\d+)y\) to the point \((-?\d+), (-?\d+)\)", text)
    assert m, f"could not parse dilation question: {text!r}"
    lam1_str, lam2_str, x_str, y_str = m.groups()
    lambda1, lambda2 = int(lam1_str), int(lam2_str)
    x, y = int(x_str), int(y_str)
    return lambda1 * x, lambda2 * y


def _derive_rotation(text):
    m = re.search(r"Rotate the point \((-?\d+), (-?\d+)\) by (\d+) degrees counterclockwise about the origin", text)
    assert m, f"could not parse rotation question: {text!r}"
    x, y, theta = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if theta == 90:
        return -y, x
    if theta == 180:
        return -x, -y
    if theta == 270:
        return y, -x
    raise AssertionError(f"unexpected angle {theta} in {text!r}")


def _derive_reflection(text):
    m = re.search(r"Reflect the point \((-?\d+), (-?\d+)\) in (.+?)\. Find the image point", text)
    assert m, f"could not parse reflection question: {text!r}"
    x, y = int(m.group(1)), int(m.group(2))
    line_phrase = m.group(3)
    if line_phrase == "the x-axis":
        return x, -y
    if line_phrase == "the y-axis":
        return -x, y
    if line_phrase == "the line y = x":
        return y, x
    if line_phrase == "the line y = -x":
        return -y, -x
    raise AssertionError(f"unrecognised line phrase {line_phrase!r} in {text!r}")


@pytest.mark.parametrize("seed", _SEEDS)
def test_dilation_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="easy")
    assert _derive_dilation(q.text) == _parse_answer(q.answer_text)


@pytest.mark.parametrize("seed", _SEEDS)
def test_rotation_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="medium")
    assert _derive_rotation(q.text) == _parse_answer(q.answer_text)


@pytest.mark.parametrize("seed", _SEEDS)
def test_reflection_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="hard")
    assert _derive_reflection(q.text) == _parse_answer(q.answer_text)


def test_oracle_coverage():
    """Documents how many questions this oracle re-derives (3 tiers x 60 seeds)."""
    total = 3 * len(_SEEDS)
    assert total == 180
