"""Independent oracle test for the n-th roots of unity skill (ACMSM087).

This test file does NOT import mathgen.generators.p7c_nth_roots_build. It
parses the rendered question text and answer_text and re-derives the
correct answer from scratch, using cmath/math -- a different route from the
generator's own exact-integer-fraction construction.
"""

from __future__ import annotations

import cmath
import math
import re

import pytest

from mathgen import make

TOPIC = "nth_roots_of_unity"
SEEDS = range(70)  # >= 60 per tier

_EASY_RE = re.compile(r"For n = (\d+), what is the argument \(in degrees\) of root k = (-?\d+)\?$")
_HARD_RE = re.compile(r"For n = (\d+), how many of the n-th roots of unity have a negative real part\?$")
_ANSWER_DEG_RE = re.compile(r"^(-?\d+)°$")


def _root_angle_degrees_raw(n: int, k: int) -> float:
    """Raw angle (degrees, in (-180, 180]) of root k via a complex exponential."""
    z = cmath.exp(complex(0, 2 * math.pi * k / n))
    return math.degrees(cmath.phase(z))


def _root_angle_degrees(n: int, k: int) -> float:
    """Angle (degrees, 0..360) of root k, rounded then wrapped.

    ``cmath.phase`` wraps to ``(-pi, pi]``, so a point landing exactly on the
    positive real axis after a full turn (k a multiple of n) can come back as
    a tiny negative angle from floating-point noise. Rounding to the nearest
    integer BEFORE wrapping (rather than adding 360 to the raw float first)
    turns that noise into exactly 0, not 359.999... -> 360.
    """
    return round(_root_angle_degrees_raw(n, k)) % 360


def _negative_real_count(n: int) -> int:
    """Count of roots with negative real part, purely by numeric enumeration."""
    count = 0
    for k in range(n):
        z = cmath.exp(complex(0, 2 * math.pi * k / n))
        if z.real < -1e-9:
            count += 1
    return count


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_argument_questions_oracle(difficulty):
    tested = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        match = _EASY_RE.search(q.text)
        assert match, f"could not parse question text: {q.text!r}"
        n, k = int(match.group(1)), int(match.group(2))

        raw_deg = _root_angle_degrees_raw(n, k)
        rounded_expected = _root_angle_degrees(n, k)
        assert abs(raw_deg - round(raw_deg)) < 1e-6, f"n={n} k={k}: angle {raw_deg} is not an exact integer degree"

        answer_match = _ANSWER_DEG_RE.match(q.answer_text)
        assert answer_match, f"could not parse answer_text: {q.answer_text!r}"
        got = int(answer_match.group(1))
        assert got == rounded_expected, (
            f"seed={seed} difficulty={difficulty}: question says n={n} k={k}, "
            f"oracle computes {rounded_expected}°, generator answered {got}°"
        )
        tested += 1
    assert tested >= 60


def test_count_questions_oracle():
    tested = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        match = _HARD_RE.search(q.text)
        assert match, f"could not parse question text: {q.text!r}"
        n = int(match.group(1))

        expected = _negative_real_count(n)
        got = int(q.answer_text)
        assert got == expected, f"seed={seed}: n={n}, oracle computes {expected}, generator answered {got}"
        tested += 1
    assert tested >= 60


def test_hand_checked_cases_via_oracle():
    """The two hand-checked cases, derived by the oracle's
    own numeric-enumeration route (not the generator's closed-form inequality)."""
    assert _negative_real_count(4) == 1
    assert _negative_real_count(6) == 3
