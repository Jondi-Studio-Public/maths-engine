"""Independent oracle test for the n-th roots of a complex number skill
(ACMSM088).

This test file does NOT import mathgen.generators.p7c_nth_roots_build. It
parses the rendered question text and answer_text and re-derives the
correct root from scratch using cmath, then verifies claimed_root**n equals
the reconstructed w -- the textbook verification method for this topic, a
different route from the generator's own divide-by-n construction.
"""

from __future__ import annotations

import cmath
import math
import re

import pytest

from mathgen import make

TOPIC = "nth_roots_of_complex_number"
SEEDS = range(70)  # >= 60 per tier

_POLAR_EASY_RE = re.compile(
    r"For w = r·cis\(θ°\) with r = (\d+), θ = (\d+)°, and n = (\d+), "
    r"find the principal \(k = 0\) square root of w in polar form"
)
_POLAR_MEDIUM_RE = re.compile(
    r"For w = r·cis\(θ°\) with r = (\d+), θ = (\d+)°, and n = (\d+), "
    r"find root k = (-?\d+) of w in polar form"
)
_CARTESIAN_HARD_RE = re.compile(
    r"The complex number w = (-?\d+i?) has n = (\d+)\. Find root k = (-?\d+) of w in polar form"
)
_ANSWER_RE = re.compile(r"^r = (-?\d+), theta = (-?\d+)$")


def _parse_cartesian(token: str) -> complex:
    """'16' -> 16+0j, '-16' -> -16+0j, '16i' -> 0+16j, '-16i' -> 0-16j."""
    if token.endswith("i"):
        return complex(0, int(token[:-1]))
    return complex(int(token), 0)


def _verify_root(w: complex, r_root: int, theta_root: int, n: int) -> bool:
    claimed_root = cmath.rect(r_root, math.radians(theta_root))
    reconstructed = claimed_root**n
    return abs(reconstructed.real - w.real) < 1e-6 and abs(reconstructed.imag - w.imag) < 1e-6


@pytest.mark.parametrize(
    "difficulty,pattern,polar",
    [
        ("easy", _POLAR_EASY_RE, True),
        ("medium", _POLAR_MEDIUM_RE, True),
    ],
)
def test_polar_given_oracle(difficulty, pattern, polar):
    tested = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        match = pattern.search(q.text)
        assert match, f"could not parse question text: {q.text!r}"
        if difficulty == "easy":
            r_w, theta_w, n = (int(g) for g in match.groups())
        else:
            r_w, theta_w, n, k = (int(g) for g in match.groups())

        w = cmath.rect(r_w, math.radians(theta_w))
        assert not (r_w == 1 and theta_w == 0), "oracle saw w = 1, which belongs to the sibling skill"

        answer_match = _ANSWER_RE.match(q.answer_text)
        assert answer_match, f"could not parse answer_text: {q.answer_text!r}"
        r_root, theta_root = (int(g) for g in answer_match.groups())

        assert _verify_root(w, r_root, theta_root, n), (
            f"seed={seed} difficulty={difficulty}: claimed root (r={r_root}, "
            f"theta={theta_root}) raised to the {n} does not reconstruct w={w}"
        )
        tested += 1
    assert tested >= 60


def test_cartesian_given_oracle():
    tested = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        match = _CARTESIAN_HARD_RE.search(q.text)
        assert match, f"could not parse question text: {q.text!r}"
        w_token, n_str, k_str = match.groups()
        n = int(n_str)

        w = _parse_cartesian(w_token)
        assert not (w.real == 1 and w.imag == 0), "oracle saw w = 1, which belongs to the sibling skill"

        answer_match = _ANSWER_RE.match(q.answer_text)
        assert answer_match, f"could not parse answer_text: {q.answer_text!r}"
        r_root, theta_root = (int(g) for g in answer_match.groups())

        assert _verify_root(w, r_root, theta_root, n), (
            f"seed={seed}: claimed root (r={r_root}, theta={theta_root}) raised "
            f"to the {n} does not reconstruct w={w} (from Cartesian token {w_token!r})"
        )
        tested += 1
    assert tested >= 60
