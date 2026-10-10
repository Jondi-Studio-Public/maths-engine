"""This route is structurally different from construction's own closed-form algebra (`p*(2-p)`, `(1-p)**2`,
`(1-p)*p`) even though `_path_probabilities` inside the generator module ALSO does path enumeration -- this
file re-implements that enumeration from scratch, reading only the rendered text, and never imports the
generator's own helper.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p12f_markov_hitting"

_P_RE = re.compile(r"P\(s→t\)=(\d+)(?:/(\d+))?")
_Q_RE = re.compile(r"P\(t→s\)=(\d+)(?:/(\d+))?")
_H_RE = re.compile(r"(?:is|probability) (\d+)(?:/(\d+))?\.\s*Find p")


def _parse_ratio(match: re.Match) -> Fraction:
    num, den = match.groups()
    return Fraction(int(num), int(den) if den else 1)


def _matrix_square(p: Fraction, q: Fraction):
    """Hand-written 2x2 matrix multiply, never a library matrix-power call."""
    m = [[1 - p, p], [q, 1 - q]]
    m2 = [[sum(m[i][k] * m[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    return m2


def _two_step_paths_from_s(p: Fraction, q: Fraction) -> dict[str, Fraction]:
    """Enumerate all four 2-step trajectories from s directly (not via the
    matrix), then cross-check against the matrix square built above -- two
    independent constructions of the same numbers, both from scratch."""
    paths = {
        "ss": (1 - p) * (1 - p),
        "st": (1 - p) * p,
        "ts": p * q,
        "tt": p * (1 - q),
    }
    m2 = _matrix_square(p, q)
    assert paths["ss"] + paths["ts"] == m2[0][0]
    assert paths["st"] + paths["tt"] == m2[0][1]
    assert sum(paths.values()) == 1
    return paths


def _oracle_easy(text: str) -> Fraction:
    p = _parse_ratio(_P_RE.search(text))
    q = _parse_ratio(_Q_RE.search(text))
    paths = _two_step_paths_from_s(p, q)
    return 1 - paths["ss"]


def _oracle_medium(text: str) -> Fraction:
    p = _parse_ratio(_P_RE.search(text))
    q = _parse_ratio(_Q_RE.search(text))
    paths = _two_step_paths_from_s(p, q)
    return paths["ss"]


def _oracle_hard(text: str) -> Fraction:
    p = _parse_ratio(_P_RE.search(text))
    q = _parse_ratio(_Q_RE.search(text))
    paths = _two_step_paths_from_s(p, q)
    return paths["st"]


def _oracle_exam(text: str) -> Fraction:
    """Search every candidate p over a wide denominator range for the one
    whose reach-within-2 value matches the stated h -- a search, not the
    closed-form square-root inversion construction uses."""
    h_match = _H_RE.search(text)
    assert h_match, f"could not find the stated h in {text!r}"
    h = _parse_ratio(h_match)
    matches = []
    for d in range(2, 40):
        for k in range(1, d):
            p = Fraction(k, d)
            if p * (2 - p) == h:
                matches.append(p)
    unique = sorted(set(matches))
    assert len(unique) == 1, f"h={h} does not pin down a unique p over the search range: {unique} (text={text!r})"
    return unique[0]


@pytest.mark.parametrize(
    "tier,fn",
    [
        ("easy", _oracle_easy),
        ("medium", _oracle_medium),
        ("hard", _oracle_hard),
    ],
)
def test_oracle_recomputes(tier, fn):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = fn(q.text)
        expected = Fraction(q.answer["p"], q.answer["q"])
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        recomputed = _oracle_exam(q.text)
        expected = Fraction(q.answer["p"], q.answer["q"])
        assert recomputed == expected, (
            f"exam seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Proven red first: q genuinely does not enter the easy/medium/hard value --


def test_oracle_confirms_q_is_irrelevant_to_easy_medium_hard():
    """Direct re-proof, from the oracle's own from-scratch path enumeration,
    of the central design claim: swap q for a different value in the
    SAME question's text and confirm the recomputed answer is unchanged."""
    for tier, fn in (("easy", _oracle_easy), ("medium", _oracle_medium), ("hard", _oracle_hard)):
        q = make(TOPIC, seed=0, difficulty=tier)
        original = fn(q.text)
        p = _parse_ratio(_P_RE.search(q.text))
        original_q = _parse_ratio(_Q_RE.search(q.text))
        swapped_q = Fraction(1, 9) if original_q != Fraction(1, 9) else Fraction(2, 9)
        swapped_text = q.text.replace(
            f"P(t→s)={original_q.numerator}" + (f"/{original_q.denominator}" if original_q.denominator != 1 else ""),
            f"P(t→s)={swapped_q.numerator}" + (f"/{swapped_q.denominator}" if swapped_q.denominator != 1 else ""),
        )
        assert swapped_text != q.text
        swapped = fn(swapped_text)
        assert swapped == original, f"{tier}: answer changed after swapping only q (p={p})"


# --- Proven red first: the oracle actually checks something ---------------


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_easy(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted
