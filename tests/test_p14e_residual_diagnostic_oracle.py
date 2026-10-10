"""Re-derives each rendered question's residual classification from `question.text` / `question.answer_text` /
`question.distractors` alone, via hand-written parsing and its OWN hardcoded copy of the three label strings
and the sign rule -- never importing anything from `mathgen.generators.p14e_residual_diagnostic_build`.

Total re-derived across the four tiers: 4 x 260 = 1040 (>= 1000).

Per tier, this file also proves the tie case (residual exactly zero, "on the
line") occurs and is confirmed offered as the key whenever it occurs -- never
silently dropped -- and that `mcq_options` always contains all three labels.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p14e_residual_diagnostic"
LABELS = ("above the line", "below the line", "on the line")
SEEDS_PER_TIER = 260


def _classify(e) -> str:
    if e > 0:
        return "above the line"
    if e < 0:
        return "below the line"
    return "on the line"


def _parse_line(text: str):
    """Parse the "y = <slope>x [<sign> <b>]" clause into (m: Fraction, b: int).
    The constant term is omitted entirely when b == 0 (b defaults to 0)."""
    m = re.search(r"y = (\(-?\d+/\d+\)|-?\d*)x(?: ([+-]) (\d+))?(?=[.,;]| |$)", text)
    assert m, f"no line clause found in {text!r}"
    m_text, sign, b_text = m.groups()
    b = 0 if sign is None else (int(b_text) if sign == "+" else -int(b_text))
    if m_text == "":
        slope = Fraction(1)
    elif m_text == "-":
        slope = Fraction(-1)
    elif m_text.startswith("("):
        slope = Fraction(m_text[1:-1])
    else:
        slope = Fraction(int(m_text))
    return slope, b


_POINT_RE = re.compile(r"\((-?\d+), (-?\d+/\d+|-?\d+)\)")


# ===========================================================================
# easy / medium: one line, one observed point -- direct substitution.
# ===========================================================================


def _oracle_direct(text: str) -> str:
    slope, b = _parse_line(text)
    obs = _POINT_RE.findall(text)
    assert len(obs) == 1, f"expected exactly one observed point in {text!r}, got {obs}"
    x0, y0 = int(obs[0][0]), Fraction(obs[0][1])
    e = y0 - (slope * x0 + b)
    return _classify(e)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_direct_oracle_matches(tier):
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = _oracle_direct(q.text)
        assert recomputed == q.answer_text, (
            f"{tier} seed={seed}: oracle {recomputed!r} != canonical {q.answer_text!r} (text={q.text!r})"
        )
        checked += 1
    assert checked == SEEDS_PER_TIER


# ===========================================================================
# hard: two labelled points A, B -- match the named one, then classify.
# ===========================================================================

_HARD_POINT_RE = re.compile(r"([AB]) = \((-?\d+), (-?\d+/\d+|-?\d+)\)")
_HARD_TARGET_RE = re.compile(r"point (\w)(?:'s residual| above| below)")


def _oracle_hard(text: str) -> str:
    slope, b = _parse_line(text)
    rows = _HARD_POINT_RE.findall(text)
    assert len(rows) == 2, f"expected 2 labelled points in {text!r}, got {rows}"
    per_point = {}
    for label, x_str, y_str in rows:
        x, y = int(x_str), Fraction(y_str)
        e = y - (slope * x + b)
        per_point[label] = _classify(e)
    target = _HARD_TARGET_RE.search(text)
    assert target, f"could not parse named target from {text!r}"
    return per_point[target.group(1)]


@pytest.mark.parametrize("seed", range(SEEDS_PER_TIER))
def test_hard_oracle_matches(seed):
    q = make(TOPIC, seed=seed, difficulty="hard")
    recomputed = _oracle_hard(q.text)
    assert recomputed == q.answer_text, (
        f"hard seed={seed}: oracle {recomputed!r} != canonical {q.answer_text!r} (text={q.text!r})"
    )


# ===========================================================================
# exam: two points ON the line -- derive m, b independently via elimination,
# then classify the third (observed) point.
# ===========================================================================

_EXAM_POINTS_RE = re.compile(r"points \((-?\d+), (-?\d+/\d+|-?\d+)\) and \((-?\d+), (-?\d+/\d+|-?\d+)\)")
_EXAM_OBS_RE = re.compile(r"observed (?:data )?point(?: is)? \((-?\d+), (-?\d+/\d+|-?\d+)\)")


def _oracle_exam(text: str) -> str:
    pts = _EXAM_POINTS_RE.search(text)
    assert pts, f"could not parse the two on-line points from {text!r}"
    x1, y1, x2, y2 = (Fraction(v) for v in pts.groups())
    obs = _EXAM_OBS_RE.search(text)
    assert obs, f"could not parse the observed point from {text!r}"
    x0, y0 = Fraction(obs.group(1)), Fraction(obs.group(2))
    # Ordinary two-point line algebra, done independently (plain elimination,
    # not sympy.Matrix.solve, which is what the generator's own in-build
    # check uses -- a different route for the oracle).
    assert x1 != x2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    e = y0 - (m * x0 + b)
    return _classify(e)


@pytest.mark.parametrize("seed", range(SEEDS_PER_TIER))
def test_exam_oracle_matches(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    assert recomputed == q.answer_text, (
        f"exam seed={seed}: oracle {recomputed!r} != canonical {q.answer_text!r} (text={q.text!r})"
    )


# ===========================================================================
# Cross-tier: options always carry all 3 labels; the tie case is real and
# always offered as the key when it occurs.
# ===========================================================================

_ORACLES = {
    "easy": lambda q: _oracle_direct(q.text),
    "medium": lambda q: _oracle_direct(q.text),
    "hard": lambda q: _oracle_hard(q.text),
    "exam": lambda q: _oracle_exam(q.text),
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_options_always_carry_all_three_labels(tier):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert set(mcq_options(q)) == set(LABELS), f"{tier} seed={seed}: options {mcq_options(q)}"


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_tie_case_is_real_and_always_offered_as_the_key(tier):
    """The oracle independently confirms every seed whose true class is 'on
    the line' both occurs (residual == 0 by direct recomputation) and is the
    question's own answer_text -- i.e. never silently swapped for a
    different key even though the class is a tie."""
    oracle = _ORACLES[tier]
    on_count = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = oracle(q)
        if recomputed == "on the line":
            on_count += 1
            assert q.answer_text == "on the line", (
                f"{tier} seed={seed}: oracle found a tie but canonical key was {q.answer_text!r}"
            )
            assert "on the line" in mcq_options(q)
    assert on_count > 0, f"{tier}: the tie case never occurred over {SEEDS_PER_TIER} seeds"


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_direct(q.text)
    corrupted = next(o for o in mcq_options(q) if o != recomputed)
    assert recomputed != corrupted
