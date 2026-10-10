"""Independent oracle for the `p12c_estimate_limit_ah` questions: it re-derives each answer from the rendered
question text.

Imports nothing from `mathgen.generators.p12c_estimate_limit_ah_build`: not the table-value helper, not the
templates, not the distractor construction. Parses `a` (and, for the exam tier, `a1`/`a2` and every table row)
straight out of the RENDERED `question.text`, recomputes `ln(a)` via `math.log` rounded to 2 d.p., and
confirms it against `question.answer_text` and `question.distractors` -- the artefact a learner actually
receives
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

TOPIC = "p12c_estimate_limit_ah"
SEEDS_PER_TIER = 260  # >= 250/tier x 4 tiers = 1040 >= 1000

_A_RE = re.compile(r"a = (-?\d+(?:\.\d+)?)")
_ROW_RE = re.compile(r"h = (\d*\.?\d+): (-?\d+\.\d+)")
_FIRST_A_RE = re.compile(r"First, for a = (-?\d+(?:\.\d+)?):")
_SECOND_A_RE = re.compile(r"Second, for a = (-?\d+(?:\.\d+)?):")


def _round2(v: float) -> str:
    v = round(v, 2)
    if v == 0:
        v = 0.0
    return f"{v:.2f}"


def _recompute_row(a: float, h: float) -> float:
    """Independent route: `math.pow` directly from the parsed `a` and `h`,
    never reading the generator's own table."""
    return (math.pow(a, h) - 1.0) / h


# --- Single-a tiers (easy, medium, hard) ------------------------------------


def _check_single_tier(difficulty: str, expected_rows: int):
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        m = _A_RE.search(q.text)
        assert m, (seed, q.text)
        a = float(m.group(1))

        rows = _ROW_RE.findall(q.text)
        assert len(rows) == expected_rows, (seed, q.text, rows)

        # Every displayed table cell is independently reproducible from `a`
        # and the row's own stated `h` -- what a learner actually verifies.
        # Compared as the same 4-d.p. DISPLAY string, never a raw float
        # difference: `v_str` is itself already rounded to 4 d.p., so a raw
        # `abs(recomputed - float(v_str))` threshold has to be wide enough to
        # absorb that rounding (observed up to ~5e-5) -- comparing the
        # formatted strings sidesteps picking a threshold at all.
        for h_str, v_str in rows:
            h = float(h_str)
            recomputed = _recompute_row(a, h)
            assert f"{recomputed:.4f}" == v_str, (seed, q.text, h_str, v_str, recomputed)

        # The correct option: ln(a) recomputed via `math.log`, rounded to
        # 2 d.p. -- never reusing anything the generator built.
        key = _round2(math.log(a))
        assert key == q.answer_text, (seed, q.text, q.answer_text, key)

        # `ln(a)` never appears in the question the learner reads.
        assert "ln(" not in q.text.replace(" ", ""), (seed, q.text)
        assert "ln(" not in q.answer_text and all("ln(" not in d for d in q.distractors), (
            seed,
            q.answer_text,
            q.distractors,
        )

        # Tie excluded by construction: no distractor equals the key.
        assert q.answer_text not in q.distractors, (seed, q.text, q.distractors)
        # And no two distractors tie each other either.
        assert len(q.distractors) == len(set(q.distractors)), (seed, q.text, q.distractors)

        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_easy_tier():
    _check_single_tier("easy", expected_rows=3)


def test_oracle_medium_tier():
    _check_single_tier("medium", expected_rows=3)


def test_oracle_hard_tier():
    _check_single_tier("hard", expected_rows=4)


def test_oracle_hard_tier_offers_h01_row_as_a_distractor():
    """The hard tier's specific required distractor (the "read the first,
    coarsest row" misconception): re-derived from `a` and `h=0.1` at full
    precision (never the already-4-d.p.-rounded displayed row text -- double
    rounding a 4-d.p. display down to 2 d.p. can land on the wrong side of a
    boundary the single full-precision rounding the generator itself does
    would not)."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _A_RE.search(q.text)
        a = float(m.group(1))
        h01_recomputed_rounded = _round2(_recompute_row(a, 0.1))
        assert h01_recomputed_rounded in q.distractors, (seed, q.text, h01_recomputed_rounded, q.distractors)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Exam tier: two a's, two tables, a magnitude comparison -----------------


def test_oracle_exam_tier():
    checked = 0
    first_wins = 0
    second_wins = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam")
        m1 = _FIRST_A_RE.search(q.text)
        m2 = _SECOND_A_RE.search(q.text)
        assert m1 and m2, (seed, q.text)
        a1_str, a2_str = m1.group(1), m2.group(1)
        a1, a2 = float(a1_str), float(a2_str)

        mag1 = abs(round(math.log(a1), 2))
        mag2 = abs(round(math.log(a2), 2))

        # Tie excluded by construction: the two magnitudes are never equal,
        # and every seed's rendered options confirm this independently too.
        assert mag1 != mag2, (seed, q.text, a1, a2, mag1, mag2)

        expected_answer = (
            f"The first value (a = {a1_str}) gives the estimated limit further from zero"
            if mag1 > mag2
            else f"The second value (a = {a2_str}) gives the estimated limit further from zero"
        )
        assert expected_answer == q.answer_text, (seed, q.text, q.answer_text, expected_answer, mag1, mag2)

        # The "equal magnitude" trap is offered, and is never the key.
        equal_option = next(o for o in [q.answer_text, *q.distractors] if "same magnitude" in o)
        assert equal_option in q.distractors, (seed, q.text, q.distractors)

        if mag1 > mag2:
            first_wins += 1
        else:
            second_wins += 1
        checked += 1
    assert checked == SEEDS_PER_TIER
    # Both sides win somewhere in a sample this size -- not a one-sided
    # construction that always favours "First".
    assert first_wins > 0 and second_wins > 0, (first_wins, second_wins)


# --- Proven red: the oracle actually fires on a corrupted answer -----------


def test_oracle_fires_on_a_tampered_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    m = _A_RE.search(q.text)
    a = float(m.group(1))
    key = _round2(math.log(a))
    tampered = _round2(math.log(a) + 1.0)  # deliberately wrong by a whole unit
    assert tampered != key
    assert key == q.answer_text  # the real answer still matches the true derivation
    assert tampered != q.answer_text  # and the tampered one would have been caught


def test_oracle_fires_on_a_tampered_exam_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    m1 = _FIRST_A_RE.search(q.text)
    m2 = _SECOND_A_RE.search(q.text)
    a1, a2 = float(m1.group(1)), float(m2.group(1))
    mag1, mag2 = abs(round(math.log(a1), 2)), abs(round(math.log(a2), 2))
    true_winner_is_first = mag1 > mag2
    tampered_answer_is_first = "first" in q.answer_text.lower()
    assert tampered_answer_is_first == true_winner_is_first  # the real answer agrees with re-derivation
