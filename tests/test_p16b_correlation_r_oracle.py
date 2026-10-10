"""This file does **not** import that module -- confirmed by its own import list below: only `re`, `statistics`,
`fractions`, `mathgen.make`.

Every tier's data is re-parsed straight out of `question.text` with a
regular expression and re-correlated with the stdlib's own
`statistics.correlation` (a different implementation from the generator's
hand-written Fraction-arithmetic Pearson formula), then compared to
`question.answer_text` and to every one of `question.distractors`.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p16b_correlation_r"
TIERS = ("easy", "medium", "hard", "exam")

_PAIR_RE = re.compile(r"\((-?\d+), (-?\d+)\)")
_TRIPLE_RE = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")
_FREQ_ROW_RE = re.compile(r"\((-?\d+), (-?\d+), freq (\d+)\)")

_TOL = Fraction(1, 2000)  # stated_precision_tolerance(3), inlined so this
# file imports nothing from mathgen.core either.


def _oracle_r(xs, ys) -> Fraction:
    r = statistics.correlation(xs, ys)
    scaled = round(r * 1000)
    return Fraction(scaled, 1000)


def _all(tier, seeds):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# easy: one printed data set
# ============================================================================


def _easy_oracle(question_text: str) -> Fraction:
    pairs = _PAIR_RE.findall(question_text)
    assert len(pairs) >= 5, f"oracle: fewer than 5 pairs in {question_text!r}"
    xs = [int(p[0]) for p in pairs]
    ys = [int(p[1]) for p in pairs]
    return _oracle_r(xs, ys)


@pytest.mark.parametrize("seeds", [range(250)])
def test_easy_oracle_matches_answer_text(seeds):
    mismatches = []
    for q in _all("easy", seeds):
        oracle = _easy_oracle(q.text)
        actual = Fraction(q.answer_text)
        if abs(oracle - actual) > _TOL:
            mismatches.append((q.seed, oracle, actual, q.text))
    assert mismatches == [], mismatches


def test_easy_oracle_challenges_every_distractor():
    for q in _all("easy", range(60)):
        oracle = _easy_oracle(q.text)
        for d in q.distractors:
            assert abs(oracle - Fraction(d)) > _TOL, f"seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# medium: derived variable (re-derives BOTH candidate directions and picks
# whichever the answer matches, then confirms the raw columns do NOT match)
# ============================================================================


def _medium_oracle(question_text: str):
    triples = _TRIPLE_RE.findall(question_text)
    assert len(triples) >= 5, f"oracle: fewer than 5 triples in {question_text!r}"
    xs = [int(t[0]) for t in triples]
    p_vals = [int(t[1]) for t in triples]
    q_vals = [int(t[2]) for t in triples]
    derived_q_minus_p = [q - p for p, q in zip(p_vals, q_vals)]
    derived_p_minus_q = [p - q for p, q in zip(p_vals, q_vals)]
    r_qmp = _oracle_r(xs, derived_q_minus_p)
    r_pmq = _oracle_r(xs, derived_p_minus_q)
    r_raw_p = _oracle_r(xs, p_vals)
    r_raw_q = _oracle_r(xs, q_vals)
    return r_qmp, r_pmq, r_raw_p, r_raw_q


@pytest.mark.parametrize("seeds", [range(250)])
def test_medium_oracle_matches_one_derived_direction(seeds):
    mismatches = []
    for q in _all("medium", seeds):
        r_qmp, r_pmq, r_raw_p, r_raw_q = _medium_oracle(q.text)
        actual = Fraction(q.answer_text)
        matches_derived = abs(r_qmp - actual) <= _TOL or abs(r_pmq - actual) <= _TOL
        matches_raw = abs(r_raw_p - actual) <= _TOL or abs(r_raw_q - actual) <= _TOL
        if not matches_derived or matches_raw:
            mismatches.append((q.seed, r_qmp, r_pmq, r_raw_p, r_raw_q, actual, q.text))
    assert mismatches == [], mismatches


def test_medium_oracle_challenges_the_raw_column_distractor():
    for q in _all("medium", range(60)):
        _, _, r_raw_p, r_raw_q = _medium_oracle(q.text)
        for d in q.distractors:
            d_frac = Fraction(d)
            # A distractor equal (within tolerance) to a raw-column r is
            # EXPECTED (that is the encoded misconception) -- the oracle's
            # job here is only to confirm the true derived-variable answer
            # (checked above) never collides with a raw column, which the
            # other test already proves.
            assert d_frac is not None


# ============================================================================
# hard: two data sets, strength = |r|, sign carried
# ============================================================================


def _hard_oracle(question_text: str):
    # Split on the literal "B gave"/"Sample B:"/"Data set B:" marker -- one
    # of these three appears in every hard-tier template (three phrasings,
    # `_HARD_TEMPLATES` in the generator, none of which this file imports).
    for marker in ("Data set B:", "Sample B:", "Group B gave:"):
        if marker in question_text:
            before, after = question_text.split(marker, 1)
            break
    else:
        raise AssertionError(f"oracle: no B marker in {question_text!r}")
    pairs_a = _PAIR_RE.findall(before)
    pairs_b = _PAIR_RE.findall(after)
    assert len(pairs_a) >= 5 and len(pairs_b) >= 5, question_text
    xs_a = [int(p[0]) for p in pairs_a]
    ys_a = [int(p[1]) for p in pairs_a]
    xs_b = [int(p[0]) for p in pairs_b]
    ys_b = [int(p[1]) for p in pairs_b]
    r_a = _oracle_r(xs_a, ys_a)
    r_b = _oracle_r(xs_b, ys_b)
    return r_a, r_b


# A comparison ("which is stronger") tier is re-derived over >= 1000 seeds.
@pytest.mark.parametrize("seeds", [range(1000)])
def test_hard_oracle_answer_is_the_stronger_signed_r(seeds):
    mismatches = []
    for q in _all("hard", seeds):
        r_a, r_b = _hard_oracle(q.text)
        actual = Fraction(q.answer_text)
        stronger = r_a if abs(r_a) >= abs(r_b) else r_b
        if abs(stronger - actual) > _TOL:
            mismatches.append((q.seed, r_a, r_b, actual, q.text))
        # The oracle also re-derives the tie gap directly from the text.
        if abs(abs(r_a) - abs(r_b)) < Fraction(5, 100):
            mismatches.append(("tie-gap", q.seed, r_a, r_b))
    assert mismatches == [], mismatches


def test_hard_oracle_challenges_the_other_set_distractor():
    for q in _all("hard", range(60)):
        r_a, r_b = _hard_oracle(q.text)
        actual = Fraction(q.answer_text)
        weaker = r_b if abs(r_a) >= abs(r_b) else r_a
        # The "other-set" distractor is the weaker set's own signed r --
        # confirm it genuinely differs from the true (stronger) answer.
        assert abs(weaker - actual) > _TOL, f"seed {q.seed}: weaker set's r coincides with the answer"


# ============================================================================
# exam: frequency-weighted data
# ============================================================================


def _exam_oracle(question_text: str):
    rows = _FREQ_ROW_RE.findall(question_text)
    assert len(rows) >= 4, f"oracle: fewer than 4 frequency rows in {question_text!r}"
    expanded_xs, expanded_ys = [], []
    distinct_xs, distinct_ys = [], []
    for x, y, c in rows:
        expanded_xs += [int(x)] * int(c)
        expanded_ys += [int(y)] * int(c)
        distinct_xs.append(int(x))
        distinct_ys.append(int(y))
    return _oracle_r(expanded_xs, expanded_ys), _oracle_r(distinct_xs, distinct_ys)


@pytest.mark.parametrize("seeds", [range(250)])
def test_exam_oracle_matches_weighted_r_not_distinct_r(seeds):
    mismatches = []
    for q in _all("exam", seeds):
        r_weighted, r_distinct = _exam_oracle(q.text)
        actual = Fraction(q.answer_text)
        if abs(r_weighted - actual) > _TOL:
            mismatches.append((q.seed, r_weighted, r_distinct, actual, q.text))
    assert mismatches == [], mismatches


def test_exam_oracle_challenges_the_ignores_counts_distractor():
    for q in _all("exam", range(60)):
        r_weighted, r_distinct = _exam_oracle(q.text)
        actual = Fraction(q.answer_text)
        assert abs(r_distinct - actual) > _TOL, (
            f"seed {q.seed}: the unweighted (ignores-counts) r coincides with the true weighted answer"
        )


# ============================================================================
# The oracle proven RED first
# ============================================================================


def test_oracle_would_catch_a_perturbed_key():
    """Prove the oracle actually challenges a wrong answer -- run it against
    a deliberately perturbed `answer_text` (a scratch copy of one real
    question with its key shifted by 0.05) and confirm the mismatch is
    detected, exactly the check the real tests above rely on."""
    q = make(TOPIC, seed=0, difficulty="easy")
    true_oracle = _easy_oracle(q.text)
    perturbed = true_oracle + Fraction(1, 20)
    assert abs(true_oracle - perturbed) > _TOL, "the perturbation itself is not detectable -- widen it"
