"""Tests the reversed-sign hard tier of `bisection-interval-update`.

Bisection-interval-update reversed-sign hard tier."""

from __future__ import annotations

import re
from math import isqrt
from fractions import Fraction

from mathgen import make
from mathgen.core import PRESETS, GRADER_TOLERANCE
from mathgen.grading.typed import grade_pair

HARD_VALUE_HI = PRESETS["bisection_interval_update"]["hard"]["value_hi"]

# Regex for the rendered question text/latex: latex embeds the reversed-sign
# function as "f(x)=<c>-x^2,\\quad [a,b]=[<left>,<right>],..." -- parse both
# `c` and the INITIAL bracket from there (see Required behaviour 3d: this
# is the pre-update bracket, not `question.answer`, which is the AFTER
# state).
_LATEX_C_RE = re.compile(r"f\(x\)=(\d+)-x\^2")
_LATEX_BRACKET_RE = re.compile(r"\[a,b\]=\[(-?\d+),(-?\d+)\]")


def _analytic_left_half_fraction(value_hi: int) -> Fraction:
    """Exact probability that the hard path's answer lands in the LEFT half
    of [left, right], under the actual sampling distribution: c is drawn
    uniformly from the non-square integers in [2, value_hi] (each c equally
    likely, since squares are rejected and resampled), and THEN, for that
    c, (left, right) is drawn uniformly and independently from [0, k] x
    [k+1, k+4]. Because c is the outer, equally-weighted draw, the correct
    aggregate is the MEAN of each c's own left-half fraction, one term per
    c -- not the count of left-half pairs divided by the total pair count
    across all c (that second quantity over-weights larger c/k, which have
    more (left, right) combinations but no higher probability of being
    drawn).

    The root m = (left+right)/2 lands in the left half of [left, right]
    exactly when m > sqrt(c), i.e. (left+right)**2 > 4*c. (An earlier draft
    of this derivation used `left + right >= 2k+1` instead; that shortcut
    ignores where c sits within its [k**2, (k+1)**2) band and is wrong
    whenever `right - left` is odd -- c=8 is a counterexample: k=2, left=1,
    right=4 gives sum=5=2k+1, which the shortcut reads as "left half", but
    m=2.5, m**2=6.25 < 8, so the root is actually in the RIGHT half.)
    """
    per_c_fractions = []
    for c in range(2, value_hi + 1):
        k = isqrt(c)
        if k * k == c:
            continue
        left_count = 0
        total_for_c = 0
        for left in range(0, k + 1):
            for right in range(k + 1, k + 5):
                total_for_c += 1
                if (left + right) ** 2 > 4 * c:
                    left_count += 1
        per_c_fractions.append(Fraction(left_count, total_for_c))
    return sum(per_c_fractions, Fraction(0)) / len(per_c_fractions)


def test_reversed_sign_template_puts_constant_before_x_squared():
    for seed in range(60):
        question = make("bisection_interval_update", seed=seed, difficulty="hard", reversed_sign=True)
        assert "−" in question.text
        assert "x²" in question.text
        # The reversed-sign template is "c − x²" (constant first); the
        # default template is "x² − c" (x² first). Assert the reversed
        # order literally, rather than the vacuous `or True` this test
        # started with.
        assert question.text.index("−") < question.text.index("x²")


def test_default_template_puts_x_squared_before_constant():
    for seed in range(60):
        question = make("bisection_interval_update", seed=seed, difficulty="hard")
        assert question.text.index("x²") < question.text.index("−")


def test_reversed_sign_still_correctly_brackets_the_root():
    # Both conventions must keep a HALF-WIDTH bracket that still contains
    # the root -- this is what the generator's own `check()` calls verify
    # at generation time; re-assert it here as a behavioural guarantee.
    for seed in range(60):
        question = make("bisection_interval_update", seed=seed, difficulty="hard", reversed_sign=True)
        left = Fraction(question.answer["left"])
        right = Fraction(question.answer["right"])
        assert right > left


def test_reversed_sign_distractors_clear_the_grader_by_construction():
    for seed in range(60):
        question = make("bisection_interval_update", seed=seed, difficulty="hard", reversed_sign=True)
        for distractor in question.distractors:
            assert distractor != question.answer_text


# These four functions are new; the four above are untouched.


def test_reversed_sign_both_halves_appear_over_600_seeds():
    """The hard path (`reversed_sign=True`) must land in EITHER half of the bracket, not always the right half
    (the defect this fixes). Classify each seed's answer against the INITIAL bracket parsed
    from `question.latex` and check the aggregate split against the exact analytic derivation above (~33.36%
    left / 66.64% right for `value_hi=80`, with right drawn from [k+1,k+4]: the narrowest range meeting the
    30/70 target; [k+1,k+3] gave ~25.48%).
    """
    left_count = 0
    right_count = 0
    for seed in range(600):
        question = make("bisection_interval_update", seed=seed, value_hi=HARD_VALUE_HI, reversed_sign=True)
        c_match = _LATEX_C_RE.search(question.latex)
        bracket_match = _LATEX_BRACKET_RE.search(question.latex)
        assert c_match and bracket_match
        left0 = Fraction(int(bracket_match.group(1)))
        right0 = Fraction(int(bracket_match.group(2)))
        midpoint = (left0 + right0) / 2
        ans_left = Fraction(question.answer["left"])
        ans_right = Fraction(question.answer["right"])
        if ans_right == midpoint:
            left_count += 1
        elif ans_left == midpoint:
            right_count += 1
        else:  # pragma: no cover - defensive, would indicate a parsing bug
            raise AssertionError(
                f"seed={seed}: answer {ans_left, ans_right} does not match either half of [{left0}, {right0}]"
            )
    assert left_count > 0
    assert right_count > 0
    assert left_count + right_count == 600

    expected_fraction = _analytic_left_half_fraction(HARD_VALUE_HI)
    assert expected_fraction == Fraction(15131, 45360)
    assert expected_fraction >= Fraction(3, 10)  # the record's 30/70 target
    # Binomial sd for n=600, p~=0.3336 is ~11.5; allow a generous ~+/-5.6sd (65)
    # band around the analytic expectation (~200) to avoid flakiness from the
    # deterministic seed-0..599 sample while still catching a broken split.
    # The record's 30/70 target, on the rendered sample itself: the band below
    # is too wide to tell [k+1,k+4] (217 here) from the old [k+1,k+3] (167).
    assert left_count >= 180, f"left_count={left_count} is under 30% of 600"
    expected_count = round(600 * expected_fraction)
    assert abs(left_count - expected_count) <= 65, (
        f"observed left_count={left_count}, analytic expected~{expected_count} "
        f"(fraction {float(expected_fraction):.4f})"
    )


def test_default_reversed_sign_false_hard_tier_still_right_half_only_over_600_seeds():
    """Control: the default path (`reversed_sign=False`, explicit) at the
    hard tier's own `value_hi` must stay untouched -- still 600/0 right-half
    only, proving the fix did not perturb the default path's RNG stream or
    output."""
    right_count = 0
    for seed in range(600):
        question = make("bisection_interval_update", seed=seed, value_hi=HARD_VALUE_HI, reversed_sign=False)
        c_match = re.search(r"f\(x\)=x\^2-(\d+)", question.latex)
        bracket_match = _LATEX_BRACKET_RE.search(question.latex)
        assert c_match and bracket_match
        left0 = Fraction(int(bracket_match.group(1)))
        right0 = Fraction(int(bracket_match.group(2)))
        midpoint = (left0 + right0) / 2
        ans_left = Fraction(question.answer["left"])
        ans_right = Fraction(question.answer["right"])
        assert ans_left == midpoint, (
            f"seed={seed}: default path must always keep [midpoint, right] "
            f"(the right half); got [{ans_left}, {ans_right}] against "
            f"[{left0}, {right0}]"
        )
        right_count += 1
    assert right_count == 600


def test_reversed_sign_distractors_clear_the_grader_over_600_seeds():
    """Assert against that real behaviour directly."""
    for seed in range(600):
        question = make("bisection_interval_update", seed=seed, value_hi=HARD_VALUE_HI, reversed_sign=True)
        for distractor in question.distractors:
            result = grade_pair(question, distractor)
            assert not result.correct, (
                f"seed={seed}: distractor {distractor!r} is graded correct "
                f"against answer {question.answer_text!r} "
                f"(tolerance={float(GRADER_TOLERANCE)})"
            )


def test_reversed_sign_root_bracketed_parsed_from_rendered_text_over_600_seeds():
    """3d: independent of the generator's own `check` calls, parse the INITIAL `c`, `left`, `right` out of the
    rendered question and assert the root really is bracketed: left**2 < c < right**2."""
    for seed in range(600):
        question = make("bisection_interval_update", seed=seed, value_hi=HARD_VALUE_HI, reversed_sign=True)
        c_match = _LATEX_C_RE.search(question.latex)
        bracket_match = _LATEX_BRACKET_RE.search(question.latex)
        assert c_match and bracket_match, f"seed={seed}: could not parse c/bracket from latex {question.latex!r}"
        c = Fraction(int(c_match.group(1)))
        left0 = Fraction(int(bracket_match.group(1)))
        right0 = Fraction(int(bracket_match.group(2)))
        assert left0**2 < c < right0**2, f"seed={seed}: [{left0}, {right0}] does not bracket sqrt({c})"
