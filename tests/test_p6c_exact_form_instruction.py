"""Wording fix: every question whose canonical answer is not exactly representable as a terminating decimal must
tell the learner it wants an exact value.

Two ways a canonical answer can fail to be a terminating decimal:
1. it is irrational-valued (names pi, log or sqrt in its displayed form) --
   never representable as any terminating decimal at all;
2. it is a rational number whose reduced denominator has a prime factor
   other than 2 or 5 -- its decimal expansion never terminates (e.g. 1/9,
   1/12), so a learner who reports a correctly-rounded decimal is typing
   something that is not exactly the canonical answer.

A rational answer whose reduced denominator has only 2s and 5s (e.g. 5/16
= 0.3125) needs no instruction: the exact decimal is finite, and the
grader accepts it as-is (confirmed directly, not assumed).

Without the instruction where it is needed, a learner who reports a
correct decimal approximation (the natural response to an un-scoped "find
the volume" prompt) is marked wrong by a grader that compares exactly --
not because they made a mistake, but because the question never said what
form it wanted. Confirmed directly against the real grader on this branch:
`senior-integrate-by-parts` (medium/hard tiers on pi/log grounds, plus a
non-terminating-rational case in its own easy tier -- 20 of 40 sampled
easy-tier questions, whenever the drawn `k` is a multiple of 3), `senior-
use-partial-fractions-where-necessary-for-integration-in` (all tiers,
log-valued), `senior-integrate-expressions-of-the-form-frac-pm1-sqrt` (all
tiers, pi-valued) and `senior-determine-volumes-of-solids-of-revolution-
about-either` (all tiers, pi-valued) all rejected a correct decimal typed
for a canonical this rule now catches.
`senior-calculate-total-change-by-integrating-instantaneous-or-marginal`
is exempt from both conditions: every tier's canonical is a plain,
terminating rational, confirmed directly over the full seed sweep below,
not assumed.

This file owns the shared rule and both generator modules it checks
(`p6c_integration_techniques_a_build.py`,
`p6c_integration_techniques_b_build.py`); it is not itself a single generator's
own test file.
"""

from __future__ import annotations

import pytest
from sympy import Rational, sympify

from mathgen import make

_IRRATIONAL_MARKERS = ("pi", "log", "sqrt")
_EXACT_FORM_PHRASE = "exact value"

# topic -> presets checked. total_change is deliberately absent from the
# "must carry" table (its own dedicated test below proves the exemption
# empirically instead of assuming it).
_TOPICS_AND_PRESETS = {
    "integrate_by_parts": ("easy", "medium", "hard"),
    "use_partial_fractions_where_necessary_for_integration_in": ("easy", "medium", "hard"),
    "integrate_expressions_of_the_form_frac_pm1_sqrt": ("easy", "medium", "hard"),
    "determine_volumes_of_solids_of_revolution_about_either": ("easy", "medium", "hard"),
}

_SEEDS = 60


def _is_non_terminating_rational(value) -> bool:
    """True if `value` is a sympy rational whose reduced denominator has a
    prime factor other than 2 or 5. False for a non-rational value (the
    irrational-marker check handles those)."""
    if not value.is_rational:
        return False
    denom = Rational(value).q
    for prime in (2, 5):
        while denom % prime == 0:
            denom //= prime
    return denom != 1


def _answer_needs_exact_form(answer_text: str) -> bool:
    """Whether a canonical answer needs an exact-form instruction: either
    its displayed form names pi/log/sqrt, or (failing that) it parses as a
    rational number that does not terminate in decimal."""
    if any(marker in answer_text for marker in _IRRATIONAL_MARKERS):
        return True
    try:
        value = sympify(answer_text.replace("^", "**"))
    except Exception:
        # Cannot parse -- conservatively assume it might need the note
        # rather than silently pass an answer this rule cannot classify.
        return True
    if not value.is_rational:
        return True
    return _is_non_terminating_rational(value)


def _has_exact_form_instruction(text: str) -> bool:
    """Whether a prompt tells the learner it wants an exact value."""
    return _EXACT_FORM_PHRASE in text


@pytest.mark.parametrize("topic", sorted(_TOPICS_AND_PRESETS))
def test_canonical_answers_needing_exact_form_carry_the_instruction(topic):
    """Every seed, every preset: if the canonical answer does not
    terminate in decimal (irrational, or a non-terminating rational), the
    prompt must say it wants an exact value. Run over the full 60-seed
    sweep per preset, not a sample -- `integrate_by_parts`'s easy tier only
    trips the non-terminating-rational branch on some seeds (whenever the
    drawn `k` is a multiple of 3), so a shallow sample could pass by
    accident even with the defect present."""
    missing = []
    for preset in _TOPICS_AND_PRESETS[topic]:
        for seed in range(_SEEDS):
            q = make(topic, seed=seed, difficulty=preset)
            if _answer_needs_exact_form(q.answer_text) and not _has_exact_form_instruction(q.text):
                missing.append(f"{topic} difficulty={preset} seed={seed}: {q.text!r} -> {q.answer_text!r}")
    assert not missing, "\n".join(missing[:10])


def test_total_change_never_needs_the_instruction():
    """Confirms the exemption is a measured fact, not an assumption: sweep
    every seed/preset of the rational-only skill and assert none of its
    canonicals -- irrational or non-terminating-rational -- ever trip the
    widened check."""
    topic = "calculate_total_change_by_integrating_instantaneous_or_marginal"
    for preset in ("easy", "medium", "hard"):
        for seed in range(_SEEDS):
            q = make(topic, seed=seed, difficulty=preset)
            assert not _answer_needs_exact_form(q.answer_text), (
                f"{topic} difficulty={preset} seed={seed}: canonical {q.answer_text!r} "
                "needs an exact-form instruction (irrational or non-terminating rational) "
                "-- the exemption no longer holds, this skill needs the instruction too"
            )


# The rule above must be able to fail, not just pass vacuously.


def test_the_rule_itself_catches_a_missing_instruction_pi_valued():
    bad_text = "Find the volume of the resulting solid."
    bad_answer = "288*pi"
    assert _answer_needs_exact_form(bad_answer)
    assert not _has_exact_form_instruction(bad_text)


def test_the_rule_itself_catches_a_missing_instruction_non_terminating_rational():
    """A plain rational like 1/9 needs the instruction too, even though it names none of pi/log/sqrt."""
    bad_text = "Find the value."
    bad_answer = "1/9"
    assert _answer_needs_exact_form(bad_answer)
    assert not _has_exact_form_instruction(bad_text)


def test_the_rule_itself_accepts_a_correctly_annotated_question():
    good_text = "Find the volume of the resulting solid. Give your answer as an exact value in terms of pi."
    good_answer = "288*pi"
    assert _answer_needs_exact_form(good_answer)
    assert _has_exact_form_instruction(good_text)


def test_the_rule_itself_does_not_flag_a_terminating_rational_missing_the_phrase():
    """A terminating rational (5/16 = 0.3125 exactly) must never be
    required to carry the instruction -- the exact decimal is finite and
    the grader accepts it as-is, so rejecting a rounded 0.31 is correct
    behaviour, not a defect this rule should flag."""
    terminating_text = "Find the value."
    terminating_answer = "5/16"
    assert not _answer_needs_exact_form(terminating_answer)


def test_non_terminating_rational_detector_matches_hand_worked_cases():
    """Direct unit coverage of `_is_non_terminating_rational` against the
    exact denominators `integrate_by_parts`'s easy tier can produce
    (k in 2..6, so k^2 in {4, 9, 16, 25, 36})."""
    terminating = [Rational(1, 4), Rational(1, 16), Rational(1, 25), Rational(3, 4)]
    non_terminating = [Rational(1, 9), Rational(2, 9), Rational(1, 12), Rational(5, 36)]
    for value in terminating:
        assert not _is_non_terminating_rational(value), f"{value} wrongly flagged as non-terminating"
    for value in non_terminating:
        assert _is_non_terminating_rational(value), f"{value} wrongly accepted as terminating"
