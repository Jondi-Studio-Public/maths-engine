"""Tests the `single_fraction` answer form in `mathgen/grading/typed.py`, which accepts a single combined
fraction and rejects an unsimplified sum.

`answer_form="single_fraction"` in `mathgen/grading/typed.py`.

Two measurements motivate this form:

1. `answer_form="expanded"` rejects the canonical "(21*x - 1)/6" against
   itself -- its bracket check only strips a bracket spanning the WHOLE
   string, and this text ends in "6", not ")".
2. sympy parses "(5*x+1)/2 + (3*x-2)/3" and "(21*x-1)/6" to the IDENTICAL
   internal object, so no check on the PARSED expression can ever tell an
   unmerged sum of two fraction terms from one combined fraction -- only a
   raw-text, bracket-depth-aware scan can.

Every adversarial case below is asserted against `_single_fraction_flagged_raw`
directly (the exact function under test) and against `grade_expression`
end-to-end (so a regression in either the wiring or the bracket-depth logic
is caught).
"""

from __future__ import annotations

import types

from mathgen.core import Question

from mathgen.grading import typed as grading
from mathgen.grading.typed import _single_fraction_flagged_raw, grade_expression


def _q(**kwargs) -> Question:
    defaults = dict(
        topic="test",
        level=0,
        seed=1,
        text="q",
        latex="q",
        answer=None,
        answer_text="",
        answer_latex="",
        working=[],
        difficulty=None,
        distractors=[],
    )
    defaults.update(kwargs)
    return Question(**defaults)


# -- Decision 1's two motivating probes, re-verified (not re-derived) -------


def test_probe_expanded_false_rejects_the_canonical_single_fraction():
    """`answer_form="expanded"` rejects "(21*x - 1)/6" typed back at itself --
    the bug this whole module exists to work around."""
    q = types.SimpleNamespace(
        answer_text="(21*x - 1)/6",
        answer="(21*x - 1)/6",
        answer_form="expanded",
        answer_alt=None,
    )
    result = grade_expression(q, "(21*x - 1)/6")
    assert result.correct is False


def test_probe_sympy_parses_unmerged_sum_and_single_fraction_identically():
    from mathgen.grading.typed import _parse_expression

    unmerged = _parse_expression("(5*x + 1)/2 + (3*x - 2)/3")
    combined = _parse_expression("(21*x - 1)/6")
    assert unmerged == combined


# -- adversarial cases, `_single_fraction_flagged_raw` directly -------------


def test_flagged_raw_accepts_canonical_single_fraction():
    assert _single_fraction_flagged_raw("(21*x - 1)/6") is False


def test_flagged_raw_rejects_unmerged_sum():
    assert _single_fraction_flagged_raw("(5*x + 1)/2 + (3*x - 2)/3") is True


def test_flagged_raw_accepts_leading_unary_minus():
    assert _single_fraction_flagged_raw("-(21*x - 1)/6") is False


def test_flagged_raw_accepts_unary_minus_immediately_after_open_paren():
    assert _single_fraction_flagged_raw("(1 - 21*x)/(-6)") is False


def test_flagged_raw_treats_minus_after_caret_as_unary():
    # The "-" in "x^-2" is never mistaken for a binary operator -- no other
    # +/- in the string, so the whole thing is not flagged.
    assert _single_fraction_flagged_raw("3*x^-2") is False


def test_grade_expression_single_fraction_accepts_minus_after_caret():
    q = _q(answer="3*x**-2", answer_text="3*x^-2", answer_form="single_fraction")
    r = grading.grade("expression", q, "3*x^-2")
    assert r.correct is True


def test_flagged_raw_accepts_leading_whitespace_before_unary_minus():
    # `stripped` is not itself whitespace-trimmed (same caveat
    # `_strip_redundant_enclosing_brackets` documents) -- a leading "-"
    # preceded only by spaces is still the first non-space character, so it
    # must stay unary, not raise on `prev_non_space is None`.
    assert _single_fraction_flagged_raw(" -(21*x - 1)/6") is False


def test_grade_expression_single_fraction_accepts_leading_whitespace_unary_minus():
    q = _q(answer="-(21*x - 1)/6", answer_text="-(21*x - 1)/6", answer_form="single_fraction")
    r = grading.grade("expression", q, " -(21*x - 1)/6")
    assert r.correct is True


def test_flagged_raw_rejects_unmerged_sum_with_leading_whitespace():
    assert _single_fraction_flagged_raw("  (5*x + 1)/2 + 3") is True


def test_flagged_raw_still_evaluates_a_later_plus_on_its_own_merits():
    # The "-" in "x^-1" is correctly classified as unary (does not itself
    # flag), but the "+ 3" that follows is a genuine depth-0 binary "+" and
    # is still flagged on its own merits.
    assert _single_fraction_flagged_raw("2*x^-1 + 3") is True


def test_flagged_raw_rejects_unmerged_difference():
    assert _single_fraction_flagged_raw("(5*x + 1)/2 - (3*x - 2)/3") is True


def test_flagged_raw_accepts_monomial_fraction_with_no_plus_minus():
    assert _single_fraction_flagged_raw("4*x/9") is False


def test_flagged_raw_rejects_fraction_plus_bare_integer():
    assert _single_fraction_flagged_raw("(5*x + 1)/2 + 3") is True


# -- the same adversarial cases through `grade_expression` end-to-end -------


def test_grade_expression_single_fraction_accepts_canonical():
    q = _q(answer="(21*x - 1)/6", answer_text="(21*x - 1)/6", answer_form="single_fraction")
    r = grading.grade("expression", q, "(21*x - 1)/6")
    assert r.correct is True


def test_grade_expression_single_fraction_rejects_unmerged_sum():
    q = _q(answer="(21*x - 1)/6", answer_text="(21*x - 1)/6", answer_form="single_fraction")
    r = grading.grade("expression", q, "(5*x + 1)/2 + (3*x - 2)/3")
    assert r.correct is False


def test_grade_expression_single_fraction_accepts_leading_unary_minus():
    q = _q(answer="-(21*x - 1)/6", answer_text="-(21*x - 1)/6", answer_form="single_fraction")
    r = grading.grade("expression", q, "-(21*x - 1)/6")
    assert r.correct is True


def test_grade_expression_single_fraction_accepts_unary_minus_after_open_paren():
    q = _q(answer="(1 - 21*x)/(-6)", answer_text="(1 - 21*x)/(-6)", answer_form="single_fraction")
    r = grading.grade("expression", q, "(1 - 21*x)/(-6)")
    assert r.correct is True


def test_grade_expression_single_fraction_rejects_unmerged_difference():
    q = _q(answer="(x - 3)/6", answer_text="(x - 3)/6", answer_form="single_fraction")
    r = grading.grade("expression", q, "(5*x + 1)/2 - (3*x - 2)/3")
    assert r.correct is False


def test_grade_expression_single_fraction_accepts_monomial_with_no_plus_minus():
    q = _q(answer="4*x/9", answer_text="4*x/9", answer_form="single_fraction")
    r = grading.grade("expression", q, "4*x/9")
    assert r.correct is True


def test_grade_expression_single_fraction_rejects_fraction_plus_bare_integer():
    q = _q(answer="(5*x + 1)/2", answer_text="(5*x + 1)/2", answer_form="single_fraction")
    r = grading.grade("expression", q, "(5*x + 1)/2 + 3")
    assert r.correct is False


# -- the √ back door is closed, same discipline as `rationalised` -----------


def test_grade_expression_single_fraction_runs_the_surd_normalisation_pass():
    # `_single_fraction_flagged_raw` calls `_normalize_surd_notation` before
    # scanning, same discipline as `rationalised`/`simplified_surd` -- this
    # particular "√" input's depth-0 "+" would still be caught without the
    # rewrite (the paren-form "√(x)" does not move any bracket depth), but
    # the spec requires the normalisation call regardless, so this asserts
    # the intended behaviour end-to-end rather than proving the rewrite is
    # load-bearing for this specific case.
    q = _q(answer="(x + 1)/2", answer_text="(x + 1)/2", answer_form="single_fraction")
    r = grading.grade("expression", q, "√(x)/2 + 3")
    assert r.correct is False
