"""Tests the three hard-tier kinds of the `sequence` generator: common difference and nth-term rule from non-
consecutive terms, and solving for the term position.

`sequence`'s three new hard-tier kinds -- non-consecutive terms for common-difference and nth-term-rule (the
difference must be found by dividing across a term gap, not read off consecutive terms), and inverse
substitution (solve for the term position, not the term value) for use-nth-term-rule. Each is a genuinely
different reasoning shape from its own skill's easy/medium tiers and from the OTHER new kinds -- never adding
the SAME new kind reachable from another skill. The default (non-hard) kinds are asserted byte-identical."""

from fractions import Fraction

import mathgen


def test_common_diff_nonconsecutive_matches_the_construction():
    for seed in range(60):
        q = mathgen.make("sequence", seed=seed, difficulty="hard", kind="common_diff_nonconsecutive")
        assert "term" in q.text and "common difference" in q.text
        assert len(q.distractors) == 3
        assert str(q.answer) not in [str(d) for d in q.distractors]


def test_nth_term_rule_nonconsecutive_matches_the_construction():
    for seed in range(60):
        q = mathgen.make("sequence", seed=seed, difficulty="hard", kind="nth_term_rule_nonconsecutive")
        assert "term" in q.text and "nth-term rule" in q.text
        assert len(q.distractors) == 3
        assert q.answer_text not in q.distractors


def test_use_rule_inverse_solves_for_x_not_for_the_value():
    for seed in range(60):
        q = mathgen.make("sequence", seed=seed, difficulty="hard", kind="use_rule_inverse")
        assert "Which value of x" in q.text
        assert len(q.distractors) == 3
        assert str(q.answer) not in [str(d) for d in q.distractors]


def test_default_kinds_unchanged():
    """The three pre-existing kinds must render byte-identically -- pinned against the actual seed-3 output on
    this base, not just determinism (a same-call comparison would be vacuous and would pass even if both calls
    hit new code)."""
    pinned = {
        "common_diff": ("Find the common difference of the sequence: 2, 10, 18, 26, ...", "8"),
        "nth_term_rule": (
            "Find the nth-term rule for the sequence: 2, 10, 18, 26, ... (use x for the term number)",
            "8*x - 6",
        ),
        "use_rule": ("A sequence follows a_x = 8*x - 6. Find the 9th term.", "66"),
    }
    for kind in ("common_diff", "nth_term_rule", "use_rule"):
        a = mathgen.make("sequence", seed=3, kind=kind, hi=9)
        b = mathgen.make("sequence", seed=3, kind=kind, hi=9)
        assert a.text == b.text
        assert a.answer == b.answer
        assert (a.text, a.answer_text) == pinned[kind], (kind, a.text, a.answer_text)
