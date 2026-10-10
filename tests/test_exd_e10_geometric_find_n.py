"""Tests the find_n hard tier of `senior-use-the-formula-t-n-r-n-1`.

Senior-use-the-formula-t-n-r-n-1 find_n hard tier."""

from __future__ import annotations

import re

from mathgen import make

# Matches a literal term-subscript reference like "t_5 = 40" (a numbered
# term stated by index) -- but NOT the generic formula token "t_n" (no
# digits) that appears in every rendered prompt's "Using t_n = r^(n-1) x t1"
# clause.
_TERM_SUBSCRIPT_RE = re.compile(r"t_(\d+)\s*=")


def test_find_n_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("geometric_nth_term_formula", seed=seed, variant="find_n")
        assert q.answer_text not in q.distractors
        assert "which term number is it?" in q.text


def test_find_n_reachable_from_hard_source():
    q = make("geometric_nth_term_formula", seed=3, difficulty="hard", variant="find_n")
    assert q.answer_text


def test_find_n_never_prints_the_answer_as_a_term_subscript_over_1000_seeds():
    # Swept over 1000 seeds here (not sampled) because the original defect was total, not occasional
    # -- a smaller sweep could not have distinguished "fixed" from "fixed by luck".
    checked = 0
    for seed in range(1000):
        q = make("geometric_nth_term_formula", seed=seed, variant="find_n")
        for m in _TERM_SUBSCRIPT_RE.finditer(q.text):
            assert m.group(1) != q.answer_text, (
                f"seed={seed}: prompt states t_{m.group(1)} = ..., which IS the answer ({q.answer_text}): {q.text!r}"
            )
        checked += 1
    assert checked == 1000
