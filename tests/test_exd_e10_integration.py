"""Integrate-polynomial/integrate-trig/ integration-by-substitution hard tiers."""

from __future__ import annotations

import re

from mathgen.grading.typed import grade_expression
from mathgen import make

_NONLINEAR_TEXT_RE = re.compile(
    r"∫ \((?:(\d+)\*)?x\*\(x\*\*2 \+ (\d+)\)\*\*(\d+)\) dx using the substitution u = x\*\*2 \+ \d+"
)


def test_rewrite_condition_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("integration", seed=seed, kinds=("rewrite_condition",))
        assert "F(1)" not in q.text or "F(x)" in q.text
        assert q.answer_text not in q.distractors
        assert "+ C" not in q.answer_text  # particular antiderivative, no free constant


def test_phase_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("integration", seed=seed, kinds=("phase",))
        assert q.answer_text.endswith("+ C")
        assert q.answer_text not in q.distractors


def test_nonlinear_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("integration", seed=seed, kinds=("nonlinear",))
        assert q.answer_text.endswith("+ C")
        assert q.answer_text not in q.distractors


def test_all_three_reachable_from_hard_sources():
    p = make("integration", seed=1, difficulty="hard", kinds=("rewrite_condition",))
    t = make("integration", seed=1, difficulty="hard", kinds=("phase",))
    s = make("integration", seed=1, difficulty="hard", kinds=("nonlinear",))
    assert p.answer_text and t.answer_text and s.answer_text


def test_nonlinear_u_form_answer_accepted_by_the_real_grader_over_300_seeds():
    # The u-form antiderivative a student actually derives via u-substitution -- k*(x**2+d)**(n+1) /
    # (2*(n+1)) + C, built here independently from the rendered prompt's own integers, never
    # imported from the generator -- must be ACCEPTED by the real grader.
    checked = 0
    for seed in range(300):
        q = make("integration", seed=seed, kinds=("nonlinear",))
        m = _NONLINEAR_TEXT_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        k_val = int(m.group(1)) if m.group(1) is not None else 1
        d_val, n_val = int(m.group(2)), int(m.group(3))
        u_form_answer = f"{k_val}*(x**2 + {d_val})**{n_val + 1}/{2 * (n_val + 1)} + C"
        result = grade_expression(q, u_form_answer)
        assert result.correct, (
            f"seed={seed}: u-form antiderivative rejected by the real grader: "
            f"{u_form_answer!r} (canonical={q.answer_text!r})"
        )
        checked += 1
    assert checked == 300
