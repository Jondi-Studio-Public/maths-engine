"""Real-grader check for the two-variable hard tier."""

from __future__ import annotations

from mathgen.grading.typed import grade_expression
from mathgen import make


def test_real_grader_accepts_canonical_and_rejects_distractors():
    for seed in range(60):
        q = make("difference_of_squares", seed=seed, kind="two_var")
        result = grade_expression(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical answer rejected: {q.answer_text!r}"
        for distractor in q.distractors:
            result = grade_expression(q, distractor)
            assert not result.correct, f"seed={seed}: distractor wrongly accepted: {distractor!r}"
