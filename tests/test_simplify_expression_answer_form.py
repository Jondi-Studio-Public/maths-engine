"""`simplify_expression` is graded as an expanded answer: restating the bracketed question is not a simplification."""

from __future__ import annotations

from mathgen import make
from mathgen.grading import GRADERS


def test_restating_the_question_is_rejected_and_the_key_accepted():
    grade = GRADERS["expression"]
    for tier in ("easy", "medium", "hard"):
        for seed in range(80):
            q = make("simplify_expression", seed=seed, difficulty=tier)
            assert q.answer_form == "expanded", (tier, seed)
            assert grade(q, q.answer_text).correct, (tier, seed, q.answer_text)
            pasted = q.text.removeprefix("Simplify").strip()
            assert not grade(q, pasted).correct, (tier, seed, q.text)
