"""Distinct-answer floors for two tiers that used to ask a handful of questions (round-2 scope section (f)).

`rearrange` had 8, 4 and 2 distinct answers on easy, medium and hard; `exd_x_s3_interp` had 2.
"""

import pytest

from mathgen import make


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rearrange_has_at_least_30_distinct_answers(difficulty):
    answers = {make("rearrange", seed=s, difficulty=difficulty).answer_text for s in range(300)}
    assert len(answers) >= 30, f"rearrange/{difficulty}: {len(answers)} distinct answers over 300 seeds"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rearrange_tiers_keep_their_step_counts(difficulty):
    """Widening the bank must not move a question between tiers: the hard tier is squared formulas only."""
    texts = {make("rearrange", seed=s, difficulty=difficulty).text for s in range(300)}
    squared = {t for t in texts if "²" in t}
    assert (squared == texts) if difficulty == "hard" else not squared


def test_exam_interpolation_has_at_least_30_distinct_answers():
    answers = {make("exd_x_s3_interp", seed=s, difficulty="exam").answer_text for s in range(100)}
    assert len(answers) >= 30, f"exd_x_s3_interp/exam: {len(answers)} distinct answers over 100 seeds"
