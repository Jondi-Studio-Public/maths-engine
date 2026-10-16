"""compare-groups-evidence: a draw whose usual wrong values collapse onto the key still gets two distractors.

These seeds raised VerificationError ("fewer than two usable distractors") while building the release question
banks: the median-difference part when the largest-value, upper-quartile and smallest-value differences equal the
key, and the exam chain's interquartile-range parts when two of their three wrong values equal the key.
"""

from __future__ import annotations

import pytest

from mathgen import make

# (topic, tier, seed): each failed before the spare distractors were added.
SEEDS = [
    ("ir2a_compare_groups_evidence", "easy", 509),
    ("ir2a_compare_groups_evidence", "easy", 1141),
    ("ir2a_compare_groups_evidence", "medium", 470),
    ("ir2a_compare_groups_evidence", "hard", 944),
    ("exd_x_ir2a_compare_groups_evidence", "exam", 771),
    ("exd_x_ir2a_compare_groups_evidence", "exam", 1560),
    ("exd_x_ir2a_compare_groups_evidence", "exam", 9549),
    ("exd_x_ir2a_compare_groups_evidence", "exam", 18320),
]


@pytest.mark.parametrize("topic,tier,seed", SEEDS)
def test_every_number_part_keeps_two_distinct_distractors(topic, tier, seed):
    question = make(topic, seed=seed, difficulty=tier)
    numeric = [p for p in question.parts if p.answer_kind == "number"]
    assert numeric
    for part in numeric:
        shown = list(part.distractors)
        assert len(shown) >= 2, part.prompt
        assert part.answer_text not in shown, part.prompt
        assert len(set(shown)) == len(shown), part.prompt
