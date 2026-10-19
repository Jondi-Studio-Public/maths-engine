"""``uc2-forced-oscillator-resonance``: the generators grade their own answers right and every distractor wrong, tiers differ, exam parts chain."""

from __future__ import annotations

import pytest

from mathgen import make

from s2c1_support import second_route  # noqa: F401  (autouse fixture)
from s2c1_support import SEEDS, check_follow_through, check_grading, check_text, questions, tier_topic

NAME = "forced_oscillator_resonance"
ALL = ("easy", "medium", "hard", "exam")


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    texts = {q.text for q in qs}
    assert len(texts) >= 12, len(texts)
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))
