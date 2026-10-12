"""Which sources a review draws from at each stage (mathgen/scheduling/stages.py)."""

from __future__ import annotations

import pytest

from mathgen import engine
from mathgen.scheduling import stages
from mathgen.skills import Skill, Source

N = 7  # stages 0..6, then maintenance passed as 7


def _skill(*sources):
    return Skill(id="s", name="s", domain="d", answer="number", status="active", sources=list(sources))


def _labels(found):
    return [src.difficulty for src in found]


LADDER = _skill(
    Source(topic="t", difficulty="easy", stages=[0, 1, 2]),
    Source(topic="t", difficulty="medium", stages=[3, 4]),
    Source(topic="t", difficulty="hard", stages=[5, 6, "maintenance"]),
    Source(topic="t", difficulty="exam", stages=["exam"]),
)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [(0, ["easy"]), (2, ["easy"]), (3, ["medium"]), (4, ["medium"]), (6, ["hard"]), (N, ["hard"])],
)
def test_normal_ladder(stage, expected):
    assert _labels(stages.eligible_sources(LADDER, stage, N)) == expected


def test_exam_source_never_serves_a_review():
    exam_only = _skill(Source(topic="t", difficulty="exam", stages=["exam"]))
    assert all(stages.eligible_sources(exam_only, s, N) == [] for s in range(N + 1))
    assert stages.non_exam_sources(LADDER) == LADDER.sources[:3]
    assert stages.is_exam_source(LADDER.sources[3]) and not stages.is_exam_source(LADDER.sources[0])


def test_unstaged_source_serves_every_stage():
    any_stage = _skill(Source(topic="t", difficulty="easy"))
    assert all(_labels(stages.eligible_sources(any_stage, s, N)) == ["easy"] for s in range(N + 1))


def test_a_gap_uses_the_nearest_stage_not_every_source():
    late = _skill(
        Source(topic="t", difficulty="easy", stages=[4, 5, 6]),
        Source(topic="t", difficulty="medium", stages=[5, 6]),
        Source(topic="t", difficulty="hard", stages=[6, "maintenance"]),
    )
    assert _labels(stages.eligible_sources(late, 0, N)) == ["easy"]
    early = _skill(Source(topic="t", difficulty="easy", stages=[1, 2, 3]))
    assert _labels(stages.eligible_sources(early, 0, N)) == ["easy"]
    assert _labels(stages.eligible_sources(early, N, N)) == ["easy"]


def test_a_tie_prefers_the_easier_side():
    split = _skill(
        Source(topic="t", difficulty="easy", stages=[1]),
        Source(topic="t", difficulty="hard", stages=[3]),
    )
    assert _labels(stages.eligible_sources(split, 2, N)) == ["easy"]


def test_a_stage_past_the_ladder_counts_as_maintenance():
    long_ladder = _skill(
        Source(topic="t", difficulty="easy", stages=[0, 1, 2]),
        Source(topic="t", difficulty="hard", stages=[9, 10]),
    )
    assert _labels(stages.eligible_sources(long_ladder, N, N)) == ["hard"]
    assert _labels(stages.eligible_sources(long_ladder, 6, N)) == ["hard"]  # nearest staged
    assert _labels(stages.eligible_sources(long_ladder, 1, N)) == ["easy"]


def test_pick_sources_rotates_from_start():
    src = ["a", "b", "c"]
    assert stages.pick_sources(src, 2) == ["a", "b"]
    assert stages.pick_sources(src, 2, start=2) == ["c", "a"]
    assert stages.pick_sources(src, 4, start=1) == ["b", "c", "a", "b"]
    assert stages.pick_sources([], 3) == [] and stages.pick_sources(src, 0) == []


def test_every_active_skill_in_the_library_serves_every_stage():
    _, skills = engine.load_skills()
    missing = [
        (skill.id, stage)
        for skill in skills
        if skill.status == "active" and stages.non_exam_sources(skill)
        for stage in range(N + 1)
        if not stages.eligible_sources(skill, stage, N)
    ]
    assert missing == []
