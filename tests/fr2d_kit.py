"""Shared checks for the `test_fr2d_*` files: the chance-and-sampling skills.

Each test file re-derives its own answers from the rendered question text; nothing here imports a generator module.
"""

from __future__ import annotations

from functools import lru_cache

from mathgen.engine import grade, load_skills, make_for_skill
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SEEDS = range(1, 151)


@lru_cache(maxsize=None)
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def question(skill_id: str, tier: str, seed: int):
    return make_for_skill(skill(skill_id), seed, source=tier)


def check_tiers(skill_id: str, tiers=("easy", "medium", "hard"), exam=False) -> None:
    """The tiers are wired as the skill says, and each asks a different shape of question."""
    sk = skill(skill_id)
    assert [s.difficulty for s in sk.sources] == list(tiers) + (["exam"] if exam else [])
    shapes = {t: {prompt_shape(question(skill_id, t, s)) for s in range(1, 41)} for t in tiers}
    for a in tiers:
        for b in tiers:
            if a < b:
                assert shapes[a] != shapes[b] or len(shapes[a]) > 1, (a, b)
    assert len(set().union(*shapes.values())) >= 3


def check_text(skill_id: str, tier: str) -> None:
    for seed in SEEDS:
        q = question(skill_id, tier, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.text.strip() and "\n" not in q.text


def check_determinism(skill_id: str, tier: str) -> None:
    for seed in (1, 7, 99):
        assert question(skill_id, tier, seed).to_json() == question(skill_id, tier, seed).to_json()


def check_options(skill_id: str, tier: str, min_options: int = 2) -> None:
    """The key grades correct; every wrong option grades wrong and carries a catalogue id."""
    sk = skill(skill_id)
    for seed in SEEDS:
        q = question(skill_id, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= min_options, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (seed, d, q.answer_text)
            tag = q.option_notes.get(d)
            assert tag is None or tag in CATALOGUE, (seed, d, tag)
        assert sum(d in q.option_notes for d in q.distractors) >= len(q.distractors) - 1


def check_exam(skill_id: str, parts: int = 4) -> None:
    """The composed key earns full marks; every wrong option of every part earns that part nothing."""
    sk = skill(skill_id)
    for seed in SEEDS:
        q = question(skill_id, "exam", seed)
        assert len(q.parts) == parts, q.text
        canon = {p.label: p.answer_text for p in q.parts}
        result = grade(sk, q, canon)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts)
        for p in q.parts:
            assert p.distractors, (seed, p.label)
            assert all(p.option_notes.get(d, "x") in CATALOGUE or d not in p.option_notes for d in p.distractors)
            for d in p.distractors:
                assert grade(sk, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d, q.text)
        assert not grade(sk, q, {p.label: "999999" for p in q.parts}).correct
        assert find_text_defects(q) == [], (seed, q.text)
