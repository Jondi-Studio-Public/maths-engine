"""Every question of an active `read_aloud` skill is short enough to read aloud to a young learner."""

from __future__ import annotations

import functools

import pytest

from mathgen import engine
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills

SEEDS = range(20)


@functools.lru_cache(maxsize=1)
def _read_aloud_skills():
    return [s for s in load_skills()[1] if s.read_aloud and s.status == "active"]


def _words(text: str) -> int:
    return len(text.split())


def test_the_field_defaults_to_false():
    skills = load_skills()[1]
    assert any(not s.read_aloud for s in skills)


@pytest.mark.parametrize("skill", _read_aloud_skills(), ids=lambda s: s.id)
def test_every_prompt_is_short(skill):
    for index, source in enumerate(skill.sources):
        for seed in SEEDS:
            q = engine.make_for_skill(skill, seed, source=index)
            prompts = [p.prompt for p in q.parts] if q.parts else [q.text]
            for prompt in prompts:
                assert _words(prompt) <= READ_ALOUD_MAX_WORDS, (source.topic, source.difficulty, seed, prompt)
