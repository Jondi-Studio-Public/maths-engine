"""The working of the triangle-area source states the area in the same unit as the prompt and figure."""

from __future__ import annotations

import re

import pytest

from mathgen.engine import load_skills, make_for_skill

_SKILL = next(s for s in load_skills()[1] if s.id == "triangle-area-base-height")
_UNIT = re.compile(r"(?:cm|mm|m)²")


@pytest.mark.parametrize("source", range(3))
@pytest.mark.parametrize("seed", range(60))
def test_working_unit_matches_prompt(source, seed):
    q = make_for_skill(_SKILL, seed, source=source)
    prompt_units = set(_UNIT.findall(q.text))
    assert len(prompt_units) == 1
    working_units = {u for part in q.working for u in _UNIT.findall(part)}
    assert working_units <= prompt_units
