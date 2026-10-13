"""A follow-through from a wrong earlier answer is stated at the part's own precision, so the prompt must state it."""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills

_SKILL = next(s for s in load_skills()[1] if s.id == "logarithm-laws")
_SOURCE = next(i for i, src in enumerate(_SKILL.sources) if src.topic == "exd_x_sc5_log_equation")


@pytest.mark.parametrize("seed", range(30))
def test_prompts_state_whole_number_precision(seed):
    q = make("exd_x_sc5_log_equation", seed=seed)
    assert "nearest whole number" in q.parts[1].prompt
    assert "nearest whole number" in q.parts[2].prompt


def test_followed_through_value_is_credited_at_the_stated_precision():
    from mathgen.engine import make_for_skill

    q = make_for_skill(_SKILL, 3, source=_SOURCE)
    # N = 9 carries into x = 6.24, which the prompt says to give to the nearest whole number: 6.
    wrong_n = "9"
    assert wrong_n in q.parts[0].distractors
    result = grade(_SKILL, q, {"a": wrong_n, "b": "6", "c": ""})
    assert result.parts["b"].follow_through and result.parts["b"].marks == 3
