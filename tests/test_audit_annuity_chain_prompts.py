"""The annuity chain's canonical (c) and (d) come from the rounded earlier answers, so the prompts must say so."""

from __future__ import annotations

import pytest

from mathgen import make


@pytest.mark.parametrize("seed", range(40))
def test_later_parts_name_the_answers_they_use(seed):
    q = make("exd_x_r2e_annuity_chain", seed=seed)
    assert "answer to (b)" in q.parts[2].prompt
    assert "answers to (a) and (c)" in q.parts[3].prompt
