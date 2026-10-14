"""The shown working for the weekly fixed costs must add up to the printed answer."""

from __future__ import annotations

import re

import pytest

from mathgen import make


@pytest.mark.parametrize("seed", range(200))
def test_budget_working_adds_to_answer(seed):
    q = make("exd_x_n1_budget_to_goal", seed=seed)
    part = q.parts[0]
    total = re.match(r"Total = (.*) = \$", part.working[1]).group(1)
    summed = sum(float(x.replace("$", "").replace(",", "")) for x in total.split(" + "))
    assert f"{summed:.2f}" == part.answer_text.replace("$", "").replace(",", "")
