"""Every live skill's tiers give enough distinct answers over the 60-seed sweep (`check_answer_variety`).

A tier whose answer is nearly always the same can be passed by typing that answer. Tiers below the minimum
today are listed, at their current count, in `skill_contract_answer_floors.yaml`: each may only rise, and
leaves the file when it reaches the minimum. A new skill or tier has to meet the minimum.
"""

from __future__ import annotations

import pytest

from scripts import skill_contract as sc


@pytest.mark.parametrize("skill_id", sc.live_ids())
def test_skill_gives_enough_distinct_answers(skill_id):
    sc.check_answer_variety(skill_id)


def test_every_floor_names_a_live_skill():
    assert not set(sc.answer_floors()) - set(sc.live_ids())


def test_floors_are_below_the_minimum():
    for skill_id, tiers in sc.answer_floors().items():
        for key, count in tiers.items():
            assert 1 <= count < sc.MIN_DISTINCT_ANSWERS, (skill_id, key, count)
