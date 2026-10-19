"""Every live skill against the contract's question-quality families, over the 60-seed sweep:

- answers: each tier gives enough distinct answers (`check_answer_variety`: 12 for a typed exam tier, else 6); a
  tier whose answer is nearly always the same can be passed by typing it.
- options_quality: no option faults (`check_options_quality`): a multiple-choice tier with too few or a varying
  number of options, number options where the answer's rank is predictable, a constant option that is never the
  answer, a probability offered a wrong option outside [0, 1].

The tiers that fail today are recorded in `skill_contract_answer_floors.yaml` and
`skill_contract_option_baseline.yaml` (`scripts/build_contract_floors.py`). Those lists may only shrink: a new
skill or tier has to pass, a widened tier is recorded at its new count, and a fixed tier has to leave its list.
"""

from __future__ import annotations

import pytest

from scripts import skill_contract as sc


@pytest.mark.parametrize("skill_id", sc.live_ids())
def test_skill_meets_the_quality_families(skill_id):
    sc.check_answer_variety(skill_id)
    sc.check_options_quality(skill_id)


def test_every_listed_skill_is_live():
    live = set(sc.live_ids())
    assert not set(sc.answer_floors()) - live
    for family, by_skill in sc.option_baseline().items():
        assert family in sc.OPTION_FAMILIES, family
        assert not set(by_skill) - live, family


def test_floors_are_below_the_minimum():
    for skill_id, tiers in sc.answer_floors().items():
        for key, count in tiers.items():
            assert 1 <= count < sc.answer_minimum(skill_id, key.rsplit("/", 1)[1]), (skill_id, key, count)
