"""Shared skill contract: the checks every per-batch `_build` file repeats, once.

One pytest case per live skill and check family (`scripts/skill_contract.py` holds
the checks), so a new skill needs no boilerplate test file beyond its own oracle,
and a change to one skill (or the generator it uses) selects only that skill's cases:

    pytest tests/test_skill_contract.py -k "<skill-id>"

Skills that predate the contract and miss a family on purpose-or-neglect are
listed in `tests/skill_contract_baseline.yaml` as `family: {skill-id: reason}`.
Their case is a strict xfail: it must still fail, so a fixed skill drops off the
list and a new skill can never join it unnoticed in review. Regenerate with
`python scripts/skill_contract_baseline.py` (the whole library; slow).

Skill-specific oracles stay in their own test files.
"""

from pathlib import Path

import pytest
import yaml

from scripts import skill_contract as sc

BASELINE_PATH = Path(__file__).with_name("skill_contract_baseline.yaml")
BASELINE = (yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8")) or {}) if BASELINE_PATH.exists() else {}


def _cases(family: str):
    known = BASELINE.get(family) or {}
    return [
        pytest.param(sid, marks=pytest.mark.xfail(strict=True, reason=known[sid]), id=sid)
        if sid in known
        else pytest.param(sid, id=sid)
        for sid in sc.live_ids()
    ]


@pytest.mark.parametrize("skill_id", _cases("shape"))
def test_shape(skill_id):
    sc.check_shape(skill_id)


@pytest.mark.parametrize("skill_id", _cases("determinism"))
def test_determinism(skill_id):
    sc.check_determinism(skill_id)


@pytest.mark.parametrize("skill_id", _cases("distractors"))
def test_distractors(skill_id):
    sc.check_distractors(skill_id)


@pytest.mark.parametrize("skill_id", _cases("options"))
def test_options(skill_id):
    sc.check_options(skill_id)


@pytest.mark.parametrize("skill_id", _cases("variety"))
def test_variety(skill_id):
    sc.check_variety(skill_id)


@pytest.mark.parametrize("skill_id", _cases("escalation"))
def test_escalation(skill_id):
    sc.check_escalation(skill_id)


@pytest.mark.parametrize("skill_id", _cases("grader"))
def test_grader_round_trip(skill_id):
    sc.check_grader(skill_id)


def test_baseline_names_only_live_skills_and_known_families():
    live = set(sc.live_ids())
    assert set(BASELINE) <= set(sc.FAMILIES)
    for family, skills in BASELINE.items():
        assert set(skills) <= live, f"{family}: baseline lists a skill that is not live"
        assert all(isinstance(r, str) and r for r in skills.values()), f"{family}: every entry needs a reason"


def test_baseline_growth_is_detected():
    from scripts.skill_contract_baseline import grown

    old = {"variety": {"a": "x"}}
    assert grown(old, {"variety": {"a": "x"}}) == []
    assert grown(old, {"variety": {}}) == []
    assert grown(old, {"variety": {"a": "x", "b": "y"}, "options": {"a": "z"}}) == ["options: a", "variety: b"]
    assert grown(None, {"variety": {"a": "x"}}) == ["variety: a"]


def test_pool_floor_changes_are_detected():
    from scripts.skill_contract_baseline import weakened_floors

    old = {"a": {"t/easy": 6}}
    assert weakened_floors(old, {"a": {"t/easy": 6}}) == []
    assert weakened_floors(old, {"a": {"t/easy": 9}}) == []
    assert weakened_floors(old, {}) == []
    assert weakened_floors(old, {"a": {"t/easy": 5}}) == ["pool floor lowered: a t/easy 6 -> 5"]
    assert weakened_floors(old, {"b": {"t/easy": 3}}) == ["pool floor added: b t/easy=3"]
