"""Shared skill contract: the checks every per-batch `_build` file repeats, once.

One pytest case per live skill and check family (`scripts/skill_contract.py` holds
the checks), so a new skill needs no boilerplate test file beyond its own oracle,
and a change to one skill (or the generator it uses) selects only that skill's cases:

    pytest tests/test_skill_contract_part_*.py -k "<skill-id>"

The cases live in `tests/test_skill_contract_part_<k>.py` (built by
`tests/skill_contract_cases.py`), split so CI can spread them over its shards; this
file checks the baseline itself.

Skills that predate the contract and miss a family on purpose-or-neglect are
listed in `tests/skill_contract_baseline.yaml` as `family: {skill-id: reason}`.
Their case is a strict xfail: it must still fail, so a fixed skill drops off the
list and a new skill can never join it unnoticed in review. Regenerate with
`python scripts/skill_contract_baseline.py` (the whole library; slow).

Skill-specific oracles stay in their own test files.
"""

from pathlib import Path


from scripts import skill_contract as sc

from tests.skill_contract_cases import BASELINE, PARTS, part_ids


def test_the_parts_cover_every_live_skill_once():
    ids = [sid for part in range(PARTS) for sid in part_ids(part)]
    assert sorted(ids) == sorted(sc.live_ids())
    assert len(ids) == len(set(ids))
    for part in range(PARTS):
        assert Path(__file__).with_name(f"test_skill_contract_part_{part}.py").is_file()
    assert not Path(__file__).with_name(f"test_skill_contract_part_{PARTS}.py").exists()


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
