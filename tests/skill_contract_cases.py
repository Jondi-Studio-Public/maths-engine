"""The per-skill contract cases, split into parts so CI can spread them over its shards.

`scripts/skill_contract.py` holds the checks; `tests/test_skill_contract.py` explains the contract and the
baseline. CI shards by test file, and one file holding every case (about 5,000, over an hour of CPU) would make
one shard the slowest by far, so `tests/test_skill_contract_part_<k>.py` each run every family for the live
skills whose position in `live_ids()` is k modulo PARTS. A case id is still the skill id, so

    pytest tests/test_skill_contract_part_*.py -k "<skill-id>"

runs one skill's cases wherever they landed.
"""

from pathlib import Path

import pytest
import yaml

from scripts import skill_contract as sc

PARTS = 8

BASELINE_PATH = Path(__file__).with_name("skill_contract_baseline.yaml")
BASELINE = (yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8")) or {}) if BASELINE_PATH.exists() else {}

CHECKS = {
    "shape": sc.check_shape,
    "determinism": sc.check_determinism,
    "distractors": sc.check_distractors,
    "options": sc.check_options,
    "variety": sc.check_variety,
    "escalation": sc.check_escalation,
    "grader": sc.check_grader,
}


def part_ids(part: int) -> list[str]:
    return [sid for i, sid in enumerate(sc.live_ids()) if i % PARTS == part]


def _cases(family: str, part: int):
    known = BASELINE.get(family) or {}
    return [
        pytest.param(sid, marks=pytest.mark.xfail(strict=True, reason=known[sid]), id=sid)
        if sid in known
        else pytest.param(sid, id=sid)
        for sid in part_ids(part)
    ]


def family_test(family: str, part: int):
    """A test function running `family`'s check for each skill in `part`; a baselined skill is a strict xfail."""
    check = CHECKS[family]

    @pytest.mark.parametrize("skill_id", _cases(family, part))
    def test(skill_id):
        check(skill_id)

    test.__name__ = f"test_{family}"
    return test
