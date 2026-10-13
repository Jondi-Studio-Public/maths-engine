"""Every recorded question still regenerates to the same id.

`tests/golden/questions.txt` holds, for every active skill's every source and seeds 0, 1 and 2, the question
id (skill / source key / seed / content hash). A change to a generator, a source's params, difficulty or key
changes the hashes of the questions it touches, and this test names them.

The questions are checked in BUCKETS skills at a time so pytest-xdist spreads the work across workers.
"""

import json
import zlib
from collections import defaultdict

import pytest

from scripts import golden_questions as gq

BUCKETS = 16

HOW_TO_FIX = (
    "\nThese questions no longer match tests/golden/questions.txt. If the change is NOT intended, fix the "
    "generator or skill file. If it is intended, regenerate the file with\n"
    "    python scripts/golden_questions.py --write\n"
    "and call out in the pull request that existing questions changed (which skills, and why)."
)


def _bucket(skill_id: str) -> int:
    return zlib.crc32(skill_id.encode("utf-8")) % BUCKETS


@pytest.fixture(scope="module")
def recorded():
    by_skill = defaultdict(list)
    for entry in gq.read_golden():
        by_skill[entry["id"].split("/", 1)[0]].append(entry)
    return by_skill


@pytest.fixture(scope="module")
def skills():
    return gq.skills_to_record()


def test_the_golden_file_is_sorted_and_well_formed():
    lines = gq.GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    ids = [e["id"] for e in entries]
    assert ids == sorted(ids), "tests/golden/questions.txt is generated; do not edit it by hand"
    assert len(ids) == len(set(ids))
    assert all(len(e["id"].split("/")) == 4 for e in entries)


def test_the_golden_file_covers_exactly_the_active_sources_and_seeds(recorded, skills):
    expected = {stem for skill in skills for stem in gq.stems_for(skill)}
    have = {e["id"].rsplit("/", 1)[0] for entries in recorded.values() for e in entries}
    missing, extra = sorted(expected - have), sorted(have - expected)
    assert not missing and not extra, (
        f"{len(missing)} question(s) not recorded, e.g. {missing[:5]}; {len(extra)} recorded but gone, "
        f"e.g. {extra[:5]}." + HOW_TO_FIX
    )


@pytest.mark.parametrize("bucket", range(BUCKETS))
def test_recorded_questions_regenerate_unchanged(bucket, recorded, skills):
    current = [e for skill in skills if _bucket(skill.id) == bucket for e in gq.entries_for(skill)]
    wanted = [e for sid, entries in recorded.items() if _bucket(sid) == bucket for e in entries]
    diffs = gq.differences(wanted, current)
    assert not diffs, f"{len(diffs)} question(s) changed:\n" + "\n".join(diffs[:40]) + HOW_TO_FIX
