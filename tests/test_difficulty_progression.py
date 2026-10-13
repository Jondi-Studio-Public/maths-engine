"""A structured difficulty_progression: loaded into TierDemand entries and checked by validate()."""

from __future__ import annotations

import pytest

from mathgen.skills import TierDemand, load_skills, validate

SKILL = """\
id: add-two
name: Add two
domain: arithmetic
status: active
prerequisites: []
answer: number
sources:
  - topic: addition
    params: {{}}
    difficulty: easy
  - topic: addition
    params: {{}}
    difficulty: medium
  - topic: addition
    params: {{}}
    difficulty: hard
{progression}"""


def _errors(tmp_path, progression):
    (tmp_path / "arithmetic").mkdir(parents=True)
    (tmp_path / "arithmetic" / "add-two.yaml").write_text(SKILL.format(progression=progression), encoding="utf-8")
    domains, skills = load_skills(tmp_path)
    return skills[0], [e for e in validate(domains, skills).errors if "difficulty_progression" in e]


GOOD = """\
difficulty_progression:
  easy: {steps: 1, demand: Add two one-digit numbers.}
  medium: {steps: 2, demand: Add with one regroup.}
  hard: {steps: 2, demand: Add three numbers with a regroup.}
"""


def test_a_structured_progression_loads_into_tier_demands(tmp_path):
    skill, errors = _errors(tmp_path, GOOD)
    assert errors == []
    assert skill.difficulty_progression["easy"] == TierDemand(steps=1, demand="Add two one-digit numbers.")
    assert [t.steps for t in skill.difficulty_progression.values()] == [1, 2, 2]


def test_prose_is_still_accepted(tmp_path):
    skill, errors = _errors(tmp_path, "difficulty_progression: Hard adds a regroup.\n")
    assert errors == [] and skill.difficulty_progression == "Hard adds a regroup."


def test_a_harder_tier_may_not_ask_fewer_steps(tmp_path):
    _, errors = _errors(tmp_path, GOOD.replace("hard: {steps: 2", "hard: {steps: 1"))
    assert errors == [
        "add-two: difficulty_progression.hard asks 1 step(s), fewer than medium (2); a harder tier must not ask less"
    ]


def test_every_used_tier_must_be_described(tmp_path):
    _, errors = _errors(tmp_path, "\n".join(GOOD.splitlines()[:3]) + "\n")
    assert errors == ["add-two: difficulty_progression does not describe tier(s) ['hard']"]


def test_a_tier_no_source_uses_is_rejected(tmp_path):
    _, errors = _errors(tmp_path, GOOD + "  exam: {steps: 3, demand: A three-part sum.}\n")
    assert errors == ["add-two: difficulty_progression describes tier(s) ['exam'] that no source uses"]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("{steps: 0, demand: x}", "steps must be a whole number of at least 1"),
        ("{steps: 1.5, demand: x}", "steps must be a whole number of at least 1"),
        ("{steps: 1, demand: ''}", "demand must say what the tier asks"),
    ],
)
def test_each_entry_needs_whole_steps_and_a_sentence(tmp_path, entry, message):
    _, errors = _errors(tmp_path, GOOD.replace("{steps: 1, demand: Add two one-digit numbers.}", entry))
    assert len(errors) == 1 and message in errors[0] and "difficulty_progression.easy" in errors[0]


def test_an_unknown_tier_name_is_rejected(tmp_path):
    _, errors = _errors(tmp_path, GOOD + "  expert: {steps: 4, demand: x}\n")
    assert "add-two: difficulty_progression names unknown tier(s) ['expert']" in errors


def test_every_structured_progression_in_the_library_validates():
    domains, skills = load_skills()
    structured = [s for s in skills if isinstance(s.difficulty_progression, dict)]
    from mathgen.skills import _progression_errors

    assert [e for s in structured for e in _progression_errors(s)] == []
