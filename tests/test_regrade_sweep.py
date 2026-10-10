"""Tests for scripts/regrade_sweep.py."""

from pathlib import Path

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from scripts.regrade_sweep import _perturb_last_decimal_place, sweep_skill

ROOT = Path(__file__).resolve().parents[1]


def _load_skill(skill_id: str):
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    return by_id[skill_id]


class _FakeResult:
    def __init__(self, correct: bool):
        self.correct = correct


@pytest.mark.parametrize(
    "skill_id",
    ["f10-fractions-half-quarter-eighth", "f10-number-represent-order-to-120"],
)
def test_real_shipped_skill_sweeps_clean_at_20_seeds(skill_id):
    skill = _load_skill(skill_id)
    result = sweep_skill(skill, seeds=20)
    assert result.questions > 0
    assert result.false_rejects == []
    assert result.false_accepts == []
    assert result.exceptions == []


def test_monkeypatched_accept_everything_grader_produces_false_accepts():
    skill = _load_skill("f10-fractions-half-quarter-eighth")
    fake_graders = dict(GRADERS)
    fake_graders["fraction"] = lambda question, given: _FakeResult(True)

    result = sweep_skill(skill, seeds=5, graders=fake_graders)
    assert result.false_accepts, "an accept-everything grader must register false accepts"


def test_monkeypatched_reject_everything_grader_produces_false_rejects():
    skill = _load_skill("f10-fractions-half-quarter-eighth")
    fake_graders = dict(GRADERS)
    fake_graders["fraction"] = lambda question, given: _FakeResult(False)

    result = sweep_skill(skill, seeds=5, graders=fake_graders)
    assert result.false_rejects, "a reject-everything grader must register false rejects"


@pytest.mark.parametrize(
    "canonical,expected",
    [
        ("0.37", "0.38"),
        ("159", "160"),
        ("14.10", "14.11"),
        ("f(3) = 7", "f(3) = 8"),
        ("-2.5", "-2.6"),
    ],
)
def test_number_synthetic_wrong_answer_differs_by_one_unit_in_last_stated_place(canonical, expected):
    perturbed = _perturb_last_decimal_place(canonical)
    assert perturbed == expected
    assert perturbed != canonical
