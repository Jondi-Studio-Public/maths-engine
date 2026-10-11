"""Independent oracle for senior-define-the-imaginary-number-i-as-a-root.

Derives the expected answer purely from ``question.text`` (never from the
generator's own tables or helpers) and compares it against ``question.
answer_text``. Does not import anything from
``mathgen.generators.p7c_imaginary_root_build``.
"""

import math
import re

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-define-the-imaginary-number-i-as-a-root"

_SQRT_RE = re.compile(r"Simplify sqrt\((-\d+)\)\.")
_POWER_RE = re.compile(r"Evaluate i\^(\(-?\d+\)|-?\d+)\.")


def _oracle_answer(text: str) -> str:
    """Independently re-derive the answer i-notation string from the
    question text alone."""
    sqrt_match = _SQRT_RE.search(text)
    if sqrt_match:
        radicand = int(sqrt_match.group(1))
        assert radicand < 0, f"radicand not negative: {text}"
        magnitude = -radicand
        root = math.isqrt(magnitude)
        assert root * root == magnitude, f"radicand not a perfect square: {text}"
        return "i" if root == 1 else f"{root}i"

    power_match = _POWER_RE.search(text)
    if power_match:
        n = int(power_match.group(1).strip("()"))
        r = n % 4
        table = {0: "1", 1: "i", 2: "-1", 3: "-i"}
        return table[r]

    raise AssertionError(f"question text matches neither known shape: {text!r}")


def _skill_sources():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill.sources


@pytest.mark.parametrize("seed", range(60))
def test_oracle_matches_answer_text_every_tier(seed):
    for spec in _skill_sources():
        if spec.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        question = make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))
        expected = _oracle_answer(question.text)
        assert expected == question.answer_text, (
            f"tier={spec.difficulty} seed={seed}: question={question.text!r} "
            f"oracle={expected!r} stated={question.answer_text!r}"
        )


def test_oracle_coverage():
    sources = _skill_sources()
    assert len(sources) == 4  # easy, medium, hard and exam
    total = len([s for s in sources if s.difficulty != "exam"]) * 60
    assert total == 180
