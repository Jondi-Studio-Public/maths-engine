"""A single-value question asks for one rounding, never two.

"Give your answer to 2 decimal places ... to 3 significant figures" leaves the learner to guess which
instruction counts, and the number grader cannot check either (`requested_rounding` reads no count from a
text that states two). Swept over every active skill's `number` answers, part by part for a multi-part
question, a few seeds per source.
"""

from __future__ import annotations

import pytest

from mathgen import engine
from mathgen.text_quality import conflicting_rounding

SEEDS = range(4)
_SKILLS = [skill for skill in engine.load_skills()[1] if skill.status == "active" and skill.sources]


def _number_texts(skill, question):
    """(where, text) for every single number answer in `question`."""
    if question.parts:
        return [(f"part {part.label}", part.text) for part in question.parts if part.answer_kind == "number"]
    return [("question", question.text)] if skill.answer == "number" else []


@pytest.mark.parametrize("skill", _SKILLS, ids=lambda skill: skill.id)
def test_a_number_answer_is_rounded_one_way(skill):
    problems = []
    for index, source in enumerate(skill.sources):
        for seed in SEEDS:
            question = engine.make_for_skill(skill, seed=seed, source=index)
            for where, text in _number_texts(skill, question):
                reason = conflicting_rounding(text or "")
                if reason:
                    problems.append(f"{source.topic}/{source.difficulty} seed {seed} {where}: {reason}: {text!r}")
    assert not problems, "\n".join(problems[:5])


@pytest.mark.parametrize(
    "text",
    [
        "Find x to 2 decimal places. Give your answer to 3 significant figures.",
        "Find x, correct to 1 decimal place, to the nearest whole number.",
        "Find x to 2 decimal places and then to 3 decimal places.",
        "Find x to 2 significant figures (3 s.f.).",
    ],
)
def test_two_roundings_are_flagged(text):
    assert conflicting_rounding(text)


@pytest.mark.parametrize(
    "text",
    [
        "Find x to 2 decimal places.",
        "Find x, each value rounded to 1 decimal place, to 1 decimal place.",
        "The ratios (each rounded to 4 d.p.) are shown. Find x to 2 decimal places.",
        "Give your answer to 3 significant figures.",
        "Round 20.26 to the nearest tenth.",
        "Find the total cost.",
    ],
)
def test_one_rounding_is_not_flagged(text):
    assert conflicting_rounding(text) is None


def test_seconds_ending_a_sentence_are_not_significant_figures():
    assert (
        conflicting_rounding("It completes one cycle every T = 3 s. Find its wavelength to 2 decimal places.") is None
    )
