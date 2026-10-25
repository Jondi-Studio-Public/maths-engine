"""Round 4 fixes: a variable x is not shown as a times sign, and a decimal is not a fraction."""

from __future__ import annotations

import re

import pytest

from mathgen import engine
from mathgen.core import _learner_text, display_question
from mathgen.skills import load_skills

_SKILLS = {s.id: s for s in load_skills()[1]}
SEEDS = range(60)


def _lines(q):
    yield q.text
    yield q.answer_text
    yield from q.working
    for part in getattr(q, "parts", None) or []:
        yield from _lines(part)


def _shown(d):
    yield d["text"]
    yield d["answer_text"]
    yield from d["working"]
    for part in d.get("parts", []):
        yield from _shown(part)


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("Find lim(x→∞) x ln((x - 2)/(x + 4)).", "Find lim(x→∞) x ln((x - 2)/(x + 4))."),
        ("tan^4 x = tan^2 x sec^2 x - tan^2 x", "tan^4 x = tan^2 x sec^2 x - tan^2 x"),
        ("sec^3 x tan^3 x = sec^2 x tan^2 x", "sec^3 x tan^3 x = sec^2 x tan^2 x"),
        ("Find 3 x sin(30)", "Find 3 × sin(30)"),
        ("Find (2 + 1) x ln(5)", "Find (2 + 1) × ln(5)"),
    ],
)
def test_letter_x_versus_times(raw, shown):
    assert _learner_text(raw) == shown


@pytest.mark.parametrize("skill_id", ["ucv1-lhopital-rule", "integrate-tan-sec-powers"])
def test_variable_x_survives_in_every_tier(skill_id):
    skill = _SKILLS[skill_id]
    for source in range(len(skill.sources)):
        for seed in SEEDS:
            q = engine.make_for_skill(skill, seed, source=source)
            for raw, shown in zip(_lines(q), _shown(display_question(q))):
                if " x " in raw and "×" not in raw and "*" not in raw:
                    assert "×" not in shown, (skill_id, source, seed, raw, shown)


def test_lhopital_hard_limit_keeps_x():
    skill = _SKILLS["ucv1-lhopital-rule"]
    seen = 0
    for seed in range(200):
        q = engine.make_for_skill(skill, seed, source=2)
        if re.search(r"lim\(x→∞\) x ln", q.text):
            seen += 1
            assert "lim(x→∞) x ln" in display_question(q)["text"]
    assert seen >= 5


@pytest.mark.parametrize("source", [0, 1, 2])
def test_decimal_is_not_a_fraction(source):
    skill = _SKILLS["f10-fraction-decimal-links"]
    checked = 0
    for seed in range(40):
        q = engine.make_for_skill(skill, seed, source=source)
        decimal = re.search(r"\d*\.\d+", q.text)
        if q.answer_kind != "fraction" or decimal is None:
            continue
        checked += 1
        assert engine.grade(skill, q, q.answer_text).correct, q.text
        assert not engine.grade(skill, q, decimal.group()).correct, (q.text, decimal.group())
    if source == 0:
        assert checked >= 20


def test_decimal_answers_still_accepted_where_a_decimal_is_asked():
    skill = _SKILLS["f10-fraction-decimal-links"]
    for source in (1, 2):
        for seed in range(20):
            q = engine.make_for_skill(skill, seed, source=source)
            assert engine.grade(skill, q, q.answer_text).correct, q.text
