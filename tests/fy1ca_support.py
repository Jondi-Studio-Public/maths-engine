"""Shared checks for the first-year calculus builds (`tests/test_fy1ca_*.py`).

Each test file keeps its own oracle (it re-derives the answers from the question text); what is common to every skill
lives here: the skill is wired as expected, the canonical answer is accepted, every distractor is rejected and tagged,
the text is clean, questions are deterministic, and an exam chain gives follow-through marks.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def load_skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def check_wiring(skill, topic: str, exam_topic: str, answer: str) -> None:
    assert skill.status == "active" and skill.answer == answer
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]


def check_single_tier(skill, topic: str, tier: str, seeds=range(80)) -> None:
    """Canonical accepted, every distractor rejected, tagged, text clean, deterministic."""
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(skill, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= 2, (tier, seed)
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
            assert text in q.distractor_values
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def _reply_text(kind: str, template: str, value) -> str:
    """`value` written the way the part's canonical answer is written."""
    if kind == "number":
        places = len(template.split(".")[1]) if "." in template else 0
        return f"{float(sympy.N(value)):.{places}f}"
    value = sympy.nsimplify(value) if not isinstance(value, (int, Fraction, sympy.Rational)) else value
    if kind == "fraction":
        value = sympy.Rational(value)
        return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"
    if kind == "number":
        places = len(template.split(".")[1]) if "." in template else 0
        return f"{float(value):.{places}f}"
    return str(value)


def check_exam(skill, topic: str, labels: list[str], marks: list[int], kinds: list[str], seeds=range(50)) -> None:
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == labels
        assert [p.marks for p in q.parts] == marks
        assert [p.answer_kind for p in q.parts] == kinds
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE, (seed, p.label, text)
            assert grade(skill, q, {**canon, p.label: ""}).parts[p.label].marks == 0


def check_follow_through(skill, topic: str, seeds=range(40), minimum: float = 0.5) -> None:
    """A wrong earlier answer, carried correctly into a later part, still earns that part's marks."""
    tried = credited = 0
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        by_label = {p.label: p for p in q.parts}
        for p in q.parts:
            if not p.follow_through:
                continue
            names = sorted(n for n in (f"part_{x.label}" for x in q.parts) if n in p.follow_through)
            if len(names) != 1:
                continue
            source = by_label[names[0].removeprefix("part_")]
            if source.answer_kind not in ("number", "fraction"):
                continue
            for wrong in source.distractors:
                value = sympy.Rational(wrong) if source.answer_kind == "fraction" else sympy.Float(wrong)
                target = sympy.sympify(p.follow_through).subs({sympy.Symbol(names[0]): value})
                if not target.is_number:
                    continue
                reply = {**canon, source.label: wrong, p.label: _reply_text(p.answer_kind, p.answer_text, target)}
                result = grade(skill, q, reply)
                tried += 1
                credited += result.parts[p.label].marks == p.marks
                assert result.parts[source.label].marks == 0
    assert tried >= 20, tried
    assert credited / tried >= minimum, (credited, tried)
