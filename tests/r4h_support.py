"""Checks shared by the second-year numerical-methods tests (`tests/test_r4h_*.py`).

Each skill's own test file supplies an oracle that works from the question text alone; the checks here are the ones
every skill shares: the key grades correct and every distractor grades wrong through `mathgen.grading.grade`, the
distractors are tagged with catalogue ids, the exam chain has the stated shape and marks, follow-through is honoured
from a learner's wrong earlier reply, and the questions are deterministic and free of text defects.
"""

from __future__ import annotations

import re

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.grading.parts import follow_through_target, named_parts, reply_value
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def skill_of(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def clean(text):
    """Stem text with the Unicode minus as a hyphen-minus."""
    return text.replace("−", "-")


def numbers(text):
    """Every (possibly negative) integer in a stem, with the Unicode minus read as a minus."""
    return [int(t) for t in re.findall(r"-?\d+", text.replace("−", "-"))]


def stem_of(q):
    return q.stem if q.parts else q.text


def check_wiring(skill, topic, exam):
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(topic, t) for t in TIERS] + [(exam, "exam")]
    assert skill.sources[-1].stages == ["exam"]
    assert skill.status == "active" and tuple(skill.year_level) == ("university",) and "university-year-2" in skill.tags


def check_key_and_distractors(skill, topic, tiers=TIERS, seeds=30, min_options=3):
    for tier in tiers:
        for seed in range(seeds):
            q = make(topic, seed=seed, difficulty=tier)
            assert grade(skill, q, q.answer_text).correct, (tier, seed, q.text, q.answer_text)
            assert len(q.distractors) >= min_options, (tier, seed)
            assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors
            assert set(q.option_notes) == set(q.distractors)
            assert all(note in CATALOGUE for note in q.option_notes.values())
            for text in q.distractors:
                assert not grade(skill, q, text).correct, (tier, seed, q.text, text)


def check_text_and_determinism(topic, tiers=TIERS, seeds=40):
    for tier in tiers:
        for seed in range(seeds):
            q = make(topic, seed=seed, difficulty=tier)
            assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
            assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def check_variety(topic, tiers=TIERS, seeds=80, distinct=30, repeats=6, by="answer_text"):
    """Distinct answers (or distinct stems, for a tier whose answer is a small whole number) across seeds."""
    for tier in tiers:
        answers = [getattr(make(topic, seed=seed, difficulty=tier), by) for seed in range(seeds)]
        assert len(set(answers)) >= distinct, (tier, len(set(answers)))
        assert max(answers.count(a) for a in set(answers)) <= repeats, tier


def exam(exam_topic, seed):
    return make(exam_topic, seed=seed, difficulty="exam")


def canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def check_exam_shape(exam_topic, kinds, marks, follow, seeds=40, min_options=2):
    labels = [chr(ord("a") + i) for i in range(len(kinds))]
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        assert [p.label for p in q.parts] == labels
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert [p.follow_through is not None for p in q.parts] == list(follow)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        for p in q.parts:
            assert len(p.distractors) >= min_options, (seed, p.label)
            assert set(p.option_notes) == set(p.distractors)
            assert all(note in CATALOGUE for note in p.option_notes.values())


def check_exam_grading(skill, exam_topic, seeds=20):
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        full = grade(skill, q, canonical(q))
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts), (seed, full)
        canon = canonical(q)
        for p in q.parts:
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def check_follow_through(skill, exam_topic, seeds=40, minimum=20):
    """A wrong earlier reply that the question's own distractor stands for earns the later part's marks when the
    later reply is right for that wrong value."""
    checked = 0
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        canon = canonical(q)
        by_label = {p.label: p for p in q.parts}
        values = {p.label: reply_value(p.answer_kind, p.answer_text) for p in q.parts if p.answer_kind != "mcq"}
        for later in q.parts:
            if not later.follow_through:
                continue
            for name in named_parts(later):
                for wrong in by_label[name].distractors:
                    swapped = {**values, name: reply_value(by_label[name].answer_kind, wrong)}
                    target = follow_through_target(later, swapped)
                    if target is None or target.answer_text == later.answer_text:
                        continue
                    result = grade(skill, q, {**canon, name: wrong, later.label: target.answer_text})
                    part = result.parts[later.label]
                    if part.marks == later.marks and part.follow_through is True:
                        checked += 1
    assert checked >= minimum, checked


def check_exam_variety(exam_topic, seeds=100, distinct=60):
    answers = [exam(exam_topic, seed).answer_text for seed in range(seeds)]
    assert len(set(answers)) >= distinct
    assert exam(exam_topic, 7).to_json() == exam(exam_topic, 7).to_json()


def sympy_expr(text):
    """A student-style expression (x^2, 3x, (1/2)x, x² or x³) read by SymPy."""
    from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

    sup = str.maketrans(
        {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    )
    text = clean(text)
    text = re.sub(r"([x-y])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: f"{m.group(1)}**{m.group(2).translate(sup)}", text)
    text = text.replace("^", "**")
    return parse_expr(text, transformations=standard_transformations + (implicit_multiplication_application,))
