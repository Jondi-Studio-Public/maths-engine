"""Shared checks for the `tests/test_w6c_*.py` files (second-year skills built by the `w6c_*` generators).

Each test file supplies its own independent oracle (the key re-derived from the question text with SymPy or a direct
calculation); this module holds what every one of them also checks: the canonical answer grades correct through
`mathgen.grading.grade`, every distractor of every part grades wrong and names a catalogue misconception, a part
follows through from a wrong earlier answer, a question is deterministic and free of text defects, and the tiers
differ in shape.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.grading import grade as grade_reply
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TIERS = ("easy", "medium", "hard")


def skill_named(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def question(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


def check_wiring(skill, topic, exam, prerequisites=()):
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam, "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"] and skill.sources[-1].params == {}
    assert skill.status == "active"
    for needed in prerequisites:
        assert needed in skill.prerequisites


def check_tier(skill, topic, tier, seeds=80, min_distractors=2):
    """Key accepted, every distractor rejected and tagged, no text defects, deterministic."""
    for seed in range(seeds):
        q = question(topic, tier, seed)
        assert grade(skill, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= min_distractors, (tier, seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
    for seed in range(0, seeds, 9):
        assert question(topic, tier, seed).to_json() == question(topic, tier, seed).to_json()


def check_exam_grading(skill, exam, marks, seeds=60):
    """Full marks for the key; zero marks for each part's every distractor; text defects none; deterministic."""
    for seed in range(seeds):
        q = make(exam, seed=seed, difficulty="exam")
        assert [p.marks for p in q.parts] == list(marks), seed
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            if p.answer_kind == "mcq":
                assert len(p.distractors) >= 3
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        if seed % 9 == 0:
            assert q.to_json() == make(exam, seed=seed, difficulty="exam").to_json()


def follow_through_value(part, replies):
    """The key of `part` re-derived from its `follow_through` string and the earlier parts' replies."""
    expr = sympy.sympify(part.follow_through, locals={"floor": sympy.floor})
    names = {str(s): s for s in expr.free_symbols}
    return expr.subs({names[n]: sympy.Rational(replies[n[-1]]) for n in names})


def check_follow_through(skill, exam, seeds=40):
    """A wrong earlier answer still earns the later part's marks when the later reply follows from it."""
    checked = 0
    for seed in range(seeds):
        q = make(exam, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            if not p.follow_through:
                continue
            sources = sorted({str(s)[-1] for s in sympy.sympify(p.follow_through).free_symbols})
            for source in sources:
                earlier = next(e for e in q.parts if e.label == source)
                for wrong_text in earlier.distractors:
                    replies = {**canon, source: wrong_text}
                    try:
                        target = follow_through_value(p, replies)
                    except (TypeError, ZeroDivisionError):
                        continue
                    if not target.is_number or not target.is_finite:
                        continue
                    frac = sympy.Rational(target)
                    reply = str(frac.p) if frac.q == 1 else f"{frac.p}/{frac.q}"
                    result = grade(skill, q, {**replies, p.label: reply})
                    if reply == canon[p.label]:
                        continue
                    part = result.parts[p.label]
                    if part.marks == p.marks:
                        assert part.follow_through is True, (seed, p.label, source, wrong_text, reply)
                        checked += 1
    assert checked >= 5, checked


def check_tier_shapes_differ(topic, seeds=60, minimum=3):
    """Each tier offers at least `minimum` prompt shapes and the hard tier adds one the lower tiers lack."""
    shapes = {t: {prompt_shape(question(topic, t, s)) for s in range(seeds)} for t in TIERS}
    for tier, found in shapes.items():
        assert len(found) >= minimum, (tier, len(found))
    assert shapes["hard"] - (shapes["easy"] | shapes["medium"]), "the hard tier adds no prompt shape"


def check_answer_space(topic, tier, seeds=120, distinct=6, share=0.4):
    """The key is not concentrated on one or two values (so it cannot be guessed)."""
    answers = Counter(question(topic, tier, s).answer_text for s in range(seeds))
    assert len(answers) >= distinct, answers.most_common(5)
    assert max(answers.values()) <= share * seeds, answers.most_common(3)


def check_grades_via_public_api(skill, topic, seeds=10):
    """`mathgen.grading.grade` (the public entry point) agrees with the engine's grader on a typed reply."""
    for seed in range(seeds):
        q = question(topic, "easy", seed)
        assert grade_reply(q.answer_kind, q, q.answer_text).correct


_INT = re.compile(r"-?\d+")


def ints(text):
    return [int(m) for m in _INT.findall(text)]
