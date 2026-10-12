"""Shared checks for the `test_exd_x_r3c_*` files: the parts every multi-part exam source must satisfy.

Nothing here imports a generator module. Each test file supplies an `oracle(question)` that re-derives every part's
answer text from the rendered stem alone.
"""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def skill_for(skill_id):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def exam_questions(topic, n):
    return [make(topic, seed=s, difficulty="exam") for s in range(n)]


def canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def check_wired(skill, topic):
    exam = [s for s in skill.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert exam[0].params == {"tier": "exam"}
    assert list(exam[0].stages) == ["exam"]


def check_shapes(skill, topic):
    """At least three prompt shapes, and one the skill's hard tier never produces."""
    exam = {prompt_shape(make(topic, seed=s, difficulty="exam")) for s in range(60)}
    below = [s for s in skill.sources if s.difficulty != "exam"]
    hard_src = next((s for s in below if s.difficulty == "hard"), below[-1])
    hard = {prompt_shape(make(hard_src.topic, seed=s, **(hard_src.params or {}))) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


def check_oracle(topic, oracle, n=300):
    bad = []
    for q in exam_questions(topic, n):
        if oracle(q) != canonical(q):
            bad.append((q.text, oracle(q), canonical(q)))
    assert bad == [], bad[:2]


def check_text(topic, n=120):
    for q in exam_questions(topic, n):
        assert find_text_defects(q, include_working=True) == [], q.text


def check_full_marks_and_distractors(skill, topic, n=40):
    """The composed answer earns every mark; each distractor of each part earns that part nothing."""
    for q in exam_questions(topic, n):
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (q.text, q.answer_text)
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert p.answer_text not in p.distractors
            assert len(set(p.distractors)) == len(p.distractors)
            assert set(p.option_notes) >= set(p.distractors), (q.text, p.label)
            for d in p.distractors:
                got = grade(skill, q, {**canon, p.label: d})
                assert got.parts[p.label].marks == 0, (q.text, p.label, d)


def check_all_wrong(skill, topic, n=30):
    for q in exam_questions(topic, n):
        got = grade(skill, q, {p.label: "999999" for p in q.parts})
        assert got.marks == 0 and not got.correct


def check_follow_through(skill, topic, chain, n=30, first="a", ft=None):
    """A wrong first answer carried correctly through the rest earns the later parts.

    `chain(q, wrong_text)` returns the replies for the later parts (a dict label -> text), worked out by the test
    from the rendered stem.  The first part earns nothing; every later part earns its marks, and those in `ft` (default all) are flagged as follow-through.
    """
    for q in exam_questions(topic, n):
        wrong = q.parts[0].distractors[0]
        later = chain(q, wrong)
        got = grade(skill, q, {first: wrong, **later})
        assert got.parts[first].marks == 0, (q.text, wrong)
        for label in later:
            assert got.parts[label].marks == got.parts[label].max_marks, (q.text, label, wrong, later)
            if ft is None or label in ft:
                assert got.parts[label].follow_through is True, (q.text, label)
        assert not got.correct


def check_answer_space(topic, n=300, floor=100, cap=0.05):
    """At least `floor` distinct answer tuples over `n` seeds, none above `cap` of the draws."""
    tuples = Counter(tuple(canonical(q).values()) for q in exam_questions(topic, n))
    assert len(tuples) >= floor, len(tuples)
    assert max(tuples.values()) / n <= cap, tuples.most_common(3)


def check_deterministic(topic):
    assert make(topic, seed=7, difficulty="exam").to_dict() == make(topic, seed=7, difficulty="exam").to_dict()
