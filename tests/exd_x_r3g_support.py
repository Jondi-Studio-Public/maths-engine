"""Shared checks for the `test_exd_x_r3g_*` files: structure, answer space, grading and format of multi-part exams.

Nothing here imports a generator module. Each test file supplies its own independent re-derivation of the answers
from the rendered stem and part prompts.
"""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

SEEDS = range(300)


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def questions(topic: str, n: int = 300):
    return [make(topic, seed=s, tier="exam") for s in range(n)]


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_wired(skill_id: str, topic: str):
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert [list(s.stages) for s in exam] == [["exam"]]
    assert dict(exam[0].params) == {"tier": "exam"}


def check_variety(topic: str, hard_topic: str, hard_params: dict, n: int = 60):
    exam = {prompt_shape(make(topic, seed=s, tier="exam")) for s in range(n)}
    hard = {prompt_shape(make(hard_topic, seed=s, **hard_params)) for s in range(n)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def check_clean_and_deterministic(topic: str, n: int = 60):
    for seed in range(n):
        q = make(topic, seed=seed, tier="exam")
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == make(topic, seed=seed, tier="exam").to_json()


def check_answer_space(topic: str, n: int = 300):
    tuples = Counter(tuple(canonical(q).values()) for q in questions(topic, n))
    assert len(tuples) >= 100, len(tuples)
    top = max(tuples.values())
    assert top <= 0.05 * n, (top, tuples.most_common(3))


def _places(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else -1


def check_grading(skill_id: str, topic: str, n: int = 150):
    """The labelled key earns full marks; each distractor of each part earns that part nothing; the format of every
    distractor of a number part matches its key (same decimal point, same places)."""
    sk = skill(skill_id)
    bad = []
    for q in questions(topic, n):
        canon = canonical(q)
        full = grade(sk, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks):
            bad.append(("key rejected", q.text, q.answer_text))
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label, p.distractors)
            assert len(p.distractors) == len(set(p.distractors))
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                if result.parts[p.label].marks != 0:
                    bad.append(("distractor accepted", q.text, p.label, d))
                if p.answer_kind == "number" and _places(d) != _places(p.answer_text):
                    bad.append(("format differs from key", p.label, d, p.answer_text))
            for d in p.option_notes:
                assert d in p.distractors
        assert all(p.option_notes and len(p.option_notes) == len(p.distractors) for p in q.parts), q.text
    assert bad == [], bad[:3]


def check_tagged(topic: str):
    from mathgen.misconceptions import CATALOGUE

    seen = set()
    for q in questions(topic, 60):
        for p in q.parts:
            seen |= set(p.option_notes.values())
    assert seen, "no tagged distractors"
    assert seen <= set(CATALOGUE), seen - set(CATALOGUE)
    assert not (seen & {"wrong-value", "reported-other-quantity"})


def check_follow_through(skill_id: str, topic: str, replies, n: int = 80, at_least: int = 20):
    """`replies(q)` returns (reply dict, set of labels that must earn full marks, set that must earn nothing), or None to
    skip a question."""
    sk = skill(skill_id)
    checked = 0
    for q in questions(topic, n):
        planned = replies(q)
        if planned is None:  # the engine withholds follow-through from a reply that merely repeats an earlier one
            continue
        reply, full, zero = planned
        result = grade(sk, q, reply)
        for label in full:
            part = result.parts[label]
            assert part.marks == part.max_marks, (q.text, label, reply)
        for label in zero:
            assert result.parts[label].marks == 0, (q.text, label, reply)
        assert not result.correct
        checked += 1
    assert checked >= at_least, checked
