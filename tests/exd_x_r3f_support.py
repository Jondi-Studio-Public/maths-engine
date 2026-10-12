"""Shared assertions for the `exd_x_r3f_*` multi-part exam-tier test files (not a test module itself)."""

from __future__ import annotations

import collections
import re

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = None


def skill(skill_id: str):
    global _SKILLS
    if _SKILLS is None:
        _SKILLS = {s.id: s for s in load_skills()[1]}
    return _SKILLS[skill_id]


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def exam_sources(skill_id: str):
    return [s for s in skill(skill_id).sources if s.difficulty == "exam"]


def assert_wired(skill_id: str, topic: str) -> None:
    (source,) = exam_sources(skill_id)
    assert source.topic == topic
    assert dict(source.params) == {"tier": "exam"}
    assert list(source.stages) == ["exam"]


def _below_exam_questions(skill_id: str, seed: int):
    """Questions from the highest practice tier the skill has below its exam tier."""
    sources = [s for s in skill(skill_id).sources if s.difficulty != "exam"]
    order = {"easy": 0, "medium": 1, "hard": 2}
    top = max(sources, key=lambda s: order.get(s.difficulty, -1))
    return make(top.topic, seed=seed, difficulty=top.difficulty, **dict(top.params))


def assert_shapes(skill_id: str, topic: str) -> None:
    exam = {prompt_shape(exam_question(topic, s)) for s in range(60)}
    below = {prompt_shape(_below_exam_questions(skill_id, s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - below, "the exam tier adds no prompt shape beyond the practice tiers"


def assert_clean_and_deterministic(topic: str) -> None:
    for seed in range(60):
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_answer_space(topic: str, seeds: int = 300, minimum: int = 100) -> None:
    """At least `minimum` distinct answer tuples over `seeds` draws, no tuple above 5%."""
    counts = collections.Counter(exam_question(topic, s).answer_text for s in range(seeds))
    assert len(counts) >= minimum, len(counts)
    assert max(counts.values()) / seeds <= 0.05, counts.most_common(3)


def assert_full_marks_and_distractors(skill_id: str, topic: str, seeds=range(120), min_options: int = 1) -> None:
    """The labelled key earns full marks, every distractor earns nothing in its part, and (for a skill graded as
    a number) every distractor has the key's decimal point and number of decimal places."""
    sk = skill(skill_id)
    number_skill = sk.answer == "number"
    for seed in seeds:
        q = exam_question(topic, seed)
        full = sum(p.marks for p in q.parts)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == full, (seed, q.answer_text)
        keys = canonical(q)
        for part in q.parts:
            assert len(part.distractors) >= min_options, (seed, part.label, part.distractors)
            assert len(set(part.distractors)) == len(part.distractors)
            assert set(part.option_notes) == set(part.distractors), (seed, part.label)
            for d in part.distractors:
                graded = grade(sk, q, {**keys, part.label: d})
                assert graded.parts[part.label].marks == 0, (seed, part.label, d, part.answer_text)
                if number_skill and part.answer_kind == "number":
                    assert ("." in d) == ("." in part.answer_text), (seed, part.label, d, part.answer_text)
                    places = lambda t: len(t.split(".")[1]) if "." in t else 0  # noqa: E731
                    assert places(d) == places(part.answer_text), (seed, part.label, d, part.answer_text)


def assert_all_wrong_earns_nothing(skill_id: str, topic: str, wrong: dict) -> None:
    sk = skill(skill_id)
    for seed in range(40):
        q = exam_question(topic, seed)
        result = grade(sk, q, wrong)
        assert result.marks == 0 and not result.correct, (seed, wrong)


def assert_follow_through(skill_id: str, topic: str, replies_for, wrong_labels, followed_labels, seeds=range(120)):
    """`replies_for(q)` returns a reply dict (or None to skip the seed) whose `wrong_labels` parts are deliberately
    wrong and whose later parts are right for the learner's own wrong values. Those later parts must earn full marks
    flagged as follow-through, and the wrong parts none."""
    sk = skill(skill_id)
    checked = 0
    for seed in seeds:
        q = exam_question(topic, seed)
        replies = replies_for(q)
        if replies is None:
            continue
        earlier = {}
        repeats = False
        for label in replies:
            repeats = repeats or (label in followed_labels and replies[label] in earlier.values())
            earlier[label] = replies[label]
        parts_by_label = {part.label: part for part in q.parts}
        coincides = any(replies[label] == parts_by_label[label].answer_text for label in followed_labels)
        if repeats or coincides:
            continue  # a slip that lands on the right number is simply right; a repeated reply is not following through
        result = grade(sk, q, replies)
        for label in wrong_labels:
            assert result.parts[label].marks == 0, (seed, label, replies)
        for label in followed_labels:
            part = result.parts[label]
            assert part.marks == part.max_marks and part.follow_through is True, (seed, label, replies, part)
        assert not result.correct
        checked += 1
    assert checked >= 40, f"only {checked} seeds were usable"


def numbers_in(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"-?\d+", text)]
