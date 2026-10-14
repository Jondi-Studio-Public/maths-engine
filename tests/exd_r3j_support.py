"""Checks shared by the `exd_x_r3j_*` multi-part exam tests: key accepted, distractors rejected and tagged,
number-part distractors at the key's decimal places, and the size of the answer space."""

from __future__ import annotations

import re
from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_source_wired(skill_id: str, topic: str) -> None:
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert [s.stages for s in exam] == [["exam"]]


def check_full_marks(skill_id: str, topic: str, seeds=range(100)) -> None:
    sk = skill(skill_id)
    for s in seeds:
        q = make(topic, seed=s)
        result = grade(sk, q, canonical(q))
        assert result.correct and result.marks == result.max_marks, (s, q.text)
        assert result.max_marks == sum(p.marks for p in q.parts)
        assert grade(sk, q, q.answer_text).correct


def check_distractors(skill_id: str, topic: str, seeds=range(150)) -> None:
    sk = skill(skill_id)
    for s in seeds:
        q = make(topic, seed=s)
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (s, p.label)
            assert len(set(p.distractors)) == len(p.distractors)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (s, p.label, d, p.option_notes.get(d))
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (s, p.label, d)
            if p.answer_kind == "number":
                places = len(p.answer_text.split(".")[1]) if "." in p.answer_text else None
                for d in p.distractors:
                    got = len(d.split(".")[1]) if "." in d else None
                    assert got == places, (s, p.label, p.answer_text, d)


def check_answer_space(topic: str, seeds=300, distinct=100, top=0.05) -> None:
    tuples = Counter(make(topic, seed=s).answer_text for s in range(seeds))
    assert len(tuples) >= distinct, len(tuples)
    assert max(tuples.values()) / seeds <= top, tuples.most_common(3)


def check_shapes(topic: str, hard_topic: str, hard_kwargs: dict, minimum=3) -> None:
    from mathgen.text_quality import find_text_defects, prompt_shape

    exam = {prompt_shape(make(topic, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(hard_topic, seed=s, **hard_kwargs)) for s in range(60)}
    assert len(exam) >= minimum
    assert exam - hard
    for s in range(60):
        assert find_text_defects(make(topic, seed=s)) == []


def numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def check_deterministic(topic: str) -> None:
    assert make(topic, seed=7).to_dict() == make(topic, seed=7).to_dict()
