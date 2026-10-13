"""Shared helpers for the `test_exd_x_sc1_*` files: the skill's exam question, its hard question, and marking.

Nothing here imports a generator module. Each test file parses the rendered prompt back into numbers and derives
its own key, so the oracle never shares code with the generator it checks.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills, make_for_skill
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}


def skill(skill_id: str):
    return _SKILLS[skill_id]


def exam(skill_id: str, seed: int):
    return make_for_skill(skill(skill_id), seed, source="exam")


def hard(skill_id: str, seed: int):
    return make_for_skill(skill(skill_id), seed, source="hard")


def money(text: str) -> list[int]:
    """The dollar amounts in a prompt, in reading order."""
    return [int(m) for m in re.findall(r"\$(\d+)", text)]


def check_wiring(skill_id: str, topic: str) -> None:
    exam_sources = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in exam_sources] == [topic]
    assert list(exam_sources[0].stages) == ["exam"]


def check_shapes(skill_id: str) -> None:
    """The exam tier shows at least three prompt shapes, and one the hard tier never shows."""
    exam_shapes = {prompt_shape(exam(skill_id, s)) for s in range(60)}
    hard_shapes = {prompt_shape(hard(skill_id, s)) for s in range(60)}
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard_shapes


def check_no_text_defects(skill_id: str) -> None:
    for seed in range(60):
        assert find_text_defects(exam(skill_id, seed)) == [], seed


def check_grading(skill_id: str, seeds: int = 120) -> None:
    """The composed key earns full marks, and every distractor of every part earns nothing for that part."""
    shown = 0
    for seed in range(seeds):
        q = exam(skill_id, seed)
        result = grade(skill(skill_id), q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                shown += 1
                res = grade(skill(skill_id), q, {**canon, p.label: d})
                assert res.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 3 * seeds


def check_tags(skill_id: str, seeds: int = 120) -> None:
    """Every tagged distractor names a catalogue id and carries the value that error reaches."""
    from mathgen.misconceptions import CATALOGUE

    tagged = 0
    for seed in range(seeds):
        q = exam(skill_id, seed)
        for p in q.parts:
            for d in p.distractors:
                note = p.option_notes.get(d)
                if note is not None:
                    tagged += 1
                    assert note in CATALOGUE
                    assert d in p.distractor_values
    assert tagged >= 2 * seeds


def check_deterministic(skill_id: str) -> None:
    assert exam(skill_id, 7).to_json() == exam(skill_id, 7).to_json()


def check_answer_space(skill_id: str, seeds: int = 300, distinct: int = 100, top: float = 0.05) -> None:
    """At least `distinct` different composed answers over `seeds`, none above `top` of them."""
    counts: dict[str, int] = {}
    for seed in range(seeds):
        text = exam(skill_id, seed).answer_text
        counts[text] = counts.get(text, 0) + 1
    assert len(counts) >= distinct, len(counts)
    assert max(counts.values()) / seeds <= top


def frac(text: str) -> Fraction:
    return Fraction(text)
