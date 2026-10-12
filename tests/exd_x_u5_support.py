"""Shared helpers for the `test_exd_x_u5_*` files: building exam questions through a skill and grading them.

Nothing here imports a generator module. Each test file re-derives the answers from the rendered prompt with its
own independent code; these helpers only build the questions, grade replies and count prompt shapes.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade, load_skills, make_for_skill
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = None


def skill_of(skill_id: str):
    global _SKILLS
    if _SKILLS is None:
        _SKILLS = {s.id: s for s in load_skills()[1]}
    return _SKILLS[skill_id]


def exam_question(skill_id: str, seed: int):
    return make_for_skill(skill_of(skill_id), seed, source="exam")


def exam_sources(skill_id: str):
    return [s for s in skill_of(skill_id).sources if s.difficulty == "exam"]


def check_shapes(skill_id: str, seeds: int = 60) -> None:
    """The exam source gives at least three prompt shapes and one the skill's highest practice tier lacks."""
    skill = skill_of(skill_id)
    exam = {prompt_shape(exam_question(skill_id, s)) for s in range(seeds)}
    top = [s for s in skill.sources if s.difficulty != "exam"][-1]
    practice = {prompt_shape(make_for_skill(skill, s, source=top)) for s in range(seeds)}
    assert len(exam) >= 3, exam
    assert exam - practice


def check_no_text_defects(skill_id: str, seeds: int = 60) -> None:
    for seed in range(seeds):
        assert find_text_defects(exam_question(skill_id, seed)) == [], seed


def canonical_reply(question) -> dict:
    return {p.label: p.answer_text for p in question.parts}


def check_key_and_distractors(skill_id: str, seeds: int = 100) -> None:
    """The labelled key earns full marks; every distractor earns nothing for its part, whatever the other parts say."""
    skill = skill_of(skill_id)
    for seed in range(seeds):
        q = exam_question(skill_id, seed)
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = canonical_reply(q)
        for part in q.parts:
            assert len(part.distractors) >= 3, (seed, part.label)
            for d in part.distractors:
                result = grade(skill, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)


def as_fraction(text: str) -> Fraction:
    return Fraction(text.replace(" ", ""))


_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def parse_expression(text: str, **symbols):
    """Learner-style algebra (``2θ² - 3θ + 8``, ``x²y(x + 2y)³``) as a SymPy expression; θ is read as ``t``."""
    s = text.replace("θ", "t").replace("²", "**2").replace("³", "**3").replace("⁴", "**4").replace("−", "-")
    s = s.replace("^", "**")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict=symbols)


def rational(value) -> Fraction:
    """A SymPy rational (or int) as an exact Fraction."""
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def survives_retarget(*values: Fraction) -> bool:
    """Whether the follow-through target for each value is read back as the same rational.

    The marker states a fraction part's follow-through answer through ``sympy.nsimplify``, which for a few
    rationals with a large denominator (4212/361, for one) returns an unrelated closed form, so a learner
    who reaches exactly such a value earns no follow-through. The follow-through tests skip those values.
    """
    return all(sympy.nsimplify(sympy.Rational(v.numerator, v.denominator)).is_Rational for v in values)
