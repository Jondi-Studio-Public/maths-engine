"""Early-years sources with leak-free stems and larger pools (``ey6a_early_variety_build``).

``ey6a_recognise_numeral`` is wired into recognise-numerals-to-5 and -to-10. The other four generators
(``ey6a_compare_length``, ``ey6a_repeating_pattern``, ``ey6a_shape_names``, ``ey6a_compare_attributes``) are
no longer wired: main rebuilt compare-lengths, continue-patterns, name-basic-shapes and
f10-compare-measurement-attributes as picture skills, and the picture skills' own tests (test_ey_shapes,
test_read_aloud, key uniqueness) forbid the ey6a sources. Those generators are still shipped, so they are
tested directly: ``make(topic, ...)`` with the parameters the sources carried on the train-2 branch, graded
against the current skill of the same name. For every seed the canonical answer grades correct through
``mathgen.engine.grade``, every distractor grades wrong, and the stem does not hand the answer over.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}

#: (skill id, topic) for every skill that still draws an ey6a source.
STILL_WIRED = [
    ("recognise-numerals-to-5", "ey6a_recognise_numeral"),
    ("recognise-numerals-to-10", "ey6a_recognise_numeral"),
]

#: (skill id, topic, [(difficulty, params), ...]) for generators tested directly. The variants are the sources
#: the skills carried on the train-2 branch, before main rebuilt them as picture skills.
DIRECT_VARIANTS = {
    ("compare-lengths", "ey6a_compare_length"): [("easy", {"lines": 3}), ("easy", {"lines": 4})],
    ("continue-patterns", "ey6a_repeating_pattern"): [("easy", {})],
    ("name-basic-shapes", "ey6a_shape_names"): [("easy", {})],
    ("f10-compare-measurement-attributes", "ey6a_compare_attributes"): [
        ("easy", {"attr_pool": ["length", "mass"], "equal_fraction": 0.0}),
        ("medium", {"attr_pool": ["capacity", "duration"], "equal_fraction": 0.0}),
        ("hard", {"attr_pool": ["length", "mass", "capacity", "duration"], "equal_fraction": 0.333}),
    ],
}

WIRED = STILL_WIRED + [pytest.param(*key, id=f"{key[0]}-direct") for key in DIRECT_VARIANTS]


def _questions(skill_id, topic, seeds=60):
    skill = _SKILLS[skill_id]
    if (skill_id, topic) in DIRECT_VARIANTS:
        variants = DIRECT_VARIANTS[(skill_id, topic)]
    else:
        sources = [s for s in skill.sources if s.topic == topic]
        assert sources, f"{skill_id} has no {topic} source"
        variants = [(s.difficulty, s.params or {}) for s in sources]
    for difficulty, params in variants:
        for seed in range(seeds):
            yield skill, make(topic, seed=seed, difficulty=difficulty, **params)


@pytest.mark.parametrize("skill_id,topic", WIRED)
def test_answer_grades_correct_and_distractors_grade_wrong(skill_id, topic):
    for skill, q in _questions(skill_id, topic, 40):
        options = mcq_options(q)  # an mcq reply the site sends is the option's index; a number is typed

        def reply(text):
            return options.index(text) if skill.answer == "mcq" else text

        assert grade(skill, q, reply(q.answer_text)).correct, (q.text, q.answer_text)
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(skill, q, reply(d)).correct, (q.text, d)
        assert find_text_defects(q) == []


@pytest.mark.parametrize("skill_id,topic", STILL_WIRED)
def test_the_skill_no_longer_draws_the_leaky_source(skill_id, topic):
    old = {"recognise_numeral", "compare_measurement_attributes"}
    assert not [s for s in _SKILLS[skill_id].sources if s.topic in old]


def test_numeral_glyph_is_not_in_the_stem():
    for skill_id in ("recognise-numerals-to-5", "recognise-numerals-to-10"):
        for _skill, q in _questions(skill_id, "ey6a_recognise_numeral"):
            assert not re.search(r"\d", q.text), q.text


def test_compare_attributes_stem_does_not_state_the_comparison():
    forbidden = re.compile(r"\b(longer|heavier|holds more|took longer|lighter|shorter)\b")
    for _skill, q in _questions("f10-compare-measurement-attributes", "ey6a_compare_attributes"):
        assert not forbidden.search(q.text.replace("What can we say?", "")), q.text


@pytest.mark.parametrize(
    "skill_id,topic,distinct",
    [
        ("compare-lengths", "ey6a_compare_length", 40),
        ("continue-patterns", "ey6a_repeating_pattern", 30),
        ("name-basic-shapes", "ey6a_shape_names", 15),
    ],
)
def test_pool_is_larger_than_the_old_one(skill_id, topic, distinct):
    texts = {q.text for _skill, q in _questions(skill_id, topic, 200)}
    assert len(texts) >= distinct


@pytest.mark.parametrize("skill_id,topic", WIRED)
def test_three_prompt_shapes(skill_id, topic):
    shapes = {prompt_shape(q) for _skill, q in _questions(skill_id, topic)}
    assert len(shapes) >= 3


@pytest.mark.parametrize("skill_id,topic", [k for k in DIRECT_VARIANTS])
def test_picture_skills_do_not_draw_the_ey6a_source(skill_id, topic):
    """Main's picture skills own these four skills; the ey6a generators stay unwired (see the module docstring)."""
    assert not [s for s in _SKILLS[skill_id].sources if s.topic == topic]


def test_picture_graph_is_one_to_one_and_graded():
    """Not wired into a skill yet (a Year 1 one-to-one picture graph skill is a separate build), so it is checked
    against the number grader directly."""
    skill = _SKILLS["f2-read-a-picture-graph"]
    for seed in range(60):
        q = make("ey6a_picture_graph_one_to_one", seed=seed)
        assert q.figure["key"]["each"] == 1
        assert find_text_defects(q) == []
        assert grade(skill, q, q.answer_text).correct
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grade(skill, q, d).correct
    assert len({prompt_shape(make("ey6a_picture_graph_one_to_one", seed=s)) for s in range(60)}) >= 3


def test_pattern_continuation_is_the_repeat():
    for _skill, q in _questions("continue-patterns", "ey6a_repeating_pattern", 100):
        cells = q.text.split("  ")[1:] if "  " in q.text else []
        assert "__" in cells
