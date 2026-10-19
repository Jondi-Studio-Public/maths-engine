"""Grading fixes from the 2026-10-19 bug hunt: units with a middle dot, angle units, whole-number keys on fraction
skills, bracketed coefficients, a unit the question asks for on an expression, "15c" for cents, dict reply to a table."""

import json

import pytest

from mathgen import engine
from mathgen.skills import load_skills

_SKILLS = {s.id: s for s in load_skills()[1]}


def _q(skill_id, seed=0, source=0):
    skill = _SKILLS[skill_id]
    return skill, engine.make_for_skill(skill, seed, source=source)


def _correct(skill, question, reply):
    return engine.grade(skill, question, reply).correct


def _number_of(question):
    return question.answer_text.rsplit("=", 1)[-1].split()[0].rstrip("°")


@pytest.mark.parametrize("skill_id", ["bending-moment-point-load", "max-bending-moment-udl", "moment-of-a-force"])
@pytest.mark.parametrize("seed", range(6))
def test_moment_key_refuses_other_units_and_accepts_its_own(skill_id, seed):
    skill, q = _q(skill_id, seed)
    n = _number_of(q)
    unit = q.answer_text.split(None, 1)[-1].split(None, 2)[-1]  # "kN·m" / "N·m"
    for ok in (q.answer_text, f"{n} {unit}", f"{n} {unit.replace('·', ' ')}", n):
        assert _correct(skill, q, ok), ok
    for bad in ("kg", "s", "J", "kN", "N", "m", "kN/m", "dollars"):
        assert not _correct(skill, q, f"{n} {bad}"), bad


@pytest.mark.parametrize("skill_id,source", [("right-triangle-angle", 0), ("cosine-rule", 3), ("sine-rule", 2)])
@pytest.mark.parametrize("seed", range(6))
def test_degree_key_refuses_other_angle_units(skill_id, source, seed):
    skill, q = _q(skill_id, seed, source)
    n = _number_of(q)
    assert _correct(skill, q, q.answer_text) and _correct(skill, q, f"{n}°") and _correct(skill, q, n)
    for bad in ("rad", "radians", "radian", "grad", "gon", "turns"):
        assert not _correct(skill, q, f"{n} {bad}"), bad


def test_radian_answers_still_take_their_unit():
    skill, q = _q("angle-degrees-to-radians-decimal")
    for ok in (q.answer_text, f"{q.answer_text} rad", f"{q.answer_text} radians"):
        assert _correct(skill, q, ok), ok
    skill, q = _q("angle-degrees-radians-pi-form")
    for ok in ("pi/4", "pi/4 rad", "π/4 radians", "pi/4 radian"):
        assert _correct(skill, q, ok), ok
    assert not _correct(skill, q, "pi/4 degrees") and not _correct(skill, q, "pi/3 rad")


@pytest.mark.parametrize(
    "skill_id,source,seeds",
    [
        ("f10-algebra-conjecture-algorithm", 0, range(8)),
        ("accumulation-function", 0, range(8)),
        ("ucv1-piecewise-continuity", 0, range(8)),
        ("uc2-newton-divided-differences", 0, range(8)),
    ],
)
def test_whole_number_key_on_a_fraction_skill_is_graded_exactly(skill_id, source, seeds):
    for seed in seeds:
        skill, q = _q(skill_id, seed, source)
        if skill.answer != "fraction" or not isinstance(q.answer, int):
            continue
        key = q.answer
        assert _correct(skill, q, str(key)) and _correct(skill, q, f"{key}.0")
        for bad in (f"{key + 0.04}", f"{key - 0.04}", f"{key}!", f"{key} !", f"{key}!!"):
            assert not _correct(skill, q, bad), (seed, bad)


def test_bracketed_coefficient_is_expanded():
    skill, q = _q("expand-single-brackets", 1, 2)
    assert q.answer_text == "2*x/3 - 3*y - 1/3"
    for ok in ("2x/3 - 3y - 1/3", "(2/3)x - 3y - 1/3", "(2/3)*x - 3*y - 1/3", "(2x)/3 - 3y - 1/3"):
        assert _correct(skill, q, ok), ok
    for bad in ("1/3(2x - 9y - 1)", "(2x - 9y - 1)/3", "(1/3)(2x - 9y - 1)", "2x/3 - 3y"):
        assert not _correct(skill, q, bad), bad


def test_an_expression_takes_the_unit_the_question_asks_for():
    skill, q = _q("senior-use-formulas-to-find-the-volume-of-pyramids", 0, 1)
    n = q.answer_text
    for ok in (n, f"{n} cm3", f"{n}cm^3", f"{n} cm³"):
        assert _correct(skill, q, ok), ok
    for bad in (f"{n} cm", f"{n} m", f"{n} kg"):
        assert not _correct(skill, q, bad), bad


@pytest.mark.parametrize("seed", range(6))
def test_cents_may_be_typed_with_c(seed):
    skill, q = _q("f10-money-count-collection", seed)
    n = q.answer_text
    for ok in (n, f"{n}c", f"{n} c", f"{n} cents"):
        assert _correct(skill, q, ok), ok
    assert not _correct(skill, q, f"{int(n) + 1}c") and not _correct(skill, q, f"{n}x")


def test_a_table_reply_may_be_a_dict():
    skill, q = _q("f10-two-way-table-relationship")
    blanks = {k: str(v) for k, v in q.answer["blanks"].items()}
    assert _correct(skill, q, blanks) and _correct(skill, q, json.dumps(blanks))
    assert not _correct(skill, q, {k: "999" for k in blanks})
