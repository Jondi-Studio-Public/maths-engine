"""Round 4 grader fixes: displayed key accepted inside tuples and sets, and a bounded factorised-as-far-as check."""

from __future__ import annotations

import time

import pytest

from mathgen import engine
from mathgen.core import display_question

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


def _displayed_key_accepted(skill_id: str, source: int, seed: int) -> tuple[bool, str]:
    skill = _SKILLS[skill_id]
    question = engine.make_for_skill(skill, seed, source=source)
    shown = display_question(question)["answer_text"]
    return engine.grade(skill, question, shown).correct, shown


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", range(12))
def test_curl_displayed_key_with_euler_constant_is_accepted(source, seed):
    ok, shown = _displayed_key_accepted("uc2-curl-of-field", source, seed)
    assert ok, shown


def test_curl_tuple_reads_e_as_euler_in_every_spelling():
    skill = _SKILLS["uc2-curl-of-field"]
    question = engine.make_for_skill(skill, 5, source=2)  # key (-2, 1 + 3*exp(1), 0)
    for reply in ("(-2, 1 + 3e, 0)", "(-2, 1 + 3*e, 0)", "(-2, 1 + 3*exp(1), 0)"):
        assert engine.grade(skill, question, reply).correct, reply
    assert not engine.grade(skill, question, "(-2, 1 + 3, 0)").correct


@pytest.mark.parametrize("source", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", range(10))
def test_trig_equations_radians_list_accepts_the_displayed_pi_form(source, seed):
    ok, shown = _displayed_key_accepted("trig-equations-radians-list", source, seed)
    assert ok, shown


def test_trig_equations_radians_list_still_rejects_wrong_sets():
    skill = _SKILLS["trig-equations-radians-list"]
    question = engine.make_for_skill(skill, 5, source=0)  # {pi/6, 5pi/6}
    for reply in ("{π/6, 5π/6}", "π/6, 5π/6", "{5π/6, π/6}", "{pi/6, 5pi/6}"):
        assert engine.grade(skill, question, reply).correct, reply
    for reply in ("{π/6, 7π/6}", "{π/6}", "{π/6, 5π/6, 7π/6}"):
        assert not engine.grade(skill, question, reply).correct, reply


@pytest.mark.parametrize("skill_id", ["factorise-quadratic", "factor-common-numerical-factor"])
@pytest.mark.parametrize("reply", ["cos((E-2)^6)", "sec((E-2)^6)", "sec((E-2)^4)*(x-2)*(x-5)", "csc((E-1)^5)"])
def test_hostile_constant_factor_is_rejected_quickly(skill_id, reply):
    skill = _SKILLS[skill_id]
    question = engine.make_for_skill(skill, 1, source=0)
    start = time.perf_counter()
    result = engine.grade(skill, question, reply)
    assert time.perf_counter() - start < 2.0
    assert not result.correct


def test_factorised_check_still_accepts_trivial_constants_and_rejects_a_real_one():
    skill = _SKILLS["factorise-quadratic"]
    question = engine.make_for_skill(skill, 1, source=0)  # (x - 2)(x - 5)
    for reply in ("(x-2)(x-5)", "1*(x-2)(x-5)", "sin(pi/2)*(x-2)(x-5)"):
        assert engine.grade(skill, question, reply).correct, reply
    for reply in ("3(x-2)(x-5)", "-(x-2)(x-5)"):
        assert not engine.grade(skill, question, reply).correct, reply
