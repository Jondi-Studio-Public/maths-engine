"""Regression tests for the bug-hunt-2026-10-19 "held" lane: defects whose fix had to wait because an existing
test parsed or pinned the old wording. Each test renders a skill over many seeds and tiers, asserts the defect is
gone from the text a learner reads, and that the key is still accepted and no distractor is.
"""

from __future__ import annotations

import re

import pytest

from mathgen import engine
from mathgen.core import display_question

_SKILLS = {s.id: s for s in engine.load_skills()[1]}
_SEEDS = range(40)


def _rendered(skill_id: str):
    """(source key, seed, question, shown) for every source of a skill over _SEEDS."""
    skill = _SKILLS[skill_id]
    for src in skill.sources:
        key = engine._key_of(skill, src)
        for seed in _SEEDS:
            q = engine.make_for_skill(skill, seed, source=src)
            yield key, seed, q, display_question(q)


def _strings(shown: dict, *, with_working: bool = True):
    out = [shown["text"], shown["answer_text"], *shown["options"]]
    if with_working:
        out += list(shown["working"])
    return out


def _forbid(skill_id: str, pattern: str, *, with_working: bool = True):
    bad = re.compile(pattern)
    for key, seed, _q, shown in _rendered(skill_id):
        for text in _strings(shown, with_working=with_working):
            assert not bad.search(text), f"{skill_id} {key} seed {seed}: {pattern!r} in {text!r}"


def _grading_holds(skill_id: str):
    """The key is accepted and no distractor is, on every source and seed."""
    skill = _SKILLS[skill_id]
    for key, seed, q, _shown in _rendered(skill_id):
        assert engine.grade(skill, q, q.answer_text).correct, f"{skill_id} {key} seed {seed}: key rejected"
        for d in q.distractors:
            assert not engine.grade(skill, q, d).correct, f"{skill_id} {key} seed {seed}: distractor {d!r} accepted"


CYCLIC = "senior-the-opposite-angles-of-a-cyclic-quadrilateral-are"
RHOMBUS = "senior-the-diagonals-of-a-parallelogram-meet-at-right"
EXP_PDF = "senior-use-numerical-integration-using-technology-f-left-t"
GMT = "senior-solve-problems-involving-greenwich-mean-time-and-the"
RATES_A = "senior-complete-calculations-with-rates-including-solving-problems-involving"
RATES_B = "senior-use-rates-to-make-comparisons"
TREND = "senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093"
SETS = "senior-use-everyday-occurrences-to-illustrate-set-descriptions-and"
GAUSS_SEIDEL = "ucv1-iterative-linear-solver-step"
PYRAMID = "square-pyramid-volume-and-surface-area"


@pytest.mark.parametrize("skill_id", [CYCLIC, RHOMBUS])
def test_quadrilateral_theorems_do_not_bracket_a_bare_term(skill_id):
    _forbid(skill_id, r"\(-?\d*x\)")
    _grading_holds(skill_id)


def test_exponential_pdf_has_no_unit_coefficient_or_exponent():
    _forbid(EXP_PDF, r"(?<![\d.])1e\^|\(-1t\)|\^\(-1t")
    _grading_holds(EXP_PDF)


def test_exponential_pdf_with_rate_one_reads_e_to_the_minus_t():
    shown_texts = [shown["text"] for _key, _seed, _q, shown in _rendered(EXP_PDF)]
    assert any("f(t) = e^(-t)" in t for t in shown_texts), "no seed drew a rate of 1 to check"
    assert not any("f(t) = 1e" in t for t in shown_texts)


def test_gmt_has_no_double_possessive_and_no_awkward_layover_clause():
    _forbid(GMT, r"'s'|'s's|'s local|'s GMT")
    _grading_holds(GMT)


def test_compare_unit_prices_options_have_no_double_hyphen_or_wrong_unit():
    _forbid("compare-unit-prices", r"--")
    for _key, seed, _q, shown in _rendered("compare-unit-prices"):
        for option in shown["options"]:
            assert "per unit" not in option, f"seed {seed}: {option!r}"
    _grading_holds("compare-unit-prices")


@pytest.mark.parametrize("skill_id", ["function-notation", "apply-a-rule-to-inputs", "input-output-tables"])
def test_function_evaluation_working_has_no_sign_or_zero_noise(skill_id):
    skill = _SKILLS[skill_id]
    for src in skill.sources:
        if src.topic != "function_evaluation":
            continue
        for seed in _SEEDS:
            q = engine.make_for_skill(skill, seed, source=src)
            line = display_question(q)["working"][0]
            assert not re.search(r"[+-] -\d|\+ 0\b|(?<![\d)])[01]\(|= 0\(", line), f"seed {seed}: {line!r}"


def test_function_evaluation_working_still_adds_up():
    checked = 0
    for skill_id in ("function-notation", "apply-a-rule-to-inputs", "input-output-tables"):
        skill = _SKILLS[skill_id]
        for src in skill.sources:
            if src.topic != "function_evaluation":
                continue
            for seed in _SEEDS:
                q = engine.make_for_skill(skill, seed, source=src)
                expr = q.working[0].split(" = ")[1]
                value = q.working[0].rsplit(" = ", 1)[1]
                python = re.sub(r"(\d|\))\(", r"\1*(", expr).replace("^", "**")
                assert eval(python) == int(value) == q.answer, f"{skill_id} seed {seed}: {q.working[0]!r}"
                checked += 1
    assert checked >= 100


@pytest.mark.parametrize("skill_id", [RATES_A, RATES_B])
def test_rates_use_an_before_a_vowel_sound(skill_id):
    _forbid(skill_id, r"\ba (8\d*|11|18)\b")
    _grading_holds(skill_id)


def test_trend_equation_is_written_without_brackets_or_plus_minus():
    _forbid(TREND, r"\+ \(|\+ -|[+-] 1×t")
    _grading_holds(TREND)


def test_set_descriptions_stem_does_not_repeat_the_options():
    for key, seed, _q, shown in _rendered(SETS):
        assert "Options:" not in shown["text"], f"{key} seed {seed}"
        assert not re.search(r"\b[A-D]: ", shown["text"]), f"{key} seed {seed}"
        assert len(shown["options"]) == 4
    _grading_holds(SETS)


def test_gauss_seidel_equations_are_written_as_on_paper():
    _forbid(GAUSS_SEIDEL, r"\(-?\d+\)[xy]|[ (]1[xy]\b|\+ -|- -")
    _grading_holds(GAUSS_SEIDEL)


def test_pyramid_answer_says_surface_area_not_surface_underscore_area():
    _forbid(PYRAMID, r"surface_area")
    _grading_holds(PYRAMID)
