"""Tests for `mathgen/generators/unknown_add_sub_equation_build.py`: `unknown_add_sub_equation` (skill
`f10-unknown-add-sub-number-equations`).

Generator-specific proofs: the equation shapes each tier draws, the solver reading the rendered equation, the
typed-number grader battery, key rank, and skill load. The independent oracle is
`tests/test_unknown_add_sub_equation_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.unknown_add_sub_equation_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "unknown_add_sub_equation"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def _equation(q):
    side = r"(?:\d+|□)(?: [+-] (?:\d+|□))*"
    m = re.search(rf"({side}) = ({side})$", q.text)
    assert m, q.text
    return m.group(1), m.group(2)


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, range(40)):
        assert find_text_defects(q) == [], f"{tier}: {find_text_defects(q)} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes_and_20_distinct_questions(tier):
    qs = _all(tier)
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 7, 59):
        assert make(TOPIC, seed=seed, difficulty=tier).to_json() == make(TOPIC, seed=seed, difficulty=tier).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_one_blank_and_the_key_is_not_already_in_the_equation(tier):
    for q in _all(tier, range(150)):
        left, right = _equation(q)
        assert (left + right).count("□") == 1
        assert str(q.answer) not in re.findall(r"\d+", f"{left} = {right}"), q.text


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_three_distinct_positive_whole_numbers_never_the_key(tier):
    for q in _all(tier, range(200)):
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors
        assert all(d.isdigit() and int(d) > 0 for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor_and_slip(tier):
    for q in _all(tier, range(100)):
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        for wrong in (q.answer + 1, q.answer - 1, q.answer + 10):
            assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_key_rank_among_the_options_is_spread_out(tier):
    ranks = Counter()
    for q in _all(tier, range(300)):
        ranks[sorted([q.answer, *map(int, q.distractors)]).index(q.answer)] += 1
    assert set(ranks) == {0, 1, 2, 3}, dict(ranks)
    assert max(ranks.values()) / sum(ranks.values()) <= 0.45, dict(ranks)


@pytest.mark.parametrize("tier", TIERS)
def test_no_single_answer_dominates(tier):
    counts = Counter(q.answer_text for q in _all(tier, range(300)))
    assert counts.most_common(1)[0][1] / 300 < 0.10, counts.most_common(3)


# -- tier progression ------------------------------------------------------------------------------------------


def test_easy_has_one_operation_the_blank_against_a_single_total_and_all_four_shapes():
    shapes = Counter()
    for q in _all("easy", range(200)):
        left, right = _equation(q)
        assert len(re.findall(r"[+-]", left + right)) == 1
        assert re.fullmatch(r"\d+", right) or re.fullmatch(r"\d+", left), q.text
        assert max(int(n) for n in re.findall(r"\d+", f"{left} = {right}")) <= 99
        shapes[re.sub(r"\d+", "n", left + " = " + right)] += 1
    assert set(shapes) == {"n + □ = n", "□ + n = n", "n - □ = n", "□ - n = n"}, shapes


def test_medium_has_an_expression_on_both_sides():
    shapes = Counter()
    for q in _all("medium", range(200)):
        left, right = _equation(q)
        assert re.search(r"[+-]", left) and re.search(r"[+-]", right), q.text
        shapes[re.sub(r"\d+", "n", left + " = " + right)] += 1
    assert len(shapes) == 4, shapes


def test_hard_uses_three_digit_numbers_and_all_three_shapes():
    shapes = Counter()
    for q in _all("hard", range(200)):
        left, right = _equation(q)
        assert max(int(n) for n in re.findall(r"\d+", f"{left} = {right}")) >= 100
        shapes[re.sub(r"\d+", "n", left + " = " + right)] += 1
    assert set(shapes) == {"n + n = n + □", "n - n = n - □", "n + □ - n = n", "□ - n + n = n"}, shapes


def test_hard_compensation_shifts_a_number_to_a_round_one():
    seen = 0
    for q in _all("hard", range(200)):
        left, right = _equation(q)
        if right.endswith("□") and re.fullmatch(r"\d+ [+-] □", right):
            first = int(right.split()[0])
            seen += 1
            assert first % 100 == 0, q.text
    assert seen > 40


# -- solver ---------------------------------------------------------------------------------------------------


def test_solver_handles_every_blank_position_and_side():
    assert gen._solve_equation_text("27 + □ = 61") == 34
    assert gen._solve_equation_text("□ + 27 = 61") == 34
    assert gen._solve_equation_text("61 - □ = 27") == 34
    assert gen._solve_equation_text("□ - 27 = 61") == 88
    assert gen._solve_equation_text("27 + 15 = □ + 20") == 22
    assert gen._solve_equation_text("63 - 28 = 80 - □") == 45
    assert gen._solve_equation_text("63 - 28 = □ - 20") == 55
    assert gen._solve_equation_text("398 + 245 = 400 + □") == 243
    assert gen._solve_equation_text("135 + □ - 48 = 400") == 313


def test_solver_fires_on_no_blank_a_cancelled_blank_or_a_fractional_solution():
    for bad in ("27 + 5 = 32", "□ + 5 = □ + 7", "□ + 5 = 6 - 1 + 0 + □"):
        with pytest.raises(VerificationError):
            gen._solve_equation_text(bad)


def test_a_key_that_disagrees_with_the_rendered_equation_fires_the_check(monkeypatch):
    original = gen._BUILDERS["add_right"]

    def wrong(rng):
        left, right, key, raw = original(rng)
        return left, right, key + 1, raw

    monkeypatch.setitem(gen._BUILDERS, "add_right", wrong)
    monkeypatch.setattr(gen, "_EASY_SHAPES", ("add_right",))
    monkeypatch.setitem(gen._SHAPES, "easy", ("add_right",))
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-unknown-add-sub-number-equations")
        assert skill.domain == "equations"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["f10-addition-facts-related-subtraction", "one-step-equations"]
        assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard"]
        stages = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages.update(source.stages)
        assert stages == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
