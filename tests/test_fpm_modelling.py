"""Tests for the three Years 7-8 modelling skills built in `fpm_modelling_build`.

`translate-word-problems-to-equations`, `linear-models` and `model-relationships-with-functions`. Each key is
re-derived from the rendered stem: an expression must solve to a whole number for the stated total, a model's
prediction is recomputed from the readings in the text, and a function rule must reproduce every row of the table
that is printed in the question while no distractor does.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.fpm_modelling_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SKILLS = {s.id: s for s in load_skills()[1]}
TIERS = ("easy", "medium", "hard")
TRANSLATE = "translate-word-problems-to-equations"
LINEAR = "linear-models"
FUNCTION = "model-relationships-with-functions"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse(text: str):
    return parse_expr(text, transformations=_TRANSFORMS)


def single(skill_id: str, tier: str, seed: int):
    src = next(s for s in SKILLS[skill_id].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=tier, **src.params)


def exam(skill_id: str, seed: int):
    src = next(s for s in SKILLS[skill_id].sources if s.difficulty == "exam")
    return make(src.topic, seed=seed, difficulty="exam", **src.params)


# -- wiring and the checks shared by every tier -------------------------------------------------------------------


@pytest.mark.parametrize("skill_id,kind", [(TRANSLATE, "expression"), (LINEAR, "number"), (FUNCTION, "expression")])
def test_skill_wiring(skill_id, kind):
    skill = SKILLS[skill_id]
    assert skill.status == "active" and skill.answer == kind
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [s.stages for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    for prerequisite in skill.prerequisites:
        assert prerequisite in SKILLS


@pytest.mark.parametrize("skill_id", [TRANSLATE, LINEAR, FUNCTION])
@pytest.mark.parametrize("tier", TIERS)
def test_canonical_correct_and_distractors_wrong(skill_id, tier):
    skill = SKILLS[skill_id]
    for seed in range(150):
        q = single(skill_id, tier, seed)
        assert grade(skill, q, q.answer_text).correct, (q.text, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (q.text, q.distractors)
        for d in q.distractors:
            assert not grade(skill, q, d).correct, (q.text, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (q.text, d, q.option_notes)
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))


@pytest.mark.parametrize("skill_id", [TRANSLATE, LINEAR, FUNCTION])
def test_tiers_differ_in_shape(skill_id):
    shapes = {tier: {prompt_shape(single(skill_id, tier, s)) for s in range(60)} for tier in TIERS}
    assert all(len(v) >= 2 for v in shapes.values())
    assert shapes["easy"] != shapes["hard"] != shapes["medium"]


@pytest.mark.parametrize("skill_id", [TRANSLATE, LINEAR, FUNCTION])
def test_deterministic(skill_id):
    assert single(skill_id, "hard", 9).to_dict() == single(skill_id, "hard", 9).to_dict()


# -- translate word problems --------------------------------------------------------------------------------------


def _total_and_variable(q):
    total = int(re.search(r"(?:right-hand side is|has|____ =) (\d+)", q.text).group(1))
    var = re.search(r"expression in (\w)", q.text).group(1)
    return total, var


@pytest.mark.parametrize("tier", TIERS)
def test_translate_expression_solves_to_a_whole_number(tier):
    var_counts = Counter()
    for seed in range(250):
        q = single(TRANSLATE, tier, seed)
        total, var = _total_and_variable(q)
        symbol = sympy.Symbol(var)
        expr = parse(q.answer_text)
        solutions = sympy.solve(sympy.Eq(expr, total), symbol)
        assert len(solutions) == 1 and solutions[0].is_integer and solutions[0] > 0, (q.text, q.answer_text)
        assert sympy.degree(expr, symbol) == 1
        var_counts[(tier, q.answer_text.count(var) > 1 or "(" in q.answer_text)] += 1
    # the hard tier writes a second quantity in terms of the first; the easy tier never needs to
    if tier == "hard":
        assert var_counts[("hard", True)] == 250
    if tier == "easy":
        assert var_counts[("easy", False)] == 250


def test_translate_hard_has_four_situations():
    kinds = Counter()
    for seed in range(200):
        text = single(TRANSLATE, "hard", seed).text
        kinds[
            "tickets"
            if "Adult tickets" in text
            else "bracket"
            if "thinks of" in text
            else "perimeter"
            if "rectangle" in text
            else "consecutive"
        ] += 1
    assert len(kinds) == 4 and min(kinds.values()) >= 20, kinds


def test_translate_distractors_do_not_solve_the_same_way():
    """A distractor is a different expression; most also have a different solution for the same total."""
    different = 0
    for seed in range(100):
        q = single(TRANSLATE, "medium", seed)
        total, var = _total_and_variable(q)
        symbol = sympy.Symbol(var)
        good = sympy.solve(sympy.Eq(parse(q.answer_text), total), symbol)
        for d in q.distractors:
            sol = sympy.solve(sympy.Eq(parse(d), total), symbol)
            different += sol != good
    assert different >= 250


def test_translate_exam_chain():
    skill = SKILLS[TRANSLATE]
    for seed in range(120):
        q = exam(TRANSLATE, seed)
        fixed, rate = (
            (int(x) for x in re.search(r"\$(\d+) .*? plus \$(\d+)|\$(\d+) .*? \$(\d+)", q.text).groups()[:2])
            if False
            else (None, None)
        )
        amounts = [int(x) for x in re.findall(r"\$(\d+)", q.stem)]
        a, b, total, c = amounts[0], amounts[1], amounts[2], amounts[3]
        var = re.search(r"Let (\w) be", q.parts[0].prompt).group(1)
        symbol = sympy.Symbol(var)
        x0 = (total - a) // b
        assert a + b * x0 == total
        assert sympy.simplify(parse(q.parts[0].answer_text) - (b * symbol + a)) == 0
        assert [p.answer_text for p in q.parts[1:]] == [str(x0), str(c * x0), str(total - c * x0)]
        assert [p.marks for p in q.parts] == [2, 2, 1, 1]
        assert [p.answer_kind for p in q.parts] == ["expression", "number", "number", "number"]
        assert [p.follow_through for p in q.parts][2:] == [f"{c}*part_b", f"{total} - part_c"]
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == 6


def test_translate_exam_distractors_and_follow_through():
    skill = SKILLS[TRANSLATE]
    for seed in range(100):
        q = exam(TRANSLATE, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        amounts = [int(x) for x in re.findall(r"\$(\d+)", q.stem)]
        total, c = amounts[2], amounts[3]
        wrong_b = int(canon["b"]) + 1
        reply = {**canon, "b": str(wrong_b), "c": str(c * wrong_b), "d": str(total - c * wrong_b)}
        if reply["c"] == reply["d"]:  # a reply that repeats the earlier value is never follow-through
            continue
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through
        assert result.parts["d"].marks == result.parts["d"].max_marks


# -- linear models ------------------------------------------------------------------------------------------------


def _readings(text: str):
    m = re.search(r"was (\d+) \w+ after (\d+) \w+ and (\d+) \w+ after (\d+) \w+", text)
    return tuple(int(x) for x in m.groups())  # y1, x1, y2, x2


def linear_oracle(tier: str, text: str) -> int:
    if tier == "easy":
        rule = re.search(r"rule (\w) = (.*?), where (\w) is", text)
        var = sympy.Symbol(rule.group(3) if False else re.search(r"(\w) is the number of", text).group(1))
        value = int(re.search(r"after (\d+) ", text.split("Use the model")[1]).group(1))
        return int(parse(rule.group(2).replace("−", "-")).subs(var, value))
    if tier == "medium":
        y1, x1, y2, x2 = _readings(text)
        x3 = int(re.search(r"predict .*? after (\d+) ", text).group(1))
        return int(Fraction(y1) + Fraction(y2 - y1, x2 - x1) * (x3 - x1))
    if "will it be" in text:
        y1, x1, y2, x2 = _readings(text)
        target = int(re.search(r"will it be (\d+)", text).group(1))
        return int(x1 + Fraction(target - y1) / Fraction(y2 - y1, x2 - x1))
    if "cost exactly the same" in text:
        a, b, c = (int(x) for x in re.findall(r"\$(\d+)", text)[:3])
        return a // (c - b)
    rule = re.search(r"modelled by (\w) = (\d+) - (\d+)(\w),", text)
    return int(rule.group(2)) // int(rule.group(3))


@pytest.mark.parametrize("tier", TIERS)
def test_linear_oracle(tier):
    kinds = Counter()
    for seed in range(300):
        q = single(LINEAR, tier, seed)
        assert int(q.answer_text) == linear_oracle(tier, q.text), (q.text, q.answer_text)
        if tier == "hard":
            kinds["valid" if "largest whole" in q.text else "plans" if "exactly the same" in q.text else "target"] += 1
    if tier == "hard":
        assert set(kinds) == {"valid", "plans", "target"} and min(kinds.values()) >= 40, kinds


def test_linear_valid_range_is_not_a_whole_division():
    for seed in range(150):
        q = single(LINEAR, "hard", seed)
        if "largest whole" not in q.text:
            continue
        rule = re.search(r"modelled by (\w) = (\d+) - (\d+)(\w),", q.text)
        start, step = int(rule.group(2)), int(rule.group(3))
        assert start % step != 0
        assert start - step * int(q.answer_text) >= 0 > start - step * (int(q.answer_text) + 1)


def test_linear_models_include_rising_and_falling_quantities():
    texts = " ".join(single(LINEAR, "medium", s).text for s in range(100))
    for fragment in ("sunflower", "savings", "tank", "candle"):
        assert fragment in texts


def test_linear_exam_chain():
    skill = SKILLS[LINEAR]
    for seed in range(150):
        q = exam(LINEAR, seed)
        y1, x1, y2, x2 = _readings(q.stem)
        rate = Fraction(y2 - y1, x2 - x1)
        start = y1 - rate * x1
        x0 = int(re.search(r"after (\d+) \w+, in", q.parts[2].prompt).group(1))
        target = int(re.search(r"will .*? be (\d+)", q.parts[3].prompt).group(1))
        want = [str(int(rate)), str(int(start)), str(int(start + rate * x0)), str(int((target - start) / rate))]
        assert [p.answer_text for p in q.parts] == want, q.text
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert min(int(w) for w in want[1:]) > 0
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_linear_exam_distractors_and_follow_through():
    skill = SKILLS[LINEAR]
    for seed in range(100):
        q = exam(LINEAR, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        y1, x1, y2, x2 = _readings(q.stem)
        x0 = int(re.search(r"after (\d+) \w+, in", q.parts[2].prompt).group(1))
        target = int(re.search(r"will .*? be (\d+)", q.parts[3].prompt).group(1))
        rate = Fraction(y2 - y1, x2 - x1) + 1  # a slip in (a), carried through
        start = y1 - rate * x1
        value = start + rate * x0
        steps = Fraction(target) - start
        if start <= 0 or value <= 0 or steps <= 0 or (steps / rate).denominator != 1:
            continue
        reply = {"a": str(int(rate)), "b": str(int(start)), "c": str(int(value)), "d": str(int(steps / rate))}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert all(result.parts[x].marks == result.parts[x].max_marks for x in "bcd"), (q.text, reply, result)


# -- functions ----------------------------------------------------------------------------------------------------


def _table(text: str):
    m = re.search(r"(\w) = (\d+(?:, \d+)*) gives (\w) = (\d+(?:, \d+)*)", text)
    xs = [int(x) for x in m.group(2).split(",")]
    ys = [int(y) for y in m.group(4).split(",")]
    return m.group(1), xs, ys


def _fits(expression: str, var: str, xs, ys) -> bool:
    expr = parse(expression)
    return all(expr.subs(sympy.Symbol(var), x) == y for x, y in zip(xs, ys))


@pytest.mark.parametrize("tier", TIERS)
def test_function_rule_fits_the_table_and_no_distractor_does(tier):
    families = Counter()
    for seed in range(250):
        q = single(FUNCTION, tier, seed)
        var, xs, ys = _table(q.text)
        assert len(xs) == len(ys) >= 4
        assert _fits(q.answer_text, var, xs, ys), (q.text, q.answer_text)
        for d in q.distractors:
            assert not _fits(d, var, xs, ys), (q.text, d)
        if tier == "hard":
            families["inverse" if "/" in q.answer_text else "square" if "^2" in q.answer_text else "falling"] += 1
    if tier == "hard":
        assert set(families) == {"inverse", "square", "falling"} and min(families.values()) >= 30, families


def test_function_easy_is_proportional_and_medium_has_a_fixed_amount():
    for seed in range(100):
        easy = single(FUNCTION, "easy", seed)
        var, xs, ys = _table(easy.text)
        assert len({Fraction(y, x) for x, y in zip(xs, ys)}) == 1
        med = single(FUNCTION, "medium", seed)
        var, xs, ys = _table(med.text)
        assert len({Fraction(y, x) for x, y in zip(xs, ys)}) > 1
        assert xs != list(range(xs[0], xs[0] + len(xs))), "medium inputs should skip values"


def test_function_exam_chain():
    skill = SKILLS[FUNCTION]
    for seed in range(150):
        q = exam(FUNCTION, seed)
        var, xs, ys = _table(q.stem)
        products = {x * y for x, y in zip(xs, ys)}
        assert len(products) == 1
        k = products.pop()
        x0 = int(re.search(r"when \w = (\d+)", q.parts[2].prompt).group(1))
        target = int(re.search(r"when \w = (\d+)", q.parts[3].prompt).group(1))
        assert [q.parts[0].answer_text, q.parts[2].answer_text, q.parts[3].answer_text] == [
            str(k),
            str(k // x0),
            str(k // target),
        ]
        assert k % x0 == 0 and k % target == 0
        assert _fits(q.parts[1].answer_text, var, xs, ys)
        assert q.parts[2].answer_text != q.parts[3].answer_text
        assert [p.answer_kind for p in q.parts] == ["number", "expression", "number", "number"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through for p in q.parts][2:] == [f"part_a/{x0}", f"part_a/{target}"]
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == 6
        assert grade(skill, q, {**{p.label: p.answer_text for p in q.parts}, "b": f"{q.parts[1].answer_text}"}).correct


def test_function_exam_distractors_and_follow_through():
    skill = SKILLS[FUNCTION]
    for seed in range(100):
        q = exam(FUNCTION, seed)
        var, xs, ys = _table(q.stem)
        canon = {p.label: p.answer_text for p in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        k = xs[0] * ys[0]
        x0 = int(re.search(r"when \w = (\d+)", q.parts[2].prompt).group(1))
        target = int(re.search(r"when \w = (\d+)", q.parts[3].prompt).group(1))
        wrong_k = k * 2
        reply = {"a": str(wrong_k), "b": canon["b"], "c": str(wrong_k // x0), "d": str(wrong_k // target)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert (
            result.parts["c"].marks == result.parts["c"].max_marks
            and result.parts["d"].marks == result.parts["d"].max_marks
        )


@pytest.mark.parametrize("skill_id", [TRANSLATE, LINEAR, FUNCTION])
def test_exam_all_wrong_earns_nothing_and_text_is_clean(skill_id):
    skill = SKILLS[skill_id]
    for seed in range(40):
        q = exam(skill_id, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert grade(skill, q, {"a": "zzz", "b": "-1234", "c": "99999", "d": "888888"}).marks == 0
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))
    assert len({prompt_shape(exam(skill_id, s)) for s in range(40)}) >= 2
