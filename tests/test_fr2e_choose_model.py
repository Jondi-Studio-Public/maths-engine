"""Tests `f10-choose-model-from-data` (topic `fr2e_choose_model`, exam chain `exd_x_fr2e_choose_model`).

The oracle reads the table back out of the question TEXT and decides the model itself from differences and ratios,
then fits the rule with SymPy (`interpolate` for lines and parabolas, a root of the row ratio for exponentials) and
checks the key and every distractor against the table or the fitted rule.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.fr2e_model_choice_build  # noqa: F401  (registers the topics)
from tests.fr2e_support import (
    SEEDS,
    TIERS,
    assert_chain_shape,
    assert_clean_and_deterministic,
    assert_exam_grading,
    assert_follow_through,
    assert_registered,
    assert_single_grading,
    assert_tiers_differ,
    exam,
    single,
)

SKILL = "f10-choose-model-from-data"
TOPIC = "fr2e_choose_model"
EXAM = "exd_x_fr2e_choose_model"
X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _table(text: str):
    xs, ys = re.search(r"x: ([\d, ]+); y: ([\d, ]+)", text).groups()
    return [int(v) for v in xs.split(",")], [int(v) for v in ys.split(",")]


def _diffs(values):
    return [b - a for a, b in zip(values, values[1:])]


def _kind(ys) -> str:
    d1 = _diffs(ys)
    if len(set(d1)) == 1:
        return "Linear"
    if len(set(_diffs(d1))) == 1:
        return "Quadratic"
    if len({sympy.Rational(b, a) for a, b in zip(ys, ys[1:])}) == 1:
        return "Exponential"
    raise AssertionError(ys)


def _rule(text: str):
    """A rule written like `y = 2 × 3^x` or `y = -x^2 + 10x + 6` as a SymPy expression."""
    body = text.split("=", 1)[1].replace("×", "*").replace("^", "**")
    return parse_expr(body, local_dict={"x": X}, transformations=_TRANSFORMS)


def _fits(rule, xs, ys) -> bool:
    return all(sympy.simplify(rule.subs(X, a) - b) == 0 for a, b in zip(xs, ys))


def _fitted(xs, ys):
    kind = _kind(ys)
    if kind == "Linear":
        return sympy.interpolate(list(zip(xs[:2], ys[:2])), X)
    if kind == "Quadratic":
        return sympy.interpolate(list(zip(xs[:3], ys[:3])), X)
    step = xs[1] - xs[0]
    base = sympy.Rational(ys[1], ys[0]) ** sympy.Rational(1, step)
    return ys[0] * base ** (X - xs[0])


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


def test_easy_names_the_right_model():
    kinds = set()
    for seed in SEEDS:
        q = single(TOPIC, "easy", seed)
        _, ys = _table(q.text)
        assert q.answer_text.startswith(_kind(ys) + ":"), (seed, q.text, q.answer_text)
        d1 = _diffs(ys)
        actual = {"Linear": abs(d1[0]), "Quadratic": abs(_diffs(d1)[0]), "Exponential": sympy.Rational(ys[1], ys[0])}[
            _kind(ys)
        ]
        for wrong in q.distractors:  # a wrong claim names another model, or the right model with the wrong number
            if wrong.startswith(_kind(ys) + ":"):
                number = re.search(r"(\d+|half)", wrong).group(1)
                assert (sympy.Rational(1, 2) if number == "half" else sympy.Integer(number)) != actual, (seed, wrong)
        kinds.add(_kind(ys))
    assert kinds == {"Linear", "Quadratic", "Exponential"}


def test_medium_rule_fits_and_distractors_do_not():
    for seed in SEEDS:
        q = single(TOPIC, "medium", seed)
        xs, ys = _table(q.text)
        assert _fits(_rule(q.answer_text), xs, ys), (seed, q.text, q.answer_text)
        for wrong in q.distractors:
            assert not _fits(_rule(wrong), xs, ys), (seed, q.text, wrong)


def test_hard_predicts_beyond_the_table_by_the_fitted_model():
    for seed in SEEDS:
        q = single(TOPIC, "hard", seed)
        xs, ys = _table(q.text)
        target = int(re.search(r"when x = (\d+)", q.text).group(1))
        assert target not in xs and target > xs[-1]
        step = int(re.search(r"x goes up by (\d+)", q.text).group(1))
        assert step == xs[1] - xs[0] and step > 1
        assert sympy.simplify(_fitted(xs, ys).subs(X, target) - int(q.answer_text)) == 0, (seed, q.text, q.answer_text)
        assert str(int(q.answer_text)) not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_tiers_differ_in_shape(tier):
    assert_tiers_differ(TOPIC)
    questions = [single(TOPIC, tier, seed) for seed in SEEDS]
    if tier == "easy":
        assert all("Which statement about the table is true?" in q.text for q in questions)
    if tier == "medium":
        assert all("Which rule fits every value" in q.text for q in questions)
    if tier == "hard":
        assert all("when x =" in q.text and "goes up by" in q.text for q in questions)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def _exam_limit(text: str, rule):
    """The first whole x after the table at which the rule breaks the stated limit, read from the part (d) sentence."""
    sentence = re.search(r"\) ([^()]*?) Find the first whole number", text).group(1)
    lower = re.search(r"negative|below|less than", sentence) is not None
    if lower:
        bound = 0 if re.search(r"negative|ground", sentence) else 1
        return next(v for v in range(0, 400) if rule.subs(X, v) < bound)
    bound = int(re.search(r"(\d+)", sentence).group(1))
    return next(v for v in range(0, 400) if rule.subs(X, v) > bound)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        xs, ys = _table(q.text)
        a, b, c, d = q.parts
        assert a.answer_text.startswith(_kind(ys) + ":"), (seed, q.text)
        rule = _rule(b.answer_text)
        assert _fits(rule, xs, ys)
        assert sympy.simplify(rule - _fitted(xs, ys)) == 0
        target = int(re.search(r"x = (\d+)", c.prompt).group(1))
        assert int(c.answer_text) == rule.subs(X, target), (seed, q.text)
        assert int(d.answer_text) == _exam_limit(q.text, rule), (seed, q.text, d.answer_text)
        assert int(d.answer_text) > xs[-1]


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["mcq", "expression", "number", "number"], [2, 3, 2, 2], [False, False, True, False])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_prediction_follows_through_from_a_wrong_rule():
    """A wrong rule in part (b) still earns part (c) when the prediction is that rule evaluated at the stated x."""

    def predicted(q, canon, wrong_rule):
        try:
            rule = _rule(wrong_rule)
        except Exception:
            return None
        target = int(re.search(r"x = (\d+)", q.parts[2].prompt).group(1))
        value = rule.subs(X, target)
        return str(int(value)) if value == int(value) and value >= 0 else None

    assert_follow_through(SKILL, EXAM, "c", "b", predicted)
