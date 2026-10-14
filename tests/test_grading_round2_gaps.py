"""Round 2 marking gaps: an MCQ option named by its letter or text, fractions written with brackets, LaTeX or
another fraction bar, derivative labels and the division sign in expressions, mixed numbers, a pair's labels,
and lowest terms where the question asks for them."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mathgen import engine
from mathgen.core import Part, Question, mcq_options
from mathgen.grading import grade
from mathgen.grading.units import unit_is_acceptable


def _q(answer, answer_text, text="q", **extra):
    return Question(topic="test", level=0, seed=1, text=text, latex="", answer=answer, answer_text=answer_text, **extra)


# -- G1: an MCQ option by its letter or its text -------------------------------------------


def _mcq_skill_question(seed=1):
    skill = next(s for s in engine.load_skills()[1] if s.status == "active" and s.answer == "mcq")
    return skill, engine.make_for_skill(skill, seed=seed)


def test_mcq_option_by_letter_and_by_text():
    for seed in range(5):
        skill, question = _mcq_skill_question(seed)
        options = mcq_options(question)
        right = options.index(question.answer_text)
        wrong = (right + 1) % len(options)
        if not any(len(o) == 1 and o.isalpha() for o in options):
            assert engine.grade(skill, question, "ABCDEFGH"[right]).correct
            assert engine.grade(skill, question, f"({'abcdefgh'[right]})").correct
            assert not engine.grade(skill, question, "ABCDEFGH"[wrong]).correct
        assert engine.grade(skill, question, f"  {question.answer_text.upper()} ").correct
        assert not engine.grade(skill, question, options[wrong]).correct


def test_mcq_digits_equal_to_an_option_text_are_that_option():
    question = _q("3", "3", distractors=["4", "5"])
    options = mcq_options(question)
    # the text "3" is the option whose text is "3", even though 3 is also an index-shaped number
    assert grade("mcq", question, "3").correct
    assert not grade("mcq", question, "4").correct
    # a digit string matching no option's text is still read as an index; an int is always an index
    assert grade("mcq", question, str(options.index("3"))).correct == ("3" == options[options.index("3")])
    assert grade("mcq", question, options.index("3")).correct


def test_mcq_letter_is_not_read_when_an_option_is_a_letter():
    question = _q("x", "x", distractors=["y", "z"])
    assert grade("mcq", question, "x").correct
    assert not grade("mcq", question, "y").correct


def _mcq_part(label, seed, answer="7"):
    return Part(
        label=label,
        prompt="p",
        answer_kind="mcq",
        answer=answer,
        answer_text=answer,
        distractors=["8", "9"],
        seed=f"{seed}:{label}",
    )


def _mcq_part_question(seed):
    parts = [_mcq_part("a", seed), _mcq_part("b", seed)]
    return _q("7", "7", parts=parts), parts


def test_mcq_index_as_int_for_every_option():
    for seed in range(5):
        skill, question = _mcq_skill_question(seed)
        options = mcq_options(question)
        right = options.index(question.answer_text)
        for index in range(len(options)):
            assert engine.grade(skill, question, index).correct == (index == right)
            assert engine.grade(skill, question, str(index)).correct == (index == right)


@pytest.mark.parametrize("given", [-1, 99, 4.0, True, False, None])
def test_mcq_index_out_of_range_or_not_an_int_is_wrong(given):
    skill, question = _mcq_skill_question()
    assert not engine.grade(skill, question, given).correct


def _seed_with_right_option_at(index):
    return next(s for s in range(1, 200) if mcq_options(_mcq_part("a", s)).index("7") == index)


def test_mcq_part_index_zero_is_a_reply_not_an_empty_box():
    # 0 is falsy: `reply or ""` once turned the first option's index into "no reply" for a part.
    seed = _seed_with_right_option_at(0)
    question, parts = _mcq_part_question(seed)
    right_b = mcq_options(parts[1]).index("7")
    for given in (
        {"a": 0, "b": right_b},
        {"a": "0", "b": str(right_b)},
        json.dumps({"a": 0, "b": right_b}),
    ):
        assert grade("mcq", question, given).marks == 2, given


def test_mcq_part_index_as_int_in_a_reply_dict():
    for seed in range(1, 20):
        question, parts = _mcq_part_question(seed)
        right_a, right_b = (mcq_options(part).index("7") for part in parts)
        assert grade("mcq", question, {"a": right_a, "b": right_b}).marks == 2
        assert grade("mcq", question, {"a": right_a, "b": True}).marks == 1


def test_one_part_question_takes_an_int_index_including_zero():
    for seed in range(1, 20):
        part = _mcq_part("a", seed)
        question = _q("7", "7", parts=[part])
        assert grade("mcq", question, mcq_options(part).index("7")).correct


# -- G3: fraction notations ------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply", ["11/19", "(11/19)", "\\frac{11}{19}", "\\dfrac{11}{19}", "11⁄19", "11 ∕ 19", "11 ÷ 19", "x = (11/19)"]
)
def test_fraction_notations_are_read(reply):
    assert grade("fraction", _q({"p": 11, "q": 19}, "11/19", text="Solve 19x = 11."), reply).correct


@pytest.mark.parametrize("reply", ["(−3/5)", "(-3/5)", "−3/5", "-\\frac{3}{5}"])
def test_negative_fraction_notations_are_read(reply):
    assert grade("fraction", _q({"p": -3, "q": 5}, "-3/5"), reply).correct


@pytest.mark.parametrize("reply", ["(11/18)", "\\frac{19}{11}", "11⁄18", "(11/19"])
def test_wrong_fractions_in_any_notation_are_wrong(reply):
    assert not grade("fraction", _q({"p": 11, "q": 19}, "11/19"), reply).correct


# -- G4: derivative labels and the division sign ---------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "f'(x) = 6x + 2",
        "f′(x) = 6x + 2",
        "y' = 6x + 2",
        "dy/dx = 6x + 2",
        "dy / dx = 6x + 2",
        "d/dx = 6x + 2",
        "Answer: 6x + 2",
        "answer = 6x + 2",
    ],
)
def test_derivative_labels_are_dropped(reply):
    question = _q("6*x + 2", "6*x + 2", text="Differentiate f(x) = 3x^2 + 2x.")
    assert grade("expression", question, reply).correct


@pytest.mark.parametrize("reply", ["f'(x) = 6x", "dy/dx = 3x^2 + 2x", "y'' = 6x + 3"])
def test_a_label_does_not_make_a_wrong_derivative_right(reply):
    question = _q("6*x + 2", "6*x + 2", text="Differentiate f(x) = 3x^2 + 2x.")
    assert not grade("expression", question, reply).correct


def test_second_derivative_label():
    question = _q("6", "6", text="Find the second derivative of y = 3x^2.")
    assert grade("expression", question, "y'' = 6").correct
    assert grade("expression", question, "d²y/dx² = 6").correct
    assert grade("expression", question, "d^2y/dx^2 = 6").correct


def test_division_sign_divides():
    question = _q("x/2", "x/2", text="Simplify 3x/6.")
    assert grade("expression", question, "x ÷ 2").correct
    assert not grade("expression", question, "2 ÷ x").correct


# -- G5: mixed numbers -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, reply",
    [(2.5, "2 1/2"), (2.5, "2½"), (-2.5, "-2 1/2"), (-2.5, "−2 ½"), (2.75, "2 3/4 m"), (9.5, "x = 9 1/2")],
)
def test_mixed_numbers_are_one_number(answer, reply):
    assert grade("number", _q(answer, str(answer)), reply).correct


@pytest.mark.parametrize("answer, reply", [(2.5, "2 5/3"), (2, "2 1/2"), (1.5, "2 1/2"), (9.5, "9 1/2 7")])
def test_what_is_not_this_mixed_number_is_wrong(answer, reply):
    assert not grade("number", _q(answer, str(answer)), reply).correct


def test_mixed_number_in_the_fraction_kind_with_a_vulgar_fraction():
    assert grade("fraction", _q({"p": 5, "q": 2}, "5/2"), "2½").correct


# -- G6: a pair's labels must be its own -----------------------------------------------------


@pytest.mark.parametrize(
    "reply, correct",
    [
        ("x = 3, y = -6", True),
        ("y = -6, x = 3", True),
        ("X = 3, Y = -6", True),
        ("(3, -6)", True),
        ("a = 3, b = -6", False),
        ("x = 3, b = -6", False),
        ("y = 3, x = -6", False),
    ],
)
def test_pair_labels_are_enforced(reply, correct):
    assert grade("pair", _q({"x": 3, "y": -6}, "x = 3, y = -6"), reply).correct is correct


def test_dropped_subscript_and_spelled_out_labels_still_count():
    beam = _q({"R_A": 7.5, "R_B": 22.5}, "R_A = 7.5 kN, R_B = 22.5 kN")
    assert grade("pair", beam, "R = 7.5 kN, R = 22.5 kN").correct
    assert grade("pair", beam, "RA = 7.5, RB = 22.5").correct
    size = _q({"h": 4, "w": 9}, "h = 4, w = 9")
    assert grade("pair", size, "height = 4, width = 9").correct


# -- G8: lowest terms when asked ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Write 6/8 in lowest terms.",
        "Write 6/8 in simplest form.",
        "Simplify 6/8.",
        "Give your answer fully simplified.",
    ],
)
def test_unreduced_fraction_is_wrong_when_lowest_terms_are_asked(text):
    question = _q({"p": 3, "q": 4}, "3/4", text=text)
    assert grade("fraction", question, "3/4").correct
    assert not grade("fraction", question, "6/8").correct
    assert not grade("fraction", question, "x = 6/8").correct


def test_unreduced_fraction_is_right_when_lowest_terms_are_not_asked():
    assert grade("fraction", _q({"p": 3, "q": 4}, "3/4", text="Find p."), "6/8").correct


def test_lowest_terms_applies_to_a_mixed_number_fraction_part():
    question = _q({"p": 5, "q": 2}, "5/2", text="Simplify 10/4, as a mixed number.")
    assert grade("fraction", question, "2 1/2").correct
    assert not grade("fraction", question, "2 2/4").correct


# -- vectors typed without spaces, and unit powers -------------------------------------------------


def _ordered(answer, text="Find the cross product."):
    return _q(answer, "(" + ", ".join(map(str, answer)) + ")", text)


@pytest.mark.parametrize("reply", ["(-2,105,287)", "-2,105,287", "(-2, 105, 287)", "<-2,105,287>"])
def test_vector_components_typed_without_spaces_are_accepted(reply):
    assert grade("ordered_list", _ordered([-2, 105, 287]), reply).correct


@pytest.mark.parametrize("reply", ["(105,-2,287)", "(-2,105,288)", "(-2105287)", "(-2,105)"])
def test_vector_components_still_compared_in_order(reply):
    assert not grade("ordered_list", _ordered([-2, 105, 287]), reply).correct


def test_one_grouped_value_is_still_one_item():
    assert grade("ordered_list", _ordered([12345, 7]), "12,345, 7").correct


def test_unit_power_is_not_read_as_the_value():
    question = SimpleNamespace(text="Convert 5 km to m.", answer_text="5000", answer=5000)
    assert unit_is_acceptable(question, "5000 m")
    assert not unit_is_acceptable(question, "5000 m2")
    assert not unit_is_acceptable(question, "5000 m^2")


@pytest.mark.parametrize("text", ["What is the sum in n = 5 terms?", "What is the answer in g(x) form?"])
def test_variable_after_in_is_not_an_asked_unit(text):
    assert unit_is_acceptable(SimpleNamespace(text=text, answer_text="7", answer=7), "7 m")
