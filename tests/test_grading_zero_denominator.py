"""A fraction written over zero is not a number: the reply is wrong, quietly.

`Fraction("1/0")` raises `ZeroDivisionError`, and the digit-string reader in `mathgen.grading.spoken` called it on
every token that looked like a number. `grade()` caught the error as a backstop, marked the reply wrong and
logged two tracebacks, which Python printed to stderr because the library had no log handler. `1/0`, `0/0`,
`5/0`, `1/0.0` and `1 1/0` are now read as "no value" (a reply with one in it has nothing to compare), and
nothing is logged.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade, typed
from mathgen.grading.spoken import check_answer, extract_values, spoken_to_expr

ZERO_DENOMINATORS = ["1/0", "0/0", "5/0", "1/0.0", "1 1/0", "-3/0", "1/00", "x = 1/0", "1/0, 2", "2, 1/0", "(1/0, 2)"]


def _question(answer, answer_text, **fields) -> Question:
    return Question(
        topic="test",
        level=0,
        seed=1,
        text="Work it out.",
        latex="Work it out.",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[],
        difficulty="easy",
        distractors=[],
        **fields,
    )


#: (kind, question) for every typed-answer kind that reads numbers from the reply.
_QUESTIONS = {
    "number, int": ("number", _question(1, "1")),
    "number, zero": ("number", _question(0, "0")),
    "number, decimal": ("number", _question(2.5, "2.5")),
    "number, p/q": ("number", _question({"p": 1, "q": 2}, "1/2")),
    "number, text answer": ("number", _question("10", "10")),
    "fraction": ("fraction", _question({"p": 1, "q": 2}, "1/2")),
    "pair, labelled": ("pair", _question({"x": 3, "y": -6}, "x = 3, y = -6")),
    "pair, unlabelled": ("pair", _question({"x": 1, "y": 2}, "(1, 2)")),
    "list": ("list", _question([1, 2], "1, 2")),
    "ordered_list": ("ordered_list", _question([1, 2, 3], "1, 2, 3")),
    "set": ("set", _question([1, 2], "{1, 2}")),
    "points": ("points", _question([[1, 2], [3, 4]], "(1, 2) (3, 4)")),
    "expression": ("expression", _question("x + 1", "x + 1")),
}


@pytest.mark.parametrize("reply", ZERO_DENOMINATORS)
@pytest.mark.parametrize("case", sorted(_QUESTIONS))
def test_a_fraction_over_zero_is_wrong_and_logs_nothing(case, reply, caplog):
    kind, question = _QUESTIONS[case]
    with caplog.at_level(logging.DEBUG, logger="mathgen"):
        result = grade(kind, question, reply)
    assert result.correct is False, (case, reply)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (case, reply)


def test_the_reader_finds_no_value_in_a_reply_with_a_fraction_over_zero():
    for reply in ZERO_DENOMINATORS:
        assert extract_values(reply) == [], reply
    assert extract_values("the answer is 2 or maybe 1/0") == []


def test_other_fractions_are_read_as_before():
    assert extract_values("3/5") == [Fraction(3, 5)]
    assert extract_values("0/5") == [Fraction(0)]
    assert extract_values("1/2 and 3") == [Fraction(1, 2), Fraction(3)]
    assert extract_values("-3/4") == [Fraction(-3, 4)]
    assert extract_values("three over five") == [Fraction(3, 5)]
    assert extract_values("three over zero") == [Fraction(3), Fraction(0)]  # the spoken form never divided by it


def test_the_spoken_expression_reader_and_the_transcript_path_do_not_raise_either(caplog):
    with caplog.at_level(logging.DEBUG, logger="mathgen"):
        for reply in ZERO_DENOMINATORS:
            spoken_to_expr(reply)
            for _, question in _QUESTIONS.values():
                check_answer(question, reply)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_zero_denominator_in_the_exact_reader_was_already_refused():
    assert typed._exact_fraction("1/0") is None
    assert typed._parse_typed_fraction("1/0") is None
    assert typed._parse_numeric_item("1/0") is None


# -- a number skill whose answer is text or a fraction, asked through a question that is not a `Question` ---------


def _number_skills_with_a_non_scalar_answer():
    skills = []
    for skill in engine.load_skills()[1]:
        if skill.status != "active" or skill.answer != "number":
            continue
        for index in range(len(skill.sources)):
            answer = engine.make_for_skill(skill, seed=1, source=index).answer
            if not isinstance(answer, (int, float)):
                skills.append((skill, index))
                break
    return skills


_NON_SCALAR = _number_skills_with_a_non_scalar_answer()


def test_the_library_has_number_skills_whose_answer_is_not_a_number():
    assert len(_NON_SCALAR) >= 8  # text such as "10" and the fraction {"p", "q"}: the shapes this file is about


@pytest.mark.parametrize("skill, index", _NON_SCALAR, ids=[skill.id for skill, _ in _NON_SCALAR])
def test_such_a_skill_grades_an_ordinary_reply_without_raising(skill, index, caplog):
    question = engine.make_for_skill(skill, seed=1, source=index)
    # a question that arrives as plain attributes (rebuilt from stored JSON, say) rather than as a `Question`
    duck = SimpleNamespace(**{name: getattr(question, name) for name in question.__dataclass_fields__})
    with caplog.at_level(logging.DEBUG, logger="mathgen"):
        for shape in (question, duck):
            assert engine.grade(skill, shape, question.answer_text).correct is True, skill.id
            assert engine.grade(skill, shape, "3").correct in (True, False)
            assert engine.grade(skill, shape, "about 2.5 units").correct in (True, False)
            assert engine.grade(skill, shape, "1/0").correct is False
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], skill.id
