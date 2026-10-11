"""The typed answer contract (mathgen/answers.py) and misconception diagnosis (mathgen/misconceptions.py).

A wrong reply that is the error a tagged distractor stands for is diagnosed as that error, however the learner spells
it; a right reply never is; and a generator that tags nothing produces exactly the question it always did.
"""

from __future__ import annotations

import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen import engine, make
from mathgen.answers import (
    ANSWER_FORMS,
    ANSWER_KINDS,
    AnswerSpec,
    Distractor,
    answer_spec,
    exact_value,
    flatten_distractors,
    kind_accepts,
    same_value,
    tagged_distractors,
)
from mathgen.core import Question, mcq_options, pick_distractors
from mathgen.grading import GRADERS, grade
from mathgen.misconceptions import CATALOGUE, describe
from sympy import Integer, Rational

# -- the contract ------------------------------------------------------------------------------------------------


def test_answer_kinds_are_the_grader_kinds():
    assert ANSWER_KINDS == set(GRADERS)


def test_answer_form_vocabulary_is_closed():
    with pytest.raises(ValueError, match="answer_form"):
        flatten_and_make({"answer_form": "tidy"})
    assert "factored" in ANSWER_FORMS


def test_unknown_answer_kind_is_refused():
    with pytest.raises(ValueError, match="answer_kind"):
        flatten_and_make({"answer_kind": "essay"})


def flatten_and_make(extra: dict):
    """Run `make()`'s checks over a hand-made generator result."""
    from mathgen import core

    topic = "_contract_probe"
    core.REGISTRY[topic] = (1, lambda rng: dict(text="q", latex="", answer=1, answer_text="1", **extra))
    try:
        return make(topic, seed=1)
    finally:
        del core.REGISTRY[topic]


def test_untagged_generator_output_is_unchanged():
    question = make("addition", seed=3)
    for key in ("answer_kind", "distractor_values", "distractor_credit", "option_notes"):
        assert key not in question.to_dict()


def test_answer_spec_reads_a_question_and_anything_shaped_like_one():
    question = make("limits", seed=1, kinds=("infinity",))
    spec = answer_spec(question)
    assert spec == AnswerSpec(
        kind="fraction", value=question.answer, text=question.answer_text, latex=question.answer_latex
    )
    part = SimpleNamespace(answer=[1, 2], answer_text="x = 1 or x = 2")
    assert answer_spec(part, kind="list").kind == "list"
    assert answer_spec(part).form is None


@pytest.mark.parametrize(
    "skill_kind, value_kind, ok",
    [("number", "number", True), ("mcq", "list", True), ("expression", "number", True), ("number", "list", False)],
)
def test_kind_accepts(skill_kind, value_kind, ok):
    assert kind_accepts(skill_kind, value_kind) is ok


# -- Distractor ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("credit", [1, Fraction(3, 2), -0.5])
def test_credit_is_a_share_below_full_marks(credit):
    with pytest.raises(ValueError, match="credit"):
        Distractor("x = 3", "sign-flip-one-root", credit=credit)


def test_distractor_needs_text():
    with pytest.raises(ValueError):
        Distractor("  ")


def test_flatten_keeps_string_distractors_as_they_are():
    result = {"distractors": ["1", "2"], "option_notes": {"1": "a"}}
    flatten_distractors(result)
    assert result == {"distractors": ["1", "2"], "option_notes": {"1": "a"}}


def test_flatten_stores_tags_values_and_credit_by_text():
    result = {
        "distractors": [
            Distractor("x = 3", "sign-flip-one-root", [3], credit=Fraction(1, 2)),
            "x = 9",
            Distractor("x = 4", None, [4]),
        ]
    }
    flatten_distractors(result)
    assert result == {
        "distractors": ["x = 3", "x = 9", "x = 4"],
        "option_notes": {"x = 3": "sign-flip-one-root"},
        "distractor_values": {"x = 3": [3], "x = 4": [4]},
        "distractor_credit": {"x = 3": "1/2"},
    }
    question = SimpleNamespace(**result)
    assert tagged_distractors(question)[0] == Distractor("x = 3", "sign-flip-one-root", [3], Fraction(1, 2))
    assert tagged_distractors(question)[1] == Distractor("x = 9")


def test_flatten_refuses_an_id_outside_the_catalogue():
    with pytest.raises(ValueError, match="catalogue"):
        flatten_distractors({"distractors": [Distractor("2", "not-a-real-error")]})


def test_flatten_refuses_one_option_with_two_tags():
    with pytest.raises(ValueError, match="tagged both"):
        flatten_distractors(
            {"distractors": [Distractor("2", "answered-zero")], "option_notes": {"2": "inverted-ratio"}}
        )


def test_pick_distractors_keeps_tags_and_drops_collisions():
    picked = pick_distractors(
        "x = 2",
        [Distractor("x = 2.0", "answered-zero"), Distractor("x = 3", "answered-zero"), "x = 4", Distractor("x = 3")],
    )
    assert picked == [Distractor("x = 3", "answered-zero"), "x = 4"]


def test_exact_value():
    assert exact_value(Fraction(6, 3)) == 2
    assert exact_value(Fraction(-7, 2)) == {"p": -7, "q": 2}
    assert exact_value(5) == 5


def test_exact_value_reads_a_sympy_rational():
    assert exact_value(Rational(-14, 4)) == {"p": -7, "q": 2}
    assert exact_value(Integer(9)) == 9
    assert exact_value(Rational(8, 4)) == 2


@pytest.mark.parametrize("number", [0.1, 2.0, True, "3/4", None])
def test_exact_value_refuses_anything_already_rounded_or_not_a_number(number):
    """0.1 as a float is 3602879701896397/36028797018963968, not the tenth the generator meant."""
    with pytest.raises(TypeError):
        exact_value(number)


@pytest.mark.parametrize(
    "left, right, kind",
    [
        (3, 3.0, None),
        (3, "3", None),
        (3, {"p": 3, "q": 1}, None),
        (Rational(1, 2), 0.5, None),
        (Fraction(7, 2), {"p": 14, "q": 4}, None),
        ([2, -1], [-1, 2], "list"),
        ([2, -1], [-1, 2], "set"),
    ],
)
def test_same_value(left, right, kind):
    assert same_value(left, right, kind)


@pytest.mark.parametrize(
    "left, right, kind",
    [
        (16.7, Fraction(50, 3), None),  # a rounded value is not the exact one
        (0.1, Fraction(1, 3), None),
        ([2, -1], [-1, 2], "ordered_list"),
        ([2, -1], [-1, 2], None),
        ("undefined", 0, None),
    ],
)
def test_not_the_same_value(left, right, kind):
    assert not same_value(left, right, kind)


# -- make() enforces the contract, whatever the generator skipped ----------------------------------------------------


def _flatten(answer, answer_text, distractors, **extra):
    result = {"answer": answer, "answer_text": answer_text, "distractors": distractors, **extra}
    flatten_distractors(result)
    return result


def test_a_distractor_that_is_the_answer_text_is_refused():
    with pytest.raises(ValueError, match="answer's own text"):
        _flatten(5, "x = 5", [Distractor(" x = 5", "answered-zero", 0)])


def test_a_string_distractor_that_is_the_answer_text_is_refused_beside_a_tagged_one():
    with pytest.raises(ValueError, match="answer's own text"):
        _flatten(5, "x = 5", ["x = 5", Distractor("x = 0", "answered-zero", 0)])


def test_a_repeated_distractor_is_refused():
    """The second would silently overwrite the first's value."""
    with pytest.raises(ValueError, match="offered twice"):
        _flatten(5, "5", [Distractor("4", "answered-zero", 4), Distractor("4", "inverted-ratio", 1)])


@pytest.mark.parametrize(
    "answer, value, kind",
    [(5, 5.0, None), (5, {"p": 10, "q": 2}, None), ({"p": 1, "q": 2}, 0.5, "fraction"), ([1, 2], [2, 1], "list")],
)
def test_a_distractor_with_the_answer_value_is_refused(answer, value, kind):
    with pytest.raises(ValueError, match="answer's own value"):
        _flatten(answer, "a", [Distractor("b", "answered-zero", value)], answer_kind=kind)


def test_an_ordered_answer_may_offer_its_reversal():
    result = _flatten([1, 2], "1, 2", [Distractor("2, 1", "sign-flip-one-root", [2, 1])], answer_kind="ordered_list")
    assert result["distractors"] == ["2, 1"]


def test_a_rounded_distractor_value_is_not_the_exact_answer():
    """16.7 is a different answer from 50/3 to a grader that reads its stated precision."""
    result = _flatten({"p": 50, "q": 3}, "50/3", [Distractor("16.7", "divides-by-final", 16.7)])
    assert result["distractor_values"] == {"16.7": 16.7}


def test_an_empty_distractor_list_is_left_alone():
    assert _flatten(5, "5", []) == {"answer": 5, "answer_text": "5", "distractors": []}
    question = flatten_and_make({"distractors": []})
    assert question.distractors == [] and tagged_distractors(question) == []


# -- the catalogue -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(CATALOGUE.values()), ids=lambda entry: entry.id)
def test_catalogue_entries_are_well_formed(entry):
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry.id)
    assert entry.description.strip() and entry.description.rstrip().endswith(".")
    assert entry.kind in ("misconception", "near_miss")


def test_skill_wording_wins_over_the_catalogue():
    skill = SimpleNamespace(misconceptions=[SimpleNamespace(id="inverted-ratio", description="Own words.")])
    assert describe("inverted-ratio", skill) == ("Own words.", "misconception")
    assert describe("inverted-ratio") == (CATALOGUE["inverted-ratio"].description, "misconception")
    assert describe("gave-change-not-percent")[1] == "near_miss"
    assert describe("nobody-knows-this") is None


# -- diagnosis -----------------------------------------------------------------------------------------------------


def _q(answer, answer_text, distractors, **extra):
    result = {"distractors": distractors}
    flatten_distractors(result)
    return Question(
        topic="t", level=1, seed=1, text="q", latex="", answer=answer, answer_text=answer_text, **result, **extra
    )


_LINEAR = _q(
    12,
    "x = 12",
    [
        Distractor("x = 50/7", "sign-error-moving-term", {"p": 50, "q": 7}),
        Distractor("x = 84", "forgot-to-divide-by-coefficient", 84, credit=Fraction(1, 2)),
    ],
)


@pytest.mark.parametrize("given", ["x = 84", "x=84", "84", "  84 ", "x = 84.0"])
def test_a_respelt_wrong_answer_is_diagnosed(given):
    result = grade("number", _LINEAR, given)
    assert not result.correct
    assert result.misconception == "forgot-to-divide-by-coefficient"
    assert result.matched_distractor == "x = 84"
    assert result.note == CATALOGUE["forgot-to-divide-by-coefficient"].description
    assert result.note_kind == "misconception"
    assert _LINEAR.distractor_credit[result.matched_distractor] == "1/2"


@pytest.mark.parametrize("given", ["50/7", "50 / 7", "100/14", "x = 50/7"])
def test_a_fraction_distractor_is_matched_by_value(given):
    assert grade("number", _LINEAR, given).misconception == "sign-error-moving-term"


@pytest.mark.parametrize("given", ["12", "x = 12"])
def test_a_right_answer_is_never_diagnosed(given):
    result = grade("number", _LINEAR, given)
    assert result.correct and result.misconception is None and result.note is None


@pytest.mark.parametrize("given", ["13", "", "banana", "7.14"])
def test_an_unexplained_wrong_answer_gets_no_diagnosis(given):
    result = grade("number", _LINEAR, given)
    assert not result.correct
    assert (result.misconception, result.matched_distractor, result.note) == (None, None, None)


def test_a_list_distractor_is_matched_in_any_order():
    question = _q([-7, 6], "x = -7 or x = 6", [Distractor("x = -6 or x = 7", "sign-flip-both-roots", [-6, 7])])
    assert grade("list", question, "7, -6").misconception == "sign-flip-both-roots"


def test_a_value_less_tag_matches_its_words_in_any_case():
    question = _q(3, "3", [Distractor("does not exist", "no-limit-at-hole")])
    assert grade("expression", question, "Does Not  Exist").misconception == "no-limit-at-hole"
    assert grade("expression", question, "exists").misconception is None


def test_the_stand_in_drops_the_answer_form():
    """A learner who expanded where the question wanted factors still made the tagged error."""
    question = _q(
        "(x - 2)*(x + 3)",
        "(x - 2)(x + 3)",
        [Distractor("(x + 2)(x - 3)", "sign-flip-both-roots", "(x + 2)*(x - 3)")],
        answer_form="factored",
    )
    assert grade("expression", question, "x^2 - x - 6").misconception == "sign-flip-both-roots"


def test_an_mcq_choice_is_diagnosed():
    question = _q(5, "5", [Distractor("0", "answered-zero", 0)])
    options = mcq_options(question)
    result = grade("mcq", question, str(options.index("0")))
    assert result.misconception == "answered-zero"
    assert grade("mcq", question, str(options.index("5"))).misconception is None


def test_a_part_shaped_object_is_diagnosed():
    """Anything with the answer attributes is graded and diagnosed the same way: a multi-part question's part."""
    part = SimpleNamespace(
        text="q",
        answer=84,
        answer_text="84",
        answer_alt=None,
        answer_form=None,
        distractors=["12"],
        option_notes={"12": "forgot-to-divide-by-coefficient"},
        distractor_values={"12": 12},
        distractor_credit={},
    )
    assert grade("number", part, "12.0").misconception == "forgot-to-divide-by-coefficient"


# -- the pilot generators ------------------------------------------------------------------------------------------

_PILOT_TOPICS = ("limits", "linear_equation", "quadratic", "function_evaluation", "percentage_change")
_SKILLS = [skill for skill in engine.load_skills()[1] if skill.status == "active"]
_PILOT_SOURCES = [
    (skill, index) for skill in _SKILLS for index, source in enumerate(skill.sources) if source.topic in _PILOT_TOPICS
]


def _plain(value) -> str:
    """A tagged value spelled the plainest way a learner would type it."""
    if isinstance(value, dict):
        return f"{value['p']}/{value['q']}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in reversed(value))
    return str(value)


@pytest.mark.parametrize("skill, index", _PILOT_SOURCES, ids=lambda v: v.id if hasattr(v, "id") else str(v))
def test_pilot_skills_diagnose_their_own_distractors(skill, index):
    tagged_seen = 0
    for seed in range(1, 31):
        question = engine.make_for_skill(skill, seed=seed, source=index)
        if question.answer_kind is not None:
            assert kind_accepts(skill.answer, question.answer_kind), (skill.id, question.answer_kind)
        right = engine.grade(skill, question, question.answer_text)
        assert right.correct and right.misconception is None
        for distractor in tagged_distractors(question):
            if distractor.misconception is None:
                continue
            tagged_seen += 1
            replies = [distractor.text] + ([_plain(distractor.value)] if distractor.value is not None else [])
            for reply in replies:
                result = engine.grade(skill, question, reply)
                assert not result.correct, (skill.id, seed, reply)
                # The first option the reply is an answer to: an earlier option can share the value.
                assert result.matched_distractor is not None, (skill.id, seed, reply)
                assert result.misconception == question.option_notes[result.matched_distractor]
                assert result.note, (skill.id, seed, reply)
    # Every pilot source in an active skill serves tagged questions, except the paths the pilot leaves alone.
    untagged_paths = {"brackets", "brackets_leading", "double_brackets", "division_form", "division_two_step"}
    untagged_paths |= {"both_sides", "both_sides_bracket", "vertex", "formula", "formula_rearrange", "factor_nonmonic"}
    if skill.sources[index].params.get("kind") not in untagged_paths:
        assert tagged_seen, skill.id


def test_skill_wording_is_used_for_a_skill_that_has_its_own():
    skill = next(s for s in _SKILLS if s.id == "limit-at-infinity")
    question = engine.make_for_skill(skill, seed=4)
    inverted = next(d for d in tagged_distractors(question) if d.misconception == "inverted-ratio")
    result = engine.grade(skill, question, inverted.text)
    assert result.note == next(m.description for m in skill.misconceptions if m.id == "inverted-ratio")


def test_percentage_change_is_set_in_australia():
    texts = {make("percentage_change", seed=seed, reverse=reverse).text for seed in range(60) for reverse in (0, 1)}
    assert any("GST" in text for text in texts)
    assert any("Brisbane" in text for text in texts)
    assert not any(text.startswith("A value") for text in texts)
