"""Exam tier of `senior-define-the-derivative-f-left-x-right-as`: reciprocal, reciprocal-square and root functions."""

import math

import exd_x_c3_support as sup

SKILL = "senior-define-the-derivative-f-left-x-right-as"
TOPIC = "exd_x_c3_first_principles"


def _function(text: str):
    """A python callable for the plain-text expression (the prompt's `·` and `^`, or the key's `*` and `^`)."""
    code = compile(text.replace("·", "*").replace("^", "**"), "<expr>", "eval")
    return lambda x: eval(code, {"sqrt": math.sqrt}, {"x": x})  # noqa: S307 - test-only, our own generated text


def test_registered():
    # The skill's exam source is now the multi-part chain; this single-answer preset stays registered, unused.
    last = sup.skill(SKILL)["sources"][-1]
    assert last["topic"] == "exd_x_r2a_first_principles_tangent"
    assert last["difficulty"] == "exam" and last["stages"] == ["exam"]


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_a_numerical_difference_quotient_of_the_rendered_function():
    """Symmetric difference quotient at h = 1e-5 on the function read back from the prompt."""
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        body = q.text.split("f(x) = ", 1)[1].split(" from first principles")[0]
        f = _function(body)
        key = _function(q.answer_text)
        for p in (1.3, 2.7):
            h = 1e-5
            numeric = (f(p + h) - f(p - h)) / (2 * h)
            assert abs(numeric - key(p)) < 1e-6 * (1 + abs(numeric)), (seed, q.text, q.answer_text)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
