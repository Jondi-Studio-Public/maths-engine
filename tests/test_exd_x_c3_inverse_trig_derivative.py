"""Exam tier of `senior-find-and-use-the-derivative-of-the-inverse`: inverse trig functions in products and chains."""

import math

import exd_x_c3_support as sup

SKILL = "senior-find-and-use-the-derivative-of-the-inverse"
TOPIC = "exd_x_c3_inverse_trig_derivative"
_NAMES = {
    "arctan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "exp": math.exp,
    "sqrt": math.sqrt,
}


def _function(text: str):
    code = compile(text.replace("·", "*").replace("^", "**"), "<expr>", "eval")
    return lambda x: eval(code, {**_NAMES}, {"x": x})  # noqa: S307 - test-only, our own generated text


def test_registered():
    # The skill's exam source now points at the multi-part tangent chain; this module's generator stays registered.
    sup.assert_registered_as_exam_source(SKILL, "exd_x_r3e_inverse_trig_tangent")


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_a_numerical_derivative_of_the_rendered_function():
    """Symmetric difference quotient (h = 1e-6) of the function read back from the prompt, at three points."""
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        body = q.text.removeprefix("Differentiate ").removesuffix(" with respect to x.")
        f, key = _function(body), _function(q.answer_text)
        for p in (0.11, 0.17, 0.23):
            h = 1e-6
            numeric = (f(p + h) - f(p - h)) / (2 * h)
            assert abs(numeric - key(p)) < 1e-6 * (1 + abs(numeric)), (seed, q.text, q.answer_text, p)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
