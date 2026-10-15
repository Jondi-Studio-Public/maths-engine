"""Exam tier of `senior-find-and-use-the-inverse-trigonometric-functions-arcsine`: compound angles."""

import math
import re
from fractions import Fraction

import exd_x_c3_support as sup

SKILL = "senior-find-and-use-the-inverse-trigonometric-functions-arcsine"
TOPIC = "exd_x_c3_inverse_trig_use"
_PROMPT = re.compile(r"Evaluate (sin|tan)\((.+)\) exactly\.")
_NAMES = {"arcsin": math.asin, "arccos": math.acos, "arctan": math.atan, "sin": math.sin, "tan": math.tan}


def _evaluate(text: str) -> float:
    """Float value of the prompt's expression, read back from its text alone."""
    m = _PROMPT.fullmatch(text)
    assert m, text
    inner = re.sub(r"(-?\d+)/(\d+)", r"(\1/\2)", m.group(2)).replace("2·", "2*")
    return eval(f"{m.group(1)}({inner})", {"__builtins__": {}}, _NAMES)  # noqa: S307 - test-only


def test_registered():
    sup.assert_registered_as_exam_source(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_key_matches_floating_point_evaluation_of_the_rendered_prompt():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        want = _evaluate(q.text)
        key = Fraction(q.answer_text)
        assert abs(float(key) - want) < 1e-9, (seed, q.text, q.answer_text, want)
        assert key.denominator < 10**4
        for d in q.distractors:
            assert abs(float(Fraction(d)) - want) > 1e-6, (seed, q.text, d)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
