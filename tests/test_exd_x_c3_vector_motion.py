"""Exam tier of `senior-differentiate-and-integrate-a-vector-function-with-respect`: acceleration to position."""

import re
from fractions import Fraction

import exd_x_c3_support as sup

SKILL = "senior-differentiate-and-integrate-a-vector-function-with-respect"
TOPIC = "exd_x_c3_vector_motion"
_PROMPT = re.compile(
    r"A particle moves in the plane with "
    r"acceleration a\(t\) = \((.+?), (.+?)\) in metres per second squared, t in seconds\. "
    r"At t = 0 its velocity is \((-?\d+), (-?\d+)\) metres per second and its position is \((-?\d+), (-?\d+)\) metres\. "
    r"Find the particle's position vector at t = (\d+) seconds\."
)


def _coefficients(text: str) -> tuple[int, int]:
    """(A, B) for the text 'At + B' (A and B nonzero, as the generator guarantees)."""
    m = re.fullmatch(r"(-?\d*)t ([+-]) (\d+)", text)
    assert m, text
    a = {"": 1, "-": -1}.get(m.group(1)) or int(m.group(1))
    return a, int(m.group(3)) * (1 if m.group(2) == "+" else -1)


def test_registered():
    # The skill's default exam source is the multi-part chain; this single-answer preset is its `exam-single` source.
    sources = sup.skill(SKILL)["sources"]
    assert sources[-2]["topic"] == "exd_x_r2a_vector_motion_chain" and "key" not in sources[-2]
    last = sources[-1]
    assert last["topic"] == TOPIC and last["key"] == "exam-single"
    assert last["difficulty"] == "exam" and last["stages"] == ["exam"]


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_double_integration_of_the_rendered_prompt():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        m = _PROMPT.fullmatch(q.text)
        assert m, q.text
        acc = [_coefficients(m.group(1)), _coefficients(m.group(2))]
        u = [int(m.group(3)), int(m.group(4))]
        p = [int(m.group(5)), int(m.group(6))]
        T = int(m.group(7))
        want = []
        for i in range(2):
            a, b = acc[i]
            want.append(Fraction(p[i]) + u[i] * T + Fraction(b * T**2, 2) + Fraction(a * T**3, 6))
        assert [float(w) for w in want] == [q.answer["x"], q.answer["y"]], (seed, q.text, want, q.answer)
        assert q.answer_text == f"x = {want[0]}, y = {want[1]}"  # Fraction prints a whole number without a denominator


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
