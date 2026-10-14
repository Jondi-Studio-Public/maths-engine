"""Exam tier of `senior-use-the-increments-formula-delta-y-cong-frac`: percentage change and a composite function."""

import re
from fractions import Fraction

import exd_x_c3_support as sup

SKILL = "senior-use-the-increments-formula-delta-y-cong-frac"
TOPIC = "exd_x_c3_increments"
_PERCENT = re.compile(
    r"A quantity is given by y = f\(x\) = (.+)\. Starting from x0 = (\d+), the value of x increases by (\d+)%\. .*"
)
_COMPOSITE = re.compile(
    r"y = f\(x\) = (\d*)\(x\^2 \+ (\d+)\)\^(\d)\. Estimate the new value of y when x changes from x0 = (-?\d+) "
    r"to x0 \+ δx, .* given δx = (-?\d+(?:/\d+)?)\."
)


def _quadratic(text: str):
    """(f, f') as Fraction functions for 'ax^2 + bx + c' text, read term by term."""
    coeffs = {2: 0, 1: 0, 0: 0}
    for sign, num, var, power in re.findall(r"([+-]?)\s*(\d*)(x?)(\^2)?", text.replace(" ", "")):
        if not (num or var):
            continue
        degree = 2 if power else (1 if var else 0)
        coeffs[degree] += (-1 if sign == "-" else 1) * (int(num) if num else 1)
    return (
        lambda v: coeffs[2] * v * v + coeffs[1] * v + coeffs[0],
        lambda v: 2 * coeffs[2] * v + coeffs[1],
    )


def _independent_key(text: str) -> Fraction:
    if m := _PERCENT.fullmatch(text):
        f, df = _quadratic(m.group(1))
        x0, pct = int(m.group(2)), Fraction(m.group(3))
        return 100 * df(x0) * (x0 * pct / 100) / f(x0)
    m = _COMPOSITE.fullmatch(text)
    assert m, text
    k, c, n, x0, dx = int(m.group(1) or 1), int(m.group(2)), int(m.group(3)), int(m.group(4)), Fraction(m.group(5))
    u = x0 * x0 + c
    return k * u**n + n * k * u ** (n - 1) * 2 * x0 * dx


def test_registered():
    # The default exam source is the multi-part chain; this single-answer preset is its `exam-single` source.
    sources = sup.skill(SKILL)["sources"]
    assert sources[-2]["topic"] == "exd_x_r2b_increments_chain" and "key" not in sources[-2]
    sup.assert_registered_as_exam_source(SKILL, TOPIC)
    assert sources[-1]["key"] == "exam-single"


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_an_independent_evaluation_of_the_rendered_prompt():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        want = _independent_key(q.text)
        assert Fraction(q.answer_text) == want, (seed, q.text, q.answer_text, want)
        assert q.answer == {"p": want.numerator, "q": want.denominator}


def test_both_kinds_occur():
    texts = [sup.exam_question(TOPIC, s).text for s in range(40)]
    assert any("percentage" in t for t in texts) and any("new value" in t for t in texts)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
