"""Exam tier of the multiplication-as-a-linear-transformation skill: infer the multiplier, then apply or report it.

The oracle reads the complex numbers out of the rendered prompt, divides with exact fractions, and compares
with the pair in `answer_text`. It shares no code with the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_complex_mult_transform"

# A complex number as written in the prompts: "-4 - 3i", "10i", "-i", "7".
_COMPLEX = re.compile(r"(?<![A-Za-z\d])(?:-?\d+ [+-] \d*i|-?\d*i|-?\d+)(?![A-Za-z\d])")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard = {
        prompt_shape(make("complex_mult_linear_transform", seed=s, difficulty="hard", variant="general"))
        for s in range(60)
    }
    assert len(shapes - hard) >= 3, (shapes, hard)


def _parse(token: str) -> tuple:
    t = token.replace(" ", "")
    if t.endswith("i"):
        t = t[:-1] + "j"
    if t in ("j", "+j"):
        t = "1j"
    if t.endswith("-j") or t.endswith("+j"):
        t = t[:-1] + "1j"
    if t == "-j":
        t = "-1j"
    value = complex(t)
    return int(value.real), int(value.imag)


def _mul(w, z):
    return (w[0] * z[0] - w[1] * z[1], w[0] * z[1] + w[1] * z[0])


def _div(num, den):
    norm = den[0] ** 2 + den[1] ** 2
    return (Fraction(num[0] * den[0] + num[1] * den[1], norm), Fraction(num[1] * den[0] - num[0] * den[1], norm))


def _oracle(text: str) -> str:
    # The "(x, y)" and "w = x + yi" wording holds no numbers; strip the labels w1 / w2 before reading values.
    cleaned = re.sub(r"\bw[12]\b", "w", text)
    nums = [_parse(t) for t in _COMPLEX.findall(cleaned)]
    if "successive" in text or "w1" in text:
        z, w1, w2 = nums
        ans = _mul(w2, _mul(w1, z))
    elif "Find w" in text:
        z, image = nums
        w = _div(image, z)
        assert all(v.denominator == 1 for v in w), text
        ans = (int(w[0]), int(w[1]))
    else:
        z1, image1, z2 = nums
        w = _div(image1, z1)
        assert all(v.denominator == 1 for v in w), text
        ans = _mul((int(w[0]), int(w[1])), z2)
    return f"x = {ans[0]}, y = {ans[1]}"


def test_key_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if expected != q.answer_text:
            mismatches.append((seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["pair"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert q.distractors, seed
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]


def test_deterministic_and_all_three_kinds_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    kinds = set()
    for seed in range(60):
        t = _exam(seed).text
        kinds.add("compose" if ("successive" in t or "w1" in t) else ("find w" if "Find w" in t else "apply"))
    assert kinds == {"compose", "find w", "apply"}
