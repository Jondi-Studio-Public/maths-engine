"""Exam tier of the binomial point-probability skill: tails, ranges and a conditional event, to 4 d.p.

The oracle reads n, p and the event from the rendered prompt and adds the exact probabilities of the success
counts that satisfy it, using `fractions` and `math.comb`. It shares no code with the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import comb

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_binomial_point_probability"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_easy_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    easy = {prompt_shape(make("binomial_probability", seed=s, difficulty="easy")) for s in range(60)}
    assert len(shapes - easy) >= 3, (shapes, easy)


def _pmf(n, p, k):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def _oracle(text: str) -> Fraction:
    p = Fraction(re.search(r"probability (0\.\d+)", text).group(1))
    n = int(re.search(r"(\d+) (?:questions|items|shots|seeds)", text).group(1))
    if "exactly" in text:
        x = int(re.search(r"is exactly (\d+)", text).group(1))
        return _pmf(n, p, x) / (1 - _pmf(n, p, 0))
    if "between" in text:
        a, b = (int(v) for v in re.search(r"between (\d+) and (\d+)", text).groups())
        return sum(_pmf(n, p, k) for k in range(a, b + 1))
    m = int(re.search(r"is at least (\d+)[.?] Give", text).group(1))
    return sum(_pmf(n, p, k) for k in range(m, n + 1))


def test_key_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _exam(seed)
        exact = _oracle(q.text)
        shown = Fraction(q.answer_text)
        if abs(shown - exact) > Fraction(1, 20000) or len(q.answer_text.split(".")[1]) != 4:
            mismatches.append((seed, q.text, float(exact), q.answer_text))
    assert mismatches == [], mismatches[:3]


def test_grader_accepts_key_and_rejects_every_distractor_and_near_misses():
    grade = GRADERS["number"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert q.distractors, seed
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
        key = Fraction(q.answer_text)
        for off in (Fraction(1, 10000), Fraction(-1, 10000)):
            near = f"{float(key + off):.4f}"
            if grade(q, near).correct:
                bad.append(("one-unit miss accepted", seed, near, q.answer_text))
    assert bad == [], bad[:3]


def test_deterministic_and_all_three_events_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    events = set()
    for seed in range(60):
        t = _exam(seed).text
        events.add(("exactly" in t, "between" in t))
    assert events == {(True, False), (False, True), (False, False)}
