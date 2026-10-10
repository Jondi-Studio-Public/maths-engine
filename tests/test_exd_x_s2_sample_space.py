"""Exam-tier tests for `exd_x_s2_sample_space` (conditional probability over two dice).

The oracle parses the rendered prompt (die sizes and the two clauses), enumerates the ordered outcomes itself and
counts; it shares nothing with the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s2_sample_space"

_FIXED = {
    "the sum is even": lambda a, b: (a + b) % 2 == 0,
    "the sum is a multiple of 3": lambda a, b: (a + b) % 3 == 0,
    "the product is even": lambda a, b: a * b % 2 == 0,
    "the first die shows a smaller number than the second": lambda a, b: a < b,
    "the two numbers differ by exactly 1": lambda a, b: abs(a - b) == 1,
    "the two numbers are different": lambda a, b: a != b,
    "the first number divides the second": lambda a, b: b % a == 0,
}


def _clause(name):
    if name in _FIXED:
        return _FIXED[name]
    m = re.fullmatch(r"the sum is at least (\d+)", name)
    if m:
        return lambda a, b, k=int(m.group(1)): a + b >= k
    m = re.fullmatch(r"the larger number is at most (\d+)", name)
    assert m, name
    return lambda a, b, k=int(m.group(1)): max(a, b) <= k


def _oracle(text):
    n, m = (int(x) for x in re.search(r"(\d+)(?:-sided die| faces).*?(\d+)(?:-sided die| faces)", text).groups())
    clauses = "|".join(re.escape(c) for c in _FIXED) + r"|the sum is at least \d+|the larger number is at most \d+"
    if text.startswith("Roll"):
        a_name, b_name = re.search(rf"P\(({clauses}) \| ({clauses})\)", text).groups()
    elif "Given that" in text:
        b_name, a_name = re.search(
            rf"Given that ({clauses}), what is the probability that ({clauses})\?", text
        ).groups()
    else:
        b_name, a_name = re.search(rf"known that ({clauses})\. Find the probability that ({clauses})\.", text).groups()
    fa, fb = _clause(a_name), _clause(b_name)
    space = [(a, b) for a in range(1, n + 1) for b in range(1, m + 1)]
    given = [s for s in space if fb(*s)]
    return Fraction(sum(1 for s in given if fa(*s)), len(given))


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


def test_shape_variety_and_novelty_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("finite_sample_space_probability", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert len(exam - hard) >= 3, (exam, hard)


def test_deterministic():
    for seed in (0, 1, 5, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_oracle_matches_key_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        got = _oracle(q.text)
        if got != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((seed, q.text, got, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["fraction"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct, seed
        assert len(q.distractors) >= 2, (seed, q.distractors)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
