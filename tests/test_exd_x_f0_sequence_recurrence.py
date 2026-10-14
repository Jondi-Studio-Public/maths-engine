"""Exam source of `generate-sequence-from-rule`: the term is recomputed from the rendered rule."""

from __future__ import annotations

import re

from sympy import Symbol, sympify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f0_sequence_recurrence"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("function_evaluation", seed=seed, difficulty="hard", a_lo=0)


def _coefs(rule: str, names: list[str]) -> list[int]:
    """'-t(n + 1) - 2t(n) + 3' -> coefficient of each name (and the constant last)."""
    out = []
    rest = rule.replace(" ", "")
    for name in names:
        key = name.replace(" ", "")
        m = re.search(rf"([+-]?\d*){re.escape(key)}", rest)
        sign_digits = m.group(1)
        out.append({"": 1, "+": 1, "-": -1}.get(sign_digits, None) or int(sign_digits))
        rest = rest[: m.start()] + rest[m.end() :]
    out.append(int(rest) if rest else 0)
    return out


def derive(text: str) -> int:
    m = re.match(
        r"A sequence is defined by t\(1\) = (-?\d+) and t\(n \+ 1\) = (.+?) for every n ≥ 1\. Find t\((\d+)\)\.", text
    )
    if m:
        a, rule, k = int(m.group(1)), m.group(2), int(m.group(3))
        p, q = _coefs(rule, ["t(n)"])
        t = a
        for _ in range(k - 1):
            t = p * t + q
        return t
    m = re.match(
        r"A sequence has t\(1\) = (-?\d+), t\(2\) = (-?\d+) and t\(n \+ 2\) = (.+?) for every n ≥ 1\. Find t\((\d+)\)\.",
        text,
    )
    if m:
        a, b, rule, k = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        p, q, rest = _coefs(rule, ["t(n + 1)", "t(n)"])
        assert rest == 0
        terms = [a, b]
        while len(terms) < k:
            terms.append(p * terms[-1] + q * terms[-2])
        return terms[k - 1]
    m = re.match(
        r"The sum of the first n terms of a sequence is S\(n\) = (.+?)\. Find the (\d+)(?:st|nd|rd|th) term", text
    )
    n = Symbol("n")
    s = sympify(re.sub(r"(\d)n", r"\1*n", m.group(1).replace("^", "**")), locals={"n": n})
    k = int(m.group(2))
    return int(s.subs(n, k) - s.subs(n, k - 1))


def test_exam_adds_shapes_over_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_q(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_key_matches_independent_derivation_over_300_seeds():
    bad = [(s, _q(s).text, _q(s).answer) for s in range(300) if derive(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
