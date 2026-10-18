"""Exam source for `use-nth-term-rule`: recover a rule from two terms, find the first term to exceed a
threshold, or count the terms of a rule inside an interval. Keys are re-derived from the rendered prompt."""

from __future__ import annotations

import re

from sympy import Symbol, lambdify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen
import mathgen.generators.exd_x_f4_nth_term  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_nth_term"
_N = Symbol("n")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("sequence", seed=seed, difficulty="hard", kind="use_rule_inverse")


def _oracle(text: str) -> int:
    if "arithmetic" in text:
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
        if "as its" in text:
            t1, p1, t2, p2, k = nums
        else:
            p1, t1, p2, t2, k = nums
        d = (t2 - t1) // (p2 - p1)
        assert d * (p2 - p1) == t2 - t1
        return t1 + d * (k - p1)
    rule_text = re.search(r"T_n = ([^.,]+)", text).group(1)
    rule = lambdify(_N, parse_expr(rule_text, transformations=_TRANSFORMS, local_dict={"n": _N}))
    tail = text[text.index(rule_text) + len(rule_text) :]
    nums = [int(x) for x in re.findall(r"-?\d+", tail)]
    if re.search(r"exceed|greater than|least positive", text):
        m = nums[-1]
        return next(n for n in range(1, 400) if rule(n) > m)
    lo, hi = nums[-2], nums[-1]
    return sum(1 for n in range(1, 400) if lo <= rule(n) <= hi)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_make(s)) for s in range(80)}
    hard = {prompt_shape(_hard(s)) for s in range(80)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_all_three_kinds_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("arithmetic" in t for t in texts)
    assert any(re.search(r"exceed|greater than|least positive", t) for t in texts)
    assert any(re.search(r"between|interval|at least", t) for t in texts)


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if _oracle(q.text) != q.answer:
            bad.append((seed, q.text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _make(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, d)


def test_ordinals_use_correct_suffixes():
    wrong = re.compile(
        r"\b(?:\d*(?<!1)[123]th|\d*1[123](?:st|nd|rd)|\d*(?<![1])1(?:nd|rd|th)|\d*(?<!1)2(?:st|rd|th)|\d*(?<!1)3(?:st|nd|th))\b"
    )
    seen = set()
    for seed in range(300):
        text = _make(seed).text
        assert not wrong.search(text), text
        seen.update(re.findall(r"\b\d+(?:st|nd|rd|th)\b", text))
    assert any(s.endswith(("st", "nd", "rd")) for s in seen), seen
