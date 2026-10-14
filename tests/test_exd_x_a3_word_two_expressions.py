"""Tests for the exam source `exd_x_a3_word_two_expressions` (solve-equation-word-problems).

The oracle re-derives every key from the RENDERED PROMPT TEXT alone, with sympy / exact fractions, and never imports
the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy import Symbol, parse_expr

from mathgen import make
from mathgen.grading import GRADERS, mcq_options
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a3_word_two_expressions"
KIND = "number"
HARD = [("word_problem", {"kind": "linear_eq_reverse"})]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _implicit(s: str) -> str:
    """Insert explicit multiplication into school-style algebra text, e.g. '3(x + 1)' -> '3*(x + 1)'."""
    s = s.replace("^", "**")
    return re.sub(r"(?<=[0-9a-z\)])(?=[a-z(])", "*", s)


def _expr(s: str):
    names = {c: Symbol(c) for c in set(re.findall(r"[a-z]", s))}
    return parse_expr(_implicit(s), local_dict=names)


def _equation(s: str):
    left, right = s.split("=")
    return _expr(left) - _expr(right)


def _grade(q, given):
    if KIND == "mcq":
        return GRADERS["mcq"](q, str(mcq_options(q).index(given))).correct
    return GRADERS[KIND](q, given).correct


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_make(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    hard = set()
    for topic, params in HARD:
        hard |= {prompt_shape(make(topic, seed=s, difficulty="hard", **params)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_determinism():
    for seed in (0, 1, 2, 77, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_registered_exam_preset():
    make(TOPIC, seed=0, difficulty="exam")


def test_source_is_wired_into_the_skill():
    from mathgen.skills import load_skills

    skill = {s.id: s for s in load_skills()[1]}["solve-equation-word-problems"]
    last = skill.sources[-1]
    assert last.topic == TOPIC and last.difficulty == "exam" and list(last.stages) == ["exam"]


def test_key_and_distractors_graded_over_150_seeds():
    for seed in range(150):
        q = _make(seed)
        assert _grade(q, q.answer_text), (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not _grade_text(q, d), (seed, q.text, q.answer_text, d)


def _grade_text(q, given):
    """Grade a raw typed reply (for mcq every option is graded by its own index)."""
    if KIND == "mcq":
        return given == q.answer_text
    return GRADERS[KIND](q, given).correct


def _oracle(text: str) -> int:
    n = Symbol("n")
    if "Option A" in text:
        fa, ra, fb, rb = (int(v) for v in re.findall(r"\$(\d+)", text))
        sol = sympy.solve(fa + ra * n - fb - rb * n, n)
    elif "times as old" in text:
        m = int(re.search(r"(\d+) times as old", text).group(1))
        yrs = int(re.search(r"In (\d+) years", text).group(1))
        k = int(re.search(r"will be (\d+) times", text).group(1))
        sol = sympy.solve(m * n + yrs - k * (n + yrs), n)
    else:
        speeds = [int(v) for v in re.findall(r"(\d+) km/h", text)]
        head = int(re.search(r"(\d+) hours? (?:later|afterwards)", text).group(1))
        slow, fast = speeds
        sol = sympy.solve(slow * (n + head) - fast * n, n)
    assert len(sol) == 1 and sol[0] > 0 and sol[0].q == 1, (text, sol)
    return int(sol[0])


def test_key_rederived_from_rendered_prompt_over_400_seeds():
    for seed in range(400):
        q = _make(seed)
        assert _oracle(q.text) == int(q.answer_text), (seed, q.text, q.answer_text)


def test_three_situations_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("Option A" in t for t in texts)
    assert any("times as old" in t for t in texts)
    assert any("km/h" in t for t in texts)


def test_a_distractor_is_the_contextually_impossible_or_non_integer_slip_somewhere():
    seen_fraction = any("/" in d for s in range(100) for d in _make(s).distractors)
    assert seen_fraction
