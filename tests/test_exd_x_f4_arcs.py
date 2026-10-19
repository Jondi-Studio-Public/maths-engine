"""Exam source for `senior-calculate-lengths-of-arcs-and-areas-of-sectors`: perimeter of a sector from its
area, the exact area of a segment, and the base radius of a cone made from a sector. Keys are re-derived from
the rendered prompt with exact sympy arithmetic."""

from __future__ import annotations

import re

import sympy as sp

import mathgen
import mathgen.generators.exd_x_f4_arcs  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_arcs"


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("arc_length_sector_area", seed=seed, difficulty="hard", variant="find_radius")


def _pi_multiple(token: str):
    m = re.fullmatch(r"(\d*)pi(?:/(\d+))?", token)
    assert m, token
    return sp.Rational(int(m.group(1) or 1), int(m.group(2) or 1)) * sp.pi


def _oracle(text: str):
    if "perimeter" in text:
        theta = _pi_multiple(re.search(r"(\S*pi\S*) radians", text).group(1))
        area = _pi_multiple(re.search(r"(\S*pi\S*) cm\^2", text).group(1))
        r = sp.sqrt(2 * area / theta)
        return 2 * r + r * theta
    if "base" in text:
        big_r = int(re.search(r"radius (\d+) cm", text).group(1))
        theta = _pi_multiple(re.search(r"(\S*pi\S*) radians", text).group(1))
        return big_r * theta / (2 * sp.pi)
    r = int(re.search(r"radius (\d+) cm", text).group(1))
    theta = _pi_multiple(re.search(r"(\S*pi\S*) radians", text).group(1))
    return sp.Rational(1, 2) * r**2 * (theta - sp.sin(theta))


def _agrees(q) -> bool:
    return sp.simplify(_oracle(q.text) - sp.sympify(q.answer_text)) == 0


def test_all_three_settings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("perimeter" in t for t in texts)
    assert any("segment" in t or "chord" in t for t in texts)
    assert any("cone" in t for t in texts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_make(s)) for s in range(80)}
    hard = {prompt_shape(_hard(s)) for s in range(80)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["expression"]
    for seed in range(150):
        q = _make(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, d)
