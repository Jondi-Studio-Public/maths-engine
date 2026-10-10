"""Fourier-coefficient-simple hard tier: step function split at pi/2."""

from __future__ import annotations

import re
from collections import Counter

import sympy

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

PATTERN = re.compile(
    r"On \[0, π\], let f\(x\) = (\d+) for 0 ≤ x ≤ π/2 and f\(x\) = 0 for π/2 < x ≤ π\. "
    r"Compute its exact (sine|cosine) Fourier coefficient for mode n = (\d+)\."
)


def _step(seed):
    return make("fourier_coefficient_simple", seed=seed, difficulty="hard", kind="step")


def _independent(c, family, n):
    # Closed form by cases on n mod 4, not by sympy integration.
    r = n % 4
    if family == "sine":
        num = {0: 0, 1: 1, 2: 2, 3: 1}[r]  # 1 - cos(n pi/2)
    else:
        num = {0: 0, 1: 1, 2: 0, 3: -1}[r]  # sin(n pi/2)
    return sympy.Rational(2 * c * num, 1) / (n * sympy.pi)


def test_step_answer_matches_recomputation_from_rendered_text():
    for seed in range(60):
        q = _step(seed)
        m = PATTERN.match(q.text)
        assert m, q.text
        c, family, n = int(m.group(1)), m.group(2), int(m.group(3))
        expected = _independent(c, family, n)
        assert expected != 0, seed
        assert sympy.simplify(q.answer - expected) == 0, (seed, q.text, q.answer_text)


def test_default_path_unchanged_and_new_kind_differs():
    for seed in range(20):
        default = make("fourier_coefficient_simple", seed=seed, difficulty="hard")
        explicit = make("fourier_coefficient_simple", seed=seed, difficulty="hard", kind=None)
        assert default.text == explicit.text
        assert default.answer_text == explicit.answer_text
        assert "f(x) = x^" in default.text
        assert _step(seed).text != default.text


def test_distractors_rejected_and_answer_accepted_over_200_seeds():
    grader = GRADERS["expression"]
    for seed in range(200):
        q = _step(seed)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        assert q.distractors, seed
        for d in q.distractors:
            assert d != q.answer_text
            assert not grader(q, d).correct, (seed, d, q.answer_text)


def test_equivalent_unsimplified_form_accepted():
    grader = GRADERS["expression"]
    for seed in range(200):
        q = _step(seed)
        m = PATTERN.match(q.text)
        c, family, n = int(m.group(1)), m.group(2), int(m.group(3))
        if family == "sine":
            form = f"2*{c}*(1-cos({n}*pi/2))/({n}*pi)"
        else:
            form = f"2*{c}*sin({n}*pi/2)/({n}*pi)"
        assert grader(q, form).correct, (seed, form, q.answer_text)


def test_deliberately_wrong_answer_rejected():
    q = _step(5)
    assert not GRADERS["expression"](q, f"({q.answer_text})+1").correct


def test_prompt_shape_differs_from_easy_and_medium():
    shapes = {prompt_shape(make("fourier_coefficient_simple", seed=1, difficulty=d)) for d in ("easy", "medium")}
    assert prompt_shape(_step(1)) not in shapes


def test_answer_concentration_is_reasonable():
    counts = Counter(_step(s).answer_text for s in range(200))
    assert counts.most_common(1)[0][1] / 200 < 0.25
