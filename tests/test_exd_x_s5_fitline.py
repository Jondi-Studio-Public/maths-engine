"""Tests for the exam source of senior-use-the-equation-of-a-fitted-line-to and senior-use-the-line-of-best-fit-to-make.

Both keys are re-derived from the rendered prompt: the threshold year by scanning years forward with exact
fractions, the best-fit prediction from the summary statistics.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_s5_fitline  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


SKILLS = [
    ("senior-use-the-equation-of-a-fitted-line-to", "exd_x_s5_fitted_line_threshold"),
    ("senior-use-the-line-of-best-fit-to-make", "exd_x_s5_best_fit_from_summary"),
]
HARD = {
    "senior-use-the-equation-of-a-fitted-line-to": ("use_the_equation_of_a_fitted_line_to", "hard"),
    "senior-use-the-line-of-best-fit-to-make": ("p8c_line_best_fit_predict", "hard"),
}
ANSWER_KIND = {
    "senior-use-the-equation-of-a-fitted-line-to": "number",
    "senior-use-the-line-of-best-fit-to-make": "number",
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    return next(s for s in skills if s.id == skill_id).sources


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_skill_declares_exam_source_last(skill_id, topic):
    src = _skill_sources(skill_id)[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (topic, "exam", ["exam"], {"tier": "exam"})


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_no_text_defects(skill_id, topic):
    for seed in range(60):
        q = _exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_prompt_variety_and_new_shapes_over_hard_tier(skill_id, topic):
    hard_topic, hard_difficulty = HARD[skill_id]
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {prompt_shape(make(hard_topic, seed=s, difficulty=hard_difficulty)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_determinism(skill_id, topic):
    for seed in (0, 1, 7, 99):
        assert _exam(topic, seed).to_json() == _exam(topic, seed).to_json()


@pytest.mark.parametrize("skill_id,topic", SKILLS)
def test_grader_accepts_key_and_rejects_every_distractor(skill_id, topic):
    grader = GRADERS[ANSWER_KIND[skill_id]]
    for seed in range(150):
        q = _exam(topic, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (seed, q.distractors)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d, q.answer_text)


T1, T2 = "exd_x_s5_fitted_line_threshold", "exd_x_s5_best_fit_from_summary"
_EQ = re.compile(r"V = (-?[\d.]+) t \+ ([\d.]+)")
_BASE = re.compile(r"(?:after|since) (\d{4})")
_P1 = re.compile(r"\((\d{4}), ([\d.]+)\) and \((\d{4}), ([\d.]+)\)")
_P2 = re.compile(r"was ([\d.]+) in (\d{4}) and ([\d.]+) in (\d{4})")


def _threshold_oracle(text):
    level = Fraction(re.findall(r"\d+(?:\.\d+)?", text)[-1])
    rising = "exceed" in text or "greater than" in text
    assert rising or "fall below" in text or "less than" in text
    eq = _EQ.search(text)
    if eq:
        m, c = Fraction(eq.group(1)), Fraction(eq.group(2))
        base = int(_BASE.search(text).group(1))

        def value(year):
            return m * (year - base) + c

        start = base
    else:
        g = _P1.search(text)
        if g:
            y1, v1, y2, v2 = int(g.group(1)), Fraction(g.group(2)), int(g.group(3)), Fraction(g.group(4))
        else:
            g = _P2.search(text)
            v1, y1, v2, y2 = Fraction(g.group(1)), int(g.group(2)), Fraction(g.group(3)), int(g.group(4))
        slope = (v2 - v1) / (y2 - y1)

        def value(year):
            return v1 + slope * (year - y1)

        start = y2
    year = start
    while not (value(year) > level if rising else value(year) < level):
        year += 1
        assert year < start + 200
    return year


def test_threshold_key_matches_independent_scan_over_400_seeds():
    bad = []
    for s in range(400):
        q = _exam(T1, s)
        if _threshold_oracle(q.text) != q.answer:
            bad.append((s, q.text, q.answer))
    assert bad == []


def test_threshold_uses_both_prompt_forms_and_both_directions():
    texts = [_exam(T1, s).text for s in range(120)]
    assert any("V = " in t for t in texts) and any("V = " not in t for t in texts)
    assert any("fall below" in t or "less than" in t for t in texts)
    assert any("exceed" in t or "greater than" in t for t in texts)


def _bestfit_oracle(text):
    body = text.replace("1 decimal place", "")
    nums = [Fraction(v) for v in re.findall(r"-?\d+(?:\.\d+)?", body)]
    if body.startswith("A study"):
        r, xb, sx, yb, sy, x0 = nums
    else:
        xb, sx, yb, sy, r, x0 = nums
    return yb + r * sy / sx * (x0 - xb)


def test_bestfit_key_matches_independent_derivation_over_400_seeds():
    bad = []
    for s in range(400):
        q = _exam(T2, s)
        y = _bestfit_oracle(q.text)
        if (y * 10).denominator != 1 or f"{float(y):.1f}" != q.answer_text:
            bad.append((s, q.text, q.answer_text, y))
    assert bad == []


def test_bestfit_wrong_values_rejected():
    grader = GRADERS["number"]
    q = _exam(T2, 5)
    v = float(q.answer_text)
    for wrong in (v + 0.1, v - 0.1, v + 1, -v):
        assert not grader(q, f"{wrong:.1f}").correct, wrong


def test_best_fit_correlation_sign_matches_the_context():
    from mathgen import make as _make

    for seed in range(300):
        text = _make("exd_x_s5_best_fit_from_summary", seed=seed, difficulty="exam").text
        r = float(re.search(r"r = (-?\d?\.\d+)", text).group(1))
        if "age of a car" in text or "hot drinks" in text:
            assert r < 0, text
        else:
            assert r > 0, text
