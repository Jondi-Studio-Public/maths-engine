"""Tests for the exam source of `senior-examine-examples-of-inverse-proportion`.

The oracle reads the rendered prompt only and recomputes the answer with exact fractions by a route of its own for
each question shape (constant of proportionality, worker-days, or the stated scale factors).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_DATA_A = re.compile(r"When x = (\d+), y = (\d+)")
_DATA_B = re.compile(r"y = (\d+) when x = (\d+)")


def _data(text):
    m = _DATA_A.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _DATA_B.search(text)
    return int(m.group(2)), int(m.group(1))


TOPIC = "exd_x_f2_inverse_proportion"
SKILL_ID = "senior-examine-examples-of-inverse-proportion"
KIND = "number"


def _skill():
    _, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    source = next(s for s in _skill().sources if s.difficulty == "hard")
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (source.params or {}).items()}
    return make(source.topic, seed=seed, difficulty="hard", **params)


def _grade(q, given):
    if KIND == "mcq":
        options = mcq_options(q)
        given = options.index(given) if given in options else -1
    return GRADERS[KIND](q, given).correct


# -- wiring, text quality and variety ------------------------------------------


def test_exam_source_is_the_last_source_of_the_skill():
    source = _skill().sources[-1]
    assert source.topic == TOPIC
    assert source.difficulty == "exam"
    assert list(source.stages) == ["exam"]
    assert dict(source.params) == {"tier": "exam"}
    assert _skill().answer == KIND


def test_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _exam(seed)
        defects = find_text_defects(q, include_working=True)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_exam_adds_shapes_not_present_in_the_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_determinism():
    for seed in (0, 1, 2, 3, 100, 12345):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_preset_registered_and_other_tiers_rejected():
    make(TOPIC, seed=0, difficulty="exam")
    try:
        make(TOPIC, seed=0, difficulty="hard")
    except KeyError:
        return
    raise AssertionError("only the exam preset should exist")


def test_three_distinct_distractors_and_none_equal_to_key():
    for seed in range(120):
        q = _exam(seed)
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors


# -- the grader: key accepted, every distractor rejected -----------------------


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    for seed in range(120):
        q = _exam(seed)
        assert _grade(q, q.answer_text), (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert not _grade(q, d), (seed, q.text, "distractor accepted", d)


def _expected(q):
    t = q.text
    if "workers" in t:
        n, total, d, leave = (int(v) for v in re.findall(r"\d+", t))
        return "workers", Fraction(n * total - n * d, n - leave)
    x1, y1 = _data(t)
    k = x1 * y1
    if "times" in t:
        c = int(re.search(r"(\d+) times", t).group(1))
        root = math.isqrt(k // c)
        assert root * root * c == k, (t, k, c)
        return "crossing", Fraction(root)
    moves = re.findall(r"(increased|decreased) by (\d+)%", t)
    assert len(moves) == 2
    x = Fraction(x1)
    for word, pct in moves:
        x *= 1 + (1 if word == "increased" else -1) * Fraction(int(pct), 100)
    return "percent", k / x


def test_key_matches_independent_recomputation_over_300_seeds():
    shapes = set()
    for seed in range(300):
        q = _exam(seed)
        shape, expected = _expected(q)
        shapes.add(shape)
        assert expected.denominator == 1, (seed, q.text)
        assert str(expected) == q.answer_text, (seed, q.text, expected, q.answer_text)
    assert shapes == {"workers", "crossing", "percent"}
