"""Tests for the exam source of `senior-determine-when-the-composition-of-two-functions-is`.

The oracle reads the rendered prompt only. It parses the stated f and g, decides by direct evaluation (exact
fractions for the reciprocal, floating point for the roots) where the requested composite exists on a fine grid
that includes the critical points, parses the key's description into a predicate and requires the two to agree
everywhere. It also requires each distractor to disagree with direct evaluation somewhere.
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

_F_RE = re.compile(r"f\(x\) = 1/\(x ([+-]) (\d+)\)")
_G_RE = re.compile(r"g\(x\) = sqrt\(x ([+-]) (\d+)\)")
_ASK_RE = re.compile(r"(f\(g\(f\(x\)\)\)|f\(f\(x\)\)|g\(f\(x\)\)|g\(g\(x\)\))")
_NUM = r"(-?\d+(?:/\d+)?)"


def _shift(sign, n):
    return int(n) if sign == "-" else -int(n)


def _evaluate(q):
    ask = _ASK_RE.search(q.text).group(1)
    f_m, g_m = _F_RE.search(q.text), _G_RE.search(q.text)
    k = _shift(*f_m.groups()) if f_m else None
    a = _shift(*g_m.groups()) if g_m else None

    def f(v):
        return 1 / (Fraction(v) - k)

    def g(v):
        return math.sqrt(float(v) - a)

    def composite(v):
        if ask == "f(f(x))":
            return f(f(v))
        if ask == "g(f(x))":
            return g(f(v))
        if ask == "g(g(x))":
            return g(g(v))
        inner = g(f(v))
        if abs(inner - k) < 1e-12:
            raise ZeroDivisionError
        return 1 / (inner - k)

    return ask, k, a, composite


def _defined(composite, v):
    try:
        composite(v)
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _fraction(text):
    return Fraction(text)


def _predicate(description):
    m = re.fullmatch(r"Defined for all real x, except x = " + _NUM + r" and x = " + _NUM, description)
    if m:
        bad = {_fraction(m.group(1)), _fraction(m.group(2))}
        return lambda v: v not in bad
    m = re.fullmatch(r"Defined for all real x, except x = " + _NUM, description)
    if m:
        bad = {_fraction(m.group(1))}
        return lambda v: v not in bad
    m = re.fullmatch(r"Defined for x >= " + _NUM, description)
    if m:
        lo = _fraction(m.group(1))
        return lambda v: v >= lo
    m = re.fullmatch(r"Defined for x <= " + _NUM + r"(?:, except x = " + _NUM + r")?", description)
    if m:
        hi = _fraction(m.group(1))
        hole = _fraction(m.group(2)) if m.group(2) else None
        return lambda v: v <= hi and v != hole
    interval = r"Defined for " + _NUM + r" (<|<=) x (<|<=) " + _NUM + r"(?:, except x = " + _NUM + r")?"
    m = re.fullmatch(interval, description)
    if m:
        lo, op1, op2, hi, hole = m.groups()
        lo, hi = _fraction(lo), _fraction(hi)
        hole = _fraction(hole) if hole else None

        def pred(v):
            left = lo < v if op1 == "<" else lo <= v
            right = v < hi if op2 == "<" else v <= hi
            return left and right and v != hole

        return pred
    raise AssertionError(f"unrecognised description {description!r}")


TOPIC = "exd_x_f2_when_defined"
SKILL_ID = "senior-determine-when-the-composition-of-two-functions-is"
KIND = "mcq"


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


def test_prompt_variety_over_60_seeds():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


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


def _grid(q):
    ask, k, a, _ = _evaluate(q)
    points = set()
    for i in range(-80, 81):
        points.add(Fraction(i, 4))
    centres = [Fraction(k)] if k is not None else []
    if a is not None:
        centres.append(Fraction(a))
    for c in centres:
        for i in range(-40, 41):
            points.add(c + Fraction(i, 12))
    nums = [Fraction(n) for n in re.findall(r"-?\d+(?:/\d+)?", q.answer_text + " " + " ".join(q.distractors))]
    for n in nums:
        points.update({n, n + Fraction(1, 50), n - Fraction(1, 50)})
    if k is not None:
        for n in range(1, 8):
            points.add(Fraction(k) + Fraction(1, n))
            points.add(Fraction(k) + Fraction(1, n + k * k))
            points.add(Fraction(k) + Fraction(1, n * (n + 1)))
        points.add(Fraction(k) + Fraction(1, k) if k else Fraction(0))
    return sorted(points)


def test_key_matches_direct_evaluation_over_300_seeds():
    asks = set()
    for seed in range(300):
        q = _exam(seed)
        ask, _, _, composite = _evaluate(q)
        asks.add(ask)
        pred = _predicate(q.answer_text)
        for pt in _grid(q):
            assert _defined(composite, pt) == pred(pt), (seed, q.text, q.answer_text, pt)
    assert asks == {"f(f(x))", "g(f(x))", "g(g(x))", "f(g(f(x)))"}


def test_every_distractor_disagrees_with_direct_evaluation_somewhere():
    for seed in range(120):
        q = _exam(seed)
        _, _, _, composite = _evaluate(q)
        grid = _grid(q)
        for d in q.distractors:
            pred = _predicate(d)
            assert any(_defined(composite, pt) != pred(pt) for pt in grid), (seed, q.text, d)
