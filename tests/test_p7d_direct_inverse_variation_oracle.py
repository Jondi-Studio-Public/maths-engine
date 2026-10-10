"""Independent oracle for the direct and inverse proportion questions of p7d."""

import functools

"""Independent oracle for: direct and inverse variation.

Parses each rendered question with a regex, recomputes the expected answer
from the relation stated in the question text alone, and compares it to
``question.answer_text``. This file must NOT import anything from
``mathgen.generators.p7d_direct_inverse_variation_build`` -- it only calls
``mathgen.make`` through the skills' own ``sources`` and reads
``question.text``/``question.answer_text``.
"""
import re
from fractions import Fraction

import mathgen
from mathgen.skills import load_skills

SEEDS = range(40)

DIRECT_EASY_RE = re.compile(
    r"y is directly proportional to x, so y = kx\. "
    r"When x = (\d+), y = (\d+)\. What is y when x = (\d+)\?"
)
DIRECT_MEDIUM_RE = re.compile(
    r"y is directly proportional to x, so y = kx\. "
    r"When x = (\d+), y = (\d+)\. For what value of x does y = (\d+)\?"
)
DIRECT_HARD_RE = re.compile(
    r"extension is (\d+) cm, the force is (\d+) N\. "
    r"The extension is first increased so it is multiplied by (\d+), then increased again "
    r"so it is multiplied by a further (\d+)\. Find the new force, in N\."
)

INVERSE_EASY_RE = re.compile(
    r"y is inversely proportional to x, so y = k/x\. "
    r"When x = (\d+), y = (\d+)\. What is y when x = (\d+)\?"
)
INVERSE_MEDIUM_RE = re.compile(
    r"y is inversely proportional to x, so y = k/x\. "
    r"When x = (\d+), y = (\d+)\. For what value of x does y = (\d+)\?"
)
INVERSE_HARD_RE = re.compile(
    r"volume is (\d+) L, the pressure is (\d+) kPa\. "
    r"The volume is first increased so it is multiplied by (\d+), then increased again "
    r"so it is multiplied by a further (\d+)\. Find the new pressure, in kPa\."
)


# `load_skills()` rescans every skill YAML from disk and costs ~560 ms. These
# helpers are called once per SEED, so an uncached lookup spent ~290 s of the
# suite reloading the same library. Cached: `make()` still runs per seed and
# every assertion is unchanged -- only the repeated disk scan is removed.
@functools.lru_cache(maxsize=None)
def _sources_by_difficulty(skill_id):
    _domains, skills_list = load_skills()
    by_id = {s.id: s for s in skills_list}
    skill = by_id[skill_id]
    return {source.difficulty: source for source in skill.sources}


def _make(source, seed):
    params = source.params or {}
    return mathgen.make(source.topic, seed=seed, difficulty=source.difficulty, **params)


def test_direct_proportion_oracle():
    sources = _sources_by_difficulty("senior-examine-examples-of-direct-proportion-and-linearly-related")
    re_covered = 0

    for seed in SEEDS:
        q = _make(sources["easy"], seed)
        m = DIRECT_EASY_RE.search(q.text)
        assert m, f"easy: could not parse question text: {q.text!r}"
        x1, y1, x2 = (int(g) for g in m.groups())
        k = Fraction(y1, x1)
        expected = k * x2
        assert expected.denominator == 1, f"easy seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"easy seed {seed}: derived {int(expected)} != shipped {q.answer_text} (x1={x1}, y1={y1}, x2={x2})"
        )
        re_covered += 1

    for seed in SEEDS:
        q = _make(sources["medium"], seed)
        m = DIRECT_MEDIUM_RE.search(q.text)
        assert m, f"medium: could not parse question text: {q.text!r}"
        x1, y1, y2 = (int(g) for g in m.groups())
        k = Fraction(y1, x1)
        expected = Fraction(y2) / k
        assert expected.denominator == 1, f"medium seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"medium seed {seed}: derived {int(expected)} != shipped {q.answer_text} (x1={x1}, y1={y1}, y2={y2})"
        )
        re_covered += 1

    for seed in SEEDS:
        q = _make(sources["hard"], seed)
        m = DIRECT_HARD_RE.search(q.text)
        assert m, f"hard: could not parse question text: {q.text!r}"
        x1, f1, t1, t2 = (int(g) for g in m.groups())
        # F = kx: the extension is scaled twice (by t1, then by a further
        # t2), so the new force is F2 = F1*t1*t2 -- independently recomputed
        # here from the law statement (F = kx: derive k = F1/x1, apply it to
        # the new extension x1*t1*t2), not copied from the generator's own
        # "headline" scale-factor arithmetic name.
        k = Fraction(f1, x1)
        x2 = x1 * t1 * t2
        expected = k * x2
        assert expected.denominator == 1, f"hard seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"hard seed {seed}: derived {int(expected)} != shipped {q.answer_text} (x1={x1}, f1={f1}, t1={t1}, t2={t2})"
        )
        re_covered += 1

    assert re_covered == 3 * len(SEEDS)


def test_inverse_proportion_oracle():
    sources = _sources_by_difficulty("senior-examine-examples-of-inverse-proportion")
    re_covered = 0

    for seed in SEEDS:
        q = _make(sources["easy"], seed)
        m = INVERSE_EASY_RE.search(q.text)
        assert m, f"easy: could not parse question text: {q.text!r}"
        x1, y1, x2 = (int(g) for g in m.groups())
        k = x1 * y1
        expected = Fraction(k, x2)
        assert expected.denominator == 1, f"easy seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"easy seed {seed}: derived {int(expected)} != shipped {q.answer_text} (x1={x1}, y1={y1}, x2={x2})"
        )
        re_covered += 1

    for seed in SEEDS:
        q = _make(sources["medium"], seed)
        m = INVERSE_MEDIUM_RE.search(q.text)
        assert m, f"medium: could not parse question text: {q.text!r}"
        x1, y1, y2 = (int(g) for g in m.groups())
        k = x1 * y1
        expected = Fraction(k, y2)
        assert expected.denominator == 1, f"medium seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"medium seed {seed}: derived {int(expected)} != shipped {q.answer_text} (x1={x1}, y1={y1}, y2={y2})"
        )
        re_covered += 1

    for seed in SEEDS:
        q = _make(sources["hard"], seed)
        m = INVERSE_HARD_RE.search(q.text)
        assert m, f"hard: could not parse question text: {q.text!r}"
        v1, p1, t1, t2 = (int(g) for g in m.groups())
        # P = k/V: the volume is scaled twice (by t1, then by a further t2),
        # so the new pressure is P2 = P1/(t1*t2) -- independently recomputed
        # here from the law statement (P = k/V: derive k = V1*P1, apply it to
        # the new volume V1*t1*t2), not copied from the generator's own
        # "headline" scale-factor arithmetic name.
        k = v1 * p1
        v2 = v1 * t1 * t2
        expected = Fraction(k, v2)
        assert expected.denominator == 1, f"hard seed {seed}: derived answer not an integer: {expected}"
        assert int(expected) == int(q.answer_text), (
            f"hard seed {seed}: derived {int(expected)} != shipped {q.answer_text} (v1={v1}, p1={p1}, t1={t1}, t2={t2})"
        )
        re_covered += 1

    assert re_covered == 3 * len(SEEDS)
