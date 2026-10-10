"""Calls ``mathgen.core.make`` through each skill's own ``sources``. Parses ``question.text`` and
``question.answer_text`` alone and recomputes the answer from scratch. **Must not import anything from
``mathgen.generators.t5_annulus_frustum_build``** -- not the ``PI`` constant, not the rounding helper, not a
table. The oracle is allowed to know the STATED pi value and precision (both are given in the question text
itself: "Using pi (roughly) 3.14159" and "to 1 decimal place"), so both are hardcoded here independently
rather than imported.

Run over >= 40 seeds x every tier x both skills (720 questions total).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

_ORACLE_PI = Fraction(314159, 100000)
_SEEDS = 40


def _round_half_up_tenths(frac: Fraction) -> Fraction:
    """Independent half-up-to-tenths rounding, written fresh rather than
    imported from the generator module."""
    scaled = frac * 10 + Fraction(1, 2)
    whole = scaled.numerator // scaled.denominator
    return Fraction(whole, 10)


def _skill_sources(skill_id: str):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill.sources


def _oracle_annulus_area(text: str) -> Fraction:
    """Recompute pi * (R^2 - r^2) from the rendered text alone, converting
    to a single common unit itself."""
    m = re.search(r"outer radius of (\d+) cm and an inner radius of (\d+) cm", text)
    if m:
        R, r = int(m.group(1)), int(m.group(2))
        return _ORACLE_PI * (R * R - r * r)

    m = re.search(r"outer diameter of (\d+) cm and an inner radius of (\d+) cm", text)
    if m:
        D, r = int(m.group(1)), int(m.group(2))
        R = Fraction(D, 2)
        return _ORACLE_PI * (R * R - r * r)

    m = re.search(r"outer radius of (\d+) cm and an inner diameter of (\d+) cm", text)
    if m:
        R, d = int(m.group(1)), int(m.group(2))
        r = Fraction(d, 2)
        return _ORACLE_PI * (R * R - r * r)

    m = re.search(r"outer diameter of (\d+) mm and a wall thickness of ([\d.]+) cm", text)
    if m:
        D_mm, t_cm = int(m.group(1)), Fraction(m.group(2))
        R = Fraction(D_mm, 2)
        t_mm = t_cm * 10  # the oracle's own unit reconciliation, mm throughout
        r = R - t_mm
        return _ORACLE_PI * (R * R - r * r)

    raise AssertionError(f"oracle could not parse annulus question: {text!r}")


def _oracle_frustum_volume(text: str) -> Fraction:
    """Recompute (1/3) * pi * h * (R^2 + R*r + r^2) from the rendered text
    alone. For the slant-height (hard) form, h is derived independently via
    Pythagoras (`h = sqrt(l^2 - (R-r)^2)`, integer square root -- its own
    route, not a copy of the generator's Pythagorean-triple drawing)."""
    m = re.search(r"base radius of (\d+) cm, a top radius of (\d+) cm and a height of (\d+) cm", text)
    if m:
        R, r, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return Fraction(1, 3) * _ORACLE_PI * h * (R * R + R * r + r * r)

    m = re.search(r"base diameter of (\d+) cm, a top diameter of (\d+) cm and a height of (\d+) cm", text)
    if m:
        D_base, D_top, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
        R, r = Fraction(D_base, 2), Fraction(D_top, 2)
        return Fraction(1, 3) * _ORACLE_PI * h * (R * R + R * r + r * r)

    m = re.search(r"base radius of (\d+) cm, a top radius of (\d+) cm and a slant height of (\d+) cm", text)
    if m:
        R, r, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # h = sqrt(l^2 - (R-r)^2), via Newton's method on integers (math.isqrt)
        # -- a different route from however the generator assigned a scaled
        # Pythagorean triple to (R - r, h, l).
        h_sq = l * l - (R - r) * (R - r)
        assert h_sq >= 0, f"negative h^2 in oracle frustum parse: {text!r}"
        import math

        h = math.isqrt(h_sq)
        assert h * h == h_sq, f"l^2 - (R-r)^2 not a perfect square in {text!r}: {h_sq}"
        return Fraction(1, 3) * _ORACLE_PI * h * (R * R + R * r + r * r)

    raise AssertionError(f"oracle could not parse frustum question: {text!r}")


def _oracle_cases(skill_id: str):
    for source in _skill_sources(skill_id):
        params = source.params or {}
        for seed in range(_SEEDS):
            yield source.difficulty, seed, params


@pytest.mark.parametrize("difficulty,seed,params", list(_oracle_cases("annulus-cross-section-area")))
def test_annulus_oracle_agrees(difficulty, seed, params):
    q = make("annulus_cross_section_area", seed=seed, difficulty=difficulty, **params)
    expected_raw = _oracle_annulus_area(q.text)
    expected_rounded = _round_half_up_tenths(expected_raw)
    actual = Fraction(q.answer_text)
    assert expected_rounded == actual, (
        f"annulus [{difficulty}] seed={seed}: oracle says {expected_rounded} "
        f"(raw {expected_raw}), answer_text says {actual}\nQuestion: {q.text}"
    )


@pytest.mark.parametrize("difficulty,seed,params", list(_oracle_cases("frustum-volume")))
def test_frustum_oracle_agrees(difficulty, seed, params):
    q = make("frustum_volume", seed=seed, difficulty=difficulty, **params)
    expected_raw = _oracle_frustum_volume(q.text)
    expected_rounded = _round_half_up_tenths(expected_raw)
    actual = Fraction(q.answer_text)
    assert expected_rounded == actual, (
        f"frustum [{difficulty}] seed={seed}: oracle says {expected_rounded} "
        f"(raw {expected_raw}), answer_text says {actual}\nQuestion: {q.text}"
    )
