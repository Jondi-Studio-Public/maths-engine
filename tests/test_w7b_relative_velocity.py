"""Tests `relative-velocity-vectors` (topic `w7b_relative_velocity`) and its exam chain `exd_x_w7b_relative_velocity`.

The oracles read the velocities, positions and speeds back out of the question TEXT and re-derive the key with plain
arithmetic and SymPy calculus (derivative of the squared distance, the interception quadratic), never from the
generator's own helpers.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy

import mathgen.generators.w7b_relative_velocity_build  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

TOPIC = "w7b_relative_velocity"
EXAM = "exd_x_w7b_relative_velocity"
SKILL = sup.skill_of("relative-velocity-vectors")

_VEC = r"(-?\d*[ij](?: [+-] \d*[ij])?)"
_VEC_RE = re.compile(_VEC + r" (?:km/h|m/s)")
_PT_RE = re.compile(r"\((-?\d+), (-?\d+)\)")


def _vec(s: str) -> tuple[int, int]:
    x = y = 0
    for sign, num, axis in re.findall(r"([+-]?)\s*(\d*)([ij])", s):
        value = int(num or 1) * (-1 if sign == "-" else 1)
        if axis == "i":
            x = value
        else:
            y = value
    return x, y


def _vectors(text: str):
    return [_vec(m) for m in _VEC_RE.findall(text)]


def _points(text: str):
    return [(int(a), int(b)) for a, b in _PT_RE.findall(text)]


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "approach-sign-error" in {m.id for m in SKILL.misconceptions}
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_independent_arithmetic():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "easy", seed)
        text = q.text
        if "straight road" in text or "Car A" in text:
            nums = [int(n) for n in re.findall(r"(\d+) km/h", text)]
            if "behind" in text or "ahead" in text:
                gap = int(re.search(r"(\d+) km", text).group(1))
                va, vb = nums[-2], nums[-1]
                # the car behind is A in the first wording and the car ahead is B in the second; A is faster either way
                assert gap == int(q.answer) * (va - vb), (seed, text)
                seen.add("catch-up")
            else:
                va, vb = nums
                same = "same direction" in text
                assert int(q.answer) == (abs(va - vb) if same else va + vb), (seed, text)
                seen.add("closing")
        elif text.startswith("With i east"):
            known, rel = _vectors(text)
            comp = 0 if "i-component" in text else 1
            if (
                "Find the " in text
                and "of the velocity of " in text
                and re.search(r"velocity of \w+ B relative to the ground", text)
            ):
                assert int(q.answer) == known[comp] - rel[comp], (seed, text)  # B = A - (A rel B)
                seen.add("reverse-b")
            else:
                assert int(q.answer) == rel[comp] + known[comp], (seed, text)  # A = (A rel B) + B
                seen.add("reverse-a")
        else:
            va, vb = _vectors(text)
            comp = 0 if "i-component" in text else 1
            if re.search(r"velocity of B relative to A", text):
                want = vb[comp] - va[comp]
            else:
                want = va[comp] - vb[comp]
            assert int(q.answer) == want, (seed, text)
            seen.add("component")
    assert seen == {"catch-up", "closing", "reverse-a", "reverse-b", "component"}


def test_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "medium", seed)
        text = q.text
        if text.startswith("At time 0"):
            ra, rb = _points(text)
            va, vb = _vectors(text)
            t = int(re.search(r"After (\d+) ", text).group(1))
            comp = 0 if "x-coordinate" in text else 1
            want = (ra[comp] + va[comp] * t) - (rb[comp] + vb[comp] * t)
            assert int(q.answer) == want, (seed, text)
            seen.add("position")
        elif (
            "velocity" in text
            and ("Drone" in text or "Ship" in text or "Boat" in text or "Aircraft" in text)
            and _vectors(text)
        ):
            va, vb = _vectors(text)
            assert int(q.answer) == math.isqrt((va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2)
            assert int(q.answer) ** 2 == (va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2
            seen.add("components")
        else:
            u, v = (int(n) for n in re.findall(r"(\d+) (?:km/h|m/s)", text)[:2])
            assert int(q.answer) ** 2 == u * u + v * v, (seed, text)
            seen.add("perpendicular")
    assert seen == {"position", "components", "perpendicular"}


def _closest(text: str):
    ra, rb = _points(text)
    va, vb = _vectors(text)
    t = sympy.symbols("t")
    dx = ra[0] - rb[0] + (va[0] - vb[0]) * t
    dy = ra[1] - rb[1] + (va[1] - vb[1]) * t
    d2 = sympy.expand(dx**2 + dy**2)
    tstar = sympy.solve(sympy.diff(d2, t), t)[0]
    return tstar, sympy.sqrt(d2.subs(t, tstar)), d2, t


def test_hard_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        text = q.text
        if text.startswith("A coastguard"):
            ra, rb = _points(text)
            (vb,) = _vectors(text)
            speed = int(re.search(r"speed of (\d+) km/h", text).group(1))
            t = sympy.symbols("t", positive=True)
            d0 = (rb[0] - ra[0], rb[1] - ra[1])
            roots = sympy.solve(sympy.Eq((d0[0] + vb[0] * t) ** 2 + (d0[1] + vb[1] * t) ** 2, speed**2 * t**2), t)
            assert len(roots) == 1, (seed, text, roots)  # exactly one positive root
            T = roots[0]
            if "After how many hours" in text:
                assert sympy.Integer(int(q.answer)) == T, (seed, text)
                seen.add("intercept-time")
            else:
                comp = 0 if "i-component" in text else 1
                got = sympy.Rational(rb[comp] + vb[comp] * T - ra[comp]) / T
                assert sympy.Integer(int(q.answer)) == got, (seed, text)
                assert (
                    int(q.answer) ** 2
                    + (  # the speed is respected
                        (rb[1 - comp] + vb[1 - comp] * T - ra[1 - comp]) / T
                    )
                    ** 2
                    == speed**2
                )
                seen.add("intercept-component")
        else:
            tstar, dmin, _, _ = _closest(text)
            if "After how many hours" in text:
                assert sympy.Integer(int(q.answer)) == tstar, (seed, text)
                seen.add("approach-time")
            else:
                assert sympy.Integer(int(q.answer)) == dmin, (seed, text)
                seen.add("approach-distance")
    assert seen == {"intercept-time", "intercept-component", "approach-time", "approach-distance"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 2, 2], ["number"] * 4)


def test_exam_keys_by_independent_calculus():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        tstar, dmin, d2, t = _closest(q.stem)
        a, b, c, d = q.parts
        va, vb = _vectors(q.stem)
        assert int(a.answer) ** 2 == (va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2
        assert sympy.Integer(int(b.answer)) == tstar
        assert sympy.Integer(int(c.answer)) == dmin
        rng_km = int(re.search(r"within (\d+) km", d.prompt).group(1))
        first = min(sympy.solve(sympy.Eq(d2, rng_km**2), t))
        assert sympy.Integer(int(d.answer)) == first, (seed, q.stem)
        assert first > 0
        # the least distance is smaller than the range, so the ships really do come within it
        assert dmin < rng_km


def test_exam_follow_through_from_a_wrong_closest_time():
    """A wrong time in (b) still earns (c)'s marks when (c) is the distance at that wrong time (chosen to be exact)."""
    checked = 0
    for seed in range(120):
        q = sup.exam(EXAM, seed)
        ra, rb = _points(q.stem)
        va, vb = _vectors(q.stem)
        r0 = (ra[0] - rb[0], ra[1] - rb[1])
        v = (va[0] - vb[0], va[1] - vb[1])
        for wrong_t in range(1, 15):
            sq = (r0[0] + v[0] * wrong_t) ** 2 + (r0[1] + v[1] * wrong_t) ** 2
            if wrong_t != int(q.parts[1].answer) and math.isqrt(sq) ** 2 == sq:
                canon = {p.label: p.answer_text for p in q.parts}
                reply = {**canon, "b": str(wrong_t), "c": str(math.isqrt(sq))}
                result = sup.grade(SKILL, q, reply)
                assert result.parts["c"].marks == q.parts[2].marks, (seed, wrong_t, result.parts["c"])
                assert result.parts["c"].follow_through is True
                checked += 1
                break
    assert checked >= 20


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier, 120)
        assert len(spread) >= 12, (tier, spread)
        assert max(spread.values()) <= 0.2 * 120, (tier, spread.most_common(3))
