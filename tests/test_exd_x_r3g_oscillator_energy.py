"""Tests for the multi-part exam source `exd_x_r3g_oscillator_energy` (spring-mass: energy, amplitude, speeds)."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_oscillator_energy"
SKILL = "ucv1-energy-conservation-check"


def _facts(q):
    m = int(re.search(r"(\d+) kg", q.text).group(1))
    k = int(re.search(r"(\d+) N/m", q.text).group(1))
    x1 = Fraction(re.search(r"([\d.]+) m from (?:its |the )?equilibrium", q.text).group(1))
    v1 = Fraction(re.search(r"([\d.]+) m/s", q.text).group(1))
    x2 = Fraction(re.search(r"when it is ([\d.]+) m from equilibrium", q.text).group(1))
    return m, k, x1, v1, x2


def _round(value, places: int) -> str:
    """Half-up rounding of an exact sympy value, written to `places` decimal places."""
    exact = sympy.nsimplify(value)
    scaled = sympy.floor(exact * 10**places + sympy.Rational(1, 2))
    return f"{int(scaled) / 10**places:.{places}f}"


def _derive(q):
    m, k, x1, v1, x2 = _facts(q)
    r = lambda f: sympy.Rational(f.numerator, f.denominator)  # noqa: E731
    energy = sympy.Rational(1, 2) * m * r(v1) ** 2 + sympy.Rational(1, 2) * k * r(x1) ** 2
    amp = sympy.sqrt(2 * energy / k)
    vmax = sympy.sqrt(2 * energy / m)
    v2 = sympy.sqrt((2 * energy - k * r(x2) ** 2) / m)
    return {"a": _round(energy, 2), "b": _round(amp, 1), "c": _round(vmax, 1), "d": _round(v2, 1)}, energy


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p14f_energy_conservation_check", {"variant": "implicit", "mk_hi": 8, "xv_hi": 6})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        want, _ = _derive(q)
        if sup.canonical(q) != want:
            bad.append((q.text, sup.canonical(q), want))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_energy():
    def replies(q):
        m, k, x1, v1, x2 = _facts(q)
        energy = Fraction(q.parts[0].distractors[0])
        # The wrong energy must still admit a speed at x2, or the engine has nothing to follow through
        if 2 * energy - k * x2**2 <= 0:
            return None
        amp = sympy.sqrt(2 * sympy.Rational(energy.numerator, energy.denominator) / k)
        vmax = sympy.sqrt(2 * sympy.Rational(energy.numerator, energy.denominator) / m)
        v2 = sympy.sqrt(
            (
                2 * sympy.Rational(energy.numerator, energy.denominator)
                - k * sympy.Rational(x2.numerator, x2.denominator) ** 2
            )
            / m
        )
        a1, v_max1 = _round(amp, 1), _round(vmax, 1)
        reply = {
            "a": f"{float(energy):.2f}",
            "b": a1,
            "c": _round(sympy.nsimplify(a1) * sympy.sqrt(sympy.Rational(k, m)), 1),
            "d": _round(v2, 1),
        }
        # a reply that repeats an earlier one earns no follow-through, and neither does one that rounds to zero
        if len(set(reply.values())) < 4 or float(reply["b"]) == 0 or float(reply["d"]) == 0:
            return None
        return reply, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)
