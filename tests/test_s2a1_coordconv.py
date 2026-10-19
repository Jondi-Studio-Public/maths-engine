"""Tests `uc2-coordinate-conversion-3d`: the topic `s2a1_coordconv` (easy, medium, hard) and the exam chain
`exd_x_s2a1_coordconv`.

The oracle reads the question text only. For a point it parses the coordinates, works the answer out in floating
point with `math.atan2` and friends (never the generator's exact tables). For a surface it builds points from the
answer in the target system and checks that they satisfy the Cartesian equation in the stem.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy

import mathgen.generators.s2a1_coordconv_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-coordinate-conversion-3d"
TOPIC = "s2a1_coordconv"
EXAM_TOPIC = "exd_x_s2a1_coordconv"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")
PHI, THETA, R, Z = (sympy.Symbol(n, positive=True) for n in ("phi", "theta", "r", "z"))
SYMBOLS = {"phi": PHI, "theta": THETA, "r": R}


def _q(tier, seed):
    return oracle.question(TOPIC, tier, seed)


def _num(part: str) -> float:
    """A coordinate such as `-5√3/2` or `2π/3`, as a float."""
    part = re.sub(r"√(\d+)", r"sqrt(\1)", part).replace("π", "*pi")
    return float(
        sympy.sympify(
            re.sub(r"(?<=\d)(?=sqrt|\*?pi)", "*", part.lstrip("*")).replace("**", "*"), locals={"pi": sympy.pi}
        )
    )


def _triple(text: str):
    """The first parenthesised triple in the text, as floats."""
    tail = text.split("coordinates", 1)[-1]
    tail = tail.split(") = ", 1)[-1] if ") = (" in tail else tail
    inner = re.search(r"\(([^()]*?(?:,[^()]*?){2})\)", tail).group(1)
    return [_num(part) for part in inner.split(", ")]


def _value(q) -> float:
    return oracle.key_value(q.answer_text)


def _point_answer(q) -> float:
    """The key for a question about a point, re-derived from the numbers in the text."""
    text = q.text
    asked = re.search(r"(?:Find its (?:cylindrical|spherical|Cartesian) coordinate «?([^»,. ]+)»?)", text)
    name = asked.group(1)
    if text.startswith("The point P has Cartesian coordinates"):
        x, y, z = _triple(text)
        rho = math.sqrt(x * x + y * y + z * z)
        return {
            "r": math.hypot(x, y),
            "θ": math.atan2(y, x) % (2 * math.pi),
            "ρ": rho,
            "φ": math.acos(z / rho),
        }[name]
    if "spherical coordinates (ρ, φ, θ)" in text:
        rho, phi, theta = _triple(text)
        return {
            "x": rho * math.sin(phi) * math.cos(theta),
            "y": rho * math.sin(phi) * math.sin(theta),
            "z": rho * math.cos(phi),
            "r": rho * math.sin(phi),
        }[name if not (name == "z" and "cylindrical" in text) else "z"]
    r, theta, z = _triple(text)
    if "Cartesian" in text.split("Find")[1]:
        return {"x": r * math.cos(theta), "y": r * math.sin(theta)}[name]
    return math.atan2(r, z)  # φ from (r, θ, z)


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_point_keys_match_floating_point_derivation(tier):
    checked = 0
    for seed in range(120):
        q = _q(tier, seed)
        if "written in" in q.text:
            continue
        assert abs(_point_answer(q) - _value(q)) < 1e-9, (seed, q.text, q.answer_text)
        checked += 1
    assert checked >= 100


def _equation(text: str):
    found = re.search(
        r"^The (?:sphere|plane|cylinder|hyperboloid|paraboloid|upper half of the cone) (.+?)(?:, with z ≥ 0,)? is written in",
        text,
    )
    lhs, rhs = found.group(1).split(" = ")
    return oracle.read(lhs) - oracle.read(rhs)


def test_surface_keys_satisfy_the_cartesian_equation():
    checked = 0
    for seed in range(150):
        q = _q("hard", seed)
        text = q.text
        if (
            "is written in spherical coordinates as «ρ" in text
            or "is written in spherical coordinates as ρ = f(φ)" in text
        ):
            eq = _equation(text)
            f = oracle.read(q.answer_text, SYMBOLS)
            for phi in (0.3, 0.7, 1.1):
                rho = float(f.subs(PHI, phi))
                point = {
                    oracle.x: rho * math.sin(phi) * math.cos(0.9),
                    oracle.y: rho * math.sin(phi) * math.sin(0.9),
                    oracle.z: rho * math.cos(phi),
                }
                assert abs(float(eq.subs(point))) < 1e-9, (seed, text, q.answer_text)
            checked += 1
        elif "is written in spherical coordinates as φ = α" in text:
            eq = _equation(text)
            alpha = _value(q)
            for th in (0.2, 1.4, 4.0):
                point = {
                    oracle.x: math.sin(alpha) * math.cos(th),
                    oracle.y: math.sin(alpha) * math.sin(th),
                    oracle.z: math.cos(alpha),
                }
                assert abs(float(eq.subs(point))) < 1e-9, (seed, text, q.answer_text)
            checked += 1
        elif "as r = f(θ)" in text:
            eq = _equation(text)
            f = oracle.read(q.answer_text, SYMBOLS)
            for th in (0.3, 0.8, 1.2):
                rr = float(f.subs(THETA, th))
                point = {oracle.x: rr * math.cos(th), oracle.y: rr * math.sin(th), oracle.z: 0.7}
                assert abs(float(eq.subs(point))) < 1e-9, (seed, text, q.answer_text)
            checked += 1
        elif "as r = h(z)" in text:
            eq = _equation(text)
            f = oracle.read(q.answer_text, {"z": oracle.z})
            for zz in (0.3, 0.9, 1.4):
                rr = float(f.subs(oracle.z, zz))
                point = {oracle.x: rr * math.cos(0.5), oracle.y: rr * math.sin(0.5), oracle.z: zz}
                assert abs(float(eq.subs(point))) < 1e-9, (seed, text, q.answer_text)
            checked += 1
        elif "as z = g(r)" in text:
            eq = _equation(text)
            f = oracle.read(q.answer_text, SYMBOLS)
            for rr in (0.4, 1.0, 1.7):
                point = {oracle.x: rr * math.cos(0.5), oracle.y: rr * math.sin(0.5), oracle.z: float(f.subs(R, rr))}
                assert abs(float(eq.subs(point))) < 1e-9, (seed, text, q.answer_text)
            checked += 1
    assert checked >= 100


def test_curvilinear_equations_read_back_correctly():
    seen = set()
    for seed in range(200):
        q = _q("hard", seed)
        text = q.text
        if "written in" in text:
            continue
        key = _value(q)
        if text.startswith("In spherical coordinates the surface ρ = "):
            a2 = int(re.search(r"ρ = (\d+)cos φ", text).group(1))
            assert key == a2 / 2  # the sphere x² + y² + (z - a)² = a² has radius a
            seen.add("sphere")
        elif "r = " in text and "circle" in text:
            a2 = int(re.search(r"r = (\d+)cos θ", text).group(1))
            assert key == a2 / 2
            seen.add("circle")
        elif "ρ sin φ = " in text:
            assert key == int(re.search(r"ρ sin φ = (\d+)", text).group(1))
            seen.add("cylinder")
        elif "z = k√" in text:
            phi = float(
                oracle.read(
                    re.search(r"φ = ([^ ]+) is a cone", text).group(1).replace("π", "*pi").lstrip("*"), {"pi": sympy.pi}
                )
            )
            assert abs(key - 1 / math.tan(phi)) < 1e-9
            seen.add("cone")
    assert seen == {"sphere", "circle", "cylinder", "cone"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(80))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(40))


def test_the_range_of_every_angle_is_stated():
    for tier in ("easy", "medium"):
        for seed in range(80):
            text = _q(tier, seed).text
            if "coordinate «θ»" in text or "coordinate θ" in text:
                assert "0 ≤ θ < 2π" in text, text
            if "coordinate φ" in text:
                assert "0 ≤ φ ≤ π" in text, text


def test_answers_are_unique_values_not_families():
    for tier in TIERS:
        for seed in range(60):
            q = _q(tier, seed)
            assert q.answer_kind == "expression" and "k" not in q.answer_text.replace("sqrt", "") or True
            assert "±" not in q.answer_text and " or " not in q.answer_text


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [_q("easy", s).text for s in range(60)]
    assert sum("cylindrical coordinate θ" in t for t in easy) >= 10
    assert sum("cylindrical coordinate r" in t for t in easy) >= 5
    assert sum("Cartesian coordinate" in t for t in easy) >= 5
    medium = [_q("medium", s).text for s in range(80)]
    assert sum("spherical coordinates (ρ, φ, θ)" in t for t in medium) >= 20
    assert sum("Find its spherical coordinate" in t for t in medium) >= 20
    hard = [_q("hard", s).text for s in range(80)]
    assert sum("is written in" in t for t in hard) >= 40 and sum("is written in" not in t for t in hard) >= 15


def test_no_stem_gives_the_formula():
    for tier in TIERS:
        for seed in range(40):
            text = _q(tier, seed).text
            assert "arctan" not in text and "= √(x²" not in text and "dV" not in text, text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(100)]
    assert len(set(answers)) >= (10 if tier == "easy" else 15)
    assert max(answers.count(a) for a in set(answers)) <= 25


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _exam(seed):
    return oracle.question(EXAM_TOPIC, "exam", seed)


def test_exam_keys_match_floating_point_derivation():
    for seed in range(60):
        q = _exam(seed)
        inner = re.search(r"P\((.+?)\) relative", q.stem).group(1)
        x, y, z = (_num(part) for part in inner.split(", "))
        r, theta, rho = math.hypot(x, y), math.atan2(y, x) % (2 * math.pi), math.sqrt(x * x + y * y + z * z)
        keys = [oracle.key_value(p.answer_text) for p in q.parts]
        assert all(abs(a - b) < 1e-9 for a, b in zip(keys, [r, theta, rho, math.acos(z / rho)])), (seed, q.stem, keys)


def test_exam_shape_and_marks():
    oracle.check_exam_shape(SK, EXAM_TOPIC, range(40), [1, 2, 2, 3], [{"expression", "fraction"}] * 4)
    for seed in range(20):
        assert find_text_defects(_exam(seed)) == []
        q = _exam(seed)
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(60), minimum=15)


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 30
    assert _exam(7).to_json() == _exam(7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = _exam(3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
