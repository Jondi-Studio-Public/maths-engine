"""Tests `projectile-cartesian-elimination`: oracles use y = x tan(a) - g x² (1 + tan²(a)) / (2 V²) from the question text."""

from __future__ import annotations

import math
import re

import pytest
import sympy
from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    assert_states,
    frac,
    questions,
    skill,
    sources,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("projectile-cartesian-elimination")
SEEDS = range(60)
G = 9.8
x = sympy.Symbol("x")


def f(text: str) -> float:
    return float(frac(text))


def launch_speed(text: str) -> float:
    return f(re.search(r"speed (?:of )?([\d.]+) m/s", text).group(1))


def launch_tangent(text: str) -> float:
    if m := re.search(r"tangent is (\d+)/(\d+)", text):
        return int(m.group(1)) / int(m.group(2))
    return math.tan(math.radians(f(re.search(r"at (\d+)° above", text).group(1))))


def path_coefficients(text: str):
    """y = P x - Q x² for a launch from ground level."""
    tan = launch_tangent(text)
    return tan, G * (1 + tan * tan) / (2 * launch_speed(text) ** 2)


def oracle_easy(q):
    s = q.text
    p, big_q = path_coefficients(s)
    if "Find Q" in s:
        return big_q, "coefficient"
    d = f(re.search(r"travelled ([\d.]+) m horizontally", s).group(1))
    return p * d - big_q * d * d, "height"


def test_easy_oracle_and_grading():
    assert_single_answer_rules(SK, "easy", SEEDS)
    kinds = set()
    for q in questions(SK, "easy", range(150)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_easy(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"coefficient", "height"}


def oracle_medium(q):
    s = q.text
    if m := re.search(r"lands ([\d.]+) m away", s):
        v = launch_speed(s)
        return math.degrees(0.5 * math.asin(G * f(m.group(1)) / v**2)), "angle-for-range"
    p, big_q = path_coefficients(s)
    v = launch_speed(s)
    if m := re.search(r"again at a height of ([\d.]+) m", s):
        h = f(m.group(1))
        return (p + math.sqrt(p * p - 4 * big_q * h)) / (2 * big_q), "return-height"
    if "greatest height" in s:
        return p * p / (4 * big_q), "greatest-height"
    if "horizontal range" in s:
        return p / big_q, "range"
    raise AssertionError(s)


def test_medium_oracle_and_grading():
    assert_single_answer_rules(SK, "medium", SEEDS)
    kinds = set()
    for q in questions(SK, "medium", range(200)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_medium(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"angle-for-range", "return-height", "greatest-height", "range"}


def oracle_hard(q):
    s = q.text
    if m := re.search(r"wall ([\d.]+) m high that stands ([\d.]+) m from the kicker", s):
        height, d = f(m.group(1)), f(m.group(2))
        return math.sqrt(G * d * d / (d - height)), "wall"
    v = launch_speed(s)
    d, y = (
        f(g)
        for g in re.search(r"point ([\d.]+) m horizontally from the kicker and ([\d.]+) m above the ground", s).groups()
    )
    c = G * d * d / (2 * v * v)
    # c T² - d T + (y + c) = 0 where T = tan(angle)
    root = math.sqrt(d * d - 4 * c * (y + c))
    lower, higher = (d - root) / (2 * c), (d + root) / (2 * c)
    pick = lower if "smaller one" in s else higher
    return math.degrees(math.atan(pick)), "angle-to-point"


def test_hard_oracle_and_grading():
    assert_single_answer_rules(SK, "hard", SEEDS)
    kinds = set()
    for q in questions(SK, "hard", range(200)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_hard(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"wall", "angle-to-point"}


def test_hard_angles_are_both_asked_for():
    asked = {
        ("smaller" if "smaller one" in q.text else "larger")
        for q in questions(SK, "hard", range(200))
        if "possible angles" in q.text
    }
    assert asked == {"smaller", "larger"}


def exam_oracle(q):
    h = f(re.search(r"(?:platform|cliff) ([\d.]+) m above (?:the|level) ground", q.text).group(1))
    v = f(re.search(r"(?:speed of|hand at) ([\d.]+) m/s", q.text).group(1))
    inv = v * v / G
    exact_inv = sympy.Rational(str(v)) ** 2 / sympy.Rational(98, 10)
    path = sympy.Rational(str(h)) + x - x * x / exact_inv
    top = h + inv / 4
    reach = (1 + math.sqrt(1 + 4 * h / inv)) * inv / 2
    angle = math.degrees(math.atan(abs(1 - 2 * reach / inv)))
    return path, top, reach, angle


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[3, 2, 2, 2])
    topic = sources(SK)["exam"]
    for seed in range(100):
        q = make(topic, seed=seed, difficulty="exam")
        path, top, reach, angle = exam_oracle(q)
        stated = sympy.sympify(q.parts[0].answer.replace("^", "**"), locals={"x": x})
        assert sympy.simplify(stated - path) == 0 or all(
            abs(float(stated.subs(x, s)) - float(path.subs(x, s))) < 1e-9 for s in (0.5, 3, 11, 29)
        ), (q.text, q.parts[0].answer)
        for part, expected in zip(q.parts[1:], (top, reach, angle), strict=True):
            assert_states(part, expected, q.text)


def test_exam_expression_accepts_an_equivalent_form():
    q = make(sources(SK)["exam"], seed=2, difficulty="exam")
    path = exam_oracle(q)[0]
    canon = {p.label: p.answer_text for p in q.parts}
    shuffled = str(sympy.Add(*reversed(path.as_ordered_terms()))).replace("**", "^")
    assert grade(SK, q, {**canon, "a": shuffled}).parts["a"].marks == q.parts[0].marks
    wrong = str(path.subs(x, 2 * x)).replace("**", "^")
    assert grade(SK, q, {**canon, "a": wrong}).parts["a"].marks == 0


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for p in make(sources(SK)["exam"], seed=seed, difficulty="exam").parts:
            assert_number_distractors_match_places(p)


@pytest.mark.parametrize(
    "topic,difficulty,distinct",
    [
        ("r3d_projectile_cartesian", "easy", 20),
        ("r3d_projectile_cartesian", "medium", 20),
        ("r3d_projectile_cartesian", "hard", 20),
        ("r3d_projectile_cartesian_exam", "exam", 10),
    ],
)
def test_deterministic_and_varied(topic, difficulty, distinct):
    assert_deterministic_and_varied(topic, difficulty, distinct=distinct)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "two possible angles" not in easy and "wall" not in easy
    assert "two possible angles" in hard and "wall" in hard
