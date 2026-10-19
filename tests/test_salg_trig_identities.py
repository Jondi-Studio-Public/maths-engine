"""Tests `reciprocal-trig-identities` (topic `salg_trig_identities`) and its exam chain `salg_exam_trig_identities`.
The oracle reads the given ratio and quadrant back out of the question text, builds the angle with `atan2` and
evaluates every key in floats; the simplification and the equation are evaluated at the angle or points directly."""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.salg_trig_identities_build  # noqa: F401  (registers the topics)
from mathgen import make
from salg_support import check_exam, check_follow_through, check_single, parse, skill

TOPIC = "salg_trig_identities"
EXAM = "salg_exam_trig_identities"
SKILL = skill("reciprocal-trig-identities")
TIERS = ("easy", "medium", "hard")
_QUADRANT = {"0° < θ < 90°": 1, "90° < θ < 180°": 2, "180° < θ < 270°": 3, "270° < θ < 360°": 4}


def _frac(text):
    return float(Fraction(text))


def _theta(given, value, quadrant):
    """The angle in the quadrant whose `given` function (tan, cot, sec, cosec) has `value`, from the standard formulas."""
    ratio = {"tan": value, "cot": 1 / value, "sec": None, "cosec": None}[given]
    if ratio is not None:
        base = math.atan(abs(ratio))
    elif given == "sec":
        base = math.acos(1 / abs(value))
    else:
        base = math.asin(1 / abs(value))
    return {1: base, 2: math.pi - base, 3: math.pi + base, 4: 2 * math.pi - base}[quadrant]


def _fn(name, theta):
    return {
        "sec": 1 / math.cos(theta),
        "cosec": 1 / math.sin(theta),
        "tan": math.tan(theta),
        "cot": 1 / math.tan(theta),
        "sin": math.sin(theta),
        "cos": math.cos(theta),
    }[name]


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "expression"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]


def test_easy_key_matches_float_evaluation():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        given, value, wanted = re.search(r"(tan|cot) θ = (\d+/\d+).*?(sec|cosec) θ", q.text, re.S).groups() and (
            re.search(r"(tan|cot) θ = (\d+/\d+)", q.text).group(1),
            _frac(re.search(r"(tan|cot) θ = (\d+/\d+)", q.text).group(2)),
            re.search(r"(sec|cosec) θ", q.text).group(1),
        )
        assert (given, wanted) in (("tan", "sec"), ("cot", "cosec")), q.text
        theta = _theta(given, value, 1)
        assert abs(_frac(q.answer_text) - _fn(wanted, theta)) < 1e-9, (seed, q.text)
        assert q.answer_kind == "expression"
        seen[given] += 1
    assert set(seen) == {"tan", "cot"}


def test_medium_key_matches_float_evaluation():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        m = re.search(r"(sec|cosec) θ = (-?\d+/\d+) and (\d+° < θ < \d+°).*?(tan|cot) θ", q.text)
        given, value, quad, wanted = m.group(1), _frac(m.group(2)), _QUADRANT[m.group(3)], m.group(4)
        assert quad != 1
        theta = _theta(given, value, quad)
        assert abs(_fn(given, theta) - value) < 1e-9
        assert abs(_frac(q.answer_text) - _fn(wanted, theta)) < 1e-9, (seed, q.text)
        seen[(given, quad)] += 1
    assert len(seen) >= 5


def test_hard_keys():
    kinds = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        if q.text.startswith(("Simplify", "Write")):
            expr = re.search(r"(?:Simplify|Write) (.*?)(?: as a single term)?\.$", q.text).group(1)
            python = expr
            for pattern, repl in (
                (r"cosec²x", "(1/sin(x))**2"),
                (r"(?<![a-z])sec²x", "(1/cos(x))**2"),
                (r"(?<![a-z])cos²x", "cos(x)**2"),
                (r"(?<![a-z])sin²x", "sin(x)**2"),
                (r"(?<![a-z])tan²x", "tan(x)**2"),
                (r"cot²x", "(1/tan(x))**2"),
                (r"(?<![a-z])sec x", "(1/cos(x))"),
                (r"(?<![a-z])tan x", "tan(x)"),
                (r"(?<![a-z])cos x", "cos(x)"),
                (r"(?<![a-z])sin x", "sin(x)"),
                (r"(\d)(?=[a-z(])", r"\1*"),
                (r"\)(?=[a-z(])", ")*"),
            ):
                python = re.sub(pattern, repl, python)
            env = {"cos": math.cos, "sin": math.sin, "tan": math.tan}
            for x in (0.4, 0.9, 1.2):
                want = eval(python, {"__builtins__": {}}, {**env, "x": x})  # noqa: S307 - the generator's own stem
                got = float(parse(q.answer_text).subs("x", x))
                assert abs(want - got) < 1e-9, (seed, q.text, q.answer_text)
            kinds["simplify"] += 1
        else:
            m = re.search(r"(\d*)(sec|cosec)²θ - (\d*)(tan|cot)²θ = (\d+)", q.text)
            a, b, c = int(m.group(1) or 1), int(m.group(3) or 1), int(m.group(5))
            fa, fb = m.group(2), m.group(4)
            angle = math.radians(float(q.answer))
            sec_sq = _fn(fa, angle) ** 2
            tan_sq = _fn(fb, angle) ** 2
            assert abs(a * sec_sq - b * tan_sq - c) < 1e-9, (seed, q.text, q.answer_text)
            assert q.answer_kind == "number"
            # the only acute solution: the other standard angles do not satisfy it
            for other in (30, 45, 60):
                if other != int(q.answer):
                    r = math.radians(other)
                    assert abs(a * _fn(fa, r) ** 2 - b * _fn(fb, r) ** 2 - c) > 1e-6, (seed, q.text, other)
            kinds["solve"] += 1
    assert kinds["simplify"] >= 40 and kinds["solve"] >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=100)


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(60)]
    assert all("acute" in t for t in easy)
    medium = [_q("medium", s).text for s in range(60)]
    assert all("acute" not in t and "°" in t for t in medium)
    assert any("-" in re.search(r"(?:sec|cosec) θ = (\S+)", t).group(1) for t in medium)
    hard = [_q("hard", s).text for s in range(60)]
    assert any(t.startswith(("Simplify", "Write")) for t in hard) and any("acute" in t for t in hard)
    assert not any(re.search(r"θ = \d+/\d+", t) for t in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(150)]
    assert len(set(answers)) >= (8 if tier == "easy" else 5), Counter(answers)
    assert len({_q(tier, s).text for s in range(150)}) >= 30


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert q.parts[0].sensible_range == (1, None)


def test_exam_keys_match_independent_derivation():
    quads = set()
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        m = re.search(r"tan θ = (-?\d+/\d+) and (\d+° < θ < \d+°)", q.text)
        quad = _QUADRANT[m.group(2)]
        quads.add(quad)
        theta = _theta("tan", _frac(m.group(1)), quad)
        assert abs(math.tan(theta) - _frac(m.group(1))) < 1e-9
        want = [_fn("sec", theta) ** 2, _fn("sec", theta), _fn("cos", theta), _fn("sin", theta)]
        for part, w in zip(q.parts, want):
            assert abs(_frac(part.answer_text) - w) < 1e-9, (seed, part.label, q.text)
    assert quads == {1, 2, 3, 4}


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_follow_through_from_a_wrong_sec_squared():
    def wrong(q):
        tan = Fraction(re.search(r"tan θ = (-?\d+/\d+)", q.text).group(1))
        root = 1 + abs(tan)  # sec read as 1 + tan, so sec^2 is (1 + |tan|)^2
        bad = root * root
        sign = -1 if q.parts[1].answer["p"] < 0 else 1
        sec = sign * root
        cos = 1 / sec
        return str(bad), {"b": str(sec), "c": str(cos), "d": str(tan * cos)}

    check_follow_through(SKILL, EXAM, "a", wrong, ["b", "c", "d"])
