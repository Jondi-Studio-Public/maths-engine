"""Tests `ucv1-related-rates` (topics `fy1ca_related_rates` and `fy1ca_related_rates_exam`).

The oracle reads the numbers back out of the question TEXT and re-derives each key with SymPy from the relation the
situation names (circle, sphere, Pythagoras, similar triangles, the cone's volume), never from the generator.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest
import sympy
from sympy import Rational, pi

from mathgen import make
from mathgen.engine import grade
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_related_rates"
EXAM = "fy1ca_related_rates_exam"
SKILL = load_skill("ucv1-related-rates")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.]*\))")
_RATIOS = {
    "half": Rational(1, 2),
    "equal to": 1,
    "one and a half": Rational(3, 2),
    "two-thirds": Rational(2, 3),
    "three-quarters": Rational(3, 4),
}


def _numbers(text: str) -> list[Rational]:
    return [Rational(m.group(0)) for m in _NUMBER.finditer(text)]


def _expected(q):
    return sympy.sympify(q.answer)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression")
    assert "differentiate-chain-rule" in SKILL.prerequisites


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_key_is_the_derivative_of_the_standard_formula():
    seen = Counter()
    t, r0 = sympy.symbols("t r0")
    for seed in range(150):
        q = _q("easy", seed)
        dr, r = _numbers(q.text)[:2]
        radius = r + dr * t
        if "volume" in q.text:
            formula, tag = Rational(4, 3) * pi * radius**3, "sphere"
        elif "surface area" in q.text:
            formula, tag = 4 * pi * radius**2, "surface"
        else:
            formula, tag = pi * radius**2, "circle"
        assert sympy.simplify(sympy.diff(formula, t).subs(t, 0) - _expected(q)) == 0, (seed, q.text)
        seen[tag] += 1
    assert min(seen.values()) >= 15 and len(seen) == 3, seen


def test_medium_keys_follow_the_geometry():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        nums = _numbers(q.text)
        if "ladder" in q.text:
            length, v, x = nums[:3]
            y = sympy.sqrt(length**2 - x**2)
            rate = -x * v / y
            expected = -rate if "speed" in q.text else rate
            seen["ladder"] += 1
        elif "lamp" in q.text:
            lamp, person, v = nums[:3]
            tip = lamp * v / (lamp - person)
            length_rate = person * v / (lamp - person)
            expected = tip if "tip" in q.text else length_rate
            seen["shadow"] += 1
        else:
            x, u, y, w = nums[:4]
            s = sympy.sqrt(x**2 + y**2)
            expected = (x * u - y * w) / s  # the second vehicle is moving towards the junction
            seen["vehicles"] += 1
        assert sympy.simplify(expected - _expected(q)) == 0, (seed, q.text, expected, q.answer)
    assert min(seen.values()) >= 30 and len(seen) == 3, seen


def test_hard_keys_follow_the_eliminated_variable():
    seen = Counter()
    h = sympy.Symbol("h", positive=True)
    for seed in range(200):
        q = _q("hard", seed)
        nums = _numbers(q.text)
        if "hailstone" in q.text:
            vol_rate, r0 = nums[:2]
            r = sympy.Symbol("r", positive=True)
            drdt = -vol_rate / sympy.diff(Rational(4, 3) * pi * r**3, r).subs(r, r0)
            expected = sympy.diff(4 * pi * r**2, r).subs(r, r0) * drdt
            if "decreasing, in" in q.text and "negative if" not in q.text:
                expected = -expected
            seen["hailstone"] += 1
        elif "tank" in q.text:
            top, depth, rate, h0 = nums[:4]
            ratio = (top / 2) / depth
            volume = pi * (ratio * h) ** 2 * h / 3
            dhdt = -rate / sympy.diff(volume, h).subs(h, h0)
            expected = -dhdt if "falling, in" in q.text and "negative if" not in q.text else dhdt
            seen["tank"] += 1
        else:
            rate, h0 = nums[:2]
            ratio = next(v for k, v in sorted(_RATIOS.items(), key=lambda kv: -len(kv[0])) if k in q.text)
            volume = pi * (ratio * h) ** 2 * h / 3
            expected = rate / sympy.diff(volume, h).subs(h, h0)
            seen["cone"] += 1
        assert sympy.simplify(expected - _expected(q)) == 0, (seed, q.text, expected, q.answer)
    assert min(seen.values()) >= 25 and len(seen) == 3, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(100)]
        assert len(set(answers)) >= 20, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [2, 2, 2, 2], ["number"] * 4)


def test_exam_keys_match_independent_derivation():
    seen = Counter()
    for seed in range(120):
        q = _exam(seed)
        rate = float(re.search(r"(\d+) m³/min", q.stem).group(1))
        h0 = float(re.search(r"(?:height|radius) is (\d+) m", q.parts[0].prompt).group(1))
        if "cone" in q.stem or "conical" in q.stem:
            ratio = float(next(v for k, v in sorted(_RATIOS.items(), key=lambda kv: -len(kv[0])) if k in q.stem))
            coef = math.pi * ratio**2 / 3
            seen["cone"] += 1
        else:
            coef = 4 * math.pi / 3
            seen["sphere"] += 1
        volume, slope = coef * h0**3, 3 * coef * h0**2
        assert float(q.parts[0].answer_text) == pytest.approx(volume, abs=0.0051)
        assert float(q.parts[1].answer_text) == pytest.approx(slope, abs=0.0051)
        assert float(q.parts[2].answer_text) == pytest.approx(rate / slope, abs=0.00051)
        # the time to double the size, from an independent numerical integral of dV = 3 coef h^2 dh
        n = 2000
        added = sum(3 * coef * (h0 + (i + 0.5) * h0 / n) ** 2 for i in range(n)) * h0 / n
        assert float(q.parts[3].answer_text) == pytest.approx(added / rate, abs=0.051)
    assert min(seen.values()) >= 25, seen


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_exam_rate_follows_through_from_a_wrong_derivative():
    credited = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        rate = float(re.search(r"(\d+) m³/min", q.stem).group(1))
        wrong_b = float(canon["b"]) * 2
        reply = f"{rate / wrong_b:.3f}"
        result = grade(SKILL, q, {**canon, "b": f"{wrong_b:.2f}", "c": reply})
        assert result.parts["b"].marks == 0
        credited += result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
    assert credited >= 45


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
