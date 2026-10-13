"""Tests `ucv1-applied-optimisation` (topics `fy1ca_applied_optimisation` and `fy1ca_applied_optimisation_exam`).

The oracle reads the numbers back out of the question TEXT and finds each optimum with SymPy: the stationary points
of the one-variable function are solved exactly and compared with the ends of the allowed interval.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy import Rational, sqrt

from mathgen import make
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_applied_optimisation"
EXAM = "fy1ca_applied_optimisation_exam"
SKILL = load_skill("ucv1-applied-optimisation")
x = sympy.Symbol("x", positive=True)


def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def _best(expr, lo, hi, maximise: bool):
    """The greatest (or least) value of `expr` on the closed interval [lo, hi], by comparing every candidate."""
    candidates = [lo, hi] + [r for r in sympy.solve(sympy.diff(expr, x), x) if r.is_real and lo <= r <= hi]
    values = [sympy.simplify(expr.subs(x, c)) for c in candidates]
    return max(values) if maximise else min(values)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_keys_are_the_greatest_fenced_area():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        nums = _numbers(q.text)
        p = nums[0]
        if "internal fence" in q.text or "including" in q.text:
            k = nums[1]
            sides = k + 2
            width = sympy.Symbol("w", positive=True)
            area = width * (p - sides * width) / 2
            expected = _best(area.subs(width, x), 0, Rational(p, sides), True)
            seen["dividers"] += 1
        else:
            expected = _best(x * (p - 2 * x), 0, Rational(p, 2), True)
            seen["creek"] += 1
        assert Rational(q.answer_text) == expected, (seed, q.text, expected)
    assert seen["creek"] >= 20 and seen["dividers"] >= 20, seen


def test_medium_keys_are_the_best_box_tank_or_window():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        nums = _numbers(q.text)
        if "square sheet" in q.text or "square of cardboard" in q.text or "tray" in q.text:
            a = nums[0]
            expected = _best(x * (a - 2 * x) ** 2, 0, Rational(a, 2), True)
            seen["box"] += 1
        elif "tank" in q.text:
            v = nums[0]
            expected = _best(x**2 + 4 * v / x, Rational(1, 10), 4 * v, False)
            seen["tank"] += 1
        else:
            b, h = nums[:2]
            expected = _best(x * h * (1 - x / b), 0, b, True)
            seen["gable"] += 1
        assert Rational(q.answer_text) == expected, (seed, q.text, expected)
    assert min(seen.values()) >= 15 and len(seen) == 3, seen


def test_hard_keys_compare_the_stationary_point_with_the_ends():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        nums = _numbers(q.text)
        if "easement" in q.text or "cannot exceed" in q.text:
            p, cap = nums[:2]
            expected = _best(x * (p - 2 * x), 0, cap, True)
            seen["paddock"] += 1
        else:
            a, d, c1, c2 = nums[:4]
            expected = _best(c1 * sqrt(a**2 + x**2) + c2 * (d - x), 0, d, False)
            stationary = Rational(c2 * a) / sqrt(c1**2 - c2**2)
            seen["cable-end" if stationary > d else "cable-inside"] += 1
        assert Rational(q.answer_text) == expected, (seed, q.text, expected)
    assert min(seen.values()) >= 10 and len(seen) == 3, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 5, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [1, 2, 3, 2], ["number"] * 4)


def test_exam_keys_match_independent_derivation():
    sides = Counter()
    for seed in range(120):
        q = _exam(seed)
        v = Rational(re.search(r"(?:hold|capacity of) (\d+) m³", q.stem).group(1))
        price = int(re.search(r"\$(\d+) per square metre", q.stem).group(1))
        trial = int(re.search(r"when x = (\d+)", q.parts[0].prompt).group(1))
        height = v / trial**2
        assert Rational(q.parts[0].answer_text) == height
        assert Rational(q.parts[1].answer_text) == trial**2 + 4 * trial * height
        surface = x**2 + 4 * v / x
        best_x = next(r for r in sympy.solve(sympy.diff(surface, x), x) if r.is_real)
        assert Rational(q.parts[2].answer_text) == best_x
        assert Rational(q.parts[3].answer_text) == price * surface.subs(x, best_x)
        sides[trial == best_x] += 1
    assert sides[False] == 120


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"


def test_working_nouns_agree_with_the_stem():
    """The worked steps never name a creek when the stem has a shed wall, or a cable when the stem has a pipe."""
    for tier in TIERS:
        for seed in range(60):
            q = _q(tier, seed)
            steps = " ".join(q.working).lower()
            stem = q.text.lower()
            if "creek" not in stem:
                assert "creek" not in steps, (tier, seed)
            if "cable" not in stem:
                assert "cable" not in steps, (tier, seed)
