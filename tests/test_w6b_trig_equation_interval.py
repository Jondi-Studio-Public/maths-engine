"""Tests `trig-equations-solutions-in-radians` (topic `w6b_trig_equation_interval`) and its exam chain.

The oracle parses the equation and the interval out of the question TEXT and finds the roots by scanning the function
for sign changes, which shares nothing with the generator's two-family construction.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests.w6b_support import TIERS, check_exam_generic, check_follow_through, check_tier, check_variety, check_wiring
from tests.w6b_support import q_of, skill

TOPIC = "w6b_trig_equation_interval"
EXAM = "exd_x_w6b_trig_equation_interval"
SK = skill("trig-equations-solutions-in-radians")
_C = {"√3/2": math.sqrt(3) / 2, "√2/2": math.sqrt(2) / 2, "√3/3": math.sqrt(3) / 3, "√3": math.sqrt(3), "1/2": 0.5}
_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}


def pi_value(text):
    """'5π/6' / 'π' / '2π' / '0' as a float."""
    text = text.strip()
    if text == "0":
        return 0.0
    m = re.fullmatch(r"(-?)(\d*)π(?:/(\d+))?", text)
    assert m, text
    num = int(m.group(2) or 1)
    den = int(m.group(3) or 1)
    return (-1 if m.group(1) else 1) * num / den * math.pi


def c_value(text):
    neg = text.startswith("-")
    t = text.lstrip("-")
    return (-1 if neg else 1) * (_C[t] if t in _C else float(t))


def _balanced(text, start):
    """The text between the bracket that opens at ``start`` and its partner."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
    raise AssertionError(text)


def parse(text):
    """(fn, a, b, c, lo, hi) from the sentence that states the equation."""
    m = re.search(r"(sin|cos|tan)\(", text)
    fn = m.group(1)
    inner, end = _balanced(text, m.end() - 1)
    after = re.match(r" = (-?[\d√/.]+)", text[end + 1 :])
    c = after.group(1) if after else re.findall(r"= (-?[\d√/.]+)", text)[-1]
    a, b = 1.0, 0.0
    if inner == "x":
        pass
    elif inner == "x/2":
        a = 0.5
    elif re.fullmatch(r"\d+x", inner):
        a = float(inner[:-1])
    elif re.fullmatch(r"x - [^()]+", inner):
        b = pi_value(inner[4:])
    elif re.fullmatch(r"\d+\(x - .+\)", inner):
        a = float(re.match(r"(\d+)", inner).group(1))
        b = pi_value(re.search(r"x - (.+)\)", inner).group(1))
    elif re.fullmatch(r"\(x - .+\)/2", inner):
        a = 0.5
        b = pi_value(re.search(r"x - (.+)\)", inner).group(1))
    else:  # pragma: no cover
        raise AssertionError(inner)
    iv = re.search(r"(-?[\dπ/]+) ≤ x ≤ (-?[\dπ/]+)", text)
    return fn, a, b, c_value(c.rstrip(".")), pi_value(iv.group(1)), pi_value(iv.group(2))


def scan(fn, a, b, c, lo, hi, n=40000):
    f = _FN[fn]
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    g = [f(a * (x - b)) - c for x in xs]
    roots = []
    for i in range(n + 1):
        if abs(g[i]) < 1e-9:
            roots.append(xs[i])
    for i in range(n):
        if g[i] * g[i + 1] < 0 and not (fn == "tan" and max(abs(g[i]), abs(g[i + 1])) > 100):
            roots.append(xs[i] + (xs[i + 1] - xs[i]) * g[i] / (g[i] - g[i + 1]))
    out = []
    for r in sorted(roots):
        if not out or r - out[-1] > 1e-3:
            out.append(r)
    return out


def reply_values(q):
    return [float(Fraction(t.strip())) for t in q.answer_text.split(",")]


def test_wiring():
    check_wiring(
        SK, TOPIC, EXAM, "list", ["senior-solve-equations-involving-trigonometric-functions-using-technology-and"]
    )


@pytest.mark.parametrize("tier", TIERS)
def test_tier_grading_and_distractors(tier):
    check_tier(SK, TOPIC, tier, seeds=100)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_variety(tier):
    check_variety(TOPIC, tier, min_texts=20, min_answers=15)


def test_easy_oracle_two_solutions_in_zero_to_two_pi():
    seen = set()
    for seed in range(150):
        q = q_of(TOPIC, "easy", seed)
        fn, a, b, c, lo, hi = parse(q.text)
        assert (a, b, lo, hi) == (1.0, 0.0, 0.0, 2 * math.pi)
        roots = scan(fn, a, b, c, lo, hi)
        ks = reply_values(q)
        assert len(roots) == len(ks) == 2, (seed, q.text, q.answer_text)
        assert all(abs(r - k * math.pi) < 1e-3 for r, k in zip(roots, sorted(ks))), (seed, q.text, q.answer_text)
        seen.add(fn)
    assert seen == {"sin", "cos", "tan"}


def test_medium_and_hard_exact_oracle():
    shapes = set()
    for tier in ("medium", "hard"):
        for seed in range(200):
            q = q_of(TOPIC, tier, seed)
            if "π" not in q.answer_text and "multiple of π" not in q.text:
                continue
            if "decimal" in q.text:
                continue
            fn, a, b, c, lo, hi = parse(q.text)
            ks = sorted(reply_values(q))
            if fn != "tan" and abs(abs(c) - 1) < 1e-12:
                # a tangent touch has no sign change to scan for: check each listed solution and the spacing
                assert all(abs(_FN[fn](a * (k * math.pi - b)) - c) < 1e-9 for k in ks), (seed, q.text)
                assert all(abs((k2 - k1) * math.pi * a - 2 * math.pi) < 1e-9 for k1, k2 in zip(ks, ks[1:]))
                continue
            roots = scan(fn, a, b, c, lo, hi)
            assert len(roots) == len(ks), (tier, seed, q.text, q.answer_text, len(roots))
            assert all(abs(r - k * math.pi) < 1e-3 for r, k in zip(roots, ks)), (tier, seed, q.text, q.answer_text)
            shapes.add((tier, a != 1, b != 0))
    assert ("medium", True, False) in shapes and ("medium", False, True) in shapes and ("hard", True, True) in shapes


def test_hard_decimal_oracle():
    found = 0
    for seed in range(300):
        q = q_of(TOPIC, "hard", seed)
        if "decimal" not in q.text:
            continue
        fn, a, b, c, lo, hi = parse(q.text)
        roots = scan(fn, a, b, c, lo, hi)
        vals = [float(t) for t in q.answer_text.split(",")]
        assert len(roots) == len(vals), (seed, q.text, q.answer_text)
        assert all(abs(r - v) <= 0.005 + 1e-9 for r, v in zip(roots, vals)), (seed, q.text, q.answer_text)
        found += 1
    assert found >= 40


def test_grader_accepts_unordered_and_longer_decimals_but_not_degrees():
    for seed in range(300):
        q = q_of(TOPIC, "hard", seed)
        if "decimal" not in q.text or len(q.answer) < 2:
            continue
        reversed_reply = ", ".join(reversed(q.answer_text.split(", ")))
        assert grade(SK, q, reversed_reply).correct
        assert not grade(SK, q, q.answer_text.split(", ")[0]).correct
        return
    pytest.fail("no decimal question found")


def test_exam_shape_and_marks():
    check_exam_generic(SK, EXAM, n=80, marks=[1, 1, 2, 3], kinds=["number", "number", "fraction", "list"])
    q = make(EXAM, seed=0, difficulty="exam")
    assert [p.follow_through for p in q.parts] == [None, "2*part_a", None, None]


def test_exam_oracle():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        fn, a, b, c, lo, hi = parse(q.stem)
        roots = scan(fn, a, b, c, lo, hi)
        assert q.parts[0].answer == round(a) == int(q.parts[0].answer_text)
        assert "complete periods" in q.parts[0].prompt
        assert q.parts[1].answer == len(roots) == 2 * round(a)
        assert abs(float(Fraction(q.parts[2].answer_text)) * math.pi - roots[0]) < 1e-3
        listed = sorted(float(Fraction(t.strip())) for t in q.parts[3].answer_text.split(","))
        assert len(listed) == len(roots)
        assert all(abs(r - k * math.pi) < 1e-3 for r, k in zip(roots, listed))


def test_exam_follow_through_from_a_wrong_number_of_periods():
    def wrong(q):
        n = q.parts[0].answer
        return [str(r) for r in (1, n + 1, n - 1) if r != n]

    def follow(q, reply):
        return str(2 * int(reply))

    assert check_follow_through(SK, EXAM, "a", wrong, "b", follow, n=40) >= 20
