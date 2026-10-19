"""Tests `sinusoid-features-from-equation` (topic `w6b_sinusoid_features`) and its exam chain.

The oracle reads the equation (or both equations) out of the question TEXT, evaluates the function numerically and
works the asked feature out by scanning it: extremes by sampling, the period and the translation by comparing values,
the first extreme and the asymptote by a fine search. It shares nothing with the generator's formulas.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade
from tests.w6b_support import TIERS, check_exam_generic, check_follow_through, check_tier, check_variety, check_wiring
from tests.w6b_support import q_of, skill

TOPIC = "w6b_sinusoid_features"
EXAM = "exd_x_w6b_sinusoid_features"
SK = skill("sinusoid-features-from-equation")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
_X = sympy.Symbol("x")


# -- reading the text ---------------------------------------------------------------------------------------------


def value_of(text):
    """'5π/12' / '-π/4' / '7/2' / '4' as a float."""
    text = text.strip()
    m = re.fullmatch(r"(-?)(\d*)π(?:/(\d+))?", text)
    if m:
        return (-1 if m.group(1) else 1) * int(m.group(2) or 1) / int(m.group(3) or 1) * math.pi
    return float(Fraction(text))


def equations(text):
    """Every ``y = a fn(inner) + d`` in the text, as callables (the bracket is read with SymPy)."""
    out = []
    for m in re.finditer(r"y = (-?\d*)(sin|cos|tan)\(", text):
        depth, i = 0, m.end() - 1
        for j in range(i, len(text)):
            depth += (text[j] == "(") - (text[j] == ")")
            if depth == 0:
                break
        inner = text[i + 1 : j]
        tail = re.match(r" ([+-]) (\d+)", text[j + 1 :])
        a = {"": 1, "-": -1}.get(m.group(1)) or int(m.group(1))
        d = (int(tail.group(2)) * (1 if tail.group(1) == "+" else -1)) if tail else 0
        angle = sympy.lambdify(_X, parse_expr(inner.replace("π", "pi"), transformations=_TRANSFORMS), "math")
        out.append(_make(m.group(2), a, angle, d))
    return out


def _make(fn, a, angle, d):
    f = _FN[fn]
    g = lambda x: a * f(angle(x)) + d  # noqa: E731
    g.fn, g.a, g.angle, g.d = fn, a, angle, d
    return g


def sample_extremes(g, n=40000):
    """(max, min) of ``g`` over two periods, by sampling."""
    span = 2 * smallest_period(g)
    ys = [g(span * i / n) for i in range(n + 1)]
    return max(ys), min(ys)


def smallest_period(g):
    xs = (0.13, 0.71, 1.37, 2.9)
    for k in range(1, 3000):
        t = k * math.pi / 120
        if all(abs(g(x + t) - g(x)) < 1e-9 for x in xs):
            return t
    raise AssertionError("no period found")


def smallest_shift(base, moved):
    """The translation of least size s with moved(x) = base(x - s) everywhere sampled."""
    xs = (0.13, 0.71, 1.37, 2.9, 4.4)
    for k in range(0, 1500):
        for s in (k * math.pi / 120, -k * math.pi / 120):
            if all(abs(moved(x) - base(x - s)) < 1e-9 for x in xs):
                return s
    raise AssertionError("no shift found")


def first_positive_extreme(g, goal, upto=40.0):
    """Smallest x > 0 at which g reaches ``goal`` (its maximum or minimum): the first sampled local turning point whose
    height is within 1e-3 of the goal, then refined by ternary search."""
    step = 1e-3
    x = step
    while x < upto:
        left, here, right = g(x - step), g(x), g(x + step)
        turning = (here >= left and here >= right) or (here <= left and here <= right)
        if turning and abs(here - goal) < 1e-3:
            lo, hi = x - step, x + step
            better = (lambda u, v: u > v) if goal > here - 1e-3 and here >= left else (lambda u, v: u < v)
            for _ in range(80):
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                if better(g(m1), g(m2)):
                    hi = m2
                else:
                    lo = m1
            return (lo + hi) / 2
        x += step
    raise AssertionError("no extreme found")


def first_positive_asymptote(g, upto=40.0):
    step = 1e-4
    x, prev = step, math.cos(g.angle(step))
    while x < upto:
        x += step
        cur = math.cos(g.angle(x))
        if prev * cur < 0:
            lo, hi = x - step, x
            for _ in range(60):
                mid = (lo + hi) / 2
                if math.cos(g.angle(lo)) * math.cos(g.angle(mid)) <= 0:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) / 2
        prev = cur
    raise AssertionError("no asymptote")


def oracle(text, answer_text):
    """Assert that the printed answer is the feature the text asks for, worked out numerically."""
    eqs = equations(text)
    claimed = value_of(answer_text)
    g = eqs[-1]
    lower = text.lower()
    if "horizontal translation" in lower or "moved horizontally" in lower:
        assert len(eqs) == 2, text
        base, moved = sorted(eqs, key=lambda e: abs(e.angle(0.0)) > 1e-12)  # the unshifted one has bracket bx
        got = smallest_shift(base, moved)
    elif "asymptote" in lower or "undefined" in lower:
        got = first_positive_asymptote(g)
    elif "y-intercept" in lower or "when x = 0" in lower or "cross the y-axis" in lower:
        got = g(0.0)
    elif "smallest positive" in lower or "least x > 0" in lower or "first reaches" in lower:
        hi, lo = sample_extremes(g)
        goal = hi if ("greatest" in lower) else lo
        got = first_positive_extreme(g, goal)
    elif "amplitude" in lower:
        hi, lo = sample_extremes(g)
        got = (hi - lo) / 2
    elif "period" in lower or "full cycle" in lower:
        got = smallest_period(g)
    elif "vertical translation" in lower or "midline" in lower or "halfway" in lower:
        hi, lo = sample_extremes(g)
        got = (hi + lo) / 2
    elif "maximum" in lower or "greatest" in lower or "largest" in lower:
        got = sample_extremes(g)[0]
    elif "minimum" in lower or "smallest" in lower or "least" in lower:
        got = sample_extremes(g)[1]
    else:  # pragma: no cover
        raise AssertionError(text)
    tolerance = (
        2e-3 if ("amplitude" in lower or "maximum" in lower or "minimum" in lower or "greatest" in lower) else 1e-3
    )
    assert abs(got - claimed) < tolerance, (text, answer_text, got, claimed)
    return g


# -- wiring and the shared checks ---------------------------------------------------------------------------------


def test_wiring():
    check_wiring(
        SK,
        TOPIC,
        EXAM,
        "expression",
        [
            "senior-model-periodic-motion-using-sine-and-cosine-functions",
            "senior-examine-period-changes-and-the-graphs-of-y",
        ],
    )


@pytest.mark.parametrize("tier", TIERS)
def test_tier_grading_and_distractors(tier):
    check_tier(SK, TOPIC, tier, seeds=100)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_variety(tier):
    check_variety(TOPIC, tier, seeds=200, min_texts=100, min_answers=20)


def test_misconceptions_listed_in_the_skill_cover_the_generator():
    listed = {m["id"] if isinstance(m, dict) else m.id for m in SK.misconceptions}
    used = set()
    for tier in TIERS:
        for seed in range(150):
            used.update(q_of(TOPIC, tier, seed).option_notes.values())
    for seed in range(60):
        for part in make(EXAM, seed=seed, difficulty="exam").parts:
            used.update(part.option_notes.values())
    assert used <= listed, used - listed


# -- the oracle ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answers_match_an_independent_numeric_oracle(tier):
    for seed in range(120):
        q = q_of(TOPIC, tier, seed)
        oracle(q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_wrong_by_the_oracle(tier):
    for seed in range(60):
        q = q_of(TOPIC, tier, seed)
        for text in q.distractors:
            with pytest.raises(AssertionError):
                oracle(q.text, text)


def test_no_python_leakage_in_text():
    for tier in TIERS:
        for seed in range(150):
            q = q_of(TOPIC, tier, seed)
            for text in [q.text, q.answer_text, *q.distractors, *getattr(q, "working", [])]:
                assert "None" not in text and "**" not in text and "pi" not in text.replace("π", ""), (seed, text)
                assert not re.search(r"\d/\d+x", text), (seed, text)  # "1/2x" would be ambiguous


# -- grading of π forms -------------------------------------------------------------------------------------------


def test_pi_forms_are_graded_by_value():
    found = 0
    for seed in range(300):
        q = q_of(TOPIC, "hard", seed)
        m = re.fullmatch(r"(-?)(\d*)π/(\d+)", q.answer_text)
        if not m:
            continue
        sign, num, den = m.group(1), m.group(2) or "1", m.group(3)
        for reply in (f"{sign}{num}pi/{den}", f"{sign}{num}*pi/{den}", f"{sign}({num}/{den})π", f"{sign}{num}π/{den}"):
            assert grade(SK, q, reply).correct, (q.text, reply)
        wrong = f"{sign}{num}π/{int(den) + 1}"
        assert not grade(SK, q, wrong).correct, (q.text, wrong)
        assert not grade(SK, q, str(round(value_of(q.answer_text) * 10) / 10)).correct, q.text
        found += 1
    assert found >= 30


# -- tier escalation ----------------------------------------------------------------------------------------------


def test_tiers_escalate():
    easy = [q_of(TOPIC, "easy", s).text for s in range(150)]
    medium = [q_of(TOPIC, "medium", s).text for s in range(150)]
    hard = [q_of(TOPIC, "hard", s).text for s in range(300)]
    # easy: always a factored bracket or none, a positive coefficient, sine or cosine, one feature read off
    assert all("tan" not in t for t in easy)
    assert all(not re.search(r"y = -", t) for t in easy)
    assert all(not re.search(r"\dx [+-] ", t) for t in easy)
    assert not any(re.search(r"asymptote|y-intercept|smallest positive|least x|first reaches", t) for t in easy)
    assert any(re.search(r"\(x \+ ", t) for t in easy)  # a shift to the left must be read as x + c
    # medium: expanded bracket, negative coefficients appear, and the translation is not simply read
    assert any(re.search(r"y = -\d*(sin|cos)", t) for t in medium)
    assert any(re.search(r"\dx [+-] \d*π", t) for t in medium)
    assert not any(re.search(r"asymptote|y-intercept|smallest positive|least x|first reaches", t) for t in medium)
    # hard: all three solving tasks occur
    assert any("asymptote" in t or "undefined" in t for t in hard)
    assert any("y-intercept" in t or "x = 0" in t or "y-axis" in t for t in hard)
    assert any(re.search(r"smallest positive value of x at which y|least x|first reaches", t) for t in hard)
    assert all(not re.search(r"State the amplitude|Find the period", t) for t in hard)


def test_determinism():
    for tier in TIERS:
        for seed in (0, 7, 31):
            assert q_of(TOPIC, tier, seed).to_json() == q_of(TOPIC, tier, seed).to_json()


# -- exam ---------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    check_exam_generic(SK, EXAM, n=80, marks=[1, 1, 2, 1], kinds=["expression"] * 4)
    q = make(EXAM, seed=0, difficulty="exam")
    assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_exam_oracle():
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        g = equations(q.stem)[0]
        hi, lo = sample_extremes(g)
        amp, period, shift, top = (value_of(p.answer_text) for p in q.parts)
        assert abs(amp - (hi - lo) / 2) < 2e-3, (q.stem, amp)
        assert abs(period - smallest_period(g)) < 1e-3, (q.stem, period)
        base = _make(g.fn, g.a, lambda x, b=g.angle(1.0) - g.angle(0.0): b * x, g.d)
        assert abs(smallest_shift(base, g) - shift) < 1e-3, (q.stem, shift)
        assert abs(top - hi) < 2e-3, (q.stem, top)


def test_exam_follow_through_from_a_wrong_amplitude():
    def wrong(q):
        right = q.parts[0].answer_text
        return [r for r in ("2", "8", "-4") if r != right]

    def follow(q, reply):
        part = q.parts[3]
        offset = Fraction(part.answer_text) - Fraction(q.parts[0].answer_text)
        return str(Fraction(reply) + offset)

    assert check_follow_through(SK, EXAM, "a", wrong, "d", follow, n=40) >= 20
