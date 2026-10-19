"""Tests for the wider exam chain appended to `senior-understand-the-concepts-of-a-probability-density-function`:
`exd_x_r3f_pdf_wider`, with the answers re-derived from the density in the stem by summing exact piece areas."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from tests import r3f_support as sup

SKILL = "senior-understand-the-concepts-of-a-probability-density-function"
EXAM = "exd_x_r3f_pdf_wider"


def _density(text):
    """(f as a function of Fraction x, a, b) read from the stem; k is solved from the total area."""
    ramp = re.search(r"f\(x\) = kx for 0 ≤ x ≤ (\d+), f\(x\) = (\d+)k for \1 ≤ x ≤ (\d+)", text)
    if ramp:
        a, b = int(ramp.group(1)), int(ramp.group(3))
        shape = lambda x: x if x <= a else Fraction(a)  # noqa: E731
    else:
        m = re.search(r"f\(x\) = k for 0 ≤ x ≤ (\d+), f\(x\) = k\((\d+) - x\)/(\d+) for", text)
        a, b = int(m.group(1)), int(m.group(2))
        shape = lambda x: Fraction(1) if x <= a else Fraction(b - x, b - a)  # noqa: E731
    return shape, a, b


def _area(shape, a, lo, hi):
    """Exact area under k*shape between lo and hi (k = 1): shape is linear on each side of a."""
    lo, hi = Fraction(lo), Fraction(hi)
    total = Fraction(0)
    for left, right in ((lo, min(hi, Fraction(a))), (max(lo, Fraction(a)), hi)):
        if right > left:
            total += (right - left) * (shape(left) + shape(right)) / 2
    return total


def test_wired():
    sources = [s for s in sup.skill(SKILL).sources if s.difficulty == "exam"]
    assert [s.topic for s in sources] == ["exd_x_r2k_density", EXAM]
    assert len({s.effective_key for s in sources}) == 2
    assert all(list(s.stages) == ["exam"] for s in sources)


def _questions(n=300):
    return [make(EXAM, seed=s, tier="exam") for s in range(n)]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [2, 2, 3, 3], 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, True, True, True] for z in _questions(40))
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in _questions():
        shape, a, b = _density(q.text)
        k = 1 / _area(shape, a, 0, b)
        prompts = " ".join(p.prompt for p in q.parts)
        c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", prompts).groups())
        got = {label: Fraction(text) for label, text in sup.canonical(q).items()}
        median = None
        lo, hi = Fraction(0), Fraction(b)
        for _ in range(200):  # bisection on the exact area
            mid = (lo + hi) / 2
            if k * _area(shape, a, 0, mid) < Fraction(1, 2):
                lo = mid
            else:
                hi = mid
        median = (lo + hi) / 2
        if got["a"] != k or got["b"] != k * _area(shape, a, 0, a) or abs(got["c"] - median) > Fraction(1, 10**9):
            bad.append((q.text, got))
        if got["d"] != k * _area(shape, a, c1, c2):
            bad.append((q.text, got))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_k():
    def replies(q):
        shape, a, b = _density(q.text)
        prompts = " ".join(p.prompt for p in q.parts)
        c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", prompts).groups())
        ramp = "kx for" in q.text
        for text in q.parts[0].distractors:
            wrong_k = Fraction(text)
            pb = wrong_k * _area(shape, a, 0, a)
            pd = wrong_k * _area(shape, a, c1, c2)
            pm = a + (Fraction(1, 2) - pb) / (wrong_k * a) if ramp else 1 / (2 * wrong_k)
            if 0 < pb < 1 and 0 < pd < 1 and 0 < pm < b:
                return {"a": text, "b": str(pb), "c": str(pm), "d": str(pd)}, {"b", "c", "d"}, {"a"}
        return None

    sup.check_follow_through(SKILL, EXAM, replies, n=150, at_least=10)
