"""Tests the signed-area-definite-integral skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "signed-area-definite-integral"
TOPIC = "sr2b_signed_area"
EXAM = "sr2b_exam_signed_area"
_REGION = re.compile(
    r"Between x = (\d+) and x = (\d+) the graph is (above|below) the x-axis and the region between it and the axis has area (\d+)"
)


def _regions(stem):
    return [(int(a), int(b), 1 if side == "above" else -1, int(area)) for a, b, side, area in _REGION.findall(stem)]


def _integral(regions, lo, hi):
    inside = [r for r in regions if lo <= r[0] and r[1] <= hi]
    assert inside and inside[0][0] == lo and inside[-1][1] == hi, (regions, lo, hi)
    return sum(sign * area for _, _, sign, area in inside)


def _bounds(prompt):
    found = re.search(r"from (?:x = )?(\d+) to (?:x = )?(\d+)", prompt)
    return int(found.group(1)), int(found.group(2))


def _derive(stem):
    regions = _regions(stem)
    if not regions:
        return _derive_unknown(stem)
    lo, hi = _bounds(stem.split("Between")[-1].split("area")[-1].split("units.")[-1])
    if "total area" in stem.split("units.")[-1]:
        return sum(r[3] for r in regions if lo <= r[0] and r[1] <= hi)
    scaled = re.search(r"of \((-?\d*)f\(x\) ([+-]) (\d+)\) dx", stem)
    base = _integral(regions, lo, hi)
    if scaled:
        k = {"": 1, "-": -1}.get(scaled.group(1)) or int(scaled.group(1))
        m = int(scaled.group(3)) * (1 if scaled.group(2) == "+" else -1)
        return k * base + m * (hi - lo)
    return base


def _derive_unknown(stem):
    p, q = (int(v) for v in re.search(r"crosses the x-axis at x = (\d+) and x = (\d+)", stem).groups())
    first = int(re.search(r"from x = \d+ to x = \d+ has area (\d+)", stem).group(1))
    total = int(re.search(r"is (\d+) square units, and", stem).group(1))
    integral = int(re.search(r"dx = (-?\d+)\.", stem).group(1))
    first_sign = 1 if re.search(r"is above the x-axis between x = \d+ and x = \d+, below", stem) else -1
    middle = (total - first_sign * integral) // 2
    last = total - first - middle
    asked_lo = int(re.search(r"axis from x = (\d+) to x = (\d+)\.$", stem).group(1))
    return middle if asked_lo == p else last


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "number",
        ["definite-integral", "senior-calculate-the-area-between-curves-in-simple-cases"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(200):
        q = sup.tier_question(TOPIC, tier, seed)
        assert int(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all(len(_regions(t)) == 2 for t in easy)
    assert any("total area" in t for t in hard) and any("3f(x)" in t or "f(x) +" in t or "f(x) -" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_answers_include_negative_and_positive_integrals():
    values = [int(sup.tier_question(TOPIC, "medium", s).answer_text) for s in range(120)]
    assert any(v < 0 for v in values) and any(v > 0 for v in values)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [1, 1, 2, 2], [False, True, False, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=60, top=0.05)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(150):
        q = sup.exam_question(EXAM, seed)
        regions = _regions(q.text)
        lo, hi = regions[0][0], regions[-1][1]
        whole = _integral(regions, lo, hi)
        found = re.search(r"of \((-?\d*)f\(x\) ([+-]) (\d+)\) dx", q.text)
        k = {"": 1, "-": -1}.get(found.group(1)) or int(found.group(1))
        m = int(found.group(3)) * (1 if found.group(2) == "+" else -1)
        keys = [_integral(regions, lo, regions[1][1]), whole, sum(r[3] for r in regions), k * whole + m * (hi - lo)]
        assert [int(p.answer_text) for p in q.parts] == keys, (seed, q.text)


def test_exam_follow_through_from_wrong_earlier_parts():
    sk = sup.skill_object(SKILL)
    for seed in range(30):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        regions = _regions(q.text)
        last_region = regions[-1]
        sign_area = last_region[2] * last_region[3]
        lo, hi = regions[0][0], last_region[1]
        found = re.search(r"of \((-?\d*)f\(x\) ([+-]) (\d+)\) dx", q.text)
        k = {"": 1, "-": -1}.get(found.group(1)) or int(found.group(1))
        m = int(found.group(3)) * (1 if found.group(2) == "+" else -1)
        for offset in (3, 5, 7, 9, 11, 13):
            wrong_a = int(canon["a"]) + offset
            wrong_b = wrong_a + sign_area
            carried = k * wrong_b + m * (hi - lo)
            replies = {**canon, "a": str(wrong_a), "b": str(wrong_b), "d": str(carried)}
            result = grade(sk, q, replies)
            # the grader declines a carried reply that merely repeats an earlier box or is 0, 1 or -1: try the next
            if result.parts["b"].marks != q.parts[1].marks or result.parts["d"].marks != q.parts[3].marks:
                continue
            assert grade(sk, q, {**replies, "d": str(carried + 1)}).parts["d"].marks == 0, seed
            break
        else:
            raise AssertionError(f"no usable wrong reply for seed {seed}")
