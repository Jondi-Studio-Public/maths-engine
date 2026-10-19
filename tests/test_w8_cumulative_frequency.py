"""Tests `cumulative-frequency-ogive` (topic `w8_ogive_read`) and `frequency-polygon-read` (topic `w8_polygon_read`)
and their exam chains.

The oracles read the plotted points off the FIGURE and the boundaries and midpoints off the question TEXT, and re-derive
each key from the class frequencies (differences of the cumulative heights, or the polygon heights themselves) by direct
summation over the raw values the classes stand for, never from the generators' own helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.w8_cumulative_frequency_build as gen
from mathgen import figures
from tests import w7b_support as sup

OGIVE_SKILL = sup.skill_of("cumulative-frequency-ogive")
POLYGON_SKILL = sup.skill_of("frequency-polygon-read")
CASES = [
    (OGIVE_SKILL, gen.OGIVE, gen.OGIVE_EXAM),
    (POLYGON_SKILL, gen.POLYGON, gen.POLYGON_EXAM),
]
NUM = r"(\d+(?:\.\d+)?)"


def _points(q):
    return sorted((Fraction(p["at"][0]).limit_denominator(100), Fraction(p["at"][1])) for p in q.figure["points"])


def _raw_ogive(q):
    """The values the ogive stands for, each at its class's lower boundary, from the plotted points alone."""
    pts = _points(q)
    out = []
    for (x0, y0), (_x1, y1) in zip(pts, pts[1:]):
        out += [x0] * int(y1 - y0)
    return out


def _raw_polygon(q):
    return [(x, int(y)) for x, y in _points(q)]


@pytest.mark.parametrize("skill,topic,exam", CASES)
def test_wiring(skill, topic, exam):
    sup.check_wiring(skill, topic, exam, "number")
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.domain == "statistics-probability" and skill.placement_eligible is False
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("skill,topic,exam", CASES)
@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(skill, topic, exam, tier):
    sup.check_single_tier(skill, topic, tier)


@pytest.mark.parametrize("skill,topic,exam", CASES)
def test_pools_and_shapes(skill, topic, exam):
    sup.check_pools(topic)


@pytest.mark.parametrize("skill,topic,exam", CASES)
@pytest.mark.parametrize("tier", sup.TIERS)
def test_figure_valid_unleaked_and_read_back(skill, topic, exam, tier):
    reader = figures.get_reader(topic)
    for seed in range(60):
        q = sup.single(topic, tier, seed)
        assert q.figure["kind"] == "cartesian" and figures.validate(q.figure) == []
        assert reader(q) == q.answer_text, (tier, seed, q.text)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert not re.search(r"\d", q.figure["alt"])
        assert "python" not in q.text.lower() and "**" not in q.text


@pytest.mark.parametrize("skill,topic,exam", CASES)
def test_exam_chain_shape_and_reader(skill, topic, exam):
    sup.check_exam_chain(skill, exam, [1, 2, 1, 2] if topic == gen.OGIVE else [1, 1, 2, 2], ["number"] * 4)
    reader = figures.get_reader(exam)
    for seed in range(60):
        q = sup.exam(exam, seed)
        assert reader(q) == q.answer_text, (seed, q.text)


@pytest.mark.parametrize("skill,topic,exam", CASES)
@pytest.mark.parametrize("tier", sup.TIERS)
def test_typed_tiers_have_enough_distinct_answers(skill, topic, exam, tier):
    assert len(sup.answer_spread(topic, tier, 60)) >= 12, tier


@pytest.mark.parametrize("skill,topic,exam", CASES)
@pytest.mark.parametrize("tier", sup.TIERS)
def test_the_key_is_not_predictably_ranked_among_the_options(skill, topic, exam, tier):
    ranks = Counter()
    for seed in range(120):
        q = sup.single(topic, tier, seed)
        values = sorted([float(q.answer_text), *map(float, q.distractors)])
        ranks[values.index(float(q.answer_text))] += 1
    assert len(ranks) >= 3, (tier, ranks)
    assert max(ranks.values()) <= 0.6 * 120, (tier, ranks)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_ogive_points_sit_on_gridlines_and_quartiles_are_plotted(tier):
    for seed in range(60):
        q = sup.single(gen.OGIVE, tier, seed)
        fig = q.figure
        top = fig["y_range"][1]
        pts = _points(q)
        assert pts[0][1] == 0 and pts[-1][1] == top
        assert all(y % fig["y_grid"] == 0 for _x, y in pts)
        assert all((x - fig["x_range"][0]) % fig["x_grid"] == 0 for x, _y in pts)
        assert all(a[1] < b[1] for a, b in zip(pts, pts[1:])), "the ogive must rise strictly"
        if tier != "easy":
            heights = {y for _x, y in pts}
            assert {top / 4, top / 2, 3 * top / 4} <= heights


def test_ogive_keys_by_independent_arithmetic():
    seen = set()
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single(gen.OGIVE, tier, seed)
            raw = _raw_ogive(q)
            n = len(raw)
            text = q.text.split("shows the results.")[-1]
            low = text.lower()
            key = Fraction(q.answer_text)
            spans = re.findall(rf"from {NUM} to less than {NUM}", text)
            if "altogether" in low or "total number" in low:
                assert key == n
                seen.add("total")
            elif "median" in low:
                # n/2 values lie below the median and the rest at or above it
                assert sum(1 for v in raw if v < key) == n // 2 and sum(1 for v in raw if v <= key) > n // 2
                seen.add("median")
            elif "interquartile" in low:
                lo = next(x for x, _ in _points(q) if sum(1 for v in raw if v < x) == n // 4)
                hi = next(x for x, _ in _points(q) if sum(1 for v in raw if v < x) == 3 * n // 4)
                assert key == hi - lo
                seen.add("iqr")
            elif "modal class" in low:
                counts = Counter(raw)
                assert counts[key] == max(counts.values()) and list(counts.values()).count(max(counts.values())) == 1
                seen.add("modal")
            elif "how many more" in low:
                (a1, b1), (a2, b2) = ((Fraction(a), Fraction(b)) for a, b in spans)
                c1 = sum(1 for v in raw if a1 <= v < b1)
                c2 = sum(1 for v in raw if a2 <= v < b2)
                assert key == c1 - c2 > 0
                seen.add("more")
            elif spans:
                a, b = (Fraction(x) for x in spans[0])
                count = sum(1 for v in raw if a <= v < b)
                assert key == (Fraction(100 * count, n) if "percentage" in low else count)
                seen.add("between-pct" if "percentage" in low else "between")
            else:
                x = Fraction(re.findall(NUM, text)[-1])
                count = sum(1 for v in raw if v >= x) if "or more" in low else sum(1 for v in raw if v < x)
                assert key == (Fraction(100 * count, n) if "percentage" in low else count)
                seen.add("pct" if "percentage" in low else ("atleast" if "or more" in low else "below"))
    assert seen == {"total", "median", "iqr", "modal", "more", "between", "between-pct", "pct", "atleast", "below"}


def test_polygon_keys_by_independent_arithmetic():
    seen = set()
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single(gen.POLYGON, tier, seed)
            data = _raw_polygon(q)
            width = Fraction(q.figure["x_grid"]).limit_denominator(100)
            n = sum(f for _m, f in data)
            text = q.text.split("midpoint of its class.")[-1]
            low = text.lower()
            key = Fraction(q.answer_text)
            assert data[0][1] == 0 and data[-1][1] == 0, "the polygon closes to zero at both ends"
            modal = max(data, key=lambda p: p[1])
            if "midpoint of the modal class" in low:
                assert key == modal[0]
                seen.add("modal-mid")
            elif "lower boundary of the modal class" in low:
                assert key == modal[0] - width / 2
                seen.add("modal-bound")
            elif "altogether" in low or "total number" in low:
                assert key == n
                seen.add("total")
            elif "estimate the mean" in low:
                assert key == Fraction(sum(m * f for m, f in data), n).limit_denominator(1000)
                assert Fraction(sum(m * f for m, f in data), n).denominator in (1, 2)
                seen.add("mean")
            elif "how many more" in low:
                m = Fraction(re.findall(NUM, text)[-1])
                assert key == modal[1] - next(f for mid, f in data if mid == m)
                seen.add("more")
            elif "midpoint" in low:
                m = Fraction(re.findall(NUM, text.split("midpoint")[-1])[0])
                assert key == next(f for mid, f in data if mid == m)
                seen.add("freq")
            else:
                x = Fraction(re.findall(NUM, text)[-1])
                if "or more" in low:
                    count = sum(f for m, f in data if m - width / 2 >= x)
                else:
                    count = sum(f for m, f in data if m + width / 2 <= x)
                assert key == (Fraction(100 * count, n) if "percentage" in low else count)
                seen.add("pct" if "percentage" in low else ("ge" if "or more" in low else "lt"))
    assert seen == {"modal-mid", "modal-bound", "total", "mean", "more", "freq", "pct", "ge", "lt"}


def test_ogive_exam_keys_by_independent_arithmetic():
    for seed in range(60):
        q = sup.exam(gen.OGIVE_EXAM, seed)
        raw = _raw_ogive(q)
        n = len(raw)
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == n
        lo, hi = (Fraction(v) for v in re.search(rf"from {NUM} to less than {NUM}", b.prompt).groups())
        assert Fraction(b.answer_text) == sum(1 for v in raw if lo <= v < hi)
        pts = [x for x, _ in _points(q)]
        assert Fraction(c.answer_text) == next(x for x in pts if sum(1 for v in raw if v < x) == n // 2)
        q1 = next(x for x in pts if sum(1 for v in raw if v < x) == n // 4)
        q3 = next(x for x in pts if sum(1 for v in raw if v < x) == 3 * n // 4)
        assert Fraction(d.answer_text) == q3 - q1


def test_polygon_exam_keys_and_follow_through():
    for seed in range(60):
        q = sup.exam(gen.POLYGON_EXAM, seed)
        data = _raw_polygon(q)
        width = Fraction(q.figure["x_grid"]).limit_denominator(100)
        n = sum(f for _m, f in data)
        a, b, c, d = q.parts
        m = Fraction(re.search(rf"midpoint {NUM}", a.prompt).group(1))
        assert Fraction(a.answer_text) == next(f for mid, f in data if mid == m)
        assert Fraction(b.answer_text) == n
        x = Fraction(re.findall(NUM, c.prompt)[-1])
        assert Fraction(c.answer_text) == sum(f for mid, f in data if mid - width / 2 >= x)
        x = Fraction(re.findall(NUM, d.prompt)[-1])
        below = sum(f for mid, f in data if mid + width / 2 <= x)
        assert Fraction(d.answer_text) == Fraction(100 * below, n)

    # follow-through: a wrong total in (b) still earns (d) when the percentage is worked from it
    def wrong_total(q, canon):
        data = _raw_polygon(q)
        n = sum(f for _m, f in data)
        x = Fraction(re.findall(NUM, q.parts[3].prompt)[-1])
        width = Fraction(q.figure["x_grid"]).limit_denominator(100)
        below = sum(f for mid, f in data if mid + width / 2 <= x)
        bad = n + 2
        pct = Fraction(100 * below, bad)
        if pct.denominator != 1:
            return None
        return {"b": str(bad), "d": str(int(pct))}

    assert sup.check_follow_through(POLYGON_SKILL, gen.POLYGON_EXAM, "d", wrong_total, 60) >= 1


def test_generators_are_fast():
    import time

    for topic in (gen.OGIVE, gen.POLYGON):
        for tier in sup.TIERS:
            t0 = time.perf_counter()
            for seed in range(30):
                sup.single(topic, tier, seed)
            assert (time.perf_counter() - t0) / 30 < 0.015, (topic, tier)
