"""Tests for eight senior General and Essential Mathematics skills built together.

Each skill has easy, medium, hard and a four-part exam chain. For every one:

* the key is accepted and every distractor of every part is rejected by `mathgen.engine.grade`;
* every wrong option carries a catalogued misconception and has the key's number of decimal places;
* the exam chains follow through from a wrong earlier part;
* the tiers differ in the kind of question asked, not only in size;
* an independent oracle reads the data back out of the question TEXT and recomputes the key without the
  generators' own helpers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from statistics import median

import pytest

import mathgen.generators  # noqa: F401  (registers every topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILLS = {s.id: s for s in load_skills()[1]}
TOPICS = {
    "grouped-frequency-table-class-intervals": ("r4g_grouped_frequency", "exd_x_r4g_grouped_frequency"),
    "back-to-back-stem-plot-compare": ("r4g_back_to_back_stem", "exd_x_r4g_back_to_back_stem"),
    "moving-average-median-smoothing": ("r4g_moving_average_smoothing", "exd_x_r4g_moving_average_smoothing"),
    "piecewise-linear-tariff-model": ("r4g_two_rate_tariff", "exd_x_r4g_two_rate_tariff"),
    "two-timetable-route-planning": ("r4g_two_timetable_route", "exd_x_r4g_two_timetable_route"),
    "sector-area-and-arc-from-degrees": ("r4g_sector_degrees", "exd_x_r4g_sector_degrees"),
    "longitude-local-clock-time": ("r4g_longitude_local_time", "exd_x_r4g_longitude_local_time"),
    "data-transformation-to-linearity": ("r4g_transform_linearity", "exd_x_r4g_transform_linearity"),
}
IDS = list(TOPICS)
TIERS = ("easy", "medium", "hard")
EXAM_MARKS = {
    "grouped-frequency-table-class-intervals": [2, 1, 2, 2],
    "back-to-back-stem-plot-compare": [1, 2, 2, 1],
    "moving-average-median-smoothing": [1, 2, 2, 1],
    "piecewise-linear-tariff-model": [2, 1, 1, 3],
    "two-timetable-route-planning": [1, 2, 2, 2],
    "sector-area-and-arc-from-degrees": [2, 2, 1, 2],
    "longitude-local-clock-time": [1, 1, 2, 2],
    "data-transformation-to-linearity": [2, 1, 2, 2],
}
SEEDS = range(40)


def _q(skill_id, tier, seed):
    topic = TOPICS[skill_id][1 if tier == "exam" else 0]
    return make(topic, seed=seed, difficulty=tier)


def _places(text):
    return len(text.partition(".")[2])


# -- wiring -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
def test_skill_wiring(skill_id):
    skill = SKILLS[skill_id]
    assert skill.status == "active" and skill.answer == "number"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (TOPICS[skill_id][0], "easy"),
        (TOPICS[skill_id][0], "medium"),
        (TOPICS[skill_id][0], "hard"),
        (TOPICS[skill_id][1], "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]
    assert not skill_id.startswith(("r4g", "uc2"))
    listed = {m.id for m in skill.misconceptions}
    used = set()
    for tier in (*TIERS, "exam"):
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            used |= set((q.option_notes or {}).values())
            for p in q.parts or []:
                used |= set((p.option_notes or {}).values())
    assert used <= listed, used - listed


# -- every key accepted, every distractor rejected -------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_single_answer_tiers(skill_id, tier):
    skill = SKILLS[skill_id]
    texts = set()
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert q.answer_kind == "number"
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (seed, text, q.answer_text)
            assert _places(text) == _places(q.answer_text), (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(skill_id, tier, seed).to_json()
        texts.add(q.text)
    assert len(texts) >= 36


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_chain(skill_id):
    skill = SKILLS[skill_id]
    for seed in SEEDS:
        q = _q(skill_id, "exam", seed)
        assert [p.marks for p in q.parts] == EXAM_MARKS[skill_id]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.label for p in q.parts] == list("abcd")
        assert len({p.answer_text for p in q.parts}) == 4
        assert sum(1 for p in q.parts if p.follow_through) >= 1
        assert q.parts[0].follow_through is None
        assert find_text_defects(q) == []
        assert q.to_json() == _q(skill_id, "exam", seed).to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(EXAM_MARKS[skill_id])
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert _places(text) == _places(p.answer_text), (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_follow_through_from_a_wrong_earlier_part(skill_id):
    """A listed slip in an earlier part still earns the later part whose reply follows from it."""
    from mathgen.grading.parts import follow_through_expression

    skill = SKILLS[skill_id]
    carried = 0
    for seed in SEEDS:
        q = _q(skill_id, "exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        places = {p.label: _places(p.answer_text) for p in q.parts}
        for target in q.parts:
            if not target.follow_through:
                continue
            for source in sorted(set(re.findall(r"part_([a-d])", target.follow_through))):
                for wrong in next(p for p in q.parts if p.label == source).distractors:
                    replies = {**canon, source: wrong}
                    for p in q.parts:
                        if p.label <= source or p.label >= target.label or not p.follow_through:
                            continue
                        value = follow_through_expression(p).subs(
                            {f"part_{k}": Fraction(v) for k, v in replies.items() if k < p.label}
                        )
                        replies[p.label] = _reply(value, places[p.label])
                    value = follow_through_expression(target).subs(
                        {f"part_{k}": Fraction(v) for k, v in replies.items() if k < target.label}
                    )
                    if not value.is_real:
                        continue
                    replies[target.label] = _reply(value, places[target.label])
                    if replies[target.label] == canon[target.label]:
                        continue
                    result = grade(skill, q, replies)
                    assert result.parts[source].marks == 0
                    # the grader carries an earlier reply only when it is the right size and in range
                    carried += result.parts[target.label].marks > 0
    assert carried >= 8


def _reply(value, places):
    """A learner's typed reply for a computed value: to the part's places, or two places when that is needed."""
    return f"{float(value):.{max(places, 2)}f}".rstrip("0").rstrip(".")


def _ask(text):
    return text.split("?")[0].split(". ")[-1][:40]


@pytest.mark.parametrize("skill_id", IDS)
def test_tiers_differ_in_shape(skill_id):
    """Hard asks questions the easy tier never asks, and the answers are not a short lookup."""
    asks = {t: Counter(_ask(_q(skill_id, t, s).text) for s in range(60)) for t in TIERS}
    assert set(asks["hard"]) - set(asks["easy"]), (asks["hard"], asks["easy"])
    answers = [_q(skill_id, "hard", s).answer_text for s in range(80)]
    assert len(set(answers)) >= 7


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_answers_vary(skill_id):
    answers = [_q(skill_id, "exam", s).answer_text for s in range(60)]
    assert len(set(answers)) >= 25


# -- independent oracles: read the data out of the TEXT -------------------------------------------------------------------


def _table(text):
    body = text.split("): frequency: ", 1)[1].split(". ", 1)[0]
    rows = re.findall(r"(\d+) to under (\d+): (\d+|x)", body)
    return [(int(a), int(b), None if f == "x" else int(f)) for a, b, f in rows]


def test_grouped_frequency_from_the_printed_table():
    seen = Counter()
    for seed in range(120):
        q = _q("grouped-frequency-table-class-intervals", "easy", seed)
        rows = _table(q.text)
        top = max(f for _, _, f in rows)
        lo, hi, _ = next(r for r in rows if r[2] == top)
        assert Fraction(q.answer_text) == Fraction(lo + hi, 2)
        q = _q("grouped-frequency-table-class-intervals", "medium", seed)
        rows = _table(q.text)
        n = sum(f for _, _, f in rows)
        mean = Fraction(sum(f * (lo + hi) for lo, hi, f in rows), 2 * n)
        assert abs(Fraction(q.answer_text) - mean) <= Fraction(5, 100), (seed, q.text)
        q = _q("grouped-frequency-table-class-intervals", "hard", seed)
        rows = _table(q.text)
        if "contains the median" in q.text:
            seen["median"] += 1
            n = sum(f for _, _, f in rows)
            cum = 0
            for lo, hi, f in rows:
                cum += f
                if cum >= (n + 1) / 2:
                    assert int(q.answer_text) == lo
                    break
        else:
            seen["missing"] += 1
            mean = Fraction(re.search(r"mean from the class midpoints is ([\d.]+)", q.text).group(1))
            known = [(lo, hi, f) for lo, hi, f in rows if f is not None]
            hidden = next(r for r in rows if r[2] is None)
            s = sum(f * Fraction(lo + hi, 2) for lo, hi, f in known)
            n = sum(f for _, _, f in known)
            mid = Fraction(hidden[0] + hidden[1], 2)
            assert (s + mid * int(q.answer_text)) / (n + int(q.answer_text)) == mean
    assert seen["median"] > 10 and seen["missing"] > 10


def _plot(text, groups):
    ga, gb = groups
    body = text.split(f"{ga} leaves | stem | {gb} leaves: ", 1)[1].split(". ", 1)[0]
    a, b = [], []
    for row in body.split("; "):
        left, stem, right = row.split(" | ")
        a += [10 * int(stem) + int(x) for x in left.split()]
        b += [10 * int(stem) + int(x) for x in right.split()]
    return sorted(a), sorted(b)


def _quartile_pair(values):
    n = len(values)
    return median(values[: n // 2]), median(values[(n + 1) // 2 :])


def test_back_to_back_from_the_printed_plot():
    seen = Counter()
    for tier in TIERS:
        for seed in range(100):
            q = _q("back-to-back-stem-plot-compare", tier, seed)
            names = re.search(r"rows written as (.+?) leaves \| stem \| (.+?) leaves:", q.text).groups()
            a, b = _plot(q.text, names)
            answer = Fraction(q.answer_text)
            if tier == "easy":
                mine = a if f"for {names[0]}." in q.text else b
                assert answer == Fraction(median(mine)).limit_denominator(2), (seed, q.text)
            elif tier == "medium":
                mine = a if f"for {names[0]}." in q.text else b
                q1, q3 = _quartile_pair(mine)
                assert answer == Fraction(q3 - q1).limit_denominator(2)
            elif "By how much" in q.text:
                seen["gap"] += 1
                ia, ib = (lambda p: p[1] - p[0])(_quartile_pair(a)), (lambda p: p[1] - p[0])(_quartile_pair(b))
                assert answer == Fraction(abs(ia - ib)).limit_denominator(2)
            else:
                seen["count"] += 1
                mine, other = (a, b) if f"for {names[0]} are" in q.text else (b, a)
                q3 = _quartile_pair(other)[1]
                assert int(answer) == sum(1 for v in mine if v > q3)
    assert seen["gap"] > 10 and seen["count"] > 5


def _series(text):
    return [
        int(v)
        for v in re.findall(
            r": (\d+)(?:;|\.)", text.split("records", 1)[1].split("Find")[0].split("Smooth")[0].split("For ")[0]
        )
    ]


def test_smoothing_from_the_printed_series():
    seen = Counter()
    for tier in TIERS:
        for seed in range(100):
            q = _q("moving-average-median-smoothing", tier, seed)
            s = _series(q.text)
            if "3-point moving mean for" in q.text and tier == "easy":
                k = int(re.search(r"mean for \w+ (\d+)", q.text).group(1)) - 1
                assert Fraction(q.answer_text) == Fraction(sum(s[k - 1 : k + 2]), 3)
                seen["mean3"] += 1
            elif "3-point moving medians" in q.text:
                k = int(re.search(r"smoothed value for \w+ (\d+)", q.text).group(1)) - 1
                assert int(q.answer_text) == sorted(s[k - 1 : k + 2])[1]
                seen["median3"] += 1
            elif "5-point moving mean for" in q.text:
                k = int(re.search(r"mean for \w+ (\d+)", q.text).group(1)) - 1
                assert abs(Fraction(q.answer_text) - Fraction(sum(s[k - 2 : k + 3]), 5)) < Fraction(1, 20)
                seen["mean5"] += 1
            elif "by how much" in q.text:
                k = int(re.search(r"For \w+ (\d+), by how much", q.text).group(1)) - 1
                w = s[k - 1 : k + 2]
                assert Fraction(q.answer_text) == Fraction(sum(w), 3) - sorted(w)[1]
                seen["gap"] += 1
            else:
                means = {d: sum(s[d - 3 : d + 2]) for d in range(3, len(s) - 1)}
                assert int(q.answer_text) == max(means, key=means.get)
                seen["peak"] += 1
    assert all(seen[k] > 5 for k in ("mean3", "median3", "mean5", "gap", "peak"))


def _money(m):
    return Fraction(m.replace("$", "").replace(",", ""))


def _tariff(text):
    """(fixed, rate 1, threshold, rate 2) in dollars from the first tariff sentence."""
    amounts = [_money(m) for m in re.findall(r"\$\d+\.\d\d", text.split("Plan B")[0])]
    t = int(re.search(r"first (\d+) ", text).group(1))
    if "includes the first" in text:
        return amounts[0], Fraction(0), t, amounts[1]
    return amounts[0], amounts[1], t, amounts[2]


def _cost(tf, u):
    f, r1, t, r2 = tf
    return f + r1 * min(u, t) + r2 * max(u - t, 0)


def test_tariff_from_the_printed_rates():
    seen = Counter()
    for tier in TIERS:
        for seed in range(120):
            q = _q("piecewise-linear-tariff-model", tier, seed)
            if "A tariff has" in q.text:
                seen["rate"] += 1
                costs = [_money(m) for m in re.findall(r"cost is (\$\d+\.\d\d)", q.text)]
                us = [
                    int(u)
                    for u in re.findall(r"(?:of|for) (\d+) (?:kilometres|days|gigabytes|kilograms)", q.text, re.I)
                ]
                assert Fraction(q.answer_text) == (costs[1] - costs[0]) / (us[1] - us[0])
                continue
            tf = _tariff(q.text)
            if tier == "easy":
                u = int(re.search(r"cost of [a-z ]*?(\d+) ", q.text).group(1))
                assert Fraction(q.answer_text) == _cost(tf, u), (seed, q.text)
            elif tier == "medium":
                total = _money(re.search(r"total was (\$\d+\.\d\d)", q.text).group(1))
                hits = [u for u in range(0, 120) if _cost(tf, u) == total]
                assert hits == [int(q.answer_text)], (seed, q.text)
            else:
                seen["break_even"] += 1
                fb, rb = (_money(m) for m in re.findall(r"\$\d+\.\d\d", q.text.split("Plan B")[1])[:2])
                hits = [u for u in range(0, 200) if _cost(tf, u) == fb + rb * u]
                assert hits == [int(q.answer_text)], (seed, q.text)
    assert seen["rate"] > 5 and seen["break_even"] > 15


def _times(text, mode, a, b):
    head = f"{mode.capitalize()} timetable, {a} to {b}: "
    body = text.split(head)[1].split(". ")[0]
    rows = re.findall(r"leaves (\d{4}) arrives (\d{4})", body)
    return [(int(h[:2]) * 60 + int(h[2:]), int(r[:2]) * 60 + int(r[2:])) for h, r in rows]


def _clock(minutes_):
    return (minutes_ // 60) * 100 + minutes_ % 60


def test_timetable_from_the_printed_tables():
    for tier in TIERS:
        for seed in range(100):
            q = _q("two-timetable-route-planning", tier, seed)
            m = re.search(r"(\w+) timetable, (.+?) to (.+?): .*?\. (\w+) timetable, \3 to (.+?): ", q.text)
            m1, o, c, m2, d = m.group(1).lower(), m.group(2), m.group(3), m.group(4).lower(), m.group(5)
            first, second = _times(q.text, m1, o, c), _times(q.text, m2, c, d)
            transfer = int(re.search(r"at least (\d+) minutes to change", q.text).group(1))
            if tier == "hard":
                deadline = re.search(r"by (\d{4})\.", q.text).group(1)
                deadline = int(deadline[:2]) * 60 + int(deadline[2:])
                best = max(d1 for d1, a1 in first for d2, a2 in second if d2 >= a1 + transfer and a2 <= deadline)
                assert int(q.answer_text) == _clock(best), (seed, q.text)
                continue
            ready = re.search(r"reaches \w[\w ]* at (\d{4})", q.text).group(1)
            ready = int(ready[:2]) * 60 + int(ready[2:])
            d1, a1 = next(s for s in first if s[0] >= ready)
            arrival = min(a2 for d2, a2 in second if d2 >= a1 + transfer)
            expected = _clock(arrival) if tier == "easy" else arrival - ready
            assert int(q.answer_text) == expected, (seed, tier, q.text)


def test_sector_from_the_printed_radius_and_angle():
    seen = Counter()
    for tier in TIERS:
        for seed in range(120):
            q = _q("sector-area-and-arc-from-degrees", tier, seed)
            answer = Fraction(q.answer_text)
            theta = re.search(r"(\d+)°", q.text)
            if "area of the path" in q.text:
                seen["path"] += 1
                r, big = (int(v) for v in re.findall(r"radius (\d+) ", q.text))
                t = int(theta.group(1))
                assert abs(float(answer) - t / 360 * math.pi * (big**2 - r**2)) < 0.051
            elif "Find the radius" in q.text:
                seen["radius"] += 1
                area = float(re.search(r"area is ([\d.]+) ", q.text).group(1))
                t = int(theta.group(1))
                assert abs(float(answer) - math.sqrt(area * 360 / (math.pi * t))) < 0.051
            elif "centre angle of the sector" in q.text:
                seen["angle"] += 1
                r = int(re.search(r"radius (\d+) ", q.text).group(1))
                arc = float(re.search(r"is ([\d.]+) \w+ long", q.text).group(1))
                assert int(answer) == round(arc * 360 / (2 * math.pi * r))
            else:
                r = int(re.search(r"radius (\d+) ", q.text).group(1))
                t = int(theta.group(1))
                if "Find its area" in q.text:
                    expected = t / 360 * math.pi * r * r
                elif "perimeter" in q.text:
                    expected = t / 360 * 2 * math.pi * r + 2 * r
                else:
                    expected = t / 360 * 2 * math.pi * r
                assert abs(float(answer) - expected) < 0.051, (seed, q.text)
    assert all(seen[k] > 5 for k in ("path", "radius", "angle"))


def _lon(text):
    return [(1 if d == "E" else -1) * int(n) for n, d in re.findall(r"(\d+)°([EW])", text)][:2]


def test_local_time_from_the_printed_longitudes():
    for tier in TIERS:
        for seed in range(100):
            q = _q("longitude-local-clock-time", tier, seed)
            la, lb = _lon(q.text)
            if tier == "hard":
                t = re.search(r"leaves Port \w+ at (\d{4})", q.text).group(1)
                h, mnt = re.search(r"takes (\d+) hours(?: (\d+) minutes)?", q.text).groups()
                flight = int(h) * 60 + int(mnt or 0)
            else:
                t = re.search(r"is (\d{4}),", q.text).group(1)
                flight = 0
            minutes_ = int(t[:2]) * 60 + int(t[2:]) + flight + (lb - la) * 4
            assert int(q.answer_text) == _clock(minutes_), (seed, q.text)


def test_transformation_from_the_printed_points():
    seen = Counter()
    for seed in range(150):
        q = _q("data-transformation-to-linearity", "medium", seed)
        (x1, y1), (x2, y2) = [(int(a), Fraction(b)) for a, b in re.findall(r"\((\d+), ([\d.]+)\)", q.text)]
        x0 = int(re.search(r"when x = (\d+)", q.text).group(1))
        kind = (
            "sq" if "upward" in q.text else "log" if "flattening out." in q.text and "increasing" in q.text else "rec"
        )
        seen[kind] += 1
        f = {"sq": lambda x: x * x, "log": lambda x: math.log10(x), "rec": lambda x: 1 / x}[kind]
        slope = (float(y2) - float(y1)) / (f(x2) - f(x1))
        assert abs(float(q.answer_text) - (float(y1) + slope * (f(x0) - f(x1)))) < 0.051, (seed, q.text)
        q = _q("data-transformation-to-linearity", "hard", seed)
        xs = [int(v) for v in re.search(r"For x = ([\d, ]+), the values", q.text).group(1).split(", ")]
        ys = [
            float(v.rstrip("."))
            for v in re.search(r"values of y are ([\d., ]+?)\.(?: |$)", q.text).group(1).split(", ")
        ]
        fits = []
        for name, f in (("sq", lambda x: x * x), ("log", math.log10), ("rec", lambda x: 1 / x)):
            g = [(ys[i + 1] - ys[i]) / (f(xs[i + 1]) - f(xs[i])) for i in range(3)]
            if max(g) - min(g) < 1e-6:
                fits.append((name, f, g[0]))
        assert len(fits) == 1, (seed, q.text)
        _, f, slope = fits[0]
        a = ys[0] - slope * f(xs[0])
        if "find the value of x" in q.text:
            yt = float(re.search(r"gives y = ([\d.]+)", q.text).group(1).rstrip("."))
            z = (yt - a) / slope
            expected = {"sq": math.sqrt, "log": lambda v: 10**v, "rec": lambda v: 1 / v}[fits[0][0]](z)
            assert abs(float(q.answer_text) - expected) < 1e-6 * expected + 1e-9
        else:
            x0 = int(re.search(r"predict y when x = (\d+)", q.text).group(1))
            assert abs(float(q.answer_text) - (a + slope * f(x0))) < 0.051
    assert all(seen[k] > 20 for k in ("sq", "log", "rec"))
