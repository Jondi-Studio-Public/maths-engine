"""Tests for the General and Essential number, measure and data skills (`r3j_*`): Heron's formula, salary and pay
conversions, classifying variables, recurrence models, sample spaces, imperial lengths, the slope of a cost graph and
speed, distance and time.

Every skill has easy, medium, hard and exam sources. For each one:

* the key is accepted and every distractor of every part is rejected by `mathgen.grading.grade`;
* the exam chains follow through from a wrong earlier part;
* an independent oracle reads the numbers back out of the question TEXT (or the drawn figure) and recomputes the key
  without calling the generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import product
from math import sqrt

import pytest

import mathgen.generators.r3j_classify_build  # noqa: F401  (registers the topics)
import mathgen.generators.r3j_heron_build  # noqa: F401
import mathgen.generators.r3j_imperial_build  # noqa: F401
import mathgen.generators.r3j_ratio_share_build  # noqa: F401
import mathgen.generators.r3j_recurrence_build  # noqa: F401
import mathgen.generators.r3j_salary_build  # noqa: F401
import mathgen.generators.r3j_sample_space_build  # noqa: F401
import mathgen.generators.r3j_slope_context_build  # noqa: F401
import mathgen.generators.r3j_speed_build  # noqa: F401
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.generators._r3j_kit import number_text
from mathgen.generators.pv1a_line_figure_build import read_intersection, read_line
from mathgen.grading.parts import follow_through_expression
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILLS = {s.id: s for s in load_skills()[1]}
IDS = [
    "triangle-area-heron",
    "salary-conversion-pay",
    "compound-recurrence-model",
    "build-sample-space",
    "convert-imperial-length",
    "interpret-slope-in-context",
    "speed-distance-time-average-speed",
]
TOPICS = {
    "triangle-area-heron": ("r3j_heron", "exd_x_r3j_heron_plot"),
    "salary-conversion-pay": ("r3j_salary", "exd_x_r3j_salary_vs_hourly"),
    "compound-recurrence-model": ("r3j_recurrence", "exd_x_r3j_recurrence_savings"),
    "build-sample-space": ("r3j_sample_space", "exd_x_r3j_sample_space_events"),
    "convert-imperial-length": ("r3j_imperial", "exd_x_r3j_imperial_voyage"),
    "interpret-slope-in-context": ("r3j_slope_context", "exd_x_r3j_slope_two_plans"),
    "speed-distance-time-average-speed": ("r3j_speed", "exd_x_r3j_speed_journey"),
}
ANSWER = {
    "triangle-area-heron": "number",
    "salary-conversion-pay": "number",
    "compound-recurrence-model": "number",
    "build-sample-space": "set",
    "convert-imperial-length": "number",
    "interpret-slope-in-context": "number",
    "speed-distance-time-average-speed": "number",
}
TIERS = ("easy", "medium", "hard")
EXAM_MARKS = {
    "triangle-area-heron": [1, 2, 2, 2],
    "salary-conversion-pay": [1, 1, 2, 2],
    "compound-recurrence-model": [2, 2, 2, 2],
    "build-sample-space": [1, 2, 2, 1],
    "convert-imperial-length": [1, 1, 2, 2],
    "interpret-slope-in-context": [1, 1, 2, 2],
    "speed-distance-time-average-speed": [1, 1, 2, 2],
}
DEEPENED = {
    "classify-data": ("r3j_classify_variable", "exd_x_r3j_classify_variables"),
    "divide-quantity-in-ratio": (None, "exd_x_r3j_ratio_shares"),
}


def _q(skill_id, tier, seed):
    topic = TOPICS[skill_id][1 if tier == "exam" else 0]
    return make(topic, seed=seed, difficulty=tier)


def _close(question, expected, slack=0):
    """True when the key is the expected value rounded to the key's own stated places."""
    places = len(question.answer_text.partition(".")[2])
    return abs(float(question.answer_text) - float(expected)) <= 0.5 * 10**-places + slack + 1e-9


# -- wiring -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
def test_skill_wiring(skill_id):
    skill = SKILLS[skill_id]
    assert skill.answer == ANSWER[skill_id] and skill.status == "active"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (TOPICS[skill_id][0], "easy"),
        (TOPICS[skill_id][0], "medium"),
        (TOPICS[skill_id][0], "hard"),
        (TOPICS[skill_id][1], "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]
    assert skill.prerequisites
    assert not skill_id.startswith(("r3j", "uc2"))
    assert skill.misconceptions


def test_deepened_skills_keep_their_old_sources_and_gain_new_ones():
    classify = SKILLS["classify-data"]
    assert [(s.topic, s.difficulty) for s in classify.sources] == [
        ("w3_classify_easy", "easy"),  # w3 replaced the guessable easy tier
        ("r3j_classify_variable", "medium"),
        ("r3j_classify_variable", "hard"),
        ("exd_x_r3j_classify_variables", "exam"),
    ]
    ratio = SKILLS["divide-quantity-in-ratio"]
    assert [s.topic for s in ratio.sources][:4] == ["p13a_divide_ratio"] * 4
    assert ratio.sources[-1].topic == "exd_x_r3j_ratio_shares" and ratio.sources[-1].stages == ["exam"]


# -- every key accepted, every distractor rejected --------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_single_answer_tiers(skill_id, tier):
    skill = SKILLS[skill_id]
    texts = set()
    for seed in range(40):
        q = _q(skill_id, tier, seed)
        assert q.answer_kind == ANSWER[skill_id], (seed, q.answer_kind)
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        places = len(q.answer_text.partition(".")[2])
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (seed, text, q.answer_text)
            if q.answer_kind == "number":
                assert len(text.partition(".")[2]) == places, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(skill_id, tier, seed).to_json()
        texts.add(q.text)
    assert len(texts) >= {"build-sample-space": 4, "interpret-slope-in-context": 20}.get(skill_id, 30), len(texts)


@pytest.mark.parametrize("skill_id", IDS)
def test_tiers_do_not_share_question_texts(skill_id):
    by_tier = {tier: {_q(skill_id, tier, seed).text for seed in range(30)} for tier in TIERS}
    assert not (by_tier["easy"] & by_tier["medium"]) and not (by_tier["medium"] & by_tier["hard"])


# -- the exam chains -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_chain(skill_id):
    skill = SKILLS[skill_id]
    for seed in range(40):
        q = _q(skill_id, "exam", seed)
        assert [p.marks for p in q.parts] == EXAM_MARKS[skill_id]
        assert [p.label for p in q.parts] == list("abcd")
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
                assert p.option_notes[text] in CATALOGUE


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_follow_through_from_a_wrong_earlier_part(skill_id):
    """A wrong earlier part (a listed slip) still earns a later part whose reply follows from it."""
    skill = SKILLS[skill_id]
    carried = 0
    for seed in range(40):
        q = _q(skill_id, "exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        places = {p.label: len(p.answer_text.partition(".")[2]) for p in q.parts if p.answer_kind == "number"}
        for target in q.parts:
            if not target.follow_through or target.answer_kind not in ("number", "fraction"):
                continue
            for source in sorted(set(re.findall(r"part_([a-d])", target.follow_through))):
                src = next(p for p in q.parts if p.label == source)
                if src.answer_kind not in ("number", "fraction"):
                    continue
                wrong = src.distractors[0]
                replies = {**canon, source: wrong}
                broken = False
                for p in q.parts:
                    if p.label <= source or p.label > target.label or not p.follow_through:
                        continue
                    expr = follow_through_expression(p)
                    names = set(re.findall(r"part_([a-d])", p.follow_through))
                    if any(n not in replies for n in names):
                        broken = True
                        break
                    used = {k: Fraction(_as_fraction(replies[k])) for k in names}
                    value = expr.subs({f"part_{k}": v for k, v in used.items()})
                    value = Fraction(float(value)).limit_denominator(10**9)
                    if value in (0, 1, -1) or value in used.values():
                        broken = True  # the engine withholds a trivial carry: it needs an actual working step
                        break
                    replies[p.label] = (
                        number_text(value, places[p.label]) if p.answer_kind == "number" else _fraction_text(value)
                    )
                if broken or replies[target.label] == canon[target.label]:
                    continue
                result = grade(skill, q, replies)
                assert result.parts[source].marks == 0
                assert result.parts[target.label].marks > 0, (seed, source, target.label, replies)
                carried += 1
    assert carried >= 20


def _as_fraction(text):
    return Fraction(str(text).replace(" ", ""))


def _fraction_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# -- independent oracles ----------------------------------------------------------------------------------------------------


def _heron(a, b, c):
    s = (a + b + c) / 2
    return sqrt(s * (s - a) * (s - b) * (s - c))


def test_heron_keys_from_the_printed_sides():
    for tier in ("easy", "medium"):
        for seed in range(150):
            q = _q("triangle-area-heron", tier, seed)
            sides = re.findall(r"([\d.]+) (m|cm)", q.text.split("Find its area")[0])
            lengths = [float(v) / (100 if unit == "cm" else 1) for v, unit in sides]
            assert len(lengths) == 3
            assert _close(q, _heron(*lengths)), (seed, q.text, q.answer_text, _heron(*lengths))
    for seed in range(150):
        q = _q("triangle-area-heron", "hard", seed)
        sides = [float(v) for v in re.findall(r"([\d.]+) m", q.text.split(". ")[0])]
        area = _heron(*sides)
        if "right angles" in q.text:
            expected = 2 * area / max(sides)
        else:
            rate = int(re.search(r"\$(\d+) per square metre", q.text).group(1))
            expected = area * rate
        assert _close(q, expected), (seed, q.text, q.answer_text, expected)


def test_heron_exam_from_the_printed_sides():
    for seed in range(100):
        q = _q("triangle-area-heron", "exam", seed)
        a, b, c = (float(v) for v in re.findall(r"([\d.]+) m", q.text))
        rate = int(re.search(r"\$(\d+) per square metre", q.text).group(1))
        area = _heron(a, b, c)
        keys = [float(p.answer_text) for p in q.parts]
        assert abs(keys[0] - (a + b + c) / 2) < 0.051
        assert abs(keys[1] - area) < 0.051
        assert abs(keys[2] - 2 * area / max(a, b, c)) < 0.051
        assert abs(keys[3] - area * rate) <= 0.5 + 1e-9


def _money(text):
    return Fraction(text.replace("$", "").replace(",", ""))


_PERIODS = {"week": 52, "fortnight": 26, "month": 12}


def test_salary_keys_from_the_text():
    seen = set()
    for seed in range(200):
        q = _q("salary-conversion-pay", "easy", seed)
        if "annual salary" in q.text:
            pay = _money(re.search(r"is paid (\$[\d,.]+) each", q.text).group(1))
            period = re.search(r"each (week|fortnight|month)", q.text).group(1)
            assert Fraction(q.answer_text) == pay * _PERIODS[period], (seed, q.text)
            seen.add("up")
        else:
            salary = _money(re.search(r"salary of (\$[\d,]+)", q.text).group(1))
            period = re.search(r"one (week|fortnight|month)", q.text).group(1)
            assert _close(q, salary / _PERIODS[period]), (seed, q.text)
            seen.add("down")
    assert seen == {"up", "down"}


def test_salary_medium_and_hard_from_the_text():
    seen = set()
    for seed in range(300):
        q = _q("salary-conversion-pay", "medium", seed)
        salary = _money(re.search(r"salary of (\$[\d,]+)", q.text).group(1))
        if "hourly rate" in q.text:
            hours = Fraction(re.search(r"for a ([\d.]+)-hour week", q.text).group(1))
            assert _close(q, salary / 52 / hours), (seed, q.text)
            seen.add("hourly")
        elif "allowance" in q.text:
            allowance = _money(re.search(r"allowance of (\$[\d,]+)", q.text).group(1))
            assert _close(q, salary / 26 + 2 * allowance), (seed, q.text)
            seen.add("allowance")
        else:
            pct = Fraction(re.search(r"pay rise of ([\d.]+)%", q.text).group(1))
            assert _close(q, salary * (100 + pct) / 100 / 12), (seed, q.text)
            seen.add("rise")
    assert seen == {"hourly", "allowance", "rise"}
    for seed in range(300):
        q = _q("salary-conversion-pay", "hard", seed)
        if "guaranteed" in q.text:
            low, high = (_money(m) for m in re.findall(r"\$([\d.]+) for each", q.text))
            first = int(re.search(r"first (\d+)", q.text).group(1))
            minimum = _money(re.search(r"at least (\$[\d,]+)", q.text).group(1))
            units = int(re.search(r"completes (\d+)", q.text).group(1))
            piece = low * min(units, first) + high * max(units - first, 0)
            assert Fraction(q.answer_text) == max(piece, minimum), (seed, q.text)
            seen.add("piece")
        else:
            salary = _money(re.search(r"salary of (\$[\d,]+)", q.text).group(1))
            rate = _money(re.search(r"pays (\$[\d.,]+) an hour", q.text).group(1))
            hours, weeks = (
                int(v) for v in re.search(r"for (\d+) hours a week, but only for (\d+) weeks", q.text).groups()
            )
            assert _close(q, rate * hours * weeks - salary), (seed, q.text)
            seen.add("year")
    assert {"piece", "year"} <= seen


def test_salary_exam_from_the_text():
    for seed in range(100):
        q = _q("salary-conversion-pay", "exam", seed)
        salary = _money(re.search(r"salary of (\$[\d,]+)", q.text).group(1))
        rate = _money(re.search(r"pays (\$[\d.,]+) an hour", q.text).group(1))
        extra = int(re.search(r"expect to work (\d+) hours", q.text).group(1))
        weekly_a = salary / 52
        weekly_b = 38 * rate + Fraction(3, 2) * rate * extra
        keys = [Fraction(p.answer_text) for p in q.parts]
        assert abs(keys[0] - weekly_a) <= Fraction(1, 200)
        assert abs(keys[1] - weekly_a / 38) <= Fraction(1, 200)
        assert abs(keys[2] - weekly_b) <= Fraction(1, 200)
        assert abs(keys[3] - 52 * (weekly_b - weekly_a)) <= 1


def test_recurrence_keys_from_the_text():
    seen = set()
    for seed in range(200):
        for tier in ("easy", "medium"):
            q = _q("compound-recurrence-model", tier, seed)
            v0 = _amount(re.search(r"(?:with|holds|has) (\$?[\d,]+)", q.text).group(1))
            pct = Fraction(re.search(r"([\d.]+)%", q.text).group(1))
            grows = "interest" in q.text or "earns" in q.text
            r = 1 + pct / 100 if grows else 1 - pct / 100
            n = int(re.search(r"after (\d+) ", q.text.split("What is")[1]).group(1))
            d = Fraction(0)
            if tier == "medium":
                sentence = re.search(
                    r"(?:after that loss|after the interest is added|Straight after that),([^.]*)\.", q.text
                ).group(1)
                d = _amount(re.search(r"(\$?[\d,]+)", sentence).group(1))
                if re.search(r"withdraws|paid out|drawn off|removed", sentence):
                    d = -d
            v = v0
            for _ in range(n):
                v = r * v + d
            assert _close(q, v), (seed, q.text, q.answer_text, float(v))
            seen.add(tier)
    assert seen == {"easy", "medium"}


def _amount(text):
    return Fraction(text.replace("$", "").replace(",", ""))


def test_recurrence_exam_from_the_text():
    for seed in range(100):
        q = _q("compound-recurrence-model", "exam", seed)
        v0 = _amount(re.search(r"with (\$[\d,]+)", q.text).group(1))
        p = Fraction(re.search(r"interest at ([\d.]+)%", q.text).group(1))
        d = _amount(re.search(r"deposits (\$[\d,]+)", q.text).group(1))
        r = 1 + p / 100
        values = [v0]
        for _ in range(40):
            values.append(r * values[-1] + d)
        assert abs(Fraction(q.parts[1].answer_text) - values[3]) <= Fraction(1, 200)
        assert abs(Fraction(q.parts[2].answer_text) - (values[3] - v0 - 3 * d)) <= Fraction(1, 200)
        target = _amount(re.search(r"reach (\$[\d,]+)", q.parts[3].prompt).group(1))
        assert int(q.parts[3].answer_text) == next(n for n, v in enumerate(values) if v >= target)
        assert (
            grade(
                SKILLS["compound-recurrence-model"],
                q,
                {
                    "a": f"{plain_r(r)}V + {d}",
                    "b": q.parts[1].answer_text,
                    "c": q.parts[2].answer_text,
                    "d": q.parts[3].answer_text,
                },
            )
            .parts["a"]
            .correct
        )


def plain_r(r):
    return f"{float(r):g}"


def test_sample_space_medium_events_by_brute_force():
    for seed in range(150):
        q = _q("build-sample-space", "medium", seed)
        if m := re.search(r"two fair (four|six)-sided dice", q.text):
            a = b = 4 if m.group(1) == "four" else 6
        else:
            a, b = (
                int(v)
                for v in re.search(r"numbered 1 to (\d+) and then a spinner numbered 1 to (\d+)", q.text).groups()
            )
        k = int(re.search(r"(?:add to|multiply to|differ by|is) (\d+)\. Write", q.text).group(1))
        if "add to" in q.text:
            ok = lambda x, y: x + y == k  # noqa: E731
        elif "multiply to" in q.text:
            ok = lambda x, y: x * y == k  # noqa: E731
        elif "differ by" in q.text:
            ok = lambda x, y: abs(x - y) == k  # noqa: E731
        else:
            ok = lambda x, y: max(x, y) == k  # noqa: E731
        expected = sorted((x, y) for x in range(1, a + 1) for y in range(1, b + 1) if ok(x, y))
        assert sorted(tuple(p) for p in q.answer) == expected, (seed, q.text)


def test_sample_space_easy_is_the_full_product():
    for seed in range(40):
        q = _q("build-sample-space", "easy", seed)
        firsts = {p[0] for p in q.answer}
        seconds = {p[1] for p in q.answer}
        assert len(q.answer) == len(firsts) * len(seconds) and len(set(map(tuple, q.answer))) == len(q.answer)


def test_sample_space_hard_lists():
    seen = set()
    for seed in range(200):
        q = _q("build-sample-space", "hard", seed)
        got = sorted(tuple(p) if isinstance(p, list) else p for p in q.answer)
        if "sum of at least" in q.text:
            sides = int(re.search(r"numbered 1 to (\d+)", q.text).group(1))
            t, d = (
                int(v) for v in re.search(r"sum of at least (\d+) or a difference of exactly (\d+)", q.text).groups()
            )
            expected = sorted(
                (x, y) for x in range(1, sides + 1) for y in range(1, sides + 1) if x + y >= t or abs(x - y) == d
            )
            seen.add("dice")
        elif "coin" in q.text:
            outcomes = ["".join(o) for o in product("HT", repeat=3)]
            if "at least 2 tails" in q.text:
                expected = [o for o in outcomes if o.count("T") >= 2]
            elif "no more than 1 head" in q.text:
                expected = [o for o in outcomes if o.count("H") <= 1]
            else:
                assert "first two tosses different" in q.text
                expected = [o for o in outcomes if o[0] != o[1]]
            expected = sorted(expected)
            seen.add("coin")
        else:
            a, b = (
                int(v)
                for v in re.search(r"numbered 1 to (\d+) and then a spinner numbered 1 to (\d+)", q.text).groups()
            )
            k = int(re.search(r"shows (\d+)", q.text).group(1))
            want = 1 if "exactly one" in q.text else 0
            expected = sorted((x, y) for x in range(1, a + 1) for y in range(1, b + 1) if (x == k) + (y == k) == want)
            seen.add("shows")
        assert got == expected, (seed, q.text)
    assert seen == {"dice", "coin", "shows"}


def test_sample_space_exam_probabilities():
    for seed in range(100):
        q = _q("build-sample-space", "exam", seed)
        n = int(q.parts[0].answer_text)
        event = q.parts[1].answer
        assert Fraction(q.parts[2].answer_text) == Fraction(len(event), n)
        assert Fraction(q.parts[3].answer_text) == 1 - Fraction(len(event), n)


_FACTS = {
    "1 foot = 12 inches": ("foot", "inch", 12),
    "1 yard = 3 feet": ("yard", "foot", 3),
    "1 mile = 1760 yards": ("mile", "yard", 1760),
}
_PLURALS = {"inches": "inch", "feet": "foot", "yards": "yard", "miles": "mile"}


def test_imperial_easy_from_the_printed_factor():
    for seed in range(150):
        q = _q("convert-imperial-length", "easy", seed)
        value = Fraction(re.search(r"as ([\d.]+) (inches|feet|yards|miles)", q.text).group(1))
        have = _PLURALS[re.search(r"as [\d.]+ (inches|feet|yards|miles)", q.text).group(1)]
        want = _PLURALS[re.search(r"how many (inches|feet|yards|miles)", q.text).group(1)]
        fact = next(f for f in _FACTS if f in q.text)
        big, small, factor = _FACTS[fact]
        expected = value * factor if (have, want) == (big, small) else value / factor
        assert Fraction(q.answer_text) == expected, (seed, q.text)


def test_imperial_medium_and_hard_from_the_text():
    for seed in range(150):
        q = _q("convert-imperial-length", "medium", seed)
        value = Fraction(re.search(r"(?:as|is) ([\d.]+) (feet|yards|metres)", q.text).group(1))
        unit = re.search(r"(?:as|is) [\d.]+ (feet|yards|metres)", q.text).group(1)
        cm = {"feet": value * 12 * Fraction("2.54"), "yards": value * 36 * Fraction("2.54"), "metres": value * 100}[
            unit
        ]
        expected = cm / 100 if unit != "metres" else cm / Fraction("2.54") / 12
        assert _close(q, expected), (seed, q.text, q.answer_text)
    for seed in range(150):
        q = _q("convert-imperial-length", "hard", seed)
        if "nautical" in q.text:
            nm = Fraction(re.search(r"is (\d+) nautical", q.text).group(1))
            expected = nm * Fraction("1.852") / Fraction("1.609")
        else:
            m1, y1, m2, y2 = (
                int(v)
                for v in re.search(
                    r"first is (\d+) miles? (\d+) yards and the second is (\d+) miles? (\d+) yards", q.text
                ).groups()
            )
            expected = Fraction((m1 + m2) * 1760 + y1 + y2, 1760) * Fraction("1.609")
        assert _close(q, expected), (seed, q.text, q.answer_text)


def test_imperial_exam_from_the_text():
    for seed in range(100):
        q = _q("convert-imperial-length", "exam", seed)
        nm, mi = (
            int(v)
            for v in re.search(
                r"(?:leg is|logs two legs of a voyage:) (\d+) nautical miles(?: from a marina to an island, then|"
                r" and the second leg is) (\d+) miles",
                q.text,
            ).groups()
        )
        a, b = nm * Fraction("1.852"), mi * Fraction("1.609")
        keys = [Fraction(p.answer_text) for p in q.parts]
        assert abs(keys[0] - a) <= Fraction(1, 200) and abs(keys[1] - b) <= Fraction(1, 200)
        assert abs(keys[2] - (a + b)) <= Fraction(1, 200)
        assert abs(keys[3] - (a + b) / Fraction("1.852")) <= Fraction(1, 20)


def test_speed_keys_from_the_text():
    seen = set()
    for seed in range(200):
        q = _q("speed-distance-time-average-speed", "easy", seed)
        if "How many minutes" in q.text:
            d, v = (Fraction(x) for x in re.search(r"([\d.]+) km at an average speed of (\d+) km/h", q.text).groups())
            assert int(q.answer_text) == d / v * 60, (seed, q.text)
            seen.add("time")
        elif "How far" in q.text:
            v = Fraction(re.search(r"speed of (\d+) km/h", q.text).group(1))
            h = re.search(r"for (?:(\d+) hours?)? ?(?:(\d+) minutes)?\.", q.text)
            hours, minutes = int(h.group(1) or 0), int(h.group(2) or 0)
            assert _close(q, v * (hours * 60 + minutes) / 60), (seed, q.text)
            seen.add("distance")
        else:
            d = Fraction(re.search(r" ([\d.]+) km in", q.text).group(1))
            h = re.search(r"in (?:(\d+) hours?)? ?(?:(\d+) minutes)?\.", q.text)
            hours, minutes = int(h.group(1) or 0), int(h.group(2) or 0)
            assert _close(q, d * 60 / (hours * 60 + minutes)), (seed, q.text)
            seen.add("speed")
    assert seen == {"time", "distance", "speed"}
    for seed in range(200):
        q = _q("speed-distance-time-average-speed", "medium", seed)
        if "returns" in q.text:
            d1, v1, d2, v2 = (
                Fraction(x)
                for x in re.search(
                    r"(\d+) km to a town at (\d+) km/h and returns along the same road, (\d+) km, at (\d+) km/h", q.text
                ).groups()
            )
        else:
            d1, v1, d2, v2 = (
                Fraction(x)
                for x in re.search(r"(\d+) km at (\d+) km/h and then (\d+) km at (\d+) km/h", q.text).groups()
            )
        assert _close(q, (d1 + d2) / (d1 / v1 + d2 / v2)), (seed, q.text)


def test_speed_hard_and_exam_from_the_text():
    seen = set()
    for seed in range(200):
        q = _q("speed-distance-time-average-speed", "hard", seed)
        if "To average" in q.text:
            d1, total, v1, target = (
                Fraction(x)
                for x in re.search(
                    r"first (\d+) km of a (\d+) km trip at (\d+) km/h\. To average ([\d.]+) km/h", q.text
                ).groups()
            )
            d2 = Fraction(re.search(r"the last (\d+) km", q.text).group(1))
            assert d1 + d2 == total
            assert Fraction(q.answer_text) == d2 / (total / target - d1 / v1), (seed, q.text)
            seen.add("target")
        else:
            d1, v1, d2, v2, stop, d3, v3 = (
                Fraction(x)
                for x in re.search(
                    r"(\d+) km at (\d+) km/h, then (\d+) km at (\d+) km/h\. \w+ then rests for (\d+) minutes and finishes with a further (\d+) km at (\d+) km/h",
                    q.text,
                ).groups()
            )
            time = d1 / v1 + d2 / v2 + stop / 60 + d3 / v3
            assert _close(q, (d1 + d2 + d3) / time), (seed, q.text)
            seen.add("stop")
    for seed in range(100):
        q = _q("speed-distance-time-average-speed", "exam", seed)
        d1, v1, stop, d2, v2 = (
            Fraction(x)
            for x in re.search(
                r"(\d+) km at (\d+) km/h, rests for (\d+) minutes, and then goes a further (\d+) km at (\d+) km/h",
                q.text,
            ).groups()
        )
        total = d1 / v1 + d2 / v2 + stop / 60
        assert abs(Fraction(q.parts[2].answer_text) - total) <= Fraction(1, 200)
        assert abs(Fraction(q.parts[3].answer_text) - (d1 + d2) / total) <= Fraction(1, 20)
    assert seen == {"target", "stop"}


def test_slope_keys_from_the_drawn_figure():
    seen = set()
    for tier in TIERS:
        for seed in range(80):
            q = _q("interpret-slope-in-context", tier, seed)
            assert q.figure and not figures.validate(q.figure)
            m, c = read_line(q.figure)
            if tier != "hard":
                assert _close(q, m), (seed, q.text)
                assert (m.denominator == 1) == (tier == "easy")
            elif match := re.search(r"for (\d+) more", q.text):
                assert _close(q, m * int(match.group(1)))
                seen.add("extra")
            elif match := re.search(r"extra \$([\d,.]+)", q.text):
                assert int(q.answer_text) == Fraction(match.group(1).replace(",", "")) / m
                seen.add("count")
            else:
                n = int(re.search(r"cost be for (\d+)", q.text).group(1))
                assert _close(q, m * n + c)
                seen.add("predict")
            read = figures.get_reader(q.topic)(q)
            assert grade(SKILLS["interpret-slope-in-context"], q, read).correct, (seed, q.text, read)
    assert seen == {"extra", "count", "predict"}


def test_slope_exam_from_the_drawn_figure():
    for seed in range(60):
        q = _q("interpret-slope-in-context", "exam", seed)
        assert q.figure and len(q.figure["lines"]) == 2 and not figures.validate(q.figure)
        (m_a, _), (m_b, _) = read_line(q.figure, 0), read_line(q.figure, 1)
        x, _y = read_intersection(q.figure, 0, 1)
        keys = [Fraction(p.answer_text) for p in q.parts]
        assert keys == [m_a, m_b, m_a - m_b, x] and m_a > m_b


# -- classify-data (deepened) and divide-quantity-in-ratio (deepened) ----------------------------------------------------------


_FIXTURES = {
    "the postcode of each customer's address": "Nominal categorical",
    "the finishing place (1st, 2nd, 3rd and so on) of each runner in a race": "Ordinal categorical",
    "each person's age in completed years": "Discrete numerical",
    "the height of each student, recorded to the nearest centimetre": "Continuous numerical",
    "the number of siblings each student has": "Discrete numerical",
    "the colour of each car in a car park": "Nominal categorical",
}


def test_classify_pool_matches_hand_labelled_fixtures():
    from mathgen.generators.r3j_classify_build import POOL

    labels = {phrase: cls for phrase, cls, _ in POOL}
    assert len(POOL) >= 60
    for phrase, expected in _FIXTURES.items():
        assert labels[phrase] == expected


@pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
def test_classify_data_sources(tier):
    skill = SKILLS["classify-data"]
    source = next(i for i, s in enumerate(skill.sources) if s.difficulty == tier)
    from mathgen.engine import make_for_skill

    for seed in range(1, 60):
        q = make_for_skill(skill, seed=seed, source=source)
        if q.parts:
            full = grade(skill, q, q.answer_text)
            assert full.correct and full.marks == full.max_marks
            counts = [int(p.answer_text) for p in q.parts[:3]]
            assert sum(counts) == 6 and len(set(counts)) == 3
            continue
        assert grade(skill, q, q.answer_text).correct
        for text in q.distractors:
            assert not grade(skill, q, text).correct
        if "Which of these" not in q.text:
            phrase = re.search(r"(?:records|is|on) (.*?)\. (?:How|What|Classify)", q.text).group(1) if False else None
            del phrase
            assert q.answer_text in (
                "Nominal categorical",
                "Ordinal categorical",
                "Discrete numerical",
                "Continuous numerical",
            )


def test_ratio_three_shares_chain():
    skill = SKILLS["divide-quantity-in-ratio"]
    for seed in range(100):
        q = make("exd_x_r3j_ratio_shares", seed=seed, difficulty="exam")
        a, b, c = (int(v) for v in re.search(r"ratio (\d+) : (\d+) : (\d+)", q.text).groups())
        diff = _amount(re.search(r"is (\$[\d,]+) more", q.text).group(1))
        unit = diff / (c - a)
        keys = [Fraction(p.answer_text) for p in q.parts]
        assert keys[0] == unit and keys[1] == unit * (a + b + c) and keys[2] == unit * b
        assert abs(keys[3] - Fraction(100 * a, a + b + c)) <= Fraction(1, 20)
        assert grade(skill, q, q.answer_text).correct
