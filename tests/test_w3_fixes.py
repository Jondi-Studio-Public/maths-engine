"""Tests the replacement and extra tiers in `mathgen/generators/w3_fixes_build.py`: percentage change, classify data
(easy), arithmetic sequences, the ESTs-and-LSTs critical-path skill and bearings with general legs.

The oracles read the numbers back out of the question TEXT and re-derive the key independently (Fractions, an exhaustive
search, an earliest/latest-start pass over the table, or `math` trigonometry for bearings).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.w3_fixes_build as mod
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects
from mathgen.generators.r3j_classify_build import POOL
from tests import w3_support as w3
from tests import w7b_support as sup


def tenth(x):
    return Fraction(int(x * 10 + Fraction(1, 2)), 10)


def test_catalogue_has_every_module_misconception():
    w3.check_catalogue(mod)


CASES = [
    ("percentage-change", "w3_percentage_change", ("easy", "medium", "hard")),
    ("classify-data", "w3_classify_easy", ("easy",)),
    ("arithmetic-sequences", "w3_arith_recognise", ("easy", "medium", "hard")),
    ("senior-use-ests-and-lsts-to-locate-the-critical", "w3_ests_lsts", ("easy", "medium", "hard")),
    ("senior-solve-problems-involving-bearings", "w3_bearings_general", ("medium", "hard")),
]


def check_single_choice(skill, topic, tier, seeds=80):
    """The key grades correct, every distractor grades wrong and carries a catalogue id, text is clean, output is stable."""
    for seed in range(seeds):
        q = sup.single(topic, tier, seed)
        assert not q.parts
        assert grade(skill, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= 2 and len(set(q.distractors)) == len(q.distractors)
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == sup.single(topic, tier, seed).to_json()


@pytest.mark.parametrize("skill_id,topic,tiers", CASES)
class TestShared:
    def test_wired_into_the_skill(self, skill_id, topic, tiers):
        sk = sup.skill_of(skill_id)
        assert {s.difficulty for s in sk.sources if s.topic == topic} == set(tiers)
        have = {m.id for m in sk.misconceptions}
        for ident, _desc, area in mod.MISCONCEPTIONS:
            if area == skill_id:
                assert ident in have

    def test_key_accepted_and_distractors_rejected(self, skill_id, topic, tiers):
        for tier in tiers:
            check_single_choice(sup.skill_of(skill_id), topic, tier)

    def test_pools(self, skill_id, topic, tiers):
        for tier in tiers:
            qs = [sup.single(topic, tier, seed) for seed in range(60)]
            assert len({(q.text, tuple(sorted([q.answer_text, *q.distractors]))) for q in qs}) >= 20, tier

    def test_options_are_distinct_and_the_key_is_among_few(self, skill_id, topic, tiers):
        for tier in tiers:
            for seed in range(60):
                q = sup.single(topic, tier, seed)
                options = [q.answer_text, *q.distractors]
                assert len(set(options)) == len(options) >= 3, (tier, seed, options)


# -- percentage change ------------------------------------------------------------------------------------------------


def numbers(text):
    return [Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


def test_percentage_change_keys_and_spread():
    answers = set()
    for seed in range(200):
        q = sup.single("w3_percentage_change", "easy", seed)
        orig, final = (Fraction(x) for x in re.search(r"from \$?(\d+) to \$?(\d+)", q.text).groups())
        assert Fraction(q.answer_text) == (final - orig) / orig * 100, (seed, q.text)
        answers.add(q.answer_text)
    assert len(answers) >= 15
    for seed in range(200):
        q = sup.single("w3_percentage_change", "medium", seed)
        orig, final = (Fraction(x) for x in re.search(r"from \$?(\d+) to \$?(\d+)", q.text).groups())
        up = "increase" in q.text
        assert (final > orig) == up, (seed, q.text)
        assert Fraction(q.answer_text) == tenth(abs(final - orig) / orig * 100), (seed, q.text)


def test_percentage_change_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single("w3_percentage_change", "hard", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "increased by" in t:
            first, second = (int(x) for x in re.search(r"increased by (\d+)% .* decreased by (\d+)%", t).groups())
            net = Fraction((100 + first) * (100 - second), 100) - 100
            assert ans == tenth(abs(net)), (seed, t)
            assert ("increase" if net > 0 else "decrease") in t
            seen.add("net")
        elif "must the new price then fall" in t:
            a = int(re.search(r"rises by (\d+)%", t).group(1))
            assert ans == tenth(Fraction(a, 100 + a) * 100), (seed, t)
            seen.add("undo")
        else:
            r1, r2 = (Fraction(x) for x in re.search(r"from ([\d.]+)% to ([\d.]+)%", t).groups())
            assert ans == tenth((r2 - r1) / r1 * 100), (seed, t)
            seen.add("rate")
    assert seen == {"net", "undo", "rate"}


# -- classify data ----------------------------------------------------------------------------------------------------

_CLASS = {"Nominal categorical": "Categorical", "Ordinal categorical": "Categorical"}
_BY_PHRASE = {phrase: _CLASS.get(cls, cls) for phrase, cls, _twist in POOL}


def test_classify_keys():
    seen = set()
    for seed in range(200):
        q = sup.single("w3_classify_easy", "easy", seed)
        t = q.text
        if t.startswith("Which of these is an example of"):
            asked = re.search(r"example of (.*) data\?", t).group(1)
            options = [q.answer_text, *q.distractors]
            classes = [_BY_PHRASE[o[0].lower() + o[1:]] for o in options]
            assert classes[0].lower() == asked and classes.count(classes[0]) == 1, (seed, t, options)
            seen.add("which")
        else:
            phrase = next(p for p in _BY_PHRASE if p in t)
            assert q.answer_text == _BY_PHRASE[phrase], (seed, t)
            assert sorted([q.answer_text, *q.distractors]) == sorted(
                ["Categorical", "Discrete numerical", "Continuous numerical"]
            )
            seen.add("classify")
        assert not any(twist for p, c, twist in POOL if p in t)
    assert seen == {"which", "classify"}


# -- arithmetic sequences ---------------------------------------------------------------------------------------------


def seq(text):
    return [int(x) for x in text.split(", ")]


def diffs(terms):
    return [terms[i + 1] - terms[i] for i in range(len(terms) - 1)]


def test_arithmetic_easy_has_exactly_one_arithmetic_option():
    for seed in range(200):
        q = sup.single("w3_arith_recognise", "easy", seed)
        options = [seq(o) for o in (q.answer_text, *q.distractors)]
        flags = [len(set(diffs(o))) == 1 for o in options]
        assert flags == [True, False, False, False], (seed, options)


def test_arithmetic_medium_statements():
    for seed in range(200):
        q = sup.single("w3_arith_recognise", "medium", seed)
        terms = seq(re.search(r"sequence ([-\d, ]+)\. Which", q.text).group(1))
        d = diffs(terms)
        if len(set(d)) == 1:
            assert q.answer_text == f"It is arithmetic, with a common difference of {d[0]}.", (seed, q.text)
        else:
            brk = next(i for i in range(1, len(d)) if d[i] != d[0]) + 1
            assert q.answer_text == f"It is not arithmetic: the difference changes after term {brk}.", (seed, q.text)
        assert len(set(diffs(terms))) >= 1


def test_arithmetic_hard_solves_for_k():
    for seed in range(200):
        q = sup.single("w3_arith_recognise", "hard", seed)
        a, b, c = (int(x) for x in re.search(r"k \+ (\d+), 2k \+ (\d+) and k \+ (\d+)", q.text).groups())
        solutions = [k for k in range(-100, 101) if 2 * (2 * k + b) == (k + a) + (k + c)]
        assert solutions == [int(q.answer_text)], (seed, q.text)


# -- activity networks ------------------------------------------------------------------------------------------------


def parse_activities(text):
    rows = {}
    for m in re.finditer(r"([A-G]) \(duration (\d+), after ([A-G, ]+|none)(?:, EST (\d+), LST (\d+))?\)", text):
        name, dur, pre, est, lst = m.groups()
        rows[name] = (int(dur), [] if pre == "none" else pre.split(", "), est and int(est), lst and int(lst))
    return rows


def passes(rows):
    est = {}
    for n in rows:  # activities are listed with predecessors first
        est[n] = max((est[p] + rows[p][0] for p in rows[n][1]), default=0)
    total = max(est[n] + rows[n][0] for n in rows)
    succ = {n: [m for m in rows if n in rows[m][1]] for n in rows}
    lst = {}
    for n in reversed(list(rows)):
        lst[n] = min((lst[s] for s in succ[n]), default=total) - rows[n][0]
    return est, lst, total, succ


def test_ests_easy_key_is_the_zero_float_row():
    for seed in range(200):
        q = sup.single("w3_ests_lsts", "easy", seed)
        rows = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"Activity ([A-D]): EST (\d+), LST (\d+)", q.text)}
        critical = [n for n, (e, l) in rows.items() if e == l]
        assert critical == [q.answer_text], (seed, q.text)


def test_ests_medium_key_is_the_longest_path():
    for seed in range(200):
        q = sup.single("w3_ests_lsts", "medium", seed)
        rows = parse_activities(q.text)
        est, lst, total, succ = passes(rows)
        for n, (_d, _p, e, l) in rows.items():  # the printed table agrees with an independent pass
            assert (e, l) == (est[n], lst[n]), (seed, q.text)
        path = q.answer_text.split("-")
        assert sum(rows[n][0] for n in path) == total
        assert all(est[n] == lst[n] for n in path)
        assert all(b in succ[a] for a, b in zip(path, path[1:]))
        assert [n for n in rows if est[n] == lst[n]] == path


def test_ests_hard_key_is_the_latest_start_time():
    for seed in range(200):
        q = sup.single("w3_ests_lsts", "hard", seed)
        rows = parse_activities(q.text)
        est, lst, total, _succ = passes(rows)
        asked = re.search(r"of activity ([A-G]) if", q.text).group(1)
        assert total == int(re.search(r"finish is time (\d+)", q.text).group(1))
        assert int(q.answer_text) == lst[asked], (seed, q.text)
        assert lst[asked] != est[asked]


# -- bearings ---------------------------------------------------------------------------------------------------------


def comp(d, theta):
    return d * math.cos(math.radians(theta)), d * math.sin(math.radians(theta))


def test_bearings_medium_components():
    for seed in range(200):
        q = sup.single("w3_bearings_general", "medium", seed)
        d, theta = (int(x) for x in re.search(r"walks (\d+) km on a bearing of (\d+)°", q.text).groups())
        n, e = comp(d, theta)
        want = n if "north" in q.text else e
        assert abs(float(q.answer_text) - want) <= 0.0501, (seed, q.text, want)
        assert theta % 90 != 0


def test_bearings_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single("w3_bearings_general", "hard", seed)
        t = q.text
        if "due east" in t:
            d1, t1, t2 = (
                int(x) for x in re.search(r"sails (\d+) km on a bearing of (\d+)°.*bearing of (\d+)°", t).groups()
            )
            d2 = float(q.answer_text)
            north = d1 * math.cos(math.radians(t1)) + d2 * math.cos(math.radians(t2))
            assert abs(north) <= 0.05 * abs(math.cos(math.radians(t2))) + 1e-9, (seed, t, north)
            seen.add("east")
            continue
        d1, t1, d2, t2 = (
            int(x)
            for x in re.search(
                r"sails (\d+) km on a bearing of (\d+)° and then (\d+) km on a bearing of (\d+)°", t
            ).groups()
        )
        n = d1 * math.cos(math.radians(t1)) + d2 * math.cos(math.radians(t2))
        e = d1 * math.sin(math.radians(t1)) + d2 * math.sin(math.radians(t2))
        if "How far is the ship" in t:
            assert abs(float(q.answer_text) - math.hypot(n, e)) <= 0.0501, (seed, t)
            seen.add("distance")
        else:
            back = (math.degrees(math.atan2(e, n)) + 180) % 360
            assert abs(int(q.answer_text) - back) <= 0.5001, (seed, t, back)
            seen.add("back")
    assert seen == {"east", "distance", "back"}
