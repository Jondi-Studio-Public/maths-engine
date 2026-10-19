"""Tests the w3 depreciation skills: `flat-rate-depreciation`, `reducing-balance-depreciation` and
`compare-flat-and-reducing-depreciation` (topics `w3_flat_depreciation`, `w3_reducing_depreciation`,
`w3_compare_depreciation`, and their exam chains).

The oracles read the price, rates, years and values back out of the question TEXT and re-derive each key with Fraction
arithmetic and a year-by-year schedule, never from the generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen.generators.w3_depreciation_build as mod
from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

FLAT, FLAT_X = "w3_flat_depreciation", "exd_x_w3_flat_depreciation"
RED, RED_X = "w3_reducing_depreciation", "exd_x_w3_reducing_depreciation"
CMP, CMP_X = "w3_compare_depreciation", "exd_x_w3_compare_depreciation"
SK_FLAT = sup.skill_of("flat-rate-depreciation")
SK_RED = sup.skill_of("reducing-balance-depreciation")
SK_CMP = sup.skill_of("compare-flat-and-reducing-depreciation")

_MONEY = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def dollars(text):
    return [Fraction(m.replace(",", "")) for m in _MONEY.findall(text)]


def percents(text):
    return [Fraction(m) for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]


def years_in(text):
    return [int(m) for m in re.findall(r"(\d+) years?", text)]


def cents(x: Fraction) -> Fraction:
    """Round half up to the cent with integer arithmetic."""
    n = x * 100
    return Fraction(int(n + Fraction(1, 2)), 100)


def flat_value(price, rate, n):
    value = price
    for _ in range(n):
        value -= price * rate / 100
    return value


def red_value(price, rate, n):
    value = price
    for _ in range(n):
        value -= value * rate / 100
    return value


def test_catalogue_has_every_module_misconception():
    for ident, description, _area in mod.MISCONCEPTIONS:
        assert CATALOGUE[ident].description == description


@pytest.mark.parametrize(
    "skill,topic,exam",
    [(SK_FLAT, FLAT, FLAT_X), (SK_RED, RED, RED_X), (SK_CMP, CMP, CMP_X)],
    ids=["flat", "reducing", "compare"],
)
class TestShared:
    def test_wiring(self, skill, topic, exam):
        sup.check_wiring(skill, topic, exam, "number")
        steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
        assert steps == sorted(steps) and len(set(steps)) == 4

    @pytest.mark.parametrize("tier", sup.TIERS)
    def test_key_accepted_and_distractors_rejected(self, skill, topic, exam, tier):
        sup.check_single_tier(skill, topic, tier)

    def test_pools(self, skill, topic, exam):
        sup.check_pools(topic)

    def test_answers_are_not_guessable(self, skill, topic, exam):
        for tier in sup.TIERS:
            spread = sup.answer_spread(topic, tier, 60)
            assert len(spread) >= 12, (tier, len(spread))
            assert max(spread.values()) <= 10, (tier, spread.most_common(3))

    def test_money_answers_have_two_places_and_distractors_match(self, skill, topic, exam):
        for tier in sup.TIERS:
            for seed in range(60):
                q = sup.single(topic, tier, seed)
                places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
                for d in q.distractors:
                    assert (len(d.split(".")[1]) if "." in d else 0) == places, (tier, seed, d, q.answer_text)


# -- flat rate --------------------------------------------------------------------------------------------------------


def test_flat_easy_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(FLAT, "easy", seed)
        t = q.text
        price, rate = dollars(t)[0], percents(t)[0]
        ans = Fraction(q.answer_text)
        if "each year, in dollars" in t:
            assert ans == price * rate / 100, (seed, t)
            seen.add("annual")
        elif "value after" in t:
            n = years_in(t)[-1]
            assert ans == flat_value(price, rate, n), (seed, t)
            seen.add("value")
        else:
            n = years_in(t)[-1]
            assert ans == price - flat_value(price, rate, n), (seed, t)
            seen.add("total")
    assert seen == {"annual", "value", "total"}


def test_flat_medium_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(FLAT, "medium", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "What percentage of the purchase price" in t:
            price, value = dollars(t)
            n = years_in(t)[0]
            assert (price - value) / n * 100 / price == ans, (seed, t)
            seen.add("rate")
        elif "After how many years will its value be" in t:
            price, target = dollars(t)
            rate = percents(t)[0]
            assert (price - target) / (price * rate / 100) == ans, (seed, t)
            seen.add("years")
        elif "straight-line" in t:
            price, scrap = dollars(t)
            assert (price - scrap) / years_in(t)[0] == ans, (seed, t)
            seen.add("scrap")
        else:
            m = re.search(r"V = (\d+) - (\d+)n", t)
            n = years_in(t)[-1]
            assert int(m.group(1)) - int(m.group(2)) * n == ans, (seed, t)
            seen.add("rule")
    assert seen == {"rate", "years", "scrap", "rule"}


def test_flat_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(FLAT, "hard", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "purchase price, in dollars" in t:
            va, vb = dollars(t)
            a, b = years_in(t)[:2]
            annual = (va - vb) / (b - a)
            assert va + annual * a == ans and vb + annual * b == ans, (seed, t)
            seen.add("price")
        elif "first year its value falls below" in t:
            price, limit = dollars(t)
            rate = percents(t)[0]
            n = 0
            while flat_value(price, rate, n) >= limit:
                n += 1
            assert n == ans, (seed, t)
            seen.add("first-below")
        elif "will its value be zero" in t:
            price, value = dollars(t)
            a = years_in(t)[0]
            assert price / ((price - value) / a) == ans, (seed, t)
            seen.add("zero")
        else:
            remaining, n = percents(t)[0], years_in(t)[0]
            assert (100 - remaining) / n == ans, (seed, t)
            seen.add("share")
    assert seen == {"price", "first-below", "zero", "share"}


def test_flat_hard_boundary_case_is_asked():
    """A target met exactly at the end of year n needs n + 1: the boundary distractor must be present and wrong."""
    hits = 0
    for seed in range(300):
        q = sup.single(FLAT, "hard", seed)
        if "first year its value falls below" not in q.text:
            continue
        price, limit = dollars(q.text)
        rate = percents(q.text)[0]
        n = int(q.answer_text)
        if flat_value(price, rate, n - 1) == limit:
            hits += 1
            assert str(n - 1) in q.distractors, (seed, q.text)
    assert hits >= 10


def test_flat_exam_keys():
    for seed in range(100):
        q = sup.exam(FLAT_X, seed)
        price, rate = dollars(q.stem)[0], percents(q.stem)[0]
        a, b, c, d = q.parts
        annual = price * rate / 100
        assert Fraction(a.answer_text) == annual
        n = years_in(b.prompt)[0]
        assert Fraction(b.answer_text) == flat_value(price, rate, n)
        target = dollars(c.prompt)[0]
        k = 0
        while flat_value(price, rate, k) != target:
            k += 1
        assert int(c.answer_text) == k
        limit = dollars(d.prompt)[0]
        j = 0
        while flat_value(price, rate, j) >= limit:
            j += 1
        assert int(d.answer_text) == j


def test_flat_exam_chain_shape_and_follow_through():
    sup.check_exam_chain(SK_FLAT, FLAT_X, [1, 2, 2, 2], ["number"] * 4)

    def wrong(q, canon):
        price = dollars(q.stem)[0]
        bad = Fraction(q.parts[0].answer_text) + 10
        n = years_in(q.parts[1].prompt)[0]
        return {"a": f"{bad:.2f}", "b": f"{float(price - bad * n):.2f}"}

    assert sup.check_follow_through(SK_FLAT, FLAT_X, "b", wrong) >= 30


# -- reducing balance -------------------------------------------------------------------------------------------------


def test_reducing_easy_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(RED, "easy", seed)
        t = q.text
        price, rate = dollars(t)[0], percents(t)[0]
        ans = Fraction(q.answer_text)
        if "first year" in t:
            assert ans == price * rate / 100, (seed, t)
            seen.add("amount")
        elif "after 1 year" in t:
            assert ans == red_value(price, rate, 1), (seed, t)
            seen.add("one")
        else:
            assert ans == red_value(price, rate, 2), (seed, t)
            seen.add("two")
    assert seen == {"amount", "one", "two"}


def test_reducing_medium_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(RED, "medium", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "annual rate of depreciation" in t:
            price, v1 = dollars(t)
            assert (price - v1) * 100 / price == ans, (seed, t)
            seen.add("rate")
        elif "given by V" in t:
            m = re.search(r"V = (\d+) × ([\d.]+)\^n", t)
            n = years_in(t)[-1]
            expected = cents(Fraction(m.group(1)) * Fraction(m.group(2)) ** n)
            assert ans == expected, (seed, t)
            seen.add("formula")
        elif "in total" in t:
            price, rate = dollars(t)[0], percents(t)[0]
            n = years_in(t)[-1]
            assert ans == price - cents(red_value(price, rate, n)), (seed, t)
            seen.add("total")
        else:
            price, rate = dollars(t)[0], percents(t)[0]
            n = years_in(t)[-1]
            assert ans == cents(red_value(price, rate, n)), (seed, t)
            seen.add("value")
    assert seen == {"rate", "formula", "total", "value"}


def test_reducing_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(RED, "hard", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "first be below" in t:
            price, target = dollars(t)
            rate = percents(t)[0]
            n = 0
            while red_value(price, rate, n) >= target:
                n += 1
            assert n == ans, (seed, t)
            seen.add("first-below")
        elif "at the same percentage rate" in t:
            price, v2 = dollars(t)
            rates = [r for r in range(1, 100) if red_value(price, Fraction(r), 2) == v2]
            assert rates == [ans], (seed, t, rates)
            seen.add("rate")
        elif "purchase price" in t:
            rate = percents(t)[0]
            n, value = years_in(t)[0], dollars(t)[0]
            assert red_value(ans, rate, n) == value, (seed, t)
            seen.add("price")
        else:
            price, rate = dollars(t)[0], percents(t)[0]
            n = int(re.search(r"during year (\d+)", t).group(1))
            assert ans == red_value(price, rate, n - 1) - red_value(price, rate, n), (seed, t)
            seen.add("year-amount")
    assert seen == {"first-below", "rate", "price", "year-amount"}


def test_reducing_exam_keys():
    for seed in range(100):
        q = sup.exam(RED_X, seed)
        price, rate = dollars(q.stem)[0], percents(q.stem)[0]
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == red_value(price, rate, 1)
        n = years_in(b.prompt)[0]
        vn = cents(red_value(price, rate, n))
        assert Fraction(b.answer_text) == vn
        assert Fraction(c.answer_text) == price - vn
        target = dollars(d.prompt)[0]
        k = 0
        while red_value(price, rate, k) >= target:
            k += 1
        assert int(d.answer_text) == k


def test_reducing_exam_chain_shape_and_follow_through():
    sup.check_exam_chain(SK_RED, RED_X, [1, 2, 1, 3], ["number"] * 4)

    def wrong(q, canon):
        price, rate = dollars(q.stem)[0], percents(q.stem)[0]
        bad = Fraction(q.parts[0].answer_text) - 100
        n = years_in(q.parts[1].prompt)[0]
        value = bad * (1 - rate / 100) ** (n - 1)
        if (value * 100).denominator != 1 and (value * 200).denominator == 1:
            return None  # an exact half-cent tie: the rounding rule of the follow-through target is not under test
        return {"a": f"{float(bad):.2f}", "b": f"{float(cents(value)):.2f}"}

    assert sup.check_follow_through(SK_RED, RED_X, "b", wrong) >= 30


# -- comparison -------------------------------------------------------------------------------------------------------


def test_compare_easy_and_medium_keys():
    for tier in ("easy", "medium"):
        for seed in range(120):
            q = sup.single(CMP, tier, seed)
            price, rate = dollars(q.text)[0], percents(q.text)[0]
            n = years_in(q.text)[-1]
            expected = cents(red_value(price, rate, n)) - flat_value(price, rate, n)
            assert Fraction(q.answer_text) == expected, (tier, seed, q.text)
            assert expected > 0


def test_compare_hard_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(CMP, "hard", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "reducing-balance value first greater" in t:
            price = dollars(t)[0]
            r_flat, r_red = percents(t)
            n = 1
            while red_value(price, r_red, n) <= flat_value(price, r_flat, n):
                n += 1
            assert n == ans, (seed, t)
            assert flat_value(price, r_flat, n) > 0
            seen.add("crossover")
        else:
            price, rate = dollars(t)[0], percents(t)[0]
            n = years_in(t)[-1]
            lost = price - red_value(price, rate, n)
            matched = lost / n / price * 100
            assert ans == Fraction(f"{float(matched):.1f}"), (seed, t)
            assert flat_value(price, matched, n) == red_value(price, rate, n)
            seen.add("equivalent")
    assert seen == {"crossover", "equivalent"}


def test_compare_exam_keys_and_shape():
    sup.check_exam_chain(SK_CMP, CMP_X, [1, 2, 1, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(CMP_X, seed)
        price = dollars(q.stem)[0]
        r_flat, r_red = percents(q.stem)
        a, b, c, d = q.parts
        n = years_in(a.prompt)[0]
        fa, fb = flat_value(price, r_flat, n), cents(red_value(price, r_red, n))
        assert Fraction(a.answer_text) == fa and Fraction(b.answer_text) == fb
        assert Fraction(c.answer_text) == fa - fb > 0
        k = 1
        while red_value(price, r_red, k) <= flat_value(price, r_flat, k):
            k += 1
        assert int(d.answer_text) == k and n < k
