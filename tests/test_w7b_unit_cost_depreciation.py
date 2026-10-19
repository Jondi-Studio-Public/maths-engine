"""Tests `unit-cost-depreciation` (topic `w7b_unit_cost_depreciation`) and its exam chain.

The oracles read the prices, rates, units and values back out of the question TEXT and re-derive the key with Fraction
arithmetic and a unit-by-unit schedule, never from the generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen.generators.w7b_unit_cost_depreciation_build  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

TOPIC = "w7b_unit_cost_depreciation"
EXAM = "exd_x_w7b_unit_cost_depreciation"
SKILL = sup.skill_of("unit-cost-depreciation")

_MONEY = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def _dollars(text: str):
    return [Fraction(m.replace(",", "")) for m in _MONEY.findall(text)]


def _units(text: str):
    return [int(m.replace(",", "")) for m in re.findall(r"([\d,]{3,}) (?:kilometres|hours)", text)]


def _answer(q) -> Fraction:
    return Fraction(q.answer_text)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "ucd-ignores-salvage" in {m.id for m in SKILL.misconceptions}
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_every_answer_is_a_two_decimal_amount_or_a_whole_number_of_units():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single(TOPIC, tier, seed)
            if "." in q.answer_text:
                assert re.fullmatch(r"\d+\.\d\d", q.answer_text), (tier, seed, q.answer_text)
                assert all(re.fullmatch(r"\d+\.\d\d", d) for d in q.distractors), (tier, seed, q.distractors)
            else:
                assert all("." not in d for d in q.distractors), (tier, seed, q.distractors)


def test_easy_keys_by_independent_arithmetic():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "easy", seed)
        t = q.text
        money, units = _dollars(t), _units(t)
        if "cents per" in t:
            rate = Fraction(int(re.search(r"(\d+) cents", t).group(1)), 100)
        elif money:
            rate = min(money)
        if "By how many dollars has it depreciated" in t or "how much has it depreciated" in t:
            assert _answer(q) == rate * units[-1], (seed, t)
            seen.add("amount")
        elif "What is the depreciation per" in t or "Find the depreciation per" in t:
            total, n = money[0], units[0]
            assert _answer(q) == total / n, (seed, t)
            seen.add("rate")
        else:
            price, n = money[0], units[-1]
            assert _answer(q) == price - money[1] * n, (seed, t)
            seen.add("value")
    assert seen == {"amount", "rate", "value"}


def test_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "medium", seed)
        t = q.text
        money, units = _dollars(t), _units(t)
        if t.rstrip().endswith("per kilometre, in dollars?") or "depreciation per" in t:
            price, salvage, life = money[0], money[1], units[0]
            assert _answer(q) == (price - salvage) / life, (seed, t)
            seen.add("rate")
        elif "How many" in t or "After how many" in t:
            price, rate, target = max(money[:2]), min(money[:2]), money[2]  # the rate always sits below the price
            assert (price - target) / rate == _answer(q), (seed, t)
            assert _answer(q) == int(_answer(q))
            seen.add("units")
        else:
            price, salvage, life, n = money[0], money[1], units[0], units[1]
            assert _answer(q) == price - (price - salvage) / life * n, (seed, t)
            seen.add("value")
    assert seen == {"rate", "units", "value"}


def _schedule_value(price, stages, n):
    value, left = price, n
    for size, rate in stages:
        use = left if size is None else min(left, size)
        value -= rate * use
        left -= use
    return value


def test_hard_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(TOPIC, "hard", seed)
        t = q.text
        money, units = _dollars(t), _units(t)
        if "for the first" in t:
            price, r1, r2 = money[0], money[1], money[2]
            n1 = units[0]
            if "After how many" in t:
                target = money[3]
                n = int(_answer(q))
                assert _schedule_value(price, [(n1, r1), (None, r2)], n) == target, (seed, t)
                assert n > n1
                seen.add("two-stage-units")
            else:
                n = units[1]
                assert _answer(q) == _schedule_value(price, [(n1, r1), (None, r2)], n), (seed, t)
                seen.add("two-stage-value")
        elif "In its first" in t:
            price, rate = money[0], money[1]
            k = int(re.search(r"In its first (\d+) years", t).group(1))
            used = [
                int(x.replace(",", ""))
                for x in re.search(r"used for ([\d, ]+?) (?:kilometres|hours)", t).group(1).split(", ")
            ]
            assert len(used) == k
            assert _answer(q) == price - rate * sum(used), (seed, t)
            seen.add("yearly")
        else:
            v1, v2 = money[0], money[1]
            u1, u2, u3 = units
            rate = (v1 - v2) / (u2 - u1)
            assert _answer(q) == v2 - rate * (u3 - u2), (seed, t)
            seen.add("readings")
    assert seen == {"two-stage-units", "two-stage-value", "yearly", "readings"}


def test_values_never_fall_below_salvage_in_two_stage_and_yearly_questions():
    for seed in range(120):
        q = sup.single(TOPIC, "hard", seed)
        assert _answer(q) > 0


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 2, 2], ["number"] * 4)


def test_exam_keys_by_independent_arithmetic():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        price, salvage = _dollars(q.stem)
        life = _units(q.stem)[0]
        a, b, c, d = q.parts
        rate = (price - salvage) / life
        assert Fraction(a.answer_text) == rate
        use = _units(b.prompt)[0]
        assert Fraction(b.answer_text) == price - rate * use
        target = _dollars(c.prompt)[0]
        assert Fraction(c.answer_text) == (price - target) / rate
        flat = _dollars(d.prompt)[0]
        years = int(re.search(r"After (\d+) years", d.prompt).group(1))
        assert Fraction(d.answer_text) == (price - rate * use) - (price - flat * years)
        assert Fraction(d.answer_text) > 0


def test_exam_follow_through_from_a_wrong_rate():
    def wrong(q, canon):
        rate = Fraction(q.parts[0].answer_text)
        bad = rate + Fraction(1, 100)
        price = _dollars(q.stem)[0]
        use = _units(q.parts[1].prompt)[0]
        value = price - bad * use
        return {"a": f"{bad:.2f}", "b": f"{float(value):.2f}"}

    assert sup.check_follow_through(SKILL, EXAM, "b", wrong) >= 30


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier, 120)
        assert len(spread) >= 60, (tier, len(spread))
        assert max(spread.values()) <= 6, (tier, spread.most_common(3))
