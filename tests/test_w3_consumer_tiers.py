"""Tests the medium and hard tiers added to nine consumer-arithmetic and statistics skills
(`mathgen/generators/w3_consumer_tiers_build.py`).

Each oracle reads the numbers back out of the question TEXT and re-derives the key with Fraction arithmetic (or, for the
road lists, by listing every route), so a generator that wrote the wrong number into its own prompt would fail.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import ceil

import pytest

import mathgen.generators.w3_consumer_tiers_build as mod
from tests import w3_support as w3
from tests import w7b_support as sup
from tests.w3_support import cents, dollars, percents

CASES = [
    ("compound-interest", "w3_ci_tiers"),
    ("simple-interest", "w3_si_tiers"),
    ("calculate-wages", "w3_wage_tiers"),
    ("calculate-overtime", "w3_ot_tiers"),
    ("unit-pricing", "w3_unit_tiers"),
    ("flat-loan-repayment", "w3_flatloan_tiers"),
    ("moving-average", "w3_movavg_tiers"),
    ("calculate-mean", "w3_mean_tiers"),
    ("shortest-network-path", "w3_netpath_tiers"),
]
TIERS = ("medium", "hard")


def test_catalogue_has_every_module_misconception():
    w3.check_catalogue(mod)


@pytest.mark.parametrize("skill_id,topic", CASES)
class TestShared:
    def test_wired_into_the_skill(self, skill_id, topic):
        sk = sup.skill_of(skill_id)
        by_tier = {s.difficulty: s.topic for s in sk.sources if s.topic == topic}
        assert set(by_tier) == set(TIERS)
        assert sk.answer == "number"
        steps = [sk.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
        assert steps == sorted(steps) and len(set(steps)) == 4
        have = {m.id for m in sk.misconceptions}
        for ident, _desc, area in mod.MISCONCEPTIONS:
            if area == skill_id:
                assert ident in have

    @pytest.mark.parametrize("tier", TIERS)
    def test_key_accepted_and_distractors_rejected(self, skill_id, topic, tier):
        sup.check_single_tier(sup.skill_of(skill_id), topic, tier)

    def test_pools(self, skill_id, topic):
        shapes = {}
        for tier in TIERS:
            qs = [sup.single(topic, tier, seed) for seed in range(60)]
            assert len({q.text for q in qs}) >= 20, tier
            shapes[tier] = sup.distinct_shapes(topic, tier, 60)
        assert shapes["hard"] - shapes["medium"], "the hard tier adds no prompt shape over the medium tier"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answers_are_not_guessable(self, skill_id, topic, tier):
        spread = sup.answer_spread(topic, tier, 60)
        assert len(spread) >= 12, len(spread)
        assert max(spread.values()) <= 10, spread.most_common(3)

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_match_the_answer_shape(self, skill_id, topic, tier):
        for seed in range(60):
            q = sup.single(topic, tier, seed)
            places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
            for d in q.distractors:
                assert (len(d.split(".")[1]) if "." in d else 0) == places, (seed, d, q.answer_text)


def ans(q):
    return Fraction(q.answer_text)


def growth(rate, n):
    return (1 + Fraction(rate) / 100) ** n


def first_int(pattern, text):
    return int(re.search(pattern, text).group(1))


# -- compound interest ------------------------------------------------------------------------------------------------


def test_ci_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(250):
            q = sup.single("w3_ci_tiers", tier, seed)
            t = q.text
            p, pc = dollars(t)[0], percents(t)
            if "How much interest is earned in total" in t:
                n = first_int(r"for (\d+) years", t)
                assert ans(q) == cents(p * growth(pc[0], n) - p), (seed, t)
                seen.add("interest")
            elif "compounded half-yearly" in t or "compounded quarterly" in t:
                m = 2 if "half-yearly" in t else 4
                n = first_int(r"after (\d+) years?", t)
                assert ans(q) == cents(p * growth(pc[0] / m, m * n)), (seed, t)
                seen.add("periods")
            elif "How much must she invest" in t:
                n = first_int(r"in (\d+) years", t)
                assert ans(q) == round(p / growth(pc[0], n)), (seed, t)
                seen.add("present")
            elif "How much more interest does Account B" in t:
                n = first_int(r"for (\d+) years", t)
                assert ans(q) == cents(p * growth(pc[0], n) - p - p * pc[0] / 100 * n), (seed, t)
                seen.add("vs-simple")
            else:
                r1, r2 = pc
                n1 = first_int(r"first (\d+) years?", t)
                n2 = first_int(r"next (\d+) years?", t)
                assert ans(q) == cents(p * growth(r1, n1) * growth(r2, n2)), (seed, t)
                seen.add("two-rates")
    assert seen == {"interest", "periods", "present", "vs-simple", "two-rates"}


# -- simple interest --------------------------------------------------------------------------------------------------


def test_si_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(250):
            q = sup.single("w3_si_tiers", tier, seed)
            t = q.text
            if "total amount at the end" in t:
                p, r = dollars(t)[0], percents(t)[0]
                n = first_int(r"for (\d+) years", t)
                assert ans(q) == p + p * r / 100 * n, (seed, t)
                seen.add("amount")
            elif "months" in t:
                p, r = dollars(t)[0], percents(t)[0]
                months = first_int(r"for (\d+) months", t)
                assert ans(q) == cents(p * r / 100 * months / 12), (seed, t)
                seen.add("months")
            elif "annual interest rate" in t:
                p, i = dollars(t)[:2]
                n = first_int(r"over (\d+) years", t)
                assert ans(q) * p * n == i * 100, (seed, t)
                seen.add("rate")
            elif "For how many years" in t:
                p, i = dollars(t)[:2]
                r = percents(t)[0]
                assert ans(q) * p * r / 100 == i, (seed, t)
                seen.add("time")
            else:
                (i,) = dollars(t)
                r = percents(t)[0]
                n = first_int(r"in (\d+) years", t)
                assert ans(q) * r / 100 * n == i, (seed, t)
                seen.add("principal")
    assert seen == {"amount", "months", "rate", "time", "principal"}


# -- wages ------------------------------------------------------------------------------------------------------------


def test_wage_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_wage_tiers", tier, seed)
            t = q.text
            d = dollars(t)
            if "works for" in t:
                h = first_int(r"for (\d+) h", t)
                m = re.search(r"(\d+) min", t)
                hours = h + Fraction(int(m.group(1)), 60 if m else 1) if m else Fraction(h)
                assert ans(q) == cents(d[0] * hours), (seed, t)
                seen.add("hm")
            elif "hourly rate of pay" in t:
                hours = first_int(r"for (\d+) hours", t)
                assert ans(q) == cents(d[0] / hours), (seed, t)
                seen.add("rate")
            elif "worked:" in t:
                hrs = sum(Fraction(x) for x in re.findall(r"(\d+(?:\.\d)?) h", t.split("worked:")[1]))
                assert ans(q) == cents(d[0] * hrs), (seed, t)
                seen.add("week")
            elif "casual loading" in t:
                load = percents(t)[0]
                hours = first_int(r"in (\d+) hours", t)
                assert ans(q) == cents(d[0] * (1 + load / 100) * hours), (seed, t)
                seen.add("casual")
            elif "withheld as tax" in t:
                hours = first_int(r"works (\d+) hours", t)
                assert ans(q) == cents(d[0] * hours * (1 - percents(t)[0] / 100)), (seed, t)
                seen.add("net")
            elif "smallest whole number of hours" in t:
                rate, target = d
                assert ans(q) == ceil(target / rate), (seed, t)
                assert target / rate != int(target / rate)
                seen.add("hours")
            else:
                r1, r2 = d
                h1, h2 = (int(x) for x in re.findall(r"works (\d+) hours there", t))
                assert ans(q) == abs(r1 * h1 - r2 * h2), (seed, t)
                seen.add("jobs")
    assert seen == {"hm", "rate", "week", "casual", "net", "hours", "jobs"}


# -- overtime ---------------------------------------------------------------------------------------------------------


def test_overtime_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_ot_tiers", tier, seed)
            t = q.text
            d = dollars(t)
            if "What is Dee's pay" in t:
                worked = first_int(r"works (\d+) hours", t)
                assert ans(q) == cents(d[0] * 38 + d[0] * Fraction(3, 2) * (worked - 38)), (seed, t)
                seen.add("total")
            elif "overtime pay?" in t:
                worked = first_int(r"duty for (\d+) hours", t)
                assert ans(q) == cents(d[0] * Fraction(3, 2) * (worked - 38)), (seed, t)
                seen.add("only")
            elif "double time" in t and "Sunday" in t:
                base, sun = (
                    int(x)
                    for x in re.findall(r"works (\d+) hours", t)[:1] + re.findall(r"and (\d+) hours on Sunday", t)
                )
                assert ans(q) == cents(d[0] * base + d[0] * 2 * sun), (seed, t)
                seen.add("double")
            elif "next 3 hours" in t:
                worked = first_int(r"works (\d+) hours", t)
                extra = worked - 38
                assert ans(q) == cents(d[0] * 38 + d[0] * Fraction(9, 2) + d[0] * 2 * (extra - 3)), (seed, t)
                seen.add("tiered")
            elif "How many hours of overtime" in t:
                rate, pay = d
                assert (pay - rate * 38) == ans(q) * rate * Fraction(3, 2), (seed, t)
                seen.add("hours")
            else:
                (pay,) = d
                hours = first_int(r"for a week of (\d+) hours", t)
                o = hours - 38
                assert ans(q) * (38 + Fraction(3, 2) * o) == pay, (seed, t)
                seen.add("rate")
    assert seen == {"total", "only", "double", "tiered", "hours", "rate"}


# -- unit pricing -----------------------------------------------------------------------------------------------------


def test_unit_pricing_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_unit_tiers", tier, seed)
            t = q.text
            d = dollars(t)
            if "At the same price per item" in t:
                k = first_int(r"pack of (\d+)", t)
                n = first_int(r"how much do (\d+)", t)
                assert ans(q) == cents(d[0] / k * n), (seed, t)
                seen.add("n-items")
            elif "price per 100 g" in t:
                m = re.search(r"A ([\d.]+) (g|kg) packet", t)
                grams = Fraction(m.group(1)) * (1000 if m.group(2) == "kg" else 1)
                assert ans(q) == cents(d[0] / (grams / 100)), (seed, t)
                seen.add("per100")
            elif "difference between the two prices per item" in t:
                k1, k2 = (int(x) for x in re.findall(r"pack of (\d+)", t))
                assert ans(q) == round(abs(d[0] / k1 - d[1] / k2) * 100), (seed, t)
                seen.add("difference")
            elif "is saved by buying" in t:
                k1, k2 = (int(x) for x in re.findall(r"pack of (\d+)", t))
                n = first_int(r"buying (\d+)", t)
                assert ans(q) == cents(abs(d[0] / k1 - d[1] / k2) * n), (seed, t)
                seen.add("saving")
            else:
                dear, cheap = d
                assert ans(q) == round((dear - cheap) / dear * 100), (seed, t)
                seen.add("percent")
    assert seen == {"n-items", "per100", "difference", "saving", "percent"}


# -- flat loans -------------------------------------------------------------------------------------------------------


def test_flat_loan_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_flatloan_tiers", tier, seed)
            t = q.text
            d = dollars(t)
            if "How much interest is charged in total" in t and "equal monthly payments" in t:
                p, monthly = d
                months = first_int(r"with (\d+) equal", t)
                assert ans(q) == monthly * months - p, (seed, t)
                seen.add("from-monthly")
            elif "How much interest is charged in total" in t:
                n = first_int(r"for (\d+) years", t)
                assert ans(q) == d[0] * percents(t)[0] / 100 * n, (seed, t)
                seen.add("interest")
            elif "total amount to be repaid" in t:
                n = first_int(r"for (\d+) years", t)
                assert ans(q) == d[0] * (1 + percents(t)[0] / 100 * n), (seed, t)
                seen.add("total")
            elif "monthly repayment" in t:
                price = d[0]
                dep, rate = percents(t)
                n = first_int(r"over (\d+) years", t)
                loan = price * (1 - dep / 100)
                assert ans(q) == cents(loan * (1 + rate / 100 * n) / (12 * n)), (seed, t)
                seen.add("deposit")
            elif "annual flat interest rate" in t:
                p, monthly = d
                months = first_int(r"with (\d+) equal", t)
                years = Fraction(months, 12)
                assert ans(q) == round_tenth((monthly * months - p) / p / years * 100), (seed, t)
                seen.add("rate")
            else:
                p, i = d
                assert ans(q) * p * percents(t)[0] / 100 == i, (seed, t)
                seen.add("term")
    assert seen == {"interest", "total", "deposit", "rate", "term", "from-monthly"}


def round_tenth(x):
    return Fraction(int(x * 10 + Fraction(1, 2)), 10)


# -- moving average ---------------------------------------------------------------------------------------------------


def test_moving_average_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_movavg_tiers", tier, seed)
            t = q.text
            if "centred on period" in t and "-point moving average centred on period" in t:
                series = [int(x) for x in re.search(r"were: ([\d, ]+)\.", t).group(1).split(", ")]
                w = first_int(r"(\d+)-point moving average centred on period", t)
                k = first_int(r"centred on period (\d+)", t) - 1
                window = series[k - w // 2 : k + w // 2 + 1]
                assert len(window) == w and ans(q) * w == sum(window), (seed, t)
                seen.add("kth")
            elif "missing value" in t:
                w = first_int(r"The (\d+)-point", t)
                shown = re.search(r"values ([\d?, ]+) is (\d+)", t)
                items = shown.group(1).split(", ")
                mean = int(shown.group(2))
                assert len(items) == w
                known = sum(int(x) for x in items if x != "?")
                assert ans(q) == mean * w - known, (seed, t)
                seen.add("missing")
            elif "By how much does the moving average change" in t:
                w = first_int(r"A (\d+)-point", t)
                out, new = (int(x) for x in re.search(r"the value (\d+) drops out .* value (\d+) comes in", t).groups())
                assert ans(q) * w == new - out, (seed, t)
                seen.add("change")
            else:
                v = [int(x) for x in re.search(r"values are ([\d, ]+)\.", t).group(1).split(", ")]
                assert ans(q) * 8 == v[0] + 2 * (v[1] + v[2] + v[3]) + v[4], (seed, t)
                seen.add("centred4")
    assert seen == {"kth", "missing", "change", "centred4"}


# -- mean -------------------------------------------------------------------------------------------------------------


def test_mean_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(300):
            q = sup.single("w3_mean_tiers", tier, seed)
            t = q.text
            if "timed" in t:
                vals = [Fraction(x) for x in re.search(r"seconds: ([\d., ]+)\.", t).group(1).split(", ")]
                assert ans(q) == cents(sum(vals) / len(vals)), (seed, t)
                seen.add("decimal")
            elif "how many students got each score" in t:
                rows = re.findall(r"score (\d+): (\d+)", t)
                total = sum(int(s) * int(f) for s, f in rows)
                count = sum(int(f) for _, f in rows)
                assert ans(q) == cents(Fraction(total, count)), (seed, t)
                seen.add("freq")
            elif "must the student score" in t:
                known = [int(x) for x in re.search(r"scored ([\d, ]+) in the first", t).group(1).split(", ")]
                n = first_int(r"in test (\d+)", t)
                target = first_int(r"mean of (\d+) over", t)
                assert len(known) == n - 1 and ans(q) == target * n - sum(known), (seed, t)
                seen.add("missing")
            elif "Class A has" in t:
                n1, m1, n2, m2 = (
                    int(x)
                    for x in re.search(
                        r"has (\d+) students with a mean mark of (\d+). Class B has (\d+) students with a mean mark of (\d+)",
                        t,
                    ).groups()
                )
                assert ans(q) == round_tenth(Fraction(n1 * m1 + n2 * m2, n1 + n2)), (seed, t)
                seen.add("combined")
            else:
                n, m, x = (
                    int(v) for v in re.search(r"mean of (\d+) numbers is (\d+). A new number, (\d+)", t).groups()
                )
                assert ans(q) == round_tenth(Fraction(n * m + x, n + 1)), (seed, t)
                seen.add("new")
    assert seen == {"decimal", "freq", "missing", "combined", "new"}


# -- road lists -------------------------------------------------------------------------------------------------------


def routes(edges, start, end):
    adj = {}
    for (a, b), w in edges.items():
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    found = []

    def go(node, seen, length):
        if node == end:
            found.append((length, tuple(seen)))
            return
        for nxt, w in adj.get(node, ()):
            if nxt not in seen:
                go(nxt, seen + [nxt], length + w)

    go(start, [start], 0)
    return found


def read_edges(text):
    return {(m[0], m[1]): int(m[2]) for m in re.findall(r"\b([A-G])([A-G]) = (\d+)", text)}


def test_network_keys():
    seen = set()
    for tier in TIERS:
        for seed in range(200):
            q = sup.single("w3_netpath_tiers", tier, seed)
            t = q.text
            edges = read_edges(t)
            end = "F" if "Six towns" in t else "G"
            assert ("A", end) not in edges
            if "is closed" in t:
                closed = re.search(r"The road ([A-G])([A-G]) is closed", t).groups()
                edges = {e: w for e, w in edges.items() if e != closed}
                best = min(routes(edges, "A", end))[0]
                seen.add("closed")
            elif "stop at town" in t:
                via = re.search(r"stop at town ([A-G])", t).group(1)
                best = min(r for r in routes(edges, "A", end) if via in r[1])[0]
                seen.add("via")
            else:
                best = min(routes(edges, "A", end))[0]
                seen.add("plain")
            assert ans(q) == best, (seed, t)
    assert seen == {"closed", "via", "plain"}
