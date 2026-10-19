"""Tests `profit-loss-and-markup` and `commission-pay` (topics `w3_profit_markup`, `w3_commission_pay`, exams).

The oracles read the amounts and percentages back out of the question TEXT and re-derive each key with Fraction
arithmetic, never from the generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.w3_retail_pay_build as mod
from tests import w3_support as w3
from tests import w7b_support as sup
from tests.w3_support import dollars, percents

PM, PM_X = "w3_profit_markup", "exd_x_w3_profit_markup"
COM, COM_X = "w3_commission_pay", "exd_x_w3_commission_pay"
SK_PM = sup.skill_of("profit-loss-and-markup")
SK_COM = sup.skill_of("commission-pay")


def test_catalogue_has_every_module_misconception():
    w3.check_catalogue(mod)


class TestProfitShared(w3.shared_checks(SK_PM, PM, PM_X)):
    pass


class TestCommissionShared(w3.shared_checks(SK_COM, COM, COM_X)):
    pass


# -- profit, loss and mark-up -----------------------------------------------------------------------------------------


def test_pm_easy_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(PM, "easy", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "What is the profit" in t or "What is the loss" in t:
            cost, sale = dollars(t)
            assert ans == (sale - cost if "profit" in t else cost - sale) > 0, (seed, t)
            seen.add("gap")
        elif "What is the selling price" in t:
            cost, r = dollars(t)[0], percents(t)[0]
            assert ans == cost + cost * r / 100, (seed, t)
            seen.add("price")
        else:
            cost, r = dollars(t)[0], percents(t)[0]
            assert ans == cost * r / 100, (seed, t)
            seen.add("markup")
    assert seen == {"gap", "price", "markup"}


def test_pm_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(PM, "medium", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "percentage profit" in t:
            cost, sale = dollars(t)
            assert (sale - cost) * 100 / cost == ans, (seed, t)
            seen.add("profit")
        elif "percentage loss" in t:
            cost, sale = dollars(t)
            assert (cost - sale) * 100 / cost == ans and sale < cost, (seed, t)
            seen.add("loss")
        else:
            n = int(re.search(r"buys (\d+)", t).group(1))
            cost, sale = dollars(t)
            assert ans == n * (sale - cost), (seed, t)
            seen.add("batch")
    assert seen == {"profit", "loss", "batch"}


def test_pm_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(PM, "hard", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "What did the shop pay" in t:
            sale, r = dollars(t)[0], percents(t)[0]
            assert ans * (100 + r) / 100 == sale, (seed, t)
            seen.add("cost")
        elif "percentage of the selling price" in t:
            cost, sale = dollars(t)
            assert (sale - cost) * 100 / sale == ans, (seed, t)
            seen.add("margin")
        elif "reduced by" in t:
            cost = dollars(t)[0]
            r, dsc = percents(t)
            net = cost * (100 + r) / 100 * (100 - dsc) / 100 - cost
            assert ans == abs(net), (seed, t)
            assert ("profit" in t) == (net > 0)
            seen.add("markup-discount")
        elif "which was a loss of" in t:
            low = dollars(t)[0]
            loss, gain = percents(t)
            cost = low * 100 / (100 - loss)
            assert ans == cost * (100 + gain) / 100, (seed, t)
            seen.add("loss-to-profit")
        else:
            cost = dollars(t)[0]
            m = percents(t)[0]
            assert ans * (100 - m) / 100 == cost, (seed, t)
            seen.add("margin-price")
    assert seen == {"cost", "margin", "markup-discount", "loss-to-profit", "margin-price"}


def test_pm_exam_keys_and_shape():
    sup.check_exam_chain(SK_PM, PM_X, [1, 1, 2, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(PM_X, seed)
        cost, r = dollars(q.stem)[0], percents(q.stem)[0]
        a, b, c, d = q.parts
        dsc = percents(b.prompt)[0]
        marked = cost * (100 + r) / 100
        sale = marked * (100 - dsc) / 100
        assert Fraction(a.answer_text) == marked and Fraction(b.answer_text) == sale
        assert Fraction(c.answer_text) == sale - cost > 0
        assert Fraction(d.answer_text) == (sale - cost) * 100 / cost


def test_pm_exam_follow_through():
    def wrong(q, canon):
        cost = dollars(q.stem)[0]
        bad = Fraction(q.parts[0].answer_text) + 20
        dsc = percents(q.parts[1].prompt)[0]
        sale = bad * (100 - dsc) / 100
        return {"a": f"{float(bad):.2f}", "b": f"{float(sale):.2f}"}

    assert sup.check_follow_through(SK_PM, PM_X, "b", wrong) >= 30


# -- commission -------------------------------------------------------------------------------------------------------


def test_com_easy_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(COM, "easy", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "What is the commission rate" in t:
            sales, comm = dollars(t)
            assert comm * 100 / sales == ans, (seed, t)
            seen.add("rate")
        elif "sales are" in t:
            r = percents(t)[0]
            assert ans == sum(dollars(t)) * r / 100, (seed, t)
            seen.add("several")
        else:
            r, sales = percents(t)[0], dollars(t)[0]
            assert ans == sales * r / 100, (seed, t)
            seen.add("simple")
    assert seen == {"rate", "several", "simple"}


def test_com_medium_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(COM, "medium", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "how much more does Plan A pay" in t:
            base, sales = dollars(t)
            r1, r2 = percents(t)
            assert ans == base + sales * r1 / 100 - sales * r2 / 100 > 0, (seed, t)
            seen.add("plans")
        elif "net sales" in t:
            sales, ret = dollars(t)
            r = percents(t)[0]
            assert ans == (sales - ret) * r / 100, (seed, t)
            seen.add("net")
        elif "What were the sales" in t:
            base, pay = dollars(t)
            r = percents(t)[0]
            assert (pay - base) * 100 / r == ans, (seed, t)
            seen.add("sales")
        else:
            base, sales = dollars(t)
            r = percents(t)[0]
            assert ans == base + sales * r / 100, (seed, t)
            seen.add("pay")
    assert seen == {"plans", "net", "sales", "pay"}


def test_com_hard_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(COM, "hard", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "do the two plans pay the same" in t:
            base = dollars(t)[0]
            r1, r2 = percents(t)
            assert base + ans * r1 / 100 == ans * r2 / 100, (seed, t)
            seen.add("break-even")
        elif "What were the sales" in t:
            tier, comm = dollars(t)
            r1, r2 = percents(t)
            above = ans - tier
            assert above > 0 and tier * r1 / 100 + above * r2 / 100 == comm, (seed, t)
            seen.add("inverse")
        else:
            tier, sales = dollars(t)
            r1, r2 = percents(t)
            assert sales > tier and ans == tier * r1 / 100 + (sales - tier) * r2 / 100, (seed, t)
            seen.add("tiered")
    assert seen == {"break-even", "inverse", "tiered"}


def test_com_exam_keys_and_shape():
    sup.check_exam_chain(SK_COM, COM_X, [1, 2, 2, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(COM_X, seed)
        base, tier = dollars(q.stem)
        r1, r2 = percents(q.stem)
        a, b, c, d = q.parts
        s1, s2, pay3, _ = None, None, None, None
        s1 = dollars(a.prompt)[0]
        s2 = dollars(b.prompt)[0]
        pay3 = dollars(c.prompt)[0]
        flat = percents(d.prompt)[0]
        assert Fraction(a.answer_text) == base + s1 * r1 / 100 and s1 <= tier
        assert Fraction(b.answer_text) == base + tier * r1 / 100 + (s2 - tier) * r2 / 100
        sales3 = Fraction(c.answer_text)
        assert base + tier * r1 / 100 + (sales3 - tier) * r2 / 100 == pay3
        assert Fraction(d.answer_text) == Fraction(b.answer_text) - s2 * flat / 100


def test_com_exam_follow_through():
    def wrong(q, canon):
        s2 = dollars(q.parts[1].prompt)[0]
        flat = percents(q.parts[3].prompt)[0]
        bad = Fraction(q.parts[1].answer_text) + 100
        return {"b": f"{float(bad):.2f}", "d": f"{float(bad - s2 * flat / 100):.2f}"}

    assert sup.check_follow_through(SK_COM, COM_X, "d", wrong) >= 30
