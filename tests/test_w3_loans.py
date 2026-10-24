"""Tests the w3 loan skills: `interest-only-loan` and `future-value-annuity-table` (topics `w3_interest_only_loan`,
`w3_annuity_table` and their exam chains).

The oracles read the loan, rates, years and table entries back out of the question TEXT and re-derive each key with
Fraction arithmetic. The annuity oracle also checks every printed table entry against the exact annuity formula, and
reads the factor FROM THE PRINTED TABLE, so a wrong cell in the question would fail.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen.generators.w3_loans_build as mod
from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

IOL, IOL_X = "w3_interest_only_loan", "exd_x_w3_interest_only_loan"
ANN, ANN_X = "w3_annuity_table", "exd_x_w3_annuity_table"
SK_IOL = sup.skill_of("interest-only-loan")
SK_ANN = sup.skill_of("future-value-annuity-table")

_MONEY = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def dollars(text):
    return [Fraction(m.replace(",", "")) for m in _MONEY.findall(text)]


def percents(text):
    return [Fraction(m) for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]


def test_catalogue_has_every_module_misconception():
    for ident, description, _area in mod.MISCONCEPTIONS:
        assert CATALOGUE[ident].description == description


@pytest.mark.parametrize(
    "skill,topic,exam", [(SK_IOL, IOL, IOL_X), (SK_ANN, ANN, ANN_X)], ids=["interest-only", "annuity-table"]
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

    def test_distractors_match_the_answer_shape(self, skill, topic, exam):
        for tier in sup.TIERS:
            for seed in range(60):
                q = sup.single(topic, tier, seed)
                places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
                for d in q.distractors:
                    assert (len(d.split(".")[1]) if "." in d else 0) == places, (tier, seed, d, q.answer_text)


# -- interest-only loans ----------------------------------------------------------------------------------------------


def test_iol_easy_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(IOL, "easy", seed)
        t = q.text
        loan, rate = dollars(t)[0], percents(t)[0]
        ans = Fraction(q.answer_text)
        if "How much interest is paid in one year" in t:
            assert ans == loan * rate / 100, (seed, t)
            seen.add("year")
        elif "each monthly payment" in t:
            assert ans * 12 == loan * rate / 100, (seed, t)
            seen.add("month")
        else:
            n = int(re.search(r"over (\d+) years", t).group(1))
            assert ans == sum(loan * rate / 100 for _ in range(n)), (seed, t)
            seen.add("total")
    assert seen == {"year", "month", "total"}


def test_iol_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(IOL, "medium", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "How much was borrowed" in t:
            monthly, rate = dollars(t)[0], percents(t)[0]
            assert ans * rate / 100 / 12 == monthly, (seed, t)
            seen.add("loan")
        elif "interest rate per annum, as a percentage" in t:
            loan, monthly = dollars(t)
            assert Fraction(f"{float(monthly * 12 * 100 / loan):.1f}") == ans, (seed, t)
            assert monthly * 12 * 100 / loan == ans
            seen.add("rate")
        else:
            loan, rate = dollars(t)[0], percents(t)[0]
            n = int(re.search(r"At the end of (\d+) years", t).group(1))
            assert ans == loan + n * loan * rate / 100, (seed, t)
            seen.add("repaid")
    assert seen == {"loan", "rate", "repaid"}


def test_iol_hard_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(IOL, "hard", seed)
        t = q.text
        ans = Fraction(q.answer_text)
        if "for the first" in t:
            loan = dollars(t)[0]
            r1, r2 = percents(t)
            n, k = (int(x) for x in re.search(r"for (\d+) years.*?first (\d+) years", t).groups())
            months = [r1 if m < 12 * k else r2 for m in range(12 * n)]
            assert ans == sum(loan * r / 100 / 12 for r in months), (seed, t)
            seen.add("change")
        elif "increases by" in t:
            r1 = percents(t)[0]
            rise = Fraction(re.search(r"rises by ([\d.]+) percentage", t).group(1))
            extra = dollars(t)[0]
            loan = extra * 12 * 100 / rise
            assert ans == loan * r1 / 100 / 12, (seed, t)
            seen.add("rise")
        else:
            loan, lump = dollars(t)[:2]
            rate = percents(t)[0]
            n, k = (int(x) for x in re.search(r"for (\d+) years. After (\d+) years", t).groups())
            owing = [loan if m < 12 * k else loan - lump for m in range(12 * n)]
            assert ans == sum(o * rate / 100 / 12 for o in owing), (seed, t)
            seen.add("lump")
    assert seen == {"change", "rise", "lump"}


def test_iol_hard_distractors_include_each_named_slip():
    tags = set()
    for seed in range(120):
        q = sup.single(IOL, "hard", seed)
        tags |= set(q.option_notes.values())
    assert {"iol-first-rate-throughout", "iol-average-rate", "iol-rise-as-payment", "iol-lump-ignored"} <= tags


def test_iol_exam_keys_and_shape():
    sup.check_exam_chain(SK_IOL, IOL_X, [1, 2, 1, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(IOL_X, seed)
        loan, rate = dollars(q.stem)[0], percents(q.stem)[0]
        a, b, c, d = q.parts
        monthly = loan * rate / 1200
        assert Fraction(a.answer_text) == monthly
        n = int(re.search(r"over (\d+) years", b.prompt).group(1))
        assert Fraction(b.answer_text) == sum(monthly for _ in range(12 * n))
        assert Fraction(c.answer_text) == Fraction(b.answer_text) + loan
        lump = dollars(d.prompt)[0]
        assert Fraction(d.answer_text) == (loan - lump) * rate / 1200


def test_iol_exam_follow_through():
    def wrong(q, canon):
        bad = Fraction(q.parts[0].answer_text) + 7
        n = int(re.search(r"over (\d+) years", q.parts[1].prompt).group(1))
        return {"a": f"{float(bad):.2f}", "b": f"{float(bad * 12 * n):.2f}"}

    assert sup.check_follow_through(SK_IOL, IOL_X, "b", wrong) >= 30


# -- annuity table ----------------------------------------------------------------------------------------------------


def exact_factor(rate_pct: Fraction, n: int) -> Fraction:
    i = rate_pct / 100
    value = ((1 + i) ** n - 1) / i
    scaled = value * 10_000
    return Fraction(int(scaled + Fraction(1, 2)), 10_000)


def read_table(text):
    """{(rate percent, periods): factor} from the printed table, checking each entry against the annuity formula."""
    table = {}
    for m in re.finditer(r"at ([\d.]+)% per period: (.*?)(?=; at |\. )", text):
        rate = Fraction(m.group(1))
        for n, f in re.findall(r"(\d+) periods (\d+\.\d{4})", m.group(2)):
            table[(rate, int(n))] = Fraction(f)
            assert Fraction(f) == exact_factor(rate, int(n)), (rate, n, f)
    assert table
    return table


_PER = {"monthly": 12, "quarterly": 4, "half-yearly": 2}


def conversion(text):
    kind = next(k for k in _PER if f"{k} " in text)
    annual = int(re.search(r"pays (\d+)% per annum", text).group(1))
    years = int(re.search(r"(\d+) years", text).group(1))
    return Fraction(annual, _PER[kind]), years * _PER[kind]


def test_annuity_easy_and_medium_keys():
    seen = set()
    for tier in ("easy", "medium"):
        for seed in range(150):
            q = sup.single(ANN, tier, seed)
            t = q.text
            table = read_table(t)
            rate = (
                Fraction(re.search(r"The interest rate is ([\d.]+)% per period", t).group(1))
                if "The interest rate is" in t
                else Fraction(re.search(r"paying ([\d.]+)% per period", t).group(1))
            )
            n = int(re.search(r"(?:for|of each period for) (\d+) periods", t).group(1))
            f = table[(rate, n)]
            ans = Fraction(q.answer_text)
            if "payment needed each period" in t:
                target = dollars(t)[0]
                assert ans * f == target, (seed, t)
                seen.add("payment")
            elif "how much interest is earned" in t:
                pay = dollars(t)[0]
                assert ans == pay * f - pay * n, (seed, t)
                seen.add("interest")
            else:
                pay = dollars(t)[0]
                assert ans == pay * f, (seed, t)
                seen.add("fv")
    assert seen == {"payment", "interest", "fv"}


def test_annuity_hard_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(ANN, "hard", seed)
        t = q.text
        table = read_table(t)
        i, n = conversion(t)
        f = table[(i, n)]
        ans = Fraction(q.answer_text)
        if "payment needed each period" in t:
            target = dollars(t)[0]
            assert ans * f == target, (seed, t)
            seen.add("payment")
        else:
            pay = dollars(t)[0]
            assert ans == pay * f, (seed, t)
            seen.add("fv")
    assert seen == {"payment", "fv"}


def test_annuity_hard_table_holds_the_wrong_rate_rows_a_learner_would_pick():
    for seed in range(60):
        q = sup.single(ANN, "hard", seed)
        table = read_table(q.text)
        i, n = conversion(q.text)
        rates = {r for r, _ in table}
        assert i in rates and len(rates) == 3
        assert {m for _, m in table} >= {n}


def test_annuity_exam_keys_and_shape():
    sup.check_exam_chain(SK_ANN, ANN_X, [2, 1, 1, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(ANN_X, seed)
        table = read_table(q.stem)
        i, n = conversion(q.stem)
        pay = dollars(q.stem)[0]
        a, b, c, d = q.parts
        f = table[(i, n)]
        assert Fraction(a.answer_text) == pay * f
        assert Fraction(b.answer_text) == pay * n
        assert Fraction(c.answer_text) == pay * f - pay * n
        target = dollars(d.prompt)[0]
        assert Fraction(d.answer_text) * f == target


def test_annuity_exam_follow_through():
    def wrong(q, canon):
        pay = dollars(q.stem)[0]
        bad = Fraction(q.parts[0].answer_text) + 50
        contributed = Fraction(q.parts[1].answer_text)
        return {"a": f"{float(bad):.2f}", "c": f"{float(bad - contributed):.2f}"}

    assert sup.check_follow_through(SK_ANN, ANN_X, "c", wrong) >= 30
