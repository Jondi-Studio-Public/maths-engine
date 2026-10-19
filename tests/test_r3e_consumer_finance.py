"""Tests for the consumer-finance skills (tax, payslips, super, HELP debts, GST, credit cards, BNPL, tariffs, CPI).

Every skill has easy, medium, hard and exam sources. For each one:

* the key is accepted and every distractor of every part is rejected by `mathgen.grading.grade`;
* the exam chains follow through from a wrong earlier part;
* the tiers differ in shape, not just in size;
* an independent oracle reads the figures back out of the question TEXT (the printed tariff or table included) and
  recomputes the key without calling the generator's own helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.r3e_bnpl_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.generators._r3e_kit import number_text
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILLS = {s.id: s for s in load_skills()[1]}
IDS = [
    "income-tax-bracket-calculation",
    "payg-payslip-reading",
    "superannuation-contributions",
    "hecs-help-repayment",
    "gst-invoice-bas",
    "credit-card-interest-statement",
    "bnpl-fees-and-instalments",
    "utility-bill-tiered-tariff",
    "cpi-and-real-value",
]
TOPICS = {
    "income-tax-bracket-calculation": ("r3e_income_tax", "exd_x_r3e_income_tax"),
    "payg-payslip-reading": ("r3e_payg_payslip", "exd_x_r3e_payg_payslip"),
    "superannuation-contributions": ("r3e_superannuation", "exd_x_r3e_superannuation"),
    "hecs-help-repayment": ("r3e_hecs_help", "exd_x_r3e_hecs_help"),
    "gst-invoice-bas": ("r3e_gst_invoice_bas", "exd_x_r3e_gst_invoice_bas"),
    "credit-card-interest-statement": ("r3e_credit_card", "exd_x_r3e_credit_card"),
    "bnpl-fees-and-instalments": ("r3e_bnpl", "exd_x_r3e_bnpl"),
    "utility-bill-tiered-tariff": ("r3e_utility_bill", "exd_x_r3e_utility_bill"),
    "cpi-and-real-value": ("r3e_cpi", "exd_x_r3e_cpi"),
}
TIERS = ("easy", "medium", "hard")
EXAM_PARTS = {  # marks per part
    "income-tax-bracket-calculation": [1, 3, 1, 2],
    "payg-payslip-reading": [1, 1, 2, 2],
    "superannuation-contributions": [1, 1, 1, 3],
    "hecs-help-repayment": [1, 3, 1, 2],
    "gst-invoice-bas": [2, 2, 1, 2],
    "credit-card-interest-statement": [2, 1, 2, 1],
    "bnpl-fees-and-instalments": [1, 2, 2, 2],
    "utility-bill-tiered-tariff": [1, 3, 1, 2],
    "cpi-and-real-value": [1, 2, 2, 2],
}


def _q(skill_id, tier, seed):
    topic = TOPICS[skill_id][1 if tier == "exam" else 0]
    return make(topic, seed=seed, difficulty=tier)


def _money(text):
    return Fraction(text.replace("$", "").replace(",", ""))


_MONEY = r"\$[\d,]+(?:\.\d+)?"


def _amounts(text):
    return [_money(m) for m in re.findall(_MONEY, text)]


# -- wiring -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
def test_skill_wiring(skill_id):
    skill = SKILLS[skill_id]
    assert skill.answer == "number" and skill.domain == "financial-discrete"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (TOPICS[skill_id][0], "easy"),
        (TOPICS[skill_id][0], "medium"),
        (TOPICS[skill_id][0], "hard"),
        (TOPICS[skill_id][1], "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]
    assert all(p for p in skill.prerequisites)
    assert not skill_id.startswith(("r3e", "uc2"))


# -- every key accepted, every distractor rejected ------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_single_answer_tiers(skill_id, tier):
    skill = SKILLS[skill_id]
    texts = set()
    for seed in range(40):
        q = _q(skill_id, tier, seed)
        assert q.answer_kind == "number"
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        places = len(q.answer_text.partition(".")[2])
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (seed, text, q.answer_text)
            assert len(text.partition(".")[2]) == places, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(skill_id, tier, seed).to_json()
        texts.add(q.text)
    assert len(texts) >= 36


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_chain(skill_id):
    skill = SKILLS[skill_id]
    for seed in range(40):
        q = _q(skill_id, "exam", seed)
        assert [p.marks for p in q.parts] == EXAM_PARTS[skill_id]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.label for p in q.parts] == list("abcd")
        assert len({p.answer_text for p in q.parts}) == 4
        assert sum(1 for p in q.parts if p.follow_through) >= 1
        assert q.parts[0].follow_through is None
        assert find_text_defects(q) == []
        assert q.to_json() == _q(skill_id, "exam", seed).to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(EXAM_PARTS[skill_id])
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
    from mathgen.grading.parts import follow_through_expression

    skill = SKILLS[skill_id]
    carried = 0
    for seed in range(40):
        q = _q(skill_id, "exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        places = {p.label: len(p.answer_text.partition(".")[2]) for p in q.parts}
        for target in q.parts:
            if not target.follow_through:
                continue
            for source in sorted(set(re.findall(r"part_([a-d])", target.follow_through))):
                wrong = next(p for p in q.parts if p.label == source).distractors[0]
                replies = {**canon, source: wrong}
                # every part between the source and the target is re-derived from the learner's own replies
                for p in q.parts:
                    if p.label <= source or p.label >= target.label or not p.follow_through:
                        continue
                    expr = follow_through_expression(p)
                    value = expr.subs({f"part_{k}": Fraction(v) for k, v in replies.items() if k < p.label})
                    replies[p.label] = number_text(Fraction(int(value.p), int(value.q)), places[p.label])
                expr = follow_through_expression(target)
                value = expr.subs({f"part_{k}": Fraction(v) for k, v in replies.items() if k < target.label})
                replies[target.label] = number_text(Fraction(int(value.p), int(value.q)), places[target.label])
                if replies[target.label] == canon[target.label]:
                    continue
                result = grade(skill, q, replies)
                assert result.parts[source].marks == 0
                assert result.parts[target.label].marks > 0, (seed, source, target.label, replies)
                carried += 1
    assert carried >= 20


# -- independent oracles ----------------------------------------------------------------------------------------------


def _scale(text):
    """[(lower threshold, rate)] read from rows like `$18,201 to $45,000 | 16 cents for each $1 over $18,200`."""
    bands = [(Fraction(0), Fraction(0))]
    for rate, low in re.findall(r": ([\d.]+) cents for each \$1 over (\$[\d,]+)", text):
        bands.append((_money(low), Fraction(rate)))
    return bands


def _banded(bands, income):
    total = Fraction(0)
    for i, (low, rate) in enumerate(bands):
        high = bands[i + 1][0] if i + 1 < len(bands) else income
        total += max(Fraction(0), min(income, high) - low) * rate / 100
    return total


def _expected_tax(text, tier):
    bands = _scale(text)
    first = text.split("Use these resident tax rates")[0]
    if tier == "easy":
        return _banded(bands, _amounts(first)[0])
    if tier == "medium":
        income = _amounts(first)[-1]
        tax = _banded(bands, income)
        return income - tax if "left once" in text else tax
    return None


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_tax_key_from_the_printed_scale(tier):
    for seed in range(120):
        q = _q("income-tax-bracket-calculation", tier, seed)
        assert Fraction(q.answer_text) == _expected_tax(q.text, tier), (seed, q.text)


def test_tax_hard_variants_from_the_printed_scale():
    seen = Counter()
    for seed in range(200):
        q = _q("income-tax-bracket-calculation", "hard", seed)
        bands = _scale(q.text)
        amounts = _amounts(q.text.split("Use these resident tax rates")[0])
        if "Medicare levy" in q.text:
            gross, ded = amounts[0], amounts[1]
            levy = Fraction(re.search(r"levy is ([\d.]+)%", q.text).group(1))
            taxable = gross - ded
            expected = taxable - _banded(bands, taxable) - taxable * levy / 100
            seen["net"] += 1
        elif "promotion" in q.text:
            start, rise = amounts[0], amounts[1]
            expected = _banded(bands, start + rise) - _banded(bands, start)
            seen["rise"] += 1
        elif "tax bill" in q.text:
            bill = amounts[0]
            lows = [low for low, _r in bands]
            expected = None
            for income in range(0, 400001, 100):
                if _banded(bands, Fraction(income)) == bill:
                    expected = Fraction(income)
                    break
            seen["reverse"] += 1
        else:
            income = amounts[0]
            expected = _banded(bands, income) * 100 / income
            seen["average"] += 1
        assert expected is not None
        places = len(q.answer_text.partition(".")[2])
        assert abs(Fraction(q.answer_text) - expected) <= Fraction(1, 2 * 10**places), (seed, q.text)
    assert min(seen.values()) >= 20 and len(seen) == 4


def test_tax_exam_chain_from_the_printed_scale():
    for seed in range(100):
        q = _q("income-tax-bracket-calculation", "exam", seed)
        bands = _scale(q.stem)
        gross, deductions = _amounts(q.stem.split("Use these resident tax rates")[0])[:2]
        levy = Fraction(re.search(r"levy is ([\d.]+)%", q.stem).group(1))
        taxable = gross - deductions
        tax = _banded(bands, taxable)
        values = [Fraction(p.answer_text) for p in q.parts]
        assert values == [taxable, tax, taxable * levy / 100, taxable - tax - taxable * levy / 100], seed


def test_gst_keys_from_the_text():
    seen = Counter()
    for seed in range(120):
        q = _q("gst-invoice-bas", "easy", seed)
        total = _amounts(q.text)[0]
        assert Fraction(q.answer_text) == total - total / Fraction(11, 10), seed
    for seed in range(120):
        q = _q("gst-invoice-bas", "medium", seed)
        listed = q.text.split("excluding GST: ")[1].split(" GST of 10%")[0]
        price = {name: _money(v) for name, v in re.findall(r"(?:^|; )(.+?): (\$[\d,]+(?:\.\d+)?)", listed)}
        taxable = re.search(r"applies to the (.+?) and the (.+?)\.", q.text).groups()
        total = sum(price.values()) + sum(price[t] for t in taxable) / 10
        assert Fraction(q.answer_text) == total, (seed, q.text)
    for seed in range(160):
        q = _q("gst-invoice-bas", "hard", seed)
        a = _amounts(q.text)
        if "quotes" in q.text:
            expected = a[0] / 11 + a[1] / 10
            seen["quote"] += 1
        else:
            expected = a[0] / 11 - a[2] / 11
            seen["bas"] += 1
        assert abs(Fraction(q.answer_text) - expected) <= Fraction(1, 200), (seed, q.text)
    assert min(seen.values()) >= 30


def test_gst_exam_from_the_text():
    for seed in range(80):
        q = _q("gst-invoice-bas", "exam", seed)
        sales, free_sales, purchases, _free_purchases = _amounts(q.stem)
        values = [Fraction(p.answer_text) for p in q.parts]
        assert values[0] == sales / 11 and values[1] == purchases / 11
        assert values[2] == values[0] - values[1]
        assert values[3] == sales - sales / 11 + free_sales


def test_credit_card_keys_from_the_text():
    seen = Counter()
    for seed in range(120):
        q = _q("credit-card-interest-statement", "easy", seed)
        balance = _amounts(q.text)[0]
        days = int(re.search(r"for (\d+) days", q.text).group(1))
        rate = Fraction(re.search(r"([\d.]+)% a year", q.text).group(1))
        expected = balance * rate / 100 / 365 * days
        assert abs(Fraction(q.answer_text) - expected) <= Fraction(1, 200), seed
    for seed in range(200):
        q = _q("credit-card-interest-statement", "hard", seed)
        rate = Fraction(re.search(r"rate of ([\d.]+)%", q.text).group(1))
        balance, payment = _amounts(q.text)[:2]
        i = rate / 1200
        if "How many monthly payments" in q.text:
            owing, n = balance, 0
            for n in range(1, 400):
                owing = owing * (1 + i) - payment
                if owing <= 0:
                    break
            assert int(q.answer_text) == n, (seed, q.text)
            seen["months"] += 1
        elif "balance after the third" in q.text:
            owing = balance
            for _ in range(3):
                owing = owing * (1 + i) - payment
            assert abs(Fraction(q.answer_text) - owing) <= Fraction(1, 200), seed
            seen["balance"] += 1
        else:
            owing, paid = balance, Fraction(0)
            for _ in range(400):
                owing = owing * (1 + i)
                if owing <= payment:
                    paid += owing
                    break
                owing -= payment
                paid += payment
            assert abs(Fraction(q.answer_text) - (paid - balance)) <= Fraction(1, 200), seed
            seen["interest"] += 1
    assert min(seen.values()) >= 20 and len(seen) == 3


def test_utility_keys_from_the_printed_tariff():
    def tariff(text):
        block = int(re.search(r"first (\d+) kWh", text).group(1))
        r1, r2 = (Fraction(x) for x in re.findall(r": (\d+) cents per kWh[;.]", text)[:2])
        supply = Fraction(re.search(r"Supply charge: (\d+) cents per day", text).group(1))
        return block, r1, r2, supply

    seen = Counter()
    for seed in range(150):
        q = _q("utility-bill-tiered-tariff", "easy", seed)
        block, r1, r2, _s = tariff(q.text)
        kwh = int(re.search(r"used (\d+) kWh", q.text).group(1))
        assert Fraction(q.answer_text) == (r1 * block + r2 * (kwh - block)) / 100, seed
    for seed in range(150):
        q = _q("utility-bill-tiered-tariff", "medium", seed)
        block, r1, r2, supply = tariff(q.text)
        kwh = int(re.search(r"used (\d+) kWh", q.text).group(1))
        days = int(re.search(r"(\d+)-day", q.text).group(1))
        net = supply * days / 100 + (r1 * block + r2 * (kwh - block)) / 100
        assert abs(Fraction(q.answer_text) - net * Fraction(11, 10)) <= Fraction(1, 200), seed
    for seed in range(200):
        q = _q("utility-bill-tiered-tariff", "hard", seed)
        if "usage charge of" in q.text:
            block, r1, r2, _s = tariff(q.text)
            charge = _amounts(q.text)[0]
            kwh = int(q.answer_text)
            assert (r1 * block + r2 * (kwh - block)) / 100 == charge
            seen["reverse"] += 1
        elif "water" in q.text:
            kl = int(re.search(r"used (\d+) kL", q.text).group(1))
            rows = re.findall(r": \$(\d+) per kL", q.text)
            r1, r2, r3 = (int(x) for x in rows)
            b1 = int(re.search(r"First (\d+) kL", q.text).group(1))
            b2 = int(re.search(r"up to (\d+) kL", q.text).group(1))
            fixed = _amounts(q.text)[0]
            assert Fraction(q.answer_text) == r1 * b1 + r2 * (b2 - b1) + r3 * (kl - b2) + fixed
            seen["water"] += 1
        else:
            block, r1, r2, supply = tariff(q.text)
            kwh = int(re.search(r"used (\d+) kWh", q.text).group(1))
            export = int(re.search(r"exported (\d+) kWh", q.text).group(1))
            feed = Fraction(re.search(r"credit: (\d+) cents per kWh exported", q.text).group(1))
            days = int(re.search(r"(\d+)-day", q.text).group(1))
            net = supply * days / 100 + (r1 * block + r2 * (kwh - block)) / 100
            assert abs(Fraction(q.answer_text) - (net * Fraction(11, 10) - feed * export / 100)) <= Fraction(1, 200), (
                seed
            )
            seen["solar"] += 1
    assert min(seen.values()) >= 20 and len(seen) == 3


def test_super_bnpl_cpi_hecs_payslip_keys_from_the_text():
    for seed in range(100):
        q = _q("superannuation-contributions", "easy", seed)
        salary = _amounts(q.text)[0]
        rate = Fraction(re.search(r"of ([\d.]+)% of salary", q.text).group(1))
        assert Fraction(q.answer_text) == salary * rate / 100 / 4, seed
        q = _q("superannuation-contributions", "medium", seed)
        salary, sacrifice = _amounts(q.text)[:2]
        rate = Fraction(re.search(r"super of ([\d.]+)%", q.text).group(1))
        tax = Fraction(re.search(r"taxed at (\d+)%", q.text).group(1))
        assert Fraction(q.answer_text) == (salary * rate / 100 + sacrifice) * (100 - tax) / 100, seed
        q = _q("bnpl-fees-and-instalments", "easy", seed)
        price, fee = _amounts(q.text)[:2]
        assert Fraction(q.answer_text) == (price + fee) / 4, seed
        q = _q("bnpl-fees-and-instalments", "medium", seed)
        price = _amounts(q.text)[0]
        n_late = int(re.search(r"pays (\d) of the instalments late", q.text).group(1))
        fee_each = Fraction(re.search(r"a \$(\d+) late fee", q.text).group(1))
        cap = Fraction(re.search(r"more than (\d+)% of the price", q.text).group(1)) * price / 100
        setup = Fraction(re.search(r"set-up fee of \$(\d+)", q.text).group(1)) if "set-up" in q.text else 0
        assert Fraction(q.answer_text) == price + setup + min(n_late * fee_each, cap), seed
        q = _q("cpi-and-real-value", "easy", seed)
        ca, cb = (Fraction(x) for x in re.search(r"CPI was ([\d.]+) in \d+ and ([\d.]+) in", q.text).groups())
        assert abs(Fraction(q.answer_text) - (cb - ca) * 100 / ca) <= Fraction(1, 20), seed
        q = _q("cpi-and-real-value", "medium", seed)
        ca, cb = (Fraction(x) for x in re.search(r"CPI was ([\d.]+) in \d+ and ([\d.]+) in", q.text).groups())
        price = _amounts(q.text)[0]
        assert abs(Fraction(q.answer_text) - price * cb / ca) <= Fraction(1, 200), seed
        q = _q("payg-payslip-reading", "easy", seed)
        rows = dict(re.findall(r"(Gross pay|PAYG tax withheld|Union fees): (\$[\d,.]+)", q.text))
        gross, payg, union = (_money(rows[k]) for k in ("Gross pay", "PAYG tax withheld", "Union fees"))
        assert Fraction(q.answer_text) == gross - payg - union, seed
        q = _q("hecs-help-repayment", "easy", seed)
        income = _amounts(q.text)[0]
        rows = re.findall(r"(\$[\d,]+) (?:up to but not including \$[\d,]+|or more): (\d+)%", q.text)
        rate = Fraction(0)
        for low, r in rows:
            if income >= _money(low):
                rate = Fraction(r)
        assert Fraction(q.answer_text) == income * rate / 100, seed


def test_hecs_years_to_clear_by_closed_form():
    for seed in range(120):
        q = _q("hecs-help-repayment", "hard", seed)
        if "How many yearly repayments" not in q.text:
            continue
        debt, repayment = _amounts(q.text)[:2]
        idx = Fraction(re.search(r"indexed by (\d+)%", q.text).group(1))
        g = 1 + idx / 100
        n = int(q.answer_text)
        before = debt * g ** (n - 1) - repayment * g * (g ** (n - 1) - 1) / (g - 1)
        assert before <= repayment
        if n > 1:
            earlier = debt * g ** (n - 2) - repayment * g * (g ** (n - 2) - 1) / (g - 1)
            assert earlier > repayment


# -- shape and variety -------------------------------------------------------------------------------------------


def _ask(text):
    """The question sentence: what is asked for, in the first 45 characters of the sentence holding the `?`."""
    return text.split("?")[0].split(". ")[-1][:45]


@pytest.mark.parametrize("skill_id", IDS)
def test_tiers_differ_in_shape(skill_id):
    """Hard is not easy with bigger numbers: it asks questions the easy tier never asks, in more than one shape."""
    asks = {t: Counter(_ask(_q(skill_id, t, s).text) for s in range(40)) for t in TIERS}
    assert len(asks["hard"]) >= 2, asks["hard"]
    assert set(asks["hard"]) - set(asks["easy"]), (asks["hard"], asks["easy"])
    # the answer is not a lookup: many different answers over many seeds
    answers = [_q(skill_id, "hard", s).answer_text for s in range(60)]
    assert len(set(answers)) >= 30


@pytest.mark.parametrize("skill_id", IDS)
def test_exam_answers_vary(skill_id):
    answers = [_q(skill_id, "exam", s).answer_text for s in range(60)]
    assert len(set(answers)) >= 40
