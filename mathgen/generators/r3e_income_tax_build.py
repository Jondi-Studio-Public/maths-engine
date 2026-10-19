"""Income tax from a stated scale: `r3e_income_tax` (easy, medium, hard) and its exam chain `exd_x_r3e_income_tax`.

``income-tax-bracket-calculation`` (``financial-discrete``, answer ``number``). Every question prints the resident
tax scale it uses as a table (and, where it matters, the Medicare levy as a percentage), so nothing depends on a
particular year's schedule; the scale is drawn from small sets of thresholds and rates.

* easy: tax payable on a taxable income that reaches only the first taxed band.
* medium: tax payable, or the income left after tax, on an income that reaches the third or fourth band (several
  slices, each at its own rate).
* hard: one of four multi-step questions: tax plus levy on an income net of deductions; the tax a pay rise adds when
  it carries income over a threshold; the taxable income that produced a stated tax bill; the average rate of tax.
* exam (``exd_x_r3e_income_tax``): taxable income from gross income and deductions, the tax, the levy, and what
  remains, each following through from the one before.

Every answer is checked a second way: the tax is rebuilt from the running total at the start of the band the income
reaches, rather than summed slice by slice.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import Banded, dollars, listing, number_part, number_text, percent, single_number

_T1 = [18200, 20000, 22000]
_T2 = [45000, 50000, 60000]
_T3 = [120000, 135000, 150000]
_T4 = [180000, 190000, 200000]
_R1 = [16, 18, 19]
_R2 = [30, 32, 35]
_R3 = [37, 39, 40]
_R4 = [45, 47]
_LEVY = [Fraction(3, 2), Fraction(2)]

_PEOPLE = [
    ("Mei", "a primary school teacher"),
    ("Tom", "an electrician"),
    ("Aisha", "a paramedic"),
    ("Liam", "a bus driver"),
    ("Chloe", "a physiotherapist"),
    ("Ben", "a surveyor"),
    ("Priya", "a pharmacist"),
    ("Jack", "a diesel mechanic"),
    ("Sophie", "an accountant"),
    ("Daniel", "a civil engineer"),
]


def _scale(rng) -> Banded:
    return Banded(
        [
            (0, 0),
            (rng.choice(_T1), rng.choice(_R1)),
            (rng.choice(_T2), rng.choice(_R2)),
            (rng.choice(_T3), rng.choice(_R3)),
            (rng.choice(_T4), rng.choice(_R4)),
        ]
    )


def _scale_text(sc: Banded) -> str:
    rows = [(a, b) for a, b in sc.table_rows()]
    return "Use these resident tax rates, by taxable income: " + listing(rows)


def _tax_by_base(sc: Banded, income) -> Fraction:
    """The tax from the running total at the start of the band, then the band's own rate."""
    income = Fraction(income)
    i = sc.band_index(income)
    base = Fraction(0)
    for j in range(i):
        base += (sc.bands[j + 1][0] - sc.bands[j][0]) * sc.bands[j][1] / 100
    return base + (income - sc.bands[i][0]) * sc.bands[i][1] / 100


def _tax(sc: Banded, income) -> Fraction:
    value = sc.amount(income)
    check(value == _tax_by_base(sc, income), "tax disagrees between the two methods")
    return value


def _income_in(rng, sc: Banded, band: int, step: int = 100) -> int:
    low = int(sc.bands[band][0])
    high = int(sc.bands[band + 1][0]) if band + 1 < len(sc.bands) else low + 90000
    span = (high - low) // step
    return low + step * rng.randint(max(2, span // 6), max(3, span - span // 6))


def _slice_working(sc: Banded, income) -> list[str]:
    lines = []
    for i, (low, rate) in enumerate(sc.bands):
        high = sc.bands[i + 1][0] if i + 1 < len(sc.bands) else None
        if Fraction(income) <= low or rate == 0:
            continue
        top = Fraction(income) if high is None or Fraction(income) < high else high
        lines.append(f"{percent(rate)}% of {dollars(top - low)} = {dollars(sc.slice_amounts(income)[i], 2)}.")
    return lines


def _slips_for_tax(sc: Banded, income):
    income = Fraction(income)
    i = sc.band_index(income)
    rate = sc.bands[i][1]
    tax = sc.amount(income)
    slips = [
        (income * rate / 100, "r3e-rate-on-whole-income"),
        (income - sc.bands[i][0], "r3e-rate-not-applied-to-excess"),
    ]
    if i >= 2:
        slips.append((tax - sc.slice_amounts(income)[1], "r3e-middle-band-left-out"))
        slips.append(((income - sc.bands[i][0]) * rate / 100, "r3e-top-slice-only"))
    else:
        slips.append(((income - sc.bands[i][0]) * rate / 1000, "r3e-percent-decimal-slip"))
    slips.append((income * sc.bands[1][1] / 100, "r3e-first-rate-on-whole-income"))
    slips.append((sum(sc.slice_amounts(income)) + sc.bands[1][0] * sc.bands[1][1] / 100, "r3e-nil-band-taxed"))
    return slips


_EASY = (
    "{name}, {job}, has a taxable income of {income} for the year and is a resident for tax purposes. {scale} "
    "How much income tax is payable? Give your answer in dollars, to the nearest cent."
)
_MEDIUM_TAX = (
    "{name} is {job}. After deductions, {name}'s taxable income for the year is {income}. {scale} "
    "Find the income tax payable, in dollars, to the nearest cent."
)
_MEDIUM_LEFT = (
    "{name} is {job} with a taxable income of {income} for the year. {scale} "
    "How much of the taxable income is left once the income tax has been paid? Give your answer in dollars, to the "
    "nearest cent."
)


def _easy(rng):
    sc = _scale(rng)
    income = _income_in(rng, sc, 1)
    tax = _tax(sc, income)
    name, job = rng.choice(_PEOPLE)
    text = _EASY.format(name=name, job=job, income=dollars(income), scale=_scale_text(sc))
    return single_number(
        text, tax, 2, _slice_working(sc, income) + [f"Tax payable: {dollars(tax, 2)}."], _slips_for_tax(sc, income)
    )


def _medium(rng):
    sc = _scale(rng)
    income = _income_in(rng, sc, rng.choice([3, 3, 4]))
    tax = _tax(sc, income)
    name, job = rng.choice(_PEOPLE)
    working = _slice_working(sc, income) + [f"Tax payable: {dollars(tax, 2)}."]
    if rng.random() < 0.5:
        text = _MEDIUM_TAX.format(name=name, job=job, income=dollars(income), scale=_scale_text(sc))
        return single_number(text, tax, 2, working, _slips_for_tax(sc, income))
    text = _MEDIUM_LEFT.format(name=name, job=job, income=dollars(income), scale=_scale_text(sc))
    left = income - tax
    slips = [
        (tax, "r3e-gave-tax-not-remainder"),
        (income - income * sc.marginal_rate(income) / 100, "r3e-rate-on-whole-income"),
    ]
    slips += [(income - t, tag) for t, tag in _slips_for_tax(sc, income)]
    return single_number(
        text, left, 2, working + [f"{dollars(income)} - {dollars(tax, 2)} = {dollars(left, 2)}."], slips
    )


def _hard_net(rng):
    sc = _scale(rng)
    levy = rng.choice(_LEVY)
    taxable = _income_in(rng, sc, rng.choice([2, 3]))
    ded = 100 * rng.randint(12, 80)
    gross = taxable + ded
    tax = _tax(sc, taxable)
    lev = taxable * levy / 100
    net = taxable - tax - lev
    name, job = rng.choice(_PEOPLE)
    text = (
        f"{name} is {job} earning {dollars(gross)} for the year and claims {dollars(ded)} in work-related "
        f"deductions. Tax is paid on the income that is left after deductions. {_scale_text(sc)} "
        f"The Medicare levy is {percent(levy)}% of taxable income. How much of {name}'s taxable income remains after "
        "the income tax and the Medicare levy are paid? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (gross - tax - lev, "r3e-deductions-not-taken-off-first"),
        (taxable - tax, "r3e-levy-left-out"),
        (taxable - lev, "r3e-tax-left-out"),
        (taxable - tax - gross * levy / 100, "r3e-levy-on-gross"),
        (taxable - _tax(sc, gross) - lev, "r3e-tax-on-gross-income"),
    ]
    working = (
        [f"Taxable income = {dollars(gross)} - {dollars(ded)} = {dollars(taxable)}."]
        + _slice_working(sc, taxable)
        + [
            f"Income tax = {dollars(tax, 2)}; Medicare levy = {percent(levy)}% of {dollars(taxable)} = {dollars(lev, 2)}.",
            f"{dollars(taxable)} - {dollars(tax, 2)} - {dollars(lev, 2)} = {dollars(net, 2)}.",
        ]
    )
    return single_number(text, net, 2, working, slips)


def _hard_rise(rng):
    for _ in attempts(200, "a pay rise across a threshold"):
        sc = _scale(rng)
        band = rng.choice([1, 2, 3])
        edge = int(sc.bands[band + 1][0])
        below = 100 * rng.randint(5, 30)
        rise = below + 100 * rng.randint(5, 40)
        start = edge - below
        end = start + rise
        if end <= edge:
            continue
        if end > (int(sc.bands[band + 2][0]) if band + 2 < len(sc.bands) else 10**9):
            continue
        break
    extra = _tax(sc, end) - _tax(sc, start)
    r_low, r_high = sc.bands[band][1], sc.bands[band + 1][1]
    name, job = rng.choice(_PEOPLE)
    text = (
        f"{name} is {job} with a taxable income of {dollars(start)}. A promotion raises the taxable income by "
        f"{dollars(rise)}. {_scale_text(sc)} How much more income tax is payable after the promotion? Give your "
        "answer in dollars, to the nearest cent."
    )
    slips = [
        (rise * r_low / 100, "r3e-old-rate-on-whole-rise"),
        (rise * r_high / 100, "r3e-new-rate-on-whole-rise"),
        (end * r_high / 100 - start * r_low / 100, "r3e-rate-on-whole-income"),
        (extra * (100 - r_high) / 100, "r3e-after-tax-gain-given"),
    ]
    working = [
        f"The first {dollars(below)} of the rise is taxed at {percent(r_low)}% and the remaining {dollars(rise - below)} at {percent(r_high)}%.",
        f"Extra tax = {dollars(below * r_low / 100, 2)} + {dollars((rise - below) * r_high / 100, 2)} = {dollars(extra, 2)}.",
    ]
    return single_number(text, extra, 2, working, slips)


def _hard_reverse(rng):
    sc = _scale(rng)
    band = rng.choice([2, 3])
    income = _income_in(rng, sc, band, 500)
    tax = _tax(sc, income)
    r = sc.bands[band][1]
    name, job = rng.choice(_PEOPLE)
    text = (
        f"{name} is {job}. {name}'s income tax bill for the year was {dollars(tax, 2)}. {_scale_text(sc)} "
        f"What was {name}'s taxable income? Give your answer in dollars."
    )
    base = tax - (income - sc.bands[band][0]) * r / 100
    slips = [
        (tax * 100 / r, "r3e-rate-on-whole-income"),
        (sc.bands[band][0] + tax * 100 / r, "r3e-base-tax-ignored"),
        (sc.bands[band - 1][0] + (tax - base) * 100 / r, "r3e-wrong-band-threshold"),
        (
            tax * 100 / sc.bands[band + 1][1] if band + 1 < len(sc.bands) else tax * 100 / sc.bands[1][1],
            "r3e-wrong-band-rate",
        ),
    ]
    working = [
        f"Tax on income up to {dollars(sc.bands[band][0])} is {dollars(base, 2)}.",
        f"The rest, {dollars(tax - base, 2)}, is {percent(r)}% of the income above {dollars(sc.bands[band][0])}, "
        f"which is {dollars(income - sc.bands[band][0])}.",
        f"Taxable income = {dollars(sc.bands[band][0])} + {dollars(income - sc.bands[band][0])} = {dollars(income)}.",
    ]
    return single_number(text, income, 0, working, slips)


def _hard_average(rng):
    sc = _scale(rng)
    income = _income_in(rng, sc, rng.choice([2, 3]), 1000)
    tax = _tax(sc, income)
    avg = tax * 100 / income
    name, job = rng.choice(_PEOPLE)
    text = (
        f"{name} is {job} with a taxable income of {dollars(income)}. {_scale_text(sc)} "
        "What is the average rate of tax, as a percentage of taxable income? Give your answer to 1 decimal place."
    )
    slips = [
        (sc.marginal_rate(income), "r3e-marginal-rate-given-as-average"),
        (tax * 100 / (income - sc.bands[1][0]), "r3e-average-over-taxed-income"),
        (
            sum(r for _t, r in sc.bands[1 : sc.band_index(income) + 1]) / sc.band_index(income),
            "r3e-band-rates-averaged",
        ),
        (Fraction(tax, 1) / income, "r3e-ratio-not-percentage"),
    ]
    working = [
        f"Tax payable = {dollars(tax, 2)}.",
        f"Average rate = {dollars(tax, 2)} / {dollars(income)} x 100 = {number_text(avg, 1)}%.",
    ]
    return single_number(text, avg, 1, working, slips)


@generator(
    "r3e_income_tax",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_income_tax(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_net, _hard_rise, _hard_reverse, _hard_average])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_income_tax", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_income_tax(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    sc = _scale(rng)
    levy = rng.choice(_LEVY)
    taxable = _income_in(rng, sc, rng.choice([2, 3]))
    ded = 100 * rng.randint(12, 80)
    gross = taxable + ded
    tax = _tax(sc, taxable)
    lev = taxable * levy / 100
    left = taxable - tax - lev
    name, job = rng.choice(_PEOPLE)
    rate = sc.marginal_rate(taxable)
    text = (
        f"{name} is {job} and earned {dollars(gross)} this year. {name} can claim {dollars(ded)} of work-related "
        f"expenses as deductions. {name} is a resident for tax purposes. {_scale_text(sc)} "
        f"The Medicare levy is {percent(levy)}% of taxable income."
    )
    parts = [
        number_part(
            "Find the taxable income, in dollars.",
            taxable,
            2,
            1,
            [f"{dollars(gross)} - {dollars(ded)} = {dollars(taxable)}."],
            [
                (gross + ded, "r3e-deduction-added-to-income"),
                (gross, "r3e-deductions-ignored"),
                (ded, "r3e-deduction-taken-as-income"),
                (gross - ded / 10, "r3e-deduction-decimal-slip"),
            ],
        ),
        number_part(
            "Hence find the income tax payable, in dollars.",
            tax,
            2,
            3,
            _slice_working(sc, taxable) + [f"Income tax = {dollars(tax, 2)}."],
            [
                (taxable * rate / 100, "r3e-rate-on-whole-income"),
                (sc.amount(gross), "r3e-tax-on-gross-income"),
                (taxable - sc.bands[sc.band_index(taxable)][0], "r3e-rate-not-applied-to-excess"),
                (tax + sc.bands[1][0] * sc.bands[1][1] / 100, "r3e-nil-band-taxed"),
            ],
            follow_through=sc.expression("part_a"),
        ),
        number_part(
            "Find the Medicare levy, in dollars.",
            lev,
            2,
            1,
            [f"{percent(levy)}% of {dollars(taxable)} = {dollars(lev, 2)}."],
            [
                (tax * levy / 100, "r3e-levy-on-tax"),
                (gross * levy / 100, "r3e-levy-on-gross"),
                (taxable * levy / 1000, "r3e-percent-decimal-slip"),
            ],
            follow_through=f"{levy.numerator}/{levy.denominator * 100}*part_a",
        ),
        number_part(
            "How much of the taxable income remains after the income tax and the Medicare levy are paid? Give your answer in dollars.",
            left,
            2,
            2,
            [f"{dollars(taxable)} - {dollars(tax, 2)} - {dollars(lev, 2)} = {dollars(left, 2)}."],
            [
                (taxable - tax, "r3e-levy-left-out"),
                (taxable - lev, "r3e-tax-left-out"),
                (tax + lev, "r3e-gave-tax-not-remainder"),
                (gross - tax - lev, "r3e-deductions-not-taken-off-first"),
            ],
            follow_through="part_a - part_b - part_c",
        ),
    ]
    return dict(text=text, parts=parts)
