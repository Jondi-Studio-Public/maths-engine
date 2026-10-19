"""Generates the `w3_*` retail and pay topics: profit, loss and mark-up, and commission pay.

* ``profit-loss-and-markup`` (``w3_profit_markup``, exam ``exd_x_w3_profit_markup``): easy: the profit or loss in dollars,
  the selling price after a mark-up, the mark-up in dollars. Medium: the percentage profit or loss on cost, the total profit
  on a batch. Hard: the cost back from a selling price and a mark-up, a profit as a percentage of the SELLING price, a
  mark-up followed by a discount, the price that turns a stated loss into a stated profit.
* ``commission-pay`` (``w3_commission_pay``, exam ``exd_x_w3_commission_pay``): easy: the commission on a sale or on
  several sales, the commission rate. Medium: base pay plus commission, the sales for a stated commission or pay,
  commission on net sales, a comparison of two pay plans. Hard: tiered commission, the sales that earn a stated tiered
  commission, the sales at which two plans pay the same.

All amounts are built backwards from whole-dollar costs and whole-number percentages, so every answer is an exact amount.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._w3_kit import MONEY, WHOLE, art, dollars, pct_text
from ._w6a_kit import TooFewDistractors, enough, part, retry_draw

PM_TOPIC = "w3_profit_markup"
PM_EXAM = "exd_x_w3_profit_markup"
COM_TOPIC = "w3_commission_pay"
COM_EXAM = "exd_x_w3_commission_pay"

_PM = "profit-loss-and-markup"
_COM = "commission-pay"

MISCONCEPTIONS = [
    ("pm-adds-cost-and-price", "Adds the cost and the selling price instead of finding their difference.", _PM),
    ("pm-gives-selling-price", "Gives the selling price where the profit or loss was asked.", _PM),
    ("pm-gives-cost", "Gives the cost price where the profit or loss was asked.", _PM),
    ("pm-percent-for-dollars", "Gives the profit as a percentage where the amount in dollars was asked.", _PM),
    ("pm-markup-only", "Gives the mark-up in dollars where the selling price was asked.", _PM),
    ("pm-adds-percent-as-dollars", "Adds the percentage to the cost as if it were a number of dollars.", _PM),
    ("pm-takes-off", "Takes the percentage off the cost instead of adding it.", _PM),
    ("pm-rate-not-converted", "Multiplies the cost by the percentage without dividing by 100.", _PM),
    ("pm-profit-on-selling-price", "Finds the profit as a percentage of the selling price instead of the cost.", _PM),
    (
        "pm-forgets-subtraction",
        "Divides the selling price by the cost and gives that as the percentage profit, forgetting to subtract 100%.",
        _PM,
    ),
    ("pm-dollars-for-percent", "Gives the profit in dollars where a percentage was asked.", _PM),
    ("pm-loss-on-selling-price", "Finds the loss as a percentage of the selling price instead of the cost.", _PM),
    ("pm-ratio-inverted", "Divides the cost price by the selling price.", _PM),
    ("pm-over-average", "Divides the profit or loss by the average of the cost and the selling price.", _PM),
    ("pm-per-item-only", "Gives the profit on one item where the profit on the whole batch was asked.", _PM),
    ("pm-revenue-for-profit", "Gives the money taken from the sale of the batch where the profit was asked.", _PM),
    ("pm-total-cost-for-profit", "Gives the total cost of the batch where the profit was asked.", _PM),
    ("pm-percent-off-price", "Takes the mark-up percentage off the selling price to find the cost.", _PM),
    (
        "pm-cost-adds-markup",
        "Adds the mark-up percentage of the selling price to the selling price to find the cost.",
        _PM,
    ),
    ("pm-cost-from-markup-dollars", "Divides the selling price by the mark-up percentage.", _PM),
    (
        "pm-markup-for-margin",
        "Gives the mark-up on cost where the profit as a percentage of the selling price was asked.",
        _PM,
    ),
    (
        "pm-margin-over-cost",
        "Divides the profit by the cost when the profit as a percentage of the selling price was asked.",
        _PM,
    ),
    ("pm-margin-dollars", "Gives the profit in dollars where the percentage of the selling price was asked.", _PM),
    (
        "pm-percentages-added",
        "Adds the mark-up and subtracts the discount as percentages of the cost (for example +40% then -25% is +15%).",
        _PM,
    ),
    ("pm-discount-ignored", "Ignores the discount and gives the profit at the marked-up price.", _PM),
    ("pm-discount-only", "Gives the discount in dollars where the profit was asked.", _PM),
    ("pm-cost-from-loss-wrong", "Adds the loss percentage to the selling price to find the cost.", _PM),
    ("pm-profit-on-loss-price", "Adds the profit percentage to the reduced selling price instead of to the cost.", _PM),
    ("pm-price-for-cost", "Gives the cost price where the new selling price was asked.", _PM),
    (
        "pm-margin-price",
        "Adds the margin percentage to the cost to find the selling price, instead of dividing the cost by what is left.",
        _PM,
    ),
    ("com-rate-not-converted", "Multiplies the sales by the rate without dividing by 100.", _COM),
    ("com-divides-by-rate", "Divides the sales by the rate instead of finding that percentage of them.", _COM),
    ("com-decimal-slip", "Misplaces the decimal point when finding the commission.", _COM),
    ("com-sales-over-commission", "Divides the sales by the commission instead of the commission by the sales.", _COM),
    ("com-commission-over-100", "Divides the commission by 100 instead of by the sales.", _COM),
    ("com-largest-only", "Finds the commission on the largest sale only.", _COM),
    ("com-misses-one-sale", "Leaves one of the sales out of the total.", _COM),
    ("com-commission-only", "Gives the commission alone where the total pay was asked.", _COM),
    ("com-base-only", "Gives the base pay alone where the total pay was asked.", _COM),
    ("com-base-times-rate", "Finds the commission rate of the base pay instead of the sales.", _COM),
    ("com-rate-as-dollars", "Adds the commission rate to the base pay as a number of dollars.", _COM),
    ("com-multiplies-for-sales", "Multiplies the commission by the rate instead of dividing by it.", _COM),
    ("com-sales-from-total-pay", "Treats the whole pay as commission when finding the sales.", _COM),
    ("com-sales-adds-base", "Adds the base pay to the pay before finding the sales.", _COM),
    ("com-gross-sales", "Finds the commission on the gross sales, ignoring the returns.", _COM),
    ("com-returns-only", "Finds the commission on the returns instead of on the net sales.", _COM),
    ("com-plan-total-only", "Gives the pay under one plan where the difference between the plans was asked.", _COM),
    ("com-plan-sum", "Adds the pay under the two plans instead of finding the difference.", _COM),
    ("com-second-rate-everywhere", "Applies the higher rate to all the sales, ignoring the tier.", _COM),
    ("com-first-rate-everywhere", "Applies the lower rate to all the sales, ignoring the tier.", _COM),
    ("com-tier-excess-only", "Finds the commission on the sales above the tier only, forgetting the first tier.", _COM),
    ("com-second-rate-on-all-above", "Applies the higher rate to all the sales and adds the first tier as well.", _COM),
    ("com-inverse-first-rate", "Divides the commission by the first rate to find the sales, ignoring the tier.", _COM),
    (
        "com-inverse-second-rate",
        "Divides the commission by the higher rate to find the sales, ignoring the first tier.",
        _COM,
    ),
    (
        "com-inverse-adds-tier",
        "Adds the whole tier to sales found from the commission, without taking off the first tier's commission.",
        _COM,
    ),
    (
        "com-break-even-ignores-first",
        "Divides the base pay by the higher rate, ignoring the commission the other plan also pays.",
        _COM,
    ),
    ("com-break-even-sums-rates", "Divides the base pay by the sum of the rates instead of their difference.", _COM),
    (
        "com-break-even-no-hundred",
        "Divides the base pay by the difference in rates but forgets that the rates are percentages.",
        _COM,
    ),
    ("com-break-even-first-rate", "Divides the base pay by the lower rate alone.", _COM),
    ("com-tier-base-dropped", "Leaves the base pay out of the total.", _COM),
]

# (singular, plural, lowest and highest realistic cost in dollars)
_ITEMS = (
    ("jacket", "jackets", 40, 200),
    ("toaster", "toasters", 20, 80),
    ("tent", "tents", 60, 400),
    ("pair of boots", "pairs of boots", 40, 200),
    ("bicycle helmet", "bicycle helmets", 20, 120),
    ("coffee machine", "coffee machines", 100, 600),
    ("backpack", "backpacks", 20, 120),
    ("desk lamp", "desk lamps", 20, 100),
    ("cricket bat", "cricket bats", 60, 300),
    ("bookshelf", "bookshelves", 60, 300),
    ("skateboard", "skateboards", 40, 200),
    ("garden hose", "garden hoses", 20, 80),
)
_PLURAL = {name: plural for name, plural, _lo, _hi in _ITEMS}
_RANGE = {name: (lo, hi) for name, _p, lo, hi in _ITEMS}


def _item(rng, unit):
    """An item whose realistic price range holds several multiples of `unit`."""
    return rng.choice([name for name, _p, _lo, hi in _ITEMS if hi >= 3 * unit])


def _price(rng, item, unit) -> Fraction:
    """A realistic price for `item`, a whole multiple of `unit` dollars."""
    lo, hi = _RANGE[item]
    first = max(-(-lo // unit), 2)
    return Fraction(rng.randint(first, hi // unit) * unit)


def _mult(total_percent) -> str:
    """A percentage as the multiplier it stands for: 140 -> 1.4."""
    return f"{total_percent / 100:g}"


# -- profit and loss: easy ----------------------------------------------------------------------------------------------


def _pm_profit_loss(rng):
    item = _item(rng, 10)
    cost = _price(rng, item, 10)
    loss = rng.random() < 0.4
    gap = Fraction(rng.randint(2, min(30, int(cost) // 5 - 1)) * 5)
    sale = cost - gap if loss else cost + gap
    check(sale > 0, "sale price")
    word = "loss" if loss else "profit"
    d = MONEY.dist(
        gap,
        [
            (cost + sale, "pm-adds-cost-and-price"),
            (sale, "pm-gives-selling-price"),
            (cost, "pm-gives-cost"),
            (gap * 100 / cost, "pm-percent-for-dollars"),
            (gap * 10, "pm-rate-not-converted"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and sells it for {dollars(sale)}. "
        f"What is the {word} on the {item}, in dollars?"
    )
    working = [f"{word.capitalize()} = {dollars(max(cost, sale))} - {dollars(min(cost, sale))} = {dollars(gap)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(gap), answer_kind="number")


def _pm_markup_price(rng):
    item = _item(rng, 20)
    cost = _price(rng, item, 20)
    r = rng.choice(range(10, 101, 10))
    markup = cost * r / 100
    sale = cost + markup
    d = MONEY.dist(
        sale,
        [
            (markup, "pm-markup-only"),
            (cost + r, "pm-adds-percent-as-dollars"),
            (cost - markup, "pm-takes-off"),
            (cost * r, "pm-rate-not-converted"),
            (cost * r / 1000 + cost, "pm-markup-only"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and marks it up by {r}% of the cost price. "
        f"What is the selling price, in dollars?"
    )
    working = [
        f"Mark-up = {r}% of {dollars(cost)} = {dollars(markup)}.",
        f"Selling price = {dollars(cost)} + {dollars(markup)} = {dollars(sale)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(sale), answer_kind="number")


def _pm_markup_amount(rng):
    item = _item(rng, 20)
    cost = _price(rng, item, 20)
    r = rng.choice(range(10, 101, 10))
    markup = cost * r / 100
    d = MONEY.dist(
        markup,
        [
            (cost + markup, "pm-gives-selling-price"),
            (cost * r, "pm-rate-not-converted"),
            (markup / 10, "pm-rate-not-converted"),
            (cost / r, "pm-cost-from-markup-dollars"),
            (cost - markup, "pm-takes-off"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and adds a mark-up of {r}% of the cost price. "
        f"How much is the mark-up, in dollars?"
    )
    working = [f"Mark-up = {r}% of {dollars(cost)} = {dollars(markup)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(markup), answer_kind="number")


# -- medium -------------------------------------------------------------------------------------------------------------


def _pm_percent_profit(rng):
    item = _item(rng, 20)
    cost = _price(rng, item, 20)
    p = rng.choice(range(10, 101, 5))
    sale = cost * (100 + p) / 100
    profit = sale - cost
    d = WHOLE.dist(
        p,
        [
            (profit * 100 / sale, "pm-profit-on-selling-price"),
            (sale * 100 / cost, "pm-forgets-subtraction"),
            (cost * 100 / sale, "pm-ratio-inverted"),
            (profit * 200 / (cost + sale), "pm-over-average"),
            (profit, "pm-dollars-for-percent"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and sells it for {dollars(sale)}. "
        f"What is the percentage profit on the cost price?"
    )
    working = [
        f"Profit = {dollars(sale)} - {dollars(cost)} = {dollars(profit)}.",
        f"Percentage profit = {dollars(profit)} / {dollars(cost)} × 100 = {p}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(p), answer_kind="number")


def _pm_percent_loss(rng):
    item = _item(rng, 20)
    cost = _price(rng, item, 20)
    p = rng.choice(range(10, 61, 5))
    sale = cost * (100 - p) / 100
    loss = cost - sale
    d = WHOLE.dist(
        p,
        [
            (loss * 100 / sale, "pm-loss-on-selling-price"),
            (sale * 100 / cost, "pm-forgets-subtraction"),
            (cost * 100 / sale, "pm-ratio-inverted"),
            (loss * 200 / (cost + sale), "pm-over-average"),
            (loss, "pm-dollars-for-percent"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and sells it for {dollars(sale)}, which is less than it paid. "
        f"What is the percentage loss on the cost price?"
    )
    working = [
        f"Loss = {dollars(cost)} - {dollars(sale)} = {dollars(loss)}.",
        f"Percentage loss = {dollars(loss)} / {dollars(cost)} × 100 = {p}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(p), answer_kind="number")


def _pm_batch(rng):
    item = _item(rng, 5)
    n = rng.randint(6, 40)
    cost = _price(rng, item, 5)
    gap = Fraction(rng.randint(1, 12) * 5)
    sale = cost + gap
    total = n * gap
    d = MONEY.dist(
        total,
        [
            (gap, "pm-per-item-only"),
            (n * sale, "pm-revenue-for-profit"),
            (n * cost, "pm-total-cost-for-profit"),
            (n * (cost + sale), "pm-adds-cost-and-price"),
            (total * 10, "pm-rate-not-converted"),
        ],
    )
    enough(d, 3)
    text = (
        f"A trader buys {n} {_PLURAL[item]} at {dollars(cost)} each and sells every one of them for {dollars(sale)} each. "
        f"What is the total profit, in dollars?"
    )
    working = [
        f"Profit on one {item} = {dollars(sale)} - {dollars(cost)} = {dollars(gap)}.",
        f"Total profit = {n} × {dollars(gap)} = {dollars(total)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


# -- hard ---------------------------------------------------------------------------------------------------------------


def _pm_cost_back(rng):
    item = _item(rng, 20)
    r = rng.choice((10, 20, 25, 40, 50, 60, 80, 100))
    # the selling price is a whole-dollar amount and the cost works back to a whole-dollar amount
    cost = _price(rng, item, 20)
    sale = cost * (100 + r) / 100
    d = MONEY.dist(
        cost,
        [
            (sale * (100 - r) / 100, "pm-percent-off-price"),
            (sale * (100 + r) / 100, "pm-cost-adds-markup"),
            (sale / r, "pm-cost-from-markup-dollars"),
            (sale - r, "pm-adds-percent-as-dollars"),
            (sale - sale * r / 1000, "pm-percent-off-price"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop sells {art(item)} {item} for {dollars(sale)}. This price includes a mark-up of {r}% on the cost price. "
        f"What did the shop pay for the {item}, in dollars?"
    )
    working = [
        f"Selling price = {100 + r}% of the cost, so cost = {dollars(sale)} / {_mult(100 + r)} = {dollars(cost)}."
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(cost), answer_kind="number")


def _pm_margin(rng):
    item = _item(rng, 20)
    m = rng.choice((10, 20, 25, 30, 40, 50))
    sale = _price(rng, item, 20)
    cost = sale * (100 - m) / 100
    profit = sale - cost
    d = WHOLE.dist(
        m,
        [
            (profit * 100 / cost, "pm-margin-over-cost"),
            (profit, "pm-margin-dollars"),
            (sale * 100 / cost, "pm-forgets-subtraction"),
            (cost * 100 / sale, "pm-ratio-inverted"),
            (profit * 200 / (cost + sale), "pm-over-average"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and sells it for {dollars(sale)}. "
        f"The profit is what percentage of the selling price?"
    )
    working = [
        f"Profit = {dollars(profit)}.",
        f"As a percentage of the selling price: {dollars(profit)} / {dollars(sale)} × 100 = {m}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(m), answer_kind="number")


_MARK_DISCOUNT = [
    (r, dsc)
    for r in range(20, 101, 10)
    for dsc in range(10, 51, 5)
    if (100 + r) * (100 - dsc) % 100 == 0 and (100 + r) * (100 - dsc) // 100 != 100
]


def _pm_markup_discount(rng):
    item = _item(rng, 100)
    r, dsc = rng.choice(_MARK_DISCOUNT)
    cost = _price(rng, item, 100)
    marked = cost * (100 + r) / 100
    sale = marked * (100 - dsc) / 100
    check((sale * 100).denominator == 1 and (marked * 100).denominator == 1, "exact")
    net = sale - cost
    profit = net > 0
    gap = abs(net)
    word = "profit" if profit else "loss"
    d = MONEY.dist(
        gap,
        [
            (cost * abs(r - dsc) / 100, "pm-percentages-added"),
            (marked - cost, "pm-discount-ignored"),
            (marked * dsc / 100, "pm-discount-only"),
            (sale, "pm-gives-selling-price"),
            (gap * 10, "pm-rate-not-converted"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop buys {art(item)} {item} for {dollars(cost)} and marks it up by {r}% of the cost price. "
        f"During a sale the marked price is reduced by {dsc}%. What is the {word} on each {item} sold in the sale, in dollars?"
    )
    working = [
        f"Marked price = {dollars(cost)} × {_mult(100 + r)} = {dollars(marked)}.",
        f"Sale price = {dollars(marked)} × {_mult(100 - dsc)} = {dollars(sale)}.",
        f"{word.capitalize()} = {dollars(max(sale, cost))} - {dollars(min(sale, cost))} = {dollars(gap)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(gap), answer_kind="number")


def _pm_loss_to_profit(rng):
    item = _item(rng, 100)
    loss = rng.choice((10, 20, 25, 40, 50))
    gain = rng.choice((10, 20, 25, 30, 50))
    cost = _price(rng, item, 100)
    low = cost * (100 - loss) / 100
    target = cost * (100 + gain) / 100
    d = MONEY.dist(
        target,
        [
            (low * (100 + gain) / 100, "pm-profit-on-loss-price"),
            (cost, "pm-price-for-cost"),
            (low + low * loss / 100, "pm-cost-from-loss-wrong"),
            (target - cost, "pm-markup-only"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop sold {art(item)} {item} for {dollars(low)}, which was a loss of {loss}% on the cost price. "
        f"For what price should the shop have sold it to make a profit of {gain}% on the cost price, in dollars?"
    )
    working = [
        f"{dollars(low)} is {100 - loss}% of the cost, so cost = {dollars(low)} / {_mult(100 - loss)} = {dollars(cost)}.",
        f"Price for a {gain}% profit = {dollars(cost)} × {_mult(100 + gain)} = {dollars(target)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(target), answer_kind="number")


def _pm_margin_price(rng):
    item = _item(rng, 20)
    m = rng.choice((10, 20, 25, 40, 50))
    sale = _price(rng, item, 20)
    cost = sale * (100 - m) / 100
    d = MONEY.dist(
        sale,
        [
            (cost * (100 + m) / 100, "pm-margin-price"),
            (cost + m, "pm-adds-percent-as-dollars"),
            (cost * m / 100, "pm-markup-only"),
            (cost / m * 100, "pm-cost-from-markup-dollars"),
            (cost * (100 + m) / 100 + cost * m / 1000, "pm-margin-price"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop pays {dollars(cost)} for {art(item)} {item}. It wants the profit to be {m}% of the selling price. "
        f"What selling price gives that profit, in dollars?"
    )
    working = [
        f"The cost is {100 - m}% of the selling price, so price = {dollars(cost)} / {_mult(100 - m)} = {dollars(sale)}."
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(sale), answer_kind="number")


_PM_EASY = (_pm_profit_loss, _pm_markup_price, _pm_markup_amount)
_PM_MEDIUM = (_pm_percent_profit, _pm_percent_loss, _pm_batch)
_PM_HARD = (_pm_cost_back, _pm_margin, _pm_markup_discount, _pm_loss_to_profit, _pm_margin_price)


@generator(
    PM_TOPIC,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_profit_markup(rng, tier="easy"):
    """Profit, loss and mark-up; the tiers are described in the module docstring."""
    pool = {"easy": _PM_EASY, "medium": _PM_MEDIUM, "hard": _PM_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(PM_EXAM, level=17, presets={"exam": dict(tier="exam")})
def exd_x_w3_profit_markup(rng, tier="exam"):
    """Exam chain: marked price, sale price, profit on the sale, profit as a percentage of the cost."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    combos = [(r, dsc) for r, dsc in _MARK_DISCOUNT if (100 + r) * (100 - dsc) // 100 > 100]

    def build(rng):
        item = _item(rng, 100)
        r, dsc = rng.choice(combos)
        cost = _price(rng, item, 100)
        marked = cost * (100 + r) / 100
        sale = marked * (100 - dsc) / 100
        profit = sale - cost
        p = profit * 100 / cost
        check(p.denominator == 1 and profit > 0, "profit")
        a_d = MONEY.dist(
            marked,
            [
                (cost * r / 100, "pm-markup-only"),
                (cost + r, "pm-adds-percent-as-dollars"),
                (cost * (100 - r) / 100, "pm-takes-off"),
                (cost * r, "pm-rate-not-converted"),
            ],
        )
        b_d = MONEY.dist(
            sale,
            [
                (marked * dsc / 100, "pm-discount-only"),
                (marked - dsc, "pm-adds-percent-as-dollars"),
                (marked, "pm-discount-ignored"),
                (cost * (100 + r - dsc) / 100, "pm-percentages-added"),
            ],
        )
        c_d = MONEY.dist(
            profit,
            [
                (cost * (r - dsc) / 100, "pm-percentages-added"),
                (marked - cost, "pm-discount-ignored"),
                (marked * dsc / 100, "pm-discount-only"),
                (sale, "pm-gives-selling-price"),
            ],
        )
        d_d = WHOLE.dist(
            p,
            [
                (profit * 100 / sale, "pm-profit-on-selling-price"),
                (profit, "pm-dollars-for-percent"),
                (r - dsc, "pm-percentages-added"),
                (sale * 100 / cost, "pm-forgets-subtraction"),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = f"A shop buys {art(item)} {item} for {dollars(cost)}. It marks the {item} up by {r}% of the cost price."
        return dict(
            text=stem,
            parts=[
                part(
                    "Find the marked price, in dollars.",
                    "number",
                    MONEY.fields(marked),
                    1,
                    [f"Marked price = {dollars(cost)} × {_mult(100 + r)} = {dollars(marked)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"In a sale the marked price is reduced by {dsc}%. Find the sale price, in dollars.",
                    "number",
                    MONEY.fields(sale),
                    1,
                    [f"Sale price = {dollars(marked)} × {_mult(100 - dsc)} = {dollars(sale)}."],
                    b_d,
                    follow_through=f"part_a*{100 - dsc}/100",
                    sensible_range=(0, None),
                ),
                part(
                    "Find the profit made on each one sold in the sale, in dollars.",
                    "number",
                    MONEY.fields(profit),
                    2,
                    [f"Profit = {dollars(sale)} - {dollars(cost)} = {dollars(profit)}."],
                    c_d,
                    follow_through=f"part_b - {cost}",
                    sensible_range=(0, None),
                ),
                part(
                    "Express this profit as a percentage of the cost price.",
                    "number",
                    WHOLE.fields(p),
                    2,
                    [f"{dollars(profit)} / {dollars(cost)} × 100 = {pct_text(p)}%."],
                    d_d,
                    follow_through=f"part_c*100/{cost}",
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


# -- commission ---------------------------------------------------------------------------------------------------------

_ROLES = (
    "a car salesperson",
    "a real estate agent",
    "a furniture salesperson",
    "a phone-plan salesperson",
    "a travel agent",
    "an insurance agent",
    "a solar-panel salesperson",
)
_RATES = (2, 3, 4, 5, 6, 8, 10, 12, 15)


def _sales(rng, lo=10, hi=120, unit=500) -> Fraction:
    return Fraction(rng.randint(lo, hi) * unit)


def _com_simple(rng):
    role, r, s = rng.choice(_ROLES), rng.choice(_RATES), _sales(rng, 4, 80, 250)
    c = s * r / 100
    d = MONEY.dist(
        c,
        [
            (s * r, "com-rate-not-converted"),
            (s / r, "com-divides-by-rate"),
            (c / 10, "com-decimal-slip"),
            (c * 10, "com-decimal-slip"),
            (s - c, "com-commission-only"),
        ],
    )
    enough(d, 3)
    text = f"{role.capitalize()} is paid a commission of {r}% of the value of the sales made. In one week the sales total {dollars(s)}. What commission is earned, in dollars?"
    working = [f"Commission = {r}% of {dollars(s)} = {dollars(c)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(c), answer_kind="number")


def _com_rate(rng):
    role, r, s = rng.choice(_ROLES), rng.choice(_RATES), _sales(rng, 4, 80, 250)
    c = s * r / 100
    d = WHOLE.dist(
        r,
        [
            (s / c, "com-sales-over-commission"),
            (c / 100, "com-commission-over-100"),
            (c * 100 / (s - c), "com-sales-over-commission"),
            (r + 1, "com-decimal-slip"),
            (r * 10, "com-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} sells goods worth {dollars(s)} in a week and earns {dollars(c)} in commission. "
        f"What is the commission rate, as a percentage of the sales?"
    )
    working = [f"Rate = {dollars(c)} / {dollars(s)} × 100 = {r}%."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(r), answer_kind="number")


def _com_several(rng):
    role, r = rng.choice(_ROLES), rng.choice(_RATES)
    sales = [_sales(rng, 4, 40, 250) for _ in range(3)]
    total = sum(sales)
    c = total * r / 100
    d = MONEY.dist(
        c,
        [
            (max(sales) * r / 100, "com-largest-only"),
            ((total - min(sales)) * r / 100, "com-misses-one-sale"),
            ((total - sales[0]) * r / 100, "com-misses-one-sale"),
            (total * r, "com-rate-not-converted"),
            (total / r, "com-divides-by-rate"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} earns {r}% commission on every sale. In one week the sales are {dollars(sales[0])}, {dollars(sales[1])} and {dollars(sales[2])}. "
        f"What commission is earned for the week, in dollars?"
    )
    working = [f"Total sales = {dollars(total)}.", f"Commission = {r}% of {dollars(total)} = {dollars(c)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(c), answer_kind="number")


def _com_total_pay(rng):
    role, r, s = rng.choice(_ROLES), rng.choice(_RATES), _sales(rng, 4, 80, 250)
    base = Fraction(rng.randint(8, 30) * 50)
    c = s * r / 100
    pay = base + c
    d = MONEY.dist(
        pay,
        [
            (c, "com-commission-only"),
            (base, "com-base-only"),
            (base + s * r, "com-rate-not-converted"),
            (base * r / 100 + base, "com-base-times-rate"),
            (base + r, "com-rate-as-dollars"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} is paid {dollars(base)} a week plus a commission of {r}% of the sales made. In one week the sales total {dollars(s)}. "
        f"What is the total pay for the week, in dollars?"
    )
    working = [
        f"Commission = {r}% of {dollars(s)} = {dollars(c)}.",
        f"Pay = {dollars(base)} + {dollars(c)} = {dollars(pay)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(pay), answer_kind="number")


def _com_sales_needed(rng):
    role, r = rng.choice(_ROLES), rng.choice(_RATES)
    s = _sales(rng, 4, 80, 250)
    base = Fraction(rng.randint(8, 30) * 50)
    c = s * r / 100
    pay = base + c
    d = MONEY.dist(
        s,
        [
            (pay * 100 / r, "com-sales-from-total-pay"),
            ((pay + base) * 100 / r, "com-sales-adds-base"),
            (c * r / 100, "com-multiplies-for-sales"),
            (c / r, "com-divides-by-rate"),
            ((pay - base) * 10 / r, "com-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} is paid {dollars(base)} a week plus a commission of {r}% of the sales made. One week the total pay is {dollars(pay)}. "
        f"What were the sales for the week, in dollars?"
    )
    working = [
        f"Commission = {dollars(pay)} - {dollars(base)} = {dollars(c)}.",
        f"Sales = {dollars(c)} / {r}% = {dollars(s)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(s), answer_kind="number")


def _com_net(rng):
    role, r = rng.choice(_ROLES), rng.choice(_RATES)
    s = _sales(rng, 10, 80, 250)
    ret = Fraction(rng.randint(1, 8) * 250)
    c = (s - ret) * r / 100
    d = MONEY.dist(
        c,
        [
            (s * r / 100, "com-gross-sales"),
            (ret * r / 100, "com-returns-only"),
            ((s + ret) * r / 100, "com-gross-sales"),
            ((s - ret) * r, "com-rate-not-converted"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} earns {r}% commission on net sales (sales less goods returned). In one month the sales are {dollars(s)} and goods worth {dollars(ret)} are returned. "
        f"What commission is earned, in dollars?"
    )
    working = [
        f"Net sales = {dollars(s)} - {dollars(ret)} = {dollars(s - ret)}.",
        f"Commission = {r}% of {dollars(s - ret)} = {dollars(c)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(c), answer_kind="number")


def _com_plans(rng):
    for _ in attempts(100, "plans where A pays more"):
        role = rng.choice(_ROLES)
        r1, r2 = rng.choice((2, 3, 4, 5)), rng.choice((6, 8, 10))
        base = Fraction(rng.randint(8, 30) * 50)
        s = _sales(rng, 4, 40, 250)
        a, b = base + s * r1 / 100, s * r2 / 100
        if a > b:
            break
    else:
        raise TooFewDistractors("plans")
    diff = a - b
    d = MONEY.dist(
        diff,
        [
            (a, "com-plan-total-only"),
            (b, "com-plan-total-only"),
            (a + b, "com-plan-sum"),
            (diff * 10, "com-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} can choose Plan A, {dollars(base)} a week plus {r1}% commission on sales, or Plan B, {r2}% commission on sales and no weekly payment. "
        f"In a week with sales of {dollars(s)}, how much more does Plan A pay than Plan B, in dollars?"
    )
    working = [
        f"Plan A = {dollars(base)} + {r1}% of {dollars(s)} = {dollars(a)}.",
        f"Plan B = {r2}% of {dollars(s)} = {dollars(b)}.",
        f"Difference = {dollars(diff)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(diff), answer_kind="number")


def _tier_case(rng):
    r1 = rng.choice((2, 3, 4, 5))
    r2 = rng.choice([x for x in (6, 8, 10, 12) if x > r1 + 1])
    tier = Fraction(rng.randint(8, 40) * 1000)
    return r1, r2, tier


def _com_tiered(rng):
    role = rng.choice(_ROLES)
    r1, r2, tier = _tier_case(rng)
    s = tier + Fraction(rng.randint(2, 30) * 500)
    c = tier * r1 / 100 + (s - tier) * r2 / 100
    d = MONEY.dist(
        c,
        [
            (s * r2 / 100, "com-second-rate-everywhere"),
            (s * r1 / 100, "com-first-rate-everywhere"),
            ((s - tier) * r2 / 100, "com-tier-excess-only"),
            (tier * r1 / 100 + s * r2 / 100, "com-second-rate-on-all-above"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} earns {r1}% commission on the first {dollars(tier)} of sales in a month and {r2}% on all sales above that. "
        f"In one month the sales total {dollars(s)}. What commission is earned, in dollars?"
    )
    working = [
        f"First {dollars(tier)}: {r1}% = {dollars(tier * r1 / 100)}.",
        f"Remaining {dollars(s - tier)}: {r2}% = {dollars((s - tier) * r2 / 100)}.",
        f"Commission = {dollars(c)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(c), answer_kind="number")


def _com_tiered_inverse(rng):
    role = rng.choice(_ROLES)
    r1, r2, tier = _tier_case(rng)
    s = tier + Fraction(rng.randint(2, 30) * 500)
    c = tier * r1 / 100 + (s - tier) * r2 / 100
    d = MONEY.dist(
        s,
        [
            (c * 100 / r2, "com-inverse-second-rate"),
            (c * 100 / r1, "com-inverse-first-rate"),
            (tier + c * 100 / r2, "com-inverse-adds-tier"),
            (tier + (c - tier * r1 / 100) * 100 / r1, "com-first-rate-everywhere"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} earns {r1}% commission on the first {dollars(tier)} of sales in a month and {r2}% on all sales above that. "
        f"One month the commission earned is {dollars(c)}. What were the sales for the month, in dollars?"
    )
    working = [
        f"Commission on the first {dollars(tier)} = {dollars(tier * r1 / 100)}.",
        f"Remaining commission = {dollars(c)} - {dollars(tier * r1 / 100)} = {dollars(c - tier * r1 / 100)}, which is {r2}% of the sales above {dollars(tier)}.",
        f"Sales above the tier = {dollars(s - tier)}; total sales = {dollars(s)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(s), answer_kind="number")


def _com_break_even(rng):
    role = rng.choice(_ROLES)
    r1 = rng.choice((2, 3, 4, 5))
    r2 = rng.choice([x for x in (6, 8, 10, 12) if x > r1])
    gap = r2 - r1
    s = Fraction(rng.randint(2, 40) * 500)
    base = s * gap / 100
    if (base * 100).denominator != 1 or not 300 <= base <= 1500:
        raise TooFewDistractors("base")
    d = MONEY.dist(
        s,
        [
            (base * 100 / r2, "com-break-even-ignores-first"),
            (base * 100 / (r1 + r2), "com-break-even-sums-rates"),
            (base / gap, "com-break-even-no-hundred"),
            (base * 100 / r1, "com-break-even-first-rate"),
        ],
    )
    enough(d, 3)
    text = (
        f"{role.capitalize()} can choose Plan A, {dollars(base)} a week plus {r1}% commission on sales, or Plan B, {r2}% commission on sales and no weekly payment. "
        f"At what weekly sales do the two plans pay the same, in dollars?"
    )
    working = [
        f"Equal pay: {dollars(base)} + {r1}% of S = {r2}% of S.",
        f"{dollars(base)} = {gap}% of S, so S = {dollars(base)} / {_mult(gap)} = {dollars(s)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(s), answer_kind="number")


_COM_EASY = (_com_simple, _com_rate, _com_several)
_COM_MEDIUM = (_com_total_pay, _com_sales_needed, _com_net, _com_plans)
_COM_HARD = (_com_tiered, _com_tiered_inverse, _com_break_even)


@generator(
    COM_TOPIC,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_commission_pay(rng, tier="easy"):
    """Commission pay; the tiers are described in the module docstring."""
    pool = {"easy": _COM_EASY, "medium": _COM_MEDIUM, "hard": _COM_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(COM_EXAM, level=17, presets={"exam": dict(tier="exam")})
def exd_x_w3_commission_pay(rng, tier="exam"):
    """Exam chain on a tiered plan: pay below the tier, pay above it, sales for a stated pay, comparison with a flat plan."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        role = rng.choice(_ROLES)
        r1, r2, tier_amount = _tier_case(rng)
        base = Fraction(rng.randint(8, 30) * 50)
        s1 = Fraction(int(tier_amount) // 1000 * rng.randint(3, 9) * 100)
        s1 = min(s1, tier_amount - Fraction(500))
        s2 = tier_amount + Fraction(rng.randint(2, 30) * 500)
        s3 = tier_amount + Fraction(rng.randint(2, 30) * 500)
        flat = rng.choice([x for x in (4, 5, 6, 8) if x < r2])
        pay1 = base + s1 * r1 / 100
        pay2 = base + tier_amount * r1 / 100 + (s2 - tier_amount) * r2 / 100
        pay3 = base + tier_amount * r1 / 100 + (s3 - tier_amount) * r2 / 100
        diff = pay2 - s2 * flat / 100
        if diff <= 0 or s1 <= 0:
            raise TooFewDistractors("flat plan pays more")
        a_d = MONEY.dist(
            pay1,
            [
                (s1 * r1 / 100, "com-commission-only"),
                (base, "com-base-only"),
                (base + s1 * r1, "com-rate-not-converted"),
                (base + r1, "com-rate-as-dollars"),
            ],
        )
        b_d = MONEY.dist(
            pay2,
            [
                (base + s2 * r2 / 100, "com-second-rate-everywhere"),
                (base + s2 * r1 / 100, "com-first-rate-everywhere"),
                (base + (s2 - tier_amount) * r2 / 100, "com-tier-excess-only"),
                (tier_amount * r1 / 100 + (s2 - tier_amount) * r2 / 100, "com-tier-base-dropped"),
            ],
        )
        c_d = MONEY.dist(
            s3,
            [
                ((pay3 - base) * 100 / r2, "com-inverse-second-rate"),
                ((pay3 - base) * 100 / r1, "com-inverse-first-rate"),
                (tier_amount + (pay3 - base) * 100 / r2, "com-inverse-adds-tier"),
                (pay3 * 100 / r2, "com-sales-from-total-pay"),
            ],
        )
        d_d = MONEY.dist(
            diff,
            [
                (pay2, "com-plan-total-only"),
                (s2 * flat / 100, "com-plan-total-only"),
                (pay2 + s2 * flat / 100, "com-plan-sum"),
                (diff * 10, "com-decimal-slip"),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = (
            f"{role.capitalize()} is paid {dollars(base)} a week plus a commission of {r1}% on the first {dollars(tier_amount)} of weekly sales "
            f"and {r2}% on all weekly sales above that."
        )
        return dict(
            text=stem,
            parts=[
                part(
                    f"Find the pay for a week in which the sales are {dollars(s1)}, in dollars.",
                    "number",
                    MONEY.fields(pay1),
                    1,
                    [f"Pay = {dollars(base)} + {r1}% of {dollars(s1)} = {dollars(pay1)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"Find the pay for a week in which the sales are {dollars(s2)}, in dollars.",
                    "number",
                    MONEY.fields(pay2),
                    2,
                    [
                        f"Commission on the first {dollars(tier_amount)} = {dollars(tier_amount * r1 / 100)}; on the remaining {dollars(s2 - tier_amount)} = {dollars((s2 - tier_amount) * r2 / 100)}.",
                        f"Pay = {dollars(base)} + {dollars(tier_amount * r1 / 100)} + {dollars((s2 - tier_amount) * r2 / 100)} = {dollars(pay2)}.",
                    ],
                    b_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"In another week the pay is {dollars(pay3)}. Find the sales for that week, in dollars.",
                    "number",
                    MONEY.fields(s3),
                    2,
                    [
                        f"Commission = {dollars(pay3)} - {dollars(base)} = {dollars(pay3 - base)}; first tier gives {dollars(tier_amount * r1 / 100)}.",
                        f"Sales above the tier = {dollars(pay3 - base - tier_amount * r1 / 100)} / {r2}% = {dollars(s3 - tier_amount)}; sales = {dollars(s3)}.",
                    ],
                    c_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"A flat plan pays {flat}% of all sales and no weekly payment. With sales of {dollars(s2)}, how much more does the plan above pay, in dollars?",
                    "number",
                    MONEY.fields(diff),
                    2,
                    [
                        f"Flat plan = {flat}% of {dollars(s2)} = {dollars(s2 * flat / 100)}; difference = {dollars(pay2)} - {dollars(s2 * flat / 100)} = {dollars(diff)}."
                    ],
                    d_d,
                    follow_through=f"part_b - {s2 * flat / 100}",
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)
