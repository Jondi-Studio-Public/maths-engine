"""GST on invoices and the business activity statement: `r3e_gst_invoice_bas` (easy, medium, hard) and the exam chain
`exd_x_r3e_gst_invoice_bas`.

``gst-invoice-bas`` (``financial-discrete``, answer ``number``). The GST rate is printed in each question (it is
10% of the GST-exclusive price, so a GST-inclusive amount contains 1/11 of itself as GST) and each question says
which prices include GST and which items are GST-free.

* easy: the GST contained in a GST-inclusive total.
* medium: the total payable on an invoice in which some lines are taxable and some are GST-free.
* hard: the GST in a quote that mixes a GST-inclusive materials price with a GST-exclusive labour price; or the net
  GST payable on a short business activity statement (GST collected less GST credits).
* exam (``exd_x_r3e_gst_invoice_bas``): GST collected on sales, GST credits on purchases, the net GST payable, and
  the total sales excluding GST.

Every answer is rebuilt a second way, by converting each inclusive amount to its exclusive price and taking 10%.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import dollars, listing, number_part, single_number

_RATE = Fraction(10)

_TRADES = [
    ("a plumber", "replacing a hot-water system"),
    ("an electrician", "rewiring a kitchen"),
    ("a landscaper", "building a retaining wall"),
    ("a painter", "painting a two-storey house"),
    ("a carpenter", "building a deck"),
    ("a tiler", "retiling a bathroom"),
]
_OWNERS = [
    ("Mei", "a bakery"),
    ("Tom", "a corner cafe"),
    ("Aisha", "a catering business"),
    ("Liam", "a market stall"),
    ("Chloe", "a sandwich shop"),
    ("Hamid", "a juice bar"),
    ("Nina", "a food truck"),
]
_TAXABLE_LINES = ["printer cartridges", "stationery", "cleaning supplies", "tools", "packaging", "equipment hire"]
_FREE_LINES = ["fresh vegetables", "bread", "milk", "first-aid supplies", "fresh fruit", "eggs"]


def _gst_in(inclusive):
    gst = Fraction(inclusive) / 11
    check(gst == Fraction(inclusive) / (1 + _RATE / 100) * _RATE / 100, "GST in an inclusive amount disagrees")
    return gst


def _easy(rng):
    trade, job = rng.choice(_TRADES)
    exclusive = 10 * rng.randint(20, 900)
    total = exclusive * (100 + _RATE) / 100
    gst = _gst_in(total)
    text = (
        f"A customer pays {dollars(total, 2)} to {trade} for {job}. The price includes GST of 10%. How much of the "
        "price is GST? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (total * _RATE / 100, "r3e-gst-taken-of-inclusive-price"),
        (total - gst, "r3e-exclusive-price-given-as-gst"),
        (total * 11 / 10, "r3e-gst-added-to-inclusive-price"),
        (total * _RATE / 1000, "r3e-percent-decimal-slip"),
    ]
    working = [
        f"The price is the GST-exclusive price plus 10%, so it is 110% of the exclusive price: {dollars(total, 2)} / 1.1 = {dollars(exclusive, 2)}.",
        f"GST = {dollars(total, 2)} - {dollars(exclusive, 2)} = {dollars(gst, 2)}.",
    ]
    return single_number(text, gst, 2, working, slips)


def _invoice_lines(rng):
    taxable = rng.sample(_TAXABLE_LINES, 2)
    free = rng.sample(_FREE_LINES, 2)
    amounts = {name: Fraction(5 * rng.randint(8, 90)) for name in taxable + free}
    return taxable, free, amounts


def _medium(rng):
    taxable, free, amounts = _invoice_lines(rng)
    order = taxable + free
    rng.shuffle(order)
    rows = [(item, dollars(amounts[item], 2)) for item in order]
    sub_t = sum(amounts[i] for i in taxable)
    sub_f = sum(amounts[i] for i in free)
    gst = sub_t * _RATE / 100
    total = sub_t + sub_f + gst
    name, biz = rng.choice(_OWNERS)
    text = (
        f"{name} owns {biz}. A supplier's invoice lists these prices, all excluding GST: "
        f"{listing(rows)} "
        f"GST of 10% applies to the {taxable[0]} and the {taxable[1]}. The {free[0]} and the {free[1]} are GST-free. "
        "What is the total amount payable on the invoice? Give your answer in dollars."
    )
    allsum = sub_t + sub_f
    slips = [
        (allsum * (100 + _RATE) / 100, "r3e-gst-charged-on-gst-free-items"),
        (allsum, "r3e-gst-left-out"),
        (allsum + sub_t / 11, "r3e-gst-treated-as-inclusive"),
        (allsum + sub_f * _RATE / 100, "r3e-gst-charged-on-gst-free-items"),
        (sub_t * (100 + _RATE) / 100, "r3e-gst-free-items-left-out"),
    ]
    working = [
        f"Taxable items: {dollars(sub_t, 2)}. GST = 10% of {dollars(sub_t, 2)} = {dollars(gst, 2)}.",
        f"GST-free items: {dollars(sub_f, 2)}.",
        f"Total = {dollars(sub_t, 2)} + {dollars(sub_f, 2)} + {dollars(gst, 2)} = {dollars(total, 2)}.",
    ]
    return single_number(text, total, 2, working, slips)


def _hard_quote(rng):
    trade, job = rng.choice(_TRADES)
    parts_incl = 11 * rng.randint(30, 400)
    labour_excl = 10 * rng.randint(30, 300)
    gst = Fraction(parts_incl) / 11 + Fraction(labour_excl) * _RATE / 100
    total = parts_incl + labour_excl + Fraction(labour_excl) * _RATE / 100
    text = (
        f"{trade.capitalize()} quotes for {job}. The materials cost {dollars(parts_incl, 2)}, which already includes "
        f"GST of 10%. The labour is {dollars(labour_excl, 2)} before GST, and GST of 10% is added to it. How much GST "
        "does the customer pay in total? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(parts_incl) * _RATE / 100 + Fraction(labour_excl) * _RATE / 100, "r3e-gst-taken-of-inclusive-price"),
        (Fraction(labour_excl) * _RATE / 100, "r3e-materials-gst-left-out"),
        (Fraction(parts_incl) / 11, "r3e-labour-gst-left-out"),
        ((Fraction(parts_incl) + labour_excl) / 11, "r3e-labour-treated-as-inclusive"),
    ]
    working = [
        f"GST in the materials = {dollars(parts_incl, 2)} / 11 = {dollars(Fraction(parts_incl) / 11, 2)}.",
        f"GST on the labour = 10% of {dollars(labour_excl, 2)} = {dollars(Fraction(labour_excl) * _RATE / 100, 2)}.",
        f"Total GST = {dollars(gst, 2)}.",
    ]
    check(total - parts_incl - labour_excl == Fraction(labour_excl) / 10, "labour GST disagrees")
    return single_number(text, gst, 2, working, slips)


def _hard_bas(rng):
    name, biz = rng.choice(_OWNERS)
    sales = 11 * rng.randint(900, 4000)
    free_sales = 100 * rng.randint(5, 40)
    purchases = 11 * rng.randint(200, 850)
    collected = Fraction(sales) / 11
    credits = Fraction(purchases) / 11
    net = collected - credits
    text = (
        f"{name} owns {biz} and is completing a business activity statement. For the quarter, taxable sales "
        f"including GST were {dollars(sales)} and GST-free sales were {dollars(free_sales)}. Business purchases "
        f"including GST were {dollars(purchases)}. GST is 10%. What is the net GST payable to the tax office? Give "
        "your answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(sales + free_sales) / 11 - credits, "r3e-gst-charged-on-gst-free-items"),
        (Fraction(sales) * _RATE / 100 - Fraction(purchases) * _RATE / 100, "r3e-gst-taken-of-inclusive-price"),
        (collected, "r3e-credits-left-out"),
        (collected + credits, "r3e-credits-added"),
        (credits - collected, "r3e-gst-subtracted-wrong-way"),
    ]
    working = [
        f"GST collected = {dollars(sales)} / 11 = {dollars(collected, 2)}.",
        f"GST credits = {dollars(purchases)} / 11 = {dollars(credits, 2)}.",
        f"Net GST payable = {dollars(collected, 2)} - {dollars(credits, 2)} = {dollars(net, 2)}.",
    ]
    return single_number(text, net, 2, working, slips)


@generator(
    "r3e_gst_invoice_bas",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_gst_invoice_bas(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_quote, _hard_bas])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_gst_invoice_bas", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_gst_invoice_bas(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name, biz = rng.choice(_OWNERS)
    sales = 11 * rng.randint(1200, 5000)
    free_sales = 100 * rng.randint(5, 40)
    purchases = 11 * rng.randint(300, 1100)
    free_purchases = 100 * rng.randint(3, 25)
    collected = Fraction(sales) / 11
    credits = Fraction(purchases) / 11
    net = collected - credits
    ex_sales = Fraction(sales) - collected + free_sales
    check(ex_sales == Fraction(sales) * 10 / 11 + free_sales, "sales excluding GST disagree")
    text = (
        f"{name} owns {biz}. In one quarter the business had taxable sales of {dollars(sales)} "
        f"including GST and GST-free sales of {dollars(free_sales)}. It bought {dollars(purchases)} of goods "
        f"and services for the business including GST, and {dollars(free_purchases)} of GST-free goods. GST is 10% of "
        "the GST-exclusive price."
    )
    parts = [
        number_part(
            "Find the GST collected on the taxable sales, in dollars.",
            collected,
            2,
            2,
            [f"{dollars(sales)} / 11 = {dollars(collected, 2)}."],
            [
                (Fraction(sales) * _RATE / 100, "r3e-gst-taken-of-inclusive-price"),
                (Fraction(sales + free_sales) / 11, "r3e-gst-charged-on-gst-free-items"),
                (Fraction(sales) - collected, "r3e-exclusive-price-given-as-gst"),
            ],
        ),
        number_part(
            "Find the GST credits for the business purchases, in dollars.",
            credits,
            2,
            2,
            [f"{dollars(purchases)} / 11 = {dollars(credits, 2)}."],
            [
                (Fraction(purchases) * _RATE / 100, "r3e-gst-taken-of-inclusive-price"),
                (Fraction(purchases + free_purchases) / 11, "r3e-gst-charged-on-gst-free-items"),
                (Fraction(purchases) - credits, "r3e-exclusive-price-given-as-gst"),
            ],
        ),
        number_part(
            "Hence find the net GST payable, in dollars.",
            net,
            2,
            1,
            [f"{dollars(collected, 2)} - {dollars(credits, 2)} = {dollars(net, 2)}."],
            [
                (collected + credits, "r3e-credits-added"),
                (collected, "r3e-credits-left-out"),
                (credits - collected, "r3e-gst-subtracted-wrong-way"),
            ],
            follow_through="part_a - part_b",
        ),
        number_part(
            "The statement also asks for total sales excluding GST. Find it, in dollars.",
            ex_sales,
            2,
            2,
            [
                f"Taxable sales excluding GST = {dollars(sales)} - {dollars(collected, 2)} = {dollars(sales - collected, 2)}.",
                f"Total = {dollars(sales - collected, 2)} + {dollars(free_sales)} = {dollars(ex_sales, 2)}.",
            ],
            [
                (Fraction(sales + free_sales), "r3e-gst-not-removed-from-sales"),
                (Fraction(sales) - collected, "r3e-gst-free-sales-left-out"),
                (Fraction(sales + free_sales) - collected, "r3e-gst-free-sales-treated-as-taxable"),
                (Fraction(sales) + free_sales - Fraction(sales) / 10, "r3e-gst-taken-of-inclusive-price"),
            ],
            follow_through=f"{sales} - part_a + {free_sales}",
        ),
    ]
    return dict(text=text, parts=parts)
