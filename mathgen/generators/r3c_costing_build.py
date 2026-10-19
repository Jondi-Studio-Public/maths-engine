"""Generates `matrix-costing-model` questions (easy, medium, hard) and the exam chain `r3c_costing_chain`.

A price list is a row matrix ``P`` and the quantities each branch orders are the columns of a quantity matrix ``Q``
(a row for each item), so the cost of each branch's order is the row ``P Q``. The prompt gives the prices and the
quantities as matrices and asks for money, so the learner must set up the product the right way round.

* easy: the cost of one branch's order (a price row times one column), in dollars;
* medium: the cost of all the branches' orders together, in dollars;
* hard: two stages, ending with the cost for one named outlet. Each product is made from ingredients in a recipe matrix ``R`` (a row for each ingredient, a
  column for each product), the ingredients have a price per kilogram, and each outlet orders batches of each product.
  The ingredient cost for every outlet is ``P R Q``;
* exam (``r3c_costing_chain``): the cost of one batch of each of two products from the recipe matrix, the cost of an
  order of both (carried forward) and the profit on it (carried forward again).

All prices are in dollars and cents; every quantity is a whole number of items or a half-kilogram multiple.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r3c_kit import number_part, number_single

_BRANCHES = ("Geelong", "Bendigo", "Ballarat", "Mildura", "Cairns", "Townsville", "Mackay", "Toowoomba")

_SUPPLIES = (
    dict(
        intro="A school canteen supplier sells these items to its branches.",
        items=("sandwiches", "fruit cups", "juice bottles", "muffins"),
        price=((550, 800), (250, 400), (150, 300), (200, 350)),
        unit="order",
    ),
    dict(
        intro="A hardware chain supplies these items to its stores.",
        items=("tins of paint", "brushes", "rollers", "drop sheets"),
        price=((2500, 4000), (600, 1200), (900, 1500), (800, 1400)),
        unit="order",
    ),
    dict(
        intro="A uniform company supplies these items to its outlets.",
        items=("shirts", "shorts", "hats", "jumpers"),
        price=((2200, 3200), (1800, 2800), (900, 1500), (3500, 5000)),
        unit="order",
    ),
)


def _money(cents) -> str:
    cents = Fraction(cents)
    dollars = cents / 100
    return f"{float(dollars):.2f}"


def _price_text(row) -> str:
    return "[" + ", ".join(_money(c) for c in row) + "]"


def _qty_text(Q) -> str:
    return "[" + ", ".join("[" + ", ".join(str(int(v)) for v in row) + "]" for row in Q) + "]"


def _cost(price_cents, quantities):
    """Cost in cents of a list of quantities at a list of prices in cents."""
    return sum(Fraction(p) * Fraction(q) for p, q in zip(price_cents, quantities))


def _draw_prices(rng, ctx, k):
    return [rng.randrange(lo, hi + 1, 50) for lo, hi in ctx["price"][:k]]


def _setup(rng, k, m):
    ctx = rng.choice(_SUPPLIES)
    items = list(ctx["items"][:k])
    branches = rng.sample(_BRANCHES, m)
    prices = _draw_prices(rng, ctx, k)
    Q = [[rng.randint(2, 40) for _ in range(m)] for _ in range(k)]
    return ctx, items, branches, prices, Q


def _far(key, slips, gap=Fraction(1, 50)):
    """The slips that sit at least ``gap`` of the key away from it, so no option is a near miss by accident."""
    return [(v, t) for v, t in slips if v is not None and abs(Fraction(v) - Fraction(key)) >= abs(Fraction(key)) * gap]


def _cols(Q):
    return [[Q[i][j] for i in range(len(Q))] for j in range(len(Q[0]))]


def _list_text(names):
    names = list(names)
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]


def _easy(rng):
    for _ in attempts(300, "an easy costing question"):
        k, m = rng.choice([(3, 3), (3, 4), (4, 3)])
        ctx, items, branches, prices, Q = _setup(rng, k, m)
        cols = _cols(Q)
        j = rng.randrange(m)
        cost = _cost(prices, cols[j]) / 100
        if len(set(sum(p * q for p, q in zip(prices, c)) for c in cols)) < m:
            continue
        text = (
            f"{ctx['intro']} The price in dollars of one each of {_list_text(items)}, in that order, is the row "
            f"P = {_price_text(prices)}. The quantities ordered are given by the matrix Q = {_qty_text(Q)}, which has a row "
            f"for each item (in the same order) and a column for each of {_list_text(branches)} (in that order). "
            f"How much does the {branches[j]} branch pay for its order, in dollars?"
        )
        rev = list(reversed(prices))
        slips = [
            (_cost(rev, cols[j]) / 100, "r3c-cost-prices-mismatched"),
            (Fraction(sum(prices), 100), "r3c-cost-ignores-quantities"),
            (Fraction(sum(cols[j])), "r3c-cost-counts-items"),
            (_cost(prices, cols[(j + 1) % m]) / 100, "r3c-cost-wrong-store"),
            (_cost(prices[:-1], cols[j][:-1]) / 100, "r3c-cost-item-left-out"),
            (_cost(prices, Q[j % k][:k]) / 100 if m >= k else None, "r3c-cost-row-for-column"),
        ]
        working = [
            f"Multiply each price by the {branches[j]} quantity for the same item, then add.",
            " + ".join(f"{_money(p)} x {q}" for p, q in zip(prices, cols[j])) + f" = {_money(cost * 100)}.",
        ]
        q = number_single(text, cost, working, _far(cost, slips), dp=2, lo=Fraction(1, 100))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(300, "a medium costing question"):
        k, m = rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
        ctx, items, branches, prices, Q = _setup(rng, k, m)
        cols = _cols(Q)
        costs = [_cost(prices, c) for c in cols]
        if len(set(costs)) < m:
            continue
        total = sum(costs) / 100
        text = (
            f"{ctx['intro']} The price in dollars of one each of {_list_text(items)}, in that order, is the row "
            f"P = {_price_text(prices)}. The quantities ordered are given by the matrix Q = {_qty_text(Q)}, which has a row "
            f"for each item (in the same order) and a column for each of {_list_text(branches)} (in that order). "
            "What is the total cost, in dollars, of the orders of all the branches together?"
        )
        rev = list(reversed(prices))
        slips = [
            (sum(_cost(rev, c) for c in cols) / 100, "r3c-cost-prices-mismatched"),
            (Fraction(sum(prices) * m, 100), "r3c-cost-ignores-quantities"),
            (Fraction(sum(sum(c) for c in cols)), "r3c-cost-counts-items"),
            (costs[0] / 100, "r3c-cost-one-branch-only"),
            ((sum(costs) - costs[-1]) / 100, "r3c-cost-item-left-out"),
            (sum(prices[i] * sum(Q[i][:-1]) for i in range(k)) / 100 + Fraction(0), "r3c-cost-item-left-out"),
        ]
        working = [
            "Find each branch's cost: the price row times its column of Q, then add the costs.",
            "Costs: " + ", ".join(_money(c) for c in costs) + ".",
            f"Total = {_money(sum(costs))}.",
        ]
        q = number_single(text, total, working, _far(total, slips), dp=2, lo=Fraction(1, 100))
        if q:
            return q
    raise AssertionError  # pragma: no cover


_FOODS = (
    dict(
        maker="A catering company",
        products=("beef pies", "vegetable pasties", "sausage rolls"),
        ingredients=("flour", "mince", "vegetables", "butter"),
        price=((200, 300), (1200, 1800), (300, 500), (900, 1300)),
    ),
    dict(
        maker="A bakery",
        products=("sourdough loaves", "fruit buns", "croissants"),
        ingredients=("flour", "butter", "sugar", "fruit"),
        price=((200, 300), (900, 1300), (250, 350), (500, 900)),
    ),
)

_OUTLETS = ("the Airport kiosk", "the Station kiosk", "the Marina kiosk", "the Hospital kiosk", "the University kiosk")


def _hard(rng):
    for _ in attempts(500, "a hard costing question"):
        ctx = rng.choice(_FOODS)
        ni, npr, no = 3, 2, rng.choice((3, 3, 4))
        ingredients = ctx["ingredients"][:ni]
        products = rng.sample(ctx["products"], npr)
        prices = [rng.randrange(lo, hi + 1, 50) for lo, hi in ctx["price"][:ni]]
        R = [[Fraction(rng.randint(1, 8), 2) for _ in range(npr)] for _ in range(ni)]
        Q = [[rng.randint(5, 40) for _ in range(no)] for _ in range(npr)]
        outlets = rng.sample(_OUTLETS, no)
        batch = [sum(Fraction(prices[i]) * R[i][p] for i in range(ni)) for p in range(npr)]
        costs = [sum(batch[p] * Q[p][o] for p in range(npr)) for o in range(no)]
        if len(set(costs)) < no or any(c % 1 for c in costs):
            continue
        j = rng.randrange(no)
        key = costs[j] / 100
        R_text = "[" + ", ".join("[" + ", ".join(f"{float(v):g}" for v in row) + "]" for row in R) + "]"
        text = (
            f"{ctx['maker']} makes {_list_text(products)} in batches. The matrix R = {R_text} gives the kilograms of each "
            f"of {_list_text(ingredients)} (rows, in that order) used in one batch of each product (columns, in the order "
            f"{_list_text(products)}). The ingredients cost P = {_price_text(prices)} dollars per kilogram, in the same order. "
            f"The matrix Q = {_qty_text(Q)} gives the number of batches of each product (rows, same order) delivered to "
            f"{_list_text(outlets)} (columns, in that order) each week. "
            f"What is the weekly ingredient cost for {outlets[j]}, in dollars?"
        )
        col_j = [Q[p][j] for p in range(npr)]
        slips = [
            (sum(batch[p] * col_j[npr - 1 - p] for p in range(npr)) / 100, "r3c-cost-prices-mismatched"),
            (sum(batch) / 100, "r3c-cost-one-stage-only"),
            (sum(batch) * sum(col_j) / 100 / npr, "r3c-cost-ignores-quantities"),
            (costs[(j + 1) % no] / 100, "r3c-cost-wrong-store"),
            (batch[0] * col_j[0] / 100, "r3c-cost-item-left-out"),
            (
                sum(Fraction(prices[i]) * col_j[p] * R[i][p] for i in range(ni) for p in range(npr) if i != 0) / 100,
                "r3c-cost-item-left-out",
            ),
            (sum(prices) * sum(col_j) / 100, "r3c-cost-ignores-quantities"),
        ]
        working = [
            "First find the ingredient cost of one batch of each product: the price row times R.",
            "Cost of one batch of each product: " + ", ".join(_money(b) for b in batch) + ".",
            "Then multiply each by the batches delivered to that outlet and add: "
            + " + ".join(f"{_money(batch[p])} x {col_j[p]}" for p in range(npr))
            + f" = {_money(costs[j])}.",
        ]
        q = number_single(text, key, working, _far(key, slips), dp=2, lo=Fraction(1, 100))
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_costing",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_costing(rng, tier="easy"):
    """Costs from a price row and quantity matrices; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("r3c_costing_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_costing_chain(rng, tier="exam"):
    """Exam chain: two batch costs from a recipe matrix, the cost of an order, and the profit on it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a costing chain"):
        ctx = rng.choice(_FOODS)
        ni = 3
        ingredients = ctx["ingredients"][:ni]
        products = rng.sample(ctx["products"], 2)
        prices = [rng.randrange(lo, hi + 1, 50) for lo, hi in ctx["price"][:ni]]
        R = [[Fraction(rng.randint(1, 8), 2) for _ in range(2)] for _ in range(ni)]
        batch = [sum(Fraction(prices[i]) * R[i][p] for i in range(ni)) for p in range(2)]
        x, y = rng.randint(4, 20), rng.randint(4, 20)
        sell = [rng.randrange(int(b) + 400, int(b) + 1500, 50) for b in batch]
        cost = x * batch[0] + y * batch[1]
        revenue = x * sell[0] + y * sell[1]
        profit = revenue - cost
        if profit <= 0 or batch[0] == batch[1] or any(v % 1 for v in (cost, profit)):
            continue
        R_text = "[" + ", ".join("[" + ", ".join(f"{float(v):g}" for v in row) + "]" for row in R) + "]"
        stem = (
            f"{ctx['maker']} makes {products[0]} and {products[1]} in batches. The matrix R = {R_text} gives the kilograms "
            f"of each of {_list_text(ingredients)} (rows, in that order) used in one batch of each product (columns, in the "
            f"order {products[0]}, {products[1]}). The ingredients cost P = {_price_text(prices)} dollars per kilogram, in the "
            "same order."
        )
        pb = [b / 100 for b in batch]
        rev_prices = list(reversed(prices))
        part_a = number_part(
            f"What is the ingredient cost of one batch of {products[0]}, in dollars?",
            pb[0],
            2,
            [" + ".join(f"{_money(prices[i])} x {float(R[i][0]):g}" for i in range(ni)) + f" = {_money(batch[0])}."],
            [
                (sum(Fraction(rev_prices[i]) * R[i][0] for i in range(ni)) / 100, "r3c-cost-prices-mismatched"),
                (sum(Fraction(prices[i]) * R[i][1] for i in range(ni)) / 100, "r3c-cost-wrong-product"),
                (Fraction(sum(prices), 100), "r3c-cost-ignores-quantities"),
                (sum(R[i][0] for i in range(ni)), "r3c-cost-counts-items"),
            ],
            dp=2,
            lo=Fraction(1, 100),
            sensible=(0, None),
        )
        part_b = number_part(
            f"What is the ingredient cost of one batch of {products[1]}, in dollars?",
            pb[1],
            2,
            [" + ".join(f"{_money(prices[i])} x {float(R[i][1]):g}" for i in range(ni)) + f" = {_money(batch[1])}."],
            [
                (sum(Fraction(rev_prices[i]) * R[i][1] for i in range(ni)) / 100, "r3c-cost-prices-mismatched"),
                (sum(Fraction(prices[i]) * R[i][0] for i in range(ni)) / 100, "r3c-cost-wrong-product"),
                (Fraction(sum(prices), 100), "r3c-cost-ignores-quantities"),
                (sum(R[i][1] for i in range(ni)), "r3c-cost-counts-items"),
            ],
            dp=2,
            lo=Fraction(1, 100),
            sensible=(0, None),
        )
        part_c = number_part(
            f"One week's order is {x} batches of {products[0]} and {y} batches of {products[1]}. What is the ingredient cost of the order, in dollars?",
            cost / 100,
            2,
            [f"{x} x {_money(batch[0])} + {y} x {_money(batch[1])} = {_money(cost)}."],
            [
                (x * pb[1] + y * pb[0], "r3c-cost-wrong-product"),
                (pb[0] + pb[1], "r3c-cost-ignores-quantities"),
                ((x + y) * pb[0], "r3c-cost-one-product-price"),
                (x * pb[0], "r3c-cost-item-left-out"),
            ],
            follow=f"{x}*part_a + {y}*part_b",
            dp=2,
            lo=Fraction(1, 100),
            sensible=(0, None),
        )
        part_d = number_part(
            f"The {products[0]} sell for ${_money(sell[0])} a batch and the {products[1]} for ${_money(sell[1])} a batch. What is the profit on the order, in dollars?",
            profit / 100,
            2,
            [
                f"Sales {x} x {_money(sell[0])} + {y} x {_money(sell[1])} = {_money(revenue)}, less the cost {_money(cost)}."
            ],
            [
                (revenue / 100, "r3c-cost-revenue-not-profit"),
                (revenue / 100 + cost / 100, "r3c-cost-profit-added"),
                (x * (sell[0] - batch[0]) / 100 + y * (sell[0] - batch[1]) / 100, "r3c-cost-one-product-price"),
                (cost / 100 - revenue / 100, "r3c-cost-profit-reversed"),
            ],
            follow=f"{_money(revenue)} - part_c",
            dp=2,
            lo=Fraction(1, 100),
            sensible=(0, None),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
