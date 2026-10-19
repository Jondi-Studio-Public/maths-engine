"""Generates the `w3_*` market and chance-game topics: linear supply and demand, and fair games.

* ``supply-demand-equilibrium`` (``w3_supply_demand``, exam ``exd_x_w3_supply_demand``): linear demand Q = a - bP and
  supply Q = c + dP, with the price P in dollars and the quantity Q in units. Easy: the quantity demanded or supplied at a
  price, and the shortage or surplus there. Medium: the equilibrium price, quantity and revenue. Hard: the equilibrium
  from two data points for each line, the new equilibrium quantity after a tax on sellers, and the price at which a stated
  surplus occurs.
* ``fair-game-expected-gain`` (``w3_fair_game``, exam ``exd_x_w3_fair_game``): a game of chance with equally likely outcomes.
  Easy: the expected prize, the expected gain per play. Medium: the entry fee that makes the game fair, the organiser's
  expected profit over many plays. Hard: the missing prize that makes a game fair, a game of two draws without
  replacement, the number of plays for a stated expected profit.

Both are built backwards from chosen equilibria, probabilities and prizes so every answer is exact, and each is re-derived
a second way (by substitution into both lines, or by listing the outcomes).
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..core import attempts, check, generator
from ._w3_kit import MONEY, WHOLE, art, dollars
from ._w6a_kit import TooFewDistractors, enough, part, retry_draw

SD_TOPIC = "w3_supply_demand"
SD_EXAM = "exd_x_w3_supply_demand"
FG_TOPIC = "w3_fair_game"
FG_EXAM = "exd_x_w3_fair_game"

_SD = "supply-demand-equilibrium"
_FG = "fair-game-expected-gain"

MISCONCEPTIONS = [
    ("sd-sign-of-slope", "Adds the price term to the demand intercept instead of subtracting it.", _SD),
    ("sd-uses-other-line", "Substitutes the price into the other line (supply for demand or demand for supply).", _SD),
    ("sd-forgets-intercept", "Leaves the intercept out and multiplies the price by the slope only.", _SD),
    ("sd-adds-lines", "Adds the quantity demanded and the quantity supplied instead of finding their difference.", _SD),
    (
        "sd-ignores-price-terms",
        "Subtracts the intercepts and ignores the price terms when finding the shortage or surplus.",
        _SD,
    ),
    ("sd-gives-demand-for-gap", "Gives the quantity demanded where the shortage or surplus was asked.", _SD),
    ("sd-gives-supply-for-gap", "Gives the quantity supplied where the shortage or surplus was asked.", _SD),
    ("sd-price-for-quantity", "Gives the price where the quantity was asked.", _SD),
    ("sd-adds-intercepts", "Adds the intercepts instead of subtracting them when setting demand equal to supply.", _SD),
    ("sd-subtracts-slopes", "Subtracts the slopes instead of adding them when collecting the price terms.", _SD),
    ("sd-divides-by-one-slope", "Divides by one slope only when solving for the price.", _SD),
    ("sd-average-intercepts", "Takes the average of the two intercepts as the equilibrium quantity.", _SD),
    ("sd-quantity-for-price", "Gives the quantity where the price was asked.", _SD),
    ("sd-wrong-line-quantity", "Finds the quantity from an incorrect equilibrium price.", _SD),
    ("sd-price-plus-quantity", "Adds the price and the quantity instead of multiplying them for the revenue.", _SD),
    ("sd-revenue-quantity-only", "Gives the quantity where the revenue was asked.", _SD),
    ("sd-revenue-price-only", "Gives the price where the revenue was asked.", _SD),
    (
        "sd-slope-from-one-point",
        "Uses a quantity as the slope instead of the change in quantity for a change in price.",
        _SD,
    ),
    ("sd-slope-inverted", "Divides the change in price by the change in quantity for the slope.", _SD),
    (
        "sd-tax-subtracted-from-quantity",
        "Subtracts the tax from the quantity instead of the price in the supply line.",
        _SD,
    ),
    ("sd-tax-added-to-price", "Adds the tax to the price in the supply line instead of subtracting it.", _SD),
    ("sd-tax-ignored", "Ignores the tax and gives the original equilibrium quantity.", _SD),
    ("sd-tax-price-for-quantity", "Gives the new equilibrium price where the quantity was asked.", _SD),
    ("sd-surplus-as-price", "Divides the surplus by the sum of the slopes and gives that as the price.", _SD),
    ("sd-surplus-shortage-price", "Subtracts the extra price needed for the surplus instead of adding it.", _SD),
    ("sd-surplus-equilibrium", "Gives the equilibrium price where the price for a surplus was asked.", _SD),
    ("sd-surplus-one-slope", "Divides the surplus by one slope only to find the extra price.", _SD),
    ("fg-unweighted-average", "Averages the prizes without weighting them by their probabilities.", _FG),
    ("fg-sum-of-prizes", "Adds the prizes without weighting them by their probabilities.", _FG),
    ("fg-largest-prize", "Gives the largest prize as the expected prize.", _FG),
    ("fg-forgets-entry", "Gives the expected prize where the expected gain (after paying to play) was asked.", _FG),
    ("fg-reversed-sign", "Subtracts the expected prize from the entry fee, so the sign of the gain is wrong.", _FG),
    ("fg-adds-entry", "Adds the entry fee to the expected prize.", _FG),
    ("fg-sum-for-fee", "Gives the total of the prizes as the fair entry fee.", _FG),
    ("fg-fee-largest-prize", "Gives the largest prize as the fair entry fee.", _FG),
    ("fg-total-not-divided", "Adds the prizes on all the tickets and does not divide by the number of tickets.", _FG),
    ("fg-ignores-blanks", "Divides by the number of winning tickets only, ignoring the blank ones.", _FG),
    ("fg-fee-ignores-target-profit", "Gives the fair entry fee, leaving out the profit the organiser wants.", _FG),
    (
        "fg-fee-subtracts-target-profit",
        "Takes the profit the organiser wants off the fair entry fee instead of adding it.",
        _FG,
    ),
    ("fg-fee-is-target-profit", "Gives the profit the organiser wants as the entry fee.", _FG),
    (
        "fg-profit-sign",
        "Gives the players' expected gain over all the plays where the organiser's profit was asked.",
        _FG,
    ),
    ("fg-profit-prize-total", "Gives the expected prize paid out over all the plays where the profit was asked.", _FG),
    ("fg-profit-per-play", "Gives the profit on one play where the profit over many plays was asked.", _FG),
    ("fg-profit-fee-total", "Gives the entry fees taken where the profit was asked.", _FG),
    (
        "fg-missing-prize-no-probability",
        "Sets the prize equal to the entry fee without allowing for the chance of winning it.",
        _FG,
    ),
    (
        "fg-missing-prize-forgets-others",
        "Ignores the expected value of the other prizes when finding the missing prize.",
        _FG,
    ),
    (
        "fg-missing-prize-times-probability",
        "Multiplies by the probability instead of dividing by it when finding the missing prize.",
        _FG,
    ),
    ("fg-missing-prize-adds-others", "Adds the expected value of the other prizes instead of subtracting it.", _FG),
    (
        "fg-with-replacement",
        "Uses the same probability for the second draw, as if the first counter were put back.",
        _FG,
    ),
    ("fg-one-draw", "Uses the chance for one draw only, ignoring the second.", _FG),
    ("fg-probability-added", "Adds the two probabilities instead of multiplying them.", _FG),
    ("fg-games-per-play", "Gives the profit per play where the number of plays was asked.", _FG),
    ("fg-games-multiplies", "Multiplies the target profit by the profit per play instead of dividing.", _FG),
    ("fg-games-prize-based", "Divides the target by the expected prize instead of the expected profit.", _FG),
    ("fg-games-fee-based", "Divides the target by the entry fee instead of the expected profit.", _FG),
]

# (name, lowest and highest realistic equilibrium price in dollars)
_PRODUCTS = (
    ("bags of oranges", 4, 9),
    ("tickets to a show", 20, 60),
    ("pairs of sunglasses", 15, 50),
    ("loaves of bread", 3, 6),
    ("cups of coffee", 3, 6),
    ("phone covers", 8, 25),
    ("kayak hire sessions", 20, 60),
    ("potted plants", 5, 20),
    ("T-shirts", 10, 40),
    ("pies", 4, 12),
)
_TIME = ("week", "day", "month")


def _market(rng, c_min: int = 0, price=(4, 30)):
    """(P*, Q*, a, b, c, d): demand Q = a - bP and supply Q = c + dP meet at (P*, Q*), all whole numbers, c >= c_min."""
    for _ in attempts(500, "a market"):
        p = rng.randint(*price)
        b = rng.randint(2, 12)
        d = rng.randint(2, 12)
        if abs(b - d) < 1:
            continue
        q = rng.randint(max(10, d * p + c_min // 10 * 10), d * p + 400)
        a, c = q + b * p, q - d * p
        if c >= c_min and a - b * (p * 2) > 0:  # demand still positive at twice the equilibrium price
            return p, q, a, b, c, d
    raise AssertionError  # pragma: no cover


def _market_for(rng, c_min: int = 0):
    """(product, P*, Q*, a, b, c, d): a market for a product whose price scale is realistic."""
    product, lo, hi = rng.choice(_PRODUCTS)
    return (product, *_market(rng, c_min, (lo, hi)))


def _dem(a, b):
    return f"Q = {a} - {b}P"


def _sup(c, d):
    return f"Q = {c} + {d}P" if c else f"Q = {d}P"


def _setting(rng, product, a, b, c, d):
    return (
        f"The weekly demand for {product} at a market is given by {_dem(a, b)}, and the weekly supply is given by {_sup(c, d)}, "
        f"where P is the price in dollars and Q is the number of units."
    )


def _units(n) -> str:
    return f"{n} unit" if n == 1 else f"{n} units"


# -- supply and demand: easy --------------------------------------------------------------------------------------------


def _sd_quantity(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    price = rng.randint(2, p0 * 2 - 1)
    demanded = rng.random() < 0.5
    qd, qs = a - b * price, c + d * price
    key = qd if demanded else qs
    other = qs if demanded else qd
    cands = (
        [
            (a + b * price, "sd-sign-of-slope"),
            (other, "sd-uses-other-line"),
            (b * price, "sd-forgets-intercept"),
            (a * price, "sd-forgets-intercept"),
            (qd + qs, "sd-adds-lines"),
        ]
        if demanded
        else [
            (c - d * price, "sd-sign-of-slope"),
            (other, "sd-uses-other-line"),
            (d * price, "sd-forgets-intercept"),
            (c * price, "sd-forgets-intercept"),
            (qd + qs, "sd-adds-lines"),
        ]
    )
    dist = WHOLE.dist(key, cands)
    enough(dist, 3)
    word = "demanded" if demanded else "supplied"
    text = _setting(rng, product, a, b, c, d) + f" How many units are {word} each week when the price is ${price}?"
    working = [f"Q = {a if demanded else c} {'-' if demanded else '+'} {b if demanded else d} × {price} = {key}."]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(key), answer_kind="number")


def _sd_gap(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    offset = rng.choice([x for x in range(-p0 + 2, p0) if x != 0])
    price = p0 + offset
    qd, qs = a - b * price, c + d * price
    gap = qs - qd
    check(gap == (b + d) * offset and gap != 0, "gap")
    word = "surplus" if gap > 0 else "shortage"
    dist = WHOLE.dist(
        abs(gap),
        [
            (qd, "sd-gives-demand-for-gap"),
            (qs, "sd-gives-supply-for-gap"),
            (qd + qs, "sd-adds-lines"),
            (abs(a - c), "sd-ignores-price-terms"),
        ],
    )
    enough(dist, 3)
    text = (
        _setting(rng, product, a, b, c, d)
        + f" What is the {word} (the difference between the quantity supplied and the quantity demanded) when the price is ${price}, in units?"
    )
    working = [
        f"Demand = {a} - {b} × {price} = {qd}; supply = {c} + {d} × {price} = {qs}.",
        f"{word.capitalize()} = {_units(abs(gap))}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(abs(gap)), answer_kind="number")


# -- medium -------------------------------------------------------------------------------------------------------------


def _sd_price(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    dist = WHOLE.dist(
        p0,
        [
            (Fraction(a + c, b + d), "sd-adds-intercepts"),
            (Fraction(a - c, abs(b - d)), "sd-subtracts-slopes"),
            (Fraction(a - c, b), "sd-divides-by-one-slope"),
            (Fraction(a - c, d), "sd-divides-by-one-slope"),
            (q0, "sd-quantity-for-price"),
        ],
    )
    enough(dist, 3)
    check(a - b * p0 == c + d * p0 == q0, "both lines meet")
    text = (
        _setting(rng, product, a, b, c, d)
        + " Find the equilibrium price, in dollars (the price at which the quantity demanded equals the quantity supplied)."
    )
    working = [f"{a} - {b}P = {c} + {d}P gives {a - c} = {b + d}P.", f"P = {a - c} / {b + d} = {p0}."]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(p0), answer_kind="number")


def _sd_quantity_eq(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    wrong_p = Fraction(a + c, b + d)
    dist = WHOLE.dist(
        q0,
        [
            (p0, "sd-price-for-quantity"),
            (Fraction(a + c, 2), "sd-average-intercepts"),
            (a - b * wrong_p, "sd-wrong-line-quantity"),
            (a - b * Fraction(a - c, abs(b - d)), "sd-wrong-line-quantity"),
            (a - b * Fraction(a - c, b), "sd-wrong-line-quantity"),
        ],
    )
    enough(dist, 3)
    text = _setting(rng, product, a, b, c, d) + " Find the equilibrium quantity, in units."
    working = [f"Equilibrium price: P = {a - c} / {b + d} = {p0}.", f"Q = {a} - {b} × {p0} = {q0}."]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(q0), answer_kind="number")


def _sd_revenue(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    revenue = p0 * q0
    dist = WHOLE.dist(
        revenue,
        [
            (q0, "sd-revenue-quantity-only"),
            (p0, "sd-revenue-price-only"),
            (p0 + q0, "sd-price-plus-quantity"),
            (p0 * (a - b * Fraction(a + c, b + d)), "sd-wrong-line-quantity"),
        ],
    )
    enough(dist, 3)
    text = (
        _setting(rng, product, a, b, c, d)
        + " At the equilibrium price, how much money in dollars is spent on the product each week?"
    )
    working = [f"P = {a - c} / {b + d} = {p0}; Q = {a} - {b} × {p0} = {q0}.", f"Revenue = {p0} × {q0} = {revenue}."]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(revenue), answer_kind="number")


# -- hard ---------------------------------------------------------------------------------------------------------------


def _sd_data_points(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    p1 = rng.randint(2, p0 - 1) if p0 > 2 else 1
    gap = rng.randint(2, 6)
    p2 = p1 + gap
    q_d1, q_d2 = a - b * p1, a - b * p2
    q_s1, q_s2 = c + d * p1, c + d * p2
    if min(q_d2, q_d1, q_s1, q_s2) <= 0:
        raise TooFewDistractors("non-positive quantities")
    check((q_d1 - q_d2) == b * gap and (q_s2 - q_s1) == d * gap, "slopes")
    wrong_b = Fraction(q_d1 - q_d2, 1)
    dist = WHOLE.dist(
        p0,
        [
            (Fraction(a - c, b + d) + 1, "sd-slope-from-one-point"),
            (q0, "sd-quantity-for-price"),
            (Fraction(q_d1 + q_s1, 2), "sd-average-intercepts"),
            (Fraction(a - c, abs(b - d)), "sd-subtracts-slopes"),
            (p1 + p2 - p0, "sd-slope-inverted"),
        ],
    )
    enough(dist, 3)
    text = (
        f"For {product} sold at a market, the weekly quantity demanded falls steadily as the price rises, and the weekly quantity supplied rises steadily. "
        f"At ${p1} the quantity demanded is {q_d1} and the quantity supplied is {q_s1}. At ${p2} the quantity demanded is {q_d2} and the quantity supplied is {q_s2}. "
        f"Find the equilibrium price, in dollars."
    )
    working = [
        f"Demand falls {q_d1 - q_d2} units for a ${gap} rise, so b = {b}; supply rises {q_s2 - q_s1} units, so d = {d}.",
        f"Demand: Q = {a} - {b}P. Supply: Q = {c} + {d}P (found from the points).",
        f"{a} - {b}P = {c} + {d}P gives P = {p0}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(p0), answer_kind="number")


def _sd_tax(rng):
    product, plo, phi = rng.choice(_PRODUCTS)
    for _ in attempts(300, "a taxed market"):
        p1 = rng.randint(max(plo, 4), phi)  # new equilibrium price paid by buyers
        q1 = rng.randint(40, 500)
        b, d = rng.randint(2, 12), rng.randint(2, 12)
        t = rng.randint(1, max(1, p1 // 4))
        a = q1 + b * p1
        c = q1 - d * (p1 - t)
        if c < 0 or p1 - t < 1 or a - b * 2 * p1 <= 0:
            continue
        p0 = Fraction(a - c, b + d)
        q0 = a - b * p0
        if p0 < 2 or q0 <= q1:
            continue
        break
    else:
        raise TooFewDistractors("no taxed market")
    # second way: solve both lines for the taxed price by substitution
    price = Fraction(a - c + d * t, b + d)
    check(price == p1 and a - b * price == q1, "taxed equilibrium")
    dist = WHOLE.dist(
        q1,
        [
            (q0, "sd-tax-ignored"),
            (p1, "sd-tax-price-for-quantity"),
            ((a - b * Fraction(a - c - d * t, b + d)), "sd-tax-added-to-price"),
            ((a - b * Fraction(a - c + t, b + d)), "sd-tax-subtracted-from-quantity"),
            (q0 - t, "sd-tax-subtracted-from-quantity"),
        ],
    )
    enough(dist, 3)
    text = (
        f"The weekly demand for {product} is given by {_dem(a, b)}, and before any tax the weekly supply is given by {_sup(c, d)}, where P is the price in dollars paid by buyers. "
        f"The government then puts a tax of ${t} per unit on sellers, so sellers keep only P - {t} on each unit and the supply becomes Q = {c} + {d}(P - {t}). "
        f"Find the new equilibrium quantity, in units."
    )
    working = [
        f"{a} - {b}P = {c} + {d}(P - {t}) gives {a - c + d * t} = {b + d}P, so P = {p1}.",
        f"Q = {a} - {b} × {p1} = {q1}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(q1), answer_kind="number")


def _sd_surplus_price(rng):
    product, p0, q0, a, b, c, d = _market_for(rng)
    k = rng.randint(1, min(8, p0 - 1)) if p0 > 1 else 1
    surplus = (b + d) * k
    price = p0 + k
    check(c + d * price - (a - b * price) == surplus, "surplus")
    dist = WHOLE.dist(
        price,
        [
            (p0, "sd-surplus-equilibrium"),
            (Fraction(surplus, b + d), "sd-surplus-as-price"),
            (p0 - k, "sd-surplus-shortage-price"),
            (p0 + Fraction(surplus, d), "sd-surplus-one-slope"),
            (p0 + Fraction(surplus, b), "sd-surplus-one-slope"),
        ],
    )
    enough(dist, 3)
    text = (
        _setting(rng, product, a, b, c, d)
        + f" At what price, in dollars, is there a surplus of {surplus} units (the quantity supplied exceeds the quantity demanded by {surplus})?"
    )
    working = [
        f"Surplus = ({c} + {d}P) - ({a} - {b}P) = {b + d}P - {a - c}.",
        f"{b + d}P - {a - c} = {surplus} gives P = {price}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(price), answer_kind="number")


_SD_EASY = (_sd_quantity, _sd_gap)
_SD_MEDIUM = (_sd_price, _sd_quantity_eq, _sd_revenue)
_SD_HARD = (_sd_data_points, _sd_tax, _sd_surplus_price)


@generator(
    SD_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_supply_demand(rng, tier="easy"):
    """Linear supply and demand; the tiers are described in the module docstring."""
    pool = {"easy": _SD_EASY, "medium": _SD_MEDIUM, "hard": _SD_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(SD_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_supply_demand(rng, tier="exam"):
    """Exam chain: quantity demanded, the shortage at a price, the equilibrium price and quantity."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        product, p0, q0, a, b, c, d = _market_for(rng)
        price = p0 - rng.randint(1, max(1, min(6, p0 - 2)))
        if price < 1:
            raise TooFewDistractors("price")
        qd, qs = a - b * price, c + d * price
        gap = qd - qs
        if gap <= 0:
            raise TooFewDistractors("shortage expected below equilibrium")
        a_d = WHOLE.dist(
            qd,
            [
                (a + b * price, "sd-sign-of-slope"),
                (qs, "sd-uses-other-line"),
                (b * price, "sd-forgets-intercept"),
                (a * price, "sd-forgets-intercept"),
            ],
        )
        b_d = WHOLE.dist(
            gap,
            [
                (qd, "sd-gives-demand-for-gap"),
                (qs, "sd-gives-supply-for-gap"),
                (qd + qs, "sd-adds-lines"),
                (qd - c, "sd-uses-other-line"),
            ],
        )
        c_d = WHOLE.dist(
            p0,
            [
                (Fraction(a + c, b + d), "sd-adds-intercepts"),
                (Fraction(a - c, abs(b - d)), "sd-subtracts-slopes"),
                (Fraction(a - c, b), "sd-divides-by-one-slope"),
                (q0, "sd-quantity-for-price"),
            ],
        )
        d_d = WHOLE.dist(
            q0,
            [
                (p0, "sd-price-for-quantity"),
                (Fraction(a + c, 2), "sd-average-intercepts"),
                (a - b * Fraction(a + c, b + d), "sd-wrong-line-quantity"),
                (a - b * Fraction(a - c, b), "sd-wrong-line-quantity"),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = _setting(rng, product, a, b, c, d)
        return dict(
            text=stem,
            parts=[
                part(
                    f"How many units are demanded each week when the price is ${price}?",
                    "number",
                    WHOLE.fields(qd),
                    1,
                    [f"Q = {a} - {b} × {price} = {qd}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"What is the shortage (the quantity demanded less the quantity supplied) at ${price}, in units?",
                    "number",
                    WHOLE.fields(gap),
                    2,
                    [f"Supply = {c} + {d} × {price} = {qs}; shortage = {qd} - {qs} = {gap}."],
                    b_d,
                    follow_through=f"part_a - {qs}",
                    sensible_range=(0, None),
                ),
                part(
                    "Find the equilibrium price, in dollars.",
                    "number",
                    WHOLE.fields(p0),
                    2,
                    [f"{a} - {b}P = {c} + {d}P gives P = {a - c} / {b + d} = {p0}."],
                    c_d,
                    sensible_range=(0, None),
                ),
                part(
                    "Find the equilibrium quantity, in units.",
                    "number",
                    WHOLE.fields(q0),
                    2,
                    [f"Q = {a} - {b} × {p0} = {q0}."],
                    d_d,
                    follow_through=f"{a} - {b}*part_c",
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


# -- fair games ---------------------------------------------------------------------------------------------------------

_DEVICES = (
    ("a spinner with {n} equal sectors", "sector", "sectors", "spin"),
    ("a bag holding {n} identical counters, one of which is drawn at random", "counter", "counters", "draw"),
    ("{n} numbered tickets in a hat, one of which is drawn at random", "ticket", "tickets", "draw"),
)


def _count(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def _game(rng, outcomes=None, fee_choice=True):
    """(N, groups [(count, prize)], device text) with a whole number of cents in the expected prize."""
    for _ in attempts(300, "a game with an exact expected prize"):
        n = rng.choice((6, 8, 10, 12, 20))
        k = rng.choice((2, 3))
        counts = rng.sample(range(1, max(3, n // 2)), k) if n // 2 > k else [1] * k
        if sum(counts) >= n:
            continue
        prizes = sorted(rng.sample(range(2, 41), k), reverse=True)
        ev = Fraction(sum(c * p for c, p in zip(counts, prizes)), n)
        if (ev * 100).denominator != 1:
            continue
        return n, list(zip(counts, prizes)), ev
    raise TooFewDistractors("no exact game")


def _describe(rng, n, groups):
    device, one, many, verb = rng.choice(_DEVICES)
    parts_ = [f"${p} on {_count(c, one, many)}" for c, p in groups]
    rest = n - sum(c for c, _ in groups)
    return (
        f"A game uses {device.format(n=n)}. The prizes are "
        + ", ".join(parts_[:-1])
        + (" and " if len(parts_) > 1 else "")
        + parts_[-1]
        + f"; there is no prize on the other {_count(rest, one, many)}. All {many} are equally likely"
        + "."
    )


def _fg_dist_for_prize(ev, n, groups, *, with_fee=None):
    prizes = [p for _, p in groups]
    return [
        (Fraction(sum(prizes), len(prizes)), "fg-unweighted-average"),
        (sum(prizes), "fg-sum-of-prizes"),
        (max(prizes), "fg-largest-prize"),
        (sum(c * p for c, p in groups), "fg-total-not-divided"),
        (Fraction(sum(c * p for c, p in groups), sum(c for c, _ in groups)), "fg-ignores-blanks"),
    ]


def _fg_expected_prize(rng):
    n, groups, ev = _game(rng)
    dist = MONEY.dist(ev, _fg_dist_for_prize(ev, n, groups))
    enough(dist, 3)
    text = _describe(rng, n, groups) + " What is the expected prize for one play, in dollars?"
    terms = " + ".join(f"{c}/{n} × {p}" for c, p in groups)
    working = [f"Expected prize = {terms} = {dollars(ev)}."]
    return dict(text=text, working=working, distractors=dist, **MONEY.fields(ev), answer_kind="number")


def _fg_expected_gain(rng):
    n, groups, ev = _game(rng)
    fee = max(Fraction(2), Fraction(round(ev * rng.randint(70, 130) / 100)))
    gain = ev - fee
    if gain == 0:
        raise TooFewDistractors("fair game")
    dist = MONEY.dist(
        gain,
        [
            (ev, "fg-forgets-entry"),
            (fee - ev, "fg-reversed-sign"),
            (ev + fee, "fg-adds-entry"),
            (Fraction(sum(p for _, p in groups), len(groups)) - fee, "fg-unweighted-average"),
            (sum(p for _, p in groups) - fee, "fg-sum-of-prizes"),
        ],
        allow_negative=True,
    )
    enough(dist, 3)
    text = (
        _describe(rng, n, groups)
        + f" It costs {dollars(fee)} to play. What is the expected gain per play, in dollars? (A loss is a negative gain.)"
    )
    terms = " + ".join(f"{c}/{n} × {p}" for c, p in groups)
    working = [
        f"Expected prize = {terms} = {dollars(ev)}.",
        f"Expected gain = {dollars(ev)} - {dollars(fee)} = {dollars(gain)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **MONEY.fields(gain), answer_kind="number")


def _fg_fair_fee(rng):
    n, groups, ev = _game(rng)
    prizes = [p for _, p in groups]
    dist = MONEY.dist(
        ev,
        [
            (sum(prizes), "fg-sum-for-fee"),
            (max(prizes), "fg-fee-largest-prize"),
            (sum(c * p for c, p in groups), "fg-total-not-divided"),
            (Fraction(sum(c * p for c, p in groups), sum(c for c, _ in groups)), "fg-ignores-blanks"),
            (Fraction(sum(prizes), len(prizes)), "fg-unweighted-average"),
        ],
    )
    enough(dist, 3)
    text = (
        _describe(rng, n, groups)
        + " What entry fee, in dollars, would make this a fair game (an expected gain of zero)?"
    )
    terms = " + ".join(f"{c}/{n} × {p}" for c, p in groups)
    working = [f"A fair game has an entry fee equal to the expected prize: {terms} = {dollars(ev)}."]
    return dict(text=text, working=working, distractors=dist, **MONEY.fields(ev), answer_kind="number")


def _fg_organiser(rng):
    n, groups, ev = _game(rng)
    fee = ev + Fraction(rng.randint(1, 8))
    plays = rng.choice((50, 100, 200, 250, 400))
    profit = (fee - ev) * plays
    dist = MONEY.dist(
        profit,
        [
            (-profit, "fg-profit-sign"),
            (ev * plays, "fg-profit-prize-total"),
            (fee - ev, "fg-profit-per-play"),
            (fee * plays, "fg-profit-fee-total"),
        ],
        allow_negative=True,
    )
    enough(dist, 3)
    text = (
        _describe(rng, n, groups)
        + f" It costs {dollars(fee)} to play. The organiser expects the game to be played {plays} times. What profit does the organiser expect, in dollars?"
    )
    working = [
        f"Expected prize per play = {dollars(ev)}; organiser's expected profit per play = {dollars(fee)} - {dollars(ev)} = {dollars(fee - ev)}.",
        f"Over {plays} plays: {plays} × {dollars(fee - ev)} = {dollars(profit)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **MONEY.fields(profit), answer_kind="number")


def _fg_missing_prize(rng):
    for _ in attempts(300, "a game with a whole missing prize"):
        n = rng.choice((6, 8, 10, 12, 20))
        c1 = rng.randint(1, max(1, n // 3))
        c2 = rng.randint(1, max(1, n // 3))
        if c1 + c2 >= n:
            continue
        p1 = rng.randint(3, 30)
        x = rng.randint(2, 60)
        fee = Fraction(c1 * p1 + c2 * x, n)
        if (fee * 100).denominator != 1 or fee <= 0:
            continue
        break
    else:
        raise TooFewDistractors("no game")
    device, one, many, verb = rng.choice(_DEVICES)
    dist = WHOLE.dist(
        x,
        [
            (fee, "fg-missing-prize-no-probability"),
            (Fraction(fee * n, c2), "fg-missing-prize-forgets-others"),
            (Fraction(fee * n + c1 * p1, c2), "fg-missing-prize-adds-others"),
            (fee * c2 / n, "fg-missing-prize-times-probability"),
            (Fraction(fee - Fraction(c1 * p1, n)) * n, "fg-missing-prize-times-probability"),
        ],
    )
    enough(dist, 3)
    text = (
        f"A game uses {device.format(n=n)}. A prize of ${p1} is paid on {_count(c1, one, many)} and a prize of $x is paid on {_count(c2, one, many)}; there is no prize on the other "
        f"{_count(n - c1 - c2, one, many)}. All {many} are equally likely. The game is fair when the entry fee is {dollars(fee)}. "
        f"Find the value of the prize $x, in dollars."
    )
    working = [
        f"Fair: {c1}/{n} × {p1} + {c2}/{n} × x = {dollars(fee)}.",
        f"{c2}/{n} × x = {dollars(fee)} - {c1 * p1}/{n}, so x = {x}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(x), answer_kind="number")


def _fg_two_draws(rng):
    for _ in attempts(300, "a two-draw game"):
        n = rng.randint(5, 9)
        r = rng.randint(2, n - 2)
        both = Fraction(r * (r - 1), n * (n - 1))
        prize = rng.randint(2, 60)
        fee = Fraction(rng.randint(1, 8))
        ev = both * prize
        if (ev * 100).denominator != 1:
            continue
        gain = ev - fee
        if gain != 0:
            break
    else:
        raise TooFewDistractors("no two-draw game")
    wrong_with = Fraction(r, n) ** 2 * prize - fee
    dist = MONEY.dist(
        gain,
        [
            (wrong_with, "fg-with-replacement"),
            (Fraction(r, n) * prize - fee, "fg-one-draw"),
            ((Fraction(r, n) + Fraction(r - 1, n - 1)) * prize - fee, "fg-probability-added"),
            (ev, "fg-forgets-entry"),
            (fee - ev, "fg-reversed-sign"),
        ],
        allow_negative=True,
    )
    enough(dist, 3)
    greens = n - r
    text = (
        f"A bag holds {r} red counters and {greens} green counters. A player pays {dollars(fee)} to draw two counters at random without putting the first back. "
        f"The player wins ${prize} if both counters are red and nothing otherwise. What is the player's expected gain per play, in dollars? (A loss is a negative gain.)"
    )
    working = [
        f"P(both red) = {r}/{n} × {r - 1}/{n - 1} = {both.numerator}/{both.denominator}.",
        f"Expected prize = {both.numerator}/{both.denominator} × {prize} = {dollars(ev)}; expected gain = {dollars(ev)} - {dollars(fee)} = {dollars(gain)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **MONEY.fields(gain), answer_kind="number")


def _fg_plays_needed(rng):
    n, groups, ev = _game(rng)
    margin = Fraction(rng.randint(1, 8))
    fee = ev + margin
    plays = rng.randint(2, 40) * 25
    target = margin * plays
    dist = WHOLE.dist(
        plays,
        [
            (margin, "fg-games-per-play"),
            (target * margin, "fg-games-multiplies"),
            (target / ev, "fg-games-prize-based"),
            (target / fee, "fg-games-fee-based"),
        ],
    )
    enough(dist, 3)
    text = (
        _describe(rng, n, groups)
        + f" It costs {dollars(fee)} to play. How many plays must the organiser expect to have so that the expected profit is {dollars(target)}?"
    )
    working = [
        f"Expected prize = {dollars(ev)}; expected profit per play = {dollars(fee)} - {dollars(ev)} = {dollars(margin)}.",
        f"Plays = {dollars(target)} / {dollars(margin)} = {plays}.",
    ]
    return dict(text=text, working=working, distractors=dist, **WHOLE.fields(plays), answer_kind="number")


_FG_EASY = (_fg_expected_prize, _fg_expected_gain)
_FG_MEDIUM = (_fg_fair_fee, _fg_organiser)
_FG_HARD = (_fg_missing_prize, _fg_two_draws, _fg_plays_needed)


@generator(
    FG_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_fair_game(rng, tier="easy"):
    """Fair games and expected gain; the tiers are described in the module docstring."""
    pool = {"easy": _FG_EASY, "medium": _FG_MEDIUM, "hard": _FG_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(FG_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_fair_game(rng, tier="exam"):
    """Exam chain: expected prize, expected gain, the fair entry fee and the organiser's expected profit."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        n, groups, ev = _game(rng)
        prizes = [p for _, p in groups]
        fee = ev + Fraction(rng.randint(1, 8)) if rng.random() < 0.6 else Fraction(max(1, int(ev) - rng.randint(1, 3)))
        gain = ev - fee
        if gain == 0:
            raise TooFewDistractors("fair game")
        plays = rng.choice((100, 200, 250, 400, 500))
        profit = -gain * plays
        a_d = MONEY.dist(ev, _fg_dist_for_prize(ev, n, groups))
        b_d = MONEY.dist(
            gain,
            [
                (ev, "fg-forgets-entry"),
                (fee - ev, "fg-reversed-sign"),
                (ev + fee, "fg-adds-entry"),
                (Fraction(sum(prizes), len(prizes)) - fee, "fg-unweighted-average"),
            ],
            allow_negative=True,
        )
        target = Fraction(rng.randint(1, 5))
        new_fee = ev + target
        c_d = MONEY.dist(
            new_fee,
            [
                (ev, "fg-fee-ignores-target-profit"),
                (ev - target, "fg-fee-subtracts-target-profit"),
                (target, "fg-fee-is-target-profit"),
                (sum(prizes) + target, "fg-sum-for-fee"),
            ],
        )
        d_d = MONEY.dist(
            profit,
            [
                (-profit, "fg-profit-sign"),
                (ev * plays, "fg-profit-prize-total"),
                (-gain, "fg-profit-per-play"),
                (fee * plays, "fg-profit-fee-total"),
            ],
            allow_negative=True,
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = _describe(rng, n, groups) + f" It costs {dollars(fee)} to play."
        return dict(
            text=stem,
            parts=[
                part(
                    "What is the expected prize for one play, in dollars?",
                    "number",
                    MONEY.fields(ev),
                    2,
                    [f"Expected prize = {dollars(ev)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    "What is the expected gain per play for the player, in dollars? (A loss is a negative gain.)",
                    "number",
                    MONEY.fields(gain),
                    1,
                    [f"Expected gain = {dollars(ev)} - {dollars(fee)} = {dollars(gain)}."],
                    b_d,
                    follow_through=f"part_a - {fee}",
                ),
                part(
                    f"The organiser wants an expected profit of {dollars(target)} on each play. What entry fee, in dollars, should be charged?",
                    "number",
                    MONEY.fields(new_fee),
                    1,
                    [
                        f"A fair fee is the expected prize, {dollars(ev)}; adding the wanted profit gives {dollars(new_fee)}."
                    ],
                    c_d,
                    follow_through=f"part_a + {target}",
                    sensible_range=(0, None),
                ),
                part(
                    f"The game is played {plays} times. What profit does the organiser expect, in dollars? (A loss is a negative profit.)",
                    "number",
                    MONEY.fields(profit),
                    2,
                    [
                        f"Organiser's expected profit per play = {dollars(-gain)}; over {plays} plays = {dollars(profit)}."
                    ],
                    d_d,
                    follow_through=f"-part_b*{plays}",
                ),
            ],
        )

    return retry_draw(rng, build)
