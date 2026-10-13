"""Generates `senior-calculate-the-dividend-paid-on-a-portfolio-of` questions (ACMGM008): complete the blank
cells of a share-portfolio table. `answer: table`; every blank is a dollar amount to the cent, or a percentage
or ratio to one decimal place, and the question says which.

Scope: a portfolio of three or four fictional companies. The learner works out the dividend paid on each holding
(shares held x dividend per share, the per-share dividend sometimes given as a percentage of the share price),
the dividend yield, and the price-to-earnings (P/E) ratio, and totals the dividend paid. Which cells are blank
is random, so the answer cannot be guessed from the layout. Every value is built from exact `Fraction`
arithmetic (cents and tenths of a percent) and the generator recomputes each blank a second way from the
finished table before returning.

Ladder (reasoning escalation, not size):
  easy:   three holdings, the dividend per share given in dollars. Multiply by the shares held, then total.
  medium: four holdings. The dividend is declared either as dollars per share or as a percentage of the share
          price, so the learner converts the percentage to dollars per share before multiplying, then totals.
  hard:   four holdings with earnings per share. The learner chooses the right quotient for each column:
          dividend yield = dividend per share / share price, P/E = share price / earnings per share, as well as
          the dividend paid, each to the stated precision.
  exam:   three holdings with different cells left out of each row. The learner works backwards as well as
          forwards: the share price from the P/E ratio and earnings per share, the dividend per share from the
          yield, the number of shares from the dividend paid, and the missing yield or P/E ratio from the rest.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase

from ._senior_table_kit import build_answer, finish_wrong, grid_text, money, one_dp, pick_blanks, rounds_clear

TIERS = ("easy", "medium", "hard", "exam")

_COMPANIES = [
    "Banksia Mining",
    "Coolabah Foods",
    "Wattle Energy",
    "Boab Telecom",
    "Jarrah Retail",
    "Gumtree Health",
    "Kestrel Logistics",
    "Mallee Water",
    "Spinifex Bank",
    "Lorikeet Media",
    "Quokka Brewing",
    "Ironbark Steel",
]

_OWNERS = ["An investor", "A retiree", "A self-managed super fund", "A household"]


def _shares(rng: random.Random) -> int:
    return rng.randrange(100, 2001, 50)


def _dollars(cents) -> Fraction:
    return Fraction(cents, 100)


def _pct_text(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10}"


def _sample_unique(rng, draw, count, key):
    """`count` draws whose `key` values are all different (bounded)."""
    out, seen = [], set()
    for _ in range(count):
        item = draw_until(rng, draw, lambda v: key(v) not in seen, what="a distinct holding")
        seen.add(key(item))
        out.append(item)
    return out


def _total_wrongs(paid_cells, total):
    """Wrong totals: one holding left out, and the per-share dividends added instead."""
    return [("leaves-a-holding-out-of-the-total", total - paid) for paid in sorted(set(paid_cells))[:2]]


# --------------------------------------------------------------------------------------------------
# easy
# --------------------------------------------------------------------------------------------------


def _easy(rng):
    names = rng.sample(_COMPANIES, 3)
    holdings = _sample_unique(
        rng, lambda r: (_shares(r), r.randint(8, 190)), 3, key=lambda v: v[1]
    )  # (shares, dividend per share in cents)
    paid = [_dollars(n * d) for n, d in holdings]
    total = sum(paid)
    columns = ["Company", "Shares held", "Dividend per share ($)", "Dividend paid ($)"]
    rows = [[names[i], holdings[i][0], money(_dollars(holdings[i][1])), money(paid[i])] for i in range(3)]
    rows.append(["Total", "-", "-", money(total)])
    cells = [(i, 3) for i in range(4)]
    blanks = pick_blanks(rng, cells, 3, 4, must=[(3, 3)])
    answer = build_answer(columns, rows, blanks)
    wrong: dict[str, list] = {}
    for i in range(3):
        n, d = holdings[i]
        wrong[f"{i},3"] = [
            ("gives-the-dividend-per-share-as-the-total", _dollars(d)),
            ("slips-a-decimal-place", paid[i] * 10),
            ("slips-a-decimal-place", paid[i] / 10),
        ]
    wrong["3,3"] = _total_wrongs(paid, total) + [
        ("totals-the-per-share-dividends", sum(_dollars(d) for _n, d in holdings))
    ]
    stem = phrase(
        rng,
        [
            "{owner} holds shares in three companies. The table shows the number of shares held in each company "
            "and the dividend each company paid per share. The dividend paid on a holding is the number of shares "
            "held multiplied by the dividend per share. Complete the table, giving each amount in dollars to the "
            "cent. Fill in each ? with a number.",
            "{owner} owns shares in three companies, and each company paid a dividend per share. The dividend "
            "paid on a holding is the number of shares held multiplied by the dividend per share, and the total is "
            "the dividend paid on the whole portfolio. Complete the table, with each amount in dollars to the cent. "
            "Fill in each ? with a number.",
        ],
        owner=rng.choice(_OWNERS),
    )
    working = [f"{names[i]}: {holdings[i][0]} x {money(_dollars(holdings[i][1]))} = {money(paid[i])}" for i in range(3)]
    working.append(f"Total: {money(total)}")
    return answer, stem, wrong, working


# --------------------------------------------------------------------------------------------------
# medium
# --------------------------------------------------------------------------------------------------


def _medium_holding(rng, kind):
    """(shares, price cents, kind, dividend per share cents, declared text)."""

    def draw(r):
        price = r.randrange(800, 9001, 50)
        if kind == "pct":
            tenths = r.randint(12, 60)
            return price, tenths, Fraction(price * tenths, 1000)
        dps = r.randrange(10, 400, 5)
        return price, dps, Fraction(dps)

    price, param, dps = draw_until(
        rng,
        draw,
        lambda v: v[2].denominator == 1 and Fraction(v[2], v[0]) * 100 <= 7 and Fraction(v[2], v[0]) * 100 >= 1,
        what="a holding with a whole-cent dividend per share",
    )
    declared = f"{_pct_text(param)}% of share price" if kind == "pct" else f"${money(_dollars(param))} per share"
    return _shares(rng), price, kind, int(dps), declared, param


def _medium(rng):
    names = rng.sample(_COMPANIES, 4)
    kinds = ["pct", "pct", "dollar", "dollar"]
    rng.shuffle(kinds)
    holdings = [_medium_holding(rng, k) for k in kinds]
    paid = [_dollars(h[0] * h[3]) for h in holdings]
    total = sum(paid)
    columns = ["Company", "Shares held", "Share price ($)", "Dividend declared", "Dividend paid ($)"]
    rows = [[names[i], h[0], money(_dollars(h[1])), h[4], money(paid[i])] for i, h in enumerate(holdings)]
    rows.append(["Total", "-", "-", "-", money(total)])
    pct_rows = [i for i, h in enumerate(holdings) if h[2] == "pct"]
    must = [(4, 4), (rng.choice(pct_rows), 4)]
    blanks = pick_blanks(rng, [(i, 4) for i in range(5)], 4, 5, must=must)
    answer = build_answer(columns, rows, blanks)
    wrong: dict[str, list] = {}
    for i, h in enumerate(holdings):
        shares, _price, kind, dps, _decl, param = h
        options = [("gives-the-dividend-per-share-as-the-total", _dollars(dps))]
        if kind == "pct":
            options += [
                ("reads-the-percentage-as-dollars-per-share", Fraction(shares * param, 10)),
                ("ignores-the-share-price", Fraction(shares * param, 1000)),
            ]
        else:
            options.append(("slips-a-decimal-place", paid[i] / 10))
        wrong[f"{i},4"] = options
    wrong["4,4"] = _total_wrongs(paid, total)
    stem = phrase(
        rng,
        [
            "{owner} holds shares in four companies. Each company declared its dividend either as a number of "
            "dollars per share or as a percentage of its share price. Complete the table, where the dividend paid "
            "on a holding is the number of shares held multiplied by the dividend per share, and the total is the "
            "dividend paid on the whole portfolio. Give each amount in dollars to the cent. Fill in each ? with a "
            "number.",
            "The table shows {owner_l} portfolio of shares in four companies. A dividend is declared as dollars per "
            "share or as a percentage of the share price. Complete the dividend paid on each holding (shares held "
            "multiplied by the dividend per share) and on the whole portfolio, each in dollars to the cent. Fill in "
            "each ? with a number.",
        ],
        owner=rng.choice(_OWNERS),
        owner_l=rng.choice(["an investor's", "a retiree's", "a household's"]),
    )
    working = []
    for i, h in enumerate(holdings):
        working.append(f"{names[i]}: dividend per share ${money(_dollars(h[3]))}; {h[0]} x that = {money(paid[i])}")
    working.append(f"Total: {money(total)}")
    return answer, stem, wrong, working


# --------------------------------------------------------------------------------------------------
# hard
# --------------------------------------------------------------------------------------------------


def _hard_holding(rng):
    def draw(r):
        return r.randrange(800, 9501, 50), r.randrange(40, 900, 5), r.randrange(10, 400, 5)

    def ok(v):
        price, eps, dps = v
        pe = Fraction(price, eps)
        yield_pct = Fraction(dps, price) * 100
        return (
            9 <= pe <= 30
            and Fraction(1, 5) <= Fraction(dps, eps) <= Fraction(9, 10)
            and 1 <= yield_pct <= 7
            and rounds_clear(yield_pct, 1)
            and rounds_clear(pe, 1)
        )

    price, eps, dps = draw_until(rng, draw, ok, what="a holding with a clean yield and P/E ratio")
    return _shares(rng), price, eps, dps


def _hard(rng):
    names = rng.sample(_COMPANIES, 4)
    holdings = [_hard_holding(rng) for _ in range(4)]
    paid = [_dollars(h[0] * h[3]) for h in holdings]
    yields = [Fraction(h[3], h[1]) * 100 for h in holdings]
    pes = [Fraction(h[1], h[2]) for h in holdings]
    total = sum(paid)
    columns = [
        "Company",
        "Shares held",
        "Share price ($)",
        "Earnings per share ($)",
        "Dividend per share ($)",
        "Dividend paid ($)",
        "Dividend yield (%)",
        "P/E ratio",
    ]
    rows = [
        [
            names[i],
            h[0],
            money(_dollars(h[1])),
            money(_dollars(h[2])),
            money(_dollars(h[3])),
            money(paid[i]),
            one_dp(yields[i]),
            one_dp(pes[i]),
        ]
        for i, h in enumerate(holdings)
    ]
    rows.append(["Total", "-", "-", "-", "-", money(total), "-", "-"])
    cells = [(i, c) for i in range(4) for c in (5, 6, 7)] + [(4, 5)]
    must = [(4, 5), (rng.randrange(4), 5), (rng.randrange(4), 6), (rng.randrange(4), 7)]
    blanks = pick_blanks(rng, cells, 7, 9, must=must)
    answer = build_answer(columns, rows, blanks)
    wrong: dict[str, list] = {}
    for i, h in enumerate(holdings):
        shares, price, eps, dps = h
        wrong[f"{i},5"] = [("gives-the-dividend-per-share-as-the-total", _dollars(dps))]
        wrong[f"{i},6"] = [
            ("divides-by-the-earnings-instead-of-the-price", Fraction(dps, eps) * 100),
            ("uses-the-dividend-paid-instead-of-the-dividend-per-share", Fraction(shares * dps, price) * 100),
        ]
        wrong[f"{i},7"] = [
            ("inverts-the-price-to-earnings-ratio", Fraction(eps, price)),
            ("divides-the-price-by-the-dividend", Fraction(price, dps)),
        ]
    wrong["4,5"] = _total_wrongs(paid, total)
    stem = phrase(
        rng,
        [
            "{owner} holds shares in four companies. For each holding the table shows the number of shares held, "
            "the share price, the earnings per share and the dividend per share. Dividend paid = shares held x "
            "dividend per share; dividend yield = dividend per share / share price, as a percentage; P/E ratio = "
            "share price / earnings per share. Complete the table, giving dollar amounts to the cent (2 decimal places) and the yield "
            "and the P/E ratio to one decimal place. Fill in each ? with a number.",
            "The table compares four shares in {owner_l} portfolio. Dividend paid = shares held x dividend per share; "
            "dividend yield = dividend per share / share price, as a percentage; P/E ratio = share price / earnings "
            "per share. Complete the table, giving dollar amounts to the cent (2 decimal places) and the yield and the P/E ratio to "
            "one decimal place. Fill in each ? with a number.",
        ],
        owner=rng.choice(_OWNERS),
        owner_l=rng.choice(["an investor's", "a retiree's", "a household's"]),
    )
    working = [f"{names[i]}: paid {money(paid[i])}; yield {one_dp(yields[i])}%; P/E {one_dp(pes[i])}" for i in range(4)]
    working.append(f"Total dividend paid: {money(total)}")
    return answer, stem, wrong, working


# --------------------------------------------------------------------------------------------------
# exam
# --------------------------------------------------------------------------------------------------

_PATTERNS = ("price-given", "pe-given", "paid-given")


def _exam_holding(rng, pattern):
    """A holding as a dict of exact values (cents unless noted), built so that every given cell is exact."""

    def draw_price_given(r):
        price = r.randrange(800, 9001, 50)
        eps = r.randrange(40, 600, 5)
        return price, eps, r.randint(12, 60)

    def draw_pe_given(r):
        eps = r.randrange(60, 401, 10)
        pe_tenths = r.randint(90, 280)
        return Fraction(eps * pe_tenths, 10), eps, r.randint(12, 60), pe_tenths

    def draw_paid_given(r):
        price = r.randrange(800, 9001, 50)
        eps = r.randrange(40, 600, 5)
        return price, eps, r.randrange(10, 400, 5)

    if pattern == "price-given":
        price, eps, y_tenths = draw_until(
            rng,
            draw_price_given,
            lambda v: (
                Fraction(v[0] * v[2], 1000).denominator == 1
                and 9 <= Fraction(v[0], v[1]) <= 30
                and rounds_clear(Fraction(v[0], v[1]), 1)
                and Fraction(1, 5) <= Fraction(v[0] * v[2], 1000 * v[1]) <= Fraction(9, 10)
            ),
            what="a price-given holding",
        )
        dps = Fraction(price * y_tenths, 1000)
        yield_pct = Fraction(y_tenths, 10)
    elif pattern == "pe-given":
        price, eps, y_tenths, _pe = draw_until(
            rng,
            draw_pe_given,
            lambda v: (
                v[0].denominator == 1
                and Fraction(int(v[0]) * v[2], 1000).denominator == 1
                and Fraction(1, 5) <= Fraction(int(v[0]) * v[2], 1000 * v[1]) <= Fraction(9, 10)
            ),
            what="a P/E-given holding",
        )
        price = int(price)
        dps = Fraction(price * y_tenths, 1000)
        yield_pct = Fraction(y_tenths, 10)
    else:
        price, eps, dps = draw_until(
            rng,
            draw_paid_given,
            lambda v: (
                9 <= Fraction(v[0], v[1]) <= 30
                and rounds_clear(Fraction(v[0], v[1]), 1)
                and 1 <= Fraction(v[2], v[0]) * 100 <= 7
                and Fraction(1, 5) <= Fraction(v[2], v[1]) <= Fraction(9, 10)
                and rounds_clear(Fraction(v[2], v[0]) * 100, 1)
            ),
            what="a paid-given holding",
        )
        dps = Fraction(dps)
        yield_pct = Fraction(dps, price) * 100
    shares = _shares(rng)
    return dict(
        pattern=pattern,
        shares=shares,
        price=Fraction(price),
        eps=Fraction(eps),
        dps=dps,
        paid=shares * dps,
        yield_pct=yield_pct,
        pe=Fraction(price, eps),
    )


def _exam(rng):
    for _ in range(50):
        patterns = [rng.choice(_PATTERNS) for _ in range(3)]
        if len(set(patterns)) >= 2:
            break
    else:  # pragma: no cover - a bounded redraw that cannot fail in practice
        patterns = list(_PATTERNS)
    names = rng.sample(_COMPANIES, 3)
    holdings = [_exam_holding(rng, p) for p in patterns]
    total = sum(h["paid"] for h in holdings)
    columns = [
        "Company",
        "Shares held",
        "Earnings per share ($)",
        "Share price ($)",
        "P/E ratio",
        "Dividend per share ($)",
        "Dividend yield (%)",
        "Dividend paid ($)",
    ]
    rows, blanks = [], {(3, 7)}
    for i, h in enumerate(holdings):
        rows.append(
            [
                names[i],
                h["shares"],
                money(h["eps"] / 100),
                money(h["price"] / 100),
                one_dp(h["pe"]),
                money(h["dps"] / 100),
                one_dp(h["yield_pct"]),
                money(h["paid"] / 100),
            ]
        )
        # price-given: P/E, dividend per share, dividend paid are left out.
        # pe-given:    share price, dividend per share, dividend paid are left out.
        # paid-given:  shares held, P/E ratio, yield are left out.
        left = {"price-given": (4, 5, 7), "pe-given": (3, 5, 7), "paid-given": (1, 4, 6)}[h["pattern"]]
        blanks |= {(i, c) for c in left}
    rows.append(["Total", "-", "-", "-", "-", "-", "-", money(total / 100)])
    answer = build_answer(columns, rows, blanks)
    wrong: dict[str, list] = {"3,7": _total_wrongs([h["paid"] / 100 for h in holdings], total / 100)}
    for i, h in enumerate(holdings):
        d = h["dps"] / 100
        wrong[f"{i},7"] = [("gives-the-dividend-per-share-as-the-total", d)]
        wrong[f"{i},5"] = [("applies-the-yield-to-the-earnings", h["eps"] * h["yield_pct"] / 10000)]
        wrong[f"{i},4"] = [
            ("inverts-the-price-to-earnings-ratio", h["eps"] / h["price"]),
            ("divides-the-price-by-the-dividend", h["price"] / h["dps"]),
        ]
        wrong[f"{i},3"] = [("divides-by-the-ratio-instead-of-multiplying", h["eps"] / h["pe"] / 100)]
        wrong[f"{i},1"] = [("divides-by-the-share-price", Fraction(h["paid"], h["price"]))]
        wrong[f"{i},6"] = [
            ("divides-by-the-earnings-instead-of-the-price", h["dps"] / h["eps"] * 100),
            ("uses-the-dividend-paid-instead-of-the-dividend-per-share", h["paid"] / h["price"] * 100),
        ]
    # Only the blank cells matter; drop options for cells that are given.
    wrong = {k: v for k, v in wrong.items() if k in answer["blanks"]}
    stem = phrase(
        rng,
        [
            "The table shows part of the record of {owner_l} portfolio of shares in three companies. Some entries "
            "are missing, and different entries are missing from different rows. Dividend paid = shares held x "
            "dividend per share; dividend yield = dividend per share / share price, as a percentage; P/E ratio = "
            "share price / earnings per share. Every entry that is shown is exact. Complete the table, giving "
            "dollar amounts to the cent (2 decimal places) and the yield and the P/E ratio to one decimal place. Fill in each ? with a "
            "number.",
            "{owner_u} keeps this record of three shares, but some entries are missing and they are not the same "
            "in every row. Dividend paid = shares held x dividend per share; dividend yield = dividend per share / "
            "share price, as a percentage; P/E ratio = share price / earnings per share. Every entry that is shown "
            "is exact. Complete the table, giving dollar amounts to the cent (2 decimal places) and the yield and the P/E ratio to "
            "one decimal place. Fill in each ? with a number.",
        ],
        owner_l=rng.choice(["an investor's", "a retiree's", "a household's"]),
        owner_u=rng.choice(_OWNERS),
    )
    working = [
        f"{names[i]}: shares {h['shares']}; price {money(h['price'] / 100)}; P/E {one_dp(h['pe'])}; "
        f"dividend per share {money(h['dps'] / 100)}; yield {one_dp(h['yield_pct'])}%; paid {money(h['paid'] / 100)}"
        for i, h in enumerate(holdings)
    ]
    working.append(f"Total dividend paid: {money(total / 100)}")
    return answer, stem, wrong, working


# --------------------------------------------------------------------------------------------------


def _recheck(answer: dict) -> None:
    """Recompute the blanks a second way, from the finished table's own relationships."""
    columns = answer["columns"]
    rows = answer["rows"]
    paid_col = columns.index("Dividend paid ($)")
    body = rows[:-1]
    check(
        sum(Fraction(r[paid_col]) for r in body) == Fraction(rows[-1][paid_col]),
        "total dividend disagrees with the holdings",
    )
    if "Dividend per share ($)" in columns and "Shares held" in columns:
        dps_col, shares_col = columns.index("Dividend per share ($)"), columns.index("Shares held")
        for r in body:
            check(Fraction(r[shares_col]) * Fraction(r[dps_col]) == Fraction(r[paid_col]), "paid is not shares x dps")
    if "P/E ratio" in columns and "Share price ($)" in columns and "Earnings per share ($)" in columns:
        pe_c, p_c, e_c = (
            columns.index("P/E ratio"),
            columns.index("Share price ($)"),
            columns.index("Earnings per share ($)"),
        )
        for r in body:
            check(abs(Fraction(r[p_c]) / Fraction(r[e_c]) - Fraction(r[pe_c])) <= Fraction(1, 20), "P/E disagrees")
    if "Dividend yield (%)" in columns and "Dividend per share ($)" in columns and "Share price ($)" in columns:
        y_c, d_c, p_c = (
            columns.index("Dividend yield (%)"),
            columns.index("Dividend per share ($)"),
            columns.index("Share price ($)"),
        )
        for r in body:
            check(
                abs(Fraction(r[d_c]) / Fraction(r[p_c]) * 100 - Fraction(r[y_c])) <= Fraction(1, 20), "yield disagrees"
            )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


@generator(
    "senior_dividend_portfolio_table",
    level=13,
    presets={t: dict(tier=t) for t in TIERS},
)
def senior_dividend_portfolio_table(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in _BUILDERS:
        raise ValueError(f"unknown tier {tier!r}")
    answer, stem, wrong, working = _BUILDERS[tier](rng)
    _recheck(_filled(answer))
    answer_text, distractors, notes = finish_wrong(rng, answer, wrong)
    return dict(
        text=f"{stem} {grid_text(answer)}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
        option_notes=notes,
    )


def _filled(answer: dict) -> dict:
    """The answer with every blank filled in (rows no longer hold None), for the consistency re-check."""
    rows = [list(r) for r in answer["rows"]]
    for key, value in answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    return {"columns": answer["columns"], "rows": rows, "blanks": {}}
