"""Money from a picture: Australian coins and notes drawn with the ``money`` figure kind.

- ``money_count_collection`` (skill ``f10-money-count-collection``): the learner reads the coins
  in the picture and answers in cents. Easy counts coins of one kind (skip counting); medium adds
  mixed coins and a dollar coin (so the learner must turn a dollar into cents); hard asks how many
  MORE cents are needed to reach a stated $1 or $2, so the total is only the first step.
- ``money_compare_collections`` (skill ``f10-money-compare-collections``): ``answer: mcq``. Easy
  compares two small coin collections, usually with the collection of more coins worth less;
  medium adds notes and dollar coins; hard compares three collections for the most or least,
  built so neither the collection with most items nor the one holding the biggest single coin or
  note is the key.
- ``money_change_from_note`` (skill ``f10-money-change-from-note``): ``answer: mcq`` over dollar
  amounts. Easy is a whole-dollar price paid with a $5 or $10 note, medium adds 50c to the price,
  hard pays with a note plus coins, so the amount paid must be added up first.

The picture is the only place the coins appear: no prompt states a coin, and no label shows a
total. Each topic registers a figure reader that works from ``question.figure`` (plus the price
or target the prompt states, or the words "most"/"least"), not from the key.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.early import DENOMINATIONS, money, money_cents
from ..slots import NAMES

_SMALL = ("5c", "10c", "20c", "50c")
_DOLLAR_COINS = ("$1", "$2")
_NOTES = ("$5", "$10", "$20")
_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}


def _cents(denom: str) -> int:
    return DENOMINATIONS[denom][1]


def _total(items) -> int:
    return sum(_cents(d) for d in items)


def _dollars(cents: int) -> str:
    if cents < 100:
        return f"{cents}c"
    return f"${cents // 100}" if cents % 100 == 0 else f"${cents // 100}.{cents % 100:02d}"


def _unleaked(fig: dict, cands):
    """The candidates the picture's own labels do not show (a wrong total equal to a coin's label would read as printed)."""
    out = []
    for c in cands:
        try:
            figures.assert_no_leak(fig, [c])
        except AssertionError:
            continue
        out.append(c)
    return out


def _by_value(items):
    return sorted(items, key=lambda d: -_cents(d))


# ====================================================================== count a collection

_COUNT_TEXT = {
    "easy": (
        "{n} has these coins. How many cents altogether?",
        "{n} counts these coins. What is the total in cents?",
        "How many cents are {n}'s coins worth?",
    ),
    "medium": (
        "{n} empties a purse of coins. How many cents is that in all?",
        "What is the value of {n}'s coins, in cents?",
        "{n} has this money. How many cents is it worth?",
    ),
    "hard": (
        "{n} wants a toy that costs ${t}. These are {n}'s coins. How many more cents does {n} need?",
        "A book costs ${t}. {n} has these coins. How many more cents does {n} need?",
        "{n} is saving for ${t}. How many more cents does {n} need to save?",
    ),
}


def _draw_count(rng, tier: str):
    """(items in drawing order, target dollars or None)."""
    if tier == "easy":
        d = rng.choice(_SMALL)
        n = rng.randint(2, 5)
        return [d] * n, None
    if tier == "medium":
        n = rng.randint(3, 5)
        items = [rng.choice(_SMALL) for _ in range(n - 1)]
        items.append(rng.choice(_DOLLAR_COINS) if rng.random() < 0.5 else rng.choice(_SMALL))
        check(len(items) == n)
        return _by_value(items), None
    if rng.random() < 0.6:
        items = [rng.choice(_SMALL) for _ in range(rng.randint(3, 5))]
        target = 1
    else:
        items = [rng.choice(_SMALL) for _ in range(rng.randint(2, 4))] + ["$1"]
        target = 2
    rng.shuffle(items)
    return items, target


def _count_answer(items, target) -> int:
    t = _total(items)
    return t if target is None else target * 100 - t


def _build_count(rng, tier: str):
    for _ in attempts(500, "a readable coin collection"):
        items, target = _draw_count(rng, tier)
        total = _total(items)
        if total > 295 or (target is not None and not 0 < total < target * 100 - 4):
            continue
        if tier == "easy" and total > 200:
            continue
        answer = _count_answer(items, target)
        fig = money(collections=[items], alt="a collection of Australian coins")
        try:
            figures.assert_no_leak(fig, [str(answer)])
        except AssertionError:
            continue
        return items, target, answer, fig
    raise AssertionError("unreachable")


def _count_distractors(items, target, answer: int):
    total, n = _total(items), len(items)
    smallest = min(_cents(d) for d in items)
    cands = [n]  # counts each coin as 1
    if target is None:
        dollars = sum(1 for d in items if d in _DOLLAR_COINS)
        if dollars:
            cands.append(
                total
                - sum(_cents(d) for d in items if d in _DOLLAR_COINS)
                + sum(int(d[1:]) for d in items if d in _DOLLAR_COINS)
            )
        cands += [total - smallest, total + smallest, total + 10]
    else:
        cands += [total, answer + 100 if target == 2 else answer + 10, answer + 5, answer - 5]
    return [str(c) for c in cands if c > 0]


@generator("money_count_collection", level=2, presets=_PRESETS)
def money_count_collection(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    items, target, answer, fig = _build_count(rng, tier)
    name = rng.choice(NAMES)
    text = rng.choice(_COUNT_TEXT[tier]).format(n=name, t=target)
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, _unleaked(fig, _count_distractors(items, target, answer)))
    check(len(distractors) >= 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_count(fig, text) == answer_text, "the figure does not show the keyed total")
    if target is None:
        working = [" + ".join(_dollars(_cents(d)) if _cents(d) >= 100 else d for d in items) + f" = {answer} cents."]
    else:
        working = [f"The coins are worth {_total(items)} cents. ${target} is {target * 100} cents, so {answer} more."]
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_count(fig: dict, text: str) -> str:
    total = money_cents(fig)
    target = re.search(r"\$(\d+)", text)
    return str(total if target is None else int(target.group(1)) * 100 - total)


figures.register_reader("money_count_collection", lambda q: _read_count(q.figure, q.text))


# ====================================================================== compare collections

_SAME = "Same value"

_COMPARE_TEXT = {
    "easy": (
        "{n} has two collections of coins. Which is worth more?",
        "{n} looks at two collections of coins. Which is worth more?",
        "{n} compares two collections of coins. Which is worth more?",
    ),
    "medium": (
        "{n} has two collections of money. Which is worth more?",
        "{n} looks at the coins and notes. Which collection is worth more?",
        "{n} compares two piles of money. Which collection is worth more?",
    ),
    "hard": (
        "{n} has three collections of money. Which is worth the {w}?",
        "{n} looks at three collections of money. Which is worth the {w}?",
        "{n} compares three collections of coins and notes. Which is worth the {w}?",
    ),
}


def _pool(tier: str):
    return _SMALL if tier == "easy" else _SMALL + _DOLLAR_COINS + _NOTES


def _draw_collection(rng, tier: str, lo: int, hi: int):
    return _by_value([rng.choice(_pool(tier)) for _ in range(rng.randint(lo, hi))])


def _repartition(rng, cents: int):
    """The same value as ``cents`` made from a random choice of small coins."""
    left, out = cents, []
    while left > 0:
        out.append(rng.choice([d for d in _SMALL if _cents(d) <= left]))
        left -= _cents(out[-1])
    return _by_value(out)


def _compare_pair(rng, tier: str):
    """Two collections plus the intended key for the easy and medium tiers."""
    want = rng.choice(("Collection A", "Collection B", _SAME) if tier == "easy" else ("Collection A", "Collection B"))
    trap = rng.random() < 0.7  # the winner holds FEWER items than the loser
    for _ in attempts(3000, "a comparison with the intended key"):
        a = _draw_collection(rng, tier, 1, 6)
        b = _repartition(rng, _total(a)) if want == _SAME and _total(a) <= 200 else _draw_collection(rng, tier, 1, 6)
        ta, tb = _total(a), _total(b)
        if max(ta, tb) > 3000 or len(a) == len(b) or len(a) > 6 or len(b) > 6:
            continue
        key = _SAME if ta == tb else ("Collection A" if ta > tb else "Collection B")
        if key != want:
            continue
        if want != _SAME and trap and ((ta > tb) != (len(a) < len(b))):
            continue
        if tier == "medium" and not any(d in _NOTES for d in a + b):
            continue
        if want == _SAME and abs(ta - tb) != 0:
            continue
        return a, b, key
    raise AssertionError("unreachable")


def _compare_triple(rng):
    want_most = rng.random() < 0.5
    for _ in attempts(4000, "three collections with a clear key"):
        cols = [_draw_collection(rng, "hard", 2, 6) for _ in range(3)]
        totals = [_total(c) for c in cols]
        ordered = sorted(totals)
        if want_most and ordered[-1] == ordered[-2] or (not want_most and ordered[0] == ordered[1]):
            continue
        k = totals.index(ordered[-1] if want_most else ordered[0])
        sizes = [len(c) for c in cols]
        biggest = [max(_cents(d) for d in c) for c in cols]
        # neither "most items" nor "holds the biggest single item" may pick out the key
        if sizes.count(max(sizes)) != 1 or sizes.index(max(sizes)) == k:
            continue
        if biggest.count(max(biggest)) != 1 or biggest.index(max(biggest)) == k:
            continue
        return cols, k, want_most
    raise AssertionError("unreachable")


@generator("money_compare_collections", level=3, presets=_PRESETS)
def money_compare_collections(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    if tier == "hard":
        cols, k, most = _compare_triple(rng)
        answer_text = f"Collection {'ABC'[k]}"
        options = ["Collection A", "Collection B", "Collection C"]
        text = rng.choice(_COMPARE_TEXT[tier]).format(n=name, w="most" if most else "least")
        alt = "three collections of Australian coins and notes"
    else:
        a, b, answer_text = _compare_pair(rng, tier)
        cols = [a, b]
        options = ["Collection A", "Collection B", _SAME]
        text = rng.choice(_COMPARE_TEXT[tier]).format(n=name)
        alt = "two collections of Australian coins and notes"
    fig = money(collections=cols, alt=alt)
    distractors = pick_distractors(answer_text, options, n=2)
    check(len(distractors) == 2)
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_compare(fig, text) == answer_text, "the figure does not show the keyed collection")
    totals = ", ".join(f"Collection {'ABC'[i]} is {_dollars(money_cents(fig, i))}" for i in range(len(cols)))
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[f"Add up each collection: {totals}."],
        figure=fig,
    )


def _read_compare(fig: dict, text: str) -> str:
    totals = [money_cents(fig, i) for i in range(len(fig["groups"]))]
    if len(totals) == 3:
        pick = max(totals) if "most" in text else min(totals)
        check(totals.count(pick) == 1, "the extreme collection is not unique")
        return f"Collection {fig['groups'][totals.index(pick)]['label']}"
    if totals[0] == totals[1]:
        return _SAME
    return f"Collection {fig['groups'][0 if totals[0] > totals[1] else 1]['label']}"


figures.register_reader("money_compare_collections", lambda q: _read_compare(q.figure, q.text))


# ====================================================================== change from a note

_CHANGE_TEXT = {
    "easy": (
        "{n} buys a toy for {p}. {n} pays with this note. How much change?",
        "A book costs {p}. {n} pays with this note. What is the change?",
        "{n} pays for a kite that costs {p} with this note. How much change does {n} get?",
    ),
    "medium": (
        "{n} buys a drink for {p}. {n} pays with this note. How much change?",
        "A pencil case costs {p}. {n} hands over this note. What is the change?",
        "{n} pays for a game that costs {p}. {n} uses this note. How much change is given back?",
    ),
    "hard": (
        "{n} buys a snack for {p}. {n} pays with this money. How much change?",
        "A ticket costs {p}. {n} pays with all of this money. What is the change?",
        "{n} pays for a sticker book that costs {p} with all of this money. How much change is given back?",
    ),
}


def _draw_change(rng, tier: str):
    """(paid items, price in cents)."""
    if tier == "easy":
        note = rng.choice(("$5", "$10"))
        price = 100 * rng.randint(1, 4 if note == "$5" else 9)
        return [note], price
    if tier == "medium":
        note = rng.choice(("$5", "$10"))
        price = 100 * rng.randint(1, 4 if note == "$5" else 9) + 50
        return [note], price
    paid = [rng.choice(("$5", "$10"))] + [
        rng.choice(("50c", "$1", "$2", "20c", "10c")) for _ in range(rng.randint(1, 3))
    ]
    price = rng.randint(100, 1800) // 5 * 5
    return paid, price


def _change_text(p: int) -> str:
    return _dollars(p)


def _change_distractors(items, price: int, change: int):
    paid = _total(items)
    whole, cents = divmod(price, 100)
    top = paid // 100
    cands = [
        change + 100,
        change - 100,
        (top - whole) * 100 + cents,  # takes the dollars apart and copies the cents across
        paid + price,
    ]
    if len(items) > 1:
        cands.append(_cents(items[0]) - price)  # forgets the coins
    cands += [change + 50, change + 5]
    return [_dollars(c) for c in cands if c > 0 and c != change]


@generator("money_change_from_note", level=3, presets=_PRESETS)
def money_change_from_note(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a change question"):
        items, price = _draw_change(rng, tier)
        paid = _total(items)
        change = paid - price
        if not 0 < change < (700 if tier == "hard" else 1000):
            continue
        fig = money(collections=[items], alt="the money used to pay")
        answer_text = _dollars(change)
        try:
            figures.assert_no_leak(fig, [answer_text])
        except AssertionError:
            continue
        break
    name = rng.choice(NAMES)
    text = rng.choice(_CHANGE_TEXT[tier]).format(n=name, p=_change_text(price))
    distractors = pick_distractors(answer_text, _unleaked(fig, _change_distractors(items, price, change)), n=3)
    check(len(distractors) >= 2, "too few change distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_change(fig, text) == answer_text, "the figure does not show the keyed change")
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[
            f"The money is worth {_dollars(paid)}. {_dollars(paid)} take away {_dollars(price)} is {answer_text}."
        ],
        figure=fig,
    )


def _read_change(fig: dict, text: str) -> str:
    price = re.search(r"\$(\d+)(?:\.(\d\d))?", text)
    cents = int(price.group(1)) * 100 + (int(price.group(2)) if price.group(2) else 0)
    return _dollars(money_cents(fig) - cents)


figures.register_reader("money_change_from_note", lambda q: _read_change(q.figure, q.text))
