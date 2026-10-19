"""Generates `w7a_dp_table` questions (easy, medium, hard) and the exam chain `exd_x_w7a_dp_table`.

``uc2-dynamic-programming-tables`` (``discrete-mathematics``, answer ``number``): fill in the table of a dynamic
programme by hand. Every answer is a whole number read from the table, and every problem is one where the obvious greedy
choice is not the best.

* easy: the cheapest way up a staircase with a toll on each step (steps of one or two); the fewest coins for an amount when
  the largest-coin-first rule fails; the cheapest route through a network with two stages.
* medium: the best value of a 0/1 knapsack with four or five items; the fewest coins with four denominations; the cheapest
  route through a network with three stages; a longer staircase with moves of one, two or three steps.
* hard: the number of ways to make an amount from coins (order does not matter); the length of a longest common subsequence
  of two strings; how much more a knapsack is worth when items may be reused.
* exam (``exd_x_w7a_dp_table``): one 0/1 knapsack with four items. (a) the best value using only the first two items,
  (b) the best value using all four, (c) the weight of the one optimal selection, (d) the extra value one more unit of
  capacity would bring.

Every answer is found twice: by the table (the recurrence) and by an exhaustive search that never builds a table (every subset,
every route, every multiset of coins, every subsequence).
"""

from __future__ import annotations

from collections import deque
from itertools import product

from ..core import attempts, check, draw_until, generator, phrase
from ._w6a_kit import TooFewDistractors, enough, num_dist, num_fields, part, retry_draw

TOPIC = "w7a_dp_table"
EXAM_TOPIC = "exd_x_w7a_dp_table"

AREA = "dynamic-programming"
MISCONCEPTIONS = [
    (
        "dp-stairs-no-skipping",
        "Climbs one step at a time throughout, so pays every toll and never uses a two-step move.",
        AREA,
    ),
    ("dp-greedy-next-step", "Always moves to the cheaper of the next two steps, without looking further ahead.", AREA),
    ("dp-stops-one-short", "Gives the least cost of reaching the step before the top.", AREA),
    ("dp-coin-greedy", "Takes the largest coin that fits each time, which is not always the fewest coins.", AREA),
    (
        "dp-coin-ignores-remainder",
        "Divides the amount by the largest coin and rounds up, ignoring how the remainder is made.",
        AREA,
    ),
    (
        "dp-greedy-cheapest-arc",
        "Takes the cheapest arc out of each node in turn instead of finding the cheapest whole route.",
        AREA,
    ),
    (
        "dp-cheapest-arc-per-stage",
        "Adds the cheapest arc of each stage, although those arcs do not join up into one route.",
        AREA,
    ),
    (
        "dp-knapsack-greedy-ratio",
        "Takes items in order of value per kilogram until nothing more fits, which is not always the best.",
        AREA,
    ),
    ("dp-knapsack-greedy-value", "Takes the most valuable items first until nothing more fits.", AREA),
    ("dp-knapsack-reuse-items", "Allows an item to be taken more than once.", AREA),
    ("dp-knapsack-ignores-capacity", "Adds the values of all the items, ignoring the weight limit.", AREA),
    ("dp-ways-counts-orderings", "Counts different orders of the same coins as different ways.", AREA),
    ("dp-ways-fewest-coins", "Gives the fewest coins where the number of ways was asked for.", AREA),
    ("dp-ways-misses-a-coin", "Leaves one coin value out when counting the ways.", AREA),
    (
        "dp-lcs-contiguous",
        "Finds the longest common run of consecutive letters instead of the longest common subsequence.",
        AREA,
    ),
    ("dp-lcs-shorter-length", "Gives the length of the shorter string as the longest common subsequence.", AREA),
    (
        "dp-lcs-aligned-matches",
        "Counts the positions where the two strings have the same letter, without letting letters shift.",
        AREA,
    ),
    ("dp-difference-gives-one-value", "Gives one of the two values instead of the difference between them.", AREA),
    (
        "dp-table-wrong-row",
        "Reads the entry for all the items where the entry for the first few items was asked for.",
        AREA,
    ),
    ("dp-weight-assumed-full", "Takes the optimal selection to fill the knapsack exactly.", AREA),
    ("dp-weight-of-all-items", "Gives the total weight of all the items rather than of the ones chosen.", AREA),
    ("dp-arithmetic-slip", "Uses the right table but is out by one or two in the arithmetic.", AREA),
]


# -- stairs -----------------------------------------------------------------------------------------------------------------


def _stairs_table(tolls, moves=(1, 2)) -> list[int]:
    """f[i] = least total toll to stand on step i (step 0 is the ground, free), moving up by any of `moves` steps."""
    inf = 10**9
    f = [0]
    for i, c in enumerate(tolls, start=1):
        f.append(c + min((f[i - m] for m in moves if i - m >= 0), default=inf))
    return f


def _stairs_search(tolls, moves=(1, 2)) -> int:
    """The least toll over every way of writing n as a sum of the moves, found without a table."""
    n = len(tolls)
    best = 10**9

    def walk(at, paid):
        nonlocal best
        if at == n:
            best = min(best, paid)
            return
        for step in moves:
            if at + step <= n:
                walk(at + step, paid + tolls[at + step - 1])

    walk(0, 0)
    return best


def _stairs_greedy(tolls, moves=(1, 2)) -> int:
    """Always move to the cheapest of the steps within reach (the shorter move on a tie)."""
    n, at, paid = len(tolls), 0, 0
    while at < n:
        reach = [m for m in moves if at + m <= n]
        step = min(reach, key=lambda m: (tolls[at + m - 1], m))
        at += step
        paid += tolls[at - 1]
    return paid


def _stairs_question(rng, lo, hi, moves):
    tolls = draw_until(
        rng,
        lambda r: [r.randint(1, 9) for _ in range(r.randint(lo, hi))],
        lambda t: _stairs_greedy(t, moves) > _stairs_search(t, moves) and sum(t) > _stairs_search(t, moves) + 3,
        what="a staircase where the greedy climb is not cheapest",
    )
    f = _stairs_table(tolls, moves)
    answer = f[-1]
    check(answer == _stairs_search(tolls, moves), "table and search disagree")
    cands = [
        (sum(tolls), "dp-stairs-no-skipping"),
        (_stairs_greedy(tolls, moves), "dp-greedy-next-step"),
        (f[-2], "dp-stops-one-short"),
        (answer + 1, "dp-arithmetic-slip"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    n = len(tolls)
    listing = ", ".join(f"step {i + 1}: ${c}" for i, c in enumerate(tolls))
    moves_text = "one step or two steps" if moves == (1, 2) else "one, two or three steps"
    stem = phrase(
        rng,
        (
            "A staircase has {n} steps and a toll is paid on each step you stand on ({listing}). You start on the ground, "
            "where nothing is paid, and each move goes up {moves}. You must finish on step {n}. "
            "What is the least total toll, in dollars?",
            "To reach the top of a {n}-step staircase you pay a toll for every step you land on ({listing}). Each move climbs "
            "{moves}, starting from the ground (free), and you must land on step {n}. What is the smallest total toll, in dollars?",
        ),
        n=n,
        listing=listing,
        moves=moves_text,
    )
    working = [
        "Let f(i) be the least toll to stand on step i, with f(0) = 0. Then f(i) = toll(i) + the smallest of f(i - m) over the allowed moves m.",
        "Table: " + ", ".join(f"f({i}) = {v}" for i, v in enumerate(f)) + f". So the least toll is {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _easy_stairs(rng):
    return _stairs_question(rng, 7, 9, (1, 2))


def _medium_stairs(rng):
    return _stairs_question(rng, 10, 12, (1, 2, 3))


# -- coins ------------------------------------------------------------------------------------------------------------------


def _coins_table(coins, amount) -> int:
    inf = 10**9
    best = [0] + [inf] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    return best[amount]


def _coins_search(coins, amount) -> int:
    """Breadth-first search over the amount paid so far: the fewest coins is the fewest moves."""
    seen = {0}
    queue = deque([(0, 0)])
    while queue:
        paid, used = queue.popleft()
        if paid == amount:
            return used
        for c in coins:
            if paid + c <= amount and paid + c not in seen:
                seen.add(paid + c)
                queue.append((paid + c, used + 1))
    raise AssertionError("amount cannot be made")  # pragma: no cover


def _coins_greedy(coins, amount) -> int:
    used = 0
    for c in sorted(coins, reverse=True):
        used += amount // c
        amount %= c
    return used if amount == 0 else 10**9


_COIN_SETS = (
    (1, 3, 4),
    (1, 4, 6),
    (1, 5, 7),
    (1, 6, 9),
    (1, 5, 8),
    (1, 7, 10),
    (1, 4, 5),
    (2, 5, 6),
    (1, 6, 10),
    (1, 3, 7),
)
_COIN_SETS_BIG = (
    (1, 4, 9, 11),
    (1, 5, 8, 14),
    (1, 3, 11, 13),
    (1, 6, 10, 15),
    (1, 7, 9, 16),
    (1, 5, 12, 17),
    (1, 4, 10, 13),
    (1, 8, 11, 19),
)


def _coin_question(rng, sets, lo, hi, tier):
    def draw(r):
        coins = r.choice(sets)
        return coins, r.randint(lo, hi)

    coins, amount = draw_until(
        rng,
        draw,
        lambda cs: (
            _coins_greedy(*cs) < 10**8
            and _coins_greedy(*cs) > _coins_table(*cs)
            and _coins_table(*cs) >= (5 if tier == "easy" else 7)
        ),
        what="an amount where the largest-coin-first rule fails",
    )
    answer = _coins_table(coins, amount)
    check(answer == _coins_search(coins, amount), "table and search disagree")
    greedy = _coins_greedy(coins, amount)
    biggest = max(coins)
    cands = [
        (greedy, "dp-coin-greedy"),
        (-(-amount // biggest), "dp-coin-ignores-remainder"),
        (answer + 1, "dp-arithmetic-slip"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    names = ", ".join(f"{c}c" for c in coins[:-1]) + f" and {coins[-1]}c"
    stem = phrase(
        rng,
        (
            "A country issues only {n} kinds of coin, worth {names}, and as many of each as you like. What is the fewest coins "
            "that make exactly {amount} cents?",
            "Coins of {names} are available in unlimited supply. Using the table of fewest coins for each amount from 0 up to {amount}, "
            "what is the fewest coins needed to make {amount} cents exactly?",
        ),
        n=len(coins),
        names=names,
        amount=amount,
    )
    working = [
        "Let g(a) be the fewest coins for amount a, with g(0) = 0. Then g(a) is 1 more than the smallest of g(a - c) over the coins c that fit.",
        f"Filling the table up to {amount} gives g({amount}) = {answer}; the largest-coin-first rule would use {greedy}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _easy_coins(rng):
    return _coin_question(rng, _COIN_SETS, 28, 58, "easy")


def _medium_coins(rng):
    return _coin_question(rng, _COIN_SETS_BIG, 45, 95, "medium")


# -- staged routes ----------------------------------------------------------------------------------------------------------


def _stage_graph(rng, widths):
    layers = [["S"]]
    names = iter("ABCDEFGH")
    for w in widths:
        layers.append([next(names) for _ in range(w)])
    layers.append(["T"])
    arcs = {(u, v): rng.randint(1, 9) for a, b in zip(layers, layers[1:]) for u in a for v in b}
    return layers, arcs


def _route_table(layers, arcs) -> int:
    """Backward table: g(T) = 0 and g(node) = min over arcs of (cost + g(next))."""
    g = {"T": 0}
    for layer in reversed(layers[:-1]):
        for u in layer:
            g[u] = min(arcs[(u, v)] + g[v] for (a, v) in arcs if a == u)
    return g["S"]


def _route_search(layers, arcs) -> int:
    best = min(sum(arcs[(r[i], r[i + 1])] for i in range(len(r) - 1)) for r in product(*layers))
    return best


def _route_greedy(layers, arcs) -> int:
    at, total = "S", 0
    for layer in layers[1:]:
        v = min(layer, key=lambda x: (arcs[(at, x)], x))
        total += arcs[(at, v)]
        at = v
    return total


def _route_stage_minima(layers, arcs) -> int:
    return sum(min(arcs[(u, v)] for u in a for v in b) for a, b in zip(layers, layers[1:]))


def _route_question(rng, widths, tier):
    def draw(r):
        return _stage_graph(r, widths)

    layers, arcs = draw_until(
        rng,
        draw,
        lambda g: (
            _route_greedy(*g) > _route_table(*g)
            and _route_stage_minima(*g) < _route_table(*g)
            and _route_table(*g) >= 8
        ),
        what="a staged network where the greedy route is not cheapest",
    )
    answer = _route_table(layers, arcs)
    check(answer == _route_search(layers, arcs), "table and search disagree")
    greedy = _route_greedy(layers, arcs)
    cands = [
        (greedy, "dp-greedy-cheapest-arc"),
        (_route_stage_minima(layers, arcs), "dp-cheapest-arc-per-stage"),
        (answer + 1, "dp-arithmetic-slip"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stages = len(widths) + 1
    spec = "; ".join(f"stage {i + 1}: {' or '.join(layers[i + 1])}" for i in range(len(widths)))
    costs = ", ".join(f"{u}→{v} {c}" for (u, v), c in sorted(arcs.items(), key=lambda kv: (kv[0][0] != "S", kv[0])))
    stem = (
        f"A delivery goes from depot S to a customer T in {stages} legs, passing through exactly one place at each stage ({spec}). "
        f"Every place in one stage is linked to every place in the next. The cost of each leg, in dollars, is {costs}. "
        "What is the least total cost from S to T?"
    )
    working = [
        "Work backwards: for each place, the least cost to finish is the smallest of (cost of the leg + least cost to finish from the place it leads to).",
        f"Doing this stage by stage back to S gives {answer}; always taking the cheapest leg out of the current place would cost {greedy}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _easy_route(rng):
    return _route_question(rng, (rng.choice((2, 3)), 2), "easy")


def _medium_route(rng):
    return _route_question(rng, (2, rng.choice((2, 3)), 2), "medium")


# -- knapsack ---------------------------------------------------------------------------------------------------------------


def _knap_table(items, cap):
    """f[i][w] = best value using the first i items with capacity w."""
    f = [[0] * (cap + 1) for _ in range(len(items) + 1)]
    for i, (wt, val) in enumerate(items, start=1):
        for w in range(cap + 1):
            f[i][w] = f[i - 1][w]
            if wt <= w:
                f[i][w] = max(f[i][w], f[i - 1][w - wt] + val)
    return f


def _knap_subsets(items, cap):
    """(best value, how many subsets reach it, the heaviest... ) over every subset that fits."""
    best, count, chosen = -1, 0, None
    for mask in range(2 ** len(items)):
        wt = sum(items[i][0] for i in range(len(items)) if mask >> i & 1)
        val = sum(items[i][1] for i in range(len(items)) if mask >> i & 1)
        if wt <= cap:
            if val > best:
                best, count, chosen = val, 1, mask
            elif val == best:
                count += 1
    return best, count, chosen


def _knap_unbounded(items, cap) -> int:
    best = [0] * (cap + 1)
    for w in range(1, cap + 1):
        best[w] = best[w - 1]
        for wt, val in items:
            if wt <= w:
                best[w] = max(best[w], best[w - wt] + val)
    return best[cap]


def _knap_greedy(items, cap, key) -> int:
    total, room = 0, cap
    for wt, val in sorted(items, key=key, reverse=True):
        if wt <= room:
            room -= wt
            total += val
    return total


def _ratio_key(item):
    return (item[1] / item[0], item[1])


def _value_key(item):
    return (item[1], -item[0])


def _draw_items(rng, n, cap_lo, cap_hi):
    items = [(rng.randint(2, 7), rng.randint(3, 15)) for _ in range(n)]
    return items, rng.randint(cap_lo, cap_hi)


def _knap_ok(items, cap, require_unique=False, both=True) -> bool:
    best, count, _ = _knap_subsets(items, cap)
    if require_unique and count != 1:
        return False
    if sum(w for w, _ in items) <= cap:
        return False
    greedy_r, greedy_v = _knap_greedy(items, cap, _ratio_key), _knap_greedy(items, cap, _value_key)
    fails = (greedy_r < best, greedy_v < best)
    return (all(fails) if both else any(fails)) and len({w for w, _ in items}) == len(items)


def _item_text(items) -> str:
    return ", ".join(f"item {chr(80 + i)} weighs {w} kg and is worth ${v}" for i, (w, v) in enumerate(items))


def _medium_knapsack(rng):
    n = rng.choice((4, 5))
    items, cap = draw_until(
        rng,
        lambda r: _draw_items(r, n, 8, 12),
        lambda g: _knap_ok(*g),
        tries=6000,
        what="knapsack items where both greedy rules fail",
    )
    f = _knap_table(items, cap)
    answer = f[n][cap]
    check(answer == _knap_subsets(items, cap)[0], "table and subset search disagree")
    cands = [
        (_knap_greedy(items, cap, _ratio_key), "dp-knapsack-greedy-ratio"),
        (_knap_greedy(items, cap, _value_key), "dp-knapsack-greedy-value"),
        (_knap_unbounded(items, cap), "dp-knapsack-reuse-items"),
        (sum(v for _, v in items), "dp-knapsack-ignores-capacity"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = phrase(
        rng,
        (
            "A bag holds at most {cap} kg. There are {n} items, each available once: {items}. Using a table of the best value for "
            "the first i items and each capacity w, what is the greatest total value that fits in the bag, in dollars?",
            "A hiker can carry at most {cap} kg and may take each of {n} items at most once: {items}. What is the greatest total value, "
            "in dollars, the hiker can carry?",
        ),
        cap=cap,
        n=n,
        items=_item_text(items),
    )
    working = [
        "Let f(i, w) be the best value using the first i items with capacity w: f(i, w) = max(f(i - 1, w), f(i - 1, w - weight) + value), where the second choice needs weight ≤ w.",
        f"Filling the table to f({n}, {cap}) gives {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


# -- hard -------------------------------------------------------------------------------------------------------------------


def _ways_table(coins, amount) -> int:
    ways = [1] + [0] * amount
    for c in coins:
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]


def _ways_search(coins, amount) -> int:
    """Count multisets of coins summing to `amount` by choosing how many of each coin (no table)."""

    def walk(i, left):
        if i == len(coins):
            return 1 if left == 0 else 0
        return sum(walk(i + 1, left - k * coins[i]) for k in range(left // coins[i] + 1))

    return walk(0, amount)


def _orderings_table(coins, amount) -> int:
    seq = [1] + [0] * amount
    for a in range(1, amount + 1):
        seq[a] = sum(seq[a - c] for c in coins if c <= a)
    return seq[amount]


_WAY_COINS = ((2, 3, 5), (1, 3, 4), (2, 5, 6), (3, 4, 7), (2, 3, 7), (1, 2, 5), (3, 5, 8), (2, 4, 5), (1, 4, 6))


def _hard_ways(rng):
    coins, amount = draw_until(
        rng,
        lambda r: (r.choice(_WAY_COINS), r.randint(13, 22)),
        lambda g: 6 <= _ways_table(*g) <= 70 and len({_ways_table(*g), _orderings_table(*g)}) == 2,
        what="an amount with a moderate number of ways",
    )
    answer = _ways_table(coins, amount)
    check(answer == _ways_search(coins, amount), "table and search disagree")
    cands = [
        (_orderings_table(coins, amount), "dp-ways-counts-orderings"),
        (_ways_table(coins[1:], amount), "dp-ways-misses-a-coin"),
        (_ways_table(coins[:-1], amount), "dp-ways-misses-a-coin"),
        (_coins_table(coins, amount), "dp-ways-fewest-coins"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v >= 3])
    enough(dist, 3)
    names = ", ".join(f"{c}c" for c in coins[:-1]) + f" and {coins[-1]}c"
    stem = phrase(
        rng,
        (
            "Coins of {names} are available in unlimited supply. In how many different ways can {amount} cents be made? Two ways are the same "
            "if they use the same number of each coin, whatever order the coins are put down in.",
            "How many different collections of coins, taken from {names} coins with as many of each as you like, total exactly {amount} cents? "
            "The order the coins are listed in does not matter.",
        ),
        names=names,
        amount=amount,
    )
    working = [
        "Let w(a) count the ways to make a, adding one coin kind at a time: after allowing a coin c, w(a) becomes w(a) + w(a - c). Start with w(0) = 1.",
        f"Allowing each coin in turn, the table ends with w({amount}) = {answer}. Counting orders as different would give {_orderings_table(coins, amount)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _lcs_table(a: str, b: str) -> int:
    f = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, start=1):
        for j, y in enumerate(b, start=1):
            f[i][j] = f[i - 1][j - 1] + 1 if x == y else max(f[i - 1][j], f[i][j - 1])
    return f[len(a)][len(b)]


def _is_subsequence(sub: str, s: str) -> bool:
    it = iter(s)
    return all(ch in it for ch in sub)


def _lcs_search(a: str, b: str) -> int:
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    best = 0
    for mask in range(2 ** len(short)):
        sub = "".join(short[i] for i in range(len(short)) if mask >> i & 1)
        if len(sub) > best and _is_subsequence(sub, long_):
            best = len(sub)
    return best


def _longest_common_substring(a: str, b: str) -> int:
    best = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a) + 1):
            if a[i:j] in b:
                best = max(best, j - i)
    return best


def _hard_lcs(rng):
    def draw(r):
        alphabet = "ACGT"
        a = "".join(r.choice(alphabet) for _ in range(r.randint(8, 9)))
        b = "".join(r.choice(alphabet) for _ in range(r.randint(8, 9)))
        return a, b

    a, b = draw_until(
        rng,
        draw,
        lambda g: (
            _lcs_table(*g) - _longest_common_substring(*g) >= 2 and _lcs_table(*g) < min(len(g[0]), len(g[1])) - 1
        ),
        what="two strings whose longest common subsequence is longer than any common run",
    )
    answer = _lcs_table(a, b)
    check(answer == _lcs_search(a, b), "table and subsequence search disagree")
    aligned = sum(x == y for x, y in zip(a, b))
    cands = [
        (_longest_common_substring(a, b), "dp-lcs-contiguous"),
        (min(len(a), len(b)), "dp-lcs-shorter-length"),
        (aligned, "dp-lcs-aligned-matches"),
        (answer + 1, "dp-arithmetic-slip"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v >= 2])
    enough(dist, 3)
    stem = phrase(
        rng,
        (
            "Two DNA strings are {a} and {b}. A common subsequence uses letters that appear in both strings in the same order, but not "
            "necessarily next to each other. What is the length of a longest common subsequence?",
            "What is the length of the longest common subsequence of {a} and {b}? (A subsequence keeps the order of the letters but may skip letters.)",
        ),
        a=a,
        b=b,
    )
    working = [
        "Let L(i, j) be the answer for the first i letters of one string and the first j of the other. If the letters match, L(i, j) = L(i - 1, j - 1) + 1; otherwise L(i, j) = max(L(i - 1, j), L(i, j - 1)).",
        f"The corner of the table gives {answer}; the longest common run of consecutive letters is only {_longest_common_substring(a, b)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _hard_reuse(rng):
    items, cap = draw_until(
        rng,
        lambda r: _draw_items(r, 5, 11, 17),
        lambda g: _knap_unbounded(*g) - _knap_subsets(*g)[0] >= 3 and sum(w for w, _ in g[0]) > g[1],
        what="knapsack items where reuse is worth several dollars more",
    )
    once = _knap_table(items, cap)[len(items)][cap]
    check(once == _knap_subsets(items, cap)[0], "table and subset search disagree")
    many = _knap_unbounded(items, cap)
    answer = many - once
    cands = [
        (many, "dp-difference-gives-one-value"),
        (once, "dp-difference-gives-one-value"),
        (answer + 1, "dp-arithmetic-slip"),
        (answer - 1, "dp-arithmetic-slip"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v and v >= 2])
    enough(dist, 3)
    stem = (
        f"A bag holds at most {cap} kg and there are {len(items)} kinds of item: {_item_text(items)}. "
        "Find the greatest total value when each kind can be taken at most once, and the greatest total value when any kind can be taken "
        "as many times as you like. How many more dollars is the unlimited-copies value than the once-only value?"
    )
    working = [
        f"With each item at most once, the table gives {once}.",
        f"With unlimited copies, f(w) = the best of f(w - weight) + value over the items that fit, which gives {many}. The difference is {many} - {once} = {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


_EASY = (_easy_stairs, _easy_coins, _easy_route)
_MEDIUM = (_medium_knapsack, _medium_coins, _medium_route, _medium_stairs)
_HARD = (_hard_ways, _hard_lcs, _hard_reuse)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7a_dp_table(rng, tier="easy"):
    """Dynamic-programming tables worked by hand; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------------


_EXAM_OPENINGS = (
    "A bag holds at most {cap} kg. There are four items, each available once: {items}.",
    "A bag can carry at most {cap} kg in total. Four items are on offer, and each can be taken once: {items}.",
    "You are packing a bag that must weigh at most {cap} kg. Each of four items may be packed once: {items}.",
)


@generator(EXAM_TOPIC, level=20, presets={"exam": dict(tier="exam")})
def exd_x_w7a_dp_table(rng, tier="exam"):
    """Exam chain on one 0/1 knapsack: the entry for two items, the best value, the weight chosen, the value of one more kg."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        items, cap = draw_until(
            rng,
            lambda r: _draw_items(r, 4, 8, 11),
            lambda g: _knap_ok(*g, require_unique=True, both=False),
            what="four items with one optimal selection",
        )
        f = _knap_table(items, cap)
        best, count, mask = _knap_subsets(items, cap)
        check(count == 1 and f[4][cap] == best, "table and subset search disagree")
        weight = sum(items[i][0] for i in range(4) if mask >> i & 1)
        two = f[2][cap]
        nxt = _knap_table(items, cap + 1)[4][cap + 1]
        gain = nxt - best
        if not (two < best and gain >= 2 and weight <= cap):
            raise TooFewDistractors("a degenerate knapsack draw")
        greedy_w = sum(items[i][0] for i in range(4))
        a_dist = num_dist(
            two,
            [
                (best, "dp-table-wrong-row"),
                (f[1][cap], "dp-table-wrong-row"),
                (f[3][cap], "dp-table-wrong-row"),
                (two + 1, "dp-arithmetic-slip"),
            ],
            3,
        )
        b_dist = num_dist(
            best,
            [
                (_knap_greedy(items, cap, _ratio_key), "dp-knapsack-greedy-ratio"),
                (_knap_greedy(items, cap, _value_key), "dp-knapsack-greedy-value"),
                (_knap_unbounded(items, cap), "dp-knapsack-reuse-items"),
                (sum(v for _, v in items), "dp-knapsack-ignores-capacity"),
            ],
            3,
        )
        c_dist = num_dist(
            weight,
            [
                (cap, "dp-weight-assumed-full"),
                (greedy_w, "dp-weight-of-all-items"),
                (weight + 1, "dp-arithmetic-slip"),
                (weight - 1, "dp-arithmetic-slip"),
            ],
            3,
        )
        d_dist = num_dist(
            gain,
            [
                (nxt, "dp-difference-gives-one-value"),
                (best, "dp-difference-gives-one-value"),
                (gain + 1, "dp-arithmetic-slip"),
                (gain + 2, "dp-arithmetic-slip"),
            ],
            3,
        )
        for d in (a_dist, b_dist, c_dist, d_dist):
            enough(d, 2)
        return items, cap, two, best, weight, nxt, gain, a_dist, b_dist, c_dist, d_dist

    items, cap, two, best, weight, nxt, gain, a_dist, b_dist, c_dist, d_dist = retry_draw(rng, build)
    # The opening is picked from the items, not drawn, so the draw order (and every question's content) is unchanged.
    opening = _EXAM_OPENINGS[sum(w + v for w, v in items) % len(_EXAM_OPENINGS)].format(
        cap=cap, items=_item_text(items)
    )
    stem = (
        f"{opening} "
        "Let f(i, w) be the greatest total value using only the first i items (P first) with a weight limit of w kg."
    )
    return dict(
        text=stem,
        parts=[
            part(
                f"Find f(2, {cap}), in dollars.",
                "number",
                num_fields(two),
                1,
                [f"Using only items P and Q with a {cap} kg limit, the best value is {two}."],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                f"Find f(4, {cap}), the greatest total value that fits in the bag, in dollars.",
                "number",
                num_fields(best),
                2,
                [f"Filling the table to f(4, {cap}) gives {best}."],
                b_dist,
                sensible_range=(0, None),
            ),
            part(
                "What is the total weight, in kg, of the one selection of items that achieves this value?",
                "number",
                num_fields(weight),
                1,
                [f"Trace back through the table to find the chosen items; their weights total {weight} kg."],
                c_dist,
                sensible_range=(0, None),
            ),
            part(
                f"How many more dollars would the best value be if the limit were {cap + 1} kg instead of {cap} kg?",
                "number",
                num_fields(gain),
                2,
                [f"f(4, {cap + 1}) = {nxt}, so the gain is {nxt} - {best} = {gain}."],
                d_dist,
                follow_through=f"{nxt} - part_b",
                sensible_range=(0, None),
            ),
        ],
    )
