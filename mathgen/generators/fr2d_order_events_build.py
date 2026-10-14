"""Generator for `f10-order-events-independent-dependent`: order chance events, then sort pairs of events.

Three tiers, all answered with an ordered list of event letters:

* easy: order four events about one bag of counters (same total, so the counts decide), least likely first.
* medium: order four events drawn from different equipment (a bag, a die, a spinner, a coin, a set of cards, a
  class), so the denominators differ and no count can be compared on its own. One event may be impossible or certain.
* hard: four pairs of events, each described by the way the chance is generated; list the letters of the pairs whose
  events are independent, in alphabetical order. The pairs come from fixed mechanisms (with or without
  replacing, separate devices, one device with two linked events), so each has one defensible answer.

Every key is re-derived a second way (cross-multiplying to a common denominator for the orderings, and for pairs by
checking P(first and second) against the product of the two chances, computed exactly from the counts).
"""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, reversed_and_adjacent_swap

from ._fr2d_kit import Short, need, redraw

_LETTERS = "ABCDEFG"
_COLOURS = ["red", "blue", "green", "yellow", "white", "black", "orange"]


def _list_d(items, tag):
    return Distractor(", ".join(items), tag, list(items))


# -- events with a known chance -------------------------------------------------------------------------------


def _ev_bag(rng):
    col = rng.choice(_COLOURS)
    n = rng.randint(8, 20)
    c = rng.randint(1, n - 1)
    verb = "is" if c == 1 else "are"
    return f"drawing a {col} counter from a bag of {n} counters, {c} of which {verb} {col}", c, n


_DIE = (
    ("rolling an even number on a fair six-sided die", 3),
    ("rolling a number greater than 4 on a fair six-sided die", 2),
    ("rolling a 6 on a fair six-sided die", 1),
    ("rolling a number less than 5 on a fair six-sided die", 4),
    ("rolling a number that is not 1 on a fair six-sided die", 5),
    ("rolling a multiple of 3 on a fair six-sided die", 2),
)


def _ev_die(rng):
    text, c = rng.choice(_DIE)
    return text, c, 6


def _ev_spinner(rng):
    col = rng.choice(_COLOURS)
    n = rng.randint(4, 10)
    c = rng.randint(1, n - 1)
    sections = "section" if c == 1 else "sections"
    return f"spinning a spinner with {n} equal sections and stopping on one of its {c} {col} {sections}", c, n


def _ev_coin(rng):
    return "tossing a fair coin and getting tails", 1, 2


_CARDS = (
    ("a multiple of 5", 4),
    ("a multiple of 4", 5),
    ("greater than 15", 5),
    ("less than 4", 3),
    ("an even number", 10),
    ("a prime number", 8),
)


def _ev_cards(rng):
    cond, c = rng.choice(_CARDS)
    return f"choosing a card at random from cards numbered 1 to 20 and it is {cond}", c, 20


def _ev_class(rng):
    n = rng.randint(18, 30)
    c = rng.randint(3, n - 3)
    return f"choosing a student at random from a class of {n}, {c} of whom walk to school, and the student walks", c, n


def _ev_certain(rng):
    return "rolling a number less than 7 on a fair six-sided die", 6, 6


def _ev_impossible(rng):
    return "rolling a 7 on a fair six-sided die", 0, 6


_MAKERS = [_ev_bag, _ev_die, _ev_spinner, _ev_coin, _ev_cards, _ev_class]


def _stem_events(letters, events):
    return "; ".join(f"{L}: {text}" for L, (text, _c, _n) in zip(letters, events))


def _ordered(letters, probs):
    """Letters from least to most likely, by cross-multiplying over a common denominator."""
    common = lcm(*[p.denominator for p in probs])
    scaled = [int(p * common) for p in probs]
    return [L for _s, L in sorted(zip(scaled, letters))]


def _ordering_options(answer, letters, events, probs):
    """Wrong orderings: greatest first, a swap in the middle, and by favourable count alone."""
    rev, swap = reversed_and_adjacent_swap(answer)
    by_count = [L for _c, L in sorted(zip([e[1] for e in events], letters))]
    return [
        _list_d(rev, "fr2d-orders-greatest-first"),
        _list_d(swap, "fr2d-order-slip"),
        _list_d(by_count, "fr2d-orders-by-favourable-count-only"),
    ]


def _finish(rng, stem, answer, options, working):
    options = [o for o in options if o.text != ", ".join(answer)]
    seen, kept = set(), []
    for o in options:
        if o.text not in seen:
            seen.add(o.text)
            kept.append(o)
    need(kept, n=2)
    text = ", ".join(answer)
    return dict(
        text=stem,
        answer_kind="ordered_list",
        answer=list(answer),
        answer_text=text,
        answer_latex=latex_answer(text),
        working=working,
        distractors=kept,
    )


def _spread_ok(probs):
    ps = sorted(probs)
    return all(b - a >= Fraction(1, 20) for a, b in zip(ps, ps[1:]))


def _easy(rng):
    k = 4
    n = rng.randint(12, 20)
    for _ in attempts(2000, "four different counts"):
        counts = [rng.randint(1, 7) for _ in range(k)]
        if len(set(counts)) == k and sum(counts) <= n:
            break
    rest = n - sum(counts)
    cols = rng.sample(_COLOURS, k + (1 if rest else 0))
    parts = [f"{c} {col}" for c, col in zip(counts, cols)]
    if rest:
        parts.append(f"{rest} {cols[k]}")
    bag = ", ".join(parts[:-1]) + " and " + parts[-1]
    letters = list(_LETTERS[:k])
    order = list(range(k))
    rng.shuffle(order)
    events = [(f"the counter is {cols[i]}", counts[i], n) for i in order]
    probs = [Fraction(counts[i], n) for i in order]
    answer = _ordered(letters, probs)
    check(answer == [letters[j] for j in sorted(range(k), key=lambda j: counts[order[j]])], "ordering differs")
    stem = (
        f"A bag holds {bag}. One counter is taken without looking. Events: {_stem_events(letters, events)}. "
        "List the events in order from least likely to most likely, separated by commas."
    )
    options = _ordering_options(answer, letters, events, probs)
    working = [f"The bag has {n} counters, so the event with the most counters of its colour is the most likely."] + [
        f"{L}: {events[letters.index(L)][1]} out of {n}" for L in answer
    ]
    return _finish(rng, stem, answer, options, working)


def _medium(rng):
    k = 4
    for _ in attempts(3000, "four events with clearly different chances"):
        makers = rng.sample(_MAKERS, k)
        events = [m(rng) for m in makers]
        if rng.random() < 0.3:
            events[rng.randrange(k)] = rng.choice([_ev_certain, _ev_impossible])(rng)
        probs = [Fraction(c, n) for _t, c, n in events]
        if len({n for _t, _c, n in events}) < 3 or not _spread_ok(probs):
            continue
        letters = list(_LETTERS[:k])
        answer = _ordered(letters, probs)
        # a learner who compares favourable counts only must land on a different order
        if answer == [L for _c, L in sorted(zip([e[1] for e in events], letters))]:
            continue
        break
    stem = (
        f"Here are four events. {_stem_events(letters, events)}. "
        "List the events in order from least likely to most likely, separated by commas."
    )
    options = _ordering_options(answer, letters, events, probs)
    working = ["Write each chance as a fraction of the whole, then compare them over a common denominator."] + [
        f"{L}: {events[letters.index(L)][1]}/{events[letters.index(L)][2]}" for L in answer
    ]
    return _finish(rng, stem, answer, options, working)


# -- pairs of events: independent or dependent -------------------------------------------------------------------
# Each template is a function of rng giving (text, independent, p_first, p_second_given_first, trap), where the
# chances are exact, so independence is checked as P(second | first) == P(second) == the second event's overall chance.


def _pair_with_replacing(rng):
    col = rng.choice([c for c in _COLOURS if c != "black"])
    r, b = rng.randint(2, 6), rng.randint(2, 6)
    n = r + b
    text = (
        f"Ben draws a counter from a bag of {r} {col} and {b} black counters, looks at it and puts it back, then "
        f"draws again. First: the first counter is {col}. Second: the second counter is {col}"
    )
    return text, Fraction(r, n), Fraction(r, n), Fraction(r, n), "fr2d-treats-repeat-as-dependent"


def _pair_without_replacing(rng):
    col = rng.choice([c for c in _COLOURS if c != "black"])
    r, b = rng.randint(2, 6), rng.randint(2, 6)
    n = r + b
    text = (
        f"Ben draws a counter from a bag of {r} {col} and {b} black counters and keeps it, then draws a second "
        f"counter. First: the first counter is {col}. Second: the second counter is {col}"
    )
    return text, Fraction(r, n), Fraction(r - 1, n - 1), Fraction(r, n), "fr2d-ignores-no-replacement"


def _pair_two_coins(rng):
    return (
        "A fair coin is tossed twice. First: the first toss is heads. Second: the second toss is heads",
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
        "fr2d-treats-repeat-as-dependent",
    )


def _pair_die_and_coin(rng):
    return (
        "A fair die is rolled and a fair coin is tossed. First: the die shows a 6. Second: the coin lands tails",
        Fraction(1, 6),
        Fraction(1, 2),
        Fraction(1, 2),
        "fr2d-treats-repeat-as-dependent",
    )


def _pair_two_spinners(rng):
    n1, n2 = rng.randint(4, 8), rng.randint(3, 6)
    return (
        f"Two spinners are spun, one with {n1} equal sections and one with {n2} equal sections. First: the "
        f"first spinner stops on blue (one section). Second: the second spinner stops on blue (one section)",
        Fraction(1, n1),
        Fraction(1, n2),
        Fraction(1, n2),
        "fr2d-treats-repeat-as-dependent",
    )


def _pair_raffle(rng):
    n = rng.randint(20, 60)
    return (
        f"{n} raffle tickets are in a barrel and two tickets are drawn, one after the other, and not put back. "
        "First: Mia's ticket is drawn first. Second: Mia's ticket is drawn second",
        Fraction(1, n),
        Fraction(0),
        Fraction(1, n),
        "fr2d-ignores-no-replacement",
    )


def _pair_die_linked(rng):
    return (
        "A fair six-sided die is rolled once. First: the die shows an even number. Second: the die shows a 2",
        Fraction(1, 2),
        Fraction(1, 3),
        Fraction(1, 6),
        "fr2d-treats-linked-events-as-independent",
    )


def _pair_number_linked(rng):
    return (
        "A number is chosen at random from 1 to 10. First: the number is odd. Second: the number is a multiple of 3",
        Fraction(1, 2),
        Fraction(2, 5),
        Fraction(3, 10),
        "fr2d-treats-linked-events-as-independent",
    )


def _pair_cards(rng):
    return (
        "Two cards are dealt from a pack and not put back. First: the first card is a heart. Second: the second "
        "card is a heart",
        Fraction(13, 52),
        Fraction(12, 51),
        Fraction(13, 52),
        "fr2d-ignores-no-replacement",
    )


_PAIR_MAKERS = (
    _pair_with_replacing,
    _pair_without_replacing,
    _pair_two_coins,
    _pair_die_and_coin,
    _pair_two_spinners,
    _pair_raffle,
    _pair_die_linked,
    _pair_number_linked,
    _pair_cards,
)


def _hard(rng):
    k = 4
    for _ in attempts(500, "four pairs with a mix of independent and dependent"):
        makers = rng.sample(_PAIR_MAKERS, k)
        pairs = [m(rng) for m in makers]
        flags = [p[3] == p[2] for p in pairs]  # P(second | first) == P(second)
        if 1 <= sum(flags) <= 3 and len({p[0] for p in pairs}) == k:
            break
    letters = list(_LETTERS[:k])
    # Second route: independence as P(first and second) == P(first) * P(second).
    both = [p[1] * p[2] for p in pairs]
    flags2 = [b == p[1] * p[3] for b, p in zip(both, pairs)]
    check(flags == flags2, "the two independence tests disagree")
    answer = [L for L, f in zip(letters, flags) if f]
    wrong_in = [L for L, f in zip(letters, flags) if not f]
    stem = (
        "Each pair below describes two events. "
        + " ".join(f"Pair {L}: {p[0]}." for L, p in zip(letters, pairs))
        + " Which pairs are made of independent events? List their letters in alphabetical order, separated by commas."
    )
    options = [_list_d(wrong_in, "fr2d-swaps-independent-dependent")]
    for L, p, f in zip(letters, pairs, flags):
        if f:
            options.append(_list_d([x for x in answer if x != L], p[4]) if len(answer) > 1 else None)
        else:
            options.append(_list_d(sorted(answer + [L]), p[4]))
    options = [o for o in options if o is not None]
    working = [
        f"Pair {L}: P(second) = {rational(p[2])}, P(second after first) = {rational(p[3])}: "
        + ("the same, so independent." if f else "different, so dependent.")
        for L, p, f in zip(letters, pairs, flags)
    ]
    return _finish(rng, stem, answer, options, working)


def rational(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


@generator(
    "fr2d_order_events",
    level=4,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@redraw
def fr2d_order_events(rng, tier: str = "easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
