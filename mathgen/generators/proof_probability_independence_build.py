"""Generates `proof_independence` items: establish and use the formula P(A and B) = P(A) P(B) for independent
events.

Every question defines the events in words, names what the learner may use (the definition of independent
events, the addition rule, complements) and asks for a complete argument. The proof is marked against a rubric,
so `answer` is a marking dict (see `_free_proof_kit`), not a value.

Ladder:

- easy: show two concretely described events are independent, by counting outcomes of one die or two coins.
- medium: use the formula once: independence passes to a complement, or a given union probability gives
  independence, or a card-draw pair is independent.
- hard: general results with a short chain (both complements, mutually exclusive events are not independent),
  and a two-dice pair whose independence is not obvious.
- exam: an equivalence or a conditional statement, an "exactly one" formula, three events that are pairwise but not
  mutually independent, and the certain or impossible event.

Every statement asked is true: the concrete ones are checked by exhaustive enumeration with exact fractions, the
general ones by an exact check on random probability values.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from ..core import check, generator
from ._free_proof_kit import Fm, Prompt, TAILS, attempts, finish, frac_plain, frac_tex, marking

AB = Fm("P(A\\cap B)", "P(A∩B)")
PA = Fm("P(A)", "P(A)")
PB = Fm("P(B)", "P(B)")
DEF = "Events are independent when the probability that both happen is the product of their probabilities, P(A∩B) = P(A)P(B). "


def _tail(rng) -> str:
    return rng.choice(TAILS)


def _fr(num: int, den: int) -> Fm:
    return Fm(frac_tex(num, den), frac_plain(num, den))


def _prob(event: set, space: list) -> Fraction:
    return Fraction(len(event), len(space))


# ---------------------------------------------------------------------------
# Event libraries (predicate, phrase) on one die of n sides and on two dice.
# ---------------------------------------------------------------------------
def _die_events(n: int) -> list:
    ev = [
        (lambda v: v % 2 == 0, "the number is even"),
        (lambda v: v % 2 == 1, "the number is odd"),
        (lambda v: v % 3 == 0, "the number is a multiple of 3"),
    ]
    for k in range(2, n):
        ev.append((lambda v, k=k: v <= k, f"the number is at most {k}"))
        ev.append((lambda v, k=k: v > k, f"the number is greater than {k}"))
    return ev


def _two_dice_events() -> list:
    return [
        (lambda a, b: a % 2 == 0, "the first die shows an even number"),
        (lambda a, b: a <= 3, "the first die shows 3 or less"),
        (lambda a, b: a > 4, "the first die shows more than 4"),
        (lambda a, b: b % 2 == 1, "the second die shows an odd number"),
        (lambda a, b: b >= 5, "the second die shows 5 or 6"),
        (lambda a, b: (a + b) % 2 == 0, "the sum of the two numbers is even"),
        (lambda a, b: a == b, "the two numbers are equal"),
        (lambda a, b: a + b == 7, "the sum of the two numbers is 7"),
        (lambda a, b: a + b >= 10, "the sum of the two numbers is at least 10"),
        (lambda a, b: a > b, "the first number is larger than the second"),
    ]


def _independent(ea: set, eb: set, space: list) -> bool:
    return _prob(ea & eb, space) == _prob(ea, space) * _prob(eb, space)


def _nontrivial(e: set, space: list) -> bool:
    return 0 < len(e) < len(space)


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _ind_one_die(rng) -> dict:
    n = rng.choice((6, 6, 8, 12))
    space = list(range(1, n + 1))
    events = _die_events(n)
    for _ in attempts(500, "an independent pair of die events"):
        (fa, pa_), (fb, pb_) = rng.sample(events, 2)
        ea, eb = {v for v in space if fa(v)}, {v for v in space if fb(v)}
        if _nontrivial(ea, space) and _nontrivial(eb, space) and ea != eb and _independent(ea, eb, space):
            break
    inter = ea & eb
    check(_prob(inter, space) == _prob(ea, space) * _prob(eb, space), "the events are independent")
    na, nb, ni = len(ea), len(eb), len(inter)
    p = Prompt()
    p.t(f"A fair {n}-sided die, numbered 1 to {n}, is rolled once. Event A is that {pa_}; event B is that {pb_}. ")
    p.t(DEF).t("Prove that A and B are independent. ").t(_tail(rng))
    ref = [
        f"The {n} outcomes are equally likely. A has {na} outcomes, so P(A) = {frac_plain(na, n)}.",
        f"B has {nb} outcomes, so P(B) = {frac_plain(nb, n)}.",
        f"A and B both happen for {ni} outcome(s), so P(A∩B) = {frac_plain(ni, n)}.",
        f"P(A)P(B) = {frac_plain(na, n)} × {frac_plain(nb, n)} = {frac_plain(na * nb, n * n)} = {frac_plain(ni, n)}.",
        "So P(A∩B) = P(A)P(B), and A and B are independent by the definition.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"finds P(A), P(B) and P(A∩B) correctly: {frac_plain(na, n)}, {frac_plain(nb, n)}, {frac_plain(ni, n)}",
            ),
            ("logical_completeness", "compares P(A∩B) with the product P(A)P(B) and concludes"),
            ("definitions_assumptions", "uses the definition of independence and equally likely outcomes"),
            ("communication", "lists the outcomes in A, in B and in both"),
        ],
        traps=[
            ("wrong_intersection", "does it count the outcomes in both A and B, not the outcomes in A or B?"),
            ("assumes_independence", "does it assume independence instead of showing P(A∩B) = P(A)P(B)?"),
        ],
    )
    return finish(p, "A and B are independent", mark)


def _ind_two_coins(rng) -> dict:
    k = rng.choice((2, 3))
    space = list(product("HT", repeat=k))
    pos = rng.randrange(k)
    qos = (pos + rng.randint(1, k - 1)) % k
    ordinal = ("first", "second", "third")
    ea = {o for o in space if o[pos] == "H"}
    eb = {o for o in space if o[qos] == "T"}
    check(_independent(ea, eb, space), "different tosses are independent")
    n = len(space)
    p = Prompt()
    p.t(
        f"A fair coin is tossed {k} times. Event A is that the {ordinal[pos]} toss is a head; event B is that the {ordinal[qos]} toss is a tail. "
    )
    p.t(DEF).t("List the equally likely outcomes and prove that A and B are independent. ").t(_tail(rng))
    ref = [
        f"There are 2^{k} = {n} equally likely outcomes (each toss is H or T).",
        f"A has {len(ea)} outcomes, so P(A) = {frac_plain(len(ea), n)}. B has {len(eb)} outcomes, so P(B) = {frac_plain(len(eb), n)}.",
        f"A and B both happen for {len(ea & eb)} outcome(s) (the {ordinal[pos]} toss is H and the {ordinal[qos]} toss is T), so P(A∩B) = {frac_plain(len(ea & eb), n)}.",
        f"P(A)P(B) = {frac_plain(len(ea), n)} × {frac_plain(len(eb), n)} = {frac_plain(len(ea & eb), n)}, equal to P(A∩B), so A and B are independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"counts {len(ea)}, {len(eb)} and {len(ea & eb)} outcomes out of {n}"),
            (
                "logical_completeness",
                "compares P(A∩B) with P(A)P(B) by calculation, not by saying the tosses 'do not affect each other'",
            ),
            ("definitions_assumptions", "states that the outcomes are equally likely"),
            ("communication", "lists or counts the sample space clearly"),
        ],
        traps=[
            (
                "independence_by_assertion",
                "does it claim independence because the tosses are separate, without checking the formula?",
            ),
            ("miscounted_space", f"does it use {n} outcomes in total?"),
        ],
    )
    return finish(p, "A and B are independent", mark)


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _ind_complement(rng) -> dict:
    p = Prompt()
    p.t("Events A and B are independent. ").t(DEF).t(
        "Prove that A and B' are independent, where B' is the event that B does not happen. You may use P(B') = 1 - P(B) and P(A) = P(A∩B) + P(A∩B'). "
    ).t(_tail(rng))
    ref = [
        "A happens either with B or with B', and these cannot both happen, so P(A) = P(A∩B) + P(A∩B').",
        "So P(A∩B') = P(A) - P(A∩B).",
        "A and B are independent, so P(A∩B) = P(A)P(B). Hence P(A∩B') = P(A) - P(A)P(B) = P(A)(1 - P(B)).",
        "Since 1 - P(B) = P(B'), P(A∩B') = P(A)P(B'). So A and B' are independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "reaches P(A∩B') = P(A)(1 - P(B)) = P(A)P(B')"),
            ("logical_completeness", "justifies P(A∩B') = P(A) - P(A∩B) before using independence"),
            ("definitions_assumptions", "uses independence of A and B exactly once, as P(A∩B) = P(A)P(B)"),
            ("communication", "states what is being proved and ends with the formula for A and B'"),
        ],
        traps=[
            (
                "treats_complement_as_exclusive_of_a",
                "does it write P(A∩B') = P(A) - P(B) or similar, losing the factor P(A)?",
            ),
            ("assumes_what_is_to_be_shown", "does it start from P(A∩B') = P(A)P(B')?"),
        ],
    )
    return finish(p, "A and B' are independent", mark)


def _ind_from_union(rng) -> dict:
    pa = Fraction(rng.randint(1, 4), 5)
    den = rng.choice((3, 4, 6))
    pb = Fraction(rng.randint(1, den - 1), den)
    pu = pa + pb - pa * pb
    check(0 < pa < 1 and 0 < pb < 1 and max(pa, pb) <= pu <= 1, "valid probabilities")
    p = Prompt()
    p.t(
        f"For events A and B, P(A) = {frac_plain(pa.numerator, pa.denominator)}, P(B) = {frac_plain(pb.numerator, pb.denominator)} and P(A∪B) = {frac_plain(pu.numerator, pu.denominator)}. "
    )
    p.t(DEF).t("Prove that A and B are independent. You may use P(A∪B) = P(A) + P(B) - P(A∩B). ").t(_tail(rng))
    inter = pa + pb - pu
    check(inter == pa * pb, "the union value makes A and B independent")
    ref = [
        "By the addition rule, P(A∩B) = P(A) + P(B) - P(A∪B).",
        f"So P(A∩B) = {frac_plain(pa.numerator, pa.denominator)} + {frac_plain(pb.numerator, pb.denominator)} - {frac_plain(pu.numerator, pu.denominator)} = {frac_plain(inter.numerator, inter.denominator)}.",
        f"P(A)P(B) = {frac_plain(pa.numerator, pa.denominator)} × {frac_plain(pb.numerator, pb.denominator)} = {frac_plain(inter.numerator, inter.denominator)}.",
        "P(A∩B) = P(A)P(B), so A and B are independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds P(A∩B) = {frac_plain(inter.numerator, inter.denominator)} using the addition rule"),
            ("logical_completeness", "compares P(A∩B) with P(A)P(B) and states the conclusion"),
            ("definitions_assumptions", "rearranges the addition rule for P(A∩B), without assuming independence"),
            ("communication", "shows each fraction calculation"),
        ],
        traps=[
            ("uses_union_as_product", "does it treat P(A∪B) as P(A)P(B)?"),
            ("assumes_independence_first", "does it assume independence to find P(A∩B) and then 'confirm' it?"),
        ],
    )
    return finish(p, "A and B are independent", mark)


def _ind_cards(rng) -> dict:
    suits = ("heart", "diamond", "club", "spade")
    suit = rng.choice(suits)
    rank_phrase, rank_n = rng.choice(
        (
            ("a king", 4),
            ("an ace", 4),
            ("a face card (jack, queen or king)", 12),
            ("a card numbered 2 to 6", 20),
            ("a card numbered 2 to 10", 36),
        )
    )
    # the pack is 52 equally likely cards; rank classes are spread evenly over the four suits
    per_suit = rank_n // 4
    check(rank_n % 4 == 0, "each suit holds the same number of the chosen cards")
    n_suit, n_rank, n_both = 13, rank_n, per_suit
    check(Fraction(n_both, 52) == Fraction(n_suit, 52) * Fraction(n_rank, 52), "the card events are independent")
    p = Prompt()
    p.t(
        f"One card is drawn at random from a standard pack of 52 cards (13 cards in each of 4 suits). Event A is that the card is a {suit}; event B is that the card is {rank_phrase}. "
    )
    p.t(DEF).t("Prove that A and B are independent. ").t(_tail(rng))
    ref = [
        f"All 52 cards are equally likely. P(A) = {frac_plain(13, 52)} = {frac_plain(1, 4)}, since 13 cards are of that suit.",
        f"P(B) = {frac_plain(rank_n, 52)}, since {rank_n} of the 52 cards are in B.",
        f"The cards in both A and B are the {per_suit} such card(s) in that suit, so P(A∩B) = {frac_plain(per_suit, 52)}.",
        f"P(A)P(B) = {frac_plain(1, 4)} × {frac_plain(rank_n, 52)} = {frac_plain(rank_n, 208)} = {frac_plain(per_suit, 52)}.",
        "So P(A∩B) = P(A)P(B) and A and B are independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds P(A) = 1/4, P(B) = {frac_plain(rank_n, 52)}, P(A∩B) = {frac_plain(per_suit, 52)}"),
            ("logical_completeness", "compares P(A∩B) with the product and concludes"),
            ("definitions_assumptions", "states that each card is equally likely"),
            ("communication", "shows how many cards are in A, in B and in both"),
        ],
        traps=[
            ("counts_union", f"does it count the cards that are {suit} or {rank_phrase} instead of both?"),
            ("suit_rank_assumed_unrelated", "does it assume the suit and the rank are unrelated without calculating?"),
        ],
    )
    return finish(p, "A and B are independent", mark)


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _ind_both_complements(rng) -> dict:
    p = Prompt()
    p.t("Events A and B are independent. ").t(DEF).t(
        "Prove that A' and B' are independent, where A' and B' are the events that A and B do not happen. You may use the addition rule P(A∪B) = P(A) + P(B) - P(A∩B) and P((A∪B)') = 1 - P(A∪B). "
    ).t(_tail(rng))
    ref = [
        "A' and B' both happen exactly when neither A nor B happens, so P(A'∩B') = P((A∪B)') = 1 - P(A∪B).",
        "By the addition rule, 1 - P(A∪B) = 1 - P(A) - P(B) + P(A∩B).",
        "A and B are independent, so P(A∩B) = P(A)P(B). Hence P(A'∩B') = 1 - P(A) - P(B) + P(A)P(B).",
        "This factorises: (1 - P(A))(1 - P(B)) = 1 - P(A) - P(B) + P(A)P(B).",
        "So P(A'∩B') = P(A')P(B'), and A' and B' are independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "reaches P(A'∩B') = (1 - P(A))(1 - P(B))"),
            ("logical_completeness", "justifies A'∩B' = (A∪B)' before using the addition rule"),
            ("definitions_assumptions", "uses the independence of A and B only to replace P(A∩B)"),
            ("communication", "factorises the expression and names the conclusion"),
        ],
        traps=[
            ("de_morgan_wrong", "does it take A'∩B' to be (A∩B)' instead of (A∪B)'?"),
            ("sign_error_in_expansion", "does the expansion of (1 - P(A))(1 - P(B)) have the correct signs?"),
        ],
    )
    return finish(p, "A' and B' are independent", mark)


def _ind_exclusive_not_independent(rng) -> dict:
    p = Prompt()
    p.t(
        "Events A and B are mutually exclusive (they cannot both happen), and each has a probability greater than 0. "
    ).t(DEF).t("Prove that A and B are not independent. ").t(_tail(rng))
    ref = [
        "Mutually exclusive events cannot both happen, so P(A∩B) = 0.",
        "P(A) > 0 and P(B) > 0, so the product P(A)P(B) > 0.",
        "So P(A∩B) = 0 ≠ P(A)P(B). The independence formula fails, so A and B are not independent.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "states P(A∩B) = 0 and P(A)P(B) > 0"),
            ("logical_completeness", "concludes that the equality P(A∩B) = P(A)P(B) fails"),
            ("definitions_assumptions", "uses both probabilities being greater than 0, which the question gives"),
            ("communication", "ends by stating that A and B are not independent"),
        ],
        traps=[
            (
                "confuses_exclusive_with_independent",
                "does it claim that exclusive events are independent because they do not affect each other?",
            ),
            ("ignores_positivity", "does it explain why the product is not 0?"),
        ],
    )
    return finish(p, "A and B are not independent", mark)


def _ind_two_dice(rng) -> dict:
    space = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    events = _two_dice_events()
    for _ in attempts(1000, "an independent pair of two-dice events"):
        (fa, pa_), (fb, pb_) = rng.sample(events, 2)
        ea = {o for o in space if fa(*o)}
        eb = {o for o in space if fb(*o)}
        if _nontrivial(ea, space) and _nontrivial(eb, space) and ea != eb and ea & eb and _independent(ea, eb, space):
            break
    na, nb, ni = len(ea), len(eb), len(ea & eb)
    check(Fraction(ni, 36) == Fraction(na, 36) * Fraction(nb, 36), "the events are independent")
    p = Prompt()
    p.t(f"Two fair six-sided dice are rolled. Event A is that {pa_}; event B is that {pb_}. ")
    p.t(DEF).t("By counting equally likely outcomes, prove that A and B are independent. ").t(_tail(rng))
    ref = [
        "The 36 ordered pairs (first die, second die) are equally likely.",
        f"A contains {na} pairs, so P(A) = {frac_plain(na, 36)}. B contains {nb} pairs, so P(B) = {frac_plain(nb, 36)}.",
        f"{ni} pairs are in both A and B, so P(A∩B) = {frac_plain(ni, 36)}.",
        f"P(A)P(B) = {frac_plain(na, 36)} × {frac_plain(nb, 36)} = {frac_plain(na * nb, 1296)} = {frac_plain(ni, 36)}.",
        "So P(A∩B) = P(A)P(B), and A and B are independent, even though the sum depends on both dice, because the counts match.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"counts {na}, {nb} and {ni} pairs out of 36"),
            (
                "logical_completeness",
                "compares P(A∩B) with P(A)P(B); does not rely on 'the dice do not affect each other'",
            ),
            ("definitions_assumptions", "uses the 36 equally likely ordered pairs"),
            ("communication", "lists or describes the pairs in A, B and A∩B"),
        ],
        traps=[
            (
                "independence_from_physical_separation",
                "does it argue only that the dice are separate, although one event involves both dice?",
            ),
            ("counts_unordered_pairs", "does it count 21 unordered pairs instead of 36 ordered pairs?"),
        ],
    )
    return finish(p, "A and B are independent", mark)


def _ind_scaled(rng) -> dict:
    """If A, B independent, then P(A or B) in terms of the two probabilities; a use of the formula."""
    pa = Fraction(rng.randint(1, 5), rng.choice((6, 7, 8)))
    den = rng.choice((6, 7, 9))
    pb = Fraction(rng.randint(1, 5), den)
    check(0 < pa < 1 and 0 < pb < 1, "valid probabilities")
    neither = (1 - pa) * (1 - pb)
    at_least = 1 - neither
    check(at_least == pa + pb - pa * pb, "two ways to find P(A or B) agree")
    p = Prompt()
    p.t(
        f"Events A and B are independent, with P(A) = {frac_plain(pa.numerator, pa.denominator)} and P(B) = {frac_plain(pb.numerator, pb.denominator)}. "
    )
    p.t(DEF).t(
        f"Prove that the probability that at least one of A and B happens is {frac_plain(at_least.numerator, at_least.denominator)}. You may use that if A and B are independent then A' and B' are independent. "
    ).t(_tail(rng))
    ref = [
        "At least one of A and B happens exactly when it is not true that neither happens: P(A∪B) = 1 - P(A'∩B').",
        "A and B are independent, so A' and B' are independent (the complement of an independent event is independent), giving P(A'∩B') = P(A')P(B').",
        f"P(A') = 1 - {frac_plain(pa.numerator, pa.denominator)} = {frac_plain((1 - pa).numerator, (1 - pa).denominator)} and P(B') = 1 - {frac_plain(pb.numerator, pb.denominator)} = {frac_plain((1 - pb).numerator, (1 - pb).denominator)}.",
        f"P(A'∩B') = {frac_plain((1 - pa).numerator, (1 - pa).denominator)} × {frac_plain((1 - pb).numerator, (1 - pb).denominator)} = {frac_plain(neither.numerator, neither.denominator)}.",
        f"So P(A∪B) = 1 - {frac_plain(neither.numerator, neither.denominator)} = {frac_plain(at_least.numerator, at_least.denominator)}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches P(A∪B) = {frac_plain(at_least.numerator, at_least.denominator)}"),
            (
                "logical_completeness",
                "justifies the use of the product formula for the complements, or uses P(A∪B) = P(A) + P(B) - P(A)P(B) with P(A∩B) from independence",
            ),
            ("definitions_assumptions", "states that independence is used to replace P(A∩B) (or P(A'∩B'))"),
            ("communication", "shows each fraction step"),
        ],
        traps=[
            ("adds_probabilities", "does it add P(A) and P(B) without subtracting the overlap?"),
            ("complement_of_wrong_event", "does it compute 1 - P(A∩B) instead of 1 - P(A'∩B')?"),
        ],
    )
    return finish(p, f"P(A∪B) = {frac_plain(at_least.numerator, at_least.denominator)}", mark)


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _ind_iff(rng) -> dict:
    p = Prompt()
    p.t("A and B are events. ").t(DEF).t(
        "Prove that A and B are independent if and only if P(A∩B') = P(A)P(B'), where B' is the event that B does not happen. You may use P(B') = 1 - P(B) and P(A) = P(A∩B) + P(A∩B'). "
    ).t(_tail(rng))
    ref = [
        "From P(A) = P(A∩B) + P(A∩B') we get P(A∩B') = P(A) - P(A∩B) for any events A and B.",
        "Forward direction: suppose A and B are independent. Then P(A∩B) = P(A)P(B), so P(A∩B') = P(A) - P(A)P(B) = P(A)(1 - P(B)) = P(A)P(B').",
        "Reverse direction: suppose P(A∩B') = P(A)P(B'). Then P(A) - P(A∩B) = P(A)(1 - P(B)) = P(A) - P(A)P(B).",
        "Subtracting P(A) from both sides and multiplying by -1 gives P(A∩B) = P(A)P(B), so A and B are independent.",
        "Both directions are proved, so A and B are independent if and only if P(A∩B') = P(A)P(B').",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "derives P(A∩B') = P(A) - P(A∩B) and uses it in both directions"),
            ("logical_completeness", "proves both directions, not only 'independent implies the formula for B''"),
            ("definitions_assumptions", "states the assumption at the start of each direction"),
            ("communication", "labels the two directions and the conclusion"),
        ],
        traps=[
            ("one_direction_only", "does it prove only one of the two implications?"),
            ("circular_reverse", "in the reverse direction, does it assume independence to get the result?"),
        ],
    )
    return finish(p, "A and B are independent if and only if P(A∩B') = P(A)P(B')", mark)


def _ind_conditional(rng) -> dict:
    p = Prompt()
    p.t("A and B are independent events with 0 < P(B) < 1, and B' is the event that B does not happen. ").t(DEF).t(
        "The conditional probability of A given B is P(A|B) = P(A∩B) ÷ P(B). Prove that P(A|B) = P(A|B'). You may use P(A∩B') = P(A)P(B') for independent A and B. "
    ).t(_tail(rng))
    ref = [
        "Since A and B are independent, P(A∩B) = P(A)P(B). Because P(B) > 0, P(A|B) = P(A)P(B) ÷ P(B) = P(A).",
        "Since 0 < P(B) < 1, P(B') = 1 - P(B) > 0, so P(A|B') is defined.",
        "P(A∩B') = P(A)P(B') (given, from independence), so P(A|B') = P(A)P(B') ÷ P(B') = P(A).",
        "So P(A|B) = P(A) = P(A|B').",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "shows P(A|B) = P(A) and P(A|B') = P(A)"),
            ("logical_completeness", "divides by P(B) and P(B') and explains why neither is 0"),
            ("definitions_assumptions", "uses 0 < P(B) < 1 so both conditional probabilities exist"),
            ("communication", "concludes with the equality of the two conditional probabilities"),
        ],
        traps=[
            ("cancels_without_nonzero", "does it divide by P(B') without saying P(B') > 0?"),
            ("swaps_conditioning", "does it use P(B|A) instead of P(A|B)?"),
        ],
    )
    return finish(p, "P(A|B) = P(A|B')", mark)


def _ind_exactly_one(rng) -> dict:
    pa = Fraction(rng.randint(1, 4), rng.choice((5, 6)))
    pb = Fraction(rng.randint(1, 4), rng.choice((5, 7, 8)))
    check(0 < pa < 1 and 0 < pb < 1, "valid probabilities")
    exactly = pa * (1 - pb) + pb * (1 - pa)
    check(exactly == pa + pb - 2 * pa * pb, "the two forms agree")
    p = Prompt()
    p.t(
        f"A and B are independent events with P(A) = {frac_plain(pa.numerator, pa.denominator)} and P(B) = {frac_plain(pb.numerator, pb.denominator)}. "
    )
    p.t(DEF).t(
        f"Prove that the probability that exactly one of A and B happens is {frac_plain(exactly.numerator, exactly.denominator)}. You may use that A and B', and A' and B, are independent. "
    ).t(_tail(rng))
    ref = [
        "Exactly one happens when A happens without B, or B happens without A. These two cases are mutually exclusive.",
        "So P(exactly one) = P(A∩B') + P(A'∩B).",
        "By independence, P(A∩B') = P(A)P(B') and P(A'∩B) = P(A')P(B).",
        f"P(A)P(B') = {frac_plain(pa.numerator, pa.denominator)} × {frac_plain((1 - pb).numerator, (1 - pb).denominator)} = {frac_plain((pa * (1 - pb)).numerator, (pa * (1 - pb)).denominator)}.",
        f"P(A')P(B) = {frac_plain((1 - pa).numerator, (1 - pa).denominator)} × {frac_plain(pb.numerator, pb.denominator)} = {frac_plain((pb * (1 - pa)).numerator, (pb * (1 - pa)).denominator)}.",
        f"The sum is {frac_plain(exactly.numerator, exactly.denominator)}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"obtains the two products and the total {frac_plain(exactly.numerator, exactly.denominator)}",
            ),
            ("logical_completeness", "explains why the two cases are exclusive, so the probabilities add"),
            ("definitions_assumptions", "uses independence of A with B' and of A' with B"),
            ("communication", "shows each product before adding"),
        ],
        traps=[
            ("includes_both", "does it include P(A∩B) in the total, which is 'at least one'?"),
            ("adds_without_exclusivity", "does it add the two cases without saying they are mutually exclusive?"),
        ],
    )
    return finish(p, f"P(exactly one of A and B) = {frac_plain(exactly.numerator, exactly.denominator)}", mark)


def _ind_pairwise_not_mutual(rng) -> dict:
    space = list(product("HT", repeat=2))
    ea = {o for o in space if o[0] == "H"}
    eb = {o for o in space if o[1] == "H"}
    ec = {o for o in space if o[0] == o[1]}
    quarter = Fraction(1, 4)
    for x, y in ((ea, eb), (ea, ec), (eb, ec)):
        check(_prob(x & y, space) == _prob(x, space) * _prob(y, space), "pairwise independent")
    check(
        _prob(ea & eb & ec, space) != _prob(ea, space) * _prob(eb, space) * _prob(ec, space), "not mutually independent"
    )
    check(_prob(ea & eb & ec, space) == quarter, "A and B and C is HH")
    p = Prompt()
    p.t(
        "A fair coin is tossed twice. Event A is that the first toss is a head; event B is that the second toss is a head; event C is that both tosses give the same result. "
    )
    p.t(DEF).t(
        "Prove that A and C are independent, that B and C are independent, and that P(A∩B∩C) ≠ P(A)P(B)P(C). "
    ).t(_tail(rng))
    ref = [
        "The four outcomes HH, HT, TH, TT are equally likely. A = {HH, HT}, B = {HH, TH}, C = {HH, TT}.",
        "P(A) = P(B) = P(C) = 1/2.",
        "A∩C = {HH}, so P(A∩C) = 1/4 = (1/2)(1/2) = P(A)P(C). B∩C = {HH}, so P(B∩C) = 1/4 = P(B)P(C). So the pairs are independent.",
        "A∩B∩C = {HH}, so P(A∩B∩C) = 1/4.",
        "But P(A)P(B)P(C) = (1/2)(1/2)(1/2) = 1/8, and 1/4 ≠ 1/8.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "lists the events as sets of outcomes and finds the probabilities 1/2, 1/4, 1/4 and 1/4"),
            ("logical_completeness", "proves both independence statements and the failure of the three-way product"),
            ("definitions_assumptions", "uses the four equally likely outcomes"),
            ("communication", "treats the three required statements separately"),
        ],
        traps=[
            ("only_checks_triple", "does it prove the three-way inequality but not the two independence statements?"),
            ("wrong_triple_probability", "does it find P(A∩B∩C) = 1/4 (the outcome HH only)?"),
        ],
    )
    return finish(p, "A and C, B and C are independent but P(A∩B∩C) ≠ P(A)P(B)P(C)", mark)


def _ind_certain_or_impossible(rng) -> dict:
    which = rng.choice(("certain", "impossible"))
    if which == "certain":
        claim_ev = "P(A) = 1"
        ref = [
            "P(A) = 1 means A' has probability 0, so P(A'∩B) = 0 (A'∩B is part of A').",
            "B happens either with A or without A, so P(B) = P(A∩B) + P(A'∩B), and P(A'∩B) ≤ P(A') = 0. So P(A∩B) = P(B).",
            "P(A)P(B) = 1 × P(B) = P(B).",
            "So P(A∩B) = P(A)P(B): A and B are independent, for any event B.",
        ]
        step = "shows P(A∩B) = P(B) using P(A'∩B) ≤ P(A') = 0"
        extra = ("sets_intersection_to_one", "does it write P(A∩B) = 1 instead of P(B)?")
    else:
        claim_ev = "P(A) = 0"
        ref = [
            "A∩B is part of A, so 0 ≤ P(A∩B) ≤ P(A) = 0, and P(A∩B) = 0.",
            "P(A)P(B) = 0 × P(B) = 0.",
            "So P(A∩B) = 0 = P(A)P(B): A and B are independent, for any event B.",
        ]
        step = "shows P(A∩B) = 0 because A∩B is contained in A"
        extra = ("confuses_with_exclusive", "does it use mutual exclusivity instead of the fact A∩B is contained in A?")
    p = Prompt()
    p.t(f"A is an event with {claim_ev}, and B is any event. ").t(DEF).t(
        "Prove that A and B are independent. Your proof must work for every event B. "
    ).t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", step),
            ("logical_completeness", "works for every event B, not only one example"),
            ("definitions_assumptions", "states the definition P(A∩B) = P(A)P(B) as the target"),
            ("communication", "ends with the conclusion"),
        ],
        traps=[
            ("example_only", "does it check only the given probability of B instead of every B?"),
            extra,
        ],
    )
    return finish(p, f"if {claim_ev}, A and B are independent", mark)


FAMILIES = {
    "easy": (_ind_one_die, _ind_two_coins),
    "medium": (_ind_complement, _ind_from_union, _ind_cards),
    "hard": (_ind_both_complements, _ind_exclusive_not_independent, _ind_two_dice, _ind_scaled),
    "exam": (_ind_iff, _ind_conditional, _ind_exactly_one, _ind_pairwise_not_mutual, _ind_certain_or_impossible),
}

_PRESETS = {t: dict(variant=t) for t in ("easy", "medium", "hard", "exam")}


@generator("proof_independence", level=14, presets=_PRESETS)
def proof_independence(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-p-a-cap`. Rubric-assessed (`answer: rubric`): every question
    returned carries a per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
