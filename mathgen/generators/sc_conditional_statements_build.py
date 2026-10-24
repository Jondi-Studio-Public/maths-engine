"""Converse, contrapositive, negation and equivalence of "if P then Q" on integer and set statements, answered with
typed whole numbers: `sc_conditional_counts` (easy, medium, hard) and `sc_conditional_counts_exam`.

Skill: ``implication-converse-contrapositive-counts``. A statement S is "if P(n) then Q(n)" about a whole number n from 1
to N. The questions never ask a learner to pick a wording from a list. They ask how many values of n (or which smallest
value) make a related statement fail, and that number is only right if the learner knows what each transformation
fails on:

* S fails at n exactly when P(n) is true and Q(n) is false (a counterexample);
* the contrapositive "if not Q then not P" fails at exactly the same n, so it is equivalent to S;
* the converse "if Q then P" fails when Q is true and P is false, a different set of n;
* the inverse "if not P then not Q" fails exactly where the converse does;
* the negation of S is true exactly where S fails;
* "P if and only if Q" fails wherever exactly one of P and Q is true.

* easy: S is true. The smallest counterexample to its converse, or the number of counterexamples to the converse.
  Or S is false and the smallest counterexample to S is asked for.
* medium: S is false, with counterexamples in both directions. Counterexamples to the contrapositive, how many n make the
  negation true, how many n make S true (an implication with a false hypothesis is true), how many make the converse true.
* hard: how many n make the inverse true, make "P if and only if Q" true, or make the negation of the converse true; or
  the same on explicit finite sets A and B ("x is in A", "x is in B").
* exam (``sc_conditional_counts_exam``): (a) the number of n with P, (b) the number with Q, (c) the number of
  counterexamples to the contrapositive, (d) the number of counterexamples to the converse, found from (a), (b) and (c)
  (|B| - |A| + (c)), and (e) the number of n for which "P if and only if Q" is false, found from (c) and (d).

Every count is computed twice: from the sets of n for which P and Q hold, and by evaluating the named logical form at
every n (the truth table of the statement, its converse, contrapositive and so on) and counting the failures.
"""

from __future__ import annotations

import math

from ..core import attempts, check, generator
from ._r3b_kit import Redraw, int_dist, int_fields, part, pick_ints, retry

# ------------------------------------------------------------------------------------------------ predicates


class Pred:
    """A statement about n: its words, the words for 'not', and the test."""

    def __init__(self, text, neg, test):
        self.text, self.neg, self.test = text, neg, test


def _multiple(k):
    return Pred(f"n is a multiple of {k}", f"n is not a multiple of {k}", lambda n, k=k: n % k == 0)


def _remainder(k, r):
    return Pred(
        f"n leaves a remainder of {r} when divided by {k}",
        f"n does not leave a remainder of {r} when divided by {k}",
        lambda n, k=k, r=r: n % k == r,
    )


def _remainders(k, rs):
    ws = " or ".join(str(r) for r in rs)
    others = " or ".join(str(r) for r in range(k) if r not in rs)
    return Pred(
        f"n leaves a remainder of {ws} when divided by {k}",
        f"n leaves a remainder of {others} when divided by {k}",
        lambda n, k=k, rs=tuple(rs): n % k in rs,
    )


def _factor(m):
    return Pred(f"n is a factor of {m}", f"n is not a factor of {m}", lambda n, m=m: m % n == 0)


def _is_square(n):
    return math.isqrt(n) ** 2 == n


def _is_cube(n):
    return round(n ** (1 / 3)) ** 3 == n


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))


_ODD = Pred("n is odd", "n is even", lambda n: n % 2 == 1)
_EVEN = Pred("n is even", "n is odd", lambda n: n % 2 == 0)
_SQUARE = Pred("n is a perfect square", "n is not a perfect square", _is_square)
_CUBE = Pred("n is a perfect cube", "n is not a perfect cube", _is_cube)
_PRIME = Pred("n is prime", "n is not prime", _is_prime)
_ENDS5 = Pred("n ends in the digit 5", "n does not end in the digit 5", lambda n: n % 10 == 5)
_ENDS0 = Pred("n ends in the digit 0", "n does not end in the digit 0", lambda n: n % 10 == 0)


def _above(c):
    return Pred(f"n is greater than {c}", f"n is at most {c}", lambda n, c=c: n > c)


def _below(c):
    return Pred(f"n is less than {c}", f"n is at least {c}", lambda n, c=c: n < c)


def _hypotheses(rng):
    """Clauses that can open an implication about whole numbers."""
    return [_multiple(k) for k in (2, 3, 4, 5, 6, 8, 9, 10, 12, 15)] + [
        _ODD,
        _EVEN,
        _PRIME,
        _SQUARE,
        _CUBE,
        _ENDS5,
        _ENDS0,
        _factor(rng.choice((48, 60, 72, 90))),
    ]


def _conclusions(rng):
    """Clauses a statement can conclude with: divisibility and remainder facts."""
    return [_multiple(k) for k in (2, 3, 4, 5, 6, 9, 10)] + [
        _ODD,
        _EVEN,
        _ENDS5,
        _remainders(4, (0, 1)),
        _remainders(3, (0, 1)),
        _remainders(5, (0, 1, 4)),
        _remainder(3, 1),
        _remainder(4, 1),
    ]


_UNIVERSES = (30, 36, 40, 45, 48, 50, 60, 64, 72)


def _sets(p, q, universe):
    ns = range(1, universe + 1)
    return {n for n in ns if p.test(n)}, {n for n in ns if q.test(n)}


# ------------------------------------------------------------------------------------------------ logical forms
# Each form is a function of (P(n), Q(n)) returning whether that statement is true at n. This is the route that does
# not use the "counterexample" reasoning, so a count from the sets is checked against a plain truth table.

_FORMS = {
    "statement": lambda p, q: (not p) or q,
    "converse": lambda p, q: (not q) or p,
    "inverse": lambda p, q: p or (not q),
    "contrapositive": lambda p, q: q or (not p),
    "negation": lambda p, q: p and (not q),
    "iff": lambda p, q: p == q,
    "negation of the converse": lambda p, q: q and (not p),
}


def _true_count(form, a, b, universe):
    return sum(1 for n in range(1, universe + 1) if _FORMS[form](n in a, n in b))


def _counts(a, b, universe):
    """Both routes at once: from the sets, and from the truth table. Returns the set counts."""
    only_a, only_b = len(a - b), len(b - a)
    check(_true_count("statement", a, b, universe) == universe - only_a, "statement count differs")
    check(_true_count("converse", a, b, universe) == universe - only_b, "converse count differs")
    check(_true_count("contrapositive", a, b, universe) == universe - only_a, "contrapositive count differs")
    check(_true_count("inverse", a, b, universe) == universe - only_b, "inverse count differs")
    check(_true_count("negation", a, b, universe) == only_a, "negation count differs")
    check(_true_count("iff", a, b, universe) == universe - only_a - only_b, "iff count differs")
    check(_true_count("negation of the converse", a, b, universe) == only_b, "negation of converse count differs")
    return only_a, only_b


# ------------------------------------------------------------------------------------------------ drawing


def _draw_pair(rng, relation):
    """(P, Q, A, B, N) for a statement of the stated relation, with every count big enough to be interesting.

    `relation`: "true" (A inside B, B has values outside A), "false" (counterexamples both ways), "false-one" (A not inside
    B, B inside A is not required but there are counterexamples to S)."""
    for _ in attempts(4000, f"a {relation} statement"):
        p, q = rng.choice(_hypotheses(rng)), rng.choice(_conclusions(rng))
        if p.text == q.text:
            continue
        universe = rng.choice(_UNIVERSES)
        a, b = _sets(p, q, universe)
        only_a, only_b = len(a - b), len(b - a)
        if len(a) < 4 or len(b) < 4 or len(a) > universe - 4 or len(b) > universe - 4:
            continue
        if relation == "true" and only_a == 0 and only_b >= 3:
            return p, q, a, b, universe
        if relation == "false" and only_a >= 2 and only_b >= 2 and len(a & b) >= 2:
            return p, q, a, b, universe
        if relation == "false-one" and only_a >= 2:
            return p, q, a, b, universe
    raise Redraw("no pair")


def _nested_remainders(rng):
    """(P, Q, A, B, N) for "if n leaves remainder r when divided by k, then n leaves remainder r when divided by j"
    with j a factor of k and r below j. The first value of Q is r, which satisfies P too, so the smallest counterexample
    to the converse is r + j, never the first value of Q."""
    for _ in attempts(200, "nested remainders"):
        j = rng.choice((2, 3, 4, 5, 6))
        k = j * rng.choice((2, 3))
        r = rng.randint(1, j - 1)
        universe = rng.choice(_UNIVERSES)
        p, q = _remainder(k, r), _remainder(j, r)
        a, b = _sets(p, q, universe)
        if a <= b and len(b - a) >= 3 and len(a) >= 4:
            return p, q, a, b, universe
    raise Redraw("no nested pair")


def _nth(values, i):
    """The i-th smallest of the values (counting from 0), or None when there are not that many."""
    vals = sorted(values)
    return vals[i] if len(vals) > i else None


def _vals(k):
    """ "1 value" or "5 values"."""
    return f"{k} value" if k == 1 else f"{k} values"


def _c(pred, which):
    """A clause quoted for use inside a sentence."""
    return f"“{getattr(pred, which)}”"


def _statement(p, q):
    return f"“if {p.text}, then {q.text}”"


def _intro(p, q, universe):
    return (
        f"Let n be a whole number from 1 to {universe}. Consider the statement S: {_statement(p, q)}, "
        "with hypothesis P and conclusion Q."
    )


def _numbers(values, limit=14):
    vals = sorted(values)
    shown = ", ".join(str(v) for v in vals[:limit])
    return shown + (", ..." if len(vals) > limit else "")


def _dist(rng, key, cands):
    """Wrong counts: a count is a positive whole number no bigger than the number of values in play."""
    return pick_ints(key, [(v, t) for v, t in cands if v is not None and v > 0], count=3, rng=rng)


def _dist_spread(rng, key, cands):
    """Three wrong counts, how many of them are below the key drawn at random, so the key has no fixed rank."""
    wrong = {}
    for v, t in cands:
        if v is not None and v > 0 and v != key:
            wrong.setdefault(v, t)
    if len(wrong) < 3:
        raise Redraw("fewer than three distinct wrong counts")
    below = [(v, t) for v, t in wrong.items() if v < key]
    above = [(v, t) for v, t in wrong.items() if v > key]
    k = rng.randint(max(0, 3 - len(above)), min(3, len(below)))
    chosen = rng.sample(below, k) + rng.sample(above, 3 - k)
    rng.shuffle(chosen)
    return [int_dist(v, t) for v, t in chosen]


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("smallest-converse", "count-converse", "smallest-statement"))
    if variant == "smallest-statement":
        for _ in attempts(200, "a counterexample that is not the first value satisfying P"):
            p, q, a, b, universe = _draw_pair(rng, "false-one")
            ans = min(a - b)
            if ans != min(a):  # else "the first n that satisfies P" would answer without testing Q
                break
        text = _intro(p, q, universe) + " S is false. Find the smallest value of n that is a counterexample to S."
        cands = [
            (min(a), "sc-counterexample-hypothesis-only"),
            (min(set(range(1, universe + 1)) - b), "sc-counterexample-conclusion-false"),
            (min(b - a) if b - a else None, "sc-counterexample-of-converse"),
            (min(a & b) if a & b else None, "sc-counterexample-both-true"),
        ]
        working = [
            f"A counterexample to S has the hypothesis {_c(p, 'text')} true and the conclusion {_c(q, 'text')} false.",
            f"The values of n that satisfy the hypothesis start {_numbers(a, 6)}; the first of them that fails the conclusion is {ans}.",
        ]
        return dict(text=text, working=working, distractors=_dist(rng, ans, cands), **int_fields(ans))
    if variant == "smallest-converse" and rng.random() < 0.7:
        p, q, a, b, universe = _nested_remainders(rng)
    else:
        for _ in attempts(200, "a counterexample that is not the first value satisfying Q"):
            p, q, a, b, universe = _draw_pair(rng, "true")
            # The converse's hypothesis is Q; for "smallest" the first n satisfying Q is usually not the answer. A few
            # draws that are (1 in 5) stay in: the ordinary pairs where it is not are few and fix the key to 3 values.
            if variant != "smallest-converse" or min(b - a) != min(b) or rng.random() < 0.2:
                break
    only_a, only_b = _counts(a, b, universe)
    check(only_a == 0, "S should be true")
    converse = f"“if {q.text}, then {p.text}”"
    if variant == "smallest-converse":
        ans = min(b - a)
        text = (
            _intro(p, q, universe)
            + f" S is true. Its converse is {converse}, which is false. Find the smallest value of n that is a counterexample to the converse."
        )
        cands = [
            (min(a), "sc-counterexample-hypothesis-only"),
            (min(b), "sc-counterexample-conclusion-true-only"),
            (min(set(range(1, universe + 1)) - a), "sc-counterexample-conclusion-false"),
            (min(a & b) if a & b else None, "sc-counterexample-both-true"),
            # the same slips made on a later value, so the key is not always the largest option
            (_nth(b, 1), "sc-counterexample-conclusion-true-only"),
            (_nth(b, 2), "sc-counterexample-conclusion-true-only"),
            (_nth(a, 1), "sc-counterexample-hypothesis-only"),
            (_nth(a & b, 1), "sc-counterexample-both-true"),
            (_nth(b, 3), "sc-counterexample-conclusion-true-only"),
            (_nth(a, 2), "sc-counterexample-hypothesis-only"),
            (_nth(a & b, 2), "sc-counterexample-both-true"),
        ]
        working = [
            f"A counterexample to the converse has its hypothesis ({_c(q, 'text')}) true and its conclusion ({_c(p, 'text')}) false.",
            f"The values of n that satisfy {_c(q, 'text')} start {_numbers(b, 6)}; the first of them that fails {_c(p, 'text')} is {ans}.",
        ]
        return dict(text=text, working=working, distractors=_dist_spread(rng, ans, cands), **int_fields(ans))
    ans = only_b
    text = (
        _intro(p, q, universe)
        + f" S is true. Its converse is {converse}. How many of the values of n are counterexamples to the converse?"
    )
    cands = [
        (len(b), "sc-counterexample-hypothesis-only"),
        (len(a), "sc-counts-the-wrong-set"),
        (universe - len(a), "sc-counterexample-conclusion-false"),
        (len(b & a), "sc-counterexample-both-true"),
        (universe - only_b, "sc-counts-true-instances-not-failures"),
    ]
    working = [
        f"A counterexample to the converse has {_c(q, 'text')} true and {_c(p, 'text')} false: a value counted by Q but not by P.",
        f"{_vals(len(b))} satisfy Q, and all {_vals(len(a))} that satisfy P are among them, so the number of counterexamples is {len(b)} - {len(a)} = {ans}.",
    ]
    return dict(text=text, working=working, distractors=_dist(rng, ans, cands), **int_fields(ans))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("contrapositive", "negation", "statement-true", "converse-true"))
    p, q, a, b, universe = _draw_pair(rng, "false")
    only_a, only_b = _counts(a, b, universe)
    base = _intro(p, q, universe)
    conv = f"“if {q.text}, then {p.text}”"
    if variant == "contrapositive":
        ans = only_a
        text = base + " How many of these values of n are counterexamples to the contrapositive of S?"
        cands = [
            (only_b, "sc-contrapositive-counts-converse-failures"),
            (universe - len(b), "sc-counts-hypothesis-of-contrapositive"),
            (len(a), "sc-counterexample-hypothesis-only"),
            (len(a & b), "sc-counterexample-both-true"),
            (universe - only_a, "sc-counts-true-instances-not-failures"),
        ]
        working = [
            f"The contrapositive of S is “if {q.neg}, then {p.neg}”.",
            "It is equivalent to S, so it fails at exactly the same values: those with P true and Q false.",
            f"{_vals(len(a))} satisfy P and {len(a & b)} of them also satisfy Q, so {len(a)} - {len(a & b)} = {ans} fail.",
        ]
    elif variant == "negation":
        ans = only_a
        text = base + " For how many of these values of n is the negation of S true?"
        cands = [
            (universe - len(a & b), "sc-negation-as-if-p-then-not-q"),
            (only_b, "sc-negation-as-converse-failure"),
            (len(a & b), "sc-negation-counts-both-true"),
            (universe - only_a, "sc-counts-true-instances-not-failures"),
            (len(a), "sc-counterexample-hypothesis-only"),
        ]
        working = [
            "The negation of “if P then Q” is “P and not Q”, which is true exactly at the counterexamples to S.",
            f"{_vals(len(a))} satisfy P; {len(a & b)} of them also satisfy Q; the other {ans} satisfy P and not Q.",
        ]
    elif variant == "statement-true":
        ans = universe - only_a
        text = (
            base
            + " For how many of these values of n is the statement S true? (For a given n, S is true when it does not fail.)"
        )
        cands = [
            (len(a & b), "sc-vacuous-truth-ignored"),
            (only_a, "sc-counts-true-instances-not-failures"),
            (universe - only_b, "sc-converse-counts-statement-failures"),
            (len(b), "sc-counts-the-wrong-set"),
            (universe - len(a), "sc-vacuous-truth-only"),
        ]
        working = [
            "For a given n, S is false only when P is true and Q is false. When P is false the statement is true (vacuously).",
            f"S fails at {_vals(only_a)} ({len(a)} with P, of which {len(a & b)} also have Q), so it is true at {universe} - {only_a} = {ans}.",
        ]
    else:
        ans = universe - only_b
        text = base + f" The converse of S is {conv}. For how many of these values of n is the converse true?"
        cands = [
            (len(a & b), "sc-vacuous-truth-ignored"),
            (only_b, "sc-counts-true-instances-not-failures"),
            (universe - only_a, "sc-converse-counts-statement-failures"),
            (len(a), "sc-counts-the-wrong-set"),
            (universe - len(b), "sc-vacuous-truth-only"),
        ]
        working = [
            "The converse is false only when Q is true and P is false; otherwise it is true at n.",
            f"{_vals(len(b))} have Q and {len(a & b)} of them also have P, so it fails at {_vals(only_b)} and is true at {universe} - {only_b} = {ans}.",
        ]
    return dict(text=text, working=working, distractors=_dist(rng, ans, cands), **int_fields(ans))


# ------------------------------------------------------------------------------------------------ hard


def _set_text(values):
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def _hard_sets(rng):
    """Explicit finite sets A and B in a universe 1..N; statements are about "x is in A"."""
    for _ in attempts(500, "explicit sets"):
        universe = rng.choice((12, 14, 15, 16, 18, 20))
        pop = list(range(1, universe + 1))
        a = set(rng.sample(pop, rng.randint(5, 9)))
        b = set(rng.sample(pop, rng.randint(6, 11)))
        only_a, only_b = len(a - b), len(b - a)
        if only_a >= 2 and only_b >= 2 and a & b and len(a | b) < universe:
            break
    only_a, only_b = _counts(a, b, universe)
    variant = rng.choice(("inverse", "iff", "negation-converse"))
    stem = (
        f"The universal set is U = {{1, 2, ..., {universe}}}. A = {_set_text(a)} and B = {_set_text(b)}. "
        "For a value x in U, let P be “x is in A” and Q be “x is in B”."
    )
    if variant == "inverse":
        ans = universe - only_b
        text = (
            stem + " The inverse of “if P then Q” is “if not P, then not Q”. For how many x in U is the inverse true?"
        )
        cands = [
            (universe - only_a, "sc-converse-counts-statement-failures"),
            (only_b, "sc-counts-true-instances-not-failures"),
            (len(a - b) + len(b - a), "sc-biconditional-counts-both-directions"),
            (universe - len(a | b), "sc-vacuous-truth-only"),
        ]
        working = [
            "The inverse fails when ‘not P’ is true and ‘not Q’ is false: x is not in A but is in B.",
            f"Those are the {_vals(only_b)} of B outside A, so it is true at {universe} - {only_b} = {ans}.",
        ]
    elif variant == "iff":
        ans = universe - only_a - only_b
        text = stem + " For how many x in U is “P if and only if Q” true?"
        cands = [
            (len(a & b), "sc-biconditional-counts-both-true"),
            (universe - only_a, "sc-biconditional-counts-one-direction"),
            (universe - only_b, "sc-biconditional-counts-one-direction"),
            (only_a + only_b, "sc-counts-true-instances-not-failures"),
        ]
        working = [
            "“P if and only if Q” is true at x when P and Q are both true or both false.",
            f"Both true: {_vals(len(a & b))}. Both false: {_vals(universe - len(a | b))}. Total {ans}.",
        ]
    else:
        ans = only_b
        text = stem + " For how many x in U is the negation of the converse of “if P then Q” true?"
        cands = [
            (only_a, "sc-negation-of-converse-as-negation"),
            (universe - only_b, "sc-counts-true-instances-not-failures"),
            (len(b), "sc-counterexample-hypothesis-only"),
            (len(a & b), "sc-negation-counts-both-true"),
        ]
        working = [
            "The converse is “if Q then P”; its negation is “Q and not P”.",
            f"These are the values in B that are not in A: {len(b)} - {len(a & b)} = {ans}.",
        ]
    return dict(text=text, working=working, distractors=_dist(rng, ans, cands), **int_fields(ans))


def _hard_integers(rng):
    variant = rng.choice(("inverse", "iff", "negation-converse"))
    p, q, a, b, universe = _draw_pair(rng, "false")
    only_a, only_b = _counts(a, b, universe)
    base = _intro(p, q, universe)
    if variant == "inverse":
        ans = universe - only_b
        text = (
            base
            + " The inverse of S is “if not P, then not Q”, where not P and not Q negate the two clauses of S. For how many of these values of n is the inverse true?"
        )
        cands = [
            (universe - only_a, "sc-converse-counts-statement-failures"),
            (only_b, "sc-counts-true-instances-not-failures"),
            (only_a + only_b, "sc-biconditional-counts-both-directions"),
            (universe - len(a | b), "sc-vacuous-truth-only"),
        ]
        working = [
            f"The inverse is “if {p.neg}, then {q.neg}”. It fails where not P is true and not Q is false: P is false but Q is true.",
            f"That is the converse's set of failures: {len(b)} - {len(a & b)} = {_vals(only_b)}. So it is true at {universe} - {only_b} = {ans}.",
        ]
    elif variant == "iff":
        ans = universe - only_a - only_b
        text = base + " For how many of these values of n is “P if and only if Q” true?"
        cands = [
            (len(a & b), "sc-biconditional-counts-both-true"),
            (universe - only_a, "sc-biconditional-counts-one-direction"),
            (universe - only_b, "sc-biconditional-counts-one-direction"),
            (only_a + only_b, "sc-counts-true-instances-not-failures"),
        ]
        working = [
            "“P if and only if Q” is true at n when P and Q are both true or both false.",
            f"Both true: {len(a & b)}. Both false: {universe - len(a | b)}. Total {ans}.",
        ]
    else:
        ans = only_b
        text = base + " For how many of these values of n is the negation of the converse of S true?"
        cands = [
            (only_a, "sc-negation-of-converse-as-negation"),
            (universe - only_b, "sc-counts-true-instances-not-failures"),
            (len(b), "sc-counterexample-hypothesis-only"),
            (len(a & b), "sc-negation-counts-both-true"),
        ]
        working = [
            f"The converse is “if {q.text}, then {p.text}”; its negation is “{q.text} and {p.neg}”.",
            f"{_vals(len(b))} have Q and {len(a & b)} of them also have P, so {only_b} have Q and not P.",
        ]
    return dict(text=text, working=working, distractors=_dist(rng, ans, cands), **int_fields(ans))


def _hard(rng):
    return rng.choice((_hard_integers, _hard_integers, _hard_sets))(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("sc_conditional_counts", level=17, presets=_PRESETS)
@retry
def sc_conditional_counts(rng, tier: str = "easy"):
    """Counts and smallest counterexamples that separate a conditional from its converse, contrapositive and negation."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("sc_conditional_counts_exam", level=17, presets={"exam": {}})
@retry
def sc_conditional_counts_exam(rng):
    """Exam chain: |A|, |B|, counterexamples to the contrapositive, to the converse (from the first three), 'iff' failures."""
    p, q, a, b, universe = _draw_pair(rng, "false")
    only_a, only_b = _counts(a, b, universe)
    na, nb = len(a), len(b)
    check(only_b == nb - na + only_a, "the converse count does not follow from (a), (b), (c)")
    stem = (
        f"Let n be a whole number from 1 to {universe}. Consider the statement S: {_statement(p, q)}. "
        "S is false for some values of n. Let P be the clause after “if” and Q the clause after “then”."
    )
    pa = part(
        f"How many of these values of n satisfy P, that is, {_c(p, 'text')}?",
        f"How many of these values of n satisfy P, that is, {_c(p, 'text')}?",
        1,
        [f"Count the values from 1 to {universe} for which {_c(p, 'text')}: {_numbers(a, 8)} ({na} in all)."],
        int_fields(na),
        _dist(
            rng,
            na,
            [
                (universe - na, "sc-counts-the-wrong-set"),
                (nb, "sc-counts-the-wrong-set"),
                (na - 1, "sc-off-by-one-in-count"),
                (na + 1, "sc-off-by-one-in-count"),
            ],
        ),
        sensible_range=(0, universe + 1),
    )
    pb = part(
        f"How many satisfy Q, that is, {_c(q, 'text')}?",
        f"How many satisfy Q, that is, {_c(q, 'text')}?",
        1,
        [f"Count the values from 1 to {universe} for which {_c(q, 'text')}: {_numbers(b, 8)} ({nb} in all)."],
        int_fields(nb),
        _dist(
            rng,
            nb,
            [
                (universe - nb, "sc-counts-the-wrong-set"),
                (na, "sc-counts-the-wrong-set"),
                (nb - 1, "sc-off-by-one-in-count"),
                (nb + 1, "sc-off-by-one-in-count"),
            ],
        ),
        sensible_range=(0, universe + 1),
    )
    pc = part(
        "How many values of n are counterexamples to the contrapositive of S?",
        "How many values of n are counterexamples to the contrapositive of S?",
        2,
        [
            f"The contrapositive is “if {q.neg}, then {p.neg}”, which is equivalent to S.",
            f"Its counterexamples are the values with P true and Q false: {na} - {len(a & b)} = {only_a}.",
        ],
        int_fields(only_a),
        _dist(
            rng,
            only_a,
            [
                (only_b, "sc-contrapositive-counts-converse-failures"),
                (universe - nb, "sc-counts-hypothesis-of-contrapositive"),
                (na, "sc-counterexample-hypothesis-only"),
                (len(a & b), "sc-counterexample-both-true"),
            ],
        ),
        sensible_range=(0, universe + 1),
    )
    converse = f"“if {q.text}, then {p.text}”"
    pd = part(
        f"The converse of S is {converse}. Use your answers to (a), (b) and (c) to find how many values of n are counterexamples to the converse.",
        f"The converse of S is {converse}. Use your answers to (a), (b) and (c) to find how many values of n are counterexamples to the converse.",
        2,
        [
            f"Counterexamples to the converse have Q true and P false: {nb} - {len(a & b)}.",
            f"The values with both P and Q number {na} - {only_a} = {len(a & b)}, so there are {nb} - {len(a & b)} = {only_b}.",
        ],
        int_fields(only_b),
        _dist(
            rng,
            only_b,
            [
                (only_a, "sc-converse-counts-statement-failures"),
                (nb, "sc-counterexample-hypothesis-only"),
                (universe - na, "sc-counterexample-conclusion-false"),
                (nb - na, "sc-converse-ignores-overlap"),
            ],
        ),
        follow_through="part_b - part_a + part_c",
        sensible_range=(0, universe + 1),
    )
    pe = part(
        "For how many values of n is “P if and only if Q” false? Use your answers to (c) and (d).",
        "For how many values of n is “P if and only if Q” false? Use your answers to (c) and (d).",
        2,
        [
            "“P if and only if Q” is false when exactly one of P and Q is true: P true and Q false, or Q true and P false.",
            f"That is {only_a} + {only_b} = {only_a + only_b}.",
        ],
        int_fields(only_a + only_b),
        _dist(
            rng,
            only_a + only_b,
            [
                (only_a, "sc-biconditional-counts-one-direction"),
                (only_b, "sc-biconditional-counts-one-direction"),
                (universe - only_a - only_b, "sc-counts-true-instances-not-failures"),
                (len(a & b), "sc-biconditional-counts-both-true"),
            ],
        ),
        follow_through="part_c + part_d",
        sensible_range=(0, universe + 1),
    )
    return dict(text=stem, parts=[pa, pb, pc, pd, pe])
