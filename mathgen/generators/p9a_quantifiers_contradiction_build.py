"""Quantifiers ('for all'/'there exists') and proof by contradiction.

- ``quantifier_forall_exists_evaluate`` (senior-use-the-quantifiers-for-all-and-there-exists,
  ACMSM027) -- evaluate or classify a quantified statement over an explicit finite set of
  integers drawn from ``range(2, 100)``. Truth classes are CONSTRUCTED to target (a chosen
  predicate's passing/failing pools are sampled directly), never left to uniform random
  sampling of the domain -- uniform sampling makes 'for all' almost always False and 'there
  exists' almost always True for most of these predicates, which would make a tier's answer
  near-constant by probability alone.

- ``proof_by_contradiction_identify`` (senior-use-proof-by-contradiction, ACMSM025) --
  identify which stated fact about a number contradicts an assumption made for a proof by
  contradiction. The scope is narrow: recognising the contradicting fact, never
  authoring a proof. The question states the claim and the assumption in full and NEVER
  states the resolution -- the word "contradiction"/"contradicts" appears only in the
  setup sentence, never attached to a specific pair of adjectives, and `answer_text` never
  appears verbatim inside `text` (checked directly, every seed and tier, in this module and
  again in the test file).

Both skills' `text`/`answer_latex` deliberately do not go through real KaTeX: `text`/`latex`
are plain prose plus explicit set notation (no fractions, surds or unresolved algebra), and
`answer_text` is a plain English label. `latex = "plain:" + text` for both skills so
the front-end renderer shows exactly what `text` says, with no LaTeX-escaping risk from the braces in a
set literal or the quotes inside an MCQ option label.

`answer_latex` is `latex_answer(answer_text)` for the quantifier skill -- verified safe for its two answer
shapes: `latex_answer("True")` / `latex_answer("False")` return real LaTeX (`\\text{True}` / `\\text{False}`,
semantically correct), and the four hard-tier option strings all contain a quote or an apostrophe and so fall
through to `latex_answer`'s own `plain:` branch. It is NOT used for the contradiction skill's `answer_text`:
`latex_answer("24 is even")` and `latex_answer("5 is positive")` sympify the English word "is" as Python's
identity operator against an auto-symbol, so `24 is Symbol('even')` evaluates to the *boolean* `False` and is
returned as real LaTeX (`\\text{False}`) -- wrong, dangerous content for a TRUE statement about parity. The
contradiction skill's `answer_latex` is built as `"plain:" + answer_text` directly instead, which is the value
`latex_answer` itself would have returned had this sympify collision not intervened (these are always plain
English sentences, never real mathematics).
"""

from __future__ import annotations

import math

from ..core import check, generator, latex_answer, pick_distractors


def _pick_slips(rng, slips, n=3):
    """Up to ``n`` wrong options from ``(applies, text)`` pairs: each is a wrong verdict that a genuine
    misreading of the quantifier produces, offered only when its stated reason can be mistaken for true."""
    pool = [text for applies, text in slips if applies]
    rng.shuffle(pool)
    return pool[:n]


# ===========================================================================
# Skill 1: quantifier_forall_exists_evaluate (ACMSM027)
# ===========================================================================

_PREDICATES = [
    ("is even", lambda n: n % 2 == 0),
    ("is odd", lambda n: n % 2 == 1),
    ("is a multiple of 3", lambda n: n % 3 == 0),
    ("is a multiple of 5", lambda n: n % 5 == 0),
    ("is greater than 20", lambda n: n > 20),
    ("is a perfect square", lambda n: n >= 0 and math.isqrt(n) ** 2 == n),
]

_QUANT_DOMAIN = range(2, 100)

_SELECT_OPTIONS = [
    "'for all' only",
    "'there exists' only",
    "Both 'for all' and 'there exists'",
    "Neither",
]


def _set_text(values) -> str:
    """Explicit set notation, never Python's `[1, 2, 3]` list rendering."""
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def _pad_to_k(base: set[int], k: int, rng) -> list[int]:
    """Pad `base` up to exactly `k` unique values by drawing from the full
    quantifier domain one at a time (resampling on any collision), and return
    the sorted list. Every branch that calls this only needs the *existing*
    members of `base` to guarantee its truth class -- the pool sizes assert
    below make padding safe regardless of what gets added."""
    values = set(base)
    while len(values) < k:
        values.add(rng.choice(_QUANT_DOMAIN))
    return sorted(values)


def _predicate_pool_sizes() -> dict[str, tuple[int, int]]:
    """Diagnostic: (passing, failing) pool sizes for every predicate over
    `_QUANT_DOMAIN`, for a pool-size report."""
    return {
        name: (len([x for x in _QUANT_DOMAIN if fn(x)]), len([x for x in _QUANT_DOMAIN if not fn(x)]))
        for name, fn in _PREDICATES
    }


@generator(
    "quantifier_forall_exists_evaluate",
    level=12,
    presets={
        "easy": dict(variant="exists"),
        "medium": dict(variant="forall"),
        "hard": dict(variant="select_quantifier"),
    },
)
def quantifier_forall_exists_evaluate(rng, variant="exists"):
    """Evaluate ('there exists'/'for all') or classify (hard: which quantifier
    makes the statement true) over an explicit finite set of integers.
    """
    if variant not in ("exists", "forall", "select_quantifier"):
        raise ValueError(f"variant must be 'exists', 'forall' or 'select_quantifier'; got {variant!r}")

    p_name, p_fn = rng.choice(_PREDICATES)
    passing_pool = [x for x in _QUANT_DOMAIN if p_fn(x)]
    failing_pool = [x for x in _QUANT_DOMAIN if not p_fn(x)]
    check(
        len(passing_pool) >= 7 and len(failing_pool) >= 7,
        f"predicate {p_name!r} pool too small: {len(passing_pool)} passing, {len(failing_pool)} failing",
    )

    if variant == "exists":
        k = 5
        target = rng.choice([True, False])
        if target:
            D = _pad_to_k({rng.choice(passing_pool)}, k, rng)
        else:
            D = sorted(rng.sample(failing_pool, k))
        # Independent check: explicit-loop accumulator, a different code path
        # from the set-construction route above.
        found = False
        for n in D:
            found = found or p_fn(n)
        check(found == target, "exists-tier independent check mismatch")

        text = f"There exists n in {_set_text(D)} such that n {p_name}. Is this True or False?"
        satisfying = sum(1 for n in D if p_fn(n))
        if target:
            answer_text = "True, because at least one value in the set satisfies it"
            slips = [
                (len(D) > satisfying, "False, because not every value in the set satisfies it"),  # exists read as all
                (satisfying >= 2, "False, because more than one value in the set satisfies it"),  # exists read as one
                (not p_fn(D[0]), "False, because the first value listed does not satisfy it"),  # only looked at one
                (True, "False, because no value in the set satisfies it"),
            ]
        else:
            answer_text = "False, because no value in the set satisfies it"
            slips = [
                (True, "True, because at least one value in the set does not satisfy it"),  # not-P read as P
                (True, "True, because some whole numbers outside the set satisfy it"),  # strays outside the set
                (True, "True, because at least one value in the set satisfies it"),
            ]
        distractors = _pick_slips(rng, slips)
        working = [
            f"Check n {p_name} for each n in {_set_text(D)}.",
            (
                "At least one value satisfies it, so the existential statement is True."
                if target
                else "No value satisfies it, so the existential statement is False."
            ),
        ]
        answer = target

    elif variant == "forall":
        k = 6
        target = rng.choice([True, False])
        if target:
            D = sorted(rng.sample(passing_pool, k))
        else:
            D = _pad_to_k({rng.choice(failing_pool)}, k, rng)
        # Independent check: explicit-loop accumulator, a different code path
        # from the sample-construction route above.
        ok = True
        for n in D:
            ok = ok and p_fn(n)
        check(ok == target, "forall-tier independent check mismatch")

        text = f"For ALL n in {_set_text(D)}, n {p_name}. Is this True or False?"
        satisfying = sum(1 for n in D if p_fn(n))
        if target:
            answer_text = "True, because every value in the set satisfies it"
            slips = [
                (True, "False, because at least one value in the set fails it"),
                (True, "False, because the statement must hold for every integer, not only those in the set"),
                (True, "False, because checking examples can never prove a universal statement"),
            ]
        else:
            answer_text = "False, because at least one value in the set fails it"
            slips = [
                (True, "True, because every value in the set satisfies it"),
                (satisfying >= 1, "True, because at least one value in the set satisfies it"),  # all read as exists
                (satisfying * 2 > len(D), "True, because most of the values in the set satisfy it"),
                (p_fn(D[0]), "True, because the first value listed satisfies it"),  # only looked at one
                (True, "True, because the statement has not been disproved"),  # burden of proof reversed
            ]
        distractors = _pick_slips(rng, slips)
        working = [
            f"Check n {p_name} for each n in {_set_text(D)}.",
            (
                "Every value satisfies it, so the universal statement is True."
                if target
                else "At least one value fails it, so the universal statement is False."
            ),
        ]
        answer = target

    else:  # select_quantifier
        k = 7
        target_class = rng.choice(["both", "exists_only", "neither"])
        if target_class == "both":
            D = sorted(rng.sample(passing_pool, k))
        elif target_class == "neither":
            D = sorted(rng.sample(failing_pool, k))
        else:  # exists_only
            D = _pad_to_k({rng.choice(passing_pool), rng.choice(failing_pool)}, k, rng)

        # Independent check: recompute BOTH all_true/any_true via the
        # explicit-loop route (not the builtin/set-construction route used
        # above), and confirm the (all_true, any_true) pair matches
        # target_class. "'for all' only" (all_true, any_true) == (True,
        # False) is structurally unreachable: a non-empty D with all_true
        # True always makes any_true True too.
        all_true, any_true = True, False
        for n in D:
            all_true = all_true and p_fn(n)
            any_true = any_true or p_fn(n)
        expected_pair = {"both": (True, True), "exists_only": (False, True), "neither": (False, False)}[target_class]
        check((all_true, any_true) == expected_pair, "select_quantifier independent check mismatch")

        answer_text = {
            "both": _SELECT_OPTIONS[2],
            "exists_only": _SELECT_OPTIONS[1],
            "neither": _SELECT_OPTIONS[3],
        }[target_class]
        check(answer_text != _SELECT_OPTIONS[0], "\"'for all' only\" must never be the correct answer")
        text = (
            f"Which quantifier makes the following statement TRUE for the set {_set_text(D)}: "
            f"'__ n in {_set_text(D)}, n {p_name}'? Choose from for-all, there-exists, both, or neither."
        )
        distractors = pick_distractors(answer_text, [o for o in _SELECT_OPTIONS if o != answer_text])
        working = [
            f"For each n in {_set_text(D)}, check n {p_name}.",
            f"all_true = {all_true}, any_true = {any_true}, so the answer is {answer_text!r}.",
        ]
        answer = target_class

    latex = f"plain:{text}"
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


# ===========================================================================
# Skill 2: proof_by_contradiction_identify (ACMSM025)
# ===========================================================================

_PAIRS = [
    ("even", lambda n: n % 2 == 0, "odd", lambda n: n % 2 == 1, list(range(2, 50))),
    ("positive", lambda n: n > 0, "negative", lambda n: n < 0, [n for n in range(-30, 31) if n != 0]),
    ("a multiple of 3", lambda n: n % 3 == 0, "not a multiple of 3", lambda n: n % 3 != 0, list(range(2, 50))),
    ("a multiple of 4", lambda n: n % 4 == 0, "not a multiple of 4", lambda n: n % 4 != 0, list(range(2, 50))),
    (
        "a perfect square",
        lambda n: n >= 0 and math.isqrt(n) ** 2 == n,
        "not a perfect square",
        lambda n: not (n >= 0 and math.isqrt(n) ** 2 == n),
        list(range(2, 50)),
    ),
]

# Every entry's actual_fn/claimed_fn are complementary and mutually exclusive
# over the entry's own domain -- checked exhaustively ONCE, at import, rather
# than on every generator call (this is a static property of the pool, not of
# any one draw). `a != b` over booleans is exactly XOR: true iff exactly one
# of the two functions holds.
for _actual_adj, _actual_fn, _claimed_adj, _claimed_fn, _pair_domain in _PAIRS:
    for _x in _pair_domain:
        check(_actual_fn(_x) != _claimed_fn(_x), f"{_actual_adj!r}/{_claimed_adj!r} pair not complementary at n={_x}")
del _actual_adj, _actual_fn, _claimed_adj, _claimed_fn, _pair_domain, _x

_COMPOUND_DOMAIN = list(range(2, 50))


def _false_statement_for(n: int, pair_tuple) -> str:
    """A FALSE statement about `n` built from `pair_tuple` (a `_PAIRS` entry
    other than the one the question is built from) -- used as a proof-by-
    contradiction distractor. Exactly one of `actual_adj`/`claimed_adj`
    describes `n` truthfully; this returns the OTHER one, which is false."""
    actual_adj, actual_fn, claimed_adj, claimed_fn, _domain = pair_tuple
    if not actual_fn(n):
        return f"{n} is {actual_adj}"
    return f"{n} is {claimed_adj}"


def _adj_truth(n: int, pair_tuple, adj: str) -> bool:
    """Independent recomputation: the truth of "n is {adj}" at `n`, evaluated
    by matching `adj` back to whichever of `pair_tuple`'s two functions it
    names and calling that function directly -- not by re-reading which list
    `n` came from or trusting `_false_statement_for`'s own routing."""
    actual_adj, actual_fn, claimed_adj, claimed_fn, _domain = pair_tuple
    if adj == actual_adj:
        return actual_fn(n)
    if adj == claimed_adj:
        return claimed_fn(n)
    raise ValueError(f"adjective {adj!r} is not part of this pair")


def _single_tier(rng, pairs_slice):
    chosen = rng.choice(pairs_slice)
    actual_adj, actual_fn, claimed_adj, claimed_fn, domain = chosen
    candidates = [x for x in domain if actual_fn(x)]
    check(candidates, f"no candidates satisfying {actual_adj!r} over its domain")
    n = rng.choice(candidates)

    # Independent check: recompute directly at n, not by re-reading the list
    # n was drawn from.
    check(actual_fn(n) and not claimed_fn(n), "constructed n fails its own pair's complementary property")

    text = (
        f"Claim: {n} is {claimed_adj}. Suppose, for contradiction, that this claim is true "
        f"— that is, suppose {n} is {claimed_adj}. Which of the following is a TRUE statement "
        f"about {n} that contradicts this assumption?"
    )
    answer_text = f"{n} is {actual_adj}"
    check(answer_text not in text, "answer_text leaked into question text")

    other_tuples = rng.sample([t for t in _PAIRS if t is not chosen], 3)
    distractor_candidates = [_false_statement_for(n, t) for t in other_tuples]
    check(all(d != answer_text for d in distractor_candidates), "a distractor equals the answer")
    for cand, t in zip(distractor_candidates, other_tuples):
        stated_adj = cand[len(f"{n} is ") :]
        check(_adj_truth(n, t, stated_adj) is False, f"distractor {cand!r} is not a false statement about {n}")

    distractors = pick_distractors(answer_text, distractor_candidates)
    working = [
        f"The claim assumes {n} is {claimed_adj}, which is being tested for contradiction.",
        f"But {n} is {actual_adj} -- exactly one of {actual_adj}/{claimed_adj} can hold, "
        f"so this fact contradicts the assumption.",
    ]
    return text, answer_text, distractors, working


def _compound_claim(rng):
    intersection: list[int] = []
    t1 = t2 = None
    for _attempt in range(200):
        t1, t2 = rng.sample(_PAIRS, 2)
        actual_adj_1, actual_fn_1, claimed_adj_1, claimed_fn_1, _d1 = t1
        actual_adj_2, actual_fn_2, claimed_adj_2, claimed_fn_2, _d2 = t2
        intersection = [x for x in _COMPOUND_DOMAIN if actual_fn_1(x) and actual_fn_2(x)]
        if intersection:
            break
    check(intersection, "no pair of _PAIRS tuples has a non-empty intersection over range(2, 50)")
    actual_adj_1, actual_fn_1, claimed_adj_1, claimed_fn_1, _d1 = t1
    actual_adj_2, actual_fn_2, claimed_adj_2, claimed_fn_2, _d2 = t2
    n = rng.choice(intersection)
    false_clause = rng.choice([1, 2])

    # Independent check: recompute directly at n.
    check(actual_fn_1(n) and actual_fn_2(n), "compound-claim independent recheck failed")

    clause1_adj = claimed_adj_1 if false_clause == 1 else actual_adj_1
    clause2_adj = actual_adj_2 if false_clause == 1 else claimed_adj_2

    text = (
        f"Claim: {n} is {clause1_adj} and {n} is {clause2_adj}. Assume this compound claim "
        f"for contradiction. Which clause produces the contradiction?"
    )
    options = [
        f"The first clause ('{n} is {clause1_adj}')",
        f"The second clause ('{n} is {clause2_adj}')",
        "Both clauses",
        "Neither clause",
    ]
    answer_text = options[0] if false_clause == 1 else options[1]
    check(answer_text not in text, "answer_text leaked into question text")

    # Independent check: directly evaluate both stated clauses' truth at n
    # and confirm exactly one is False, and that it is the one false_clause
    # designates.
    truth1 = _adj_truth(n, t1, clause1_adj)
    truth2 = _adj_truth(n, t2, clause2_adj)
    check((truth1, truth2).count(False) == 1, "compound claim must have exactly one false clause")
    check((not truth1 and false_clause == 1) or (not truth2 and false_clause == 2), "false-clause designation mismatch")

    distractors = pick_distractors(answer_text, [o for o in options if o != answer_text])
    working = [
        f"The first clause states {n} is {clause1_adj}; the second clause states {n} is {clause2_adj}.",
        f"Exactly one of the two stated clauses is false at n = {n}.",
    ]
    return text, answer_text, distractors, working, false_clause


@generator(
    "proof_by_contradiction_identify",
    level=12,
    presets={
        "easy": dict(variant="single_simple"),
        "medium": dict(variant="single_computed"),
        "hard": dict(variant="compound_claim"),
    },
)
def proof_by_contradiction_identify(rng, variant="single_simple"):
    """Identify which stated fact contradicts an assumption made for a proof
    by contradiction. `text` states the claim and the assumption in full and
    never the resolution.
    """
    if variant not in ("single_simple", "single_computed", "compound_claim"):
        raise ValueError(f"variant must be 'single_simple', 'single_computed' or 'compound_claim'; got {variant!r}")

    if variant == "single_simple":
        text, answer_text, distractors, working = _single_tier(rng, _PAIRS[0:2])
    elif variant == "single_computed":
        text, answer_text, distractors, working = _single_tier(rng, _PAIRS[2:5])
    else:  # compound_claim
        text, answer_text, distractors, working, _false_clause = _compound_claim(rng)

    latex = f"plain:{text}"
    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=working,
        distractors=distractors,
    )
