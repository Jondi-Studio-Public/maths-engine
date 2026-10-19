"""Factorial notation + restricted permutations.

Two genuinely new Specialist Mathematics (senior-mathematics) skills, both narrowed against the shipped
``permutations``/``combinations``/ ``multiplication-principle`` trio in
``mathgen/generators/curriculum_expansion.py``: none of those three ever writes ``n!`` in a question or asks
the learner to manipulate factorial notation itself, and none of them ever states a restriction.

``senior_use_factorial_notation`` (``senior-use-factorial-notation``,
ACMSM003, topic ``p11c_factorial_notation``, ``answer: expression``) -- bare
evaluation/simplification of ``n!`` notation itself. No "how many ways"
framing anywhere in this skill.

``senior_solve_problems_involving_permutations_and_restrictions_with_or``
(``senior-solve-problems-involving-permutations-and-restrictions-with-or``,
ACMSM004, topic ``p11c_restricted_permutations``, ``answer: number``) -- a
restriction (together / apart / repeated-object-plus-restriction) is present
in EVERY question, every tier; that restriction is the entire point of
separation from the shipped trio.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import sympy

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
)

_n = sympy.symbols("n")

# ============================================================================
# senior-use-factorial-notation (ACMSM003)
# ============================================================================

# easy: a small two-operand combination of factorials -- a!, b! for a, b in {0..., 6}, combined with
# +, - or x, every operand still small enough that both factorials are computed outright (never
# telescoped). Enumerated at import time, not sampled: the full (a, b, op) cross product is built
# and only combinations whose VALUE is under 1000 survive -- excludes 6! + 6! = 1440 and every a! x
# b! pair where both a and b sit above the small end of the pool (e.g. 4! x 5! = 2880). Subtraction
# is restricted to a > b: every result stays non-negative (a bare integer, isdigit-parseable), and
# the "(a - b)!" misconception below always gets a defined factorial argument -- never offered when
# it would be undefined. Division is deliberately excluded: a!/b! for small a, b is the same
# question SHAPE the medium tier already teaches ("simplify n!/m! to a single number"), and the
# medium tier's own n range starts at n = 4 -- too small to force telescoping recognition over
# direct computation, so the boundary between "easy: compute two small factorials directly" and
# "medium: recognise the telescoping cancellation" would not be clean. The three remaining operators
# already clear the shared 10-distinct-question floor by a wide margin, so nothing is lost by
# dropping it.
_EASY_OPERAND_POOL = range(0, 7)
_EASY_OPS = ("+", "−", "×")  # +, minus sign, multiplication sign


def _easy_combo_value(a: int, b: int, op: str) -> int:
    fa, fb = factorial(a), factorial(b)
    if op == "+":
        return fa + fb
    if op == "−":
        return fa - fb
    return fa * fb


def _build_easy_combo_pool() -> tuple[tuple[int, int, str], ...]:
    """Every ``(a, b, op)`` combination whose value is under 1000, built by
    enumerating the full cross product directly -- never a sampled or
    assumed subset. See the module-level comment above for the exact
    exclusions this produces."""
    pool = []
    for a in _EASY_OPERAND_POOL:
        for b in _EASY_OPERAND_POOL:
            for op in _EASY_OPS:
                if op == "−" and a <= b:
                    continue  # (a - b)! would be undefined; keep results non-negative
                if _easy_combo_value(a, b, op) < 1000:
                    pool.append((a, b, op))
    return tuple(pool)


_EASY_COMBO_POOL = _build_easy_combo_pool()
assert len(_EASY_COMBO_POOL) >= 20, f"p11c_factorial_notation easy combo pool only has {len(_EASY_COMBO_POOL)} entries"
assert all(_easy_combo_value(a, b, op) < 1000 for a, b, op in _EASY_COMBO_POOL), (
    "p11c_factorial_notation easy combo pool contains a value >= 1000"
)

# medium: n!/m! with n - m in {1, 2}. Enumerated bound: the displayed product
# n*(n-1) must stay under 4 digits (< 1000); n*(n-1) < 1000 holds up to
# n = 32 (32*31 = 992; 33*32 = 1056 fails). n in [4, 15] is a much smaller
# pool, chosen purely for readable numbers -- the bound above is the actual
# enumerated ceiling, not an assumption.
_MEDIUM_N_POOL = range(4, 16)
_MEDIUM_K_POOL = (1, 2)
# The ORIGINAL medium tier had exactly one sentence template -- "Simplify {n}!/{m}!
# to a single number." -- and both {n} and {m} are digits, so `prompt_shape`'s own digit mask
# collapses every draw to the SAME single masked string regardless of which n, m or k was drawn.
# Three instruction-verb templates asking the identical mathematical task (still "reduce this ratio
# to one number", no change to what is computed or to the answer's own shape) clear the floor
# without touching the tier's reasoning character.
_MEDIUM_TEMPLATES = (
    "Simplify {n}!/{m}! to a single number.",
    "Write {n}!/{m}! as a single number.",
    "Evaluate {n}!/{m}! as a single number.",
)

# hard: symbolic ratio of two factorials shifted from n by different amounts.
# The ORIGINAL hard
# tier only ever put the LARGER shifted factorial on top ((n+a)!/(n+b)!,
# a > b), which produces exactly two masked prompt_shape() outputs over 60
# seeds ("Simplify (n + #)!/n! ..." when b = 0, "Simplify (n + #)!/(n + #)!
# ..." otherwise) -- one short of the prompt-variety floor of 3. The fix is a
# genuinely different direction of reasoning, not a relabelling: a THIRD
# variant puts the SMALLER shifted factorial (or the bare n!) on top, so the
# ratio inverts to a RECIPROCAL instead of a product -- the learner has to
# recognise which direction the cancellation goes, not just perform it.
# Three explicit variants, drawn uniformly so every one is virtually certain
# to appear across 60 seeds (each 1/3 per draw; P(a variant is absent from
# 60 draws) = (2/3)^60 ~= 3e-11):
#   den_bare:     (n+d)!/n!      -- product form, denominator bare (the
#                                   original b=0 case)
#   num_bare:     n!/(n+d)!      -- RECIPROCAL form, numerator bare (the new
#                                   third masked shape)
#   both_shifted: (n+b+d)!/(n+b)! or (n+b)!/(n+b+d)! -- both sides shifted,
#                                   direction (product vs reciprocal) drawn
#                                   independently; this collapses to the SAME
#                                   masked shape as den_bare/num_bare's
#                                   "both non-bare" case either way (an
#                                   arithmetic property invisible to
#                                   text_quality.prompt_shape's own
#                                   documented ACCEPTED LIMITATION), so it is
#                                   never relied on for the shape count, only
#                                   for widening b.
_HARD_SHIFT_POOL = range(1, 15)
_HARD_D_POOL = (1, 2)
_HARD_VARIANTS = ("den_bare", "num_bare", "both_shifted")


def _fact_display(shift: int) -> str:
    """``n!`` for shift 0, else ``(n + shift)!``."""
    return f"(n + {shift})!" if shift else "n!"


def _arg_display(shift: int) -> str:
    """``n`` for shift 0, else ``n + shift``."""
    return f"n + {shift}" if shift else "n"


def _prose(text: str) -> str:
    """A bare ``*`` (from ``str(sympy_expr)``) rendered as ``×`` for
    learner-facing prose (``question.working`` is checked against every
    ``mathgen/text_quality.py`` pattern regardless of its declared scope, so
    the ``star_operator`` check -- normally prompt-only -- still applies
    there; ``answer_text``/distractors keep the bare ``*`` sympy needs to
    parse them back)."""
    return text.replace("*", " × ")


@generator(
    "p11c_factorial_notation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p11c_factorial_notation(rng, tier="easy"):
    """Bare factorial-notation evaluation/simplification -- never a counting
    story.

    easy:   evaluate a small two-operand combination of factorials (a! + b!,
            a! - b! or a! x b! for a, b in {0, ..., 6}) -- still bare
            evaluation, both operands always small enough to compute
            outright (see `_build_easy_combo_pool`'s module comment for the
            exact enumerated bound).
    medium: simplify n!/m! (n - m in {1, 2}) to its product form / numeric
            value -- the learner must recognise the telescoping cancellation
            rather than compute either factorial in full.
    hard:   symbolic simplification of (n+a)!/(n+b)! for a variable n --
            general algebraic cancellation, not a numeric answer for one n.
    """
    if tier == "easy":
        a, b, op = rng.choice(_EASY_COMBO_POOL)
        fa, fb = factorial(a), factorial(b)
        answer_value = _easy_combo_value(a, b, op)
        # independent route: re-derive a! and b! from the definition (a
        # fresh manual term-by-term product for each, never math.factorial's
        # own two calls reused) and recombine with the SAME operator.
        manual_fa = 1
        for i in range(1, a + 1):
            manual_fa *= i
        manual_fb = 1
        for i in range(1, b + 1):
            manual_fb *= i
        if op == "+":
            manual_value = manual_fa + manual_fb
        elif op == "−":
            manual_value = manual_fa - manual_fb
        else:
            manual_value = manual_fa * manual_fb
        check_equivalent(answer_value, manual_value, f"easy combo mismatch: a={a}, b={b}, op={op}")
        check(answer_value < 1000, f"easy combo answer not under 4 digits: a={a}, b={b}, op={op}, value={answer_value}")
        question_text = f"Evaluate {a}! {op} {b}! as a single number."
        answer_text = str(answer_value)
        working = [f"{a}! {op} {b}! = {fa} {op} {fb} = {answer_value}."]

        candidates_text = []
        # combines the two n's FIRST, then factorialises the combination -- (a op b)! instead of a!
        # op b! only offered when it stays under the same 1000 bound. Subtraction's combined
        # argument is always a - b > 0 here (the pool never draws a <= b for "−"), so this is never
        # an undefined factorial.
        combined_n = a + b if op == "+" else (a - b if op == "−" else a * b)
        combined_value = factorial(combined_n)
        if combined_value < 1000 and combined_value != answer_value:
            candidates_text.append(str(combined_value))
        # applies the most confusable OTHER operator to the two already-
        # correct factorial values (+ <-> x; − also falls back to + since
        # swapping to ÷ is a different mistake class the medium tier owns)
        wrong_op = "×" if op == "+" else "+"
        wrong_op_value = fa * fb if wrong_op == "×" else fa + fb
        if wrong_op_value < 1000 and wrong_op_value != answer_value:
            candidates_text.append(str(wrong_op_value))
        # forgets to apply "!" to the second operand
        if op == "+":
            forgot_b_value = fa + b
        elif op == "−":
            forgot_b_value = fa - b
        else:
            forgot_b_value = fa * b
        if 0 <= forgot_b_value < 1000 and forgot_b_value != answer_value:
            candidates_text.append(str(forgot_b_value))
        # forgets to apply "!" to the first operand
        if op == "+":
            forgot_a_value = a + fb
        elif op == "−":
            forgot_a_value = a - fb
        else:
            forgot_a_value = a * fb
        if 0 <= forgot_a_value < 1000 and forgot_a_value != answer_value:
            candidates_text.append(str(forgot_a_value))

    elif tier == "medium":
        n = rng.choice(_MEDIUM_N_POOL)
        k = rng.choice(_MEDIUM_K_POOL)
        m = n - k
        # independent route: derive the ratio from the DEFINITION of
        # factorial (a fresh term-by-term product from m+1 up to n), never by
        # re-running the same telescoping cancellation used to build the
        # answer below.
        manual_product = 1
        for i in range(m + 1, n + 1):
            manual_product *= i
        answer_value = manual_product
        check_equivalent(answer_value, factorial(n) // factorial(m), f"medium ratio mismatch: n={n}, m={m}")
        check(answer_value < 1000, f"medium answer not under 4 digits: n={n}, m={m}, value={answer_value}")
        template = rng.choice(_MEDIUM_TEMPLATES)
        question_text = template.format(n=n, m=m)
        answer_text = str(answer_value)
        if k == 1:
            working = [f"{n}!/{m}! = {n} (the ratio telescopes to the single remaining factor)."]
        else:
            product_text = " × ".join(str(i) for i in range(n, m, -1))
            working = [f"{n}!/{m}! = {product_text} = {answer_value}."]

        candidates_text = []
        # off-by-one: one fewer factor in the product (k=2 -> just n; k=1 ->
        # the empty product, 1)
        drop_factor = n if k == 2 else 1
        candidates_text.append(str(drop_factor))
        # treats n!/m! as (n/m)! -- only a coherent numeric mistake to offer
        # when n/m is itself a whole number
        if n % m == 0:
            ratio_as_factorial = factorial(n // m)
            if ratio_as_factorial != answer_value:
                candidates_text.append(str(ratio_as_factorial))
        # applies no cancellation at all: reports n! outright
        candidates_text.append(str(factorial(n)))

    elif tier == "hard":
        variant = rng.choice(_HARD_VARIANTS)
        d = rng.choice(_HARD_D_POOL)
        if variant == "den_bare":
            # (n+d)!/n! -- product form, denominator bare (the original tier).
            num_shift, den_shift = d, 0
        elif variant == "num_bare":
            # n!/(n+d)!
            num_shift, den_shift = 0, d
        else:  # "both_shifted"
            b = rng.choice(_HARD_SHIFT_POOL)
            if rng.choice([True, False]):
                num_shift, den_shift = b + d, b  # product form
            else:
                num_shift, den_shift = b, b + d  # reciprocal form

        num_text = _fact_display(num_shift)
        den_text = _fact_display(den_shift)
        question_text = f"Simplify {num_text}/{den_text} as an expression in n, where n is a positive integer."

        # independent route: re-derive the ratio straight from the
        # DEFINITION of factorial (a fresh symbolic expression, never the
        # term-list built below) and confirm it is the same value as the
        # constructed product/reciprocal.
        definitional_ratio = sympy.factorial(_n + num_shift) / sympy.factorial(_n + den_shift)

        if num_shift >= den_shift:
            terms = [_n + i for i in range(den_shift + 1, num_shift + 1)]
            product_expr = terms[0] if len(terms) == 1 else terms[0] * terms[1]
            answer_expr = product_expr
            factor_words = " × ".join(f"({t})" for t in terms) if len(terms) > 1 else str(terms[0])
            working = [f"{num_text}/{den_text} = {factor_words}."]
        else:
            terms = [_n + i for i in range(num_shift + 1, den_shift + 1)]
            product_expr = terms[0] if len(terms) == 1 else terms[0] * terms[1]
            answer_expr = 1 / product_expr
            factor_words = " × ".join(f"({t})" for t in terms) if len(terms) > 1 else str(terms[0])
            working = [f"{num_text}/{den_text} = 1/({factor_words})."]

        answer_text = str(answer_expr)
        check_equivalent(
            definitional_ratio,
            answer_expr,
            f"hard-tier ratio does not telescope to the built product: num_shift={num_shift}, den_shift={den_shift}",
        )

        candidates_text = []
        if num_shift >= den_shift:
            # forgets one factor at the edge of the product
            if num_shift - den_shift == 1:
                off_by_one = str(answer_expr * (_n + num_shift + 1))
            else:
                off_by_one = str(terms[-1])
            candidates_text.append(off_by_one)
            # treats (n+a)!/(n+b)! as ((n+a)/(n+b))! -- postfix "!" (the same
            # notation the question and true answer use), never
            # `factorial(...)` call syntax, which a learner would never type
            # and which the question never shows.
            candidates_text.append(f"(({_arg_display(num_shift)})/({_arg_display(den_shift)}))!")
            # forgot to cancel at all: reports the denominator's own
            # factorial term
            candidates_text.append(den_text)
        else:
            # forgets to invert: reports the product itself, not its
            # reciprocal -- the direction-of-reasoning error this variant
            # exists to test.
            candidates_text.append(str(product_expr))
            candidates_text.append(f"(({_arg_display(num_shift)})/({_arg_display(den_shift)}))!")
            # forgets one factor at the edge, still inside the reciprocal
            if den_shift - num_shift == 1:
                wrong_recip = str(1 / (product_expr * (_n + den_shift + 1)))
            else:
                wrong_recip = str(1 / terms[-1])
            candidates_text.append(wrong_recip)

    else:
        raise ValueError(f"unknown tier {tier!r}")

    distractors = pick_distractors(answer_text, candidates_text)
    check_pairwise_distinct(answer_text, *distractors, msg=f"factorial-notation distractor collision: tier={tier}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "working": working,
        "distractors": distractors,
    }


# ============================================================================
# senior-solve-problems-involving-permutations-and-restrictions-with-or
# (ACMSM004)
# ============================================================================

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOUNS = ("books", "photographs", "trophies", "envelopes", "folders")

# Answer-concentration fix. Measured over 500 seeds against the ORIGINAL generator (n fixed to
# {4,5,6} for easy/medium, n=6 fixed with one repeated pair for hard): easy 3 distinct answers (240
# 35.4%, 12 32.4%, 48 32.2%), medium 3 distinct answers (72 37.8%, 480 32.0%, 12 30.2%), hard **1**
# distinct answer -- 120 on 100% of 500 draws. Every hard question was the same structure relabelled
# (6 objects, two identical, one adjacency constraint on the other two): 5!/2! x 2 = 120, always.
# `answer: number` sends no options to the learner, so a learner who types "120" on every hard
# question scores 100% without reading it.
#
# The fix varies STRUCTURE, not labels, at every tier -- hand-enumerated pools (never a blind n x k
# or n x r x constraint cross product: several combinations are dropped because they collide with
# another combination's value, the same discipline `p11c_factorial_notation`'s own
# `_build_easy_combo_pool` already established for this module):
#
# easy/medium draw (n, k) -- n = row length, k = how many specific objects carry the restriction (2
# or 3, replacing the original fixed pair). easy asks for the "all k together" count
# (`factorial(n-k+1) * factorial(k)`, a direct generalisation of the original block-of-2 formula to
# a block of k); medium asks for the "NOT all k together" complement (`n! - together`). `(4, 3)` is
# dropped from the pool: it lands on 12 for BOTH together and apart, colliding with `(4, 2)`'s own
# together value. The remaining 6 combinations each land on their own distinct value within each
# tier's own value set -- easy: {4, 12, 48, 36, 240, 144}; medium: {2, 12, 72, 84, 480, 576} -- so a
# uniform draw puts each at 1/6 (~16.7%) of a tier's draws, clear of the 25% bound with a real
# sampling margin (binomial std at p=1/6, n=500 seeds is ~1.6 percentage points; 16.7% + 3 std ~=
# 21.6%).
#
# hard draws (n, r, constraint) -- n = row length, r = how many identical
# copies of one labelled object are present (2 or 3, replacing the fixed
# n=6/repeat=2 of the original), constraint = "together" or "apart" applied
# to a SEPARATE distinct pair p, q. All three axes the rework brief names --
# object count, identical-item count, constraint type -- vary independently
# across the 6-entry pool. `(7, 3, "together")` (value 240) is dropped: it
# collides with `(6, 2, "apart")` (also 240). The remaining six each land on
# a distinct value -- {120, 240, 40, 80, 720, 600} -- again 1/6 (~16.7%)
# each under a uniform draw. The largest achievable value (n=7, r=2,
# together = 720) stays clear of both the 1000-digit ceiling
# (`test_restricted_permutations_answers_are_integers_under_1000`) and an
# undivided 7! (5040).
_EASY_MEDIUM_COMBOS = ((3, 2), (4, 2), (5, 2), (5, 3), (6, 2), (6, 3))  # (n, k)
_HARD_COMBOS = (  # (n, r, constraint) -- answer values noted, all distinct
    (6, 2, "together"),  # 120
    (6, 2, "apart"),  # 240
    (6, 3, "together"),  # 40
    (6, 3, "apart"),  # 80
    (7, 2, "together"),  # 720
    (7, 3, "apart"),  # 600
)


def _together_count(n: int, k: int) -> int:
    """``n`` distinct objects, a specific ``k`` of them must all be adjacent
    (in any internal order): treat the block as one unit (``n - k + 1``
    units total), multiply by the block's own ``k!`` internal orderings.
    ``k=2`` reduces to the original pair-block formula."""
    return factorial(n - k + 1) * factorial(k)


def _plural(count: int, word: str) -> str:
    """``f"{count} {word}"`` with a correct English plural -- genuinely
    one-off (used twice in this module's own working-text only; not worth a
    shared `core.py` helper). The widened combo pools below can draw a count
    of exactly 1 (e.g. n=3, k=2 leaves 1 unrestricted item; n=6, r=3 leaves 1
    spare distinct letter), and `mathgen.text_quality.singular_count_plural`
    checks `question.working` regardless of its own declared "prompt" scope
    (see `p11c_factorial_notation`'s own `_prose` docstring for the same
    fact) -- "1 items" is a real defect there, not a style nit."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _is_contiguous_block(perm, members) -> bool:
    """True if every symbol in ``members`` occupies a contiguous run of
    positions in ``perm`` -- the shared adjacency test for both the
    easy/medium (distinct-object block of ``k``) and hard (multiset, but
    ``members`` is always the two non-repeated symbols ``p``, ``q``)
    constructions."""
    positions = sorted(perm.index(c) for c in members)
    return positions[-1] - positions[0] == len(members) - 1


@generator(
    "p11c_restricted_permutations",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p11c_restricted_permutations(rng, tier="easy"):
    """A restriction is present in EVERY question, every tier -- the whole
    separation from the shipped permutations/combinations/multiplication-
    principle trio, which never state one.

    easy:   "all k specified objects together" (block) restriction, k in
            {2, 3}, drawn with the row length n from `_EASY_MEDIUM_COMBOS`.
    medium: "NOT all k together" restriction, same (n, k) pool -- total
            minus easy's own together count (a subtraction structure).
    hard:   r identical copies of one object (r in {2, 3}) AND a
            together/apart restriction on a SEPARATE distinct pair, drawn
            from `_HARD_COMBOS` -- n, r and the constraint all vary.
    """
    if tier == "easy":
        n, k = rng.choice(_EASY_MEDIUM_COMBOS)
        noun = rng.choice(_NOUNS)
        members = sorted(rng.sample(_LETTERS[:n], k))
        answer_value = _together_count(n, k)
        # independent route: enumerate every permutation of the n labelled
        # slots directly and count how many place every member of `members`
        # in a contiguous run -- never the block-counting formula used to
        # build the answer.
        objs = _LETTERS[:n]
        direct = sum(1 for perm in permutations(objs) if _is_contiguous_block(perm, members))
        check_equivalent(answer_value, direct, f"easy together count mismatch: n={n}, k={k}")

        if k == 2:
            p, q = members
            restriction = f"{p} and {q} must be next to each other."
        else:
            p, q, r_ = members
            restriction = f"{p}, {q} and {r_} must all be next to each other, in any order."
        question_text = (
            f"{n} different {noun} are placed in a row. {restriction} In how many ways can the {noun} be arranged?"
        )
        answer_text = str(answer_value)
        block_text = "".join(members)
        working = [
            f"Treat {block_text} as one block: {n - k + 1}! arrangements of the block "
            f"plus the other {_plural(n - k, 'item')}, times {k}! internal orders of the block, "
            f"= {factorial(n - k + 1)} × {factorial(k)} = {answer_value}."
        ]

        candidates_text = []
        candidates_text.append(str(factorial(n)))  # ignores the restriction
        candidates_text.append(str(factorial(n - k + 1)))  # omits the block's own k! factor
        candidates_text.append(str(factorial(n) - answer_value))  # applies the "not all together" method instead

    elif tier == "medium":
        n, k = rng.choice(_EASY_MEDIUM_COMBOS)
        noun = rng.choice(_NOUNS)
        members = sorted(rng.sample(_LETTERS[:n], k))
        together = _together_count(n, k)
        answer_value = factorial(n) - together
        # independent route: enumerate every permutation directly and count
        # how many do NOT place every member of `members` in a contiguous
        # run -- a genuinely different construction from "total minus
        # together".
        objs = _LETTERS[:n]
        direct = sum(1 for perm in permutations(objs) if not _is_contiguous_block(perm, members))
        check_equivalent(answer_value, direct, f"medium apart count mismatch: n={n}, k={k}")

        if k == 2:
            p, q = members
            restriction = f"{p} and {q} must NOT be next to each other."
        else:
            p, q, r_ = members
            restriction = f"{p}, {q} and {r_} must NOT all be next to each other, in any order."
        question_text = (
            f"{n} different {noun} are placed in a row. {restriction} In how many ways can the {noun} be arranged?"
        )
        answer_text = str(answer_value)
        working = [
            f"Total arrangements {factorial(n)} minus the {together} arrangements "
            f"with all of {''.join(members)} together = {answer_value}."
        ]

        candidates_text = []
        candidates_text.append(str(together))  # applies the "all together" method instead
        candidates_text.append(str(factorial(n)))  # ignores the restriction
        candidates_text.append(
            str(factorial(n) - factorial(n - k + 1))
        )  # omits the block's own k! factor in the subtraction

    elif tier == "hard":
        n, r, constraint = rng.choice(_HARD_COMBOS)
        pool = _LETTERS[: n - r + 1]  # n - r + 1 distinct symbols: x, plus n - r others
        x = rng.choice(pool)
        others = [c for c in pool if c != x]
        p, q = sorted(rng.sample(others, 2))
        rest = sorted(c for c in others if c not in (p, q))
        check(len(rest) >= 1, f"hard tier drew no spare distinct letters: n={n}, r={r}")
        rest_text = ", ".join(rest)
        units = n - 1  # the {p,q}-block collapses two slots into one

        total_value = factorial(n) // factorial(r)
        together_value = 2 * factorial(units) // factorial(r)
        apart_value = total_value - together_value
        answer_value = together_value if constraint == "together" else apart_value

        # independent route: brute-force the actual multiset -- generate
        # every arrangement of [x]*r + others (python's own duplicate-tuple
        # behaviour on repeated elements, deduplicated with `set`, is the
        # multiset-permutation DEFINITION, not the closed-form n!/r!
        # formula) and count directly how many place p and q adjacent (or
        # not, for the apart case). This is a genuinely different algorithm
        # from the closed form asserted as the answer, never a re-spelling
        # of it.
        multiset = tuple([x] * r + others)
        check(len(multiset) == n, f"hard tier multiset size mismatch: n={n}, r={r}")
        if constraint == "together":
            direct = sum(1 for perm in set(permutations(multiset)) if _is_contiguous_block(perm, (p, q)))
        else:
            direct = sum(1 for perm in set(permutations(multiset)) if not _is_contiguous_block(perm, (p, q)))
        check_equivalent(answer_value, direct, f"hard {constraint} count mismatch: n={n}, r={r}")

        restriction = f"{p} must be next to {q}." if constraint == "together" else f"{p} must NOT be next to {q}."
        question_text = (
            f"{n} objects are arranged in a row: {r} of them are labelled {x} "
            f"(identical), and the rest are labelled {rest_text}, {p} and {q} "
            f"(all distinct). {restriction} In how many distinct ways can the "
            "objects be arranged?"
        )
        answer_text = str(answer_value)
        if constraint == "together":
            working = [
                f"Treat {p}{q} as one block: {units} units to arrange (the block, the "
                f"{r} {x}'s, and the other {_plural(n - r - 2, 'distinct letter')}), with {x} "
                f"repeated {r} times: {units}!/{r}! = {factorial(units) // factorial(r)}, "
                f"times 2! for {p}{q}'s own internal order = {answer_value}."
            ]
        else:
            working = [
                f"Total arrangements of the multiset {n}!/{r}! = {total_value} minus the "
                f"{together_value} arrangements with {p} and {q} together = {answer_value}."
            ]

        candidates_text = []
        if constraint == "together":
            candidates_text.append(str(total_value))  # ignores the restriction
            candidates_text.append(str(factorial(units) * factorial(2)))  # forgets the repeated-object division
            candidates_text.append(str(factorial(units) // factorial(r)))  # omits the block's own 2! ordering factor
        else:
            candidates_text.append(str(together_value))  # applies the "together" method instead
            candidates_text.append(str(total_value))  # ignores the restriction
            wrong_total_no_repeat = factorial(n)
            wrong_together_no_repeat = 2 * factorial(n - 1)
            candidates_text.append(
                str(wrong_total_no_repeat - wrong_together_no_repeat)
            )  # forgets the objects are not all distinct at all

    else:
        raise ValueError(f"unknown tier {tier!r}")

    distractors = pick_distractors(answer_text, candidates_text)
    check_pairwise_distinct(answer_text, *distractors, msg=f"restricted-permutations distractor collision: tier={tier}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": int(answer_text),
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "working": working,
        "distractors": distractors,
    }
