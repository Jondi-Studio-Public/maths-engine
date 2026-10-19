"""Recognise and use the recursive definition of an arithmetic / geometric sequence, both ``acara-senior-v8-4``,
domain ``functions-sequences``, ``answer: pair``.

Two generators:

- ``arithmetic_recursive_definition_state``
  (``senior-recognise-and-use-the-recursive-definition-of-an``, ACMMM068) --
  given the first several TERMS of an arithmetic sequence, state the first
  term and the common difference d in t(n+1) = t(n) + d.
- ``geometric_recursive_definition_state``
  (``senior-recognise-and-use-the-recursive-definition-of-a``, ACMMM072) --
  given the first several TERMS of a geometric sequence, state the first
  term and the common ratio r in t(n+1) = r * t(n).

Both are the REVERSE inference direction of the shipping
``senior-use-recursion-to-generate-an-arithmetic-sequence`` /
``senior-use-recursion-to-generate-a-geometric-sequence`` skills
(``mathgen/generators/p3c_recursion_build.py``), which STATE t1
and the rule up front and ask the learner to GENERATE the next k terms as an
ordered_list. Here t1 and the rule's constant are never given -- only the
terms -- and the learner infers the constant and states it as a labelled
`pair` (``answer: pair``, not ``ordered_list``). Never state t1 or the
constant in the question text, and never label the given terms.
"""

from __future__ import annotations

from ..core import check, check_pairwise_distinct, generator, paren_negative, phrase, pick_distractors

# --- shared real-world contexts (hard tier) ------------------------------
# Every context is chosen to be honestly defensible under BOTH an increasing
# and a decreasing reading (arithmetic hard allows either sign of d), or, for
# the geometric alternating contexts, specifically under a sign-alternating
# reading (only ever paired with a negative r) -- see the per-generator hard
# branch below.

_ARITH_CONTEXTS = [
    "The temperature, in degrees Celsius, is recorded each hour: {terms}",
    "A hiker's altitude, in metres above sea level, is recorded at each checkpoint: {terms}",
    "A bank account balance, in dollars, is recorded at the end of each week: {terms}",
    "A drone's height above the ground, in metres, is recorded every ten seconds: {terms}",
]

_GEOM_GROWTH_CONTEXTS = [
    "A population of bacteria is counted each hour: {terms}",
    "An investment's value, in dollars, is recorded at the end of each year: {terms}",
    "The number of views on a viral video is recorded each day: {terms}",
]

_GEOM_ALTERNATING_CONTEXTS = [
    "A pendulum's displacement from its resting point, in centimetres, is measured after each swing: {terms}",
    "A signal's voltage, in millivolts, is measured after each cycle: {terms}",
]

# --- abstract (context-free) lead-ins (easy/medium tiers) -----------------
# `context is None` at easy/medium, so without these the same single
# template rendered every time (only the digits inside {terms} varied,
# which mathgen.text_quality.prompt_shape masks away -- 1 shape over 60
# seeds). `phrase()` (mathgen/core.py) picks one lead-in per render with the
# generator's own `rng`, AFTER every other rng draw the function makes, so
# the a1/d/r distribution and the resample loops are undisturbed. Every
# template ends in the exact same `{terms}. State its recursive
# definition: ...` suffix the term/answer-parsing regexes in
# tests/test_p11b_recursive_definition_build.py and
# tests/test_p11b_recursive_definition_oracle.py depend on -- only the
# sentence introducing {terms} varies.

_ARITH_ABSTRACT_TEMPLATES = [
    "An arithmetic sequence begins {terms}. State its recursive definition: the first "
    "term, and the value of d in the rule t(n+1) = t(n) + d.",
    "The first few terms of an arithmetic sequence are {terms}. State its recursive "
    "definition: the first term, and the value of d in the rule t(n+1) = t(n) + d.",
    "Consider the arithmetic sequence {terms}. State its recursive definition: the "
    "first term, and the value of d in the rule t(n+1) = t(n) + d.",
]

_GEOM_ABSTRACT_TEMPLATES = [
    "A geometric sequence begins {terms}. State its recursive definition: the first "
    "term, and the value of r in the rule t(n+1) = r × t(n).",
    "The first few terms of a geometric sequence are {terms}. State its recursive "
    "definition: the first term, and the value of r in the rule t(n+1) = r × t(n).",
    "Consider the geometric sequence {terms}. State its recursive definition: the "
    "first term, and the value of r in the rule t(n+1) = r × t(n).",
]


# --- arithmetic_recursive_definition_state --------------------------------


def _verify_arithmetic_definition(a1: int, d: int, terms: list[int]) -> None:
    """Construction builds `terms` by an iterative accumulation loop
    (cur += d, k times, with terms[0] == a1). This check recomputes d from
    the STATED TERMS alone via terms[1] - terms[0] -- a genuinely different
    derivation, not a re-run of the same loop -- cross-checked against
    terms[-1] - terms[-2] to confirm the WHOLE list is consistent, not just
    the first pair, before asserting it equals the constructed d."""
    recomputed_d = terms[1] - terms[0]
    check(
        recomputed_d == terms[-1] - terms[-2],
        "term list is not arithmetic throughout (first-pair and last-pair differences disagree)",
    )
    check(recomputed_d == d, "recomputed d disagrees with the constructed d")
    check(terms[0] == a1, "first stated term is not a1")


@generator(
    "arithmetic_recursive_definition_state",
    level=11,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def arithmetic_recursive_definition_state(rng, variant="easy"):
    """Given the first several terms of an arithmetic sequence (never t1 or
    d labelled or stated), state its recursive definition: the first term
    and the value of d in t(n+1) = t(n) + d. The graded answer is a labelled
    pair {"first_term": a1, "d": d}."""
    if variant == "easy":
        k = 3
        d = rng.randint(1, 9)  # always positive: increasing sequence
        a1_choices = list(range(1, 21))  # always non-zero by range
        context = None
    elif variant == "medium":
        k = 4
        d = rng.choice([v for v in range(-9, 10) if v != 0])  # negative allowed
        a1_choices = [v for v in range(-20, 21) if v != 0]
        context = None
    elif variant == "hard":
        k = rng.choice([3, 4])
        d = rng.choice([v for v in range(-9, 10) if v != 0])
        a1_choices = [v for v in range(-20, 21) if v != 0]
        context = rng.choice(_ARITH_CONTEXTS)
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', or 'hard'")

    # adds-instead-of-subtracts's d-value (2*a1 + d) can algebraically
    # coincide with uses-magnitude-only-d's d-value (abs(d), when d < 0 --
    # exactly when a1 == -d) -- an accidental collision BETWEEN two
    # distractors, not a construction error. Resample a1 (keep d) up to 20
    # times on collision, exactly like the geometric generator's own
    # adds-instead-of-multiplies resample below.
    for _attempt in range(20):
        a1 = rng.choice(a1_choices)
        check(a1 != 0, "a1 must be non-zero (asserted directly, not just trusted from the draw range)")
        if d >= 0 or 2 * a1 + d != abs(d):
            break
    else:
        check(False, "could not resample a1 to avoid a distractor collision after 20 attempts")

    # terms[0] == a1; build k consecutive terms [a1, a1+d, a1+2d, ...].
    terms = []
    cur = a1
    for _ in range(k):
        terms.append(cur)
        cur += d

    _verify_arithmetic_definition(a1, d, terms)

    terms_str = ", ".join(str(t) for t in terms)
    if context:
        question_text = (
            f"{context.format(terms=terms_str)}. State the sequence's recursive "
            f"definition: the first term, and the value of d in t(n+1) = t(n) + d."
        )
    else:
        question_text = phrase(rng, _ARITH_ABSTRACT_TEMPLATES, terms=terms_str)

    answer_text = f"first_term = {a1}, d = {d}"

    # Misconception distractors.
    off_by_one = {"first_term": terms[1], "d": d}
    adds_instead = {"first_term": a1, "d": terms[0] + terms[1]}
    values = [(a1, d), (off_by_one["first_term"], off_by_one["d"]), (adds_instead["first_term"], adds_instead["d"])]
    candidate_dicts = [off_by_one, adds_instead]

    if d < 0:
        # Only offered when d < 0: collides with the answer when d >= 0.
        uses_magnitude_only = {"first_term": a1, "d": abs(d)}
        values.append((uses_magnitude_only["first_term"], uses_magnitude_only["d"]))
        candidate_dicts.append(uses_magnitude_only)

    check_pairwise_distinct(*values, msg="a distractor collides with another option or the answer")

    candidates = [f"first_term = {c['first_term']}, d = {c['d']}" for c in candidate_dicts]

    return dict(
        text=question_text,
        answer={"first_term": a1, "d": d},
        answer_text=answer_text,
        answer_latex=rf"\text{{first term}} = {a1},\; d = {d}",
        working=[
            f"d = term2 - term1 = {terms[1]} - {paren_negative(terms[0])} = {d}",
            f"Check against the last pair: {terms[-1]} - {paren_negative(terms[-2])} = {d}",
            f"first_term = {a1}",
        ],
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


# --- geometric_recursive_definition_state ---------------------------------


def _verify_geometric_definition(a1: int, r: int, terms: list[int]) -> None:
    """Construction builds `terms` by an iterative repeated-multiplication
    loop (cur *= r, k times, with terms[0] == a1). This check recomputes r
    from the STATED TERMS alone via exact integer division terms[1] //
    terms[0] (asserting the division is exact with % first) -- a genuinely
    different derivation -- cross-checked against terms[-1] // terms[-2]
    for consistency across the whole list, before asserting it equals the
    constructed r."""
    check(terms[1] % terms[0] == 0, "terms[1] is not exactly divisible by terms[0]")
    recomputed_r = terms[1] // terms[0]
    check(terms[-1] % terms[-2] == 0, "terms[-1] is not exactly divisible by terms[-2]")
    check(
        terms[-1] // terms[-2] == recomputed_r,
        "term list is not geometric throughout (first-pair and last-pair ratios disagree)",
    )
    check(recomputed_r == r, "recomputed r disagrees with the constructed r")
    check(terms[0] == a1, "first stated term is not a1")


@generator(
    "geometric_recursive_definition_state",
    level=11,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def geometric_recursive_definition_state(rng, variant="easy"):
    """Given the first several terms of a geometric sequence (never t1 or r
    labelled or stated), state its recursive definition: the first term and
    the value of r in t(n+1) = r * t(n). The graded answer is a labelled
    pair {"first_term": a1, "r": r}."""
    if variant == "easy":
        k = 3
        r = rng.choice([2, 3, 4])  # always positive growth
        a1_lo, a1_hi = 1, 9
        context = None
    elif variant == "medium":
        k = 4
        r = rng.choice([2, 3, 4, -2, -3, -4])  # negative/alternating allowed
        a1_lo, a1_hi = 1, 9
        context = None
    elif variant == "hard":
        k = rng.choice([3, 4])
        r = rng.choice([2, 3, 4, -2, -3, -4])
        a1_lo, a1_hi = 1, 9
        context = rng.choice(_GEOM_ALTERNATING_CONTEXTS if r < 0 else _GEOM_GROWTH_CONTEXTS)
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', or 'hard'")

    # adds-instead-of-multiplies (terms[1] - terms[0] = a1*(r-1)) can
    # algebraically coincide with r for specific (a1, r) pairs. Resample a1
    # (keep r) up to 20 times on collision, exactly like
    # p3c_recursion_build.py's resample pattern.
    for _attempt in range(20):
        a1 = rng.randint(a1_lo, a1_hi)
        if a1 * (r - 1) != r:
            break
    else:
        check(False, "could not resample a1 to avoid the adds-instead-of-multiplies collision after 20 attempts")

    # terms[0] == a1; build k consecutive terms [a1, a1*r, a1*r**2, ...].
    terms = []
    cur = a1
    for _ in range(k):
        terms.append(cur)
        cur *= r

    _verify_geometric_definition(a1, r, terms)

    terms_str = ", ".join(str(t) for t in terms)
    if context:
        question_text = (
            f"{context.format(terms=terms_str)}. State the sequence's recursive "
            f"definition: the first term, and the value of r in t(n+1) = r × t(n)."
        )
    else:
        question_text = phrase(rng, _GEOM_ABSTRACT_TEMPLATES, terms=terms_str)

    answer_text = f"first_term = {a1}, r = {r}"

    # Misconception distractors.
    off_by_one = {"first_term": terms[1], "r": r}
    adds_instead_r = terms[1] - terms[0]
    check(adds_instead_r != r, "adds-instead-of-multiplies distractor collides with the answer")
    adds_instead = {"first_term": a1, "r": adds_instead_r}
    values = [(a1, r), (off_by_one["first_term"], off_by_one["r"]), (adds_instead["first_term"], adds_instead["r"])]
    candidate_dicts = [off_by_one, adds_instead]

    if r < 0:
        # Only offered when r < 0: collides with the answer when r >= 0.
        uses_magnitude_only = {"first_term": a1, "r": abs(r)}
        values.append((uses_magnitude_only["first_term"], uses_magnitude_only["r"]))
        candidate_dicts.append(uses_magnitude_only)

    check_pairwise_distinct(*values, msg="a distractor collides with another option or the answer")

    candidates = [f"first_term = {c['first_term']}, r = {c['r']}" for c in candidate_dicts]

    return dict(
        text=question_text,
        answer={"first_term": a1, "r": r},
        answer_text=answer_text,
        answer_latex=rf"\text{{first term}} = {a1},\; r = {r}",
        working=[
            f"r = term2 ÷ term1 = {terms[1]} ÷ {paren_negative(terms[0])} = {r}",
            f"Check against the last pair: {terms[-1]} ÷ {paren_negative(terms[-2])} = {r}",
            f"first_term = {a1}",
        ],
        distractors=pick_distractors(answer_text, candidates, n=3),
    )
