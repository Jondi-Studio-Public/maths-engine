"""Generates `t1_simplify_surds` questions: simplify a surd or a sum of surds, such as sqrt(50) to 5*sqrt(2).

Simplify-surds.

an exam-style item asks students to simplify surds: sqrt(50) -> 5*sqrt(2),
sqrt(18) + sqrt(8) -> 5*sqrt(2).
branch) added `answer_form="simplified_surd"` specifically so this skill
could be built, and fixed the raw-text parsing traps for the unicode radical
("√50") and the bare-ASCII spelling ("5sqrt2"). It left one measured hole
open, re-confirmed by direct probe through mathgen.grading before writing this
module: **`50^(1/2)` (equivalently `50**(1/2)`) is ACCEPTED under
`simplified_surd`**, because the raw-text check
(mathgen/grading/typed.py::_simplified_surd_ok / _sqrt_call_radicands) only recognises a
literal `sqrt(` call -- it never looks at a `Pow` node with a 1/2 exponent.
`mathgen/grading/typed.py` is not changed by this generator, so every question's prompt text mitigates by naming the
required form explicitly: "Give your answer in the form a√b."

Scope is confined to NUMERIC radicands. Variables (`sqrt(sin(x))`, `sqrt(xy)`,
`sqrtx`) are mis-parsed by the same machinery and are deliberately
out of scope: a purely numeric radicand can never reach that
mis-parse path.

Every canonical `answer_text` is the plain ASCII `str()` of the sympy
expression (e.g. ``"5*sqrt(2)"``, ``"2*sqrt(5) + 4*sqrt(3)"``) -- never the
unicode "√" glyph -- exactly like `p6e_quadratic_formula_build.py` does
for its own surd answers, so the canonical always round-trips through the
real grader. The unicode radical is used only in the human-facing prompt
text, which is never parsed.
"""

from __future__ import annotations

from sympy import sqrt

from ..core import check_distinct, check_equivalent, generator, latex_answer, pick_distractors

# Fixed square-free base list.
SQUAREFREE = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23]


def largest_square_factor(n: int) -> tuple[int, int]:
    """Return ``(k, m)`` with ``n == k*k*m`` and ``m`` square-free, by trial
    division.

    Specific to this skill's own construction (drawing a radicand that is
    KNOWN to have a given extractable perfect-square factor) -- not a general
    answer-checking primitive like `check_equivalent`, so it lives here
    rather than in `mathgen/core.py`.
    """
    k = 1
    m = n
    i = 2
    while i * i <= m:
        while m % (i * i) == 0:
            m //= i * i
            k *= i
        i += 1
    return k, m


def _easy(rng):
    k = rng.randint(2, 9)
    m = rng.choice(SQUAREFREE)
    n = k * k * m

    answer = k * sqrt(m)
    answer_text = str(answer)
    text = f"Simplify √{n}. Give your answer in the form a√b."

    # Independent check, second route: re-derive from the radicand exactly as
    # the question states it (n), never from k/m directly.
    check_equivalent(answer, sqrt(n), "easy construction disagrees with sqrt(n)")

    # extracts-without-rooting: pulls the perfect-square factor out but
    # forgets to take its square root, so states k*k as the coefficient.
    extracts_without_rooting = k * k * sqrt(m)
    # drops-the-radical: states just k, dropping the remaining sqrt(m) entirely.
    drops_the_radical = k
    check_distinct(
        answer,
        extracts_without_rooting,
        drops_the_radical,
        msg="easy misconception distractor must be a genuinely different value",
    )

    distractors = pick_distractors(answer_text, [str(extracts_without_rooting), str(drops_the_radical)])
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, evaluate=False),
        distractors=distractors,
        answer_form="simplified_surd",
    )


def _medium(rng):
    b = rng.choice(SQUAREFREE)
    j1 = rng.randint(2, 6)
    j2 = rng.randint(2, 6)
    while j2 == j1:
        j2 = rng.randint(2, 6)
    r1 = j1 * j1 * b
    r2 = j2 * j2 * b

    answer = (j1 + j2) * sqrt(b)
    answer_text = str(answer)
    text = f"Simplify √{r1} + √{r2}. Give your answer in the form a√b."

    # Independent check, second route: reconstruct each surd term from the
    # RAW radicands the question states (r1, r2), never from j1/j2 directly.
    check_equivalent(answer, sqrt(r1) + sqrt(r2), "medium construction disagrees with the un-simplified sum")

    # adds-radicands: treats the two radicands as addable inside one root
    # before simplifying.
    adds_radicands = sqrt(r1 + r2)
    # forgets-to-combine: states only the larger of the two already-extracted
    # terms, never finishing the addition.
    forgets_to_combine = max(j1, j2) * sqrt(b)
    check_distinct(
        answer,
        adds_radicands,
        forgets_to_combine,
        msg="medium misconception distractor must be a genuinely different value",
    )

    distractors = pick_distractors(answer_text, [str(adds_radicands), str(forgets_to_combine)])
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, evaluate=False),
        distractors=distractors,
        answer_form="simplified_surd",
    )


def _hard(rng):
    b = rng.choice(SQUAREFREE)
    c1 = rng.randint(2, 5)
    c2 = rng.randint(2, 5)
    j1 = rng.randint(2, 6)
    j2 = rng.randint(2, 6)
    while j2 == j1:
        j2 = rng.randint(2, 6)

    while True:
        total = c1 * j1 - c2 * j2
        if total <= 0:
            c1, c2 = c2, c1
            j1, j2 = j2, j1
            total = c1 * j1 - c2 * j2
        if total == 0:
            # Resample the whole pair (never render a zero result).
            c1 = rng.randint(2, 5)
            c2 = rng.randint(2, 5)
            j1 = rng.randint(2, 6)
            j2 = rng.randint(2, 6)
            while j2 == j1:
                j2 = rng.randint(2, 6)
            continue
        break

    r1 = j1 * j1 * b
    r2 = j2 * j2 * b

    answer = total * sqrt(b)
    answer_text = str(answer)
    text = f"Simplify {c1}√{r1} - {c2}√{r2}. Give your answer in the form a√b."

    # Independent check, second route: reconstruct each term as its own
    # pre-existing coefficient times sqrt(the raw stated radicand), never
    # from j1/j2 directly.
    check_equivalent(
        answer, c1 * sqrt(r1) - c2 * sqrt(r2), "hard construction disagrees with the un-simplified difference"
    )

    # adds-instead-of-subtracts: wrong operation.
    adds_instead_of_subtracts = (c1 * j1 + c2 * j2) * sqrt(b)
    # forgot-existing-coefficient: extracts each j_i correctly but drops c1
    # (uses j1 unscaled instead of c1*j1).
    forgot_existing_coefficient = (j1 - c2 * j2) * sqrt(b)
    check_distinct(
        answer,
        adds_instead_of_subtracts,
        forgot_existing_coefficient,
        msg="hard misconception distractor must be a genuinely different value",
    )

    distractors = pick_distractors(answer_text, [str(adds_instead_of_subtracts), str(forgot_existing_coefficient)])
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, evaluate=False),
        distractors=distractors,
        answer_form="simplified_surd",
    )


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "t1_simplify_surds",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def t1_simplify_surds(rng, tier: str = "easy"):
    """Simplify a surd expression, three tiers of genuine reasoning escalation:

    easy   -- single surd, one extraction step (k*sqrt(m) from sqrt(k*k*m)).
    medium -- sum of two surds that combine into ONE term after simplification
              (both radicands need extracting; the learner must also add the
              extracted coefficients).
    hard   -- two terms, each carrying an explicit pre-existing coefficient,
              joined by subtraction (the learner must extract, multiply the
              extracted root by the term's own leading coefficient, then
              subtract, keeping track of sign).
    """
    try:
        fn = _TIERS[tier]
    except KeyError:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(sorted(_TIERS))}") from None
    return fn(rng)
