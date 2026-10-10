"""## Why `answer_form="single_fraction"`, not `"expanded"`

`grade_expression` compares by VALUE. This skill's canonical answer is necessarily a fraction with a
MULTI-TERM numerator over a common denominator (e.g. "(21x - 1)/6"), which needs a bracket that does not span
the whole string -- `"expanded"`'s bracket check would reject the canonical against ITSELF.
`"single_fraction"` rejects a raw-text binary "+"/"-" at bracket depth 0 -- the signature of an unmerged sum
of two fraction terms -- while accepting the canonical's own internal, depth>=1 "+"/"-" inside its numerator's
bracket.

## The forward-construction algorithm (never a redraw for correctness)

`k` fractions (2 for easy/medium/hard, 3 for exam), each `(p_i*x + q_i)/d_i`,
combined over `D = lcm(d_1, ..., d_k)` with multiplier `m_i = D // d_i`. The
FIRST fraction (index 0) is always added (its own connecting sign is fixed
"+"); every other fraction's connecting sign (`sign_i`) is drawn from
`{+1, -1}` on tiers that teach subtraction -- medium and hard fix it to `-1`
unconditionally (subtraction IS the escalation those two tiers teach, not a
coin flip: a 50/50 draw left `test_all_topics_usually_have_enough_distractors`
under its 85% two-distractor bar at this topic's default tier, since
forgot-to-distribute-the-sign is only exercisable when a subtraction is
actually present); exam alone draws each of its two connectors independently,
so "mix of +/-" (the reasoning-variety feature) lands on exam, not medium.
`p_i` is always a POSITIVE magnitude (the numerator's own x-coefficient never
carries a sign; only the CONNECTOR does -- this is what makes
"forgot-to-distribute-the-sign" meaningful: the whole bracketed numerator gets
negated, or fails to).

**x-term-never-cancels guarantee, by construction (no redraw loop)**: the first fraction's `p_1` is drawn from
`[6, 9]` and every other `p_i` from a small fixed magnitude (`{1, 2}` on easy/medium/hard, exactly `1` on
exam). `check(A != 0...)` is still asserted at runtime as a belt-and-braces confirmation, never as a redraw
trigger. `B` (the constant term) has no such guarantee and does not need one -- a numerator that is a bare
`A*x` (B == 0) is a fine, simpler, valid output.

**Independent check, second route**: `check_equivalent` compares the
constructed `Rational(A, D)*x + Rational(B, D)` against the sum of the
original `k` fractions built directly from `p_i, q_i, d_i, sign_i` as a
sympy sum (never from `A`/`B` arithmetic).

## Denominators, one shared idea across tiers

- **easy/medium** -- two DISTINCT PRIMES from `{2, 3, 5, 7}` (`rng.sample`,
  so coprime by construction, no gcd check needed): `D` is just the product,
  no genuine LCD reasoning.
- **hard/exam** -- `d_a = g*s`, `d_b = g*t` for `g` in `{2, 3}` and a
  coprime `(s, t)` pair from a curated list: `D = lcm(d_a, d_b) = g*s*t`,
  NOT the naive product `g^2*s*t` -- a genuinely different (larger) number,
  which is exactly what the `common-denominator-error` misconception
  exploits. exam adds a THIRD denominator from `{5, 7}` (prime, so coprime
  to every `g*s*t` above, whose only prime factors are 2 and 3): `D` grows
  to `g*s*t*d_c`.

## Misconceptions

- **common-denominator-error** -- states the answer over the PRODUCT of the
  denominators instead of the true LCD, keeping the correctly-scaled
  numerator (`A`, `B` unchanged) but the wrong (larger) denominator. Only
  offered on hard/exam, where the product genuinely exceeds the LCD
  (`P = g*D` on both tiers, `g >= 2`, so always distinct); on easy/medium
  the product IS the LCD, so this would be a coincidental duplicate of the
  correct answer and is honestly dropped there (never offered, never
  `check_distinct`-checked against a candidate that would fail).
- **forgot-to-distribute-the-sign** -- on a term with connecting sign `-1`,
  applies the sign to the x-term but not the constant (keeps `+m_i*q_i`
  instead of `-m_i*q_i`). Only offered when at least one term's sign is
  `-1` in that specific draw (never exercisable on easy, which is
  addition-only); exam picks the FIRST such term if more than one qualifies.
- **added-denominators** -- states the answer over the SUM of the
  denominators (`d_1 + d_2 [+ d_3]`) instead of `D`, keeping `A`, `B`
  unchanged. Offered unconditionally on every tier: verified (see result
  file) that `sum(d_i) != D` for every denominator combination this module
  can draw, across all four tiers.

Every distractor is formatted through the same `_frac_text` helper as the
canonical answer -- one bracketed (or bare, when `B == 0`) numerator over
one denominator -- so a distractor never fails the `single_fraction` raw
check for the wrong reason (looking like an unmerged sum); it fails only on
value, exactly like the canonical answer's own shape.
"""

from __future__ import annotations

from sympy import Rational, symbols

from ..core import check, check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors

x = symbols("x")

# --- shared denominator/term construction -----------------------------------

#: easy/medium: two distinct primes -- coprime by construction (`rng.sample`
#: never repeats), so D is just the product; no LCD reasoning needed yet.
PRIMES = (2, 3, 5, 7)

#: hard/exam: a shared factor g times a coprime (s, t) pair, so
#: D = lcm(g*s, g*t) = g*s*t -- NOT the naive product g^2*s*t (the gap the
#: common-denominator-error misconception exploits). s < t always, so the
#: term built from g*s (the SMALLER denominator) always gets the LARGER
#: multiplier t -- that term is assigned p_1's big range, see _hard/_exam.
G = (2, 3)
ST_PAIRS = ((1, 2), (1, 3), (2, 3), (1, 4), (3, 4))

#: exam's third denominator -- prime, so always coprime to g*s*t above
#: (whose only prime factors are 2 and 3): D grows to exactly g*s*t*d_c,
#: never a smaller lcm, which keeps the per-draw arithmetic
#: exhaustively checkable.
THIRD_DENOMS = (5, 7)

P1_LO, P1_HI = 6, 9
#: every OTHER fraction's x-coefficient magnitude on easy/medium/hard.
P_OTHER = (1, 2)

#: any nonzero constant term, medium/hard/exam ("negative q_i allowed").
NONZERO = tuple(v for v in range(-9, 10) if v != 0)


def _term_text(p: int, q: int, d: int) -> str:
    """One fraction's own display text: a bracketed two-term numerator (the
    x-coefficient is always positive -- only the CONNECTOR carries a sign,
    see the module docstring) over its own denominator, e.g. '(4x + 3)/5'."""
    numerator = join_terms([(p, "x"), (q, "")])
    return f"({numerator})/{d}"


def _expr_text(terms) -> str:
    """The question's own literal unmerged-sum text -- exactly what a
    learner reads, and exactly the text the 'own wrong form is rejected'
    proof (this module's test file) feeds straight back to the real
    grader, no reconstruction needed."""
    p0, q0, d0, _m0, _s0 = terms[0]
    parts = [_term_text(p0, q0, d0)]
    for p, q, d, _m, sign in terms[1:]:
        parts.append((" + " if sign == 1 else " - ") + _term_text(p, q, d))
    return "".join(parts)


def _literal_sum(terms):
    """The k original fractions summed directly from p_i, q_i, d_i, sign_i
    -- the independent second route `check_equivalent` compares against,
    never built from A/B."""
    total = 0
    for p, q, d, _m, sign in terms:
        total += sign * (p * x + q) / d
    return total


def _combine(terms) -> tuple[int, int]:
    A = sum(sign * m * p for p, q, d, m, sign in terms)
    B = sum(sign * m * q for p, q, d, m, sign in terms)
    return A, B


def _frac_text(A: int, B: int, D: int) -> str:
    """The canonical/distractor shared shape: a bracketed two-term numerator
    over D, or -- when B == 0, a fine and simpler valid output, never
    special-cased away -- a bare single-term numerator with no bracket at
    all (accepted by `single_fraction` exactly like the
    '4*x/9' example)."""
    numerator = join_terms([(A, "x"), (B, "")])
    return f"({numerator})/{D}" if B != 0 else f"{numerator}/{D}"


def _frac_expr(A: int, B: int, D: int):
    return Rational(A, D) * x + Rational(B, D)


def _forgot_sign(terms, bug_index: int) -> tuple[int, int]:
    """The forgot-to-distribute-the-sign misconception: at `bug_index`
    (a term whose OWN connecting sign is -1), the x-term keeps its correct
    sign but the constant term does not -- `+m_i*q_i` instead of
    `-m_i*q_i`. A genuinely different B whenever q_i != 0 (guaranteed:
    every term's q_i is drawn nonzero on every tier this is offered on)."""
    A = 0
    B = 0
    for idx, (p, q, d, m, sign) in enumerate(terms):
        A += sign * m * p
        B += m * q if idx == bug_index else sign * m * q
    return A, B


def _sentence(rng, expr_text: str) -> str:
    return phrase(rng, _SENTENCE_PATTERNS, expr=expr_text)


# The prompt-parsing in this module's own test file keys off the literal word "Simplify" at the
# start of every question and these three exact suffixes.
_SENTENCE_PATTERNS = (
    "Simplify {expr}.",
    "Simplify {expr}, giving your answer as a single fraction.",
    "Simplify {expr} to a single fraction.",
)


# --- easy: two coprime denominators, addition only, positive q_i -----------


def _easy(rng):
    d1, d2 = sorted(rng.sample(PRIMES, 2))
    D = d1 * d2
    m1, m2 = d2, d1
    p1 = rng.randint(P1_LO, P1_HI)
    p2 = rng.choice(P_OTHER)
    q1 = rng.randint(1, 9)
    q2 = rng.randint(1, 9)
    terms = [(p1, q1, d1, m1, 1), (p2, q2, d2, m2, 1)]

    A, B = _combine(terms)
    check(A != 0, "easy: x-coefficient cancelled to zero -- construction bug")
    check_equivalent(
        _literal_sum(terms), _frac_expr(A, B, D), "easy: literal sum does not reduce to the constructed answer"
    )

    answer_expr = _frac_expr(A, B, D)
    answer_text = _frac_text(A, B, D)
    text = _sentence(rng, _expr_text(terms))

    added_den = d1 + d2
    added_expr = _frac_expr(A, B, added_den)
    added_text = _frac_text(A, B, added_den)
    check_distinct(
        answer_expr, added_expr, msg="easy: added-denominators distractor must be a genuinely different value"
    )

    return text, answer_expr, answer_text, [added_text]


# --- medium: two coprime denominators, subtraction, negative q_i ----------


def _medium(rng):
    d1, d2 = sorted(rng.sample(PRIMES, 2))
    D = d1 * d2
    m1, m2 = d2, d1
    p1 = rng.randint(P1_LO, P1_HI)
    p2 = rng.choice(P_OTHER)
    q1 = rng.choice(NONZERO)
    q2 = rng.choice(NONZERO)
    # medium's own defining escalation over easy IS subtraction -- always
    # present, not a coin flip, so forgot-to-distribute-the-sign is
    # exercisable on every question, not merely "sometimes" (measured: a
    # 50/50 draw here left `test_all_topics_usually_have_enough_distractors`
    # under its 85% two-distractor bar at this topic's default tier).
    sign2 = -1
    terms = [(p1, q1, d1, m1, 1), (p2, q2, d2, m2, sign2)]

    A, B = _combine(terms)
    check(A != 0, "medium: x-coefficient cancelled to zero -- construction bug")
    check_equivalent(
        _literal_sum(terms), _frac_expr(A, B, D), "medium: literal sum does not reduce to the constructed answer"
    )

    answer_expr = _frac_expr(A, B, D)
    answer_text = _frac_text(A, B, D)
    text = _sentence(rng, _expr_text(terms))

    candidates = []
    added_den = d1 + d2
    added_expr = _frac_expr(A, B, added_den)
    added_text = _frac_text(A, B, added_den)
    check_distinct(
        answer_expr, added_expr, msg="medium: added-denominators distractor must be a genuinely different value"
    )
    candidates.append(added_text)

    Ab, Bb = _forgot_sign(terms, bug_index=1)
    bug_expr = _frac_expr(Ab, Bb, D)
    bug_text = _frac_text(Ab, Bb, D)
    check_distinct(
        answer_expr,
        bug_expr,
        msg="medium: forgot-to-distribute-the-sign distractor must be a genuinely different value",
    )
    candidates.append(bug_text)

    return text, answer_expr, answer_text, candidates


# --- hard: shared-factor denominators (genuine LCD), subtraction ----------


def _hard(rng):
    g = rng.choice(G)
    s, t = rng.choice(ST_PAIRS)
    d1, d2 = g * s, g * t
    D = g * s * t
    m1, m2 = t, s
    p1 = rng.randint(P1_LO, P1_HI)
    p2 = rng.choice(P_OTHER)
    q1 = rng.choice(NONZERO)
    q2 = rng.choice(NONZERO)
    # hard carries medium's subtraction forward (always present, see
    # medium's own comment) and adds the genuine-LCD denominator step.
    sign2 = -1
    terms = [(p1, q1, d1, m1, 1), (p2, q2, d2, m2, sign2)]

    A, B = _combine(terms)
    check(A != 0, "hard: x-coefficient cancelled to zero -- construction bug")
    check_equivalent(
        _literal_sum(terms), _frac_expr(A, B, D), "hard: literal sum does not reduce to the constructed answer"
    )

    answer_expr = _frac_expr(A, B, D)
    answer_text = _frac_text(A, B, D)
    text = _sentence(rng, _expr_text(terms))

    candidates = []
    added_den = d1 + d2
    added_expr = _frac_expr(A, B, added_den)
    added_text = _frac_text(A, B, added_den)
    check_distinct(
        answer_expr, added_expr, msg="hard: added-denominators distractor must be a genuinely different value"
    )
    candidates.append(added_text)

    P = d1 * d2
    check(P != D, "hard: product denominator coincides with the true LCD -- construction bug")
    cde_expr = _frac_expr(A, B, P)
    cde_text = _frac_text(A, B, P)
    check_distinct(
        answer_expr, cde_expr, msg="hard: common-denominator-error distractor must be a genuinely different value"
    )
    candidates.append(cde_text)

    Ab, Bb = _forgot_sign(terms, bug_index=1)
    bug_expr = _frac_expr(Ab, Bb, D)
    bug_text = _frac_text(Ab, Bb, D)
    check_distinct(
        answer_expr, bug_expr, msg="hard: forgot-to-distribute-the-sign distractor must be a genuinely different value"
    )
    candidates.append(bug_text)

    return text, answer_expr, answer_text, candidates


# --- exam: three fractions, at least one shared-factor pair ---------------


def _exam(rng):
    g = rng.choice(G)
    s, t = rng.choice(ST_PAIRS)
    d_pair = g * s * t
    d_dom = rng.choice(THIRD_DENOMS)
    D = d_pair * d_dom
    d_b, d_c = g * s, g * t
    m_dom, m_b, m_c = d_pair, t * d_dom, s * d_dom

    p1 = rng.randint(P1_LO, P1_HI)
    p2 = p3 = 1
    q1 = rng.choice(NONZERO)
    q2 = rng.choice(NONZERO)
    q3 = rng.choice(NONZERO)
    sign2 = rng.choice((1, -1))
    sign3 = rng.choice((1, -1))
    terms = [(p1, q1, d_dom, m_dom, 1), (p2, q2, d_b, m_b, sign2), (p3, q3, d_c, m_c, sign3)]

    A, B = _combine(terms)
    check(A != 0, "exam: x-coefficient cancelled to zero -- construction bug")
    check_equivalent(
        _literal_sum(terms), _frac_expr(A, B, D), "exam: literal sum does not reduce to the constructed answer"
    )

    answer_expr = _frac_expr(A, B, D)
    answer_text = _frac_text(A, B, D)
    text = _sentence(rng, _expr_text(terms))

    candidates = []
    added_den = d_dom + d_b + d_c
    added_expr = _frac_expr(A, B, added_den)
    added_text = _frac_text(A, B, added_den)
    check_distinct(
        answer_expr, added_expr, msg="exam: added-denominators distractor must be a genuinely different value"
    )
    candidates.append(added_text)

    P = d_dom * d_b * d_c
    check(P != D, "exam: product denominator coincides with the true LCD -- construction bug")
    cde_expr = _frac_expr(A, B, P)
    cde_text = _frac_text(A, B, P)
    check_distinct(
        answer_expr, cde_expr, msg="exam: common-denominator-error distractor must be a genuinely different value"
    )
    candidates.append(cde_text)

    bug_index = None
    if sign2 == -1:
        bug_index = 1
    elif sign3 == -1:
        bug_index = 2
    if bug_index is not None:
        Ab, Bb = _forgot_sign(terms, bug_index)
        bug_expr = _frac_expr(Ab, Bb, D)
        bug_text = _frac_text(Ab, Bb, D)
        check_distinct(
            answer_expr,
            bug_expr,
            msg="exam: forgot-to-distribute-the-sign distractor must be a genuinely different value",
        )
        candidates.append(bug_text)

    return text, answer_expr, answer_text, candidates


@generator(
    "exd_e2_algebraic_fraction_add_sub",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e2_algebraic_fraction_add_sub(rng, tier: str = "medium"):
    if tier == "easy":
        text, answer_expr, answer_text, cands = _easy(rng)
    elif tier == "medium":
        text, answer_expr, answer_text, cands = _medium(rng)
    elif tier == "hard":
        text, answer_expr, answer_text, cands = _hard(rng)
    elif tier == "exam":
        text, answer_expr, answer_text, cands = _exam(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="single_fraction",
    )
