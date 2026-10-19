"""Discrete recurrence + ODE characteristic roots.

Two skills, one module:

- ``p12e_recurrence_two_step`` (``ucv1-recurrence-two-step``, ``answer:
  number``) -- evaluate/invert a second-order linear recurrence
  ``a_n = p*a_{n-1} + q*a_{n-2}`` from two stated initial terms.
- ``p12e_characteristic_roots`` (``ucv1-second-order-characteristic-roots``,
  ``answer: pair``, labelled) -- the characteristic roots of
  ``y'' + b*y' + c*y = 0``, i.e. the roots of ``r^2 + b*r + c = 0``.

Both topics are brand new; neither shares a kernel with
``generate-sequence-from-rule`` (closed-form position substitution) or
``senior-use-recursion-to-generate-an-arithmetic/geometric-sequence``
(first-order recursion, answer is a LIST of the next k terms).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Eq, I, solve, sqrt, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    paren_negative,
    phrase,
    pick_distractors,
)

_MAX_ATTEMPTS = 500

# ---------------------------------------------------------------------------
# Skill 1: ucv1-recurrence-two-step
# ---------------------------------------------------------------------------

_P_RANGE = tuple(range(2, 6))  # 2..5
_Q_RANGE = tuple(v for v in range(-3, 4) if v != 0)  # -3..-1, 1..3
_TERM_RANGE = tuple(range(-6, 7))  # a1/a2 initial-term pool, -6..6

# Exam tier: a wider adversarial q pool (|q| > p is always true here since p tops out at 5 and this
# pool starts at 6) -- p itself stays positive throughout every tier, including exam, because the
# skill's design fixes p in 2..5 for every source; "both negative" (another possible adversarial shape)
# would require relaxing that, which is not what the design
# asks for.
_EXAM_Q_RANGE = tuple(v for v in range(-9, 10) if abs(v) > 5)


def _forward_step(p, q, prev1, prev2):
    """a_n = p*a_{n-1} + q*a_{n-2}. Shared by construction's iterative loop
    AND the hard/exam tiers' "solve, then verify by substituting back" check
    -- deliberately the SAME method on that side."""
    return p * prev1 + q * prev2


def _independent_forward_term(p, q, a1, a2, n):
    """Recompute a_n from scratch via a freshly-written, list-based unroll --
    not the sliding-window state construction's loop uses -- the independent
    route for the FORWARD tiers (easy/medium/exam's forward half)."""
    terms = [None, a1, a2]
    for i in range(3, n + 1):
        terms.append(terms[i - 1] * p + terms[i - 2] * q)
    return terms[n]


def _draw_forward(rng, p_pool=_P_RANGE, q_pool=_Q_RANGE, term_pool=_TERM_RANGE):
    p = rng.choice(p_pool)
    q = rng.choice(q_pool)
    a1 = rng.choice(term_pool)
    a2 = rng.choice(term_pool)
    return p, q, a1, a2


def _fmt_int(value) -> str:
    return str(int(value))


def _recurrence_expr(p, q) -> str:
    """ "a_n = ..." right-hand side, student-facing (no literal asterisk, no
    "- -" chain) -- e.g. "2a_(n-1) - 3a_(n-2)" or "a_(n-1) + a_(n-2)"."""
    return join_terms([(p, "a_(n-1)"), (q, "a_(n-2)")])


_FORWARD_PATTERNS = [
    "A sequence satisfies a_n = {expr} for n >= 3, with a_1 = {a1} and a_2 = {a2}. Find a_{target}.",
    "A sequence is defined by a_n = {expr} for n >= 3. Given a_1 = {a1} and a_2 = {a2}, find a_{target}.",
    "For the recurrence a_n = {expr} (n >= 3) with initial terms a_1 = {a1} and a_2 = {a2}, find a_{target}.",
]

_INVERSE_PATTERNS = [
    "A sequence satisfies a_n = {expr} for n >= 3. a_2 = {a2} and a_3 = {a3}. Find a_1.",
    "For the recurrence a_n = {expr} (n >= 3), a_2 = {a2} and a_3 = {a3}. What is a_1?",
    "A sequence follows a_n = {expr} for n >= 3, where a_2 = {a2} and a_3 = {a3}. Determine a_1.",
]

_EXAM_RECURRENCE_PATTERNS = [
    "A sequence satisfies a_n = p a_(n-1) + q a_(n-2) for n >= 3, with a_1 = {a1}, "
    "a_2 = {a2}, a_3 = {a3}. Exactly one of these (p, q) pairs is consistent with all "
    "three terms: Pair A: p = {pa}, q = {qa}. Pair B: p = {pb}, q = {qb}. Using the "
    "consistent pair, find a_4.",
    "For the recurrence a_n = p a_(n-1) + q a_(n-2) (n >= 3), the terms a_1 = {a1}, "
    "a_2 = {a2} and a_3 = {a3} are given. Only one of these (p, q) pairs fits all "
    "three terms: Pair A: p = {pa}, q = {qa}. Pair B: p = {pb}, q = {qb}. Use the "
    "correct pair to find a_4.",
    "A sequence obeys a_n = p a_(n-1) + q a_(n-2) for n >= 3. Given a_1 = {a1}, "
    "a_2 = {a2}, a_3 = {a3}, exactly one candidate pair below is consistent: "
    "Pair A: p = {pa}, q = {qa}. Pair B: p = {pb}, q = {qb}. Find a_4 using the "
    "consistent pair.",
]


def _offer_candidate(answer, value, candidates, label):
    """Append `value` to `candidates` iff it differs from `answer` and from
    every candidate already offered -- two different misconceptions can land
    on the same wrong number, and only one of them should ever be shown.
    Every candidate actually offered is proven distinct from the answer AND
    pairwise distinct from every other offered candidate, unconditionally."""
    if value == answer or any(value == existing for existing in candidates):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    check_pairwise_distinct(answer, *candidates, value, msg=f"{label} collided with another candidate")
    candidates.append(value)


def _p12e_recurrence_forward(rng, *, target_index):
    """easy (target_index=3) / medium (target_index=4): forward evaluation,
    chained for medium (a3 must be held as an intermediate result and reused
    to reach a4 -- a genuine reasoning escalation over easy's single
    substitution, not a bigger number)."""
    for _attempt in range(_MAX_ATTEMPTS):
        p, q, a1, a2 = _draw_forward(rng)
        prev2, prev1 = a1, a2
        for _ in range(target_index - 2):
            nxt = _forward_step(p, q, prev1, prev2)
            prev2, prev1 = prev1, nxt
        answer = prev1  # a_{target_index}

        # Independent check: a freshly-written unroll, not construction's
        # sliding window.
        check_equivalent(
            _independent_forward_term(p, q, a1, a2, target_index),
            answer,
            f"forward target_index={target_index}: independent check failed",
        )

        immediate_prev = prev2  # a_{target_index - 1}
        # Misconception candidates (only offered when they genuinely differ
        # from the answer; check_distinct/check_pairwise_distinct then prove
        # that of every candidate actually offered):
        candidates = []

        # off-by-one-index: returns a_{target_index-1} instead of a_{target_index}
        _offer_candidate(answer, immediate_prev, candidates, "off-by-one-index")

        # swaps-coefficients: q*a_{n-1} + p*a_{n-2}
        # Recover the two terms feeding the final step (a_{n-1}, a_{n-2})
        # independently of the construction loop's state.
        term_n_minus_1 = prev2
        term_n_minus_2 = _independent_forward_term(p, q, a1, a2, target_index - 2) if target_index > 3 else a1
        swap = q * term_n_minus_1 + p * term_n_minus_2
        _offer_candidate(answer, swap, candidates, "swaps-coefficients")

        # drops-one-term: p*a_{n-1} only, ignoring q*a_{n-2}
        drop = p * term_n_minus_1
        _offer_candidate(answer, drop, candidates, "drops-one-term")

        # sign-flip: only exercisable when q < 0 (forgets the negative sign)
        if q < 0:
            sign_flip = p * term_n_minus_1 + (-q) * term_n_minus_2
            _offer_candidate(answer, sign_flip, candidates, "sign-flip")

        if len(candidates) == 0:
            continue  # no headroom at all this draw -- resample
        break
    else:
        raise RuntimeError(f"forward target_index={target_index}: could not draw a usable question")

    answer_text = _fmt_int(answer)
    distractors = pick_distractors(answer_text, [_fmt_int(c) for c in candidates])

    expr = _recurrence_expr(p, q)
    text = phrase(rng, _FORWARD_PATTERNS, expr=expr, a1=a1, a2=a2, target=target_index)

    if target_index == 3:
        working = [
            f"a_3 = {paren_negative(p)} × {paren_negative(a2)} + {paren_negative(q)} × {paren_negative(a1)} = {answer}."
        ]
    else:
        a3_value = _independent_forward_term(p, q, a1, a2, 3)
        working = [
            f"a_3 = {paren_negative(p)} × {paren_negative(a2)} + {paren_negative(q)} × {paren_negative(a1)} = {a3_value}.",
            f"a_4 = {paren_negative(p)} × {paren_negative(a3_value)} + {paren_negative(q)} × {paren_negative(a2)} = {answer}.",
        ]

    return dict(
        text=text,
        latex=text,
        answer=int(answer),
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


def _p12e_recurrence_inverse(rng, *, q_pool=_Q_RANGE, three_terms=False):
    """hard: given a_2, a_3, p, q, find a_1 (solve a_3 = p*a_2 + q*a_1).
    exam (three_terms=True): given a_1, a_2, a_3 and TWO candidate (p, q)
    pairs, identify the consistent pair and use it to find a_4 -- a genuine
    escalation in KIND over hard: a verification step before the
    computation, not a longer forward chain and not just bigger numbers."""
    for _attempt in range(_MAX_ATTEMPTS):
        p, q, a1, a2 = _draw_forward(rng, q_pool=q_pool)
        a3 = _forward_step(p, q, a2, a1)

        if not three_terms:
            answer = a1
            # Independent check: forward-simulate a1 -> a2 -> a3 using the
            # SAME iterative method construction itself uses, then assert it
            # lands on the stated a3 -- the "solve, then verify by
            # substituting back" pattern (xy_solution's own discipline).
            check(_forward_step(p, q, a2, answer) == a3, "hard: solved a1 does not reproduce a3 by substitution")

            candidates = []
            # off-by-one-index: returns a2 instead of a1
            _offer_candidate(answer, a2, candidates, "off-by-one-index")
            # swaps-coefficients: solve as if a3 = q*a2 + p*a1 (only offered
            # when the resulting a1 is an exact integer, so it never breaks
            # the anti-shape-tell rule against the always-integer answer)
            swap_frac = Fraction(a3 - q * a2, p)
            if swap_frac.denominator == 1:
                _offer_candidate(answer, int(swap_frac), candidates, "swaps-coefficients")
            # drops-one-term: treats a3 as q*a1 alone (ignores p*a2), only
            # offered when exact
            drop_frac = Fraction(a3, q)
            if drop_frac.denominator == 1:
                _offer_candidate(answer, int(drop_frac), candidates, "drops-one-term")
            # sign-flip: only exercisable when q < 0, and only offered when exact
            if q < 0:
                flip_frac = Fraction(a3 - p * a2, -q)
                if flip_frac.denominator == 1:
                    _offer_candidate(answer, int(flip_frac), candidates, "sign-flip")

            if len(candidates) == 0:
                continue
            answer_text = _fmt_int(answer)
            distractors = pick_distractors(answer_text, [_fmt_int(c) for c in candidates])

            text = phrase(rng, _INVERSE_PATTERNS, expr=_recurrence_expr(p, q), a2=a2, a3=a3)
            working = [
                f"a_3 = {paren_negative(p)} × a_2 + {paren_negative(q)} × a_1, so {a3} = {paren_negative(p)} × {paren_negative(a2)} + {paren_negative(q)} × a_1.",
                f"a_1 = ({paren_negative(a3)} - {paren_negative(p)} × {paren_negative(a2)}) / {paren_negative(q)} = {answer}.",
            ]
            return dict(
                text=text,
                latex=text,
                answer=int(answer),
                answer_text=answer_text,
                working=working,
                distractors=distractors,
            )

        # exam: three supplied terms + a wrong-pair check, then compute a4.
        a4 = _forward_step(p, q, a3, a2)
        p_wrong = rng.choice(_P_RANGE)
        q_wrong = rng.choice(q_pool)
        if _forward_step(p_wrong, q_wrong, a2, a1) == a3:
            continue  # accidentally consistent -- not a genuine wrong pair, resample

        answer = a4
        # Independent check: freshly-written unroll from a1 (not construction's
        # loop), confirming a4 under the CORRECT pair.
        check_equivalent(_independent_forward_term(p, q, a1, a2, 4), answer, "exam: independent check on a4 failed")
        # And confirm the wrong pair really is inconsistent with the three
        # given terms (the fact the learner must actually verify).
        check(_forward_step(p_wrong, q_wrong, a2, a1) != a3, "exam: wrong pair unexpectedly reproduces a3")

        candidates = []
        # uses-wrong-pair: applies the inconsistent pair forward instead
        wrong_pair_answer = _forward_step(p_wrong, q_wrong, a3, a2)
        _offer_candidate(answer, wrong_pair_answer, candidates, "uses-wrong-pair")
        # off-by-one-index: returns a3 instead of a4
        _offer_candidate(answer, a3, candidates, "off-by-one-index")
        # swaps-coefficients (correct pair, forward): q*a3 + p*a2
        swap = q * a3 + p * a2
        _offer_candidate(answer, swap, candidates, "swaps-coefficients")
        # drops-one-term: p*a3 only
        drop = p * a3
        _offer_candidate(answer, drop, candidates, "drops-one-term")
        # sign-flip: only when q < 0
        if q < 0:
            sign_flip = p * a3 + (-q) * a2
            _offer_candidate(answer, sign_flip, candidates, "sign-flip")

        if len(candidates) == 0:
            continue
        answer_text = _fmt_int(answer)
        distractors = pick_distractors(answer_text, [_fmt_int(c) for c in candidates])

        pair_a, pair_b = (p, q), (p_wrong, q_wrong)
        if rng.random() < 0.5:
            pair_a, pair_b = pair_b, pair_a
        text = phrase(
            rng,
            _EXAM_RECURRENCE_PATTERNS,
            a1=a1,
            a2=a2,
            a3=a3,
            pa=pair_a[0],
            qa=pair_a[1],
            pb=pair_b[0],
            qb=pair_b[1],
        )
        working = [
            f"Check each pair against a_3 = p × a_2 + q × a_1: only p = {p}, q = {q} reproduces a_3 = {a3}.",
            f"a_4 = {paren_negative(p)} × {paren_negative(a3)} + {paren_negative(q)} × {paren_negative(a2)} = {answer}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=int(answer),
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("inverse: could not draw a usable question")


@generator(
    "p12e_recurrence_two_step",
    level=14,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12e_recurrence_two_step(rng, tier="easy"):
    if tier == "easy":
        return _p12e_recurrence_forward(rng, target_index=3)
    if tier == "medium":
        return _p12e_recurrence_forward(rng, target_index=4)
    if tier == "hard":
        return _p12e_recurrence_inverse(rng, q_pool=_Q_RANGE, three_terms=False)
    if tier == "exam":
        return _p12e_recurrence_inverse(rng, q_pool=_EXAM_Q_RANGE, three_terms=True)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Skill 2: ucv1-second-order-characteristic-roots
# ---------------------------------------------------------------------------

_ROOT_POOL = tuple(v for v in range(-16, 17) if v != 0)
_K_POOL = tuple(range(1, 7))  # imaginary part magnitude, always drawn positive
_EXAM_ROOT_POOL = tuple(v for v in range(-15, 16) if v != 0)


def _independent_roots_from_bc(b, c):
    """Solve r^2 + b*r + c = 0 directly via the quadratic formula in sympy,
    from b and c ALONE -- independent of however construction derived b/c
    from its chosen roots. sympy's sqrt of a negative literal evaluates to an
    imaginary number automatically, so this one function covers the real,
    repeated and complex tiers alike."""
    disc = b * b - 4 * c
    root_disc = sqrt(disc)
    return (-b + root_disc) / 2, (-b - root_disc) / 2


def _independent_bc_from_roots(r1, r2):
    """exam tier only: from the two ROOTS, solve the 2x2 linear system
    {r1^2 + b*r1 + c = 0, r2^2 + b*r2 + c = 0} for (b, c) directly -- a
    genuinely different route from Vieta's shortcut (construction's own
    b = -(r1+r2), c = r1*r2), since it never pattern-matches sum/product at
    all."""
    b_sym, c_sym = symbols("b_sym c_sym")
    eqs = [Eq(r1**2 + b_sym * r1 + c_sym, 0), Eq(r2**2 + b_sym * r2 + c_sym, 0)]
    sol = solve(eqs, [b_sym, c_sym])
    return sol[b_sym], sol[c_sym]


def _fmt_pair(label1, value1, label2, value2) -> str:
    return f"{label1} = {_fmt_int(value1)}, {label2} = {_fmt_int(value2)}"


def _ode_expr(b, c) -> str:
    """ "y'' + ..." left-hand side with NUMERIC b, c, student-facing (no
    literal asterisk, no "- -" chain)."""
    return join_terms([(1, "y''"), (b, "y'"), (c, "y")])


def _char_expr(b, c) -> str:
    """ "r^2 + ..." characteristic-equation left-hand side with NUMERIC b, c."""
    return join_terms([(1, "r^2"), (b, "r"), (c, "")])


_EASY_ROOTS_PATTERNS = [
    "{ode} = 0. Let r1 be the smaller root and r2 the larger root (r1 <= r2) of "
    "the characteristic equation {char} = 0. Find r1 and r2.",
    "The homogeneous ODE {ode} = 0 has characteristic equation {char} = 0 with "
    "two distinct real roots. Letting r1 <= r2 be the smaller and larger root, "
    "find r1 and r2.",
    "Solve the characteristic equation {char} = 0 (from {ode} = 0) for its two "
    "real roots, reporting r1 (the smaller root) and r2 (the larger root).",
]

_MEDIUM_ROOTS_PATTERNS = [
    "{ode} = 0. The characteristic equation {char} = 0 has a repeated root. Let "
    "r1 be the smaller root and r2 the larger root (here r1 = r2). Find r1 and r2.",
    "The ODE {ode} = 0 has characteristic equation {char} = 0 with a repeated "
    "real root. Report this value as both r1 and r2.",
    "Solve {char} = 0, the characteristic equation of {ode} = 0, given that it "
    "has exactly one (repeated) real root. Find r1 and r2 (equal here).",
]

_HARD_ROOTS_PATTERNS = [
    "{ode} = 0. The characteristic equation {char} = 0 has complex conjugate "
    "roots a + ki and a - ki, where k is the coefficient of i in the root "
    "with the POSITIVE imaginary part. Find a (real_part) and k (imaginary_part).",
    "The characteristic equation {char} = 0 of {ode} = 0 has complex conjugate "
    "roots. Writing them as a + ki and a - ki with k > 0, find the real part a "
    "and the imaginary part k.",
    "Solve {char} = 0 (the characteristic equation of {ode} = 0) for its complex "
    "conjugate roots a ± ki, taking k to be positive, and report a (real_part) "
    "and k (imaginary_part).",
]

_EXAM_ROOTS_PATTERNS = [
    "A homogeneous linear second-order ODE y'' + b*y' + c*y = 0 has "
    "characteristic-equation roots r1 = {r1} and r2 = {r2}. Find b and c.",
    "The characteristic equation of a homogeneous second-order ODE "
    "y'' + b*y' + c*y = 0 has roots r1 = {r1} and r2 = {r2}. Determine b and c.",
    "A second-order constant-coefficient ODE y'' + b*y' + c*y = 0 has "
    "characteristic roots r1 = {r1} and r2 = {r2}. Recover b and c.",
]


def _p12e_characteristic_roots_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        r1, r2 = rng.sample(_ROOT_POOL, 2)
        if r1 > r2:
            r1, r2 = r2, r1
        b, c = -(r1 + r2), r1 * r2

        hi, lo = _independent_roots_from_bc(b, c)
        check_equivalent(hi, r2, "easy: independent root check failed (larger)")
        check_equivalent(lo, r1, "easy: independent root check failed (smaller)")

        answer = {"r1": r1, "r2": r2}
        answer_text = _fmt_pair("r1", r1, "r2", r2)

        candidates = []
        offered_tuples = [(r1, r2)]
        # sign-flip-on-b: negates the roots
        sf = (-r2, -r1)  # keep smaller/larger order after negation
        if sf not in offered_tuples:
            check_distinct((r1, r2), sf, msg="sign-flip-on-b collided")
            offered_tuples.append(sf)
            candidates.append(_fmt_pair("r1", sf[0], "r2", sf[1]))
        # drops-discriminant-sqrt: offers -b/2 for BOTH labels (only valid to
        # offer on easy/hard, never medium, where it collides with the truth)
        half = Fraction(-b, 2)
        if half.denominator == 1:
            hp = (int(half), int(half))
            if hp not in offered_tuples:
                check_distinct((r1, r2), hp, msg="drops-discriminant-sqrt collided")
                check_pairwise_distinct(*offered_tuples, hp, msg="easy: two offered candidates collide")
                offered_tuples.append(hp)
                candidates.append(_fmt_pair("r1", hp[0], "r2", hp[1]))
        # coefficients-as-roots: reports b, c directly as if THEY were the
        # roots (skips solving the characteristic equation entirely), sorted
        # into the same smaller/larger labelling as the true answer
        cp = (min(b, c), max(b, c))
        if cp not in offered_tuples:
            check_distinct((r1, r2), cp, msg="coefficients-as-roots collided")
            check_pairwise_distinct(*offered_tuples, cp, msg="easy: two offered candidates collide")
            offered_tuples.append(cp)
            candidates.append(_fmt_pair("r1", cp[0], "r2", cp[1]))

        distractors = pick_distractors(answer_text, candidates, unordered=False)
        if len(distractors) == 0:
            continue

        ode = _ode_expr(b, c)
        char = _char_expr(b, c)
        text = phrase(rng, _EASY_ROOTS_PATTERNS, ode=ode, char=char)
        working = [
            f"{char} = 0 factors as ({join_terms([(1, 'r'), (-r1, '')])})({join_terms([(1, 'r'), (-r2, '')])}) = 0.",
            f"r1 = {r1}, r2 = {r2}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p12e_characteristic_roots_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        r = rng.choice(_ROOT_POOL)
        b, c = -2 * r, r * r

        hi, lo = _independent_roots_from_bc(b, c)
        check_equivalent(hi, r, "medium: independent root check failed (hi)")
        check_equivalent(lo, r, "medium: independent root check failed (lo)")

        answer = {"r1": r, "r2": r}
        answer_text = _fmt_pair("r1", r, "r2", r)

        candidates = []
        offered_tuples = [(r, r)]
        # sign-flip-on-b: negates the (repeated) root
        sf = (-r, -r)
        if sf not in offered_tuples:
            check_distinct((r, r), sf, msg="medium: sign-flip-on-b collided")
            offered_tuples.append(sf)
            candidates.append(_fmt_pair("r1", sf[0], "r2", sf[1]))
        # forgets-to-halve: uses raw b instead of -b/2 (forgets the /2 step)
        fh = (b, b)
        if fh not in offered_tuples:
            check_distinct((r, r), fh, msg="medium: forgets-to-halve collided")
            check_pairwise_distinct(*offered_tuples, fh, msg="medium: two offered candidates collide")
            offered_tuples.append(fh)
            candidates.append(_fmt_pair("r1", fh[0], "r2", fh[1]))
        # drops-discriminant-sqrt is NEVER offered here: -b/2 == r exactly,
        # the correct answer, so it has no headroom on this tier -- stated
        # honestly.

        distractors = pick_distractors(answer_text, candidates, unordered=False)
        if len(distractors) == 0:
            continue

        ode = _ode_expr(b, c)
        char = _char_expr(b, c)
        text = phrase(rng, _MEDIUM_ROOTS_PATTERNS, ode=ode, char=char)
        working = [
            f"Discriminant = {paren_negative(b)}^2 - 4 × {paren_negative(c)} = 0, so the root is repeated.",
            f"r1 = r2 = -{paren_negative(b)}/2 = {r}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p12e_characteristic_roots_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.choice(_ROOT_POOL)
        k = rng.choice(_K_POOL)
        b, c = -2 * a, a * a + k * k

        hi, lo = _independent_roots_from_bc(b, c)
        check_equivalent(hi, a + k * I, "hard: independent root check failed (a+ki)")
        check_equivalent(lo, a - k * I, "hard: independent root check failed (a-ki)")

        answer = {"real_part": a, "imaginary_part": k}
        answer_text = _fmt_pair("real_part", a, "imaginary_part", k)

        candidate_tuples = []
        # real-part-only / drops-discriminant-sqrt: both offer (a, a) for the
        # labelled pair -- two different misconceptions (reading the roots as
        # real, or falling back to -b/2 for both labels) that happen to land
        # on the same wrong VALUE here; offered once, deduped by pick_distractors.
        candidate_tuples.append(("real_part", a, "imaginary_part", a))
        # wrong-sign-on-imaginary-part: negates k (k always != 0)
        candidate_tuples.append(("real_part", a, "imaginary_part", -k))
        # sign-flip-on-b: negates the real part only
        candidate_tuples.append(("real_part", -a, "imaginary_part", k))

        candidates = []
        offered_tuples = [(a, k)]
        for label1, v1, label2, v2 in candidate_tuples:
            if any((v1, v2) == existing for existing in offered_tuples):
                continue  # collides with the answer or an already-offered candidate
            check_distinct((a, k), (v1, v2), msg=f"hard: candidate ({v1},{v2}) collided")
            check_pairwise_distinct(*offered_tuples, (v1, v2), msg="hard: two offered candidates collide")
            offered_tuples.append((v1, v2))
            candidates.append(_fmt_pair(label1, v1, label2, v2))

        distractors = pick_distractors(answer_text, candidates, unordered=False)
        if len(distractors) < 3:
            continue

        ode = _ode_expr(b, c)
        char = _char_expr(b, c)
        text = phrase(rng, _HARD_ROOTS_PATTERNS, ode=ode, char=char)
        working = [
            f"Discriminant = {paren_negative(b)}^2 - 4 × {paren_negative(c)} = {b * b - 4 * c} < 0.",
            f"Roots = (-{paren_negative(b)} ± √({b * b - 4 * c})) / 2 = {a} ± {k}i.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p12e_characteristic_roots_exam(rng):
    """Inverse direction: given the roots, find b and c -- a genuine
    escalation over hard, both in decision (Vieta's formulas rather than the
    quadratic formula) and in which quantity is unknown (the coefficients,
    never asked for at any other tier)."""
    for _attempt in range(_MAX_ATTEMPTS):
        r1, r2 = rng.sample(_EXAM_ROOT_POOL, 2)
        if r1 > r2:
            r1, r2 = r2, r1
        b, c = -(r1 + r2), r1 * r2

        # Independent check: solve the 2x2 linear system directly from the
        # roots, never touching Vieta's sum/product pattern-match.
        b_check, c_check = _independent_bc_from_roots(r1, r2)
        check_equivalent(b_check, b, "exam: independent b check failed")
        check_equivalent(c_check, c, "exam: independent c check failed")

        answer = {"b": b, "c": c}
        answer_text = _fmt_pair("b", b, "c", c)

        # forgets-negative-on-b: b = r1 + r2 (drops the Vieta sign)
        wrong_b = r1 + r2
        # sign-flip-on-c: c = -(r1*r2)
        wrong_c = -c
        # sum-difference-confusion: uses r2 - r1 instead of r1 + r2 for b
        wrong_b2 = -(r2 - r1)
        candidate_tuples = [
            (wrong_b, c),  # forgets-negative-on-b
            (b, wrong_c),  # sign-flip-on-c
            (wrong_b, wrong_c),  # both wrong at once
            (wrong_b2, c),  # sum-difference-confusion
        ]

        candidates = []
        offered_tuples = [(b, c)]
        for cand_b, cand_c in candidate_tuples:
            if any((cand_b, cand_c) == existing for existing in offered_tuples):
                continue  # collides with the answer or an already-offered candidate
            check_distinct((b, c), (cand_b, cand_c), msg=f"exam: candidate ({cand_b},{cand_c}) collided")
            check_pairwise_distinct(*offered_tuples, (cand_b, cand_c), msg="exam: two offered candidates collide")
            offered_tuples.append((cand_b, cand_c))
            candidates.append(_fmt_pair("b", cand_b, "c", cand_c))

        distractors = pick_distractors(answer_text, candidates, unordered=False)
        if len(distractors) < 3:
            continue

        text = phrase(rng, _EXAM_ROOTS_PATTERNS, r1=r1, r2=r2)
        working = [
            f"b = -(r1 + r2) = -({r1} + {paren_negative(r2)}) = {b}.",
            f"c = r1 × r2 = {paren_negative(r1)} × {paren_negative(r2)} = {c}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p12e_characteristic_roots",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12e_characteristic_roots(rng, tier="easy"):
    if tier == "easy":
        return _p12e_characteristic_roots_easy(rng)
    if tier == "medium":
        return _p12e_characteristic_roots_medium(rng)
    if tier == "hard":
        return _p12e_characteristic_roots_hard(rng)
    if tier == "exam":
        return _p12e_characteristic_roots_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
