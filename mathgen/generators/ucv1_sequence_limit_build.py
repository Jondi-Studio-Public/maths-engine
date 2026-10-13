"""Generates `sequence_limit` items: decide whether a proposed limit of an elementary sequence is valid, prove it
from the epsilon-N definition when it is, and give a finite epsilon witness when it is not.

Each question states a sequence a_n and a proposed limit L. Whether the claim is valid is decided here from the
true limit of the sequence (known by construction and re-checked with exact fractions at large n). For a valid
claim the generator also knows an N(epsilon) that works, and for an invalid one an epsilon_0 > 0 and an index N0
from which every term (or every term of one parity) is at least epsilon_0 from L. All of these are checked on the
terms of the sequence. The answer is marked against a rubric, so `answer` is a marking dict (see `_crit_kit`).

Ladder:

- easy: a ratio of linear expressions (3n + 5)/(2n + 7); the proposed limit is the true limit, or the value the
  first terms suggest, or the true limit shifted.
- medium: a limit plus an alternating decaying term (c + (-1)^n/(n + k)), and a ratio of quadratic expressions.
- hard: a sequence that alternates between two values (c + (-1)^n d n/(n + k)), which has no limit at all, so every
  proposed limit is invalid.
- exam: a difference of square roots, and a limit with a bounded oscillating term in the numerator (squeeze).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator
from ._crit_kit import Fm, Prompt, attempts, finish, frac_plain, frac_tex, marking

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}

TAILS_L = (
    "Decide whether the claim is valid. If it is, prove it from the definition of a limit. If it is not, give a positive number for which no index works and show that infinitely many terms stay that far from the proposed limit.",
    "Is the claim valid? If so, prove it from the epsilon-N definition by finding N for every epsilon. If not, give a fixed positive epsilon and show that infinitely many terms are at least that far from the proposed limit.",
)


def _fr(x: Fraction) -> Fm:
    return Fm(frac_tex(x.numerator, x.denominator), frac_plain(x.numerator, x.denominator))


def _ceil(x: Fraction) -> int:
    """The least integer strictly greater than x (and at least 1), so that n >= N means n > x."""
    return max(1, math.floor(x) + 1)


def _cf(x: Fraction) -> str:
    """A coefficient written to stand before '/n': whole numbers bare, fractions bracketed."""
    return str(x.numerator) if x.denominator == 1 else f"({x.numerator}/{x.denominator})"


def _sci(x: Fraction) -> str:
    return frac_plain(x.numerator, x.denominator)


class Seq:
    """A sequence with its true limit points, a valid N(eps) when it converges, and a proposed limit."""

    def __init__(self, fm: Fm, term, limit_points: list[Fraction], n_of_eps, tail_gap):
        self.fm = fm
        self.term = term  # n -> Fraction
        self.limit_points = limit_points  # one entry if convergent, two if it alternates
        self.n_of_eps = n_of_eps  # Fraction eps -> int, valid when convergent
        self.tail_gap = tail_gap  # (limit point, parity or None, eps0) -> N0


def _claim_item(rng, seq: Seq, proposed: Fraction, family_note: str) -> dict:
    pts = seq.limit_points
    valid = len(pts) == 1 and pts[0] == proposed
    p = Prompt()
    p.t("Consider the sequence ").m(Fm("a_{n}=", "a_n = ") + seq.fm).t(" for n = 1, 2, 3, ... . A student claims that ")
    p.m(Fm("a_{n}\\to ", "a_n → ") + _fr(proposed)).t(" as n tends to infinity. ").t(rng.choice(TAILS_L))
    ptxt = _sci(proposed)
    if valid:
        # verify N(eps) on the terms for several epsilons
        for eps in (Fraction(1, 2), Fraction(1, 10), Fraction(1, 100), Fraction(3, 1000)):
            n0 = seq.n_of_eps(eps)
            check(n0 >= 1, "N is a positive index")
            for n in list(range(n0, n0 + 400)) + [n0 * 3 + 7, n0 * 50 + 11]:
                check(abs(seq.term(n) - proposed) < eps, f"N(eps) fails at eps={eps}, n={n}")
        ref = [
            f"The claim is valid: the sequence tends to {ptxt}. Let epsilon > 0 be given.",
            f"{family_note}",
            "Choose N so that this bound is below epsilon for every n ≥ N; then |a_n - L| < epsilon for all n ≥ N, which is the definition of the limit.",
            "The index N is chosen after epsilon and does not depend on n.",
        ]
        steps = [
            ("correctness", f"decides the claim is valid and bounds |a_n - {ptxt}| by an explicit expression in n"),
            (
                "logical_completeness",
                "turns the bound into a choice of N in terms of epsilon (N depends on epsilon only) and checks it works for every n ≥ N",
            ),
            ("definitions_assumptions", "starts from an arbitrary epsilon > 0 and uses the definition of the limit"),
            ("communication", "ends with 'so a_n → L' and states N explicitly"),
        ]
        traps = [
            (
                "tests_values_only",
                "does it evaluate a few terms and say they get close, without an N for every epsilon?",
            ),
            ("n_depends_on_n", "does the N it chooses depend on n itself?"),
            ("equals_not_less", "does it find N for |a_n - L| < epsilon for all n ≥ N, not only for one n?"),
        ]
        verdict = f"Valid: a_n tends to {ptxt}"
    else:
        # the witness: epsilon0 = (distance from L to the farther limit point) / 2
        far = max(pts, key=lambda q: abs(q - proposed))
        eps0 = abs(far - proposed) / 2
        check(eps0 > 0, "an invalid claim has a positive witness")
        n0, parity = seq.tail_gap(far, eps0)
        idx = [n for n in range(n0, n0 + 600) if parity is None or n % 2 == parity]
        check(all(abs(seq.term(n) - proposed) >= eps0 for n in idx), "witness holds on the tail")
        check(abs(seq.term(n0 * 40 + (parity or 0)) - proposed) >= eps0, "witness holds far out")
        where = "every n ≥ N0" if parity is None else f"every {'even' if parity == 0 else 'odd'} n ≥ N0"
        ref = [
            f"The claim is not valid. {family_note}",
            f"Take epsilon_0 = {_sci(eps0)}. For {where}, where N0 = {n0}, the term a_n is within {_sci(eps0)} of {_sci(far)}, so |a_n - {ptxt}| ≥ |{_sci(far)} - {ptxt}| - {_sci(eps0)} = {_sci(abs(far - proposed) - eps0)}, which is at least epsilon_0.",
            "So for every index N there are terms beyond N (infinitely many, in fact) with |a_n - L| ≥ epsilon_0. No N can satisfy the definition for this epsilon_0, so a_n does not tend to the proposed limit.",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is not valid and gives a fixed epsilon_0 > 0 (such as {_sci(eps0)}) that does not depend on n",
            ),
            (
                "logical_completeness",
                "shows infinitely many terms (all large n, or all large n of one parity) with |a_n - L| ≥ epsilon_0, so no N works",
            ),
            (
                "definitions_assumptions",
                "uses the negation of the definition: some epsilon_0 > 0 such that for every N there is n ≥ N with |a_n - L| ≥ epsilon_0",
            ),
            ("communication", "states epsilon_0 and the tail of terms clearly"),
        ]
        traps = [
            ("one_term_only", "does it show only a single term far from L, rather than infinitely many?"),
            ("epsilon_depends_on_n", "does its epsilon_0 depend on n or shrink with n?"),
            (
                "wrong_limit_assumed",
                "does it correctly find the actual behaviour of the sequence (limit point or two limit points)?",
            ),
        ]
        if len(pts) == 2:
            traps.append(
                (
                    "alternating_has_no_limit",
                    "does it notice that the sequence alternates between two values, so no proposed limit can be valid?",
                )
            )
        verdict = f"Not valid: epsilon_0 = {_sci(eps0)} works (the terms stay at least that far from {ptxt})"
    mark = marking(reference=ref, steps=steps, traps=traps)
    return finish(p, verdict, mark)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
def _ratio_linear(rng):
    for _ in attempts(500, "a linear ratio"):
        p_, q_, r_, s_ = rng.randint(1, 6), rng.randint(1, 9), rng.randint(1, 5), rng.randint(1, 9)
        if p_ * s_ != q_ * r_:
            break
    lim = Fraction(p_, r_)
    D = abs(q_ * r_ - p_ * s_)

    def term(n):
        return Fraction(p_ * n + q_, r_ * n + s_)

    def n_of_eps(eps):
        return max(1, _ceil(Fraction(D, r_ * r_) / eps))

    def tail_gap(point, eps0):
        return max(1, _ceil(Fraction(D, r_ * r_) / eps0)), None

    fm = Fm(f"\\frac{{{p_}n+{q_}}}{{{r_}n+{s_}}}", f"({p_}n + {q_})/({r_}n + {s_})")
    seq = Seq(fm, term, [lim], n_of_eps, tail_gap)
    props = [lim, lim, Fraction(q_, s_), lim + 1, lim - Fraction(1, 2)]
    return (
        seq,
        rng.choice(props),
        (
            f"|a_n - {_sci(lim)}| = {D}/({r_}({r_}n + {s_})) < {_cf(Fraction(D, r_ * r_))}/n, so it is below epsilon once n > {_cf(Fraction(D, r_ * r_))}/epsilon; take any integer N greater than {_cf(Fraction(D, r_ * r_))}/epsilon."
        ),
        (
            f"The terms tend to {_sci(lim)}: |a_n - {_sci(lim)}| < {_cf(Fraction(D, r_ * r_))}/n, which is small for large n."
        ),
    )


def _alt_decay(rng):
    c = Fraction(rng.randint(-4, 6), rng.choice((1, 2)))
    k = rng.randint(0, 6)

    def term(n):
        return c + Fraction((-1) ** n, n + k)

    def n_of_eps(eps):
        return max(1, _ceil(1 / eps))

    def tail_gap(point, eps0):
        return max(1, _ceil(1 / eps0)), None

    ctex = frac_tex(c.numerator, c.denominator)
    cp = _sci(c)
    kk = f"n+{k}" if k else "n"
    if c == 0:
        fm = Fm(f"\\frac{{(-1)^{{n}}}}{{{kk}}}", f"(-1)^n/({kk})" if k else "(-1)^n/n")
    else:
        fm = Fm(f"{ctex}+\\frac{{(-1)^{{n}}}}{{{kk}}}", f"{cp} + (-1)^n/({kk})" if k else f"{cp} + (-1)^n/n")
    seq = Seq(fm, term, [c], n_of_eps, tail_gap)
    props = [c, c, c + 1, c - 2, c + Fraction(1, 2)]
    return (
        seq,
        rng.choice(props),
        (
            f"|a_n - {cp}| = 1/({kk}) < 1/n, so it is below epsilon once n > 1/epsilon; take any integer N greater than 1/epsilon."
        ),
        (f"The terms tend to {cp}: |a_n - {cp}| = 1/({kk}) ≤ 1/n, which is small for large n."),
    )


def _ratio_quadratic(rng):
    b = rng.randint(1, 5)
    c = rng.choice((2, 3, 5, 6))

    def term(n):
        return Fraction(n * n + b * n, n * n - c)

    def n_of_eps(eps):
        return max(c + b + 1, _ceil(Fraction(b + 1) / eps))

    def tail_gap(point, eps0):
        return max(c + b + 1, _ceil(Fraction(b + 1) / eps0)), None

    fm = Fm(f"\\frac{{n^{{2}}+{b}n}}{{n^{{2}}-{c}}}", f"(n^2 + {b}n)/(n^2 - {c})")
    seq = Seq(fm, term, [Fraction(1)], n_of_eps, tail_gap)
    props = [Fraction(1), Fraction(1), Fraction(0), Fraction(2)]
    return (
        seq,
        rng.choice(props),
        (
            f"|a_n - 1| = ({b}n + {c})/(n^2 - {c}) ≤ {b + 1}/n for n ≥ {c + b + 1}, so it is below epsilon once n > {b + 1}/epsilon; take N to be any integer at least {c + b + 1} and greater than {b + 1}/epsilon."
        ),
        (f"The terms tend to 1: |a_n - 1| = ({b}n + {c})/(n^2 - {c}) ≤ {b + 1}/n for n ≥ {c + b + 1}."),
    )


def _oscillating(rng):
    c = Fraction(rng.randint(-3, 4))
    d = Fraction(rng.randint(1, 4))
    k = rng.randint(1, 5)

    def term(n):
        return c + (-1) ** n * d * Fraction(n, n + k)

    def tail_gap(point, eps0):
        parity = 0 if point == c + d else 1
        return max(1, _ceil(d * k / eps0)), parity

    ctx = frac_tex(c.numerator, c.denominator)
    dd = "" if d == 1 else _sci(d)
    lead_t = "" if c == 0 else f"{ctx}+"
    lead_p = "" if c == 0 else f"{_sci(c)} + "
    fm = Fm(f"{lead_t}{dd}(-1)^{{n}}\\frac{{n}}{{n+{k}}}", f"{lead_p}{dd}(-1)^n n/(n + {k})")
    seq = Seq(fm, term, [c + d, c - d], lambda eps: 1, tail_gap)
    props = [c + d, c - d, c, c + 2 * d]
    return (
        seq,
        rng.choice(props),
        (
            f"The even-numbered terms tend to {_sci(c + d)} and the odd-numbered terms tend to {_sci(c - d)}: a_n = {lead_p}{dd}(-1)^n n/(n + {k}) and n/(n + {k}) tends to 1. The sequence alternates between two different values, so it has no limit."
        ),
        (
            f"The even terms are {lead_p}{dd}n/(n + {k}), within {_sci(d * k)}/(n + {k}) of {_sci(c + d)}; the odd terms are within the same amount of {_sci(c - d)}."
        ),
    )


def _sqrt_difference(rng):
    a = rng.randint(1, 6)
    lim = Fraction(a, 2)
    an = "n" if a == 1 else f"{a}n"

    def term(n):
        return Fraction(math.sqrt(n * n + a * n) - n).limit_denominator(10**12)

    def n_of_eps(eps):
        return max(1, _ceil(Fraction(a * a, 8) / eps))

    def tail_gap(point, eps0):
        return max(1, _ceil(Fraction(a * a, 8) / eps0)), None

    # the inequality 0 < a/2 - a_n < a^2/(8n) holds for every n >= 1; check it on a long run
    for n in list(range(1, 2000)) + [10**4, 10**5]:
        gap = lim - (math.sqrt(n * n + a * n) - n)
        check(0 < gap < a * a / (8 * n) + 1e-12, "square-root gap bound")
    fm = Fm(f"\\sqrt{{n^{{2}}+{an}}}-n", f"sqrt(n^2 + {an}) - n")
    seq = Seq(fm, term, [lim], n_of_eps, tail_gap)
    props = [lim, lim, Fraction(a), Fraction(0), lim + 1]
    return (
        seq,
        rng.choice(props),
        (
            f"Multiply by the conjugate: a_n = {an}/(sqrt(n^2 + {an}) + n). Since sqrt(n^2 + {an}) < n + {a}/2, a_n > {an}/(2n + {a}/2), and so 0 < {a}/2 - a_n < {a * a}/(8n); it is below epsilon once n > {a * a}/(8 epsilon): take any integer N greater than {a * a}/(8 epsilon)."
        ),
        (
            f"Multiplying by the conjugate gives a_n = {an}/(sqrt(n^2 + {an}) + n), and 0 < {a}/2 - a_n < {a * a}/(8n), so the terms tend to {a}/2."
        ),
    )


def _squeeze(rng):
    A, B = rng.randint(1, 4), rng.randint(0, 3)
    C = Fraction(rng.randint(-3, 5))

    def term(n):
        return C + Fraction(round((A * math.sin(n) + B) * 10**9), 10**9) / n

    def n_of_eps(eps):
        return max(1, _ceil(Fraction(A + B) / eps))

    def tail_gap(point, eps0):
        return max(1, _ceil(Fraction(A + B) / eps0)), None

    a_t = "" if A == 1 else str(A)
    bnum = f"{a_t}\\sin n" + (f"+{B}" if B else "")
    bplain = f"{a_t} sin(n)".strip() + (f" + {B}" if B else "")
    fm = Fm(f"{frac_tex(C.numerator, C.denominator)}+\\frac{{{bnum}}}{{n}}", f"{_sci(C)} + ({bplain})/n")
    seq = Seq(fm, term, [C], n_of_eps, tail_gap)
    props = [C, C, C + 1, C - Fraction(1, 2)]
    return (
        seq,
        rng.choice(props),
        (
            f"Since |sin n| ≤ 1, |a_n - {_sci(C)}| = |{bplain}|/n ≤ {A + B}/n. This is below epsilon once n > {A + B}/epsilon, so take any integer N greater than {A + B}/epsilon."
        ),
        (f"Since |sin n| ≤ 1, |a_n - {_sci(C)}| ≤ {A + B}/n, which tends to 0, so the terms tend to {_sci(C)}."),
    )


def _build(rng, fam) -> dict:
    out = fam(rng)
    seq, proposed = out[0], out[1]
    note_valid, note_invalid = out[2], out[3]
    pts = seq.limit_points
    valid = len(pts) == 1 and pts[0] == proposed
    return _claim_item(rng, seq, proposed, note_valid if valid else note_invalid)


def _easy(rng) -> dict:
    return _build(rng, _ratio_linear)


def _medium(rng) -> dict:
    return _build(rng, _alt_decay)


def _hard(rng) -> dict:
    return _build(rng, rng.choice((_oscillating, _ratio_quadratic)))


def _exam(rng) -> dict:
    return _build(rng, rng.choice((_sqrt_difference, _squeeze)))


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("sequence_limit", level=16, presets=_PRESETS)
def sequence_limit(rng, variant="easy"):
    """`ucv1-sequence-limit-counterexample`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
