"""Generates `p19e_geometric_limit_class` and `p19e_recurrence_long_term_class` questions: classify the long-term
behaviour of a geometric sequence or a first-order linear recurrence.

`p19e_geometric_limit_class` -- given a geometric sequence's positive first
term and common ratio r (stated directly, inferred from terms, given by an
explicit/recursive rule, or recovered from two non-consecutive terms via
r^3 = t5/t2), classify the limiting behaviour as n -> infinity: converges to
0 (|r| < 1), diverges to +infinity (r > 1), or neither (r <= -1, oscillates).

`p19e_recurrence_long_term_class` -- given a first-order linear recurrence
u(n+1) = a*u(n) + b with a > 0 (stated directly, illustrated with early
terms, compared against its own fixed point, or translated from a percentage
-rate context), classify the long-term behaviour: increases without bound,
decreases without bound, or settles to a steady state.

Both are 3-way classifications: exactly 3 fixed-wording options, shuffled
onto letters A/B/C fresh each draw so the key letter is balanced (never
touches GENUINE_TWO_OPTION_TOPICS -- a 3-way mcq always has >= 2
distractors). Each generator draws its target CLASS uniformly first, then
builds data that realises it, and re-derives the class by a second,
behavioural route (iterating the sequence/recurrence) before trusting the
constructive one.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import check, generator, latex_answer, phrase, rational_text, rounded_decimal_string

_LETTERS = ["A", "B", "C"]


def _mcq_shuffle(rng, classes: list[str], option_text: dict[str, str], target: str):
    """Shuffle `classes` onto A/B/C fresh each draw; return the rendered
    options block, the answer letter, and the two distractor letters."""
    order = list(classes)
    rng.shuffle(order)
    letter_of_class = dict(zip(_LETTERS, order))
    class_of_letter = {c: l for l, c in letter_of_class.items()}
    check(len(class_of_letter) == 3, "mcq classes must be pairwise distinct")
    opts = "   ".join(f"{l}: {option_text[letter_of_class[l]]}" for l in _LETTERS)
    answer_letter = class_of_letter[target]
    distractor_letters = [l for l in _LETTERS if l != answer_letter]
    return opts, answer_letter, distractor_letters


# ===========================================================================
# Skill A: p19e_geometric_limit_class
# ===========================================================================

GEO_CLASSES = ["zero", "infinity", "neither"]
GEO_OPTION_TEXT = {
    "zero": "Converges to 0",
    "infinity": "Diverges to +∞ (the terms stay positive and grow without bound)",
    "neither": "Neither: the terms oscillate in sign and do not approach a limit",
}

_R_ZERO_EASY = [
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(1, 3),
    Fraction(-1, 3),
    Fraction(2, 5),
    Fraction(-3, 4),
    Fraction(3, 5),
]
_R_INF_EASY = [Fraction(2), Fraction(3), Fraction(4), Fraction(3, 2), Fraction(5, 2)]
_R_NEITHER_EASY = [Fraction(-1), Fraction(-2), Fraction(-3), Fraction(-3, 2), Fraction(-5, 2)]
_R_EASY = {"zero": _R_ZERO_EASY, "infinity": _R_INF_EASY, "neither": _R_NEITHER_EASY}

_R_ZERO_MEDIUM = [Fraction(1, 2), Fraction(-2, 3), Fraction(1, 4), Fraction(-3, 4), Fraction(2, 3)]
_R_INF_MEDIUM = [Fraction(2), Fraction(3), Fraction(3, 2), Fraction(5, 2)]
_R_NEITHER_MEDIUM = [Fraction(-2), Fraction(-3), Fraction(-3, 2), Fraction(-1)]
_R_MEDIUM = {"zero": _R_ZERO_MEDIUM, "infinity": _R_INF_MEDIUM, "neither": _R_NEITHER_MEDIUM}

_R_ZERO_HARD = [Fraction(7, 8), Fraction(-9, 10), Fraction(4, 5), Fraction(-3, 4)]
_R_INF_HARD = [Fraction(11, 10), Fraction(6, 5), Fraction(9, 8)]
_R_NEITHER_HARD = [Fraction(-5, 4), Fraction(-11, 10), Fraction(-6, 5), Fraction(-1)]
_R_HARD = {"zero": _R_ZERO_HARD, "infinity": _R_INF_HARD, "neither": _R_NEITHER_HARD}

# (p, q) pairs for the exam tier, r = p/q in lowest terms, chosen per class.
_PQ_ZERO = [(1, 2), (-1, 2), (1, 3), (-1, 3), (2, 3), (-2, 3)]
_PQ_INF = [(2, 1), (3, 1), (3, 2)]
_PQ_NEITHER = [(-1, 1), (-2, 1), (-3, 1), (-3, 2)]
_PQ_EXAM = {"zero": _PQ_ZERO, "infinity": _PQ_INF, "neither": _PQ_NEITHER}


def geo_classify_by_rule(r: Fraction) -> str:
    """Construction route: classify directly from r."""
    check(r != 0 and r != 1, "geometric limit class: r must not be 0 or 1")
    if r > 1:
        return "infinity"
    if r <= -1:
        return "neither"
    return "zero"


def geo_classify_by_simulation(a1, r: Fraction, n_mid: int = 60, n_extra: int = 40) -> str:
    """Behavioural route: iterate the sequence and read off the trend --
    a genuinely different computation from the r-comparison rule above."""
    u = Fraction(a1)
    for _ in range(n_mid):
        u *= r
    mid = u
    for _ in range(n_extra):
        u *= r
    end = u
    if abs(end) < abs(mid):
        return "zero"
    signs = set()
    v = mid
    for _ in range(10):
        signs.add(1 if v > 0 else -1)
        v *= r
    if len(signs) > 1:
        return "neither"
    return "infinity"


@generator(
    "p19e_geometric_limit_class",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_geometric_limit_class(rng, tier="easy"):
    target = rng.choice(GEO_CLASSES)
    opts_text, answer_letter, distractor_letters = _mcq_shuffle(rng, GEO_CLASSES, GEO_OPTION_TEXT, target)

    if tier == "easy":
        r = rng.choice(_R_EASY[target])
        a1 = rng.randint(1, 9)
        text = phrase(
            rng,
            [
                "A geometric sequence has first term t1 = {a1} (a positive number) and "
                "common ratio r = {r}. What is the limiting behaviour of the sequence as "
                "n approaches infinity? {opts}",
                "The first term of a geometric sequence is t1 = {a1} (positive) and its "
                "common ratio is r = {r}. Classify the sequence's limiting behaviour as "
                "n → ∞. {opts}",
                "Consider a geometric sequence with positive first term t1 = {a1} and "
                "common ratio r = {r}. As n → ∞, what happens to the sequence's "
                "terms? {opts}",
            ],
            a1=a1,
            r=rational_text(r),
            opts=opts_text,
        )
    elif tier == "medium":
        r = rng.choice(_R_MEDIUM[target])
        q = r.denominator
        k = rng.choice([1, 2, 3])
        a1 = k * q * q
        t1 = a1
        t2 = a1 * r
        t3 = t2 * r
        check(
            t2.denominator == 1 and t3.denominator == 1,
            "geometric limit class medium: displayed terms must be integers",
        )
        text = phrase(
            rng,
            [
                "A geometric sequence begins {t1}, {t2}, {t3}, .... What is the limiting "
                "behaviour of the sequence as n → ∞? {opts}",
                "The first three terms of a geometric sequence are {t1}, {t2}, {t3}. "
                "Classify its limiting behaviour as n → ∞. {opts}",
                "A geometric sequence starts {t1}, {t2}, {t3}, .... As n → ∞, "
                "what happens to the sequence's terms? {opts}",
            ],
            t1=int(t1),
            t2=int(t2),
            t3=int(t3),
            opts=opts_text,
        )
    elif tier == "hard":
        r = rng.choice(_R_HARD[target])
        a1 = rng.randint(1, 9)
        if rng.random() < 0.5:
            text = phrase(
                rng,
                [
                    "A sequence is defined by t_n = t1 × r^(n-1), with t1 = {a1} and "
                    "r = {r}, for n = 1, 2, 3, .... What is the limiting behaviour of the "
                    "sequence as n → ∞? {opts}",
                    "A sequence has t_n = t1 × r^(n-1) for n = 1, 2, 3, ..., with "
                    "t1 = {a1} and r = {r}. Classify its limiting behaviour as n → "
                    "∞. {opts}",
                ],
                a1=a1,
                r=rational_text(r),
                opts=opts_text,
            )
        else:
            text = phrase(
                rng,
                [
                    "A sequence satisfies t1 = {a1} and t_(n+1) = r × t_n for n >= 1, "
                    "with r = {r}. What is the limiting behaviour of the sequence as n "
                    "→ ∞? {opts}",
                    "A sequence is generated by t1 = {a1} and the rule t_(n+1) = r × "
                    "t_n, with r = {r}. Classify its limiting behaviour as n → ∞. "
                    "{opts}",
                ],
                a1=a1,
                r=rational_text(r),
                opts=opts_text,
            )
    else:  # exam
        p, q = rng.choice(_PQ_EXAM[target])
        k = rng.choice([1, 2])
        r = Fraction(p, q)
        t2 = k * p * q**3
        t5 = k * p**4
        check(t2 != 0 and t5 != 0, "geometric limit class exam: t2/t5 must not be zero")
        text = phrase(
            rng,
            [
                "In a geometric sequence with positive first term, t2 = {t2} and "
                "t5 = {t5}. What is the limiting behaviour of the sequence as n → "
                "∞? {opts}",
                "A geometric sequence has positive first term. Its second term is "
                "t2 = {t2} and its fifth term is t5 = {t5}. Classify the sequence's "
                "limiting behaviour as n → ∞. {opts}",
            ],
            t2=t2,
            t5=t5,
            opts=opts_text,
        )
        a1 = k * q**4

    rule_class = geo_classify_by_rule(r)
    sim_class = geo_classify_by_simulation(a1, r)
    check(rule_class == target, "geometric limit class: rule route disagrees with the drawn target")
    check(sim_class == target, "geometric limit class: simulation route disagrees with the drawn target")

    working = [
        f"r = {rational_text(r)}.",
        f"{'|r| < 1' if -1 < r < 1 else ('r > 1' if r > 1 else 'r <= -1')} -> {GEO_OPTION_TEXT[target]}.",
        f"Option {answer_letter} is the only one consistent with this.",
    ]

    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=distractor_letters,
    )


# ===========================================================================
# Skill B: p19e_recurrence_long_term_class
# ===========================================================================

REC_CLASSES = ["increase", "decrease", "steady"]
REC_OPTION_TEXT = {
    "increase": "Increases without bound",
    "decrease": "Decreases without bound",
    "steady": "Settles to a steady state",
}

_A_LT1 = [Fraction(1, 2), Fraction(3, 4), Fraction(4, 5), Fraction(9, 10), Fraction(1, 4), Fraction(7, 10)]
_A_GT1 = [
    Fraction(5, 4),
    Fraction(6, 5),
    Fraction(3, 2),
    Fraction(2, 1),
    Fraction(5, 2),
    Fraction(21, 20),
    Fraction(11, 10),
]
_A_GT1_PCT = [
    (Fraction(21, 20), 5),
    (Fraction(51, 50), 2),
    (Fraction(6, 5), 20),
    (Fraction(23, 20), 15),
    (Fraction(11, 10), 10),
    (Fraction(27, 25), 8),
]
_EXAM_CONTEXTS = ["a savings account balance", "the amount owing on a loan"]


def rec_classify_by_rule(a: Fraction, b, u0) -> str:
    """Construction route: classify directly from a, b and u0."""
    if a == 1:
        if b > 0:
            return "increase"
        if b < 0:
            return "decrease"
        return "steady"
    if 0 < a < 1:
        return "steady"
    x_star = Fraction(b, 1 - a)
    if u0 == x_star:
        return "steady"
    return "increase" if u0 > x_star else "decrease"


def rec_classify_by_simulation(a: Fraction, b, u0, block: int = 60) -> str:
    """Behavioural route: iterate the recurrence in two equal blocks and
    compare the two resulting differences -- exact, since for a linear
    recurrence u(n+block) - u(n) scales by exactly a**block from one block
    to the next (u(n) - x* = a**n * (u0 - x*)). A shrinking difference means
    the sequence is settling to a fixed value; a non-shrinking difference
    means it runs away, in the direction of the first difference's sign.
    This is a genuinely different computation from the fixed-point
    comparison rule above -- no x* is ever computed here."""
    aq = Fraction(a)
    bq = Fraction(b)

    def _advance(start, n):
        u = start
        for _ in range(n):
            u = aq * u + bq
        return u

    u0v = Fraction(u0)
    u1 = _advance(u0v, block)
    d1 = u1 - u0v
    if d1 == 0:
        return "steady"
    u2 = _advance(u1, block)
    d2 = u2 - u1
    if abs(d2) < abs(d1):
        return "steady"
    return "increase" if d1 > 0 else "decrease"


def _a_display(a: Fraction) -> str:
    if a.denominator == 1:
        return str(a.numerator)
    val = a.numerator / a.denominator
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s


def _recurrence_formula(a: Fraction, b: int) -> str:
    sign = "+" if b >= 0 else "-"
    return f"u(n+1) = {_a_display(a)} × u(n) {sign} {abs(b)}"


def _fixed_point_case(rng, a: Fraction, cls: str, j_range, offset_range):
    q = a.denominator
    p = a.numerator
    j = rng.choice(j_range)
    x_star = q * j
    b = j * (q - p)
    if cls == "steady":
        u0 = x_star
    else:
        offset = rng.choice(offset_range)
        u0 = x_star + offset if cls == "increase" else x_star - offset
    return b, u0


@generator(
    "p19e_recurrence_long_term_class",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_recurrence_long_term_class(rng, tier="easy"):
    target = rng.choice(REC_CLASSES)
    opts_text, answer_letter, distractor_letters = _mcq_shuffle(rng, REC_CLASSES, REC_OPTION_TEXT, target)

    if tier == "easy":
        if target == "steady":
            a = rng.choice(_A_LT1)
            b = rng.randint(-50, 50)
        else:
            a = Fraction(1)
            mag = rng.randint(1, 50)
            b = mag if target == "increase" else -mag
        u0 = rng.randint(1, 50)
        formula = _recurrence_formula(a, b)
        text = phrase(
            rng,
            [
                "A sequence is defined by {formula}, with u(0) = {u0}. What happens to u(n) as n → ∞? {opts}",
                "Consider the recurrence {formula}, starting from u(0) = {u0}. Classify "
                "the long-term behaviour of u(n) as n → ∞. {opts}",
                "A quantity u(n) obeys {formula}, with initial value u(0) = {u0}. As "
                "n → ∞, what happens to u(n)? {opts}",
            ],
            formula=formula,
            u0=u0,
            opts=opts_text,
        )
    elif tier == "medium":
        if target == "increase":
            a = rng.choice(_A_GT1)
            b = rng.randint(0, 50)
            u0 = rng.randint(1, 50)
        elif target == "decrease":
            a = rng.choice(_A_GT1)
            b = -rng.randint(1, 50)
            u0 = -rng.randint(0, 50)
        else:
            a = rng.choice(_A_LT1)
            b = rng.choice([-1, 1]) * rng.randint(100, 900)
            u0 = rng.randint(-200, 200)
        formula = _recurrence_formula(a, b)
        terms = [Fraction(u0)]
        for _ in range(3):
            terms.append(a * terms[-1] + b)
        terms_text = ", ".join(rounded_decimal_string(t, 2) for t in terms)
        text = phrase(
            rng,
            [
                "A sequence is defined by {formula}, with u(0) = {u0}. Its first four "
                "terms are {terms}. What happens to u(n) as n → ∞? {opts}",
                "Consider the recurrence {formula}, starting from u(0) = {u0}; its first "
                "four terms are {terms}. Classify the long-term behaviour of u(n) as "
                "n → ∞. {opts}",
                "A quantity u(n) obeys {formula}, with u(0) = {u0}, giving first terms "
                "{terms}. What happens to u(n) as n → ∞? {opts}",
            ],
            formula=formula,
            u0=u0,
            terms=terms_text,
            opts=opts_text,
        )
    elif tier == "hard":
        a = rng.choice(_A_GT1)
        b, u0 = _fixed_point_case(rng, a, target, j_range=range(-6, 7), offset_range=range(2, 21))
        formula = _recurrence_formula(a, b)
        text = phrase(
            rng,
            [
                "A sequence is defined by {formula}, with u(0) = {u0} (a > 1). Compare "
                "u(0) to the recurrence's fixed point to decide: what happens to u(n) as "
                "n → ∞? {opts}",
                "Consider the recurrence {formula} (with a > 1), starting from u(0) = "
                "{u0}. By finding the fixed point and comparing it with u(0), classify "
                "the long-term behaviour of u(n). {opts}",
                "A quantity u(n) obeys {formula} where a > 1, with initial value u(0) = "
                "{u0}. Find the recurrence's fixed point and compare it to u(0) to "
                "determine what happens to u(n) as n → ∞. {opts}",
            ],
            formula=formula,
            u0=u0,
            opts=opts_text,
        )
    else:  # exam
        a, pct = rng.choice(_A_GT1_PCT)
        # j_range/offset_range chosen so x_star = q*j is always comfortably
        # positive and offset never exceeds it -- u(0) stays non-negative
        # (a dollar amount is never negative in this context).
        b, u0 = _fixed_point_case(rng, a, target, j_range=range(3, 7), offset_range=range(1, 10))
        verb = "added" if b >= 0 else "removed"
        amount = abs(b)
        context = rng.choice(_EXAM_CONTEXTS)
        context_cap = context[0].upper() + context[1:]
        text = phrase(
            rng,
            [
                "{context_cap} grows by {pct}% each period, and ${amount} is {verb} "
                "every period, starting at ${u0}. If this pattern continued indefinitely, "
                "what would happen to the amount over the long term? {opts}",
                "Suppose {context} increases by {pct}% each period; ${amount} is also "
                "{verb} every period, and the amount starts at ${u0}. Long-term, if the "
                "same rule kept applying forever, what would happen to the amount? {opts}",
            ],
            context_cap=context_cap,
            context=context,
            pct=pct,
            amount=amount,
            verb=verb,
            u0=u0,
            opts=opts_text,
        )

    rule_class = rec_classify_by_rule(a, b, u0)
    sim_class = rec_classify_by_simulation(a, b, u0)
    check(rule_class == target, "recurrence long-term class: rule route disagrees with the drawn target")
    check(sim_class == target, "recurrence long-term class: simulation route disagrees with the drawn target")

    working = [
        f"a = {_a_display(a)}, b = {b}, u(0) = {u0}.",
        f"-> {REC_OPTION_TEXT[target]}.",
        f"Option {answer_letter} is the only one consistent with this.",
    ]

    return dict(
        text=text,
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=distractor_letters,
    )
