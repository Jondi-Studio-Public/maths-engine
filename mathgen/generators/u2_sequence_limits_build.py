"""Generates `u2_sequence_limits` questions and the exam chain `exd_x_u2_sequence_limits`.

``ucv1-sequence-limits-squeeze-monotone`` (``functions-sequences``, answer ``fraction``; a whole-number limit is
accepted as a fraction): the limit of a sequence found by dividing through by the dominant term, by squeezing, or by
solving L = f(L) for a monotone bounded recursive sequence.

* easy: a ratio of two polynomials in n of the same degree (limit: ratio of the leading coefficients), or of a lower
  degree over a higher one (limit 0).
* medium: a ratio of sums of exponentials (divide by the dominant exponential), a difference of square roots
  (multiply by the conjugate), a ratio whose numerator has a bounded oscillating term (squeeze), or a linear
  recurrence a(n+1) = p a(n) + q.
* hard: a recursive sequence a(n+1) = f(a(n)) whose limit equation has two roots, where the starting value and
  monotonicity decide which root is the limit: (a(n)^2 + c)/s, the square root of (k + a(n)), or Newton's average
  (a(n) + m^2/a(n))/2.
* exam: a(1) given and a(n+1) = (a(n)^2 + c)/s. (a) the possible limits, (b) a(2), (c) the limit, (d) the limit of
  (2n + 1) a(n)/(n + 3) (follows through from (c)).

Every limit is checked by iterating the recurrence (or evaluating the expression at a very large n) in 60-digit
arithmetic and comparing with the exact value.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, paren_negative, rational_text
from ._u2_kit import frac_d, keep, int_d, redraws

TOPIC = "u2_sequence_limits"
EXAM_TOPIC = "exd_x_u2_sequence_limits"


def sub(n):
    """aₙ-style subscripts for a short expression: 'n', 'n+1', '1', '2'."""
    table = str.maketrans("0123456789n+-()", "₀₁₂₃₄₅₆₇₈₉ₙ₊₋₍₎")
    return n.translate(table)


A_N = "a" + sub("n")
A_N1 = "a" + sub("n+1")


def _poly_text(coeffs, var="n"):
    """3n² - 2n + 5 from [3, -2, 5] (highest power first)."""
    degree = len(coeffs) - 1
    out = ""
    for k, c in enumerate(coeffs):
        power = degree - k
        if c == 0:
            continue
        mono = "" if power == 0 else (var if power == 1 else var + {2: "²", 3: "³"}[power])
        mag = abs(c)
        body = (str(mag) if mag != 1 or power == 0 else "") + mono
        if not out:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out


def _fraction_answer(value):
    value = Fraction(value)
    return dict(
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=rational_text(value),
        answer_latex=rational_text(value),
    )


def _check_limit(fn, value, what, n=10**12):
    """Evaluate the n-th term very far out in 60-digit arithmetic; it must agree with the exact limit."""
    with mp.workdps(60):
        got = fn(mp.mpf(n))
        target = mp.mpf(value.numerator) / value.denominator
        check(abs(got - target) < mp.mpf(10) ** -6 * max(1, abs(target)), f"{what}: a(n) = {got}, expected {target}")


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    degree = rng.choice([2, 2, 3])
    lower = rng.random() < 0.2
    for _ in attempts(500, "a polynomial ratio"):
        top = [rng.choice([v for v in range(-6, 8) if v]) for _ in range(1)] + [
            rng.randint(-6, 7) for _ in range(degree)
        ]
        bottom = [rng.choice([v for v in range(-6, 8) if v])] + [rng.randint(-6, 7) for _ in range(degree)]
        if lower:
            top = top[1:]  # one degree lower than the denominator
            top[0] = top[0] or 1
        if top[0] == 0 or bottom[0] == 0 or len(set(map(abs, (top[0], bottom[0])))) < 2 and not lower:
            continue
        # keep the second coefficient pair distinct so the "next coefficient" slip is a different number
        if not lower and top[1] * bottom[0] == bottom[1] * top[0]:
            continue
        break
    value = Fraction(0) if lower else Fraction(top[0], bottom[0])

    def term(n):
        num = sum(c * n ** (len(top) - 1 - k) for k, c in enumerate(top))
        den = sum(c * n ** (len(bottom) - 1 - k) for k, c in enumerate(bottom))
        return num / den

    _check_limit(term, value, "polynomial ratio")
    if lower:
        cands = [
            frac_d(Fraction(top[0], bottom[0]), "seqlim-leading-ratio-unequal-degree"),
            frac_d(Fraction(top[0], bottom[1]) if bottom[1] else Fraction(1), "seqlim-leading-ratio-unequal-degree"),
            frac_d(Fraction(sum(top), sum(bottom)) if sum(bottom) else Fraction(2), "seqlim-substitute-n-equals-one"),
        ]
    else:
        cands = [
            frac_d(Fraction(bottom[0], top[0]), "seqlim-ratio-inverted"),
            frac_d(Fraction(top[1], bottom[1]) if bottom[1] else Fraction(1), "seqlim-next-coefficient"),
            frac_d(Fraction(sum(top), sum(bottom)) if sum(bottom) else Fraction(2), "seqlim-substitute-n-equals-one"),
            frac_d(Fraction(top[0]), "seqlim-ignores-denominator"),
        ]
    text = f"Find the limit as n tends to infinity of ({_poly_text(top)}) / ({_poly_text(bottom)})."
    if lower:
        working = [
            f"Divide top and bottom by n{'²' if degree == 2 else '³'}: the numerator tends to 0 and the denominator to {bottom[0]}.",
            "The limit is 0.",
        ]
    else:
        working = [
            f"Divide top and bottom by n{'²' if degree == 2 else '³'}: every term except the leading one tends to 0.",
            f"The limit is {top[0]}/{bottom[0]} = {rational_text(value)}.",
        ]
    return dict(text=text, working=working, distractors=keep(rational_text(value), cands, 4), **_fraction_answer(value))


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_exponential(rng):
    for _ in attempts(500, "an exponential ratio"):
        big = rng.choice([4, 5, 6])
        lo1, lo2 = rng.sample([2, 3, 4], 2)
        if big in (lo1, lo2):
            continue
        a, b, c, d = (rng.choice([v for v in range(-6, 8) if abs(v) > 1]) for _ in range(4))
        if lo1 == lo2 or abs(a) == abs(c):
            continue
        break
    value = Fraction(a, c)

    def term(n):
        return (a * mp.mpf(big) ** n + b * mp.mpf(lo1) ** n) / (c * mp.mpf(big) ** n + d * mp.mpf(lo2) ** n)

    _check_limit(term, value, "exponential ratio", n=400)
    cands = [
        frac_d(Fraction(b, d), "seqlim-picks-smaller-base"),
        frac_d(Fraction(a + b, c + d) if c + d else Fraction(3), "seqlim-substitute-n-equals-one"),
        frac_d(Fraction(c, a), "seqlim-ratio-inverted"),
        frac_d(Fraction(a * big, c * big + d), "seqlim-ignores-denominator"),
    ]
    text = f"Find the limit as n tends to infinity of ({_exp_text(a, big, b, lo1)}) / ({_exp_text(c, big, d, lo2)})."
    working = [
        f"Divide top and bottom by {big}ⁿ. Since ({lo1}/{big})ⁿ and ({lo2}/{big})ⁿ tend to 0, the limit is {a}/{c} = {rational_text(value)}."
    ]
    return dict(text=text, working=working, distractors=keep(rational_text(value), cands, 4), **_fraction_answer(value))


def _exp_text(a, base1, b, base2):
    return f"{a}·{base1}ⁿ {'+' if b > 0 else '-'} {abs(b)}·{base2}ⁿ"


def _term(c, suffix) -> str:
    """' + 3/n' or ' - 3/n' (nothing for 0): a signed term following another."""
    return "" if c == 0 else (f" + {c}{suffix}" if c > 0 else f" - {-c}{suffix}")


def _n(k):
    """k n, written as 'n' when k is 1 and '-n' when k is -1."""
    return "n" if k == 1 else "-n" if k == -1 else f"{k}n"


def _medium_root(rng):
    if rng.random() < 0.5:
        a = rng.choice([v for v in range(-9, 10) if v and v % 2 == 0 or v in (3, 5, 7, -3, -5)])
        b = rng.randint(1, 9)
        value = Fraction(a, 2)

        def term(n):
            return mp.sqrt(n * n + a * n + b) - n

        _check_limit(term, value, "root minus n")
        cands = [
            frac_d(Fraction(a), "seqlim-root-conjugate-dropped-half"),
            frac_d(Fraction(0), "seqlim-infinity-minus-infinity-is-zero"),
            frac_d(Fraction(a, 2) + b, "seqlim-root-constant-term"),
            frac_d(Fraction(b, 2), "seqlim-root-constant-term"),
        ]
        text = f"Find the limit as n tends to infinity of √(n² {'+' if a > 0 else '-'} {_n(abs(a))} + {b}) - n."
        working = [
            "Multiply and divide by the conjugate √(n² + ...) + n: the numerator becomes " + f"{join_terms([(a, 'n'), (b, '')])}.",
            f"Dividing top and bottom by n gives ({a}{_term(b, '/n')}) / (√(1{_term(a, '/n')}{_term(b, '/n²')}) + 1) → {a}/2 = {rational_text(value)}.",
        ]
    else:
        a, b = rng.sample([v for v in range(-8, 9) if v], 2)
        value = Fraction(a - b, 2)

        def term(n):
            return mp.sqrt(n * n + a * n) - mp.sqrt(n * n + b * n)

        _check_limit(term, value, "difference of roots")
        cands = [
            frac_d(Fraction(a - b), "seqlim-root-conjugate-dropped-half"),
            frac_d(Fraction(0), "seqlim-infinity-minus-infinity-is-zero"),
            frac_d(Fraction(a + b, 2), "seqlim-root-sum-for-difference"),
            frac_d(Fraction(b - a, 2), "seqlim-ratio-inverted"),
        ]
        text = (
            f"Find the limit as n tends to infinity of √(n² {'+' if a > 0 else '-'} {_n(abs(a))}) - "
            f"√(n² {'+' if b > 0 else '-'} {_n(abs(b))})."
        )
        working = [
            "Multiply and divide by the conjugate: the numerator becomes " + f"{a - b}n.",
            f"Dividing top and bottom by n gives {a - b} / (√(1{_term(a, '/n')}) + √(1{_term(b, '/n')})) → {a - b}/2 = {rational_text(value)}.",
        ]
    return dict(text=text, working=working, distractors=keep(rational_text(value), cands, 4), **_fraction_answer(value))


def _medium_squeeze(rng):
    d, e = (rng.choice([v for v in range(-6, 8) if v]) for _ in range(2))
    f = rng.randint(1, 9)
    c = rng.randint(2, 9)
    osc = rng.choice(["sin(n²)", "cos(n)", "(-1)ⁿ", "sin(n)"])
    value = Fraction(d, e)

    def osc_fn(n):
        return {"sin(n²)": mp.sin(n * n), "cos(n)": mp.cos(n), "(-1)ⁿ": (-1) ** int(n), "sin(n)": mp.sin(n)}[osc]

    def term(n):
        return (d * n + c * osc_fn(n)) / (e * n + f)

    _check_limit(term, value, "squeeze ratio", n=10**12 + 1)
    cands = [
        frac_d(Fraction(c), "seqlim-oscillation-treated-as-limit"),
        frac_d(Fraction(d + c, e) if e else Fraction(2), "seqlim-oscillation-treated-as-limit"),
        frac_d(Fraction(0), "seqlim-oscillation-no-limit-claimed"),
        frac_d(Fraction(e, d), "seqlim-ratio-inverted"),
    ]
    text = (
        f"Find the limit as n tends to infinity of ({_n(d)} {'+' if c > 0 else '-'} {abs(c)}{osc}) / "
        f"({_n(e)} {'+' if f > 0 else '-'} {abs(f)})."
    )
    working = [
        f"Since -1 ≤ {osc} ≤ 1, the numerator lies between {_n(d)} - {c} and {_n(d)} + {c}.",
        f"Dividing by n: both bounds give a ratio tending to {d}/{paren_negative(e)}, so by the squeeze theorem the limit is {rational_text(value)}.",
    ]
    return dict(text=text, working=working, distractors=keep(rational_text(value), cands, 4), **_fraction_answer(value))


def _medium_linear_recurrence(rng):
    for _ in attempts(500, "a contracting linear recurrence"):
        den = rng.choice([2, 3, 4, 5])
        num = rng.choice([v for v in range(-(den - 1), den) if v])
        q = rng.randint(1, 9)
        a1 = rng.randint(-5, 9)
        p = Fraction(num, den)
        limit = Fraction(q) / (1 - p)
        if limit != a1:
            break
    # Iterate far out in exact fractions.
    value = Fraction(a1)
    for _ in range(80):
        value = p * value + q
    check(abs(value - limit) < Fraction(1, 10**4), "iteration does not approach the limit")
    cands = [
        frac_d(Fraction(q) / (1 + p) if p != -1 else Fraction(0), "seqlim-fixed-point-sign"),
        frac_d(Fraction(q) * (1 - p), "seqlim-fixed-point-multiplied"),
        frac_d(Fraction(a1), "seqlim-limit-is-first-term"),
        frac_d(Fraction(q), "seqlim-ignores-multiplier"),
    ]
    text = f"The sequence is defined by a{sub('1')} = {a1} and {A_N1} = {_lin_text(p, q)}. Find the limit of {A_N}."
    working = [
        f"Suppose {A_N} → L. Then L = ({rational_text(p)})L + {q}, so L(1 {'-' if p > 0 else '+'} {rational_text(abs(p))}) = {q}.",
        f"L = {rational_text(limit)}. (The sequence approaches this value because |{rational_text(p)}| < 1.)",
    ]
    return dict(text=text, working=working, distractors=keep(rational_text(limit), cands, 4), **_fraction_answer(limit))


def _lin_text(p, q):
    if p.denominator == 1:
        coef = "" if p == 1 else ("-" if p == -1 else str(p.numerator))
        lead = f"{coef}{A_N}"
    else:
        lead = f"({p.numerator}/{p.denominator}){A_N}"
    return f"{lead} {'+' if q > 0 else '-'} {abs(q)}"


def _medium(rng):
    return rng.choice([_medium_exponential, _medium_root, _medium_squeeze, _medium_linear_recurrence])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------


def _iterate(fn, a1, steps=400):
    value = mp.mpf(a1)
    for _ in range(steps):
        value = fn(value)
    return value


def _hard_quadratic_recurrence(rng):
    for _ in attempts(500, "a recurrence with two fixed points"):
        r1, r2 = sorted(rng.sample(range(1, 8), 2))
        a1 = rng.randint(0, r2 - 1)
        if a1 == r1 or (r2 - r1 == 1 and a1 == 0):
            continue
        break
    s, c = r1 + r2, r1 * r2
    value = Fraction(r1)
    with mp.workdps(40):
        got = _iterate(lambda a: (a * a + c) / s, a1)
        check(abs(got - r1) < mp.mpf(10) ** -8, "iteration does not approach the smaller root")
    cands = [
        int_d(r2, "seqlim-wrong-fixed-point-root"),
        int_d(a1, "seqlim-limit-is-first-term"),
        int_d(-r1, "seqlim-fixed-point-sign"),
        frac_d(Fraction(c, s), "seqlim-fixed-point-multiplied"),
    ]
    text = (
        f"A sequence is defined by a{sub('1')} = {a1} and {A_N1} = ({A_N}² + {c})/{s}. "
        f"The sequence is monotonic and bounded, so it converges. Find its limit."
    )
    working = [
        f"If {A_N} → L then L = (L² + {c})/{s}, so L² - {s}L + {c} = 0 and L = {r1} or L = {r2}.",
        f"Since a{sub('1')} = {a1} is below {r2}"
        + (
            f" and above {r1}, the terms decrease towards {r1}."
            if a1 > r1
            else f", the terms increase but cannot pass {r1}."
        ),
        f"The limit is {r1}.",
    ]
    return dict(text=text, working=working, distractors=keep(str(r1), cands, 4), **_fraction_answer(value))


def _hard_root_recurrence(rng):
    m = rng.choice([2, 3, 4, 5, 6])
    k = m * (m - 1)
    a1 = rng.choice([v for v in range(1, m + 3) if v != m])
    with mp.workdps(40):
        got = _iterate(lambda a: mp.sqrt(k + a), a1, steps=200)
        check(abs(got - m) < mp.mpf(10) ** -10, "iteration does not approach the positive root")
    cands = [
        int_d(-(m - 1), "seqlim-wrong-fixed-point-root"),
        int_d(a1, "seqlim-limit-is-first-term"),
        int_d(k, "seqlim-fixed-point-squared"),
        int_d(m + 1, "seqlim-fixed-point-squared"),
    ]
    text = (
        f"A sequence is defined by a{sub('1')} = {a1} and {A_N1} = √({k} + {A_N}). "
        "It is monotonic and bounded. Find its limit."
    )
    working = [
        f"If {A_N} → L (and L ≥ 0, since every term is positive) then L = √({k} + L), so L² - L - {k} = 0.",
        f"The roots are L = {m} and L = {-(m - 1)}; the negative root is impossible, so the limit is {m}.",
    ]
    return dict(text=text, working=working, distractors=keep(str(m), cands, 4), **_fraction_answer(Fraction(m)))


def _hard_newton(rng):
    m = rng.choice([2, 3, 4, 5, 6])
    a1 = rng.choice([v for v in range(1, 12) if v != m])
    with mp.workdps(40):
        got = _iterate(lambda a: (a + m * m / a) / 2, a1, steps=100)
        check(abs(got - m) < mp.mpf(10) ** -10, "Newton iteration does not approach the root")
    cands = [
        int_d(-m, "seqlim-wrong-fixed-point-root"),
        int_d(a1, "seqlim-limit-is-first-term"),
        frac_d(Fraction(m * m, 2), "seqlim-fixed-point-squared"),
        int_d(m * m, "seqlim-fixed-point-squared"),
    ]
    text = (
        f"A sequence is defined by a{sub('1')} = {a1} and {A_N1} = ({A_N} + {m * m}/{A_N})/2. "
        "It is bounded below by 0 and, after the first term, decreasing. Find its limit."
    )
    working = [
        f"If {A_N} → L > 0 then L = (L + {m * m}/L)/2, so L = {m * m}/L and L² = {m * m}.",
        f"All terms are positive, so L = {m}.",
    ]
    return dict(text=text, working=working, distractors=keep(str(m), cands, 4), **_fraction_answer(Fraction(m)))


def _hard(rng):
    return rng.choice([_hard_quadratic_recurrence, _hard_root_recurrence, _hard_newton, _hard_quadratic_recurrence])(
        rng
    )


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_sequence_limits(rng, tier="easy"):
    """Limits of sequences: dominant terms, squeeze, monotone bounded recursion; tiers in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_sequence_limits(rng, tier="exam"):
    """a(1) given, a(n+1) = (a(n)^2 + c)/s: the fixed points, a(2), the limit, and a limit that uses it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a recurrence with two fixed points"):
        r1, r2 = sorted(rng.sample(range(1, 8), 2))
        a1 = rng.randint(1, r2 - 1)
        if a1 == r1 or r1 + r2 > 11:
            continue
        break
    s, c = r1 + r2, r1 * r2
    a2 = Fraction(a1 * a1 + c, s)
    with mp.workdps(40):
        got = _iterate(lambda a: (a * a + c) / s, a1)
        check(abs(got - r1) < mp.mpf(10) ** -8, "iteration does not approach the smaller root")

        def d_term(n):
            return (2 * n + 1) * _iterate(lambda a: (a * a + c) / s, a1, 300) / (n + 3)

        check(abs(d_term(mp.mpf(10) ** 15) - 2 * r1) < mp.mpf(10) ** -6, "limit of the combined sequence differs")

    def list_dist(values, tag):
        text = ", ".join(str(v) for v in values)
        return Distractor(text, tag, list(values))

    a_cands = [
        list_dist(sorted([-r1, -r2]), "seqlim-fixed-point-sign"),
        list_dist(sorted([r1, -r2]), "seqlim-fixed-point-sign"),
        list_dist([0, s], "seqlim-fixed-point-multiplied"),
    ]
    b_cands = [
        frac_d(Fraction(a1 * a1) + Fraction(c, s), "seqlim-precedence-in-recurrence"),
        frac_d(Fraction(a1 + c, s), "seqlim-square-dropped"),
        frac_d(Fraction(a1 * a1 + c), "seqlim-divisor-dropped"),
    ]
    c_cands = [
        int_d(r2, "seqlim-wrong-fixed-point-root"),
        int_d(a1, "seqlim-limit-is-first-term"),
        frac_d(a2, "seqlim-limit-is-first-term"),
    ]
    d_cands = [
        int_d(r1, "seqlim-combined-limit-factor-dropped"),
        int_d(r1 * 3, "seqlim-combined-limit-factor-dropped"),
        frac_d(Fraction(2 * r1, 3), "seqlim-combined-limit-uses-constants"),
    ]
    list_text = ", ".join(str(v) for v in (r1, r2))
    return dict(
        text=rng.choice(
            [
                f"A sequence is defined by a{sub('1')} = {a1} and {A_N1} = ({A_N}² + {c})/{s}. "
                "It can be shown that the sequence is monotonic and bounded, so it converges.",
                f"Let ({A_N}) be a sequence with a{sub('1')} = {a1} and {A_N1} = ({A_N}² + {c})/{s}. "
                "You may assume that it is monotonic and bounded, so it converges.",
                f"Consider the sequence given by a{sub('1')} = {a1} and {A_N1} = ({A_N}² + {c})/{s}. "
                "This sequence is known to be monotonic and bounded, hence convergent.",
            ]
        ),
        parts=[
            dict(
                prompt=f"Suppose {A_N} → L. Find all possible values of L.",
                answer_kind="list",
                answer=[r1, r2],
                answer_text=list_text,
                marks=2,
                working=[f"L = (L² + {c})/{s} gives L² - {s}L + {c} = 0, so L = {r1} or L = {r2}."],
                distractors=keep(list_text, a_cands),
            ),
            dict(
                prompt=f"Find a{sub('2')}.",
                marks=1,
                working=[f"a{sub('2')} = ({a1}² + {c})/{s} = {rational_text(a2)}."],
                distractors=keep(rational_text(a2), b_cands),
                **_fraction_answer(a2),
            ),
            dict(
                prompt=f"Find the limit of {A_N}.",
                marks=3,
                working=[
                    f"a{sub('1')} = {a1} is below {r2}, so the terms stay below {r2} and approach {r1}"
                    + ("." if a1 < r1 else " from above."),
                    f"The limit is {r1}.",
                ],
                distractors=keep(str(r1), c_cands),
                **_fraction_answer(Fraction(r1)),
            ),
            dict(
                prompt=f"Hence find the limit of (2n + 1){A_N}/(n + 3).",
                marks=2,
                working=[f"(2n + 1)/(n + 3) → 2 and {A_N} → {r1}, so the limit is 2 × {r1} = {2 * r1}."],
                distractors=keep(str(2 * r1), d_cands),
                follow_through="2*part_c",
                **_fraction_answer(Fraction(2 * r1)),
            ),
        ],
    )
