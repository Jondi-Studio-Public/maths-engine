"""Generates `s2b2_recurrence_diagonalisation` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_recurrence_diagonalisation`.

``uc2-recurrence-by-diagonalisation``: a linear system xₙ₊₁ = A xₙ is solved by diagonalising A = P D P⁻¹, so that
xₙ = P Dⁿ P⁻¹ x₀; the long-run behaviour is read from the eigenvalue of largest modulus.

* easy (``expression``): a two-state Markov chain given by a column-stochastic matrix with entries in tenths and an
  initial vector of counts; the long-run count in one named state, which is the total times that share of the steady
  state.
* medium (``expression``): a coupled pair aₙ₊₁ = …, bₙ₊₁ = … with integer eigenvalues; the closed form of aₙ or bₙ,
  c₁λ₁ⁿ + c₂λ₂ⁿ.
* hard: either the closed form of one component of a coupled system of three sequences (``expression``, three
  exponentials), or the long-run count at one named station of a three-state Markov chain with a positive stochastic
  matrix whose other two eigenvalues are rational (a whole number, also ``expression``).
* exam (a two-state chain): (a) the second eigenvalue 1 - p - q, (b) the steady-state share of state 1, (c) the closed
  form of the first component, (d) its long-run limit, which follows through from (b).

Every key is checked by iterating the recurrence with exact fractions and comparing with the closed form at n = 0..8,
and the limits are checked against A¹⁰⁰ x₀ computed exactly.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, latex_answer, phrase, pick_distractors, rational_text
from ._s2b2_kit import (
    tidied,
    apply,
    conjugate,
    fm,
    inverse,
    jordan_matrix,
    mat_str,
    mmul,
    nullspace,
    number_distractor,
    primitive,
    vec_str,
)
from ._ucv1_linalg import column_value

N = sympy.Symbol("n")
_SUB = {"n": "ₙ", "+": "₊"}
_NAMES = ("x", "y", "z")


# -- text of a closed form ----------------------------------------------------------------------------------------


def _base_text(base: Fraction) -> str:
    if base.denominator == 1:
        return f"{base.numerator}^n" if base > 0 else f"(-{-base.numerator})^n"
    sign = "-" if base < 0 else ""
    return f"({sign}{abs(base.numerator)}/{base.denominator})^n"


def _term_text(coef: Fraction, base: Fraction) -> str:
    c = rational_text(coef) if coef.denominator != 1 else str(coef.numerator)
    if base == 1:
        return c
    b = _base_text(base)
    if coef == 1:
        return b
    if coef == -1:
        return "-" + b
    if coef.denominator != 1:
        c = f"({c})"
    return f"{c}*{b}"


def closed_form_text(terms) -> str:
    """`Σ c·λⁿ` from `[(c, λ), …]`, largest |λ| first and the constant term last, with sign-aware joining."""
    merged: dict[Fraction, Fraction] = {}
    for c, b in terms:
        merged[Fraction(b)] = merged.get(Fraction(b), Fraction(0)) + Fraction(c)
    ordered = sorted(((c, b) for b, c in merged.items() if c != 0), key=lambda t: (-abs(t[1]), -t[1]))
    # A constant term is last.
    ordered.sort(key=lambda t: t[1] == 1)
    pieces = [_term_text(c, b) for c, b in ordered]
    if not pieces:
        return "0"
    out = pieces[0]
    for piece in pieces[1:]:
        out += f" - {piece[1:]}" if piece.startswith("-") else f" + {piece}"
    return out


def closed_form_expr(terms):
    return sum(
        (
            sympy.Rational(c.numerator, c.denominator) * sympy.Rational(b.numerator, b.denominator) ** N
            for c, b in map(lambda t: (Fraction(t[0]), Fraction(t[1])), terms)
        ),
        sympy.Integer(0),
    )


def _decompose(p, lams, x0, row):
    """Coefficients `c_k = P[row][k] · (P⁻¹ x₀)[k]` of component `row` of xₙ = P Dⁿ P⁻¹ x₀."""
    w = apply(inverse(p), x0)
    return [p[row][k] * w[k] for k in range(len(lams))]


def _verify_closed_form(a, x0, row, terms):
    """Iterate xₙ₊₁ = A xₙ exactly and compare component `row` with the closed form at n = 0..8."""
    x = [Fraction(v) for v in x0]
    expr = closed_form_expr(terms)
    for n in range(9):
        check(
            sympy.Rational(x[row].numerator, x[row].denominator) == expr.subs(N, n), f"closed form differs at n = {n}"
        )
        x = apply(a, x)


def _sub(text: str) -> str:
    return "".join(_SUB.get(ch, ch) for ch in text)


def _system_text(a, names):
    rows = []
    for i, name in enumerate(names):
        terms = [(int(a[i][j]), f"{names[j]}ₙ") for j in range(len(names))]
        rows.append(f"{name}ₙ₊₁ = {join_terms(terms)}")
    return ", ".join(rows)


def _initial_text(names, x0):
    return ", ".join(f"{name}₀ = {v}" for name, v in zip(names, x0))


# -- closed-form distractors --------------------------------------------------------------------------------------


def _closed_form_options(terms, x0, row, p, lams):
    """Tagged wrong closed forms from the terms `(c, λ)`."""
    terms = [(Fraction(c), Fraction(b)) for c, b in terms]
    key = closed_form_expr(terms)
    candidates = []
    # Each coefficient with the other's eigenvalue.
    if len(terms) >= 2:
        rotated = [(terms[i][0], terms[(i + 1) % len(terms)][1]) for i in range(len(terms))]
        candidates.append((rotated, "rec-eigenvalues-swapped"))
    # An exponent one too high, from starting the index at 1.
    candidates.append(([(c * b, b) for c, b in terms], "rec-index-shift"))
    # The coefficients taken as the initial values themselves.
    if len(terms) == len(x0):
        candidates.append(
            ([(Fraction(x0[k]), terms[k][1]) for k in range(len(terms))], "rec-coefficients-from-initial-values")
        )
    # One eigenvalue with the wrong sign.
    flipped = [(c, -b if k == 0 else b) for k, (c, b) in enumerate(terms)]
    candidates.append((flipped, "sign-flip-one-root"))
    # The eigenvalues used as the coefficients and the coefficients as the bases.
    if all(c.denominator == 1 and c != 0 and abs(c) < 20 for c, _b in terms):
        candidates.append(([(b, c) for c, b in terms if c != 0], "rec-coefficient-base-confused"))
    # The coefficients not weighted by the components of P: c_k read from P⁻¹ x₀ only.
    w = apply(inverse(p), x0)
    candidates.append(([(w[k], terms[k][1]) for k in range(len(terms))], "rec-eigenvector-weight-dropped"))
    out = []
    for cand, tag in candidates:
        if closed_form_expr(cand) == key or any(b == 0 for _c, b in cand):
            continue
        text = closed_form_text(cand)
        out.append(Distractor(text, tag, text))
    return out


# -- easy: a two-state Markov chain -------------------------------------------------------------------------------


_EASY_STEMS = (
    "Customers of two online grocery services, {u} and {v}, change service each week. The weekly changes are given by "
    "xₙ₊₁ = A xₙ, where xₙ lists the numbers of customers of ({u}, {v}) after n weeks, A = {m}, and "
    "x₀ = {x0}. Find the number of {w} customers that the weekly count approaches as n grows large.",
    "Each week customers of {u} and {v} swap between the two services according to xₙ₊₁ = A xₙ, where "
    "A = {m} and x₀ = {x0} lists the numbers of customers of ({u}, {v}) now. Find the limit of the number of {w} "
    "customers as n → ∞.",
)
_SERVICES = (("Greenbox", "Freshpack"), ("Meadow Meals", "Harvest Direct"), ("Outback Fresh", "Coastline Grocer"))


def _tenths(m):
    """A matrix of Fractions in tenths as the decimals a learner reads: [[0.8, 0.3], [0.2, 0.7]]."""
    return "[" + ", ".join("[" + ", ".join(f"{float(v):.1f}" for v in row) + "]" for row in m) + "]"


def _markov_2x2(rng):
    for _ in attempts(2000, "a two-state chain with integer limit counts"):
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        if p + q == 10 or p == q:
            continue
        s = p + q
        total = rng.choice(range(200, 1301, 50))
        if (total * q) % s or (total * p) % s:
            continue
        first = rng.choice(range(1, 10)) * total // 10
        if first * s == total * q:
            continue
        return p, q, total, first
    raise AssertionError  # pragma: no cover


def _vec_options(candidates, answer):
    """Tagged wrong vectors from `(vector, tag)`: none equal to the answer or repeated (graded exactly, no scaling)."""
    answer = [Fraction(v) for v in answer]
    out, seen = [], set()
    for vec, tag in candidates:
        vec = [Fraction(v) for v in vec]
        text = vec_str(vec)
        if vec == answer or text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, column_value(vec)))
    return out


def _far_future(a, x0):
    """A⁵¹² x₀ exactly, by repeated squaring."""
    power = a
    for _ in range(9):
        power = mmul(power, power)
    return apply(power, x0)


def _limit_check(a, x0, expected):
    """A⁵¹² x₀ is within 10⁻⁶ of the limit, computed exactly."""
    for got, want in zip(_far_future(a, x0), expected):
        check(abs(got - want) < Fraction(1, 10**6), "A^512 x0 is not close to the limit")


def _count_options(candidates, answer):
    """Tagged wrong counts from `(value, tag)`: whole numbers, none equal to the answer or repeated."""
    out, seen = [], {int(answer)}
    for value, tag in candidates:
        value = Fraction(value)
        if value.denominator != 1 or int(value) in seen:
            continue
        seen.add(int(value))
        out.append(Distractor(str(int(value)), tag, str(int(value))))
    return out


def _easy(rng):
    p, q, total, first = _markov_2x2(rng)
    s = p + q
    a = fm([[Fraction(10 - p, 10), Fraction(q, 10)], [Fraction(p, 10), Fraction(10 - q, 10)]])
    x0 = [first, total - first]
    limit = [Fraction(total * q, s), Fraction(total * p, s)]
    _limit_check(a, x0, limit)
    u, v = rng.choice(_SERVICES)
    i = rng.randrange(2)
    other = 1 - i
    key = int(limit[i])
    text = str(key)
    ratio = [Fraction(total * q, 10), Fraction(total * p, 10)]
    candidates = [
        (x0[i], "markov-long-run-initial"),
        (Fraction(total, 2), "markov-equal-split"),
        (limit[other], "markov-state-swapped"),
        (ratio[i], "markov-ratio-not-normalised"),
        (x0[other], "markov-long-run-initial"),
    ]
    distractors = pick_distractors(text, _count_options(candidates, key), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _EASY_STEMS, m=_tenths(a), x0=vec_str(x0), u=u, v=v, w=(u, v)[i]),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=text,
        working=[
            f"A is column-stochastic, so it has eigenvalue 1; the other eigenvalue is 1 - {p / 10:g} - {q / 10:g} = {(10 - s) / 10:g}.",
            f"The eigenvalue 1 has eigenvector ({q}, {p}); the other has modulus less than 1, so xₙ tends to {total} times the steady "
            f"state, which is {vec_str(limit)}. The {(u, v)[i]} entry is {text}.",
        ],
        distractors=distractors,
    )


# -- medium and hard: integer coupled systems ---------------------------------------------------------------------

_MEDIUM_STEMS = (
    "The sequences {names} satisfy {system}, with {initial}. Find a closed form for {target}ₙ.",
    "Two sequences are defined by {system}, with {initial}. Find a closed form for {target}ₙ in terms of n.",
)
_HARD_STEMS = (
    "The sequences {names} satisfy {system}, with {initial}. Find a closed form for {target}ₙ.",
    "Three sequences are defined by {system}, with {initial}. Find a closed form for {target}ₙ in terms of n.",
)
_EIGENVALUES = [-2, -1, 1, 2, 3, 4]


def _integer_system(rng, size, cap):
    for _ in attempts(2000, "an integer system with distinct integer eigenvalues and non-zero coefficients"):
        lams = sorted(rng.sample(_EIGENVALUES, size))
        a, p = conjugate(rng, jordan_matrix([(l, 1) for l in lams]), cap=cap)
        x0 = [rng.randint(-3, 4) for _ in range(size)]
        if all(v == 0 for v in x0):
            continue
        row = rng.randrange(size)
        coefficients = _decompose(p, lams, x0, row)
        if any(c == 0 for c in coefficients) or max(abs(c) for c in coefficients) > 12:
            continue
        return a, p, lams, x0, row, coefficients
    raise AssertionError  # pragma: no cover


def _closed_form_question(rng, size, stems, cap):
    a, p, lams, x0, row, coefficients = _integer_system(rng, size, cap)
    terms = list(zip(coefficients, lams))
    _verify_closed_form(a, x0, row, terms)
    names = _NAMES[:size]
    text = closed_form_text(terms)
    options = _closed_form_options(terms, x0, row, p, lams)
    distractors = pick_distractors(text, options, 4)
    check(len(distractors) >= 3, "too few distractors")
    expr = closed_form_expr([(Fraction(c), Fraction(b)) for c, b in terms])
    return dict(
        text=phrase(
            rng,
            stems,
            names=", ".join(f"{n}ₙ" for n in names),
            system=_system_text(a, names),
            initial=_initial_text(names, x0),
            target=names[row],
        ),
        answer_kind="expression",
        answer=str(expr).replace("**", "^"),
        answer_text=text,
        answer_latex=latex_answer(expr),
        working=[
            f"The matrix of the system is A = {mat_str(a)}, with eigenvalues {', '.join(map(str, lams))}.",
            f"Writing x₀ in eigenvectors, {names[row]}ₙ = {text}.",
        ],
        distractors=distractors,
    )


def _medium(rng):
    return _closed_form_question(rng, 2, _MEDIUM_STEMS, cap=9)


# -- hard: three-state Markov chain -------------------------------------------------------------------------------

_MARKOV_STEMS = (
    "A bike-share scheme has three stations P, Q and R. Each day the bikes at each station are redistributed: "
    "xₙ₊₁ = A xₙ, where xₙ lists the numbers of bikes at (P, Q, R) after n days, A = {m} (column j gives where the "
    "bikes at station j end up) and x₀ = {x0}. Find the number of bikes at station {w} that the daily count "
    "approaches as n grows large.",
    "The numbers of bikes at three stations P, Q, R after n days satisfy xₙ₊₁ = A xₙ with A = {m} and x₀ = {x0}. "
    "Find the limit of the number of bikes at station {w} as n → ∞.",
)


def _rational_sqrt(value: Fraction):
    num, den = value.numerator, value.denominator
    if num < 0:
        return None
    rn, rd = isqrt(num), isqrt(den)
    return Fraction(rn, rd) if rn * rn == num and rd * rd == den else None


def _markov_3x3(rng):
    """(A, steady, eigenvalues): a positive column-stochastic matrix in tenths with rational eigenvalues."""
    for _ in attempts(20000, "a 3×3 stochastic matrix with rational eigenvalues"):
        cols = []
        for _c in range(3):
            a1 = rng.randint(1, 8)
            a2 = rng.randint(1, 9 - a1)
            cols.append((a1, a2, 10 - a1 - a2))
        a = fm([[Fraction(cols[j][i], 10) for j in range(3)] for i in range(3)])
        trace = a[0][0] + a[1][1] + a[2][2]
        minors = sum(a[i][i] * a[j][j] - a[i][j] * a[j][i] for i, j in ((0, 1), (0, 2), (1, 2)))
        # (λ - 1)(λ² - (t - 1)λ + d) with d = m - (t - 1).
        d = minors - (trace - 1)
        disc = (trace - 1) ** 2 - 4 * d
        root = _rational_sqrt(disc)
        if root is None or root == 0:
            continue
        r1, r2 = ((trace - 1) + root) / 2, ((trace - 1) - root) / 2
        if r1 == 0 or r2 == 0 or abs(r1) >= 1 or abs(r2) >= 1:
            continue
        return a, [r1, r2]
    raise AssertionError  # pragma: no cover


def _steady_state(a):
    """The vector with sum 1 in the null space of A - I, exactly."""
    shifted = [[a[i][j] - (1 if i == j else 0) for j in range(3)] for i in range(3)]
    vec = nullspace(shifted)[0]
    total = sum(vec)
    return [v / total for v in vec]


def _hard_markov(rng):
    for _ in attempts(200, "a stochastic matrix with a small steady-state denominator"):
        a, others = _markov_3x3(rng)
        steady = _steady_state(a)
        denominator = 1
        for v in steady:
            denominator = denominator * v.denominator // gcd(denominator, v.denominator)
        if denominator <= 150 and len(set(steady)) > 1:
            break
    for _ in attempts(500, "an initial vector whose total makes the limit whole"):
        total = denominator * rng.randint(1, 60) * 10
        if total < 200 or total > 1500:
            continue
        x0 = [rng.randint(1, 8) * total // 10 for _ in range(2)]
        x0.append(total - sum(x0))
        if x0[2] <= 0:
            continue
        limit = [total * v for v in steady]
        if limit == [Fraction(v) for v in x0]:
            continue
        break
    check(all(v.denominator == 1 for v in limit), "limit counts are not whole")
    _limit_check(a, x0, limit)
    # SymPy: the eigenvalues are 1 and the two rational ones, and the steady state is an eigenvector.
    m = sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in a])
    eig = m.eigenvals()
    wanted = {sympy.Integer(1): 1}
    for r in others:
        wanted[sympy.Rational(r.numerator, r.denominator)] = (
            wanted.get(sympy.Rational(r.numerator, r.denominator), 0) + 1
        )
    check(eig == wanted, "eigenvalues differ from the construction")
    i = rng.randrange(3)
    key = int(limit[i])
    text = str(key)
    prim = primitive(steady)
    candidates = [
        (Fraction(total, 3), "markov-equal-split"),
        (x0[i], "markov-long-run-initial"),
        (prim[i], "markov-eigenvector-not-scaled"),
        (limit[(i + 1) % 3], "markov-state-swapped"),
        (limit[(i + 2) % 3], "markov-state-swapped"),
        (sum(x0) - key, "markov-long-run-initial"),
    ]
    distractors = pick_distractors(text, _count_options(candidates, key), 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _MARKOV_STEMS, m=_tenths(a), x0=vec_str(x0), w="PQR"[i]),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=text,
        working=[
            f"A is column-stochastic with positive entries, so its eigenvalue 1 has an eigenvector and the other eigenvalues "
            f"({rational_text(others[0])} and {rational_text(others[1])}) have modulus below 1.",
            f"The steady state is {vec_str(steady)}; xₙ tends to {total} times it, which is {vec_str(limit)}. "
            f"The entry for station {'PQR'[i]} is {text}.",
        ],
        distractors=distractors,
    )


def _hard(rng):
    if rng.random() < 0.5:
        return _closed_form_question(rng, 3, _HARD_STEMS, cap=8)
    return _hard_markov(rng)


@generator(
    "s2b2_recurrence_diagonalisation",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_recurrence_diagonalisation(rng, tier="easy"):
    """Closed forms and long-run limits of xₙ₊₁ = A xₙ by diagonalisation; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_recurrence_diagonalisation", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_recurrence_diagonalisation(rng, tier="exam"):
    """Exam chain on a two-state chain: the second eigenvalue, the steady-state share, the closed form, the limit."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p, q, total, first = _markov_2x2(rng)
    s = p + q
    a = fm([[Fraction(10 - p, 10), Fraction(q, 10)], [Fraction(p, 10), Fraction(10 - q, 10)]])
    x0 = [first, total - first]
    r = Fraction(10 - s, 10)
    share = Fraction(q, s)
    limit = Fraction(total * q, s)
    terms = [(Fraction(first) - limit, r), (limit, Fraction(1))]
    _verify_closed_form(a, x0, 0, terms)
    _limit_check(a, x0, [limit, total - limit])
    eig = sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in a]).eigenvals()
    check(set(eig) == {sympy.Integer(1), sympy.Rational(r.numerator, r.denominator)}, "SymPy's eigenvalues differ")
    text_c = closed_form_text(terms)
    expr_c = closed_form_expr(terms)
    u, v = rng.choice(_SERVICES)

    def decimal(f):
        return f"{float(f):g}"

    a_dist = pick_distractors(
        decimal(r),
        [
            Distractor(decimal(1 - Fraction(p, 10)), "markov-second-eigenvalue-one-entry", float(1 - Fraction(p, 10))),
            Distractor(decimal(Fraction(p + q, 10)), "markov-second-eigenvalue-sum", float(Fraction(p + q, 10))),
            Distractor(decimal(-r), "sign-flip-both-roots", float(-r)),
            Distractor(decimal(1 - Fraction(q, 10)), "markov-second-eigenvalue-one-entry", float(1 - Fraction(q, 10))),
            Distractor(decimal(1 + r), "eigval-trace-as-eigenvalue", float(1 + r)),
        ],
        3,
    )
    b_dist = pick_distractors(
        rational_text(share),
        [
            Distractor(rational_text(Fraction(p, s)), "markov-ratio-inverted", exact_value(Fraction(p, s))),
            Distractor("1/2", "markov-equal-split", exact_value(Fraction(1, 2))),
            Distractor(rational_text(Fraction(q, 10)), "markov-rate-as-share", exact_value(Fraction(q, 10))),
            Distractor(rational_text(Fraction(10 - p, 10)), "markov-rate-as-share", exact_value(Fraction(10 - p, 10))),
        ],
        3,
    )
    c_dist = pick_distractors(text_c, _closed_form_options_2state(terms, first), 3)
    d_dist = pick_distractors(
        str(int(limit)),
        [
            number_distractor(first, "markov-long-run-initial"),
            number_distractor(total - limit, "markov-state-swapped"),
            number_distractor(Fraction(total, 2), "markov-equal-split"),
            number_distractor(Fraction(total * p, s) + 0, "markov-ratio-inverted"),
        ],
        3,
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=(
            f"Customers of two online grocery services, {u} and {v}, change service each week. With xₙ listing the "
            f"numbers of customers of ({u}, {v}) after n weeks, xₙ₊₁ = A xₙ where A = {_tenths(a)} and "
            f"x₀ = {vec_str(x0)}."
        ),
        parts=[
            dict(
                prompt="A has eigenvalue 1. Find its other eigenvalue, as a decimal.",
                answer_kind="number",
                answer=float(r),
                answer_text=decimal(r),
                marks=2,
                working=[
                    f"The trace of A is {decimal(1 + r)}, so the other eigenvalue is {decimal(1 + r)} - 1 = {decimal(r)}."
                ],
                distractors=a_dist,
                sensible_range=(-1, 1),
            ),
            dict(
                prompt=f"Find the long-run proportion of customers who use {u}, as a fraction.",
                answer_kind="fraction",
                answer=exact_value(share),
                answer_text=rational_text(share),
                marks=2,
                working=[
                    f"The steady state is proportional to ({q}, {p}), so the proportion using {u} is {q}/{s} = {rational_text(share)}."
                ],
                distractors=b_dist,
                sensible_range=(0, 1),
            ),
            dict(
                prompt=f"Find a closed form for the number of {u} customers after n weeks, writing fractions as (a/b)^n.",
                answer_kind="expression",
                answer=str(expr_c).replace("**", "^"),
                answer_text=text_c,
                answer_latex=latex_answer(expr_c),
                marks=3,
                working=[f"xₙ = (steady part) + c·({rational_text(r)})ⁿ(1, -1); matching x₀ gives {text_c}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find the number of {u} customers that the closed form approaches as n grows large.",
                answer_kind="number",
                answer=int(limit),
                answer_text=str(int(limit)),
                marks=1,
                follow_through=f"{total}*part_b",
                working=[
                    f"The ({rational_text(r)})ⁿ term tends to 0, so the limit is {total} × {rational_text(share)} = {int(limit)}."
                ],
                distractors=d_dist,
            ),
        ],
    )


def _closed_form_options_2state(terms, first):
    """Wrong closed forms for the exam chain: the terms are `[(c, r), (S, 1)]`."""
    (c, r), (s_, _one) = terms
    candidates = [
        ([(s_, r), (c, Fraction(1))], "rec-eigenvalues-swapped"),
        ([(c * r, r), (s_, Fraction(1))], "rec-index-shift"),
        ([(Fraction(first), r), (s_, Fraction(1))], "rec-coefficients-from-initial-values"),
        ([(c, -r), (s_, Fraction(1))], "sign-flip-one-root"),
        ([(-c, r), (s_, Fraction(1))], "sign-error-moving-term"),
    ]
    out = []
    key = closed_form_expr(terms)
    for cand, tag in candidates:
        if closed_form_expr(cand) == key:
            continue
        text = closed_form_text(cand)
        out.append(Distractor(text, tag, text))
    return out
