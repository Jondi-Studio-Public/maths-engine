"""Generates `s2b2_matrix_exponential` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_matrix_exponential`.

``uc2-matrix-exponential`` (``linear-algebra``, answer ``expression``): the matrix exponential e^{At}, the fundamental
matrix of x' = Ax, with entries that are expressions in t. If A = P J P⁻¹ then e^{At} = P e^{Jt} P⁻¹, where a
diagonal block gives e^{λt} and a Jordan block gives e^{λt}(I + tN + t²N²/2! + …) with N the nilpotent part.

* easy: one entry of e^{At} for a dense 2×2 matrix with distinct integer eigenvalues (two exponentials).
* medium: one entry of e^{At} for a dense 2×2 matrix A = λI + N with N non-zero and nilpotent, e^{At} = e^{λt}(I + tN).
* hard: a 3×3 matrix: either one entry of e^{At}, or one component of the solution of x' = Ax, x(0) = x₀, where A has
  three distinct eigenvalues, or a block of size 2 (plus a 1×1 block), or a single block of size 3.
* exam: the 2×2 matrix of the medium tier: (a) λ, (b) and (c) two entries of e^{At}, (d) a component of the solution
  of x' = Ax, x(0) = x₀, which combines (b) and (c).

An entry is held exactly as `{λ: {m: c}}`, meaning Σ c tᵐ e^{λt}, with `Fraction` coefficients, so nothing here needs
SymPy until the final expression is written. A is P J P⁻¹ with unimodular P, so every coefficient is an integer or a
half-integer. Each key is checked two ways: the matrix E(t) built from P and J satisfies E' = AE and E(0) = I exactly,
and its entries match the Taylor series of e^{At} summed numerically at t = 1/3.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import VerificationError, attempts, check, generator, latex_answer, phrase, pick_distractors
from ._s2b1_linalg import plain_stem
from ._s2b2_kit import tidied, conjugate, inverse, jordan_matrix, mat_str, number_distractor

T = sympy.Symbol("t")
_LAMBDAS = [-3, -2, -1, 1, 2, 3, 4]

Groups = dict  # {lam: {power: Fraction}}


class NoT(dict):
    """A `Groups` value written with e^λ in place of e^(λt): the exponent without its t."""


# -- exact exponential polynomials ----------------------------------------------------------------------------------


def g_clean(g):
    out = {}
    for lam, poly in g.items():
        poly = {m: Fraction(c) for m, c in poly.items() if c != 0}
        if poly:
            out[int(lam)] = poly
    return out


def g_add(*groups, scales=None):
    out: dict = {}
    for k, g in enumerate(groups):
        s = Fraction(1) if scales is None else Fraction(scales[k])
        for lam, poly in g.items():
            target = out.setdefault(lam, {})
            for m, c in poly.items():
                target[m] = target.get(m, Fraction(0)) + s * c
    return g_clean(out)


def g_scale(g, s):
    return g_add(g, scales=[s])


def g_diff(g):
    out: dict = {}
    for lam, poly in g.items():
        target = out.setdefault(lam, {})
        for m, c in poly.items():
            target[m] = target.get(m, Fraction(0)) + lam * c
            if m:
                target[m - 1] = target.get(m - 1, Fraction(0)) + m * c
    return g_clean(out)


def g_eval(g, t):
    return sum(float(c) * t**m * math.exp(lam * t) for lam, poly in g.items() for m, c in poly.items())


def g_expr(g):
    scale = 1 if isinstance(g, NoT) else T
    return sum(
        (
            sympy.Rational(c.numerator, c.denominator) * T**m * sympy.exp(lam * scale)
            for lam, poly in g.items()
            for m, c in poly.items()
        ),
        sympy.Integer(0),
    )


def const(c):
    return g_clean({0: {0: Fraction(c)}})


def _exp_text(lam, show_t=True):
    if lam == 0:
        return ""
    if not show_t:
        return "e" if lam == 1 else f"e^({lam})"
    if lam == 1:
        return "e^t"
    if lam == -1:
        return "e^(-t)"
    return f"e^({lam}t)"


def _mono_text(c: Fraction, m: int) -> str:
    """|c| tᵐ as text (no sign)."""
    c = abs(c)
    num, den = c.numerator, c.denominator
    t = "" if m == 0 else ("t" if m == 1 else f"t^{m}")
    lead = "" if (num == 1 and m > 0) else str(num)
    return f"{lead}{t}" + (f"/{den}" if den != 1 else "")


def _poly_text(poly) -> str:
    terms = sorted(poly.items(), reverse=True)
    out = ""
    for k, (m, c) in enumerate(terms):
        body = _mono_text(c, m)
        if k == 0:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out


def _group_text(lam, poly, show_t=True) -> str:
    exp = _exp_text(lam, show_t)
    if lam == 0:
        return _poly_text(poly)
    if poly == {0: Fraction(1)}:
        return exp
    if poly == {0: Fraction(-1)}:
        return "-" + exp
    body = _poly_text(poly)
    return f"({body}){exp}" if len(poly) > 1 else f"{body}{exp}"


def g_text(g, show_t=None) -> str:
    """The sum written as a student would, largest λ first, constants last, signs joined."""
    if show_t is None:
        show_t = not isinstance(g, NoT)
    pieces = []
    for lam in sorted(g, key=lambda l: (l == 0, -l)):
        if g[lam]:
            pieces.append(_group_text(lam, g[lam], show_t))
    if not pieces:
        return "0"
    out = pieces[0]
    for piece in pieces[1:]:
        out += f" - {piece[1:]}" if piece.startswith("-") else f" + {piece}"
    return out


def g_equal(a, b) -> bool:
    return g_clean(a) == g_clean(b)


# -- the matrix exponential ---------------------------------------------------------------------------------------


def exponential(p, blocks):
    """E(t) = P e^{Jt} P⁻¹ as a matrix of `Groups`, for P a Fraction matrix and `blocks` a list of `(λ, size)`."""
    n = len(p)
    pinv = inverse(p)
    e = [[{} for _ in range(n)] for _ in range(n)]
    offset = 0
    for lam, size in blocks:
        for u in range(size):
            for v in range(u, size):
                power = v - u
                factor = Fraction(1, math.factorial(power))
                for i in range(n):
                    for j in range(n):
                        c = p[i][offset + u] * pinv[offset + v][j] * factor
                        if c:
                            cell = e[i][j].setdefault(lam, {})
                            cell[power] = cell.get(power, Fraction(0)) + c
        offset += size
    return [[g_clean(cell) for cell in row] for row in e]


def _series(a, t0, terms=60):
    n = len(a)
    at = [[float(a[i][j]) * t0 for j in range(n)] for i in range(n)]
    term = [[float(i == j) for j in range(n)] for i in range(n)]
    total = [row[:] for row in term]
    for k in range(1, terms):
        term = [[sum(term[i][m] * at[m][j] for m in range(n)) / k for j in range(n)] for i in range(n)]
        total = [[total[i][j] + term[i][j] for j in range(n)] for i in range(n)]
    return total


def verify(a, e):
    """E' = A E and E(0) = I exactly; and E(1/3) equals the Taylor series of e^{At} at t = 1/3."""
    n = len(a)
    for i in range(n):
        for j in range(n):
            lhs = g_diff(e[i][j])
            rhs = g_add(*[e[k][j] for k in range(n)], scales=[a[i][k] for k in range(n)])
            check(g_equal(lhs, rhs), "E' differs from A E")
            at_zero = sum((c for poly in e[i][j].values() for m, c in poly.items() if m == 0), Fraction(0))
            check(at_zero == (1 if i == j else 0), "E(0) is not the identity")
    t0 = 1 / 3
    series = _series(a, t0)
    for i in range(n):
        for j in range(n):
            value = g_eval(e[i][j], t0)
            check(abs(value - series[i][j]) < 1e-9 * max(1.0, abs(value)), "Taylor series differs from the formula")


def component(e, row, x0):
    """Component `row` of E(t) x₀ as `Groups`."""
    return g_add(*[e[row][k] for k in range(len(x0))], scales=list(x0))


def _wrong(key, items):
    """Tagged options from `(groups, tag)`, dropping any equal to `key` as a function or repeated."""
    out, seen = [], set()
    for groups, tag in items:
        if not isinstance(groups, NoT) and g_equal(groups, key):
            continue
        text = g_text(groups)
        if text == g_text(key) or text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
    return out


def _answer(key):
    expr = g_expr(key)
    return dict(
        answer_kind="expression",
        answer=str(expr).replace("**", "^"),
        answer_text=g_text(key),
        answer_latex=latex_answer(expr),
    )


def _entry_name(i, j):
    return f"({i + 1}, {j + 1})"


def lam_only(lam, poly):
    return {lam: {m: Fraction(c) for m, c in poly.items()}}


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Find the {pos} entry of e^(At) for A = {m}. Write e^(kt) for the exponential e^{{kt}}.",
    "For A = {m}, the matrix exponential e^(At) has entries that are expressions in t. Find its {pos} entry, writing "
    "e^(kt) for e^{{kt}}.",
    "Let A = {m}. Find the {pos} entry of e^(At) as an expression in t, writing e^(kt) for e^{{kt}}.",
)


def _easy(rng):
    for _ in attempts(500, "a diagonalisable 2×2 matrix with all entries of e^{At} using both exponentials"):
        l1, l2 = rng.sample(_LAMBDAS, 2)
        blocks = [(l1, 1), (l2, 1)]
        a, p = conjugate(rng, jordan_matrix(blocks), cap=10)
        e = exponential(p, blocks)
        if all(len(e[i][j]) == 2 for i in range(2) for j in range(2)):
            break
    verify(a, e)
    i, j = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    key = e[i][j]
    p_inv = inverse(p)
    d1, d2 = (l1, l2)
    # P⁻¹ e^{Dt} P in place of P e^{Dt} P⁻¹.
    reversed_form = g_add({d1: {0: p_inv[i][0] * p[0][j]}}, {d2: {0: p_inv[i][1] * p[1][j]}})
    wrong = [
        ({int(a[i][j]): {0: Fraction(1)}}, "matrix-exp-entrywise"),
        ({l1: key.get(l2, {}), l2: key.get(l1, {})}, "matrix-exp-eigenvalues-swapped"),
        (reversed_form, "diag-conjugation-reversed"),
        (NoT(key), "matrix-exp-t-dropped"),
        ({-lam: poly for lam, poly in key.items()}, "matrix-exp-sign-in-exponent"),
    ]
    distractors = pick_distractors(g_text(key), _wrong(key, wrong), 4)
    check(len(distractors) >= 3, "too few distractors")
    text = g_text(key)
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(a), pos=_entry_name(i, j)),
        working=[
            f"A has eigenvalues {l1} and {l2}, so A = P D P⁻¹ with D = diag({l1}, {l2}) and e^(At) = P e^(Dt) P⁻¹.",
            f"The {_entry_name(i, j)} entry of P diag(e^({l1}t), e^({l2}t)) P⁻¹ is {text}.",
        ],
        distractors=distractors,
        **_answer(key),
    )


# -- medium -------------------------------------------------------------------------------------------------------

_MEDIUM_STEMS = (
    "Find the {pos} entry of e^(At) for A = {m}. Write e^(kt) for the exponential e^{{kt}}.",
    "Find the {pos} entry of the matrix exponential e^(At), where A = {m}. Write e^(kt) for e^{{kt}}.",
    "A = {m}. Write the {pos} entry of e^(At) as an expression in t, using e^(kt) for e^{{kt}}.",
)


def _nilpotent_2x2(rng):
    """(lam, N): a non-zero integer 2×2 nilpotent matrix N = s [[-pq, p²], [-q², pq]] with all four entries non-zero."""
    for _ in attempts(500, "a dense nilpotent 2×2 matrix"):
        p, q = rng.randint(1, 3), rng.randint(1, 3)
        q = q * rng.choice([-1, 1])
        s = rng.choice([1, 1, 2, -1, -2])
        n = [[-s * p * q, s * p * p], [-s * q * q, s * p * q]]
        if all(v != 0 for row in n for v in row) and max(abs(v) for row in n for v in row) <= 9:
            return rng.choice([-3, -2, -1, 1, 2, 3, 4]), n
    raise AssertionError  # pragma: no cover


def _medium_matrix(rng):
    lam, n = _nilpotent_2x2(rng)
    check(n[0][0] * n[1][1] - n[0][1] * n[1][0] == 0 and n[0][0] + n[1][1] == 0, "N is not nilpotent")
    a = [[Fraction(lam * (i == j) + n[i][j]) for j in range(2)] for i in range(2)]
    e = [
        [{lam: {m: c for m, c in {0: Fraction(i == j), 1: Fraction(n[i][j])}.items() if c != 0}} for j in range(2)]
        for i in range(2)
    ]
    return lam, n, a, e


def _medium(rng):
    lam, n, a, e = _medium_matrix(rng)
    verify(a, e)
    i, j = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    key = e[i][j]
    delta = Fraction(1 if i == j else 0)
    nij = Fraction(n[i][j])
    wrong = [
        ({lam: {0: delta}}, "matrix-exp-nilpotent-term-dropped"),
        ({lam: {0: delta + nij}}, "matrix-exp-nilpotent-t-factor-dropped"),
        ({lam: {0: delta, 1: -nij}}, "sign-error-moving-term"),
        ({lam: {0: Fraction(1), 1: nij}}, "matrix-exp-identity-added-offdiagonal"),
        ({int(a[i][j]): {0: Fraction(1)}}, "matrix-exp-entrywise"),
        ({0: {0: delta, 1: nij}}, "matrix-exp-eigenvalue-factor-dropped"),
    ]
    distractors = pick_distractors(g_text(key), _wrong(key, wrong), 4)
    check(len(distractors) >= 3, "too few distractors")
    text = g_text(key)
    return dict(
        text=phrase(rng, _MEDIUM_STEMS, m=mat_str(a), pos=_entry_name(i, j)),
        working=[
            f"The only eigenvalue is {lam}, and N = A - {lam}I = {mat_str(n)} satisfies N² = 0.",
            f"So e^(At) = e^({lam}t)(I + tN), and its {_entry_name(i, j)} entry is {text}.",
        ],
        distractors=distractors,
        **_answer(key),
    )


# -- hard ---------------------------------------------------------------------------------------------------------

_HARD_ENTRY_STEMS = (
    "Find the {pos} entry of e^(At) for A = {m}. Write e^(kt) for the exponential e^{{kt}}.",
    "For A = {m}, find the {pos} entry of the matrix exponential e^(At), writing e^(kt) for e^{{kt}}.",
)
_HARD_IVP_STEMS = (
    "The vector x(t) solves x' = Ax with A = {m} and x(0) = {x0}. Find {name}(t), writing e^(kt) for e^{{kt}}.",
    "Solve x' = Ax, x(0) = {x0}, for A = {m}, and give the component {name}(t). Write e^(kt) for e^{{kt}}.",
)


def _distinct3(rng):
    return [(l, 1) for l in sorted(rng.sample(_LAMBDAS + [0], 3))]


def _block21(rng):
    lam, mu = rng.sample(_LAMBDAS, 2)
    return [(lam, 2), (mu, 1)]


def _block3(rng):
    return [(rng.choice(_LAMBDAS), 3)]


_STRUCTURES = (_distinct3, _block21, _block3)


def _draw_hard(rng):
    for _ in attempts(500, "a 3×3 matrix whose e^{At} has rich entries"):
        blocks = rng.choice(_STRUCTURES)(rng)
        a, p = conjugate(rng, jordan_matrix(blocks), cap=12)
        e = exponential(p, blocks)
        rich = [
            (i, j)
            for i in range(3)
            for j in range(3)
            if e[i][j] and (len(e[i][j]) >= 2 or max(len(poly) for poly in e[i][j].values()) >= 2)
        ]
        if rich:
            return a, p, blocks, e, rich
    raise AssertionError  # pragma: no cover


def _constant_terms(g):
    """Each polynomial replaced by its constant term: the nilpotent contribution dropped."""
    return g_clean({lam: {0: poly.get(0, Fraction(0))} for lam, poly in g.items()})


def _flip_nonconstant(g):
    """Every t-dependent polynomial term negated: the sign error moving the nilpotent part across."""
    return g_clean({lam: {m: (-c if m else c) for m, c in poly.items()} for lam, poly in g.items()})


def _no_factorials(g):
    """The t² coefficient doubled, as if t²/2! were written t²."""
    return g_clean({lam: {m: (c * 2 if m == 2 else c) for m, c in poly.items()} for lam, poly in g.items()})


def _hard_once(rng):
    a, p, blocks, e, rich = _draw_hard(rng)
    verify(a, e)
    if rng.random() < 0.5:
        i, j = rng.choice(rich)
        key = e[i][j]
        stem = phrase(rng, _HARD_ENTRY_STEMS, m=mat_str(a), pos=_entry_name(i, j))
        working_head = f"The {_entry_name(i, j)} entry of e^(At) = P e^(Jt) P⁻¹ is"
        entry = int(a[i][j])
    else:
        i = rng.randrange(3)
        x0 = [rng.choice([-2, -1, 0, 1, 2, 3]) for _ in range(3)]
        if all(v == 0 for v in x0):
            x0[0] = 1
        key = component(e, i, x0)
        if not key:
            x0[i] = 1
            key = component(e, i, x0)
        check(len(key) >= 2 or max(len(poly) for poly in key.values()) >= 2, "solution is a bare exponential")
        stem = phrase(
            rng, _HARD_IVP_STEMS, m=mat_str(a), x0="(" + ", ".join(str(v) for v in x0) + ")ᵀ", name=f"x{i + 1}"
        )
        working_head = f"x(t) = e^(At)x(0), whose component x{i + 1}(t) is"
        entry = None
    wrong = [
        (_flip_nonconstant(key), "sign-error-moving-term"),
        (_constant_terms(key), "matrix-exp-nilpotent-term-dropped"),
        (NoT(key), "matrix-exp-t-dropped"),
        ({-lam: poly for lam, poly in key.items()}, "matrix-exp-sign-in-exponent"),
    ]
    lams = sorted(key)
    if len(lams) >= 2:
        swapped = dict(key)
        swapped[lams[0]], swapped[lams[-1]] = key[lams[-1]], key[lams[0]]
        wrong.append((swapped, "matrix-exp-eigenvalues-swapped"))
    if any(size >= 3 for _l, size in blocks):
        wrong.append((_no_factorials(key), "matrix-exp-factorial-dropped"))
    if entry is not None:
        wrong.append(({entry: {0: Fraction(1)}}, "matrix-exp-entrywise"))
    distractors = pick_distractors(g_text(key), _wrong(key, wrong), 4)
    check(len(distractors) >= 3, "too few distractors")
    text = g_text(key)
    block_text = ", ".join(f"{s}×{s} block for {lam}" for lam, s in blocks)
    return dict(
        text=stem,
        working=[
            f"A is similar to a Jordan matrix with a {block_text}; e^(Jt) has e^(λt) on each block, times 1, t, t²/2 along the superdiagonals.",
            f"{working_head} {text}.",
        ],
        distractors=distractors,
        **_answer(key),
    )


def _hard(rng):
    """The first hard draw whose answer is not a bare exponential and has enough distinct distractors."""
    for _ in attempts(50, "a hard draw with rich answer and distractors"):
        try:
            return _hard_once(rng)
        except VerificationError as exc:
            if str(exc) not in ("solution is a bare exponential", "too few distractors"):
                raise
    raise AssertionError  # pragma: no cover


@generator(
    "s2b2_matrix_exponential",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
@tidied
def s2b2_matrix_exponential(rng, tier="easy"):
    """Entries of e^{At} as expressions in t; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_matrix_exponential", level=19, presets={"exam": dict(tier="exam")})
@plain_stem
@tidied
def exd_x_s2b2_matrix_exponential(rng, tier="exam"):
    """Exam chain on A = λI + N: λ, the t-coefficients of two entries of e^{At}, and of a component of x' = Ax.

    Every part is a whole number so the framework's follow-through check is cheap (a symbolic expression part is not);
    the full entries and the solution are verified here with exact closed forms, Taylor series and x' = Ax."""
    check(tier == "exam", f"unknown tier {tier!r}")
    lam, n, a, e = _medium_matrix(rng)
    verify(a, e)
    for _ in attempts(200, "an initial vector whose solution keeps its t-term"):
        x0 = [rng.choice([-2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2])]
        row = rng.randrange(2)
        if x0[0] * n[row][0] + x0[1] * n[row][1] != 0:
            break
    entry_b, entry_c = e[row][0], e[row][1]
    solution = component(e, row, x0)
    check(g_equal(solution, g_add(entry_b, entry_c, scales=x0)), "x(t) differs from e^{At} x(0)")
    xs = [component(e, r, x0) for r in range(2)]
    for r in range(2):
        check(g_equal(g_diff(xs[r]), g_add(*xs, scales=[a[r][0], a[r][1]])), "x' differs from Ax")
    trace = int(a[0][0] + a[1][1])
    delta_b, delta_c = (1 if row == 0 else 0), (1 if row == 1 else 0)
    nb, nc = int(n[row][0]), int(n[row][1])
    s_key = x0[0] * nb + x0[1] * nc
    # The t-coefficients read back from the exact closed forms.
    check(entry_b[lam][1] == nb and entry_c[lam][1] == nc and solution[lam][1] == s_key, "t-coefficients differ")
    const_part = x0[0] * delta_b + x0[1] * delta_c

    def numbers(key, cands):
        return pick_distractors(str(key), [number_distractor(v, t) for v, t in cands if v != key], 3)

    a_dist = numbers(
        lam,
        [
            (-lam, "sign-flip-both-roots"),
            (trace, "eigval-trace-as-eigenvalue"),
            (int(a[0][0]), "eigval-diagonal-entry"),
            (int(a[1][1]), "eigval-diagonal-entry"),
            (2 * lam + 1, "eigval-trace-difference-sign"),
        ],
    )
    b_dist = numbers(
        nb,
        [
            (0, "matrix-exp-nilpotent-term-dropped"),
            (-nb, "sign-error-moving-term"),
            (delta_b + nb, "matrix-exp-nilpotent-t-factor-dropped"),
            (int(a[row][0]), "matrix-exp-entrywise"),
        ],
    )
    c_dist = numbers(
        nc,
        [
            (0, "matrix-exp-nilpotent-term-dropped"),
            (-nc, "sign-error-moving-term"),
            (delta_c + nc, "matrix-exp-nilpotent-t-factor-dropped"),
            (int(a[row][1]), "matrix-exp-entrywise"),
        ],
    )
    d_dist = numbers(
        s_key,
        [
            (0, "matrix-exp-nilpotent-term-dropped"),
            (-s_key, "sign-error-moving-term"),
            ((x0[0] + x0[1]) * (nb + nc), "matrix-exp-columns-not-weighted"),
            (x0[0] * nb, "matrix-exp-columns-not-weighted"),
            (s_key + const_part, "matrix-exp-nilpotent-t-factor-dropped"),
        ],
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    i = row + 1
    rng_range = (-80, 80)
    return dict(
        text=phrase(
            rng,
            (
                "Let A = {m}.",
                "Consider the matrix A = {m}.",
                "The matrix A = {m} is given.",
            ),
            m=mat_str(a),
        ),
        parts=[
            dict(
                prompt="A has a single repeated eigenvalue λ. Find λ.",
                answer_kind="number",
                answer=lam,
                answer_text=str(lam),
                marks=1,
                working=[
                    f"The trace is {trace} and the eigenvalues are equal, so λ = {lam}; A - {lam}I = {mat_str(n)} is non-zero and nilpotent."
                ],
                distractors=a_dist,
                sensible_range=(-20, 20),
            ),
            dict(
                prompt=f"The ({i}, 1) entry of e^(At) has the form (p + qt)e^(λt), with λ the eigenvalue from part (a). Find q.",
                answer_kind="number",
                answer=nb,
                answer_text=str(nb),
                marks=2,
                working=[
                    f"e^(At) = e^({lam}t)(I + tN), so the ({i}, 1) entry is {g_text(entry_b)} and q is the ({i}, 1) entry of N, {nb}."
                ],
                distractors=b_dist,
                sensible_range=rng_range,
            ),
            dict(
                prompt=f"The ({i}, 2) entry of e^(At) has the form (p + qt)e^(λt). Find q.",
                answer_kind="number",
                answer=nc,
                answer_text=str(nc),
                marks=2,
                working=[f"Likewise the ({i}, 2) entry is {g_text(entry_c)}, so q = {nc}."],
                distractors=c_dist,
                sensible_range=rng_range,
            ),
            dict(
                prompt=(
                    f"The component x{i}(t) of the solution of x' = Ax with x(0) = ({x0[0]}, {x0[1]})ᵀ has the form "
                    "(r + st)e^(λt). Find s."
                ),
                answer_kind="number",
                answer=s_key,
                answer_text=str(s_key),
                marks=3,
                follow_through=f"({x0[0]})*part_b + ({x0[1]})*part_c",
                working=[
                    f"x(t) = e^(At)x(0), so x{i}(t) = {x0[0]}·[({i}, 1) entry] + {x0[1]}·[({i}, 2) entry] = {g_text(solution)}, and s = {s_key}."
                ],
                distractors=d_dist,
                sensible_range=rng_range,
            ),
        ],
    )
