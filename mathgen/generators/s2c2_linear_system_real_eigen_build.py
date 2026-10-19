"""Generates `s2c2_linear_system_real_eigen` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_linear_system_real_eigen`.

``uc2-linear-system-real-eigen`` (``ordinary-differential-equations``, answer ``expression``): the initial value
problem x' = A x when A has real, distinct eigenvalues. The learner reads the system, finds the eigenvalues and
eigenvectors, solves the initial values for the constants and writes one component of the solution.

* easy: a 2x2 system with a full matrix, asked for x(t).
* medium: a 3x3 system whose matrix is upper triangular, asked for x(t), the component that depends on the other two.
* hard: a 3x3 system with a full matrix, so the eigenvalues come from a cubic and every eigenvector needs solving.
* exam (``exd_x_s2c2_linear_system_real_eigen``): a 2x2 system in four parts: (a) the larger eigenvalue,
  (b) the first entry of its eigenvector scaled to end in 1, (c) x(t), (d) x at a stated time to 3 decimal places.

The matrix is built as P D P^(-1) with P an integer matrix of determinant 1, so the eigenvectors are the columns
of P. Each answer is then checked by integrating x' = A x numerically (Runge-Kutta), a route with no eigenvalues
in it, and the exam chain also by Cayley-Hamilton: the coefficient of each exponential is read off the projections
(A - lambda_j I) / (lambda_i - lambda_j) applied to the initial vector.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, fmt_signed, generator, pick_distractors, rational_text, rounds_unambiguously
from ._s2c2_common import (
    Term,
    fn_differs,
    fn_machine,
    fn_text,
    fn_value,
    initial_text,
    matmul,
    matvec,
    merged,
    rk4_state,
    system_text,
)
from ._ucv1_texsay import say_marked

_NAMES = "xyz"


def _inverse_unimodular(p):
    """The integer inverse of an integer matrix of determinant 1 (2x2 or 3x3), by the adjugate."""
    n = len(p)
    if n == 2:
        return [[p[1][1], -p[0][1]], [-p[1][0], p[0][0]]]
    inv = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rows = [r for r in range(3) if r != j]
            cols = [c for c in range(3) if c != i]
            minor = p[rows[0]][cols[0]] * p[rows[1]][cols[1]] - p[rows[0]][cols[1]] * p[rows[1]][cols[0]]
            inv[i][j] = (-1) ** (i + j) * minor
    return inv


def _det(p):
    if len(p) == 2:
        return p[0][0] * p[1][1] - p[0][1] * p[1][0]
    return (
        p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
        - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
        + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])
    )


def _build_matrix(p, lams):
    inv = _inverse_unimodular(p)
    check(_det(p) == 1, "P is not unimodular")
    d = [[lams[i] if i == j else 0 for j in range(len(lams))] for i in range(len(lams))]
    return matmul(matmul(p, d), inv)


def _draw_p2(rng):
    for _ in attempts(500, "a 2x2 integer matrix of determinant 1"):
        p = [[rng.randint(-2, 3), rng.randint(-2, 3)], [rng.randint(-2, 3), rng.randint(-2, 3)]]
        if _det(p) == 1 and all(p[i][j] != 0 for i in range(2) for j in range(2)):
            return p
    raise AssertionError("unreachable")


def _draw_p3_upper(rng):
    for _ in attempts(500, "a unit upper-triangular P"):
        p = [[1, rng.randint(-2, 2), rng.randint(-2, 2)], [0, 1, rng.randint(-2, 2)], [0, 0, 1]]
        if p[0][1] and p[0][2] and p[1][2]:
            return p
    raise AssertionError("unreachable")


def _draw_p3_full(rng):
    for _ in attempts(500, "a full integer matrix of determinant 1"):
        lo = [[1, 0, 0], [rng.randint(-1, 2), 1, 0], [rng.randint(-1, 2), rng.randint(-1, 2), 1]]
        up = [[1, rng.randint(-1, 2), rng.randint(-1, 2)], [0, 1, rng.randint(-1, 2)], [0, 0, 1]]
        p = matmul(lo, up)
        if lo[1][0] and lo[2][0] and lo[2][1] and sum(1 for row in p for v in row if v == 0) <= 2:
            return p
    raise AssertionError("unreachable")


def _lams(rng, n, lo=-3, hi=3, allow_zero=False):
    pool = [v for v in range(lo, hi + 1) if allow_zero or v != 0]
    return rng.sample(pool, n)


class _Setup:
    """A solved system: matrix a, eigenvalues, eigenvector matrix p, constants c, initial vector x0."""

    def __init__(self, p, lams, c):
        self.p, self.lams, self.c = p, lams, c
        self.n = len(lams)
        self.a = _build_matrix(p, lams)
        self.x0 = matvec(p, c)

    def component(self, k, exps=None, coef=None, entry=True):
        """Component k of the solution as terms: sum_i coef_i * p[k][i] * e^(exps_i t)."""
        exps = exps or self.lams
        coef = coef or self.c
        terms = merged(Term(Fraction(coef[i] * (self.p[k][i] if entry else 1)), a=exps[i]) for i in range(self.n))
        return sorted(terms, key=lambda t: -t.a)

    def stem(self):
        return system_text(self.a, _NAMES), initial_text(self.x0, _NAMES)


def _verify(setup, k, answer):
    """Integrate x' = A x from x(0) = x0 and compare component k with the answer, at t = 0.5 and 1."""
    for t_end in (0.5, 1.0):
        state = rk4_state(setup.a, setup.x0, t_end)
        want = fn_value(answer, t_end)
        check(abs(state[k] - want) <= 1e-6 * max(1.0, abs(want)), f"x' = Ax disagrees with the answer at t = {t_end}")
    check(abs(fn_value(answer, 0.0) - setup.x0[k]) < 1e-9, "the answer misses the initial value")


def _candidates(setup, k):
    n = setup.n
    cyc = [setup.lams[(i + 1) % n] for i in range(n)]
    other = (k + 1) % n
    return [
        ("ode2-exponent-sign-flipped", setup.component(k, exps=[-v for v in setup.lams])),
        ("ode2-eigenvalue-eigenvector-mismatch", setup.component(k, exps=cyc)),
        ("ode2-initial-values-as-coefficients", setup.component(k, coef=setup.x0)),
        ("ode2-eigenvector-entry-dropped", setup.component(k, entry=False)),
        ("ode2-other-component-given", setup.component(other)),
    ]


def _fn_distractors(answer, candidates, n=4):
    scored = [Distractor(fn_text(f), tag, fn_machine(f)) for tag, f in candidates if f and fn_differs(answer, f)]
    return pick_distractors(fn_text(answer), scored, n)


_STEMS = (
    "Solve the initial value problem «{system}» with «{ic}». Find «{comp}(t)».",
    "A system satisfies «{system}», with «{ic}». Find «{comp}(t)».",
    "Given «{system}» and «{ic}», find «{comp}(t)».",
)


def _single(rng, tier):
    for _ in attempts(500, "a real-eigenvalue system with a clean answer"):
        if tier == "easy":
            p = _draw_p2(rng)
            lams = _lams(rng, 2, -4, 4)
            k = 0
        elif tier == "medium":
            p = _draw_p3_upper(rng)
            lams = _lams(rng, 3, -3, 3, allow_zero=True)
            k = 0
        else:
            p = _draw_p3_full(rng)
            lams = _lams(rng, 3, -3, 3, allow_zero=True)
            k = rng.choice([0, 1, 2])
        n = len(lams)
        c = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(n)]
        setup = _Setup(p, lams, c)
        if max(abs(v) for row in setup.a for v in row) > (9 if n == 2 else 11):
            continue
        if sum(1 for row in setup.a for v in row if v == 0) > (1 if n == 2 else 3):
            continue
        if tier == "hard" and any(sum(1 for v in row if v) < 3 for row in setup.a):
            continue
        if len({tuple(row) for row in setup.a}) < n:
            continue
        answer = setup.component(k)
        if len(answer) != n or any(abs(t.c) > 14 for t in answer):
            continue
        cands = _candidates(setup, k)
        distractors = _fn_distractors(answer, cands)
        if len(distractors) < 3:
            continue
        return setup, k, answer, distractors
    raise AssertionError("unreachable")


@generator(
    "s2c2_linear_system_real_eigen",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c2_linear_system_real_eigen(rng, tier="easy"):
    setup, k, answer, distractors = _single(rng, tier)
    _verify(setup, k, answer)
    system, ic = setup.stem()
    text, latex = say_marked(rng.choice(_STEMS).format(system=system, ic=ic, comp=_NAMES[k]))
    lam_text = ", ".join(str(v) for v in sorted(setup.lams))
    return dict(
        text=text,
        latex=latex,
        answer=fn_machine(answer),
        answer_text=fn_text(answer),
        answer_kind="expression",
        distractors=distractors,
        working=[
            f"The eigenvalues of the coefficient matrix are {lam_text}; each eigenvector gives one exponential.",
            f"Fitting the initial values to the constants gives {_NAMES[k]}(t) = {fn_text(answer)}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _n(v) -> str:
    return f"({v})"


def _exam_draw(rng):
    for _ in attempts(1000, "an exam real-eigenvalue system"):
        p = _draw_p2(rng)
        lams = _lams(rng, 2, -4, 4)
        c = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(2)]
        setup = _Setup(p, lams, c)
        (a, b), (cc, d) = setup.a
        if cc == 0 or max(abs(v) for row in setup.a for v in row) > 9:
            continue
        answer = setup.component(0)
        if len(answer) != 2:
            continue
        t0 = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2)])
        value = fn_value(answer, float(t0))
        if not (0.05 <= abs(value) <= 600 and rounds_unambiguously(value, 3, 0.1)):
            continue
        return setup, answer, t0, value
    raise AssertionError("unreachable")


class _Redraw(Exception):
    pass


@generator("exd_x_s2c2_linear_system_real_eigen", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_linear_system_real_eigen(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    setup, answer, t0, value = _exam_draw(rng)
    (a, b), (c, d) = setup.a
    lam1, lam2 = max(setup.lams), min(setup.lams)
    x0, y0 = setup.x0
    trace = a + d
    _verify(setup, 0, answer)
    # Cayley-Hamilton: the coefficient of e^(lam_i t) in x is the first entry of (A - lam_j I) x0 / (lam_i - lam_j).
    for lam_i, lam_j in ((lam1, lam2), (lam2, lam1)):
        coef = Fraction((a - lam_j) * x0 + b * y0, lam_i - lam_j)
        have = next((t.c for t in answer if t.a == lam_i), Fraction(0))
        check(coef == have, "projection coefficient disagrees with the answer")

    # (a) the larger eigenvalue
    det = a * d - b * c
    wrong_a = [
        ("ode2-characteristic-sign-error", -lam1),
        ("ode2-wrong-eigenvalue-chosen", lam2),
        ("ode2-diagonal-entries-as-eigenvalues", max(a, d)),
        ("ode2-trace-as-eigenvalue", trace),
        ("ode2-determinant-as-eigenvalue", det),
    ]
    a_dist = pick_distractors(str(lam1), [Distractor(str(v), tag, v) for tag, v in wrong_a if v != lam1], 4)

    # (b) the first entry of the eigenvector whose last entry is 1
    v1 = Fraction(lam1 - d, c)
    check(Fraction(a - lam1) * v1 + b == 0, "the vector is not an eigenvector")
    wrong_b = [
        ("ode2-eigenvector-wrong-diagonal-entry", Fraction(lam1 - a, c)),
        ("ode2-eigenvector-of-other-eigenvalue", Fraction(lam2 - d, c)),
        ("ode2-eigenvector-sign-slip", Fraction(d - lam1, c)),
        ("ode2-eigenvector-not-normalised", Fraction(lam1 - d)),
    ]
    b_dist = pick_distractors(
        rational_text(v1), [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in wrong_b if v != v1], 4
    )

    # (c) x(t)
    c_dist = _fn_distractors(answer, _candidates(setup, 0))

    # (d) x(t0) to 3 decimal places
    t0f = float(t0)
    d_text = f"{value:.3f}"
    d_cands = []
    for tag, f in _candidates(setup, 0):
        v = fn_value(f, t0f)
        if abs(v) < 1e6 and abs(round(v, 3) - round(value, 3)) > 0.006:
            d_cands.append(Distractor(f"{v:.3f}", tag, round(v, 3)))
    d_dist = pick_distractors(d_text, d_cands, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    system, ic = setup.stem()
    stem, stem_latex = say_marked(
        rng.choice(
            [
                "A system satisfies «{s}», with «{i}».",
                "Consider the initial value problem «{s}», «{i}».",
            ]
        ).format(s=system, i=ic)
    )
    t0_text = rational_text(t0) if t0.denominator == 1 else f"{t0f:g}"
    t0_ft = str(t0.numerator) if t0.denominator == 1 else f"{t0.numerator}/{t0.denominator}"
    lam2_ft = f"({trace} - part_a)"
    follow_c = (
        f"(({a} - {lam2_ft})*{_n(x0)} + {b}*{_n(y0)})/(2*part_a - {trace})*exp(part_a*t)"
        f" + (({a} - part_a)*{_n(x0)} + {b}*{_n(y0)})/({trace} - 2*part_a)*exp({lam2_ft}*t)"
    )

    def prompt(template, **slots):
        text, tex = say_marked(template.format(**slots))
        return dict(prompt=text, prompt_latex=tex)

    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Find the larger eigenvalue of the coefficient matrix of the system."),
                answer_kind="number",
                answer=lam1,
                answer_text=str(lam1),
                marks=1,
                working=[f"The eigenvalues are {lam2} and {lam1}; the larger is {lam1}."],
                distractors=a_dist,
                sensible_range=(-12, 12),
            ),
            dict(
                **prompt(
                    "An eigenvector for the eigenvalue in part (a) can be scaled so that its second entry is 1. "
                    "Find its first entry."
                ),
                answer_kind="fraction",
                answer=exact_value(v1),
                answer_text=rational_text(v1),
                marks=2,
                follow_through=f"(part_a - {_n(d)})/{_n(c)}",
                working=[
                    f"Row 2 of (A - λI)v = 0 with λ = {lam1} gives {fmt_signed(c, 'v1')} = {lam1 - d}, so v1 = {rational_text(v1)}."
                ],
                distractors=b_dist,
            ),
            dict(
                **prompt("Find «x(t)»."),
                answer_kind="expression",
                answer=fn_machine(answer),
                answer_text=fn_text(answer),
                marks=3,
                follow_through=follow_c,
                working=[f"Fitting the initial values to the two exponentials gives x(t) = {fn_text(answer)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt("Find «x({t0})», correct to 3 decimal places.", t0=t0_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=d_text,
                marks=1,
                follow_through=f"Subs(part_c, (t, e), ({t0_ft}, E))",
                working=[f"x({t0_text}) = {d_text}."],
                distractors=d_dist,
                sensible_range=(-1000, 1000),
            ),
        ],
    )
