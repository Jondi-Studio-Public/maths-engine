"""Generates `s2c2_linear_system_repeated_eigen` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_linear_system_repeated_eigen`.

``uc2-linear-system-repeated-eigen`` (``ordinary-differential-equations``, answer ``expression``): the initial value
problem x' = A x when A has a repeated eigenvalue and only one independent eigenvector, so the solution needs a
generalised eigenvector and a term in t e^(lambda t).

* easy: a 2x2 defective matrix with a small eigenvector, asked for x(t).
* medium: a 2x2 defective matrix with a larger eigenvector and scale, asked for y(t).
* hard: a 3x3 matrix with a repeated eigenvalue (a Jordan block of size 2) and a third, different eigenvalue, asked
  for one component, so the t e^(lambda t) term and the separate exponential both appear.
* exam (``exd_x_s2c2_linear_system_repeated_eigen``): (a) the repeated eigenvalue, (b) the first entry of
  (A - lambda I) x(0) (follows through from (a)), (c) x(t) (follows through from (a) and (b)), (d) x at a stated
  time to 3 decimal places.

The 2x2 matrix is A = lambda I + N with N = k [[-v1 v2, v1^2], [-v2^2, v1 v2]], so N^2 = 0, N v = 0 and
x(t) = e^(lambda t)(x0 + t N x0). The 3x3 matrix is S J S^(-1) with S of determinant 1. Each answer is checked by
integrating x' = A x numerically (Runge-Kutta), a route with no eigenvectors in it.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
    rational_text,
    rounds_unambiguously,
)
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
from .s2c2_linear_system_real_eigen_build import _inverse_unimodular

_NAMES = "xyz"


class _Defective2:
    """x' = A x with A = lambda I + N, N nilpotent with N v = 0."""

    def __init__(self, lam, v, k, x0):
        self.lam, self.v, self.k, self.x0 = lam, v, k, list(x0)
        v1, v2 = v
        n = [[-k * v1 * v2, k * v1 * v1], [-k * v2 * v2, k * v1 * v2]]
        self.n = n
        self.a = [[lam + n[0][0], n[0][1]], [n[1][0], lam + n[1][1]]]
        self.nx = matvec(n, self.x0)
        check(matmul(n, n) == [[0, 0], [0, 0]], "N is not nilpotent")

    def component(self, i, *, lam=None, drop_t=False, sign=1, ax=False):
        """Component i of e^(lam t)(x0 + t N x0), with optional named slips."""
        lam = self.lam if lam is None else lam
        step = matvec(self.a, self.x0) if ax else self.nx
        terms = [Term(Fraction(self.x0[i]), a=lam)]
        if not drop_t:
            terms.append(Term(Fraction(sign * step[i]), a=lam, tp=1))
        return merged(terms)


class _Defective3:
    """x' = A x with A = S J S^(-1), J a size-2 Jordan block for lam plus a simple eigenvalue mu."""

    def __init__(self, s, lam, mu, c):
        self.s, self.lam, self.mu, self.c = s, lam, mu, c
        inv = _inverse_unimodular(s)
        j = [[lam, 1, 0], [0, lam, 0], [0, 0, mu]]
        self.a = matmul(matmul(s, j), inv)
        self.x0 = matvec(s, c)

    def component(self, i, *, lam=None, mu=None, drop_t=False, sign=1):
        lam = self.lam if lam is None else lam
        mu = self.mu if mu is None else mu
        s, (c1, c2, c3) = self.s, self.c
        terms = [Term(Fraction(s[i][0] * c1 + s[i][1] * c2), a=lam), Term(Fraction(s[i][2] * c3), a=mu)]
        if not drop_t:
            terms.append(Term(Fraction(sign * s[i][0] * c2), a=lam, tp=1))
        return merged(terms)


def _verify(a, x0, k, answer):
    for t_end in (0.5, 1.0):
        state = rk4_state(a, x0, t_end)
        want = fn_value(answer, t_end)
        check(abs(state[k] - want) <= 1e-6 * max(1.0, abs(want)), f"x' = Ax disagrees with the answer at t = {t_end}")
    check(abs(fn_value(answer, 0.0) - x0[k]) < 1e-9, "the answer misses the initial value")


def _fn_distractors(answer, candidates, n=4):
    scored = [Distractor(fn_text(f), tag, fn_machine(f)) for tag, f in candidates if f and fn_differs(answer, f)]
    return pick_distractors(fn_text(answer), scored, n)


def _candidates2(system, i):
    return [
        ("ode2-repeated-eigenvalue-no-t-term", system.component(i, drop_t=True)),
        ("ode2-matrix-used-instead-of-nilpotent-part", system.component(i, ax=True)),
        ("ode2-t-term-sign", system.component(i, sign=-1)),
        ("ode2-exponent-sign-flipped", system.component(i, lam=-system.lam)),
        ("ode2-other-component-given", system.component(1 - i)),
    ]


def _candidates3(system, i):
    return [
        ("ode2-repeated-eigenvalue-no-t-term", system.component(i, drop_t=True)),
        ("ode2-t-term-sign", system.component(i, sign=-1)),
        ("ode2-eigenvalue-eigenvector-mismatch", system.component(i, lam=system.mu, mu=system.lam)),
        ("ode2-exponent-sign-flipped", system.component(i, lam=-system.lam, mu=-system.mu)),
        ("ode2-other-component-given", system.component((i + 1) % 3)),
    ]


_PRIMITIVE = [(1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]
_LARGER = [(1, 3), (3, 1), (2, 3), (3, 2), (1, -3), (3, -1), (2, -3), (3, -2), (1, 2), (2, 1), (1, -2), (2, -1)]


def _draw2(rng, tier):
    for _ in attempts(2000, "a defective 2x2 system with a clean answer"):
        lam = rng.choice([-3, -2, -1, 1, 2, 3])
        if tier == "easy":
            v, k = rng.choice(_PRIMITIVE), rng.choice([-1, 1])
            i = 0
        else:
            v, k = rng.choice(_LARGER), rng.choice([-2, -1, 1, 2])
            i = 1
        x0 = [rng.randint(-4, 4), rng.randint(-4, 4)]
        system = _Defective2(lam, v, k, x0)
        if max(abs(e) for row in system.a for e in row) > 12 or 0 in x0:
            continue
        answer = system.component(i)
        if len(answer) != 2 or any(abs(t.c) > 24 for t in answer):
            continue
        distractors = _fn_distractors(answer, _candidates2(system, i))
        if len(distractors) < 3:
            continue
        return system, i, answer, distractors
    raise AssertionError("unreachable")


def _draw3(rng):
    for _ in attempts(3000, "a defective 3x3 system with a clean answer"):
        lo = [[1, 0, 0], [rng.randint(-1, 2), 1, 0], [rng.randint(-1, 2), rng.randint(-1, 2), 1]]
        up = [[1, rng.randint(-1, 2), rng.randint(-1, 2)], [0, 1, rng.randint(-1, 2)], [0, 0, 1]]
        s = matmul(lo, up)
        lam, mu = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        c = [rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])]
        system = _Defective3(s, lam, mu, c)
        if max(abs(e) for row in system.a for e in row) > 12 or sum(1 for row in system.a for e in row if e == 0) > 3:
            continue
        i = rng.randrange(3)
        answer = system.component(i)
        if len(answer) != 3 or any(abs(t.c) > 24 for t in answer):
            continue
        distractors = _fn_distractors(answer, _candidates3(system, i))
        if len(distractors) < 3:
            continue
        return system, i, answer, distractors
    raise AssertionError("unreachable")


_STEMS = (
    "Solve the initial value problem «{system}» with «{ic}». Find «{comp}(t)».",
    "A system satisfies «{system}», with «{ic}». Find «{comp}(t)».",
    "Given «{system}» and «{ic}», find «{comp}(t)».",
)


@generator(
    "s2c2_linear_system_repeated_eigen",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c2_linear_system_repeated_eigen(rng, tier="easy"):
    if tier == "hard":
        system, i, answer, distractors = _draw3(rng)
        names = "xyz"
        working = [
            f"The eigenvalue {system.lam} is repeated with a single eigenvector, so the solution has a "
            f"t e^({system.lam}t) term; the other eigenvalue is {system.mu}."
        ]
    else:
        system, i, answer, distractors = _draw2(rng, tier)
        names = "xy"
        working = [
            f"The eigenvalue {system.lam} is repeated with a single eigenvector, so the solution is "
            f"e^(λt)(x0 + t(A - λI)x0) with λ = {system.lam}."
        ]
    _verify(system.a, system.x0, i, answer)
    text, latex = say_marked(
        rng.choice(_STEMS).format(system=system_text(system.a, names), ic=initial_text(system.x0, names), comp=names[i])
    )
    return dict(
        text=text,
        latex=latex,
        answer=fn_machine(answer),
        answer_text=fn_text(answer),
        answer_kind="expression",
        distractors=distractors,
        working=[*working, f"{names[i]}(t) = {fn_text(answer)}."],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


def _exam_draw(rng):
    for _ in attempts(2000, "an exam defective system"):
        lam = rng.choice([-3, -2, -1, 1, 2, 3])
        v, k = rng.choice(_PRIMITIVE + _LARGER), rng.choice([-2, -1, 1, 2])
        x0 = [rng.randint(-4, 4), rng.randint(-4, 4)]
        system = _Defective2(lam, v, k, x0)
        if max(abs(e) for row in system.a for e in row) > 12 or 0 in x0 or v[0] == 0:
            continue
        answer = system.component(0)
        if len(answer) != 2:
            continue
        t0 = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2)])
        value = fn_value(answer, float(t0))
        if not (0.05 <= abs(value) <= 500 and rounds_unambiguously(value, 3, 0.1)):
            continue
        return system, answer, t0, value
    raise AssertionError("unreachable")


@generator("exd_x_s2c2_linear_system_repeated_eigen", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_linear_system_repeated_eigen(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    system, answer, t0, value = _exam_draw(rng)
    lam = system.lam
    (a, b), (c, d) = system.a
    x0, y0 = system.x0
    trace, det = a + d, a * d - b * c
    _verify(system.a, system.x0, 0, answer)
    check(2 * lam == trace and lam * lam == det, "lambda is not a double root of the characteristic equation")
    nx1 = system.nx[0]
    check(nx1 == (a - lam) * x0 + b * y0, "N x0 is inconsistent")

    wrong_a = [
        ("ode2-characteristic-sign-error", -lam),
        ("ode2-trace-as-eigenvalue", trace),
        ("ode2-determinant-as-eigenvalue", det),
        ("ode2-diagonal-entries-as-eigenvalues", a),
        ("ode2-diagonal-entries-as-eigenvalues", d),
    ]
    a_dist = pick_distractors(str(lam), [Distractor(str(v), tag, v) for tag, v in wrong_a if v != lam], 4)

    wrong_b = [
        ("ode2-nilpotent-part-sign", (a + lam) * x0 + b * y0),
        ("ode2-matrix-used-instead-of-nilpotent-part", a * x0 + b * y0),
        ("ode2-off-diagonal-term-dropped", (a - lam) * x0),
        ("ode2-other-component-given", system.nx[1]),
    ]
    b_dist = pick_distractors(str(nx1), [Distractor(str(v), tag, v) for tag, v in wrong_b if v != nx1], 4)

    c_dist = _fn_distractors(answer, _candidates2(system, 0))

    t0f = float(t0)
    d_text = f"{value:.3f}"
    d_cands = []
    for tag, f in _candidates2(system, 0):
        v = fn_value(f, t0f)
        if abs(v) < 1e6 and abs(round(v, 3) - round(value, 3)) > 0.006:
            d_cands.append(Distractor(f"{v:.3f}", tag, round(v, 3)))
    d_dist = pick_distractors(d_text, d_cands, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem, stem_latex = say_marked(
        rng.choice(
            [
                "A system satisfies «{s}», with «{i}». Its coefficient matrix has a repeated eigenvalue.",
                "Consider the initial value problem «{s}», «{i}». The coefficient matrix has a repeated eigenvalue.",
            ]
        ).format(s=system_text(system.a, "xy"), i=initial_text(system.x0, "xy"))
    )
    t0_text = rational_text(t0) if t0.denominator == 1 else f"{t0f:g}"
    t0_ft = str(t0.numerator) if t0.denominator == 1 else f"{t0.numerator}/{t0.denominator}"
    follow_c = f"exp(part_a*t)*(({x0}) + part_b*t)"

    def prompt(template, **slots):
        text, tex = say_marked(template.format(**slots))
        return dict(prompt=text, prompt_latex=tex)

    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Find the repeated eigenvalue «λ» of the coefficient matrix."),
                answer_kind="number",
                answer=lam,
                answer_text=str(lam),
                marks=1,
                working=[f"The trace is {trace}, so λ = {lam}; the determinant {det} is λ²."],
                distractors=a_dist,
                sensible_range=(-12, 12),
            ),
            dict(
                **prompt(
                    "Let «N = A - λI», where «A» is the coefficient matrix. "
                    "Find the first entry of «N» multiplied by the initial vector «(x(0), y(0))»."
                ),
                answer_kind="number",
                answer=nx1,
                answer_text=str(nx1),
                marks=2,
                follow_through=f"({a} - part_a)*({x0}) + ({b})*({y0})",
                working=[f"({a} - {paren_negative(lam)})({x0}) + ({b})({y0}) = {nx1}."],
                distractors=b_dist,
                sensible_range=(-200, 200),
            ),
            dict(
                **prompt("Find «x(t)»."),
                answer_kind="expression",
                answer=fn_machine(answer),
                answer_text=fn_text(answer),
                marks=3,
                follow_through=follow_c,
                working=[f"x(t) = e^({lam}t)({join_terms([(x0, ''), (nx1, 't')])}) = {fn_text(answer)}."],
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
