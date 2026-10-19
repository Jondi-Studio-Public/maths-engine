"""Generates `s2c2_linear_system_complex_eigen` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_linear_system_complex_eigen`.

``uc2-linear-system-complex-eigen`` (``ordinary-differential-equations``, answer ``expression``): the real-valued
solution of x' = A x for a 2x2 matrix whose eigenvalues are alpha +- beta i.

* easy: alpha = 0, a pure oscillation, asked for x(t).
* medium: alpha is not zero, so the solution spirals; asked for y(t), which needs the other row of the matrix.
* hard: the second initial value is not given: the stem gives x(0) and x'(0), so y(0) must be recovered from the
  first equation before the solution can be written; asked for y(t).
* exam (``exd_x_s2c2_linear_system_complex_eigen``): (a) the real part alpha, (b) the imaginary part beta (follows
  through from (a)), (c) x(t) (follows through from (a) and (b)), (d) x at a stated time to 3 decimal places.

The matrix is A = [[alpha + m, q], [r, alpha - m]] with q r = -(beta^2 + m^2). Then N = A - alpha I satisfies
N^2 = -beta^2 I, so the solution is x(t) = e^(alpha t)(x0 cos(beta t) + (N x0 / beta) sin(beta t)). Each answer is
checked by integrating x' = A x numerically (Runge-Kutta), a route with no eigenvalues in it.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor, exact_value
from ._s2_options import spot_check
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text, rounds_unambiguously
from ._s2c2_common import Term, fn_differs, fn_machine, fn_text, fn_value, initial_text, merged, rk4_state, system_text
from ._ucv1_texsay import say_marked

_NAMES = "xy"


class _System:
    """x' = A x with complex eigenvalues alpha +- beta i, and the initial vector."""

    def __init__(self, alpha, beta, m, q, r, x0, y0):
        self.alpha, self.beta, self.m, self.q, self.r = alpha, beta, m, q, r
        self.a = [[alpha + m, q], [r, alpha - m]]
        self.x0 = [x0, y0]
        check(q * r == -(beta * beta + m * m), "q r is not -(beta^2 + m^2)")
        # N x0, with N = A - alpha I
        self.nx = [m * x0 + q * y0, r * x0 - m * y0]

    def sine_coefficient(self, k):
        return Fraction(self.nx[k], self.beta)

    def component(self, k, *, alpha=None, beta=None, swap=False, scale=1, sign=1):
        """e^(alpha t)(c cos(beta t) + s sin(beta t)) for component k, with optional named slips."""
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        c, s = Fraction(self.x0[k]), self.sine_coefficient(k) * scale * sign
        if swap:
            c, s = s, c
        return merged([Term(c, a=alpha, trig="cos", b=beta), Term(s, a=alpha, trig="sin", b=beta)])


def _verify(system, k, answer):
    check(abs(fn_value(answer, 0.0) - system.x0[k]) < 1e-9, "the answer misses the initial value")
    if not spot_check(system.a, system.x0, k):  # the integration is the slow second route: one question in four
        return
    for t_end in (0.5, 1.0):
        state = rk4_state(system.a, system.x0, t_end)
        want = fn_value(answer, t_end)
        check(abs(state[k] - want) <= 1e-6 * max(1.0, abs(want)), f"x' = Ax disagrees with the answer at t = {t_end}")


def _candidates(system, k):
    out = [
        ("ode2-cos-sin-coefficients-swapped", system.component(k, swap=True)),
        ("ode2-frequency-from-discriminant", system.component(k, beta=2 * system.beta)),
        ("ode2-sine-coefficient-not-divided-by-frequency", system.component(k, scale=system.beta)),
        ("ode2-sine-term-sign", system.component(k, sign=-1)),
        ("ode2-growth-factor-dropped", system.component(k, alpha=0)),
        ("ode2-exponent-sign-flipped", system.component(k, alpha=-system.alpha)),
        ("ode2-other-component-given", system.component(1 - k)),
    ]
    return out


def _fn_distractors(answer, candidates, n=4):
    scored = [Distractor(fn_text(f), tag, fn_machine(f)) for tag, f in candidates if f and fn_differs(answer, f)]
    return pick_distractors(fn_text(answer), scored, n)


def _divisor_pairs(k):
    return [(d, k // d) for d in range(1, k + 1) if k % d == 0]


def _draw(rng, tier):
    """A system of the tier: (system, asked component, answer terms)."""
    for _ in attempts(2000, "a complex-eigenvalue system with a clean answer"):
        beta = rng.choice([1, 2, 2, 3])
        m = rng.randint(-3, 3)
        if tier == "easy":
            alpha = 0
            if m == 0 and beta == 1:
                continue
        elif tier == "medium":
            alpha = rng.choice([-2, -1, 1, 2])
        else:
            alpha = rng.choice([-2, -1, 1, 2])
            if m == 0:
                continue
        d, e = rng.choice(_divisor_pairs(beta * beta + m * m))
        q, r = (d, -e) if rng.random() < 0.5 else (-d, e)
        x0, y0 = rng.randint(-4, 4), rng.randint(-4, 4)
        if x0 == 0 or y0 == 0:
            continue
        system = _System(alpha, beta, m, q, r, x0, y0)
        if max(abs(v) for row in system.a for v in row) > 9:
            continue
        if any(n % beta for n in system.nx):
            continue
        k = 0 if tier == "easy" else 1
        answer = system.component(k)
        if len(answer) != 2 or any(abs(t.c) > 12 for t in answer):
            continue
        if tier == "hard" and abs(q) < 2:
            continue
        distractors = _fn_distractors(answer, _candidates(system, k))
        if len(distractors) < 3:
            continue
        return system, k, answer, distractors
    raise AssertionError("unreachable")


_STEMS = (
    "Solve the initial value problem «{system}» with «{ic}». Find the real-valued solution «{comp}(t)».",
    "A system satisfies «{system}», with «{ic}». Find the real-valued solution «{comp}(t)».",
    "Given «{system}» and «{ic}», find «{comp}(t)» as a real-valued function.",
)
_HARD_STEMS = (
    "A system satisfies «{system}», with «x(0) = {x0}» and «x'(0) = {v0}». Find the real-valued solution «y(t)».",
    "Given «{system}», «x(0) = {x0}» and «x'(0) = {v0}», find «y(t)» as a real-valued function.",
)


@generator(
    "s2c2_linear_system_complex_eigen",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c2_linear_system_complex_eigen(rng, tier="easy"):
    system, k, answer, distractors = _draw(rng, tier)
    _verify(system, k, answer)
    sys_text = system_text(system.a, _NAMES)
    if tier == "hard":
        x0, y0 = system.x0
        v0 = system.a[0][0] * x0 + system.a[0][1] * y0
        check(v0 == (system.alpha + system.m) * x0 + system.q * y0, "x'(0) is not the first equation at t = 0")
        text, latex = say_marked(rng.choice(_HARD_STEMS).format(system=sys_text, x0=x0, v0=v0))
        steps = [f"From x' = {join_terms([(system.a[0][0], 'x'), (system.a[0][1], 'y')])} at t = 0, y(0) = {y0}."]
    else:
        text, latex = say_marked(
            rng.choice(_STEMS).format(system=sys_text, ic=initial_text(system.x0, _NAMES), comp=_NAMES[k])
        )
        steps = []
    sign = "+" if system.alpha >= 0 else "-"
    return dict(
        text=text,
        latex=latex,
        answer=fn_machine(answer),
        answer_text=fn_text(answer),
        answer_kind="expression",
        distractors=distractors,
        working=[
            *steps,
            f"The eigenvalues are {system.alpha} {sign} {system.beta}i "
            f"(real part {system.alpha}, imaginary part {system.beta}).",
            f"The real solution is {_NAMES[k]}(t) = {fn_text(answer)}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


def _exam_draw(rng):
    for _ in attempts(2000, "an exam complex-eigenvalue system"):
        beta = rng.choice([1, 2, 2, 3])
        m = rng.randint(-3, 3)
        alpha = rng.choice([-2, -1, 1, 2])
        d, e = rng.choice(_divisor_pairs(beta * beta + m * m))
        q, r = (d, -e) if rng.random() < 0.5 else (-d, e)
        x0, y0 = rng.randint(-4, 4), rng.randint(-4, 4)
        if x0 == 0:
            continue
        system = _System(alpha, beta, m, q, r, x0, y0)
        if max(abs(v) for row in system.a for v in row) > 9 or any(n % beta for n in system.nx):
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


@generator("exd_x_s2c2_linear_system_complex_eigen", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_linear_system_complex_eigen(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    system, answer, t0, value = _exam_draw(rng)
    alpha, beta = system.alpha, system.beta
    (a, b), (c, d) = system.a
    x0, y0 = system.x0
    trace, det = a + d, a * d - b * c
    _verify(system, 0, answer)
    check(
        2 * alpha == trace and alpha * alpha + beta * beta == det, "alpha and beta disagree with trace and determinant"
    )

    wrong_a = [
        ("ode2-real-part-not-halved", trace),
        ("ode2-exponent-sign-flipped", -alpha),
        ("ode2-real-and-imaginary-parts-swapped", beta),
        ("ode2-determinant-as-eigenvalue", det),
    ]
    a_dist = pick_distractors(str(alpha), [Distractor(str(v), tag, v) for tag, v in wrong_a if v != alpha], 4)

    wrong_b = [
        ("ode2-imaginary-part-not-rooted", beta * beta),
        ("ode2-imaginary-part-not-halved", 2 * beta),
        ("ode2-modulus-as-imaginary-part", math.isqrt(det) if math.isqrt(det) ** 2 == det else None),
        ("ode2-real-and-imaginary-parts-swapped", abs(alpha)),
    ]
    b_dist = pick_distractors(
        str(beta), [Distractor(str(v), tag, v) for tag, v in wrong_b if v is not None and v != beta], 4
    )

    c_dist = _fn_distractors(answer, _candidates(system, 0))

    t0f = float(t0)
    d_text = f"{value:.3f}"
    d_cands = []
    for tag, f in _candidates(system, 0):
        v = fn_value(f, t0f)
        if abs(v) < 1e6 and abs(round(v, 3) - round(value, 3)) > 0.006:
            d_cands.append(Distractor(f"{v:.3f}", tag, round(v, 3)))
    d_dist = pick_distractors(d_text, d_cands, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem, stem_latex = say_marked(
        rng.choice(
            [
                "A system satisfies «{s}», with «{i}».",
                "Consider the initial value problem «{s}», «{i}». Its matrix has complex eigenvalues.",
            ]
        ).format(s=system_text(system.a, _NAMES), i=initial_text(system.x0, _NAMES))
    )
    t0_text = rational_text(t0) if t0.denominator == 1 else f"{t0f:g}"
    t0_ft = str(t0.numerator) if t0.denominator == 1 else f"{t0.numerator}/{t0.denominator}"
    follow_c = f"exp(part_a*t)*(({x0})*cos(part_b*t) + (({a} - part_a)*({x0}) + ({b})*({y0}))/part_b*sin(part_b*t))"

    def prompt(template, **slots):
        text, tex = say_marked(template.format(**slots))
        return dict(prompt=text, prompt_latex=tex)

    sign = "+" if alpha >= 0 else "-"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("The eigenvalues of the coefficient matrix are «α ± βi» with «β > 0». Find «α»."),
                answer_kind="number",
                answer=alpha,
                answer_text=str(alpha),
                marks=1,
                working=[f"Half the trace is {alpha}, so α = {alpha}."],
                distractors=a_dist,
                sensible_range=(-12, 12),
            ),
            dict(
                **prompt("Find «β»."),
                answer_kind="number",
                answer=beta,
                answer_text=str(beta),
                marks=2,
                follow_through=f"sqrt({det} - part_a**2)",
                working=[
                    f"The determinant is α² + β², so β² = {det} - {alpha * alpha} = {beta * beta} and β = {beta}."
                ],
                distractors=b_dist,
                sensible_range=(0, 12),
            ),
            dict(
                **prompt("Find the real-valued solution «x(t)»."),
                answer_kind="expression",
                answer=fn_machine(answer),
                answer_text=fn_text(answer),
                marks=3,
                follow_through=follow_c,
                working=[
                    f"With eigenvalues {alpha} {sign} {beta}i, x(t) = e^({alpha}t)(A cos({beta}t) + B sin({beta}t)) "
                    f"with A = {x0}, giving x(t) = {fn_text(answer)}."
                ],
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
