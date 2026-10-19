"""Generates `r3h_logistic` questions (easy, medium, hard) and the exam chain `exd_x_r3h_logistic`.

``logistic-equation-solution`` (``differential-equations``, answers ``expression`` and ``number``): the logistic
equation ``dP/dt = kP(1 - P/M)`` solved as ``P = M / (1 + A e^(-kt))`` with ``A = (M - P(0)) / P(0)``. The learner
is not asked to formulate the equation (the senior formulation skill does that); the work is in the solution.

* easy: ``k``, ``M`` and ``P(0)`` are given; write ``P`` as a function of ``t`` (expression).
* medium: either find ``k`` from one later reading (number, 3 decimal places) or find ``P`` at a later time from
  ``k`` (number, nearest whole).
* hard: from ``P(0)`` and one later reading, find the time at which the population reaches a stated fraction of ``M``,
  or the time at which it grows fastest (number, 1 decimal place). Neither time is read off a formula.
* exam (``exd_x_r3h_logistic``): (a) ``A`` (follows the form given), (b) ``k`` from the later reading, (c) the time to
  reach 90% of ``M``, (d) the greatest growth rate, ``kM/4`` (each following through from the earlier parts).

Every solution is checked against a fourth-order Runge-Kutta integration of the equation itself, and every
time or rate is checked by substituting back into the solution.
"""

from __future__ import annotations

import math

import sympy

from ..core import attempts, check, generator, rounds_unambiguously
from ._r3h_kit import (
    T,
    build_with_redraws,
    expr_options,
    fraction_part,
    machine,
    number_options,
    number_part,
    plain,
    Redraw,
    say_marked,
)

_R = sympy.Rational

# (what P measures, time unit, multipliers for the capacity)
_CONTEXTS = (
    ("the number of koalas on a predator-free island", "years", (3, 4, 5, 8)),
    ("the number of fish in a stocked dam", "months", (4, 5, 6, 10)),
    ("the number of students in a school who have heard a rumour", "days", (2, 3, 4, 6)),
    ("the area (in hectares) of a conservation reserve covered by a weed", "years", (1, 2, 3, 4)),
    ("the number (in thousands) of bacteria in a laboratory culture", "hours", (1, 2, 3, 5)),
)
_K = (_R(1, 10), _R(1, 5), _R(1, 4), _R(3, 10), _R(2, 5), _R(1, 2), _R(3, 5))
_A = (1, 3, 4, 9, 19)
_R_CHOICES = (_R(1, 2), _R(1, 3), _R(1, 4), _R(2, 3), _R(3, 4))


def _rk4(k: float, M: float, p0: float, t_end: float, steps: int = 400) -> float:
    """P(t_end) by integrating dP/dt = kP(1 - P/M) directly."""
    h = t_end / steps
    p = p0

    def f(v):
        return k * v * (1 - v / M)

    for _ in range(steps):
        a = f(p)
        b = f(p + h * a / 2)
        c = f(p + h * b / 2)
        d = f(p + h * c)
        p += h * (a + 2 * b + 2 * c + d) / 6
    return p


def _solution(M, A, k):
    return M / (1 + A * sympy.exp(-k * T))


def _draw_reading(rng):
    """M, A, P0, a time t1 and the reading P1 there, with e^(-k t1) = r rational so that P1 is a whole number."""
    for _ in attempts(2000, "a logistic reading that is a whole number"):
        subject, unit, mults = rng.choice(_CONTEXTS)
        m = rng.choice(mults)
        M = 100 * m * rng.choice([1, 2, 3, 4, 5, 6])
        A = rng.choice(_A)
        r = rng.choice(_R_CHOICES)
        t1 = rng.choice([2, 3, 4, 5, 6])
        if (M % (1 + A)) or A == 1:
            continue
        p0 = M // (1 + A)
        p1 = sympy.Rational(M) / (1 + A * r)
        if not p1.is_Integer or p1 <= p0:
            continue
        k = -math.log(float(r)) / t1
        if not 0.05 < k < 1.2:
            continue
        return dict(subject=subject, unit=unit, M=M, A=A, p0=p0, r=r, t1=t1, p1=int(p1), k=k)
    raise AssertionError("unreachable")


def _k_wrongs(c):
    """Slips in finding k from P(t1) = P1 (all numeric)."""
    M, A, p0, t1, p1 = c["M"], c["A"], c["p0"], c["t1"], c["p1"]
    q = M / p1 - 1
    true = math.log(A / q) / t1
    return [
        ("r3h-logistic-k-sign", -true),
        ("r3h-logistic-forgot-minus-one", math.log((M / p0) / q) / t1),
        ("r3h-logistic-k-not-divided-by-time", math.log(A / q)),
        ("r3h-logistic-log-base-ten", math.log10(A / q) / t1),
    ]


def _fmt(v):
    return float(v)


def _time_to_fraction(A, k, f):
    return math.log(A * f / (1 - f)) / k


def _context_sentence(c, extra=""):
    return f"The number of {c['thing']} P at time t {c['unit']}"


# ---------------------------------------------------------------------------------------------------------------
# practice tiers


def _easy(rng):
    for _ in attempts(500, "a logistic solution"):
        subject, unit, mults = rng.choice(_CONTEXTS)
        M = 100 * rng.choice(mults) * rng.choice([1, 2, 3, 4, 5, 6])
        A = rng.choice(_A[1:])
        if M % (1 + A):
            continue
        p0 = M // (1 + A)
        k = rng.choice(_K)
        break
    sol = _solution(M, A, k)
    ode = f"dP/dt = {_kk(k)}P(1 - P/{M})"
    stem, latex = say_marked(
        rng.choice(
            [
                f"Let «P» be {subject} at time «t» {unit}. It satisfies the logistic equation «{ode}», with «P(0) = {p0}». Find «P» as a function of «t».",
                f"{subject.capitalize()} at time «t» {unit}, written «P», is modelled by «{ode}» with «P(0) = {p0}». Find «P(t)».",
            ]
        )
    )
    # A mistakes: forgot -1, inverse, k dropped, sign of exponent
    wrongs = [
        ("r3h-logistic-forgot-minus-one", _solution(M, _R(M, p0), k)),
        ("r3h-logistic-constant-inverted", _solution(M, _R(p0, M - p0), k)),
        ("r3h-logistic-k-dropped", _solution(M, A, 1)),
        ("r3h-logistic-exponent-sign", M / (1 + A * sympy.exp(k * T))),
        ("r3h-logistic-exponential-model", p0 * sympy.exp(k * T)),
    ]
    options = expr_options(sol, wrongs, 4, T)
    check(len(options) >= 2, "too few logistic distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=machine(sol),
        answer_text=plain(sol),
        answer_kind="expression",
        distractors=options,
        working=[f"A = ({M} - {p0})/{p0} = {A}.", f"P = {M}/(1 + {A}e^(-{_kk(k)}t))."],
    )


def _kk(k) -> str:
    """k as a decimal when it terminates (0.25), written the way the stem shows it."""
    return f"{float(k):g}"


def _medium(rng):
    for _ in attempts(200, "a reading whose k rounds cleanly"):
        c = _draw_reading(rng)
        k_true = math.log(c["A"] / (c["M"] / c["p1"] - 1)) / c["t1"]
        if rounds_unambiguously(k_true, 3, 0.1):
            break
    if rng.random() < 0.5:
        # find k from one reading
        check(abs(_rk4(k_true, c["M"], c["p0"], c["t1"]) - c["p1"]) < 1e-6 * c["p1"], "k fails the RK4 check")
        stem, latex = say_marked(
            f"Let «P» be {c['subject']} at time «t» {c['unit']}. It satisfies «dP/dt = kP(1 - P/{c['M']})», where «k» is a constant. "
            f"Initially «P = {c['p0']}», and after {c['t1']} {c['unit']} «P = {c['p1']}». Find «k», correct to 3 decimal places."
        )
        wrongs = _k_wrongs(c)
        dist = number_options(k_true, 3, wrongs)
        check(len(dist) >= 2, "too few distractors for k")
        return dict(
            text=stem,
            latex=latex,
            answer=round(k_true, 3),
            answer_text=f"{k_true:.3f}",
            answer_kind="number",
            distractors=dist,
            working=[
                f"P = {c['M']}/(1 + {c['A']}e^(-kt)); at t = {c['t1']}: {c['p1']} = {c['M']}/(1 + {c['A']}e^(-{c['t1']}k)).",
                f"k = {k_true:.3f}.",
            ],
        )
    # find P at a later time from k
    for _ in attempts(500, "a logistic value that rounds cleanly"):
        k = rng.choice(_K)
        t = rng.choice([2, 3, 4, 5, 6, 8])
        val = c["M"] / (1 + c["A"] * math.exp(-float(k) * t))
        if rounds_unambiguously(val, 0, 0.1) and c["p0"] < val < 0.97 * c["M"]:
            break
    check(abs(_rk4(float(k), c["M"], c["p0"], t) - val) < 1e-6 * val, "value fails the RK4 check")
    stem, latex = say_marked(
        f"Let «P» be {c['subject']} at time «t» {c['unit']}. It satisfies «dP/dt = {_kk(k)}P(1 - P/{c['M']})», with «P(0) = {c['p0']}». "
        f"Find «P({t})», correct to the nearest whole number."
    )
    M, A, p0 = c["M"], c["A"], c["p0"]
    wrongs = [
        ("r3h-logistic-forgot-minus-one", M / (1 + (M / p0) * math.exp(-float(k) * t))),
        ("r3h-logistic-exponent-sign", M / (1 + A * math.exp(float(k) * t))),
        ("r3h-logistic-exponential-model", p0 * math.exp(float(k) * t)),
        ("r3h-logistic-k-dropped", M / (1 + A * math.exp(-t))),
    ]
    dist = number_options(val, 0, wrongs, gap=3)
    check(len(dist) >= 2, "too few distractors for P")
    return dict(
        text=stem,
        latex=latex,
        answer=round(val),
        answer_text=f"{round(val)}",
        answer_kind="number",
        distractors=dist,
        working=[f"A = {A}, so P = {M}/(1 + {A}e^(-{_kk(k)}t)) and P({t}) = {val:.1f}."],
    )


def _hard(rng):
    for _ in attempts(400, "a logistic timing question"):
        c = _draw_reading(rng)
        M, A, t1, p1 = c["M"], c["A"], c["t1"], c["p1"]
        k = math.log(A / (M / p1 - 1)) / t1
        if rng.random() < 0.5:
            f = rng.choice([_R(4, 5), _R(9, 10), _R(19, 20), _R(3, 4)])
            fv = float(f)
            ans = _time_to_fraction(A, k, fv)
            what = f"the time at which P first reaches {int(f * 100)}% of its capacity"
            wrongs = [
                ("r3h-logistic-fraction-not-converted", math.log(A * fv / (1 - fv) * 100) / k),
                ("r3h-logistic-forgot-constant", math.log(fv / (1 - fv)) / k),
                ("r3h-logistic-time-times-k", math.log(A * fv / (1 - fv)) * k),
                ("r3h-logistic-half-capacity", math.log(A) / k),
            ]
            check(abs(_rk4(k, M, c["p0"], ans, 2000) - fv * M) < 1e-4 * M, "time fails the RK4 check")
        else:
            ans = math.log(A) / k
            what = "the time at which P is increasing most rapidly"
            wrongs = [
                ("r3h-logistic-time-times-k", math.log(A) * k),
                ("r3h-logistic-fraction-not-converted", math.log(A * 2) / k),
                ("r3h-logistic-forgot-constant", math.log(2) / k),
                ("r3h-logistic-half-capacity", t1),
                ("r3h-logistic-max-rate-at-start", 0.0),
            ]
            check(abs(_rk4(k, M, c["p0"], ans, 2000) - M / 2) < 1e-4 * M, "inflection fails the RK4 check")
        if 0.5 < ans < 60 and abs(ans - t1) > 0.5 and rounds_unambiguously(ans, 1, 0.1):
            break
    stem, latex = say_marked(
        f"Let «P» be {c['subject']} at time «t» {c['unit']}. «P» follows a logistic model with capacity «{M}». "
        f"Initially «P = {c['p0']}», and after {t1} {c['unit']} «P = {p1}». "
        f"Find {what}, in {c['unit']} correct to 1 decimal place."
    )
    dist = number_options(ans, 1, wrongs)
    check(len(dist) >= 2, "too few timing distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=round(ans, 1),
        answer_text=f"{ans:.1f}",
        answer_kind="number",
        distractors=dist,
        working=[f"A = {A}, k = {k:.4f}, and the time is {ans:.1f}."],
    )


@generator(
    "r3h_logistic",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_logistic(rng, tier="easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    c = _draw_reading(rng)
    M, A, p0, t1, p1 = c["M"], c["A"], c["p0"], c["t1"], c["p1"]
    k = math.log(A / (M / p1 - 1)) / t1
    t90 = math.log(9 * A) / k
    rate = k * M / 4
    if not (
        rounds_unambiguously(k, 4, 0.1) and rounds_unambiguously(t90, 1, 0.1) and rounds_unambiguously(rate, 1, 0.1)
    ):
        raise Redraw
    check(abs(_rk4(k, M, p0, t90, 2000) - 0.9 * M) < 1e-4 * M, "t90 fails the RK4 check")
    unit = c["unit"]
    stem, latex = say_marked(
        f"Let «P» be {c['subject']} at time «t» {unit}. It satisfies «dP/dt = kP(1 - P/{M})», where «k» is a positive constant. "
        f"Initially «P = {p0}», and after {t1} {unit} «P = {p1}». The solution has the form «P = {M}/(1 + Ae^(-kt))»."
    )
    q = M / p1 - 1
    parts = [
        fraction_part(
            "Find the value of «A».",
            A,
            1,
            [f"A = ({M} - {p0})/{p0} = {A}."],
            [
                ("r3h-logistic-forgot-minus-one", _R(M, p0)),
                ("r3h-logistic-constant-inverted", _R(p0, M - p0)),
                ("r3h-logistic-capacity-added", _R(M + p0, p0)),
            ],
        ),
        number_part(
            "Hence find «k», correct to 4 decimal places.",
            k,
            4,
            2,
            [f"{p1} = {M}/(1 + {A}e^(-{t1}k)), so k = ln({A}/{_R(M, p1) - 1})/{t1} = {k:.4f}."],
            [
                ("r3h-logistic-k-sign", -k),
                ("r3h-logistic-forgot-minus-one", math.log((M / p0) / q) / t1),
                ("r3h-logistic-k-not-divided-by-time", math.log(A / q)),
                ("r3h-logistic-log-base-ten", math.log10(A / q) / t1),
            ],
            follow_through=f"-log(({M}/{p1} - 1)/part_a)/{t1}",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the time, in {unit} correct to 1 decimal place, at which «P» first reaches 90% of {M}.",
            t90,
            1,
            2,
            [f"0.9 = 1/(1 + {A}e^(-kt)) gives e^(-kt) = 1/{9 * A}, so t = ln({9 * A})/{k:.4f} = {t90:.1f}."],
            [
                ("r3h-logistic-forgot-constant", math.log(9) / k),
                ("r3h-logistic-time-times-k", math.log(9 * A) * k),
                ("r3h-logistic-fraction-not-converted", math.log(0.9 * A) / k),
                ("r3h-logistic-half-capacity", math.log(A) / k),
            ],
            follow_through="log(9*part_a)/part_b",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the greatest rate of increase of «P», in units of «P» per "
            + unit[:-1]
            + ", correct to 1 decimal place.",
            rate,
            1,
            1,
            [f"The rate is greatest when P = {M}/2, where dP/dt = k({M}/2)(1/2) = {rate:.1f}."],
            [
                ("r3h-logistic-max-rate-missing-quarter", k * M),
                ("r3h-logistic-max-rate-half", k * M / 2),
                ("r3h-logistic-max-rate-at-start", k * p0 * (1 - p0 / M)),
                ("r3h-logistic-max-rate-at-capacity", k * M / 8),
            ],
            follow_through=f"part_b*{M}/4",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_logistic", level=18, presets={"exam": dict(tier="exam")})
def exd_x_r3h_logistic(rng, tier="exam"):
    """Exam tier: the constant, k, the time to 90% of capacity and the greatest growth rate."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
