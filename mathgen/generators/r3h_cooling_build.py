"""Generates `r3h_cooling` questions (easy, medium, hard) and the exam chain `exd_x_r3h_cooling`.

``newton-cooling-shifted-exponential`` (``differential-equations``, answers ``expression`` and ``number``): Newton's
law ``dT/dt = -k(T - A)`` for an object cooling or warming towards the surrounding temperature ``A``, solved as
``T = A + (T0 - A) e^(-kt)``. The temperature the object approaches is ``A``, not zero, so the exponential is shifted.

* easy: ``k``, ``A`` and ``T(0)`` are given; write ``T`` as a function of ``t`` (expression).
* medium: either find ``k`` from one later reading (number, 3 decimal places) or find ``T`` at a later time from ``k``
  (number, 1 decimal place).
* hard: from ``T(0)`` and one later reading, find the time to reach a stated temperature (number, 1 decimal place),
  for an object that cools or warms.
* exam (``exd_x_r3h_cooling``): (a) the initial excess ``T(0) - A``, (b) ``k`` from the later reading, (c) the
  temperature at a stated later time, (d) the time to reach a stated temperature (each following through).

Every solution is checked against a fourth-order Runge-Kutta integration of the equation itself.
"""

from __future__ import annotations

import math

import sympy

from ..core import attempts, check, generator, rounds_unambiguously
from ._exd_distinct_parts import answers_distinct
from ._r3h_kit import (
    T,
    PARAM,
    Redraw,
    build_with_redraws,
    expr_options,
    expr_part,
    machine,
    number_options,
    number_part,
    plain,
    say_marked,
)

_R = sympy.Rational

# (object, unit phrase for the stem, ambient temperatures, initial temperatures, time unit)
_CONTEXTS = (
    ("a cup of tea just poured", (18, 20, 22, 24), (82, 85, 88, 92), "minutes"),
    ("a metal bar taken from a kiln", (20, 22, 25), (220, 260, 300, 340), "minutes"),
    ("a pie taken from an oven", (20, 22, 24), (160, 172, 184, 196), "minutes"),
    ("a can of drink taken from a fridge into a hot room", (26, 28, 30), (4,), "minutes"),
    ("a frozen roast taken from a freezer into a kitchen", (20, 22, 24), (-18,), "minutes"),
)
_K = (_R(1, 20), _R(1, 10), _R(3, 20), _R(1, 5), _R(1, 4), _R(3, 10))
_RATIO = (_R(1, 2), _R(1, 3), _R(1, 4), _R(2, 3), _R(3, 4), _R(1, 5), _R(3, 5))


def _rk4(k: float, ambient: float, t0: float, t_end: float, steps: int = 400) -> float:
    """T(t_end) by integrating dT/dt = -k(T - A) directly."""
    h = t_end / steps
    v = t0

    def f(u):
        return -k * (u - ambient)

    for _ in range(steps):
        a = f(v)
        b = f(v + h * a / 2)
        c = f(v + h * b / 2)
        d = f(v + h * c)
        v += h * (a + 2 * b + 2 * c + d) / 6
    return v


def _band(c, values):
    """Drop wrong temperatures that no learner would believe: far outside the range from the start to the surroundings."""
    lo, hi = min(c["T0"], c["A"]) - 12, max(c["T0"], c["A"]) + 12
    return [(tag, v) for tag, v in values if v is not None and lo <= v <= hi]


def _fmt(value) -> str:
    value = _R(value)
    return str(value.p) if value.q == 1 else f"{float(value):g}"


def _minus(value) -> str:
    """``- 20`` for 20, ``+ 5`` for -5: subtracting a number without ever writing ``- -``."""
    return f"- {_fmt(value)}" if _R(value) >= 0 else f"+ {_fmt(-_R(value))}"


def _plus_term(a, d, exponent) -> str:
    """``a + d e^(exponent)`` written without ``+ -``."""
    return f"{_fmt(a)} {'+' if d >= 0 else '-'} {_fmt(abs(_R(d)))}e^({exponent})"


def _kk(k) -> str:
    return f"{float(k):g}"


def _draw_reading(rng):
    """An object, its ambient and starting temperatures, and a later reading T1 = A + D r (a one-decimal number)."""
    for _ in attempts(2000, "a cooling reading with a clean temperature"):
        obj, ambients, starts, unit = rng.choice(_CONTEXTS)
        A = rng.choice(ambients)
        T0 = rng.choice(starts)
        D = T0 - A
        r = rng.choice(_RATIO)
        t1 = rng.choice([3, 4, 5, 6, 8, 10])
        T1 = A + D * r
        if not (_R(T1) * 10).is_Integer:
            continue
        k = -math.log(float(r)) / t1
        return dict(obj=obj, unit=unit, A=A, T0=T0, D=D, r=r, t1=t1, T1=T1, k=k)
    raise AssertionError("unreachable")


def _solution(A, D, k):
    return A + D * sympy.exp(-k * T)


def _verb(c):
    return "cools" if c["D"] > 0 else "warms"


def _setup(c) -> str:
    return f"Let «T» be the temperature (in °C) of {c['obj']}, «t» {c['unit']} after the start. It {_verb(c)} towards the surrounding temperature of {c['A']} °C"


def _k_wrongs(c):
    A, T0, T1, t1, D = c["A"], c["T0"], float(c["T1"]), c["t1"], c["D"]
    true = math.log(D / (T1 - A)) / t1
    return [
        ("r3h-cooling-k-sign", -true),
        ("r3h-cooling-ambient-ignored", math.log(T0 / T1) / t1 if T1 and T0 / T1 > 0 else None),
        ("r3h-cooling-k-not-divided-by-time", math.log(D / (T1 - A))),
        ("r3h-cooling-log-base-ten", math.log10(D / (T1 - A)) / t1),
        ("r3h-cooling-ambient-ignored", math.log((T0 - A) / T1) / t1 if T1 and (T0 - A) / T1 > 0 else None),
    ]


@generator(
    "r3h_cooling",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_cooling(rng, tier="easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


def _easy(rng):
    obj, ambients, starts, unit = rng.choice(_CONTEXTS)
    A, T0, k = rng.choice(ambients), rng.choice(starts), rng.choice(_K)
    D = T0 - A
    c = dict(obj=obj, unit=unit, A=A, D=D)
    sol = _solution(A, D, k)
    stem, latex = say_marked(
        f"{_setup(c)}, so that «dT/dt = -{_kk(k)}(T {_minus(A)})». Initially «T = {T0}». Find «T» as a function of «t»."
    )
    wrongs = [
        ("r3h-cooling-ambient-ignored", D * sympy.exp(-k * T)),
        ("r3h-cooling-no-shift", T0 * sympy.exp(-k * T)),
        ("r3h-cooling-ambient-added-to-initial", A + T0 * sympy.exp(-k * T)),
        ("r3h-cooling-excess-sign", A - D * sympy.exp(-k * T)),
        ("r3h-cooling-exponent-sign", A + D * sympy.exp(k * T)),
    ]
    options = expr_options(sol, wrongs, 4, T)
    check(len(options) >= 2, "too few cooling distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=machine(sol),
        answer_text=plain(sol),
        answer_kind="expression",
        distractors=options,
        working=[f"T {_minus(A)} = {_fmt(D)}e^(-{_kk(k)}t), so T = {_plus_term(A, D, '-' + _kk(k) + 't')}."],
    )


def _medium(rng):
    for _ in attempts(300, "a medium cooling question that rounds cleanly"):
        c = _draw_reading(rng)
        A, T0, D, T1, t1 = c["A"], c["T0"], c["D"], c["T1"], c["t1"]
        if rng.random() < 0.5:
            k = c["k"]
            if not rounds_unambiguously(k, 3, 0.1):
                continue
            check(abs(_rk4(k, A, T0, t1) - float(T1)) < 1e-6 * max(1, abs(float(T1))), "k fails the RK4 check")
            stem, latex = say_marked(
                f"{_setup(c)}, so that «dT/dt = -k(T {_minus(A)})» for a constant «k». "
                f"Initially «T = {T0}» and after {t1} {c['unit']} «T = {_fmt(T1)}». Find «k», correct to 3 decimal places."
            )
            dist = number_options(k, 3, _k_wrongs(c))
            check(len(dist) >= 2, "too few distractors for k")
            return dict(
                text=stem,
                latex=latex,
                answer=round(k, 3),
                answer_text=f"{k:.3f}",
                answer_kind="number",
                distractors=dist,
                working=[
                    f"{_fmt(T1)} {_minus(A)} = {_fmt(D)}e^(-{t1}k), so k = ln({_fmt(D)}/{_fmt(T1 - A)})/{t1} = {k:.3f}."
                ],
            )
        kk = rng.choice(_K)
        t = rng.choice([3, 5, 8, 10, 15])
        val = A + D * math.exp(-float(kk) * t)
        if not rounds_unambiguously(val, 1, 0.1):
            continue
        check(abs(_rk4(float(kk), A, T0, t) - val) < 1e-6 * max(1, abs(val)), "value fails the RK4 check")
        stem, latex = say_marked(
            f"{_setup(c)}, so that «dT/dt = -{_kk(kk)}(T {_minus(A)})». Initially «T = {T0}». "
            f"Find «T» after {t} {c['unit']}, correct to 1 decimal place."
        )
        kf = float(kk)
        wrongs = [
            ("r3h-cooling-no-shift", T0 * math.exp(-kf * t)),
            ("r3h-cooling-ambient-ignored", D * math.exp(-kf * t)),
            ("r3h-cooling-excess-sign", A - D * math.exp(-kf * t)),
            ("r3h-cooling-exponent-sign", A + D * math.exp(kf * t)) if abs(kf * t) < 6 else None,
            ("r3h-cooling-ambient-added-to-initial", A + T0 * math.exp(-kf * t)),
        ]
        dist = number_options(val, 1, _band(c, wrongs))
        check(len(dist) >= 2, "too few distractors for T")
        return dict(
            text=stem,
            latex=latex,
            answer=round(val, 1),
            answer_text=f"{val:.1f}",
            answer_kind="number",
            distractors=dist,
            working=[f"T = {_plus_term(A, D, '-' + _kk(kk) + 't')}, so T({t}) = {val:.1f}."],
        )
    raise AssertionError("unreachable")


def _hard(rng):
    for _ in attempts(500, "a cooling timing question"):
        c = _draw_reading(rng)
        A, T0, D, T1, t1, k = c["A"], c["T0"], c["D"], c["T1"], c["t1"], c["k"]
        # a target between the start and ambient, further along than the reading
        frac = rng.choice([_R(1, 10), _R(1, 5), _R(1, 8), _R(1, 6), _R(1, 4), _R(1, 3), _R(1, 12)])
        Tf = A + D * frac
        if not _R(Tf).is_Integer or abs(Tf - A) < 1 or abs(float(Tf) - float(T1)) < 1:
            continue
        ans = math.log(float(D / (Tf - A))) / k
        if not (0.5 < ans < 90 and abs(ans - t1) > 0.5 and rounds_unambiguously(ans, 1, 0.1)):
            continue
        check(
            abs(_rk4(k, A, T0, ans, 4000) - float(Tf)) < 1e-4 * max(1, abs(float(Tf) - A)), "time fails the RK4 check"
        )
        break
    c["Tf"] = Tf
    stem, latex = say_marked(
        f"{_setup(c)}, following Newton's law of cooling. Initially «T = {T0}», and after {t1} {c['unit']} «T = {_fmt(T1)}». "
        f"Find the time, in {c['unit']} correct to 1 decimal place, at which the temperature reaches {_fmt(Tf)} °C."
    )
    Dd, Td = float(D), float(Tf - A)
    wrongs = [
        ("r3h-cooling-ambient-ignored", math.log(T0 / float(Tf)) / k if Tf and T0 / float(Tf) > 0 else None),
        ("r3h-cooling-time-times-k", math.log(Dd / Td) * k),
        ("r3h-cooling-ratio-inverted", -ans),
        ("r3h-cooling-k-not-divided-by-time", ans * t1),
        ("r3h-cooling-target-not-shifted", math.log(Dd / float(Tf)) / k if Tf and Dd / float(Tf) > 0 else None),
    ]
    dist = number_options(ans, 1, wrongs)
    check(len(dist) >= 2, "too few timing distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=round(ans, 1),
        answer_text=f"{ans:.1f}",
        answer_kind="number",
        distractors=dist,
        working=[
            f"k = ln({_fmt(D)}/{_fmt(T1 - A)})/{t1} = {k:.4f}; {_fmt(Tf - A)} = {_fmt(D)}e^(-kt) gives t = {ans:.1f}."
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    c = _draw_reading(rng)
    A, T0, D, T1, t1, k, unit = c["A"], c["T0"], c["D"], c["T1"], c["t1"], c["k"], c["unit"]
    t2 = rng.choice([t1 + 2, 2 * t1, 2 * t1 + 3, 3 * t1])
    val = A + D * math.exp(-k * t2)
    frac = rng.choice([_R(1, 10), _R(1, 5), _R(1, 8), _R(1, 6), _R(1, 4), _R(1, 3), _R(1, 12)])
    Tf = A + D * frac
    if not _R(Tf).is_Integer or abs(float(Tf) - float(T1)) < 1 or (float(Tf) - A) * (float(T1) - A) <= 0:
        raise Redraw
    ans = math.log(float(D / (Tf - A))) / k
    if not (
        rounds_unambiguously(k, 4, 0.1) and rounds_unambiguously(val, 1, 0.1) and rounds_unambiguously(ans, 1, 0.1)
    ):
        raise Redraw
    if abs(_rk4(k, A, T0, t2, 2000) - val) > 1e-5 * max(1, abs(val)):
        raise AssertionError("RK4 disagrees about T(t2)")
    stem, latex = say_marked(
        f"{_setup(c)}, following Newton's law of cooling «dT/dt = -k(T {_minus(A)})» for a constant «k». "
        f"Initially «T = {T0}», and after {t1} {unit} «T = {_fmt(T1)}»."
    )
    T1f = float(T1)
    parts = [
        expr_part(
            "Solve the differential equation to write «T» in terms of «t» and «k».",
            A + D * sympy.exp(-PARAM * T),
            2,
            [f"T {_minus(A)} = {_fmt(D)}e^(-kt), so T = {_plus_term(A, D, '-kt')}."],
            [
                ("r3h-cooling-no-shift", T0 * sympy.exp(-PARAM * T)),
                ("r3h-cooling-ambient-ignored", D * sympy.exp(-PARAM * T)),
                ("r3h-cooling-ambient-added-to-initial", A + T0 * sympy.exp(-PARAM * T)),
                ("r3h-cooling-excess-sign", A - D * sympy.exp(-PARAM * T)),
                ("r3h-cooling-exponent-sign", A + D * sympy.exp(PARAM * T)),
            ],
            var=T,
        ),
        number_part(
            "Hence find «k», correct to 4 decimal places.",
            k,
            4,
            2,
            [f"{_fmt(T1)} {_minus(A)} = {_fmt(D)}e^(-{t1}k), so k = ln({_fmt(D)}/{_fmt(T1 - A)})/{t1} = {k:.4f}."],
            [
                ("r3h-cooling-k-sign", -k),
                ("r3h-cooling-ambient-ignored", math.log(T0 / T1f) / t1 if T1f and T0 / T1f > 0 else None),
                ("r3h-cooling-k-not-divided-by-time", math.log(D / (T1f - A))),
                ("r3h-cooling-log-base-ten", math.log10(D / (T1f - A)) / t1),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the temperature of the object after {t2} {unit}, correct to 1 decimal place.",
            val,
            1,
            2,
            [f"T = {_plus_term(A, D, '-' + format(k, '.4f') + ' x ' + str(t2))} = {val:.1f}."],
            _band(
                c,
                [
                    ("r3h-cooling-no-shift", T0 * math.exp(-k * t2)),
                    ("r3h-cooling-ambient-ignored", D * math.exp(-k * t2)),
                    ("r3h-cooling-excess-sign", A - D * math.exp(-k * t2)),
                    ("r3h-cooling-ambient-added-to-initial", A + T0 * math.exp(-k * t2)),
                ],
            ),
            follow_through=f"{A} + ({D})*exp(-part_b*{t2})",
            t2=t2,
            unit=unit,
        ),
        number_part(
            "Find the time, in {unit} correct to 1 decimal place, at which the temperature reaches {Tf} °C.",
            ans,
            1,
            2,
            [f"{_fmt(Tf - A)} = {_fmt(D)}e^(-kt) gives t = ln({_fmt(D)}/{_fmt(Tf - A)})/{k:.4f} = {ans:.1f}."],
            [
                ("r3h-cooling-time-times-k", math.log(float(D / (Tf - A))) * k),
                (
                    "r3h-cooling-ambient-ignored",
                    math.log(T0 / float(Tf)) / k if float(Tf) and T0 / float(Tf) > 0 else None,
                ),
                (
                    "r3h-cooling-target-not-shifted",
                    math.log(float(D) / float(Tf)) / k if float(Tf) and float(D) / float(Tf) > 0 else None,
                ),
                ("r3h-cooling-ratio-inverted", -ans),
            ],
            follow_through=f"log(({D})/({_fmt(Tf - A)}))/part_b",
            sensible_range=(0, None),
            unit=unit,
            Tf=_fmt(Tf),
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


def _exam_distinct(rng):
    """`_exam_once`, drawn again when two parts share an answer (a learner could copy one into the other)."""
    built = _exam_once(rng)
    if not answers_distinct(built["parts"]):
        raise Redraw
    return built


@generator("exd_x_r3h_cooling", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r3h_cooling(rng, tier="exam"):
    """Exam tier: the initial excess, k, a later temperature and the time to reach a temperature."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_distinct)
