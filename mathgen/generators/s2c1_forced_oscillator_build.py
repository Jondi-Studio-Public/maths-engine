"""Generates `s2c1_forced_oscillator_resonance` questions (easy, medium, hard) and the exam chain
`exd_x_s2c1_forced_oscillator_resonance`.

``uc2-forced-oscillator-resonance`` (``ordinary-differential-equations``, answer ``expression``): a mass on a spring
driven by a force ``F0 cos(wt)``, modelled by ``m x'' + c x' + k x = F0 cos(wt)``. The stems give masses, stiffnesses
and forces in SI units; the learner builds the equation.

* easy: no damping and a driving frequency away from the natural frequency; find the forced response
  ``x_p(t) = A cos(wt)``.
* medium: from rest, either driven at the natural frequency without being told so (``x = F0 t sin(w0 t)/(2 m w0)``),
  or driven close to it (``x = A (cos wt - cos w0 t)``).
* hard: one of four numerical questions: the driving frequency that makes the steady amplitude greatest in a damped
  system, the beat period, the greatest displacement during the beats, or the time for the resonant amplitude to
  reach a stated size.
* exam (``exd_x_s2c1_forced_oscillator_resonance``): resonance, (a) the natural frequency, (b) the growth rate of the
  amplitude (follows through from (a)), (c) x(t) (follows through from (a) and (b)), (d) the time to reach a stated
  amplitude (follows through from (b)); or beats, (a) the natural frequency, (b) the coefficient A in
  ``x = A (cos wt - cos w0 t)``, (c) x(t), (d) the beat period (follows through from (a)).

The closed forms are checked by substituting them into ``m x'' + c x' + k x = F0 cos(wt)`` numerically at three times
and against ``x(0) = x'(0) = 0``; the hard answers are also recomputed from the closed form of x(t).
"""

from __future__ import annotations

import math

import sympy

from ..answers import exact_value
from ..core import attempts, check, generator, rational_text, rounds_unambiguously
from ._s2c1_kit import (
    T,
    expr_options,
    fraction_options,
    machine,
    number_options,
    plain,
    prompt_part,
    say_marked,
    small_fraction,
    solves_ode,
)

t = T
_R = sympy.Rational
_CONTEXTS = [
    (
        "lab",
        "A {m} kg mass hangs from a spring of stiffness {k} N/m in a laboratory.",
        [1, 2, 3, 4, 5],
        "a motor-driven arm",
    ),
    (
        "pump",
        "A {m} kg pump is mounted on springs with combined stiffness {k} N/m.",
        [10, 12, 15, 20, 25, 40],
        "its unbalanced motor",
    ),
]
_F0 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40]


def _draw_system(rng):
    """(context text template, m, k, w0, source) with k = m w0² a whole number."""
    for _ in attempts(100, "a spring-mass system"):
        name, template, masses, source = rng.choice(_CONTEXTS)
        m = rng.choice(masses)
        w0 = rng.choice([3, 4, 5, 6, 7, 8] if name == "lab" else [5, 6, 7, 8, 9, 10])
        return template, m, m * w0 * w0, w0, source
    raise AssertionError("unreachable")


def _draw_force(rng, m):
    """A driving force no more than ten times the mass in newtons (an acceleration of at most 10 m/s² at rest)."""
    return rng.choice([f for f in _F0 if f <= 10 * m] or [1])


def _system_text(template, m, k, source, F0, w, extra="") -> str:
    return (
        f"{template.format(m=m, k=k)} {source[0].upper() + source[1:]} applies the force «{F0}cos({_wt(w)})» newtons "
        f"along the spring{extra}."
    )


def _cos(w):
    return sympy.cos(w * t)


def _coef(n) -> str:
    """A coefficient written in front of a variable: nothing at all for 1."""
    return "" if n == 1 else str(n)


def _wt(w) -> str:
    """The argument of a trig function in the driving force: ``t`` for 1, else ``4t``."""
    return "t" if w == 1 else f"{w}t"


def _check_motion(x_sol, m, c, k, force, x0=0, v0=0):
    pts = (_R(7, 10), _R(13, 10), _R(21, 10))
    check(solves_ode(x_sol, (m, c, k), force, t, pts), "the motion does not satisfy m x'' + c x' + k x = F")
    check(x_sol.subs(t, 0) == x0 and sympy.diff(x_sol, t).subs(t, 0) == v0, "initial conditions are not met")


# ---------------------------------------------------------------------------------------------------------------
# easy and medium: closed forms in t


def _easy(rng):
    for _ in attempts(300, "a non-resonant forced oscillator"):
        template, m, k, w0, source = _draw_system(rng)
        w = rng.choice([w for w in range(1, 13) if w != w0])
        F0 = _draw_force(rng, m)
        A = _R(F0, m * (w0 * w0 - w * w))
        if not small_fraction(A, 20, 40) or not (_R(1, 100) <= abs(A) <= 2):
            continue
        xp = A * _cos(w)
        pts = (_R(7, 10), _R(13, 10), _R(21, 10))
        check(solves_ode(xp, (m, 0, k), F0 * _cos(w), t, pts), "x_p does not satisfy the equation")
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w)
            + " Ignoring damping, find the forced response «x_p(t)», the displacement in metres from equilibrium "
            "that oscillates at the driving frequency."
        )
        wrongs = [
            ("forced-oscillator-force-not-divided-by-mass", _R(F0, w0 * w0 - w * w) * _cos(w)),
            ("forced-oscillator-sign-of-detuning", -A * _cos(w)),
            ("forced-oscillator-frequencies-added", _R(F0, m * (w0 * w0 + w * w)) * _cos(w)),
            ("forced-oscillator-sin-for-cos", A * sympy.sin(w * t)),
            ("forced-oscillator-stiffness-as-frequency", _R(F0, m * (k - w * w)) * _cos(w)),
        ]
        dist = expr_options(xp, wrongs, 4, var=t)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(xp),
            answer_text=plain(xp),
            answer_kind="expression",
            distractors=dist,
            working=[
                f"{_coef(m)}x'' + {_coef(k)}x = {F0}cos({_wt(w)}), so x'' + {_coef(w0 * w0)}x = {rational_text(_R(F0, m))}cos({_wt(w)}).",
                f"Try x_p = A cos({_wt(w)}): A({w0 * w0} - {w * w}) = {rational_text(_R(F0, m))}, so A = {rational_text(A)}.",
            ],
        )
    raise AssertionError("unreachable")


def _medium_resonance(rng):
    for _ in attempts(300, "a resonant forced oscillator"):
        template, m, k, w0, source = _draw_system(rng)
        F0 = _draw_force(rng, m)
        rate = _R(F0, 2 * m * w0)
        if not small_fraction(rate, 24, 40) or rate > _R(3, 2):
            continue
        x_sol = rate * t * sympy.sin(w0 * t)
        _check_motion(x_sol, m, 0, k, F0 * _cos(w0))
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w0)
            + " The mass starts at rest at its equilibrium position and damping is negligible. Find the displacement "
            "«x(t)» in metres."
        )
        wrongs = [
            ("forced-oscillator-resonance-factor-two", _R(F0, m * w0) * t * sympy.sin(w0 * t)),
            ("forced-oscillator-resonance-no-growth", _R(F0, m * w0 * w0) * _cos(w0)),
            ("forced-oscillator-force-not-divided-by-mass", _R(F0, 2 * w0) * t * sympy.sin(w0 * t)),
            ("forced-oscillator-resonance-wrong-trig", rate * t * _cos(w0)),
            ("forced-oscillator-resonance-frequency-slip", _R(F0, 2 * m * w0 * w0) * t * sympy.sin(w0 * t)),
        ]
        dist = expr_options(x_sol, wrongs, 4, var=t)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(x_sol),
            answer_text=plain(x_sol),
            answer_kind="expression",
            distractors=dist,
            working=[
                f"The natural angular frequency is sqrt({k}/{m}) = {w0} rad/s, equal to the driving frequency, so the "
                f"particular solution is t(A cos({w0}t) + B sin({w0}t)).",
                f"From rest, x = {plain(x_sol)}.",
            ],
        )
    raise AssertionError("unreachable")


def _medium_beats(rng):
    for _ in attempts(300, "a beating forced oscillator"):
        template, m, k, w0, source = _draw_system(rng)
        w = w0 + rng.choice([-2, -1, 1, 2])
        if w < 1:
            continue
        F0 = _draw_force(rng, m)
        A = _R(F0, m * (w0 * w0 - w * w))
        if not small_fraction(A, 20, 40) or not (_R(1, 100) <= abs(A) <= 2):
            continue
        x_sol = A * (_cos(w) - _cos(w0))
        _check_motion(x_sol, m, 0, k, F0 * _cos(w))
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w)
            + " The mass starts at rest at its equilibrium position and damping is negligible. Find the displacement "
            "«x(t)» in metres."
        )
        wrongs = [
            ("forced-oscillator-initial-conditions-ignored", A * _cos(w)),
            ("forced-oscillator-sign-of-detuning", A * (_cos(w0) - _cos(w))),
            ("forced-oscillator-force-not-divided-by-mass", _R(F0, w0 * w0 - w * w) * (_cos(w) - _cos(w0))),
            ("forced-oscillator-frequencies-added", _R(F0, m * (w0 * w0 + w * w)) * (_cos(w) - _cos(w0))),
            ("forced-oscillator-homogeneous-sine", A * (_cos(w) - _cos(w0)) + A * w * sympy.sin(w0 * t) / w0),
        ]
        dist = expr_options(x_sol, wrongs, 4, var=t)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(x_sol),
            answer_text=plain(x_sol),
            answer_kind="expression",
            distractors=dist,
            working=[
                f"x_p = {plain(A * _cos(w))}; the complementary function {w0}-frequency term is fixed by x(0) = 0 and "
                f"x'(0) = 0, giving x = {plain(x_sol)}."
            ],
        )
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# hard: numerical questions


def _hard_peak_frequency(rng):
    for _ in attempts(500, "a damped system with a whole-number peak frequency"):
        name, template, masses, source = rng.choice(_CONTEXTS)
        m = rng.choice(masses)
        cs = [c for c in range(2, 41, 2) if c * c % (2 * m) == 0]
        if not cs:
            continue
        c = rng.choice(cs)
        wr = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
        k = m * wr * wr + c * c // (2 * m)
        w0 = math.sqrt(k / m)
        wd = math.sqrt(k / m - c * c / (4 * m * m))
        # independent check: the amplitude 1/sqrt((k - m w²)² + (c w)²) is greatest at w = wr
        amp = lambda w: 1 / math.sqrt((k - m * w * w) ** 2 + (c * w) ** 2)  # noqa: E731
        check(amp(wr) > amp(wr - 0.01) and amp(wr) > amp(wr + 0.01), "the peak is not at the stated frequency")
        F0 = rng.choice([5, 10, 20, 25, 50])
        damper = "A dashpot" if name == "pump" else "A damper"
        body = "the pump" if name == "pump" else "the mass"
        stem, stem_latex = say_marked(
            f"{template.format(m=m, k=k)} {damper} gives a damping force of «{c}v» newtons when {body} moves with "
            f"velocity «v» m/s, and {source} applies a force «{F0}cos(wt)» newtons, where «w» can be adjusted. At which "
            "angular frequency «w» (rad/s) is the amplitude of the steady vibration greatest? Give your answer "
            "correct to 2 decimal places."
        )
        slip_a = k / m - c * c / (m * m)
        slip_b = k / m - c * c / (2 * m)
        wrongs = [
            ("forced-oscillator-natural-frequency-used", w0),
            ("forced-oscillator-damped-frequency-used", wd),
            ("forced-oscillator-peak-damping-slip", math.sqrt(slip_a) if slip_a > 0 else None),
            ("forced-oscillator-peak-damping-slip", math.sqrt(slip_b) if slip_b > 0 else None),
        ]
        dist = number_options(float(wr), 2, wrongs, 4)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=float(wr),
            answer_text=f"{wr:.2f}",
            answer_kind="number",
            distractors=dist,
            working=[
                f"The steady amplitude is F0/sqrt((k - mw²)² + (cw)²), greatest when w² = k/m - c²/(2m²) = {wr * wr}, "
                f"so w = {wr:.2f} rad/s."
            ],
        )
    raise AssertionError("unreachable")


def _hard_beat_period(rng):
    for _ in attempts(300, "a beat period"):
        template, m, k, w0, source = _draw_system(rng)
        w = w0 + rng.choice([-1, 1])
        if w < 1:
            continue
        F0 = _draw_force(rng, m)
        period = 2 * math.pi / abs(w0 - w)
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w)
            + " Damping is negligible, so the vibration grows and fades repeatedly. Find the time in seconds between "
            "successive moments when the amplitude of the vibration is greatest, correct to 3 decimal places."
        )
        wrongs = [
            ("forced-oscillator-beat-period-slip", 2 * math.pi / (w0 + w)),
            ("forced-oscillator-beat-period-slip", 4 * math.pi / abs(w0 - w)),
            ("forced-oscillator-beat-period-slip", math.pi / abs(w0 - w)),
            ("forced-oscillator-beat-period-slip", 2 * math.pi / w0),
        ]
        dist = number_options(period, 3, wrongs, 4)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=round(period, 3),
            answer_text=f"{period:.3f}",
            answer_kind="number",
            distractors=dist,
            working=[
                f"The natural frequency is {w0} rad/s and the driving frequency is {w} rad/s. From rest, "
                f"x = A(cos {w}t - cos {w0}t) = 2A sin({abs(w0 - w)}t/2) sin({w0 + w}t/2) up to sign, whose envelope "
                f"repeats every 2π/{abs(w0 - w)} = {period:.3f} s."
            ],
        )
    raise AssertionError("unreachable")


def _hard_peak_displacement(rng):
    for _ in attempts(300, "a largest displacement"):
        template, m, k, w0, source = _draw_system(rng)
        w = w0 + rng.choice([-2, -1, 1, 2])
        if w < 1:
            continue
        F0 = _draw_force(rng, m)
        peak = _R(2 * F0, m * abs(w0 * w0 - w * w))
        if not small_fraction(peak, 30, 60) or peak > 4:
            continue
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w)
            + " The mass starts at rest at its equilibrium position and damping is negligible, so the vibration "
            "beats: its displacement oscillates inside a slowly varying envelope. Find the largest value of that "
            "envelope, the greatest amplitude of the beats, in metres."
        )
        A = _R(F0, m * (w0 * w0 - w * w))
        wrongs = [
            ("forced-oscillator-beat-amplitude-slip", abs(A)),
            ("forced-oscillator-beat-amplitude-slip", 2 * abs(A) * m),
            ("forced-oscillator-force-not-divided-by-mass", _R(2 * F0, abs(w0 * w0 - w * w))),
            ("forced-oscillator-beat-amplitude-slip", _R(2 * F0, m * (w0 * w0 + w * w))),
        ]
        dist = expr_options(peak, wrongs, 4)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(peak),
            answer_text=rational_text(peak),
            answer_kind="expression",
            distractors=dist,
            working=[
                f"x = A(cos {w}t - cos {w0}t) with A = {rational_text(A)}, whose beat envelope has greatest value 2|A| = "
                f"{rational_text(peak)} m."
            ],
        )
    raise AssertionError("unreachable")


def _hard_growth_time(rng):
    for _ in attempts(500, "a resonant growth time"):
        template, m, k, w0, source = _draw_system(rng)
        F0 = _draw_force(rng, m)
        limit = rng.choice([_R(1, 20), _R(1, 10), _R(1, 5), _R(1, 4), _R(1, 2)])
        rate = _R(F0, 2 * m * w0)
        time = limit / rate
        value = float(time)
        if not (0.2 <= value <= 60 and rounds_unambiguously(value, 2, 0.1)):
            continue
        stem, stem_latex = say_marked(
            _system_text(template, m, k, source, F0, w0)
            + " The mass starts at rest at its equilibrium position and damping is negligible. The vibration is "
            f"safe only while its amplitude is below {float(limit):g} m. After how many seconds does the amplitude of "
            "the vibration reach this size? Give your answer correct to 2 decimal places."
        )
        wrongs = [
            ("forced-oscillator-resonance-factor-two", float(limit / (rate * 2))),
            ("forced-oscillator-resonance-factor-two", float(2 * limit / rate)),
            ("forced-oscillator-force-not-divided-by-mass", float(limit * 2 * w0 / F0)),
            ("forced-oscillator-resonance-frequency-slip", float(limit * 2 * m * w0 * w0 / F0)),
        ]
        dist = number_options(value, 2, wrongs, 4)
        if len(dist) < 2:
            continue
        return dict(
            text=stem,
            latex=stem_latex,
            answer=round(value, 2),
            answer_text=f"{value:.2f}",
            answer_kind="number",
            distractors=dist,
            working=[
                f"The driving frequency equals the natural frequency {w0} rad/s, so the amplitude grows as "
                f"F0 t/(2 m w0) = {rational_text(rate)}t metres, which reaches {float(limit):g} m at t = {value:.2f} s."
            ],
        )
    raise AssertionError("unreachable")


@generator(
    "s2c1_forced_oscillator_resonance",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_forced_oscillator_resonance(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return (_medium_resonance if rng.random() < 0.5 else _medium_beats)(rng)
    pick = rng.random()
    if pick < 0.3:
        return _hard_peak_frequency(rng)
    if pick < 0.55:
        return _hard_beat_period(rng)
    if pick < 0.75:
        return _hard_peak_displacement(rng)
    return _hard_growth_time(rng)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_forced_oscillator_resonance", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_forced_oscillator_resonance(rng, tier="exam"):
    """Exam tier: a resonance or beats chain in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam forced-oscillator question with enough distractors"):
        try:
            return _resonance_exam(rng) if rng.random() < 0.5 else _beats_exam(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _resonance_exam(rng):
    template, m, k, w0, source = _draw_system(rng)
    F0 = _draw_force(rng, m)
    rate = _R(F0, 2 * m * w0)
    if not small_fraction(rate, 24, 40) or rate > _R(3, 2):
        raise _Redraw
    x_sol = rate * t * sympy.sin(w0 * t)
    _check_motion(x_sol, m, 0, k, F0 * _cos(w0))
    limit = rng.choice([_R(1, 20), _R(1, 10), _R(1, 5), _R(1, 4), _R(1, 2)])
    value = float(limit / rate)
    if not (0.2 <= value <= 60 and rounds_unambiguously(value, 2, 0.1)):
        raise _Redraw

    a_dist = _int_options(
        w0,
        [
            ("forced-oscillator-frequency-slip", w0 * w0),
            ("forced-oscillator-frequency-slip", k // m * 2),
            ("forced-oscillator-frequency-slip", k),
            ("forced-oscillator-frequency-slip", m * w0),
        ],
    )
    b_wrongs = [
        ("forced-oscillator-resonance-factor-two", _R(F0, m * w0)),
        ("forced-oscillator-force-not-divided-by-mass", _R(F0, 2 * w0)),
        ("forced-oscillator-resonance-frequency-slip", _R(F0, 2 * m * w0 * w0)),
        ("forced-oscillator-resonance-frequency-slip", _R(F0, 2 * m)),
    ]
    b_dist = fraction_options(rate, b_wrongs, 4)
    c_wrongs = [
        ("forced-oscillator-resonance-factor-two", _R(F0, m * w0) * t * sympy.sin(w0 * t)),
        ("forced-oscillator-resonance-no-growth", _R(F0, m * w0 * w0) * _cos(w0)),
        ("forced-oscillator-resonance-wrong-trig", rate * t * _cos(w0)),
        ("forced-oscillator-force-not-divided-by-mass", _R(F0, 2 * w0) * t * sympy.sin(w0 * t)),
    ]
    c_dist = expr_options(x_sol, c_wrongs, 4, var=t)
    d_wrongs = [
        ("forced-oscillator-resonance-factor-two", float(limit / (rate * 2))),
        ("forced-oscillator-resonance-factor-two", float(2 * limit / rate)),
        ("forced-oscillator-resonance-frequency-slip", float(limit * w0 / rate)),
        ("forced-oscillator-resonance-frequency-slip", float(rate / limit)),
    ]
    d_dist = number_options(value, 2, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem, stem_latex = say_marked(
        _system_text(template, m, k, source, F0, w0)
        + " The mass starts at rest at its equilibrium position and damping is negligible."
    )
    limit_text = f"{float(limit):g}"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part("Find the natural angular frequency of the spring-mass system, in rad/s."),
                answer_kind="number",
                answer=int(w0),
                answer_text=str(w0),
                marks=1,
                working=[f"sqrt(k/m) = sqrt({k}/{m}) = {w0} rad/s."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    "The displacement has the form «x(t) = a t sin(w0 t)» where «w0» is the natural frequency. Find "
                    "«a», the rate in metres per second at which the amplitude grows."
                ),
                answer_kind="fraction",
                answer=exact_value(rate),
                answer_text=rational_text(rate),
                marks=3,
                follow_through=f"{F0}/(2*{m}*part_a)",
                working=[f"a = F0/(2 m w0) = {F0}/(2 x {m} x {w0}) = {rational_text(rate)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Write the displacement «x(t)» in metres."),
                answer_kind="expression",
                answer=machine(x_sol),
                answer_text=plain(x_sol),
                marks=1,
                follow_through="part_b*t*sin(part_a*t)",
                working=[f"x = {plain(x_sol)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part(
                    "The vibration is safe only while its amplitude is below «{limit}» m. How many seconds after the "
                    "start does the amplitude reach this size? Give your answer correct to 2 decimal places.",
                    limit=limit_text,
                ),
                answer_kind="number",
                answer=round(value, 2),
                answer_text=f"{value:.2f}",
                marks=2,
                follow_through=f"{limit_text}/part_b",
                working=[f"t = {limit_text}/{rational_text(rate)} = {value:.2f} s."],
                distractors=d_dist,
            ),
        ],
    )


def _int_options(answer: int, wrongs, n: int = 4):
    return number_options(float(answer), 0, [(tag, float(v)) for tag, v in wrongs], n, gap=1)


def _beats_exam(rng):
    template, m, k, w0, source = _draw_system(rng)
    w = w0 + rng.choice([-2, -1, 1, 2])
    if w < 1:
        raise _Redraw
    F0 = _draw_force(rng, m)
    A = _R(F0, m * (w0 * w0 - w * w))
    if not small_fraction(A, 20, 40) or not (_R(1, 100) <= abs(A) <= 2):
        raise _Redraw
    x_sol = A * (_cos(w) - _cos(w0))
    _check_motion(x_sol, m, 0, k, F0 * _cos(w))
    period = 2 * math.pi / abs(w0 - w)
    if not rounds_unambiguously(period, 3, 0.1):
        raise _Redraw

    a_dist = _int_options(
        w0,
        [
            ("forced-oscillator-frequency-slip", w0 * w0),
            ("forced-oscillator-frequency-slip", k // m * 2),
            ("forced-oscillator-frequency-slip", k),
            ("forced-oscillator-frequency-slip", m * w0),
        ],
    )
    b_wrongs = [
        ("forced-oscillator-sign-of-detuning", -A),
        ("forced-oscillator-force-not-divided-by-mass", _R(F0, w0 * w0 - w * w)),
        ("forced-oscillator-frequencies-added", _R(F0, m * (w0 * w0 + w * w))),
        ("forced-oscillator-frequency-slip", _R(F0, m * abs(w0 - w))),
    ]
    b_dist = fraction_options(A, b_wrongs, 4)
    c_wrongs = [
        ("forced-oscillator-initial-conditions-ignored", A * _cos(w)),
        ("forced-oscillator-sign-of-detuning", A * (_cos(w0) - _cos(w))),
        ("forced-oscillator-frequencies-added", _R(F0, m * (w0 * w0 + w * w)) * (_cos(w) - _cos(w0))),
        ("forced-oscillator-force-not-divided-by-mass", _R(F0, w0 * w0 - w * w) * (_cos(w) - _cos(w0))),
    ]
    c_dist = expr_options(x_sol, c_wrongs, 4, var=t)
    d_wrongs = [
        ("forced-oscillator-beat-period-slip", 2 * math.pi / (w0 + w)),
        ("forced-oscillator-beat-period-slip", 4 * math.pi / abs(w0 - w)),
        ("forced-oscillator-beat-period-slip", math.pi / abs(w0 - w)),
        ("forced-oscillator-beat-period-slip", 2 * math.pi / w0),
    ]
    d_dist = number_options(period, 3, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem, stem_latex = say_marked(
        _system_text(template, m, k, source, F0, w)
        + " The mass starts at rest at its equilibrium position and damping is negligible."
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part("Find the natural angular frequency of the spring-mass system, in rad/s."),
                answer_kind="number",
                answer=int(w0),
                answer_text=str(w0),
                marks=1,
                working=[f"sqrt(k/m) = sqrt({k}/{m}) = {w0} rad/s."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    "The displacement has the form «x(t) = a (cos(wt) - cos(w0 t))», where «w» is the driving "
                    "frequency and «w0» the natural frequency. Find «a»."
                ),
                answer_kind="fraction",
                answer=exact_value(A),
                answer_text=rational_text(A),
                marks=3,
                follow_through=f"{F0}/({m}*(part_a**2 - {w}**2))",
                working=[f"a = F0/(m(w0² - w²)) = {F0}/({m}({w0 * w0} - {w * w})) = {rational_text(A)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Write the displacement «x(t)» in metres."),
                answer_kind="expression",
                answer=machine(x_sol),
                answer_text=plain(x_sol),
                marks=1,
                follow_through=f"part_b*(cos({w}*t) - cos(part_a*t))",
                working=[f"x = {plain(x_sol)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part(
                    "The amplitude of the vibration rises and falls. Find the time in seconds between successive "
                    "moments of greatest amplitude, correct to 3 decimal places."
                ),
                answer_kind="number",
                answer=round(period, 3),
                answer_text=f"{period:.3f}",
                marks=2,
                follow_through=f"2*pi/(part_a - {w})" if w0 > w else f"2*pi/({w} - part_a)",
                working=[f"The envelope repeats every 2π/|w0 - w| = 2π/{abs(w0 - w)} = {period:.3f} s."],
                distractors=d_dist,
            ),
        ],
    )
