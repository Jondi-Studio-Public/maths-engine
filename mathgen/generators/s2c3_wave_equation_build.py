"""Generates `s2c3_wave_equation` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_wave_equation`.

``uc2-wave-equation`` (``fourier-analysis-and-pdes``, answer ``number``): the one-dimensional wave equation
``u_tt = c^2 u_xx``, by d'Alembert's formula on the whole line and by separation of variables on a string fixed at both
ends.

* easy: d'Alembert with a string released from rest, ``u(x, 0) = a / (b + x^2)``, ``u_t(x, 0) = 0``; the equation is
  given as ``u_tt = K u_xx`` so the speed is the square root of K. Find u at a point and time.
* medium: either d'Alembert with an initial velocity g (constant, linear or quadratic, so its integral is exact), or a
  string fixed at both ends released from rest from a sum of two sine modes (find u at a point and time).
* hard: a string fixed at both ends with tension T and density rho (so c = sqrt(T / rho)), released with both a sine-sum
  displacement and a sine-sum velocity, time in milliseconds; the velocity term of each mode is divided by its angular
  frequency.
* exam (``exd_x_s2c3_wave_equation``): the hard set-up as a chain: (a) c, (b) the fundamental frequency (follows through
  from (a)), (c) the angular frequency of the second mode (follows through from (b)), (d) u at a point and time.

Every key is checked by central differences: the value satisfies u_tt = c^2 u_xx at the point, and (for the bounded string)
the end conditions and the initial displacement and velocity.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator
from ._s2c3_common import fixed, key_ok, number_part, places_text, prompt_fields, simpson, stem_fields, wrong_numbers

DP = 3
_PDE = ("∂²u/∂t² = c² ∂²u/∂x²", r"\frac{\partial^{2}u}{\partial t^{2}}=c^{2}\,\frac{\partial^{2}u}{\partial x^{2}}")


def _num(x: float) -> str:
    return f"{x:g}"


# -- d'Alembert -------------------------------------------------------------------------------------------------------


def _profile(a: int, b: int):
    return lambda s: a / (b + s * s)


def _velocity_integral(kind: str, k: int, c: float, x: float, t: float) -> float:
    """(1/(2c)) * integral of g from x - ct to x + ct, in closed form."""
    if kind == "const":
        return k * t
    if kind == "linear":
        return k * x * t
    return k * (x * x * t + c * c * t**3 / 3)  # g = k s^2


def _g_text(kind: str, k: int) -> str:
    lead = "" if k == 1 else ("-" if k == -1 else str(k))
    return {"const": str(k), "linear": f"{lead}x", "quad": f"{lead}x²"}[kind]


def _g_fn(kind: str, k: int):
    return {"const": lambda s: k, "linear": lambda s: k * s, "quad": lambda s: k * s * s}[kind]


def _dalembert_u(f, g, c, x, t) -> float:
    n = 2000
    if g is None:
        integral = 0.0
    elif t >= 0:
        integral = simpson(lambda s: g(s) + 0 * s, x - c * t, x + c * t, n)
    else:
        integral = -simpson(lambda s: g(s) + 0 * s, x + c * t, x - c * t, n)
    return 0.5 * (f(x - c * t) + f(x + c * t)) + integral / (2 * c)


def _check_dalembert(f, g, c, x, t, value):
    """Numerical check: the value matches Simpson quadrature of the velocity term, and u_tt = c^2 u_xx there."""

    def u(xx, tt):
        return _dalembert_u(f, g, c, xx, tt)

    check(abs(u(x, t) - value) < 1e-9, "closed form and quadrature disagree")
    h = 1e-3
    utt = (u(x, t + h) - 2 * u(x, t) + u(x, t - h)) / (h * h)
    uxx = (u(x + h, t) - 2 * u(x, t) + u(x - h, t)) / (h * h)
    check(abs(utt - c * c * uxx) < 1e-3 * (1 + abs(utt)), "u does not satisfy the wave equation")
    check(abs(u(x, 0.0) - f(x)) < 1e-12, "initial displacement differs")
    if g is not None:
        dt = 1e-4
        check(abs((u(x, dt) - u(x, -dt)) / (2 * dt) - g(x)) < 1e-4 * (1 + abs(g(x))), "initial velocity differs")


_DALEMBERT_OPENINGS = (
    "The displacement u(x, t) of a long string satisfies ",
    "A very long string has displacement u(x, t) governed by ",
    "Waves travel along a long string whose displacement u(x, t) obeys ",
)


def _dalembert_pde(K: int, opening: int = 0) -> list:
    return [
        _DALEMBERT_OPENINGS[opening],
        (
            "∂²u/∂t² = %d ∂²u/∂x²" % K,
            r"\frac{\partial^{2}u}{\partial t^{2}}=%d\,\frac{\partial^{2}u}{\partial x^{2}}" % K,
        ),
        " for −∞ < x < ∞, with ",
    ]


def _easy(rng):
    for _ in attempts(200, "an easy d'Alembert problem"):
        c = rng.choice([1, 2, 3, 4])
        K = c * c
        a, b = rng.randint(1, 9), rng.choice([1, 2, 4, 5])
        f = _profile(a, b)
        x0, t0 = rng.randint(-4, 4), rng.choice([1, 2, 3])
        v = 0.5 * (f(x0 - c * t0) + f(x0 + c * t0))
        if c == 1 or not key_ok(v, DP, 0.08) or abs(v - f(x0)) < 0.01:
            continue
        _check_dalembert(f, None, c, x0, t0, v)
        cands = [
            (f(x0 - c * t0) + f(x0 + c * t0), "dalembert-half-dropped"),
            (f(x0 - c * t0), "dalembert-one-pulse-only"),
            (f(x0 + c * t0), "dalembert-one-pulse-only"),
            (0.5 * (f(x0 - K * t0) + f(x0 + K * t0)), "wave-speed-not-square-rooted"),
            (f(x0), "wave-no-time-evolution"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        return dict(
            **stem_fields(
                *_dalembert_pde(K, rng.randrange(len(_DALEMBERT_OPENINGS))),
                ("u(x, 0) = %d/(%d + x²)" % (a, b), r"u(x,0)=\frac{%d}{%d+x^{2}}" % (a, b)),
                " and ",
                ("∂u/∂t (x, 0) = 0", r"\frac{\partial u}{\partial t}(x,0)=0"),
                f". Find u({x0}, {t0}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"c² = {K}, so c = {c}. d'Alembert: u(x, t) = ½[f(x − ct) + f(x + ct)] = ½[f({x0 - c * t0}) + f({x0 + c * t0})] = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _medium_dalembert(rng):
    for _ in attempts(300, "a medium d'Alembert problem"):
        c = rng.choice([2, 3, 4])
        K = c * c
        a, b = rng.randint(1, 9), rng.choice([1, 2, 4, 5])
        f = _profile(a, b)
        kind = rng.choice(["const", "linear", "quad"])
        k = rng.choice([-3, -2, -1, 1, 2, 3, 4])
        g = _g_fn(kind, k)
        x0, t0 = rng.randint(-3, 3), rng.choice([1, 2])
        v = 0.5 * (f(x0 - c * t0) + f(x0 + c * t0)) + _velocity_integral(kind, k, c, x0, t0)
        if not key_ok(v, DP, 0.05) or abs(v) > 200:
            continue
        _check_dalembert(f, g, c, x0, t0, v)
        base = 0.5 * (f(x0 - c * t0) + f(x0 + c * t0))
        integral = 2 * c * _velocity_integral(kind, k, c, x0, t0)
        cands = [
            (base + integral, "dalembert-velocity-term-undivided"),
            (base + integral / 2, "dalembert-velocity-term-half"),
            (base, "dalembert-velocity-term-dropped"),
            (
                0.5 * (f(x0 - K * t0) + f(x0 + K * t0)) + _velocity_integral(kind, k, K, x0, t0),
                "wave-speed-not-square-rooted",
            ),
            (base - _velocity_integral(kind, k, c, x0, t0), "dalembert-velocity-sign"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        return dict(
            **stem_fields(
                *_dalembert_pde(K, rng.randrange(len(_DALEMBERT_OPENINGS))),
                ("u(x, 0) = %d/(%d + x²)" % (a, b), r"u(x,0)=\frac{%d}{%d+x^{2}}" % (a, b)),
                " and ",
                (
                    "∂u/∂t (x, 0) = " + _g_text(kind, k),
                    r"\frac{\partial u}{\partial t}(x,0)=" + _g_text(kind, k).replace("²", "^{2}"),
                ),
                f". Find u({x0}, {t0}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"c = {c}. u(x, t) = ½[f(x − ct) + f(x + ct)] + (1/2c)∫ g(s) ds from x − ct to x + ct, "
                f"so u({x0}, {t0}) = {fixed(base, DP)} {'-' if v - base < 0 else '+'} {fixed(abs(v - base), DP)} = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- bounded string ------------------------------------------------------------------------------------------------------


class String:
    """u(x, t) = sum of [A_n cos(w_n t) + (B_n / w_n) sin(w_n t)] sin(n pi x / L), w_n = n pi c / L (x in m, t in s)."""

    def __init__(self, L, c, A, B):
        self.L, self.c, self.A, self.B = L, c, A, B

    def w(self, n):
        return n * math.pi * self.c / self.L

    def value(self, x, t, w=None, velocity="divided", swap=False):
        w = w or self.w
        total = 0.0
        for n in sorted(set(self.A) | set(self.B)):
            wn = w(n)
            a, b = self.A.get(n, 0.0), self.B.get(n, 0.0)
            vel = b / wn if velocity == "divided" else b
            if swap:
                term = a * math.sin(wn * t) + vel * math.cos(wn * t)
            else:
                term = a * math.cos(wn * t) + vel * math.sin(wn * t)
            total += term * math.sin(n * math.pi * x / self.L)
        return total


def _check_string(s: String, x0: float, t0: float) -> None:
    h, dt = 1e-4 * s.L, 1e-6
    u = s.value
    utt = (u(x0, t0 + dt) - 2 * u(x0, t0) + u(x0, t0 - dt)) / (dt * dt)
    uxx = (u(x0 + h, t0) - 2 * u(x0, t0) + u(x0 - h, t0)) / (h * h)
    check(abs(utt - s.c**2 * uxx) < 2e-3 * (1 + abs(utt)), "u does not satisfy the wave equation")
    check(abs(u(0.0, t0)) < 1e-9 and abs(u(s.L, t0)) < 1e-9, "the ends move")
    for xs in (0.3 * s.L, 0.7 * s.L):
        init = sum(a * math.sin(n * math.pi * xs / s.L) for n, a in s.A.items())
        vel = sum(b * math.sin(n * math.pi * xs / s.L) for n, b in s.B.items())
        check(abs(u(xs, 0.0) - init) < 1e-9, "initial displacement differs")
        check(abs((u(xs, dt) - u(xs, -dt)) / (2 * dt) - vel) < 1e-4 * (1 + abs(vel)), "initial velocity differs")


def _modes_text(coefs: dict, L: float, first_unit: str) -> tuple[str, str]:
    plain, tex = "", ""
    for i, (n, c) in enumerate(sorted(coefs.items())):
        mag = abs(c)
        sign = ("-" if c < 0 else "") if i == 0 else (" - " if c < 0 else " + ")
        coeff = "" if mag == 1 else f"{mag} "
        arg = f"{'' if n == 1 else n}πx/{_num(L)}"
        plain += f"{sign}{coeff}sin({arg})"
        tex += f"{sign}{'' if mag == 1 else mag}\\sin\\frac{{{'' if n == 1 else n}\\pi x}}{{{_num(L)}}}"
    return plain, tex


_LENGTHS = (0.6, 0.9, 1.2, 1.5)
_SPEEDS = (30, 40, 50, 60, 80, 100)
_DENSITIES = (0.004, 0.005, 0.008, 0.01, 0.02)


def _string_candidates(s: String, x0: float, t0: float, c_from_tension=None):
    L, c = s.L, s.c
    out = [
        (s.value(x0, t0, velocity="raw"), "wave-velocity-coefficient-not-divided"),
        (s.value(x0, t0, w=lambda n: n * c / L), "wave-frequency-pi-dropped"),
        (s.value(x0, t0, w=lambda n: n * math.pi / L), "wave-frequency-c-dropped"),
        (s.value(x0, t0, w=lambda n: math.pi * c / L), "wave-mode-frequency-not-scaled"),
        (s.value(x0, t0, swap=True), "wave-cos-sin-swap"),
        (s.value(x0, 0.0), "wave-no-time-evolution"),
    ]
    return out


def _draw_string(rng, with_velocity: bool):
    L = rng.choice(_LENGTHS)
    c = rng.choice(_SPEEDS)
    m, n = sorted(rng.sample([1, 2, 3, 4], 2))
    amp = lambda: rng.choice([-1, 1]) * rng.randint(1, 8)  # noqa: E731
    A = {m: amp(), n: amp()}
    B = {}
    if with_velocity:
        picks = rng.sample([1, 2, 3], 2)
        B = {p: rng.choice([-1, 1]) * rng.randint(1, 6) * 100 for p in sorted(picks)}
    return String(L, c, A, B)


def _string_point(rng, s: String):
    j = rng.choice([1, 2, 3, 4, 5])
    return j * s.L / 6, rng.randint(1, 12)


def _string_question(rng, with_velocity: bool, by_tension: bool):
    for _ in attempts(300, "a bounded-string problem"):
        s = _draw_string(rng, with_velocity)
        x0, t_ms = _string_point(rng, s)
        t0 = t_ms / 1000
        v = s.value(x0, t0)
        if not key_ok(v, DP, 0.05) or abs(v) > 60:
            continue
        # both modes must be oscillating, not at a node, at the chosen point
        if any(abs(math.sin(n * math.pi * x0 / s.L)) < 1e-9 for n in s.A):
            continue
        _check_string(s, x0, t0)
        cands = _string_candidates(s, x0, t0)
        if with_velocity:
            cands.append((s.value(x0, t_ms), "wave-time-unit-ms-as-s"))
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        return s, x0, t_ms, v, d
    raise AssertionError  # pragma: no cover


def _tension_clause(rng, c: int):
    rho = rng.choice(_DENSITIES)
    T = round(rho * c * c, 3)
    check(abs(math.sqrt(T / rho) - c) < 1e-9, "tension and density do not give the speed")
    return rho, T


def _string_stem(rng, s: String, tension: bool, with_velocity: bool, rho=None, T=None):
    a_plain, a_tex = _modes_text(s.A, s.L, "mm")
    items = [
        f"A taut string of length {_num(s.L)} m is fixed at both ends. Its displacement u(x, t), in millimetres, "
        f"x metres along the string and t seconds after release, satisfies ",
        _PDE,
    ]
    if tension:
        items += [f", where c² = T/ρ, the tension is T = {_num(T)} N and the linear density is ρ = {_num(rho)} kg/m. "]
    else:
        items += [f" with c = {s.c} m/s. "]
    if with_velocity:
        b_plain, b_tex = _modes_text(s.B, s.L, "mm/s")
        items += [
            "The string is released with u(x, 0) = ",
            (a_plain, a_tex),
            " mm and ∂u/∂t (x, 0) = ",
            (b_plain, b_tex),
            " mm/s",
        ]
    else:
        items += ["The string is released from rest from the shape u(x, 0) = ", (a_plain, a_tex), " mm"]
    return items


def _medium_string(rng):
    s, x0, t_ms, v, d = _string_question(rng, False, False)
    return dict(
        **stem_fields(
            *_string_stem(rng, s, False, False),
            f". Find u({_num(x0)}, {t_ms / 1000:g}), correct to {places_text(DP)}.",
        ),
        **number_part(v, DP),
        working=[
            f"u = Σ Aₙ cos(ωₙt) sin(nπx/L) with ωₙ = nπc/L: here ω₁ = {fixed(s.w(1), 3)} rad/s, "
            f"and u({_num(x0)}, {t_ms / 1000:g}) = {fixed(v, DP)} mm."
        ],
        distractors=d,
    )


def _hard(rng):
    for _ in attempts(100, "a hard wave problem"):
        s, x0, t_ms, v, d = _string_question(rng, True, True)
        rho, T = _tension_clause(rng, s.c)
        cands = [
            (
                String(s.L, T / rho, s.A, s.B).value(x0, t_ms / 1000),
                "wave-speed-not-square-rooted",
            ),
            (String(s.L, math.sqrt(rho / T), s.A, s.B).value(x0, t_ms / 1000), "wave-speed-inverted"),
        ]
        d2 = wrong_numbers(v, DP, cands + [(float(x), tag) for x, tag in []], 2)
        # keep the original distractors and add the speed slips when they are far enough from the key
        merged = []
        seen = {fixed(v, DP)}
        for item in d + d2:
            if item.text not in seen:
                merged.append(item)
                seen.add(item.text)
        merged = merged[:4]
        if len(merged) < 3:
            continue
        return dict(
            **stem_fields(
                *_string_stem(rng, s, True, True, rho, T),
                f". Find u({_num(x0)}, {t_ms / 1000:g}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"c = √(T/ρ) = {s.c} m/s, ωₙ = nπc/L. Each mode is [Aₙ cos(ωₙt) + (Bₙ/ωₙ) sin(ωₙt)] sin(nπx/L); "
                f"summing at x = {_num(x0)}, t = {t_ms / 1000:g} gives {fixed(v, DP)} mm."
            ],
            distractors=merged,
        )
    raise AssertionError  # pragma: no cover


@generator(
    "s2c3_wave_equation",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_wave_equation(rng, tier="easy"):
    """The wave equation by d'Alembert and by separation of variables; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_dalembert, _medium_string])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("exd_x_s2c3_wave_equation", level=21, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_wave_equation(rng, tier="exam"):
    """Exam chain: wave speed, fundamental frequency, second-mode angular frequency and the displacement at a point."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(100, "an exam string"):
        s, x0, t_ms, v, d_val = _string_question(rng, True, True)
        rho, T = _tension_clause(rng, s.c)
        c, L = s.c, s.L
        f1 = c / (2 * L)
        w2 = 2 * math.pi * c / L
        if not (key_ok(f1, DP, 0.5) and key_ok(w2, DP, 1.0)):
            continue
        a_d = wrong_numbers(
            float(c),
            1,
            [
                (T / rho, "wave-speed-not-square-rooted"),
                (math.sqrt(T) / rho, "wave-speed-root-of-tension-only"),
                (T * rho, "wave-speed-product"),
                (math.sqrt(T / rho) * 2, "wave-speed-doubled"),
            ],
            3,
        )
        b_d = wrong_numbers(
            f1,
            DP,
            [
                (c / L, "wave-fundamental-half-wavelength-missed"),
                (c / (4 * L), "wave-fundamental-quarter-wavelength"),
                (2 * math.pi * c / L, "wave-frequency-hertz-vs-radians"),
                (L / (2 * c), "wave-frequency-inverted"),
                (c * 2 * L, "wave-frequency-inverted"),
            ],
            3,
        )
        c_d = wrong_numbers(
            w2,
            DP,
            [
                (math.pi * c / L, "wave-mode-frequency-not-scaled"),
                (2 * c / L, "wave-frequency-pi-dropped"),
                (2 * math.pi * c / L / 2, "wave-mode-frequency-not-scaled"),
                (4 * math.pi * c / L, "wave-mode-frequency-squared"),
                (2 * math.pi * f1, "wave-frequency-of-first-mode"),
            ],
            3,
        )
        d_d = d_val
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break
    else:
        raise AssertionError  # pragma: no cover
    check(abs(2 * math.pi * 2 * f1 - w2) < 1e-9, "second-mode frequency and fundamental disagree")
    xs = _num(x0)
    ts = f"{t_ms / 1000:g}"
    return dict(
        **stem_fields(*_string_stem(rng, s, True, True, rho, T), "."),
        parts=[
            dict(
                **prompt_fields("Find the wave speed c, in m/s, correct to 1 decimal place."),
                **number_part(float(c), 1),
                marks=2,
                sensible_range=(0, None),
                working=[f"c = √(T/ρ) = √({_num(T)}/{_num(rho)}) = {c}.0 m/s."],
                distractors=a_d,
            ),
            dict(
                **prompt_fields(
                    f"Find the fundamental frequency of the string, in hertz, correct to {places_text(DP)}."
                ),
                **number_part(f1, DP),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/(2*({Fraction(str(L))}))",
                working=[f"The fundamental has wavelength 2L, so f = c/(2L) = {fixed(f1, DP)} Hz."],
                distractors=b_d,
            ),
            dict(
                **prompt_fields(
                    f"Find the angular frequency ω₂ of the second mode, in rad/s, correct to {places_text(DP)}."
                ),
                **number_part(w2, DP),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*pi*part_a/({Fraction(str(L))})",
                working=[f"ω₂ = 2πc/L = {fixed(w2, DP)} rad/s."],
                distractors=c_d,
            ),
            dict(
                **prompt_fields(f"Find u({xs}, {ts}), correct to {places_text(DP)}."),
                **number_part(v, DP),
                marks=3,
                working=[f"Summing the modes [Aₙ cos(ωₙt) + (Bₙ/ωₙ) sin(ωₙt)] sin(nπx/L) gives {fixed(v, DP)} mm."],
                distractors=d_d,
            ),
        ],
    )
