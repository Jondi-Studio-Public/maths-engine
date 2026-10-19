"""Generates `s2c3_heat_equation` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_heat_equation`.

``uc2-heat-equation-initial-fit`` (``fourier-analysis-and-pdes``, answer ``number``): the heat equation
``u_t = k u_xx`` on a rod 0 < x < L, solved by separation of variables and fitted to the initial temperature.

* easy: Dirichlet ends (both held at 0) or Neumann ends (both insulated), and an initial temperature that is already
  a finite sum of modes; the learner picks sine or cosine modes from the ends, attaches the decay k (n pi/L)^2 to each
  mode, and evaluates u at a point and time.
* medium: an initial temperature that is a function (constant, ramp, tent or heated middle section); the learner finds
  the Fourier coefficients, then asks either for one coefficient of the solution or for u at a point and time (several
  terms matter, the later ones are small).
* hard: Dirichlet ends held at different temperatures T1 and T2, so the steady state T1 + (T2 - T1) x / L is taken out
  first, and the transient is fitted to what is left of a constant or tent initial temperature.
* exam (``exd_x_s2c3_heat_equation``): Dirichlet rod with a constant, tent or heated-middle initial temperature:
  (a) the decay rate of the third mode, (b) its Fourier coefficient, (c) u at a point and time, (d) the time for that
  mode's amplitude to fall to 1 degree (follows through from (a) and (b)).

Each key is checked by evaluating the series (200 terms) and confirming by central differences that it satisfies
u_t = k u_xx, the end conditions, and, at points away from any jump, the initial temperature.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator
from ._s2c3_common import fixed, key_ok, number_part, places_text, prompt_fields, stem_fields, wrong_numbers
from ._s2c3_fourier import Fn, Piece

DP = 3
_LENGTHS = (12, 18, 24, 30)
_MATERIALS = (
    ("copper", 1.1),
    ("aluminium", 0.97),
    ("brass", 0.35),
    ("iron", 0.23),
    ("steel", 0.12),
)
_PDE = ("∂u/∂t = k ∂²u/∂x²", r"\frac{\partial u}{\partial t}=k\,\frac{\partial^{2}u}{\partial x^{2}}")
_TERMS = 80


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _fmt_k(k: float) -> str:
    return f"{k:g}"


class Case:
    """A rod problem: u(x, t) = steady(x) + sum over modes of coef(n) exp(-rate(n) t) mode_n(x)."""

    def __init__(self, L, k, family, coef, mean=0.0, t1=0.0, t2=0.0, terms=_TERMS, finite=None):
        self.L, self.k, self.family = L, k, family
        self.coef, self.mean, self.t1, self.t2 = coef, mean, t1, t2
        self.terms, self.finite = terms, finite  # `finite`: sorted modes that carry the whole initial temperature

    def modes(self):
        return self.finite if self.finite is not None else range(1, self.terms + 1)

    def rate(self, n):
        return self.k * (n * math.pi / self.L) ** 2

    def steady(self, x):
        return self.t1 + (self.t2 - self.t1) * x / self.L

    def mode(self, n, x, family=None):
        arg = n * math.pi * x / self.L
        return math.sin(arg) if (family or self.family) == "sin" else math.cos(arg)

    def value(self, x, t, rate=None, family=None, only=None, with_mean=True, with_steady=True, decay_mean=False):
        rate = rate or self.rate
        total = self.steady(x) if with_steady else 0.0
        if self.family == "cos" and with_mean:
            total += self.mean * (math.exp(-rate(0) * t) if decay_mean else 1.0)
        for n in self.modes():
            if only is not None and n not in only:
                continue
            total += self.coef(n) * math.exp(-rate(n) * t) * self.mode(n, x, family)
        return total


def _check_solution(case: Case, ic, x0: float, t0: float, ends: tuple[str, float, float]) -> None:
    """u satisfies the equation, the end conditions and, where `ic` is given, the initial temperature."""
    h, dt = 1e-3 * case.L, 1e-3
    u = case.value
    ut = (u(x0, t0 + dt) - u(x0, t0 - dt)) / (2 * dt)
    uxx = (u(x0 + h, t0) - 2 * u(x0, t0) + u(x0 - h, t0)) / (h * h)
    check(abs(ut - case.k * uxx) < 1e-4 * (1 + abs(ut)) + 1e-6, "u does not satisfy the heat equation")
    kind, left, right = ends
    if kind == "dirichlet":
        check(abs(u(0.0, t0) - left) < 1e-6 and abs(u(case.L, t0) - right) < 1e-6, "end temperatures differ")
    else:
        # An insulated end has zero slope, so u differs from its end value only at second order in the step e;
        # the tolerance scales with the temperature, since that second-order term does.
        e = 1e-6 * case.L
        tol = 1e-6 * (1 + abs(u(0.0, t0)) + abs(u(case.L, t0)))
        check(abs(u(e, t0) - u(0.0, t0)) < tol and abs(u(case.L, t0) - u(case.L - e, t0)) < tol, "ends not insulated")
    if ic is not None:
        saved, case.terms = case.terms, 400  # the initial fit converges slowly near jumps and unequal ends
        try:
            scale = 1 + max(abs(v) for v in ic.values()) + abs(case.t1) + abs(case.t2)
            for q in ic:
                check(abs(case.value(q * case.L, 0.0) - ic[q]) < 0.03 * scale, "initial fit differs")
        finally:
            case.terms = saved


# -- wording --------------------------------------------------------------------------------------------------------


def _rod(rng, L, material, k, ends_kind, extra=()):
    ends = {
        "dirichlet": ("Both ends are held at 0 °C.", "Both ends are held at 0 °C."),
        "neumann": (
            "Both ends are insulated, so ∂u/∂x = 0 at x = 0 and x = %d." % L,
            r"\text{Both ends are insulated, so }\frac{\partial u}{\partial x}=0\text{ at }x=0\text{ and }x=%d." % L,
        ),
    }
    end_plain, end_tex = ends[ends_kind]
    plain_ends = (end_plain, end_tex if ends_kind == "neumann" else end_plain)
    return [
        f"The temperature u(x, t), in °C, at distance x cm from one end of {_a(material)} rod {L} cm long, "
        f"t seconds after time zero, satisfies ",
        _PDE,
        f" with k = {_fmt_k(k)} cm²/s. ",
        plain_ends[0] if ends_kind == "dirichlet" else (plain_ends[0], plain_ends[1]),
        " ",
        *extra,
    ]


def _trig_term(coef: int, n: int, family: str, L: int, first: bool) -> tuple[str, str]:
    func, tfunc = ("sin", r"\sin") if family == "sin" else ("cos", r"\cos")
    arg = f"{'' if n == 1 else n}πx/{L}"
    targ = rf"\frac{{{'' if n == 1 else n}\pi x}}{{{L}}}"
    mag = abs(coef)
    lead = "" if mag == 1 else f"{mag} "
    sign = ("-" if coef < 0 else "") if first else (" - " if coef < 0 else " + ")
    return f"{sign}{lead}{func}({arg})", f"{sign}{'' if mag == 1 else mag}{tfunc}{targ}"


def _finite_ic(coefs: dict, family: str, L: int, constant: int = 0) -> tuple[str, str]:
    plain, tex = "u(x, 0) = ", "u(x, 0) = "
    first = True
    if constant:
        plain += str(constant)
        tex += str(constant)
        first = False
    for n, c in sorted(coefs.items()):
        p, t = _trig_term(c, n, family, L, first)
        plain += p
        tex += t
        first = False
    return plain, tex


def _x_text(x0) -> str:
    return str(int(x0)) if float(x0) == int(x0) else f"{x0:g}"


# -- distractors ----------------------------------------------------------------------------------------------------


def _value_candidates(case: Case, x0: float, t0: float, series: bool):
    L, k = case.L, case.k
    out = [
        (case.value(x0, t0, rate=lambda n: k * (math.pi / L) ** 2 * n), "heat-decay-linear-in-n"),
        (case.value(x0, t0, rate=lambda n: k * n * n / (L * L)), "heat-decay-pi-dropped"),
        (case.value(x0, t0, rate=lambda n: k * n * math.pi / L), "heat-decay-wavenumber-unsquared"),
        (case.value(x0, t0, rate=lambda n: k * (n * math.pi) ** 2 / L), "heat-decay-length-unsquared"),
        (case.value(x0, t0, family="cos" if case.family == "sin" else "sin"), "heat-wrong-mode-family"),
        (case.value(x0, 0.0), "heat-no-decay"),
    ]
    if series:
        out.append((case.value(x0, t0, only={1}), "heat-series-truncated-to-first-term"))
    if case.family == "cos" and case.mean:
        out.append((case.value(x0, t0, with_mean=False), "heat-mean-term-dropped"))
    return out


def _significant_terms(case: Case, x0: float, t0: float) -> int:
    return sum(1 for n in case.modes() if abs(case.coef(n) * math.exp(-case.rate(n) * t0) * case.mode(n, x0)) > 1.5e-3)


# -- easy -----------------------------------------------------------------------------------------------------------


def _draw_t(rng, case: Case, x0: float, lo=4, hi=90, terms=(2, 6)):
    """A whole number of seconds at which the answer is a clean key that several terms of the series build, or None."""
    for t0 in rng.sample(range(lo, hi + 1), hi - lo + 1):
        v = case.value(x0, t0)
        if (
            key_ok(v, DP, 0.05)
            and abs(v) < 400
            and abs(v - case.value(x0, 0.0)) > 1.0
            and terms[0] <= _significant_terms(case, x0, t0) <= terms[1]
        ):
            return t0
    return None


def _point(rng, L):
    return (L // 6) * rng.choice([1, 2, 3, 4, 5])


def _easy(rng):
    for _ in attempts(100, "an easy heat problem"):
        L = rng.choice(_LENGTHS)
        material, k = rng.choice(_MATERIALS)
        family = rng.choice(["sin", "cos"])
        ends_kind = "dirichlet" if family == "sin" else "neumann"
        m, n = sorted(rng.sample([1, 2, 3, 4, 5], 2))
        amp = lambda: rng.choice([-1, 1]) * rng.randint(2, 18) * 5  # noqa: E731
        coefs = {m: amp(), n: amp()}
        constant = rng.choice([20, 30, 40, 50, 60]) if family == "cos" else 0
        case = Case(L, k, family, lambda j, coefs=coefs: coefs.get(j, 0.0), mean=constant, finite=sorted(coefs))
        x0 = _point(rng, L)
        found = None
        for t0 in rng.sample(range(3, 81), 78):
            v = case.value(x0, t0)
            if (
                key_ok(v, DP, 0.05)
                and all(abs(coefs[j] * math.exp(-case.rate(j) * t0) * case.mode(j, x0)) > 0.4 for j in coefs)
                and abs(v) < 400
            ):
                found = t0, v
                break
        if found is None:
            continue
        t0, v = found
        d = wrong_numbers(v, DP, _value_candidates(case, x0, t0, False), 4)
        if len(d) < 2:
            continue
        ends = ("dirichlet", 0.0, 0.0) if ends_kind == "dirichlet" else ("neumann", 0.0, 0.0)
        _check_solution(case, None, x0, t0, ends)
        check(
            abs(case.value(x0, 0.0) - (constant + sum(c * case.mode(j, x0) for j, c in coefs.items()))) < 1e-9,
            "initial",
        )
        ic_plain, ic_tex = _finite_ic(coefs, family, L, constant)
        return dict(
            **stem_fields(
                *_rod(rng, L, material, k, ends_kind),
                "Initially ",
                (ic_plain, ic_tex),
                f". Find u({_x_text(x0)}, {t0}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"The ends are {'held at 0' if family == 'sin' else 'insulated'}, so u = "
                + ("Σ bₙ e^(−k(nπ/L)²t) sin(nπx/L)" if family == "sin" else "a₀/2 + Σ aₙ e^(−k(nπ/L)²t) cos(nπx/L)")
                + f"; matching the initial temperature fixes each coefficient, giving u({_x_text(x0)}, {t0}) = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- medium and hard: initial temperature given as a function ---------------------------------------------------------


def _ic_fn(rng, L: int, kind: str) -> Fn:
    if kind == "const":
        return Fn(float(L), str(L), str(L), [Piece(Fraction(0), Fraction(1), rng.choice([30, 40, 50, 60, 80, 100]), 0)])
    if kind == "ramp":
        slope = rng.choice([1, 2, 3, 4])
        while slope * L > 100:
            slope -= 1
        return Fn(float(L), str(L), str(L), [Piece(Fraction(0), Fraction(1), 0, slope)])
    if kind == "tent":
        s = rng.choice([1, 2, 3, 4])
        while s * L > 120:
            s -= 1
        h = Fraction(1, 2)
        return Fn(float(L), str(L), str(L), [Piece(Fraction(0), h, 0, s), Piece(h, Fraction(1), s * L, -s)])
    heat = rng.choice([40, 50, 60, 80, 100])  # a middle section at a different temperature
    return Fn(
        float(L),
        str(L),
        str(L),
        [
            Piece(Fraction(0), Fraction(1, 4), 0, 0),
            Piece(Fraction(1, 4), Fraction(3, 4), heat, 0),
            Piece(Fraction(3, 4), Fraction(1), 0, 0),
        ],
    )


def _fn_coef(fn: Fn, family: str, n: int) -> float:
    """b_n = (2/L) int f sin(n pi x/L), a_n = (2/L) int f cos(n pi x/L), over [0, L]."""
    return fn.integral("b" if family == "sin" else "a", n * math.pi / fn.L) * 2 / fn.L


def _fn_mean(fn: Fn) -> float:
    return fn.integral("a", 0.0) / fn.L


def _ic_describe(fn: Fn, name="u(x, 0)"):
    return fn.describe(close_last=True, name=name, bare_single=True)


def _ic_items(fn: Fn, kind: str) -> list:
    """The opening of the sentence that states the initial temperature (no final full stop)."""
    L = int(fn.L)
    if kind == "const":
        return [f"Initially the whole rod is at {fn.pieces[0].c0} °C"]
    if kind == "step":
        return [
            f"Initially the middle half of the rod ({_x_text(L / 4)} ≤ x ≤ {_x_text(3 * L / 4)}) is at "
            f"{fn.pieces[1].c0} °C and the rest is at 0 °C"
        ]
    plain, tex = _ic_describe(fn)
    return ["Initially ", (plain, tex)]


def _case_from_fn(fn: Fn, k: float, family: str, t1=0.0, t2=0.0, shift=None) -> Case:
    """Fit u(x,0) = fn (minus the steady state, if the ends differ) with the series for `family`."""
    work = fn
    if shift is not None:
        work = Fn(fn.L, fn.L_text, fn.L_tex, [Piece(p.a, p.b, p.c0 - shift[0], p.c1 - shift[1]) for p in fn.pieces])
    mean = _fn_mean(work) if family == "cos" else 0.0
    return Case(fn.L, k, family, lambda n: _fn_coef(work, family, n), mean=mean, t1=t1, t2=t2)


def _ic_checks(fn: Fn):
    return {q: fn.value_at(Fraction(q).limit_denominator(100)) for q in (0.4, 0.6, 0.45)}


def _medium(rng):
    for _ in attempts(100, "a medium heat problem"):
        L = rng.choice(_LENGTHS)
        material, k = rng.choice(_MATERIALS)
        family = rng.choice(["sin", "sin", "cos"])
        kind = rng.choice(["const", "ramp", "tent", "step"] if family == "sin" else ["ramp", "tent", "step"])
        fn = _ic_fn(rng, L, kind)
        case = _case_from_fn(fn, k, family)
        ends_kind = "dirichlet" if family == "sin" else "neumann"
        if rng.random() < 0.35:
            # one coefficient of the solution
            n = rng.choice([1, 2, 3, 4]) if kind in ("ramp",) else rng.choice([1, 3, 5])
            v = case.coef(n)
            if not key_ok(v, DP, 0.08):
                continue
            check(
                abs(fn.check_coefficient("b" if family == "sin" else "a", n, 2 / fn.L, Fraction(0), Fraction(1)) - v)
                < 1e-7,
                "quad",
            )
            name = ("b" if family == "sin" else "a") + str(n)
            sub = "bₙ" if family == "sin" else "aₙ"
            cands = [
                (v / 2, "fourier-half-range-factor-dropped"),
                (-v, "fourier-coefficient-sign-flip"),
                (_fn_coef(fn, "cos" if family == "sin" else "sin", n), "heat-wrong-mode-family"),
                (v * math.exp(-k * (n * math.pi / L) ** 2 * 10), "heat-decay-applied-to-coefficient"),
                (fn.integral("b" if family == "sin" else "a", float(n)) * 2 / fn.L, "fourier-period-scaling-dropped"),
            ]
            d = wrong_numbers(v, DP, cands, 4)
            if len(d) < 2:
                continue
            sub_tex = sub.replace("ₙ", "_n")
            return dict(
                **stem_fields(
                    *_rod(rng, L, material, k, ends_kind),
                    *_ic_items(fn, kind),
                    ". Write the solution as ",
                    (
                        "u(x, t) = Σ " + sub + " e^(−λₙt) " + ("sin" if family == "sin" else "cos") + "(nπx/L)"
                        if family == "sin"
                        else "u(x, t) = a₀/2 + Σ aₙ e^(−λₙt) cos(nπx/L)",
                        r"u(x,t)=\sum_{n=1}^{\infty} " + sub_tex + r"e^{-\lambda_n t}\sin\frac{n\pi x}{L}"
                        if family == "sin"
                        else r"u(x,t)=\frac{a_0}{2}+\sum_{n=1}^{\infty} a_n e^{-\lambda_n t}\cos\frac{n\pi x}{L}",
                    ),
                    f" and find the coefficient {name[0]}{'₀₁₂₃₄₅₆₇₈₉'[n]}, correct to {places_text(DP)}.",
                ),
                **number_part(v, DP),
                working=[
                    f"{name} = (2/L) ∫₀ᴸ u(x, 0) {'sin' if family == 'sin' else 'cos'}(nπx/L) dx = {fixed(v, DP)}."
                ],
                distractors=d,
            )
        x0 = _point(rng, L)
        t0 = _draw_t(rng, case, x0, 2, 70, (2, 6))
        if t0 is None:
            continue
        v = case.value(x0, t0)
        _check_solution(
            case, _ic_checks(fn), x0, t0, (ends_kind, 0.0, 0.0) if ends_kind == "dirichlet" else ("neumann", 0, 0)
        )
        d = wrong_numbers(v, DP, _value_candidates(case, x0, t0, True), 4)
        if len(d) < 2:
            continue
        return dict(
            **stem_fields(
                *_rod(rng, L, material, k, ends_kind),
                *_ic_items(fn, kind),
                f". Find u({_x_text(x0)}, {t0}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"Expand u(x, 0) in {'sine' if family == 'sin' else 'cosine'} modes (coefficients (2/L)∫ u(x, 0) "
                f"{'sin' if family == 'sin' else 'cos'}(nπx/L) dx), multiply mode n by e^(−k(nπ/L)²t), and sum: "
                f"u({_x_text(x0)}, {t0}) = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _hard(rng):
    for _ in attempts(100, "a hard heat problem"):
        L = rng.choice(_LENGTHS)
        material, k = rng.choice(_MATERIALS)
        t1, t2 = rng.choice([0, 10, 20, 30, 40]), rng.choice([60, 70, 80, 90, 100])
        if rng.random() < 0.5:
            t1, t2 = t2, t1
        kind = rng.choice(["const", "tent", "step"])
        fn = _ic_fn(rng, L, kind)
        if kind == "const":
            # a uniform starting temperature between the end temperatures would be the simplest case; vary it
            fn = Fn(float(L), str(L), str(L), [Piece(Fraction(0), Fraction(1), rng.choice([20, 30, 50, 70, 90]), 0)])
        slope = (t2 - t1) / L
        case = _case_from_fn(fn, k, "sin", t1=t1, t2=t2, shift=(t1, slope))
        x0 = _point(rng, L)
        t0 = _draw_t(rng, case, x0, 2, 70, (2, 6))
        if t0 is None:
            continue
        v = case.value(x0, t0)
        # the initial temperature recovered at points away from jumps
        _check_solution(case, _ic_checks(fn), x0, t0, ("dirichlet", float(t1), float(t2)))
        raw = _case_from_fn(fn, k, "sin")  # coefficients of u(x, 0) itself, steady state not subtracted
        cands = [
            (case.value(x0, t0, with_steady=False), "heat-steady-state-omitted"),
            (
                case.steady(x0) + raw.value(x0, t0, with_steady=False),
                "heat-steady-state-not-subtracted",
            ),
            (case.value(x0, t0, rate=lambda n: k * (math.pi / L) ** 2 * n), "heat-decay-linear-in-n"),
            (case.value(x0, t0, rate=lambda n: k * n * n / (L * L)), "heat-decay-pi-dropped"),
            (case.value(x0, 0.0), "heat-no-decay"),
            (case.value(x0, t0, only={1}), "heat-series-truncated-to-first-term"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 2:
            continue
        ends_plain = f"The end x = 0 is held at {t1} °C and the end x = {L} at {t2} °C."
        return dict(
            **stem_fields(
                f"The temperature u(x, t), in °C, at distance x cm from one end of {_a(material)} rod {L} cm long, t seconds after time zero, satisfies ",
                _PDE,
                f" with k = {_fmt_k(k)} cm²/s. {ends_plain} ",
                *_ic_items(fn, kind),
                f". Find u({_x_text(x0)}, {t0}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"Steady state: {t1} + ({t2 - t1}/{L})x. The transient w = u − steady has zero ends, "
                f"so expand w(x, 0) in sine modes and decay mode n by e^(−k(nπ/L)²t): u({_x_text(x0)}, {t0}) = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


@generator(
    "s2c3_heat_equation",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_heat_equation(rng, tier="easy"):
    """Heat equation on a rod fitted to an initial temperature; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -----------------------------------------------------------------------------------------------------------

_EXAM_MODE = 3


@generator("exd_x_s2c3_heat_equation", level=21, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_heat_equation(rng, tier="exam"):
    """Exam chain: decay rate of mode 3, its coefficient, u at a point, and the time for that mode to fade to 1 degree."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = _EXAM_MODE
    for _ in attempts(300, "an exam rod"):
        L = rng.choice(_LENGTHS)
        material, k = rng.choice(_MATERIALS)
        kind = rng.choice(["const", "tent", "step"])
        fn = _ic_fn(rng, L, kind)
        case = _case_from_fn(fn, k, "sin")
        lam = case.rate(n)
        b = case.coef(n)
        x0 = _point(rng, L)
        if not (key_ok(lam, 5, 0.002) and key_ok(b, DP, 1.5) and b > 0):
            continue
        t_fade = math.log(b / 1.0) / lam
        if not (key_ok(t_fade, 1, 1.0) and t_fade < 400):
            continue
        t0 = _draw_t(rng, case, x0, 2, 60, (2, 6))
        if t0 is None:
            continue
        v = case.value(x0, t0)
        a_d = wrong_numbers(
            lam,
            5,
            [
                (k * (math.pi / L) ** 2 * n, "heat-decay-linear-in-n"),
                (k * n * n / (L * L), "heat-decay-pi-dropped"),
                (k * n * math.pi / L, "heat-decay-wavenumber-unsquared"),
                (k * (n * math.pi) ** 2 / L, "heat-decay-length-unsquared"),
            ],
            3,
        )
        b_d = wrong_numbers(
            b,
            DP,
            [
                (b / 2, "fourier-half-range-factor-dropped"),
                (-b, "fourier-coefficient-sign-flip"),
                (_fn_coef(fn, "cos", n), "heat-wrong-mode-family"),
                (case.coef(1), "heat-wrong-mode-number"),
                (b * 2, "fourier-normalisation-doubled"),
            ],
            3,
        )
        c_d = wrong_numbers(v, DP, _value_candidates(case, x0, t0, True), 3)
        d_d = wrong_numbers(
            t_fade,
            1,
            [
                (math.log(b) / (k * (math.pi / L) ** 2 * n), "heat-decay-linear-in-n"),
                (math.log(b) / (k * n * n / (L * L)), "heat-decay-pi-dropped"),
                (math.log(b / 2) / lam if b > 2 else None, "heat-fade-target-wrong"),
                (math.log(b * 100) / lam, "heat-fade-target-wrong"),
                (1 / lam, "heat-time-constant-not-fade-time"),
            ],
            3,
        )
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break
    else:
        raise AssertionError  # pragma: no cover
    _check_solution(case, _ic_checks(fn), x0, t0, ("dirichlet", 0.0, 0.0))
    check(abs(fn.check_coefficient("b", n, 2 / fn.L, Fraction(0), Fraction(1)) - b) < 1e-7, "quadrature differs")
    check(abs(b * math.exp(-lam * t_fade) - 1.0) < 1e-9, "fade time does not give 1 degree")
    lam_text = fixed(lam, 5)
    return dict(
        **stem_fields(
            *_rod(rng, L, material, k, "dirichlet"),
            *_ic_items(fn, kind),
            ". The solution is u(x, t) = Σ ",
            (
                "bₙ e^(−λₙt) sin(nπx/L)",
                r"b_n e^{-\lambda_n t}\sin\frac{n\pi x}{L}",
            ),
            ".",
        ),
        parts=[
            dict(
                **prompt_fields(f"Find the decay rate λ₃ of the third mode, in s⁻¹, correct to 5 decimal places."),
                **number_part(lam, 5),
                marks=2,
                sensible_range=(0, None),
                working=[f"λₙ = k(nπ/L)², so λ₃ = {_fmt_k(k)}(3π/{L})² = {lam_text}."],
                distractors=a_d,
            ),
            dict(
                **prompt_fields(f"Find the coefficient b₃, correct to {places_text(DP)}."),
                **number_part(b, DP),
                marks=3,
                sensible_range=(0, None),
                working=[f"b₃ = (2/L) ∫₀ᴸ u(x, 0) sin(3πx/L) dx = {fixed(b, DP)}."],
                distractors=b_d,
            ),
            dict(
                **prompt_fields(f"Find u({_x_text(x0)}, {t0}), correct to {places_text(DP)}."),
                **number_part(v, DP),
                marks=3,
                working=[f"Summing the series gives u({_x_text(x0)}, {t0}) = {fixed(v, DP)}."],
                distractors=c_d,
            ),
            dict(
                **prompt_fields(
                    "The third mode, b₃e^(−λ₃t) sin(3πx/L), has amplitude b₃e^(−λ₃t). "
                    "After how many seconds does this amplitude fall to 1 °C? Give your answer correct to 1 decimal place."
                ),
                **number_part(t_fade, 1),
                marks=2,
                sensible_range=(0, None),
                follow_through="log(part_b)/part_a",
                working=[f"b₃e^(−λ₃t) = 1 gives t = ln(b₃)/λ₃ = {fixed(t_fade, 1)}."],
                distractors=d_d,
            ),
        ],
    )
