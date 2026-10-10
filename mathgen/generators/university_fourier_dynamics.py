"""Small, closed-form slices of Fourier analysis, PDEs, dynamics and physics.

This batch is deliberately computational rather than proof-oriented.  Each
generator uses a different verification route from the construction where
practical (exact integration, discriminants, derivative evaluation, or a
second rearrangement of the physical law), so the resulting items are useful
for early university placement without pretending to assess full theory.
"""

from __future__ import annotations

from fractions import Fraction
from math import pi

import sympy

from ..core import check, check_equivalent, generator, latex_answer, pick_distractors


def _expr_text(value) -> str:
    """Use a grader-friendly exact expression (``**`` and ``pi``)."""
    return str(sympy.simplify(value)).replace("**", "^")


@generator(
    "fourier_coefficient_simple",
    level=17,
    presets={
        "easy": dict(n_hi=3),
        "medium": dict(n_hi=6),
        "hard": dict(n_hi=10),
    },
)
def fourier_coefficient_simple(rng, n_hi=6):
    """Compute one exact sine/cosine coefficient on [0, pi]."""
    x = sympy.symbols("x", real=True)
    n = rng.randint(1, n_hi)
    family = rng.choice(("sine", "cosine"))
    power = rng.choice((1, 2))
    f = x**power
    basis = sympy.sin(n * x) if family == "sine" else sympy.cos(n * x)
    coefficient = sympy.simplify(2 / sympy.pi * sympy.integrate(f * basis, (x, 0, sympy.pi)))

    # Independent checks: differentiate the antiderivative and compare a
    # high-precision numerical quadrature with the exact result.
    antiderivative = sympy.integrate(f * basis, x)
    check_equivalent(sympy.diff(antiderivative, x), f * basis, "Fourier integrand antiderivative check failed")
    numeric = sympy.N(2 / sympy.pi * sympy.integrate(f * basis, (x, 0, sympy.pi)), 30)
    check(abs(float(numeric) - float(sympy.N(coefficient, 30))) < 1e-10, "Fourier coefficient numerical check failed")

    answer_text = _expr_text(coefficient)
    wrong_sign = _expr_text(-coefficient)
    missing_factor = _expr_text(sympy.simplify(coefficient / 2))
    wrong_basis = sympy.simplify(
        2
        / sympy.pi
        * sympy.integrate(f * (sympy.cos(n * x) if family == "sine" else sympy.sin(n * x)), (x, 0, sympy.pi))
    )
    candidates = [wrong_sign, missing_factor, _expr_text(wrong_basis), _expr_text(coefficient + 1)]
    return dict(
        text=(
            f"On [0, π], let f(x) = x^{power}. Compute its exact {family} Fourier "
            f"coefficient for mode n = {n}. Use a_n = 2/π ∫₀^π f(x)cos(nx)dx "
            "or b_n = 2/π ∫₀^π f(x)sin(nx)dx as appropriate."
        ),
        latex=(
            rf"f(x)=x^{{{power}}},\quad n={n};\quad "
            rf"\text{{find the exact {family} coefficient on }}[0,\pi]."
        ),
        answer=coefficient,
        answer_text=answer_text,
        answer_latex=sympy.latex(coefficient),
        working=[
            f"Insert f(x)=x^{power} into the {family} coefficient integral.",
            f"Evaluate the exact integral to obtain {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "constant_coefficient_pde_classification",
    level=17,
    presets={
        "easy": dict(coefficient_hi=2),
        "medium": dict(coefficient_hi=4),
        "hard": dict(coefficient_hi=7),
    },
)
def constant_coefficient_pde_classification(rng, coefficient_hi=4, kind=None):
    """Classify A u_xx + B u_xy + C u_yy = 0 from B^2 - 4AC.

    ``kind=None`` (the default) preserves the original behaviour
    byte-for-byte: the equation is always shown as
    ``A u_xx + (B)u_xy + (C)u_yy = 0``. ``kind="rearranged"`` moves the
    ``C u_yy`` term across the ``=`` sign instead, rendering
    ``A u_xx + (B)u_xy = (-C)u_yy`` -- the SAME A, B, C and the SAME
    discriminant classification, but the coefficients must be identified
    from a rearranged equation first, a genuinely different first step.
    ``B`` is forced nonzero for this kind so the rearranged equation always
    shows all three terms.
    """
    target = rng.choice(("Elliptic", "Parabolic", "Hyperbolic"))
    h = max(2, coefficient_hi)
    if target == "Elliptic":
        a = rng.randint(1, h)
        c = rng.randint(1, h)
        b = rng.randint(-max(1, min(h, int((4 * a * c - 1) ** 0.5))), max(1, min(h, int((4 * a * c - 1) ** 0.5))))
        while b * b >= 4 * a * c or (kind == "rearranged" and b == 0):
            b = rng.randint(-h, h)
    elif target == "Parabolic":
        a = rng.randint(1, h)
        c = rng.randint(1, h)
        b = 2 * rng.choice((-1, 1)) * int((a * c) ** 0.5)
        while b * b != 4 * a * c or (kind == "rearranged" and b == 0):
            c = rng.randint(1, h)
            b = 2 * rng.choice((-1, 1)) * int((a * c) ** 0.5)
    else:
        a = rng.randint(1, h)
        c = -rng.randint(1, h)
        b = rng.randint(-h, h)
        if kind == "rearranged":
            while b == 0:
                b = rng.randint(-h, h)
    discriminant = b * b - 4 * a * c
    check(
        (discriminant < 0 and target == "Elliptic")
        or (discriminant == 0 and target == "Parabolic")
        or (discriminant > 0 and target == "Hyperbolic"),
        "PDE discriminant classification mismatch",
    )
    # Recompute from the quadratic-form matrix as an independent check.
    matrix_det = sympy.Matrix([[a, sympy.Rational(b, 2)], [sympy.Rational(b, 2), c]]).det()
    check(
        sympy.sign(matrix_det) == (-1 if discriminant > 0 else 1 if discriminant < 0 else 0),
        "PDE quadratic-form check failed",
    )
    answer_text = target
    others = [label for label in ("Elliptic", "Parabolic", "Hyperbolic") if label != target]
    if kind == "rearranged":
        check(b != 0, "rearranged PDE hard tier drew a zero mixed-term coefficient")
        return dict(
            text=(
                f"Classify the constant-coefficient PDE {a}u_xx "
                f"+ ({b})u_xy = ({-c})u_yy as elliptic, parabolic or hyperbolic."
            ),
            latex=rf"{a}u_{{xx}}+({b})u_{{xy}}=({-c})u_{{yy}};\quad\text{{classify}}.",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, plain=True),
            working=[
                f"Rearrange to standard form: {a}u_xx + ({b})u_xy + ({c})u_yy = 0.",
                f"Compute Δ = B² − 4AC = {discriminant}.",
                f"Since Δ is {'negative' if discriminant < 0 else 'zero' if discriminant == 0 else 'positive'}, the PDE is {target.lower()}.",
            ],
            distractors=pick_distractors(answer_text, others),
        )
    return dict(
        text=(
            f"Classify the constant-coefficient PDE {a}u_xx "
            f"+ ({b})u_xy + ({c})u_yy = 0 as elliptic, parabolic or hyperbolic."
        ),
        latex=rf"{a}u_{{xx}}+({b})u_{{xy}}+({c})u_{{yy}}=0;\quad\text{{classify}}.",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Compute Δ = B² − 4AC = {discriminant}.",
            f"Since Δ is {'negative' if discriminant < 0 else 'zero' if discriminant == 0 else 'positive'}, the PDE is {target.lower()}.",
        ],
        distractors=pick_distractors(answer_text, others),
    )


@generator(
    "autonomous_equilibrium_stability",
    level=16,
    presets={
        "easy": dict(rate_hi=2, equilibrium_hi=3),
        "medium": dict(rate_hi=5, equilibrium_hi=5),
        "hard": dict(rate_hi=9, equilibrium_hi=8),
    },
)
def autonomous_equilibrium_stability(rng, rate_hi=5, equilibrium_hi=5, kind=None):
    """Classify a hyperbolic equilibrium of a 1D autonomous ODE.

    ``kind=None`` (the default) preserves the original linear behaviour
    byte-for-byte: x' = rate(x - equilibrium), where the sign of `rate` IS
    f'(x*) directly. ``kind="logistic"`` uses a NONLINEAR x' = x(b - x)
    instead, which has TWO equilibria (x*=0 and x*=b); one is chosen at
    random and its stability must be found by actually differentiating
    f(x) = bx - x**2 to get f'(x) = b - 2x and substituting -- the student
    cannot read the answer off a given rate directly.
    """
    if kind == "logistic":
        b = rng.choice([value for value in range(-rate_hi, rate_hi + 1) if value])
        equilibrium = rng.choice([0, b])
        # f(x) = x(b - x) = bx - x^2, so f'(x) = b - 2x.
        derivative = b - 2 * equilibrium
        # Independent check: symbolic differentiation via sympy, evaluated
        # at the chosen equilibrium, must agree with the hand-derived
        # formula above.
        x = sympy.Symbol("x")
        f = x * (b - x)
        check(sympy.expand(f) == sympy.expand(b * x - x**2), "logistic expansion mismatch")
        check(int(sympy.diff(f, x).subs(x, equilibrium)) == derivative, "logistic derivative substitution mismatch")
        check(f.subs(x, equilibrium) == 0, "logistic equilibrium check failed")
        answer_text = "Stable (attracting)" if derivative < 0 else "Unstable (repelling)"
        opposite = "Unstable (repelling)" if derivative < 0 else "Stable (attracting)"
        candidates = [opposite, "Non-equilibrium point", "Neutral (zero derivative)"]
        return dict(
            text=(
                f"For the autonomous equation x' = x({b} − x), "
                f"classify the equilibrium x* = {equilibrium} using the sign of f'(x*)."
            ),
            latex=rf"x'=x({b}-x);\quad x_*={equilibrium};\quad\text{{classify stability}}.",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, plain=True),
            working=[
                f"f(x) = x({b} − x) = {b}x − x², so f({equilibrium}) = 0 and x* = {equilibrium} is an equilibrium.",
                f"f'(x) = {b} − 2x, so f'(x*) = {derivative}; a negative derivative attracts and a positive derivative repels.",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )
    rate = rng.choice([value for value in range(-rate_hi, rate_hi + 1) if value])
    equilibrium = rng.randint(-equilibrium_hi, equilibrium_hi)
    # f(x)=rate*(x-r), so f(r)=0 and f'(r)=rate.
    derivative = rate
    check(rate * (equilibrium - equilibrium) == 0, "equilibrium check failed")
    check(derivative == rate, "linearisation derivative check failed")
    answer_text = "Stable (attracting)" if derivative < 0 else "Unstable (repelling)"
    opposite = "Unstable (repelling)" if derivative < 0 else "Stable (attracting)"
    candidates = [opposite, "Non-equilibrium point", "Neutral (zero derivative)"]
    return dict(
        text=(
            f"For the autonomous equation x' = {rate}(x − ({equilibrium})), "
            f"classify the equilibrium x* = {equilibrium} using the sign of f'(x*)."
        ),
        latex=rf"x'={rate}(x-({equilibrium}));\quad x_*={equilibrium};\quad\text{{classify stability}}.",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"f({equilibrium}) = 0, so x* = {equilibrium} is an equilibrium.",
            f"f'(x*) = {derivative}; a negative derivative attracts and a positive derivative repels.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "simple_harmonic_angular_frequency",
    level=16,
    presets={
        "easy": dict(mass_hi=3, omega_hi=3),
        "medium": dict(mass_hi=7, omega_hi=6),
        "hard": dict(mass_hi=12, omega_hi=10),
    },
)
def simple_harmonic_angular_frequency(rng, mass_hi=7, omega_hi=6, kind=None):
    """Compute ω=sqrt(k/m) for a mass-spring oscillator.

    ``kind=None`` (the default) preserves the original behaviour
    byte-for-byte: m and k are given, ω is asked. ``kind="solve_for_k"``
    gives m and ω instead, and asks for the stiffness k -- solving the SAME
    relation for k = mω² (a multiplication, no square root) rather than for
    ω, a genuinely different unknown.
    """
    mass = rng.randint(1, mass_hi)
    omega = rng.randint(1, omega_hi)
    stiffness = mass * omega * omega
    check(stiffness / mass == omega * omega, "spring ratio check failed")
    check(sympy.sqrt(sympy.Rational(stiffness, mass)) == omega, "angular-frequency square-root check failed")
    if kind == "solve_for_k":
        answer_text = str(stiffness)
        candidates = [
            str(mass),
            str(omega),
            str(mass + omega),
            str(mass * omega),
            str(stiffness + 1),
            str(stiffness - 1),
        ]
        return dict(
            text=(
                f"A mass m = {mass} kg is attached to a spring and oscillates with "
                f"angular frequency ω = {omega} rad/s. Find the stiffness k in "
                "ω = √(k/m), in N/m."
            ),
            latex=rf"m={mass},\;\omega={omega};\quad k=m\omega^2.",
            answer=stiffness,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"k = mω² = {mass}×{omega}² = {stiffness}.", f"Therefore k = {stiffness} N/m."],
            distractors=pick_distractors(answer_text, candidates),
        )
    answer_text = str(omega)
    candidates = [
        str(stiffness),
        str(mass),
        str(stiffness // mass),
        str(omega * omega),
        str(omega + 1),
        str(2 * omega + 3),
    ]
    return dict(
        text=(
            f"A mass m = {mass} kg is attached to a spring with stiffness "
            f"k = {stiffness} N/m. Find the angular frequency ω in ω = √(k/m), "
            "in rad/s."
        ),
        latex=rf"m={mass},\;k={stiffness};\quad\omega=\sqrt{{k/m}}.",
        answer=omega,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"k/m = {stiffness}/{mass} = {omega * omega}.", f"Therefore ω = √({omega * omega}) = {omega} rad/s."],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "wave_frequency_from_speed_wavelength",
    level=16,
    presets={
        "easy": dict(wavelength_hi=5, frequency_hi=4),
        "medium": dict(wavelength_hi=9, frequency_hi=8),
        "hard": dict(wavelength_hi=15, frequency_hi=12),
    },
)
def wave_frequency_from_speed_wavelength(rng, wavelength_hi=9, frequency_hi=8, kind=None):
    """Compute f=v/λ from integer speed and wavelength data.

    ``kind=None`` (the default) preserves the original behaviour
    byte-for-byte: v and λ are given, f is asked. ``kind="solve_for_wavelength"``
    gives v and f instead, and asks for λ -- solving the SAME relation
    v = fλ for λ = v/f rather than f = v/λ, a genuinely different unknown.
    """
    wavelength = rng.randint(1, wavelength_hi)
    frequency = rng.randint(1, frequency_hi)
    speed = wavelength * frequency
    check(speed == frequency * wavelength, "wave relation construction failed")
    check(speed / wavelength == frequency, "wave relation division check failed")
    if kind == "solve_for_wavelength":
        answer_text = str(wavelength)
        candidates = [
            str(speed),
            str(frequency),
            str(speed * frequency),
            str(wavelength + 1),
            str(max(1, wavelength - 1)),
            str(wavelength + 2),
        ]
        return dict(
            text=(
                f"A travelling wave has speed v = {speed} m/s and frequency "
                f"f = {frequency} Hz. Find its wavelength λ using v = fλ."
            ),
            latex=rf"v={speed}\,\mathrm{{m/s}},\;f={frequency}\,\mathrm{{Hz}};\quad v=f\lambda.",
            answer=wavelength,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Rearrange v = fλ to λ = v/f = {speed}/{frequency}.", f"Thus λ = {wavelength} m."],
            distractors=pick_distractors(answer_text, candidates),
        )
    answer_text = str(frequency)
    candidates = [
        str(speed),
        str(wavelength),
        str(speed * wavelength),
        str(frequency + 1),
        str(max(1, frequency - 1)),
        str(frequency + 2),
    ]
    return dict(
        text=(
            f"A travelling wave has speed v = {speed} m/s and wavelength "
            f"λ = {wavelength} m. Find its frequency f using v = fλ."
        ),
        latex=rf"v={speed}\,\mathrm{{m/s}},\;\lambda={wavelength}\,\mathrm{{m}};\quad v=f\lambda.",
        answer=frequency,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Rearrange v = fλ to f = v/λ = {speed}/{wavelength}.", f"Thus f = {frequency} Hz."],
        distractors=pick_distractors(answer_text, candidates),
    )
