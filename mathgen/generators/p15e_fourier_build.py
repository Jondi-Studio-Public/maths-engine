"""A Gaussian's Fourier transform, and a separated wave-equation mode.

- ``p15e_fourier_transform_gaussian`` (``ucv1-fourier-transform-gaussian``,
  ``answer: expression``) -- return the Fourier transform of a supplied
  real Gaussian A*exp(-a*x^2), a > 0, under a transform convention STATED
  IN EVERY PROMPT (never assumed). Every convention here keeps the answer
  REAL (the Gaussian is centred at x = 0 and A, a are real, so the
  own recommended dodge of the open expression-grader defect: the graded
key never needs a literal `I`.
  Prompts show `exp(...)`, never `e^`, so the
  grader-hazard row for `e^(-w^2/4)` never applies here at all. The
  frequency-convention tier names the transform variable `xi`, never `f`,
  so it never doubles as the function's own name `f(x)` in the same
  sentence. Construction applies the closed-form transform directly, with
  NO symbolic verification inside `make()` -- the hot path, which
  per-topic test sweeps call on the order of a thousand times per generator. The
  independent route -- `mpmath` numeric quadrature of the DEFINING integral,
  a genuinely different (and much slower) route than the closed-form
  substitution -- lives entirely in
  ``tests/test_p15e_fourier_oracle.py``, run once per suite (60 seeds/tier).
- ``p15e_wave_equation_separation`` (``ucv1-wave-equation-separation``,
  ``answer: expression``) -- one product-form separated mode of the 1-D
  wave equation u_tt = c^2 u_xx under a stated Dirichlet, Neumann or (exam
  tier only) MIXED boundary family, with the amplitude, mode number n, and
  the sin/cos choice IN TIME pinned by a stated initial condition (the expression grader accepts equivalent forms, so a
  scalar multiple of the key is WRONG -- every prompt fixes the scale). The exam
  tier's mixed family (fixed at x=0, free at x=L:
  sin((2n-1)pi x / (2L))) has no two-candidate shortcut: unlike the earlier
  design, nothing in the prompt already contains the answer string, so
  reading it off is not an option -- the learner must construct the
  eigenfunction for a family the easier tiers never used. Construction is
  pure closed-form formula substitution -- no symbolic verification runs
  inside `make()`. That the mode SOLVES the wave PDE and satisfies
  its boundary condition is a fact about the closed FORM itself (any
  sin/cos eigenfunction of wavenumber k, paired with a time factor of the
  same k scaled by c, solves u_tt=c^2 u_xx; the three boundary families'
  own eigenfunctions vanish, or have vanishing derivative, at the stated
  ends by a trig identity, for every n) -- proved once per tier/seed in
  ``tests/test_p15e_fourier_oracle.py`` via ``sympy.diff``, not on every
  `make()` call.

Brand new module; no existing generator is edited.
"""

from __future__ import annotations

import sympy
from sympy import cos, exp, pi, sin, sqrt, symbols

from ..core import (
    check,
    fmt_signed,
    generator,
    phrase,
    pick_distractors,
)

_MAX_ATTEMPTS = 500
_X, _W = symbols("x w", real=True)
_T = symbols("t", real=True)
_XI = symbols("xi", real=True)


def _n_pi(n: int) -> str:
    """``pi`` for n = 1, ``3pi`` for n = 3 (never ``1pi``)."""
    return "pi" if n == 1 else f"{n}pi"


def _fmt(expr) -> str:
    """Display only -- every caller already passes a fully `sympy.expand`-ed
    value, so this stringifies rather than re-simplifies it (see
    `p15e_complex_build._fmt`'s docstring for the measured cost of
    `simplify`/`trigsimp` on a value that never needed it)."""
    return str(sympy.expand(expr)).replace("**", "^")


def _text_fmt(expr) -> str:
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "·")


def _eq(a, b) -> bool:
    """Exact equality for this module's values (rationals, `sqrt`, `exp`,
    `pi` -- never an unevaluated fraction needing `cancel`): see
    `p15e_complex_build._eq`'s docstring."""
    return sympy.expand(a - b) == 0


# ---------------------------------------------------------------------------
# Skill 1: ucv1-fourier-transform-gaussian
# ---------------------------------------------------------------------------

_A_POOL = tuple(v for v in range(1, 10) if v != 0)
_AMP_POOL = tuple(v for v in range(-9, 10) if v != 0)
_A_HARD_POOL = tuple(range(1, 5))  # kept small: sqrt(1/(2a)) below stays legible

_CONV_ORDINARY = ("ordinary", "F(w) = the integral over all real x of f(x) exp(-i*w*x) dx")
_CONV_UNITARY = ("unitary", "F(w) = 1/sqrt(2·pi) times the integral over all real x of f(x) exp(-i*w*x) dx")
_CONV_FREQ = (
    "frequency",
    "F(xi) = the integral over all real x of f(x) exp(-2·pi*i*xi*x) dx (xi the "
    "ordinary, not angular, frequency variable)",
)


def _ordinary_transform(A, a):
    return sympy.expand(A * sqrt(pi / a) * exp(-(_W**2) / (4 * a)))


def _unitary_transform(A, a):
    return sympy.expand(A * sqrt(sympy.Rational(1, 2) / a) * exp(-(_W**2) / (4 * a)))


def _frequency_transform(A, a):
    # Substituting w = 2*pi*xi into the ordinary-convention result. Named `xi`, never `f`: the
    # function being transformed is ALSO called `f` (f(x) = A*exp(-a*x^2)) in every prompt, so a
    # frequency variable also named `f` would be the same letter for two different things in one
    # sentence. The graded `answer_text` and every `grade` spelling probe use `xi` too.
    return sympy.expand(_ordinary_transform(A, a).subs(_W, 2 * pi * _XI))


def _offer(answer, value, offered, formatted, label):
    if _eq(value, answer) or any(_eq(value, v) for v in offered):
        return
    offered.append(value)
    formatted.append(_fmt(value))


_EASY_PATTERNS = [
    "Let f(x) = {A}exp(-x^2). Using the ordinary Fourier transform convention, {conv_desc}, find F(w).",
    "For f(x) = {A}exp(-x^2), compute its Fourier transform F(w) under the ordinary convention: {conv_desc}.",
    "Given f(x) = {A}exp(-x^2), find F(w) using the ordinary transform convention {conv_desc}.",
]

_MEDIUM_PATTERNS = [
    "Let f(x) = {A}exp(-{a_disp}). Using the ordinary Fourier transform convention, {conv_desc}, find F(w).",
    "For f(x) = {A}exp(-{a_disp}), compute its Fourier transform F(w) under the ordinary convention: {conv_desc}.",
    "Given f(x) = {A}exp(-{a_disp}), find F(w) using the ordinary transform convention {conv_desc}.",
]

_HARD_PATTERNS = [
    "Let f(x) = {A}exp(-{a_disp}). Using the UNITARY Fourier transform convention, {conv_desc}, find F(w).",
    "For f(x) = {A}exp(-{a_disp}), compute its Fourier transform F(w) under the unitary convention: {conv_desc}.",
    "Given f(x) = {A}exp(-{a_disp}), find F(w) using the unitary transform convention {conv_desc}.",
]

_EXAM_PATTERNS = [
    "Let f(x) = {A}exp(-{a_disp}). Using the ORDINARY-FREQUENCY Fourier transform convention, {conv_desc}, find F(xi).",
    "For f(x) = {A}exp(-{a_disp}), compute its Fourier transform F(xi) under the frequency convention: {conv_desc}.",
    "Given f(x) = {A}exp(-{a_disp}), find F(xi) using the frequency-domain transform convention {conv_desc}.",
]


def _amp_text(A) -> str:
    """ "" (A=1), "-" (A=-1), "5 " (A=5, trailing space so it never runs
    into "exp": "-5exp(-x^2)" needs a space), "-5 " (A=-5)."""
    if A == 1:
        return ""
    if A == -1:
        return "-"
    return f"{A} "


def _p15e_ft_easy(rng):
    for _ in range(_MAX_ATTEMPTS):
        A = rng.choice(_AMP_POOL)
        a = 1
        answer = _ordinary_transform(A, a)
        check(sympy.im(answer) == 0, "easy: transform is not real")

        offered, formatted = [], []
        _offer(answer, A * exp(-(_W**2) / 4), offered, formatted, "forgets-the-sqrt-factor")
        _offer(answer, A * sqrt(pi) * exp(-(_W**2) / 2), offered, formatted, "wrong-exponent-denominator")
        _offer(answer, A * sqrt(pi) * exp(-(_W**2)), offered, formatted, "keeps-original-a-in-exponent")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(rng, _EASY_PATTERNS, A=_amp_text(A), conv_desc=_CONV_ORDINARY[1])
        working = [
            f"F(w) = {A}sqrt(pi/1)exp(-w^2/(4×1)) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p15e_ft_medium(rng):
    for _ in range(_MAX_ATTEMPTS):
        A = rng.choice(_AMP_POOL)
        a = rng.choice(_A_POOL)
        answer = _ordinary_transform(A, a)
        check(sympy.im(answer) == 0, "medium: transform is not real")

        offered, formatted = [], []
        _offer(answer, A * exp(-(_W**2) / (4 * a)), offered, formatted, "forgets-the-sqrt-factor")
        _offer(answer, A * sqrt(pi / a) * exp(-(_W**2) / (2 * a)), offered, formatted, "wrong-exponent-denominator")
        _offer(answer, A * sqrt(pi / a) * exp(-a * _W**2), offered, formatted, "keeps-original-a-in-exponent")
        _offer(answer, A * sqrt(pi * a) * exp(-(_W**2) / (4 * a)), offered, formatted, "inverts-a-in-prefactor")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        a_disp = "x^2" if a == 1 else f"{a}x^2"
        text = phrase(rng, _MEDIUM_PATTERNS, A=_amp_text(A), a_disp=a_disp, conv_desc=_CONV_ORDINARY[1])
        working = [
            f"F(w) = {A}sqrt(pi/{a})exp(-w^2/(4×{a})) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p15e_ft_hard(rng):
    for _ in range(_MAX_ATTEMPTS):
        A = rng.choice(_AMP_POOL)
        a = rng.choice(_A_HARD_POOL)
        answer = _unitary_transform(A, a)
        check(sympy.im(answer) == 0, "hard: transform is not real")
        ordinary = _ordinary_transform(A, a)
        check(_eq(answer, ordinary / sqrt(2 * pi)), "hard: unitary transform does not match ordinary/sqrt(2*pi)")

        offered, formatted = [], []
        _offer(answer, ordinary, offered, formatted, "wrong-convention-constant")
        _offer(answer, A * exp(-(_W**2) / (4 * a)), offered, formatted, "forgets-the-sqrt-factor")
        _offer(
            answer,
            A * sqrt(sympy.Rational(1, 2) / a) * exp(-(_W**2) / (2 * a)),
            offered,
            formatted,
            "wrong-exponent-denominator",
        )

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        a_disp = "x^2" if a == 1 else f"{a}x^2"
        text = phrase(rng, _HARD_PATTERNS, A=_amp_text(A), a_disp=a_disp, conv_desc=_CONV_UNITARY[1])
        working = [
            f"Ordinary-convention transform: {_text_fmt(ordinary)}.",
            f"Divide by √(2π) for the unitary convention: {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p15e_ft_exam(rng):
    for _ in range(_MAX_ATTEMPTS):
        A = rng.choice(_AMP_POOL)
        a = rng.choice(_A_HARD_POOL)
        answer = _frequency_transform(A, a)
        check(sympy.im(answer) == 0, "exam: transform is not real")
        ordinary = _ordinary_transform(A, a)
        check(_eq(answer, ordinary.subs(_W, 2 * pi * _XI)), "exam: frequency-convention substitution mismatch")

        offered, formatted = [], []
        _offer(answer, ordinary.subs(_W, _XI), offered, formatted, "forgets-the-2-pi-substitution")
        _offer(answer, ordinary.subs(_W, pi * _XI), offered, formatted, "halves-the-2-pi-substitution")
        _offer(answer, A * exp(-(pi**2) * _XI**2 / a), offered, formatted, "forgets-the-sqrt-factor")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        a_disp = "x^2" if a == 1 else f"{a}x^2"
        text = phrase(rng, _EXAM_PATTERNS, A=_amp_text(A), a_disp=a_disp, conv_desc=_CONV_FREQ[1])
        working = [
            f"Ordinary-convention (angular) transform: {_text_fmt(ordinary)}.",
            f"Substitute w = 2πxi: F(xi) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p15e_fourier_transform_gaussian",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15e_fourier_transform_gaussian(rng, tier="medium"):
    if tier == "easy":
        return _p15e_ft_easy(rng)
    if tier == "medium":
        return _p15e_ft_medium(rng)
    if tier == "hard":
        return _p15e_ft_hard(rng)
    if tier == "exam":
        return _p15e_ft_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Skill 2: ucv1-wave-equation-separation
# ---------------------------------------------------------------------------

_C_POOL = tuple(range(1, 6))
_N_POOL = tuple(range(1, 6))
_L_POOL = tuple(range(2, 10))
_COEF_POOL = tuple(v for v in range(-9, 10) if v != 0)

_START_FROM_REST = ("cos", "the string starts from rest (initial velocity zero)")
_START_FROM_ZERO_DISPLACEMENT = ("sin", "the string starts at zero displacement with nonzero initial velocity")


def _wave_spatial(family, n, L):
    """The family's own spatial eigenfunction. "mixed" (exam tier only:
    fixed at x=0, free at x=L) uses the HALF-integer wavenumber
    (2n-1)pi/(2L) -- a genuinely different eigenfunction family from
    Dirichlet's sin(n pi x/L) or Neumann's cos(n pi x/L), not obtainable by
    copying either easier tier's formula with different numbers."""
    if family == "dirichlet":
        return sin(n * pi * _X / L)
    if family == "neumann":
        return cos(n * pi * _X / L)
    return sin((2 * n - 1) * pi * _X / (2 * L))  # mixed


def _wave_number(family, n, L):
    """The spatial wavenumber k such that the family's eigenfunction is
    sin(kx) or cos(kx) -- shared with the time factor's argument k*c*t, the
    closed-form fact that makes u_tt = c^2 u_xx hold for ANY of these three
    families without a per-draw PDE check (see the module docstring)."""
    return (2 * n - 1) * pi / (2 * L) if family == "mixed" else n * pi / L


def _wave_mode(coeff, family, time_kind, n, c, L, *, drop_c=False, wrong_time=False):
    time_arg_c = 1 if drop_c else c
    tk = ("sin" if time_kind == "cos" else "cos") if wrong_time else time_kind
    L_sym = L if isinstance(L, sympy.Basic) else sympy.Integer(L)
    spatial = _wave_spatial(family, n, L_sym)
    k = _wave_number(family, n, L_sym)
    time_factor = cos(k * time_arg_c * _T) if tk == "cos" else sin(k * time_arg_c * _T)
    return coeff * spatial * time_factor


def _offer_expr(answer, value, offered, formatted, label):
    if _eq(value, answer) or any(_eq(value, v) for v in offered):
        return
    offered.append(value)
    formatted.append(_fmt(value))


_EASY_WAVE_PATTERNS = [
    "A string on [0, pi] satisfies the wave equation u_tt = c^2 u_xx with "
    "fixed ends u(0,t) = u(pi,t) = 0 (Dirichlet), c = {c}, and {ic_words}. "
    "One separated mode has amplitude B = 1 and mode number n = {n}. Write "
    "u_n(x,t).",
    "On [0, pi] with fixed ends u(0,t) = u(pi,t) = 0, the wave equation "
    "u_tt = c^2 u_xx holds with c = {c}, and {ic_words}. Write the "
    "separated mode u_n(x,t) for n = {n}, amplitude B = 1.",
    "A string occupies [0, pi] with u(0,t) = u(pi,t) = 0, u_tt = c^2 u_xx, "
    "c = {c}, and {ic_words}. Find the mode u_n(x,t) for mode number "
    "n = {n}, amplitude B = 1.",
]

_MEDIUM_WAVE_PATTERNS = [
    "A string on [0, {L}] satisfies the wave equation u_tt = c^2 u_xx with "
    "fixed ends u(0,t) = u({L},t) = 0 (Dirichlet), c = {c}, and {ic_words}. "
    "One separated mode has amplitude B = {coeff} and mode number n = {n}. "
    "Write u_n(x,t).",
    "On [0, {L}] with fixed ends u(0,t) = u({L},t) = 0, the wave equation "
    "u_tt = c^2 u_xx holds with c = {c}, and {ic_words}. Write the "
    "separated mode u_n(x,t) for n = {n}, amplitude B = {coeff}.",
    "A string occupies [0, {L}] with u(0,t) = u({L},t) = 0, u_tt = c^2 "
    "u_xx, c = {c}, and {ic_words}. Find the mode u_n(x,t) for mode number "
    "n = {n}, amplitude B = {coeff}.",
]

_HARD_WAVE_PATTERNS = [
    "A string on [0, {L}] satisfies u_tt = c^2 u_xx. The ends are free: "
    "u_x(0,t) = u_x({L},t) = 0 (Neumann), c = {c}, and {ic_words}. One "
    "separated mode has amplitude A = {coeff} and mode number n = {n}. "
    "Write u_n(x,t).",
    "A string [0, {L}] has free ends u_x(0,t) = u_x({L},t) = 0 and obeys "
    "u_tt = c^2 u_xx, c = {c}, and {ic_words}. Write the separated mode "
    "u_n(x,t) for n = {n}, amplitude A = {coeff}.",
    "On [0, {L}], u(x,t) satisfies u_tt = c^2 u_xx with free ends "
    "(u_x(0,t) = u_x({L},t) = 0), c = {c}, and {ic_words}. Find the mode "
    "u_n(x,t) for mode number n = {n}, amplitude A = {coeff}.",
]

_MIXED_INDEXING = "n = 1 is the lowest mode, n = 2 the next, and so on"

_EXAM_WAVE_PATTERNS = [
    "On [0, {L}], u(x,t) satisfies u_tt = c^2 u_xx with c = {c}. The "
    "boundary condition is MIXED: fixed at x = 0 (u(0,t) = 0) and free at "
    "x = {L} (u_x({L},t) = 0), and {ic_words}. Modes are indexed so that "
    "{indexing}. Construct the separated mode u_n(x,t) for mode number "
    "n = {n} and amplitude C = {coeff}; derive the spatial eigenfunction "
    "from the boundary conditions themselves.",
    "A string on [0, {L}] obeys u_tt = c^2 u_xx (c = {c}) and has the "
    "MIXED boundary condition: fixed at x = 0 (u(0,t) = 0) but free at "
    "x = {L} (u_x({L},t) = 0). Given that {ic_words}, and that {indexing}, "
    "find the mode u_n(x,t) for mode number n = {n} and amplitude "
    "C = {coeff}.",
    "The string u(x,t) on [0, {L}] satisfies u_tt = c^2 u_xx, c = {c}, "
    "with the MIXED boundary condition u(0,t) = 0 (fixed) and "
    "u_x({L},t) = 0 (free), and {ic_words}. Modes are indexed so that "
    "{indexing}. For mode number n = {n} and amplitude C = {coeff}, "
    "derive the eigenfunction that satisfies both boundary conditions and "
    "give the full separated mode u_n(x,t).",
]

_BC_WORDS = {"dirichlet": "fixed ends u(0,t) = u({L},t) = 0", "neumann": "free ends u_x(0,t) = u_x({L},t) = 0"}


def _p15e_wave_easy(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        n = rng.choice(_N_POOL)
        L = sympy.pi
        family = "dirichlet"
        time_kind, ic_words = _START_FROM_REST
        answer = _wave_mode(1, family, time_kind, n, c, L)

        offered, formatted = [], []
        _offer_expr(answer, _wave_mode(1, "neumann", time_kind, n, c, L), offered, formatted, "wrong-family")
        _offer_expr(
            answer, _wave_mode(1, family, time_kind, n, c, L, wrong_time=True), offered, formatted, "wrong-time-factor"
        )
        _offer_expr(answer, _wave_mode(1, family, time_kind, n, c, L, drop_c=True), offered, formatted, "drops-c")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(rng, _EASY_WAVE_PATTERNS, c=c, n=n, ic_words=ic_words)
        working = [f"u_n(x,t) = sin({fmt_signed(n, 'x')}) cos({n} × {c} × t) = {_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p15e_wave_medium(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        family = "dirichlet"
        time_kind, ic_words = _START_FROM_REST
        answer = _wave_mode(coeff, family, time_kind, n, c, L)

        offered, formatted = [], []
        _offer_expr(answer, _wave_mode(coeff, "neumann", time_kind, n, c, L), offered, formatted, "wrong-family")
        _offer_expr(
            answer,
            _wave_mode(coeff, family, time_kind, n, c, L, wrong_time=True),
            offered,
            formatted,
            "wrong-time-factor",
        )
        _offer_expr(answer, _wave_mode(coeff, family, time_kind, n, c, L, drop_c=True), offered, formatted, "drops-c")
        _offer_expr(answer, _wave_mode(1, family, time_kind, n, c, L), offered, formatted, "wrong-amplitude")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(rng, _MEDIUM_WAVE_PATTERNS, c=c, n=n, L=L, coeff=coeff, ic_words=ic_words)
        working = [f"u_n(x,t) = {coeff} × sin({_n_pi(n)} x/{L}) cos({_n_pi(n)} × {c} × t/{L}) = {_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p15e_wave_hard(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        family = "neumann"
        time_kind, ic_words = _START_FROM_ZERO_DISPLACEMENT
        answer = _wave_mode(coeff, family, time_kind, n, c, L)

        offered, formatted = [], []
        _offer_expr(answer, _wave_mode(coeff, "dirichlet", time_kind, n, c, L), offered, formatted, "wrong-family")
        _offer_expr(
            answer,
            _wave_mode(coeff, family, time_kind, n, c, L, wrong_time=True),
            offered,
            formatted,
            "wrong-time-factor",
        )
        _offer_expr(answer, _wave_mode(coeff, family, time_kind, n, c, L, drop_c=True), offered, formatted, "drops-c")
        _offer_expr(answer, _wave_mode(1, family, time_kind, n, c, L), offered, formatted, "wrong-amplitude")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(rng, _HARD_WAVE_PATTERNS, c=c, n=n, L=L, coeff=coeff, ic_words=ic_words)
        working = [f"u_n(x,t) = {coeff} × cos({_n_pi(n)} x/{L}) sin({_n_pi(n)} × {c} × t/{L}) = {_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p15e_wave_exam(rng):
    """MIXED boundary family (fixed at x=0, free at x=L): the spatial eigenfunction sin((2n-1)pi x/(2L)) is a
    genuinely new construction, not one of two strings the prompt already contains. Distractors include
    `wrong-wavenumber-parity`: the plausible mistake of using the INTEGER wavenumber n·pi/L (as if the
    boundary were Dirichlet or Neumann) instead of the mixed family's half-integer one."""
    family = "mixed"
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        time_kind, ic_words = rng.choice([_START_FROM_REST, _START_FROM_ZERO_DISPLACEMENT])

        answer = _wave_mode(coeff, family, time_kind, n, c, L)

        offered, formatted = [], []
        _offer_expr(
            answer,
            _wave_mode(coeff, "dirichlet", time_kind, n, c, L),
            offered,
            formatted,
            "wrong-wavenumber-parity-dirichlet",
        )
        _offer_expr(
            answer,
            _wave_mode(coeff, "neumann", time_kind, n, c, L),
            offered,
            formatted,
            "wrong-wavenumber-parity-neumann",
        )
        _offer_expr(
            answer,
            _wave_mode(coeff, family, time_kind, n, c, L, wrong_time=True),
            offered,
            formatted,
            "wrong-time-factor",
        )
        _offer_expr(answer, _wave_mode(coeff, family, time_kind, n, c, L, drop_c=True), offered, formatted, "drops-c")
        _offer_expr(answer, _wave_mode(1, family, time_kind, n, c, L), offered, formatted, "wrong-amplitude")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(rng, _EXAM_WAVE_PATTERNS, c=c, n=n, L=L, coeff=coeff, ic_words=ic_words, indexing=_MIXED_INDEXING)
        working = [
            "Mixed boundary: u(0,t) = 0 rules out cos(kx); u_x(L,t) = 0 needs "
            f"cos(kL) = 0, so kL = (2n-1)pi/2, giving k = (2n-1)pi/(2·{L}).",
            f"u_n(x,t) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p15e_wave_equation_separation",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15e_wave_equation_separation(rng, tier="easy"):
    if tier == "easy":
        return _p15e_wave_easy(rng)
    if tier == "medium":
        return _p15e_wave_medium(rng)
    if tier == "hard":
        return _p15e_wave_hard(rng)
    if tier == "exam":
        return _p15e_wave_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
