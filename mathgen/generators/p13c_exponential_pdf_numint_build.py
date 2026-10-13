"""Generates `p13c_exponential_pdf_numint` questions (ACMSM127): numerically integrate an exponential density
with technology.

**THIS SKILL IS NARROWED.** The descriptor bundles two clauses: (1) a bounded
numerical integration of the exponential pdf ``f(t) = lambda*e^(-lambda*t)``
for ``t >= 0``, and (2) an open "use the exponential random variables and
associated probabilities and quantiles to model data and solve practical
problems" clause. The skill drops
clause (2) entirely -- no data-to-model decomposition, no scenario, no
authentic dataset -- and keeps only clause (1): given a stated ``lambda > 0``
and stated finite bounds, compute the value of the definite integral of the
pdf between those bounds to a stated precision. Every tier below, INCLUDING
the exam tier, is a bare numeric definite integral between STATED bounds --
never a word problem, a modelled scenario, or a supplied dataset. The
`modelling`/`authentic_data` capability gates were dropped for exactly the
clause this generator does not implement; only `technology_use` is cited in
the skill's contract, because "numerical integration using technology" here
means the learner evaluates the closed-form result on a calculator to the
stated precision, not that the generator itself runs a quadrature scheme.

Kernel: ``integral[a, b] lambda*e^(-lambda*t) dt = e^(-lambda*a) -
e^(-lambda*b)``. Construction computes this via ``math.exp`` (fast, double
precision -- ample headroom against the coarsest stated precision used here,
4 d.p.); the independent in-generator check (``_sympy_independent_route``)
recomputes the SAME integral from scratch via ``sympy.integrate`` acting
symbolically on ``sympy.Rational``/``sympy.exp`` and high-precision
``evalf``, an entirely different code path from the ``math.exp`` construction
route.

**Difficulty ladder** (every tier ``answer: number``, graded to its own
stated precision via ``stated_precision_tolerance``):
- easy: bounds ``[0, b]``, single lambda, 2 d.p. -> ``1 - e^(-lambda*b)``.
- medium: bounds ``[a, b]`` both nonzero (``0 < a < b``) -> two evaluations
  combined by subtraction, not one; 2 d.p.
- hard: same shape as medium, 4 d.p., AND lambda/bounds are drawn so
  ``e^(-lambda*a)`` and ``e^(-lambda*b)`` are close in magnitude (the ratio
  ``e^(lambda*(b-a))`` is kept under a tight cap) -- this is a
  numerical-stability trap, not a bigger number: subtracting two close
  values means a learner who truncates either intermediate term too early
  loses the significant digits the stated 4 d.p. answer needs, so the
  escalation is precision MANAGEMENT under near-cancellation, not merely
  "more decimal places for their own sake".
- exam: additivity over a stated union of two disjoint intervals
  ``[a1, b1]`` together with ``[a2, b2]`` (``b1 <= a2``); the answer is the
  SUM of the two sub-integrals. Still the same scope -- two pairs of
  stated bounds instead of one -- and a genuine escalation in kind
  (recognising a union's integral is a sum of its parts, two evaluations
  chained into one answer), with no scenario, no dataset, no word problem.

**Misconceptions** (checked distinct at the STATED-PRECISION grid, every
seed, unconditionally -- see ``_drop_if_collides``):
- swaps the sign (``e^(-lambda*b) - e^(-lambda*a)``, negative of the true
  answer);
- forgets the lower-bound term entirely (computes ``1 - e^(-lambda*b)`` when
  ``a != 0`` is stated) -- medium/hard/exam only, since easy already has
  ``a = 0`` and this "mistake" coincides with the correct formula there;
- uses ``lambda*t`` instead of ``-lambda*t`` in the exponent (a sign error
  inside the exponential itself: ``e^(lambda*b) - e^(lambda*a)``);
- (exam) sums only one of the two sub-integrals, forgetting the union's
  second interval -- offered as two separate candidates (first interval
  only, second interval only) so both are exercisable.

Every distractor's raw value is rounded to the SAME stated precision as the
answer and dropped if it collides with the answer's own rounded value
(``_drop_if_collides``) -- since a distractor formatted at the answer's own
decimal places necessarily differs by at least one whole unit in the last
place whenever it survives that check, it clears
``stated_precision_tolerance(dp)`` (half a unit in that place) BY
CONSTRUCTION, at both 2 d.p. and 4 d.p.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..core import (
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)

_t = sp.symbols("t")

_MAX_ATTEMPTS = 500


# --- shared arithmetic ------------------------------------------------------


def _to_fraction(x: Fraction) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator)


def _raw_integral(lam: Fraction, a: Fraction, b: Fraction) -> Fraction:
    """``e^(-lambda*a) - e^(-lambda*b)`` via ``math.exp`` (construction
    route). ``Fraction(float)`` captures the double's exact binary value, so
    every later comparison (rounding, distractor collision) is exact
    ``Fraction`` arithmetic from this point on -- only the initial transcendental
    evaluation itself is a float."""
    return Fraction(math.exp(-float(lam) * float(a))) - Fraction(math.exp(-float(lam) * float(b)))


def _sympy_independent_route(lam: Fraction, a: Fraction, b: Fraction) -> Fraction:
    """Independent check: symbolic ``sympy.integrate`` of the pdf between
    ``a`` and ``b``, evaluated to 40 significant figures -- a completely
    different code path from ``_raw_integral``'s ``math.exp`` construction
    route (exact ``sympy.Rational`` inputs, symbolic antiderivative, high-
    precision ``evalf``, not a double-precision closed-form evaluation)."""
    lam_r, a_r, b_r = _to_fraction(lam), _to_fraction(a), _to_fraction(b)
    integral = sp.integrate(lam_r * sp.exp(-lam_r * _t), (_t, a_r, b_r))
    return Fraction(str(integral.evalf(40)))


def _verify_independent(raw: Fraction, lam: Fraction, a: Fraction, b: Fraction, msg: str) -> None:
    independent = _sympy_independent_route(lam, a, b)
    check(abs(independent - raw) < Fraction(1, 10**9), msg)


def _drop_if_collides(raw: Fraction, dp: int, answer_rounded: Fraction) -> str | None:
    """Round a candidate distractor's raw value to the answer's own stated precision and return its display
    string -- or ``None`` if it collides with the answer at that precision, or is not a probability.

    The answer is a probability, so a candidate outside [0, 1] (a negated value, an exponent with the wrong
    sign) is dropped: a learner eliminates an impossible probability at sight."""
    rounded = round_half_up(raw, dp)
    if not (0 <= rounded <= 1):
        return None
    if rounded == answer_rounded:
        return None
    return stated_precision_string(rounded, dp)


def _fmt_param(value: Fraction) -> str:
    """Shortest exact decimal display for a lambda/bound value ("0.3", not
    "0.30")."""
    for places in range(0, 4):
        scaled = value * (10**places)
        if scaled.denominator == 1:
            sign = "-" if scaled.numerator < 0 else ""
            digits = str(abs(scaled.numerator)).zfill(places + 1)
            if places == 0:
                return f"{sign}{digits}"
            whole, frac = digits[:-places], digits[-places:]
            return f"{sign}{whole}.{frac}"
    return str(float(value))


# --- prompt templates --------------------------------------------------------

_FACT_TEMPLATES = (
    "The exponential random variable has probability density function "
    "f(t) = {lam}e^(-{lam}t) for t >= 0. Using technology, evaluate the "
    "integral of f(t) from t = {a} to t = {b}, correct to {dp} decimal places.",
    "Let f(t) = {lam}e^(-{lam}t) for t >= 0 be the probability density "
    "function of an exponential random variable with parameter lambda = {lam}. "
    "Use numerical integration by technology to find the value of the "
    "integral of f(t) between t = {a} and t = {b}, correct to {dp} decimal places.",
    "For the exponential probability density function f(t) = {lam}e^(-{lam}t), "
    "t >= 0, compute the definite integral of f(t) over the interval from "
    "t = {a} to t = {b}. Give your answer correct to {dp} decimal places.",
)

_EXAM_TEMPLATES = (
    "The exponential random variable has probability density function "
    "f(t) = {lam}e^(-{lam}t) for t >= 0. Using technology, evaluate the "
    "integral of f(t) over the union of the intervals from t = {a1} to "
    "t = {b1} and from t = {a2} to t = {b2}, correct to {dp} decimal places.",
    "Let f(t) = {lam}e^(-{lam}t) for t >= 0 be the probability density "
    "function of an exponential random variable with parameter lambda = {lam}. "
    "Use numerical integration by technology to find the value of the "
    "integral of f(t) over [{a1}, {b1}] together with [{a2}, {b2}], correct "
    "to {dp} decimal places.",
    "For the exponential probability density function f(t) = {lam}e^(-{lam}t), "
    "t >= 0, use technology to compute the sum of the definite integral of "
    "f(t) from t = {a1} to t = {b1} and the definite integral of f(t) from "
    "t = {a2} to t = {b2}. Give your answer correct to {dp} decimal places.",
)


def _finish(text: str, answer_rounded: Fraction, dp: int, distractor_candidates: list, n: int = 3) -> dict:
    answer_text = stated_precision_string(answer_rounded, dp)
    survivors = [d for d in distractor_candidates if d is not None]
    # Top-up, used only when the misconceptions above fall short: the complement of the answer (the
    # opposite tail) and values a few units in the last place away, all inside [0, 1].
    unit = Fraction(1, 10**dp)
    for extra in (1 - answer_rounded, *(answer_rounded + j * unit for j in (-10, 10, -20, 20, -30, 30))):
        extra_text = _drop_if_collides(extra, dp, answer_rounded)
        if extra_text is not None:
            survivors.append(extra_text)
    distractors = pick_distractors(answer_text, survivors, n=n)
    check(len(distractors) == n, "need a full set of in-range distractors")
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer_rounded),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )


# --- tiers -------------------------------------------------------------------


def _easy(rng) -> dict:
    dp = 2
    for _attempt in range(_MAX_ATTEMPTS):
        lam = Fraction(rng.randint(2, 25), 10)  # 0.2 .. 2.5
        b = Fraction(rng.randint(5, 60), 10)  # 0.5 .. 6.0
        x = float(lam) * float(b)
        if not (0.3 <= x <= 4.0):
            continue
        raw = _raw_integral(lam, Fraction(0), b)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (Fraction(5, 100) < answer_rounded < Fraction(95, 100)):
            continue
        break
    else:
        check(False, "easy: could not draw a non-degenerate question")

    _verify_independent(raw, lam, Fraction(0), b, f"easy: independent check failed lam={lam} b={b}")

    sign_swap_raw = -raw
    wrong_exponent_raw = Fraction(math.exp(float(lam) * float(b))) - Fraction(1)

    # In-range misconceptions: the survival probability e^(-lambda*b) (forgets "1 -"), the rate left out of
    # the exponent, and the density value at b (reads f(b) as the area).
    survival_raw = Fraction(math.exp(-float(lam) * float(b)))
    no_rate_raw = Fraction(1) - Fraction(math.exp(-float(b)))
    density_raw = lam * survival_raw
    cands = [
        _drop_if_collides(sign_swap_raw, dp, answer_rounded),
        _drop_if_collides(wrong_exponent_raw, dp, answer_rounded),
        _drop_if_collides(survival_raw, dp, answer_rounded),
        _drop_if_collides(no_rate_raw, dp, answer_rounded),
        _drop_if_collides(density_raw, dp, answer_rounded),
        _drop_if_collides(Fraction(1) - Fraction(math.exp(-float(b) / float(lam))), dp, answer_rounded),
    ]

    lam_s, b_s = _fmt_param(lam), _fmt_param(b)
    text = phrase(rng, _FACT_TEMPLATES, lam=lam_s, a="0", b=b_s, dp=str(dp))
    return _finish(text, answer_rounded, dp, cands, n=2)


def _medium(rng) -> dict:
    dp = 2
    for _attempt in range(_MAX_ATTEMPTS):
        lam = Fraction(rng.randint(2, 25), 10)  # 0.2 .. 2.5
        a = Fraction(rng.randint(2, 30), 10)  # 0.2 .. 3.0
        gap = Fraction(rng.randint(5, 40), 10)  # 0.5 .. 4.0
        b = a + gap
        raw = _raw_integral(lam, a, b)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (Fraction(5, 100) < answer_rounded < Fraction(95, 100)):
            continue
        break
    else:
        check(False, "medium: could not draw a non-degenerate question")

    _verify_independent(raw, lam, a, b, f"medium: independent check failed lam={lam} a={a} b={b}")

    sign_swap_raw = -raw
    forgot_lower_raw = Fraction(1) - Fraction(math.exp(-float(lam) * float(b)))
    wrong_exponent_raw = Fraction(math.exp(float(lam) * float(b))) - Fraction(math.exp(float(lam) * float(a)))

    # In-range misconceptions after the two above that leave [0, 1]: only the upper term, only the lower
    # term, and a one-rectangle estimate (density at a times the width).
    upper_only_raw = Fraction(math.exp(-float(lam) * float(b)))
    lower_only_raw = Fraction(math.exp(-float(lam) * float(a)))
    rectangle_raw = lam * lower_only_raw * (b - a)
    cands = [
        _drop_if_collides(sign_swap_raw, dp, answer_rounded),
        _drop_if_collides(forgot_lower_raw, dp, answer_rounded),
        _drop_if_collides(wrong_exponent_raw, dp, answer_rounded),
        _drop_if_collides(upper_only_raw, dp, answer_rounded),
        _drop_if_collides(lower_only_raw, dp, answer_rounded),
        _drop_if_collides(rectangle_raw, dp, answer_rounded),
        # treats the interval as [0, b - a]: the width alone, not where it sits
        _drop_if_collides(Fraction(1) - Fraction(math.exp(-float(lam) * float(b - a))), dp, answer_rounded),
    ]

    lam_s, a_s, b_s = _fmt_param(lam), _fmt_param(a), _fmt_param(b)
    text = phrase(rng, _FACT_TEMPLATES, lam=lam_s, a=a_s, b=b_s, dp=str(dp))
    return _finish(text, answer_rounded, dp, cands)


def _hard(rng) -> dict:
    """4 d.p., and lambda/bounds are drawn so the two exponential terms stay
    close in magnitude -- ``e^(lambda*(b-a))`` capped well under 2 -- forcing
    the learner to carry enough precision through BOTH intermediate
    evaluations to avoid a cancellation error when they are subtracted. This
    is a numerical-stability trap (a genuinely different decision: how much
    precision to keep before subtracting two close values), not a bigger
    number chosen for its own sake."""
    dp = 4
    for _attempt in range(_MAX_ATTEMPTS):
        lam = Fraction(rng.randint(5, 30), 10)  # 0.5 .. 3.0
        a = Fraction(rng.randint(20, 300), 100)  # 0.20 .. 3.00
        gap = Fraction(rng.randint(2, 15), 100)  # 0.02 .. 0.15
        b = a + gap
        ratio = math.exp(float(lam) * float(gap))
        if not (1.02 <= ratio <= 1.6):
            continue
        raw = _raw_integral(lam, a, b)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if answer_rounded <= Fraction(1, 1000):
            continue
        break
    else:
        check(False, "hard: could not draw a non-degenerate near-cancellation question")

    _verify_independent(raw, lam, a, b, f"hard: independent check failed lam={lam} a={a} b={b}")

    sign_swap_raw = -raw
    forgot_lower_raw = Fraction(1) - Fraction(math.exp(-float(lam) * float(b)))
    wrong_exponent_raw = Fraction(math.exp(float(lam) * float(b))) - Fraction(math.exp(float(lam) * float(a)))

    # In-range misconceptions after the two above that leave [0, 1]: only the upper term, only the lower
    # term, and a one-rectangle estimate (density at a times the width).
    upper_only_raw = Fraction(math.exp(-float(lam) * float(b)))
    lower_only_raw = Fraction(math.exp(-float(lam) * float(a)))
    rectangle_raw = lam * lower_only_raw * (b - a)
    cands = [
        _drop_if_collides(sign_swap_raw, dp, answer_rounded),
        _drop_if_collides(forgot_lower_raw, dp, answer_rounded),
        _drop_if_collides(wrong_exponent_raw, dp, answer_rounded),
        _drop_if_collides(upper_only_raw, dp, answer_rounded),
        _drop_if_collides(lower_only_raw, dp, answer_rounded),
        _drop_if_collides(rectangle_raw, dp, answer_rounded),
        # treats the interval as [0, b - a]: the width alone, not where it sits
        _drop_if_collides(Fraction(1) - Fraction(math.exp(-float(lam) * float(b - a))), dp, answer_rounded),
    ]

    lam_s, a_s, b_s = _fmt_param(lam), _fmt_param(a), _fmt_param(b)
    text = phrase(rng, _FACT_TEMPLATES, lam=lam_s, a=a_s, b=b_s, dp=str(dp))
    return _finish(text, answer_rounded, dp, cands)


def _exam(rng) -> dict:
    """Additivity over a stated union of two disjoint intervals. Still the
    bare scope -- two pairs of stated bounds, two evaluations summed --
    with no scenario, no dataset, no word problem."""
    dp = 4
    for _attempt in range(_MAX_ATTEMPTS):
        lam = Fraction(rng.randint(5, 25), 10)  # 0.5 .. 2.5
        a1 = Fraction(rng.randint(10, 150), 100)  # 0.10 .. 1.50
        gap1 = Fraction(rng.randint(20, 80), 100)  # 0.20 .. 0.80
        b1 = a1 + gap1
        sep = Fraction(rng.randint(10, 60), 100)  # 0.10 .. 0.60
        a2 = b1 + sep
        gap2 = Fraction(rng.randint(20, 80), 100)  # 0.20 .. 0.80
        b2 = a2 + gap2

        raw1 = _raw_integral(lam, a1, b1)
        raw2 = _raw_integral(lam, a2, b2)
        raw = raw1 + raw2
        if not rounds_unambiguously(float(raw), dp):
            continue
        if not rounds_unambiguously(float(raw1), dp) or not rounds_unambiguously(float(raw2), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if answer_rounded <= Fraction(1, 1000):
            continue
        break
    else:
        check(False, "exam: could not draw a non-degenerate disjoint-union question")

    _verify_independent(raw1, lam, a1, b1, f"exam: independent check failed (interval 1) lam={lam} a1={a1} b1={b1}")
    _verify_independent(raw2, lam, a2, b2, f"exam: independent check failed (interval 2) lam={lam} a2={a2} b2={b2}")

    first_only_raw = raw1
    second_only_raw = raw2
    sign_swap_raw = -raw

    # In-range stand-ins for the sign swap: integrating the whole hull [a1, b2] (the gap counted too), and
    # only the upper terms.
    hull_raw = _raw_integral(lam, a1, b2)
    cands = [
        _drop_if_collides(first_only_raw, dp, answer_rounded),
        _drop_if_collides(second_only_raw, dp, answer_rounded),
        _drop_if_collides(sign_swap_raw, dp, answer_rounded),
        _drop_if_collides(hull_raw, dp, answer_rounded),
    ]

    lam_s = _fmt_param(lam)
    a1_s, b1_s, a2_s, b2_s = _fmt_param(a1), _fmt_param(b1), _fmt_param(a2), _fmt_param(b2)
    text = phrase(rng, _EXAM_TEMPLATES, lam=lam_s, a1=a1_s, b1=b1_s, a2=a2_s, b2=b2_s, dp=str(dp))
    return _finish(text, answer_rounded, dp, cands)


@generator(
    "p13c_exponential_pdf_numint",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13c_exponential_pdf_numint(rng, tier="easy"):
    """See module docstring for the full ladder and the scope.

    easy:   bounds [0, b], 2 d.p. -> 1 - e^(-lambda*b).
    medium: bounds [a, b] both nonzero, 2 d.p. -> e^(-lambda*a) - e^(-lambda*b).
    hard:   same shape, 4 d.p., lambda/bounds drawn for near-cancellation
            (a numerical-stability trap, not a bigger number).
    exam:   additivity over a stated union of two disjoint intervals, 4 d.p.
    """
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    return _exam(rng)
