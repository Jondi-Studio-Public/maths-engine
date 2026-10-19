"""A one-step Euler-Maruyama update (``ucv1-stochastic-euler-step``, ``stochastic-processes``, topic
``p13f_stochastic_euler``, ``answer: fraction``).

Boundary against the prerequisite ``explicit-euler-linear-ode``
(``mathgen/generators/university_scientific_computing.py::explicit_euler_linear_ode``):
that skill takes one (or two) exact, purely DETERMINISTIC explicit-Euler
step(s) for ``y' = lambda*y + forcing``: ``y1 = y0 + h*(lambda*y0 +
forcing)``. This skill supplies the SAME deterministic quantities plus a
diffusion term built from a supplied, stated noise value ``dW`` (never
sampled/simulated -- the "Brownian-path generation" exclusion) and asks for
one Euler-Maruyama update of ``dY = (lambda*Y + forcing) dt + sigma dW``:

    Y1 = Y0 + h*(lambda*Y0 + forcing) + sigma*dW

i.e. exactly the prerequisite's deterministic update PLUS ``sigma*dW``.
``sigma`` is a stated constant (additive noise) at easy/medium, or a stated
function of the CURRENT value, ``sigma(Y) = k*Y`` (multiplicative noise), at
hard/exam. Every quantity is given in the question text; the learner
substitutes and computes exactly (``Fraction`` arithmetic throughout).

Because the deterministic-only value is IDENTICAL to what
``explicit-euler-linear-ode`` answers for the same ``(lambda, forcing, Y0,
h)``, the "omits diffusion entirely" misconception distractor is also that
prerequisite skill's own correct answer for the same draw. Every tier's draw
loop therefore enforces ``sigma*dW != 0`` (the diffusion term never
vanishes) and ``check_pairwise_distinct``s the true answer against every
misconception candidate, unconditionally, before accepting a draw.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

_ADDITIVE_TEMPLATES = (
    "Use one Euler-Maruyama step of size h = {h} for dY = {drift} "
    "+ σ dW, starting from Y(0) = {y0}, where the diffusion "
    "coefficient σ = {sigma} is constant. The stated noise value for this "
    "step is ΔW = {dW}. Find the approximation Y(h).",
    "A stochastic process satisfies dY = {drift} + σ dW "
    "with constant σ = {sigma}. Starting from Y(0) = {y0}, take one "
    "Euler-Maruyama step of size h = {h} using the stated noise value ΔW = "
    "{dW}. Find Y(h).",
    "For dY = {drift} + σ dW (constant σ = {sigma}), "
    "starting at Y(0) = {y0}, compute one Euler-Maruyama update with step "
    "size h = {h} and stated noise value ΔW = {dW}. Give Y(h).",
)

_MULTIPLICATIVE_TEMPLATES = (
    "Use one Euler-Maruyama step of size h = {h} for dY = {drift} "
    "+ σ(Y) dW, starting from Y(0) = {y0}, where the "
    "diffusion coefficient scales with the current value, σ(Y) = {kY} "
    "(multiplicative noise). The stated noise value for this step is ΔW = "
    "{dW}. Find the approximation Y(h).",
    "A stochastic process satisfies dY = {drift} + "
    "σ(Y) dW with σ(Y) = {kY} (multiplicative noise). Starting from Y(0) "
    "= {y0}, take one Euler-Maruyama step of size h = {h} using the stated "
    "noise value ΔW = {dW}. Find Y(h).",
    "For dY = {drift} + σ(Y) dW with σ(Y) = {kY} "
    "(so the diffusion coefficient scales with the current value), "
    "starting at Y(0) = {y0}, compute one Euler-Maruyama update with step "
    "size h = {h} and stated noise value ΔW = {dW}. Give Y(h).",
)

_EXAM_TEMPLATES = (
    "Take two chained Euler-Maruyama steps of size h = {h} for dY = "
    "{drift} + σ(Y) dW with σ(Y) = {kY} "
    "(multiplicative noise), starting from Y(0) = {y0}. Both the drift and "
    "the diffusion coefficient are re-evaluated at Y1 for the second step. "
    "The stated noise values are ΔW1 = {dW1} for the first step and ΔW2 = "
    "{dW2} for the second step. Find the approximation Y(2h).",
    "A stochastic process satisfies dY = {drift} + "
    "σ(Y) dW with σ(Y) = {kY} (multiplicative noise). Starting from Y(0) "
    "= {y0}, take two Euler-Maruyama steps of size h = {h}, re-evaluating "
    "both the drift and the diffusion coefficient at Y1 for the second "
    "step, using stated noise values ΔW1 = {dW1} then ΔW2 = {dW2}. Find "
    "Y(2h).",
    "For dY = {drift} + σ(Y) dW with σ(Y) = {kY} "
    "(diffusion scales with the current value), starting at Y(0) = {y0}, "
    "chain two Euler-Maruyama steps of size h = {h} — re-evaluating drift "
    "and diffusion at Y1 for the second step — with stated noise values "
    "ΔW1 = {dW1} and ΔW2 = {dW2}. Give Y(2h).",
)


def _drift(lam: int, forcing: int) -> str:
    """The drift of ``dY = lam Y dt + forcing dt`` as it is written on paper: ``3Y dt + dt``, ``-Y dt - 2 dt``."""
    parts = []
    for coef, var in ((lam, "Y dt"), (forcing, " dt")):
        if coef == 0:
            continue
        mag = abs(coef)
        body = var.strip() if mag == 1 else f"{mag}{var}"
        if not parts:
            parts.append(("-" if coef < 0 else "") + body)
        else:
            parts.append(f"{'-' if coef < 0 else '+'} {body}")
    return " ".join(parts) if parts else "0"


def _sigma_y(k: int) -> str:
    return join_terms([(k, "Y")])


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _draw_lam_forcing_y0_h(rng, coefficient_hi, step_denominator_hi, value_hi, *, require_y0_not_one=False):
    lam = rng.randint(-coefficient_hi, coefficient_hi)
    forcing = rng.randint(-coefficient_hi, coefficient_hi)
    y0 = rng.randint(-value_hi, value_hi)
    if y0 == 0 or (require_y0_not_one and y0 == 1):
        return None
    denom = rng.randint(2, step_denominator_hi)
    h = Fraction(1, denom)
    slope0 = lam * y0 + forcing
    if slope0 == 0:
        return None
    return lam, forcing, y0, h, slope0


def _draw_dw(rng, hi, denom_hi):
    """A small nonzero stated noise value, sign and magnitude varied."""
    if rng.random() < 0.5:
        val = rng.randint(1, hi)
        if rng.random() < 0.5:
            val = -val
        return Fraction(val)
    num = rng.randint(1, hi)
    if rng.random() < 0.5:
        num = -num
    den = rng.randint(2, denom_hi)
    return Fraction(num, den)


def _build_additive(rng, coefficient_hi, step_denominator_hi, value_hi, dw_hi, dw_denom_hi, sigma_hi):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        draw = _draw_lam_forcing_y0_h(rng, coefficient_hi, step_denominator_hi, value_hi)
        if draw is None:
            continue
        lam, forcing, y0, h, slope0 = draw
        sigma = rng.randint(1, sigma_hi)
        if rng.random() < 0.5:
            sigma = -sigma
        sigma = Fraction(sigma)
        dw = _draw_dw(rng, dw_hi, dw_denom_hi)
        if h == 1 or dw == h:
            continue

        det = Fraction(y0) + h * slope0
        diffusion = sigma * dw
        answer = det + diffusion
        # Independent route: same terms, regrouped and re-associated.
        direct = (Fraction(y0) + h * Fraction(lam) * y0) + (h * forcing + sigma * dw)
        check_equivalent(answer, direct, "additive Euler-Maruyama regrouping mismatch")
        check(diffusion != 0, "diffusion term vanished")

        candidates = {
            "omits-diffusion": det,
            "omits-step-size-on-diffusion": det + sigma * h * dw,
            "wrong-sign-diffusion": det - diffusion,
            "uses-dW-as-h": Fraction(y0) + dw * slope0 + diffusion,
            # Same-shape (same denominator) fallback so an integer-vs-fraction
            # answer is never the only option of its own shape.
            "flips-answer-sign": -answer,
        }
        try:
            check_pairwise_distinct(
                answer, *candidates.values(), msg=f"additive collision: {lam},{forcing},{y0},{h},{sigma},{dw}"
            )
        except VerificationError:
            continue
        return lam, forcing, y0, h, sigma, dw, answer, candidates
    raise VerificationError("could not draw a collision-free additive-noise question")


def _build_multiplicative(rng, coefficient_hi, step_denominator_hi, value_hi, dw_hi, dw_denom_hi, k_hi):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        draw = _draw_lam_forcing_y0_h(rng, coefficient_hi, step_denominator_hi, value_hi, require_y0_not_one=True)
        if draw is None:
            continue
        lam, forcing, y0, h, slope0 = draw
        k = rng.randint(1, k_hi)
        if rng.random() < 0.5:
            k = -k
        dw = _draw_dw(rng, dw_hi, dw_denom_hi)
        if h == 1 or dw == h:
            continue

        sigma_val = Fraction(k) * y0
        det = Fraction(y0) + h * slope0
        diffusion = sigma_val * dw
        answer = det + diffusion
        direct = (Fraction(y0) + h * Fraction(lam) * y0) + (h * forcing + Fraction(k) * y0 * dw)
        check_equivalent(answer, direct, "multiplicative Euler-Maruyama regrouping mismatch")
        check(diffusion != 0, "diffusion term vanished")

        candidates = {
            "omits-diffusion": det,
            "omits-step-size-on-diffusion": det + sigma_val * h * dw,
            "wrong-sign-diffusion": det - diffusion,
            "uses-dW-as-h": Fraction(y0) + dw * slope0 + diffusion,
            "forgets-Y0-scaling-in-diffusion": det + Fraction(k) * dw,
            "flips-answer-sign": -answer,
        }
        try:
            check_pairwise_distinct(
                answer, *candidates.values(), msg=f"multiplicative collision: {lam},{forcing},{y0},{h},{k},{dw}"
            )
        except VerificationError:
            continue
        return lam, forcing, y0, h, k, dw, answer, candidates
    raise VerificationError("could not draw a collision-free multiplicative-noise question")


def _build_easy(rng) -> dict:
    lam, forcing, y0, h, sigma, dw, answer, candidates = _build_additive(
        rng,
        coefficient_hi=4,
        step_denominator_hi=4,
        value_hi=5,
        dw_hi=3,
        dw_denom_hi=3,
        sigma_hi=3,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(
        rng,
        _ADDITIVE_TEMPLATES,
        drift=_drift(lam, forcing),
        y0=y0,
        h=rational_text(h),
        sigma=rational_text(sigma),
        dW=rational_text(dw),
    )
    det = Fraction(y0) + h * (lam * y0 + forcing)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Deterministic part: Y0 + h(drift at Y0) = {y0} + ({rational_text(h)})({lam * y0 + forcing}) = {rational_text(det)}.",
            f"Diffusion part: σ·ΔW = ({rational_text(sigma)})·({rational_text(dw)}) = {rational_text(sigma * dw)}.",
            f"Y1 = {join_terms([(det, ''), (sigma * dw, '')])} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _build_medium(rng) -> dict:
    lam, forcing, y0, h, sigma, dw, answer, candidates = _build_additive(
        rng,
        coefficient_hi=7,
        step_denominator_hi=8,
        value_hi=9,
        dw_hi=6,
        dw_denom_hi=5,
        sigma_hi=6,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(
        rng,
        _ADDITIVE_TEMPLATES,
        drift=_drift(lam, forcing),
        y0=y0,
        h=rational_text(h),
        sigma=rational_text(sigma),
        dW=rational_text(dw),
    )
    det = Fraction(y0) + h * (lam * y0 + forcing)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Deterministic part: Y0 + h(drift at Y0) = {y0} + ({rational_text(h)})({lam * y0 + forcing}) = {rational_text(det)}.",
            f"Diffusion part: σ·ΔW = ({rational_text(sigma)})·({rational_text(dw)}) = {rational_text(sigma * dw)}.",
            f"Y1 = {join_terms([(det, ''), (sigma * dw, '')])} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _build_hard(rng) -> dict:
    lam, forcing, y0, h, k, dw, answer, candidates = _build_multiplicative(
        rng,
        coefficient_hi=6,
        step_denominator_hi=6,
        value_hi=8,
        dw_hi=4,
        dw_denom_hi=4,
        k_hi=4,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(
        rng,
        _MULTIPLICATIVE_TEMPLATES,
        drift=_drift(lam, forcing),
        forcing=forcing,
        y0=y0,
        h=rational_text(h),
        kY=_sigma_y(k),
        dW=rational_text(dw),
    )
    det = Fraction(y0) + h * (lam * y0 + forcing)
    sigma_val = Fraction(k) * y0
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Deterministic part: Y0 + h(drift at Y0) = {y0} + ({rational_text(h)})({lam * y0 + forcing}) = {rational_text(det)}.",
            f"σ(Y0) = ({k})·{y0} = {rational_text(sigma_val)}; diffusion part = ({rational_text(sigma_val)})·({rational_text(dw)}) = {rational_text(sigma_val * dw)}.",
            f"Y1 = {join_terms([(det, ''), (sigma_val * dw, '')])} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, distractor_strs, n=6),
    )


def _build_exam(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        draw = _draw_lam_forcing_y0_h(rng, coefficient_hi=5, step_denominator_hi=5, value_hi=6, require_y0_not_one=True)
        if draw is None:
            continue
        lam, forcing, y0, h, slope0 = draw
        k = rng.randint(1, 3)
        if rng.random() < 0.5:
            k = -k
        dw1 = _draw_dw(rng, 3, 3)
        dw2 = _draw_dw(rng, 3, 3)
        if h == 1 or dw1 == h or dw2 == h or dw1 == dw2:
            continue

        sigma0 = Fraction(k) * y0
        y1 = Fraction(y0) + h * slope0 + sigma0 * dw1
        if y1 == 0:
            continue
        slope1 = lam * y1 + forcing
        sigma1 = Fraction(k) * y1
        det2 = y1 + h * slope1
        diffusion2 = sigma1 * dw2
        y2 = det2 + diffusion2
        if diffusion2 == 0:
            continue

        # Independent route: recompute step 1 and step 2 from the prerequisite's
        # own deterministic-update expression, regrouped differently.
        direct_y1 = (Fraction(y0) + h * Fraction(lam) * y0) + (h * forcing + Fraction(k) * y0 * dw1)
        check_equivalent(y1, direct_y1, "exam step-1 regrouping mismatch")
        direct_y2 = (y1 + h * Fraction(lam) * y1) + (h * forcing + Fraction(k) * y1 * dw2)
        check_equivalent(y2, direct_y2, "exam step-2 regrouping mismatch")

        # Fully deterministic (omits both diffusion terms): identical to what
        # explicit-euler-linear-ode's own two-step preset would answer.
        det_only_y1 = Fraction(y0) + h * (lam * y0 + forcing)
        det_only_y2 = det_only_y1 + h * (lam * det_only_y1 + forcing)

        candidates = {
            "omits-diffusion": det_only_y2,
            "omits-step-size-on-diffusion": det2 + sigma1 * h * dw2,
            "wrong-sign-diffusion": det2 - diffusion2,
            "uses-dW-as-h": y1 + dw2 * slope1 + diffusion2,
            "forgets-Y0-scaling-in-diffusion": det2 + Fraction(k) * dw2,
            "flips-answer-sign": -y2,
        }
        try:
            check_pairwise_distinct(
                y2, *candidates.values(), msg=f"exam collision: {lam},{forcing},{y0},{h},{k},{dw1},{dw2}"
            )
        except VerificationError:
            continue

        payload, answer_text = _fraction_payload(y2)
        text = phrase(
            rng,
            _EXAM_TEMPLATES,
            drift=_drift(lam, forcing),
            y0=y0,
            h=rational_text(h),
            kY=_sigma_y(k),
            dW1=rational_text(dw1),
            dW2=rational_text(dw2),
        )
        distractor_strs = [rational_text(v) for v in candidates.values()]
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"Y1 = {y0} + ({rational_text(h)})({slope0}) + ({rational_text(sigma0)})({rational_text(dw1)}) = {rational_text(y1)}.",
                f"σ(Y1) = ({k})·{rational_text(y1)} = {rational_text(sigma1)}; slope at Y1 = ({lam})({rational_text(y1)})+({forcing}) = {rational_text(slope1)}.",
                f"Y2 = {rational_text(y1)} + ({rational_text(h)})({rational_text(slope1)}) + ({rational_text(sigma1)})({rational_text(dw2)}) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=6),
        )
    raise VerificationError("could not draw a collision-free exam-tier question")


@generator(
    "p13f_stochastic_euler",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def stochastic_euler(rng, tier="easy"):
    """Compute one Euler-Maruyama update for a supplied scalar drift,
    diffusion, step and noise value (the skill's stated contract).

    easy/medium: additive noise, sigma a stated constant.
    hard: multiplicative noise, sigma(Y) = k*Y.
    exam: two chained multiplicative-noise steps, re-evaluating both drift
    and diffusion at Y1 for the second step.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
