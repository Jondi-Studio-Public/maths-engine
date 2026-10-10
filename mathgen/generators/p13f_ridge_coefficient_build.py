"""A one-feature ridge-regression coefficient (``ucv1-ridge-coefficient``, ``statistical-learning``, topic
``p13f_ridge_coefficient``, ``answer: fraction``).

**Derivation (checked by the independent oracle, not just trusted).** Ridge
regression on a single, CENTRED feature (mean-zero x, mean-zero y -- no
intercept) fits ``beta`` to minimise

    J(beta) = sum_i (y_i - beta*x_i)^2 + lambda*beta^2.

``dJ/dbeta = -2*sum(x_i*(y_i - beta*x_i)) + 2*lambda*beta``. Setting this to
zero: ``sum(x_i*y_i) - beta*sum(x_i^2) - lambda*beta = 0``, i.e.

    beta = Sxy / (Sxx + lambda),    Sxx = sum(x_i^2), Sxy = sum(x_i*y_i).

Boundary against the prerequisite ``least-squares-line``
(``mathgen/generators/university_applied.py``-family): that skill fits an
UNPENALISED line with an intercept to RAW (uncentred) data, answering a pair
``(m, b)``. This skill supplies CENTRED one-feature data (no intercept term
needed or asked for) plus a penalty ``lambda >= 1`` and asks for the single
closed-form ridge coefficient above. At ``lambda = 0`` this formula
degenerates to the plain OLS slope ``Sxy/Sxx``, which is (up to the missing
intercept) ``least-squares-line``'s own answer shape -- so every draw here
enforces ``lambda >= 2`` (never 0, and never 1 either: at ``lambda = 1`` the
``uses-lambda-squared`` distractor, ``Sxy/(Sxx+1)``, collides with the
answer) and ``check_pairwise_distinct``s the true answer against the
zero-penalty value on every seed.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

_GIVEN_TEMPLATES = (
    "For a centred one-feature dataset (mean-zero x, mean-zero y; fit no "
    "intercept) with Sxx = sum(x_i^2) = {Sxx} and Sxy = sum(x_i*y_i) = {Sxy}, "
    "find the ridge coefficient beta = Sxy / (Sxx + lambda) for penalty "
    "lambda = {lam}.",
    "A ridge regression on centred one-feature data (no intercept term) "
    "has Sxx = sum(x_i^2) = {Sxx} and Sxy = sum(x_i*y_i) = {Sxy}. Using "
    "penalty lambda = {lam}, find beta = Sxy / (Sxx + lambda).",
    "Given Sxx = sum(x_i^2) = {Sxx} and Sxy = sum(x_i*y_i) = {Sxy} for a "
    "centred one-feature dataset (mean-zero x, mean-zero y, no intercept), "
    "compute the ridge coefficient beta = Sxy / (Sxx + lambda) with "
    "penalty lambda = {lam}.",
)

_HARD_TEMPLATES = (
    "A dataset has these centred points (mean-zero x, mean-zero y; fit no "
    "intercept): {points}. Using penalty lambda = {lam}, compute the ridge "
    "coefficient beta = sum(x_i*y_i) / (sum(x_i^2) + lambda).",
    "For the centred one-feature points {points} (mean-zero x, mean-zero y, "
    "no intercept), find the ridge coefficient beta = sum(x_i*y_i) / "
    "(sum(x_i^2) + lambda) with penalty lambda = {lam}.",
    "A one-feature dataset, already centred to mean zero in both x and y "
    "(no intercept fitted), has points {points}. Compute the ridge "
    "coefficient beta = sum(x_i*y_i) / (sum(x_i^2) + lambda) for penalty "
    "lambda = {lam}.",
)

_EXAM_TEMPLATES = (
    "For a centred one-feature dataset, Sxx = sum(x_i^2) = {Sxx} and "
    "Sxy = sum(x_i*y_i) = {Sxy}. The ridge coefficient beta = Sxy / "
    "(Sxx + lambda) works out to {beta} for some penalty lambda > 0. Find "
    "lambda.",
    "A ridge regression on centred one-feature data has Sxx = "
    "sum(x_i^2) = {Sxx} and Sxy = sum(x_i*y_i) = {Sxy}. Its ridge "
    "coefficient is beta = {beta} = Sxy / (Sxx + lambda) for some penalty "
    "lambda > 0. Find lambda.",
    "Given Sxx = sum(x_i^2) = {Sxx} and Sxy = sum(x_i*y_i) = {Sxy} for a "
    "centred one-feature dataset, the ridge coefficient Sxy / (Sxx + "
    "lambda) equals {beta}. Find the penalty lambda (lambda > 0).",
)


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _beta_candidates(Sxx: int, Sxy: int, lam: int) -> dict:
    """The four given-directly-tier misconception values, computed by a
    plausible WRONG formula each -- never a restatement of the correct one."""
    return {
        "omits-lambda": Fraction(Sxy, Sxx),
        "subtracts-lambda": Fraction(Sxy, Sxx - lam),
        "adds-lambda-to-numerator": Fraction(Sxy + lam, Sxx),
        "uses-lambda-squared": Fraction(Sxy, Sxx + lam * lam),
    }


def _draw_easy_medium(rng, *, sxx_hi, lam_hi, beta_hi, force_fraction):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        Sxx = rng.randint(1, sxx_hi)
        lam = rng.randint(2, lam_hi)
        if Sxx == lam or Sxx + lam == 0 or Sxx + lam * lam == 0:
            continue
        if force_fraction:
            Sxy = rng.randint(-beta_hi, beta_hi)
            if Sxy == 0:
                continue
        else:
            beta_target = rng.randint(-beta_hi, beta_hi)
            if beta_target == 0:
                continue
            Sxy = beta_target * (Sxx + lam)
        answer = Fraction(Sxy, Sxx + lam)
        if force_fraction and answer.denominator == 1:
            continue
        direct = Fraction(Sxy) / Fraction(Sxx + lam)
        check_equivalent(answer, direct, "ridge closed-form recomputation mismatch")

        zero_penalty = Fraction(Sxy, Sxx) if Sxx != 0 else None
        candidates = _beta_candidates(Sxx, Sxy, lam)
        # A same-shape (same denominator) fallback so an integer-vs-fraction
        # answer is never the only option of its own shape (regrade sweep
        # convention, "the answer and every distractor formatted alike") --
        # also the "penalty-direction distractor" idea:
        # reports the ridge coefficient with the wrong sign, as if the
        # penalty pulled beta the other way.
        candidates["flips-penalty-sign"] = -answer
        values_to_check = [answer, *candidates.values()]
        try:
            check_pairwise_distinct(*values_to_check, msg=f"ridge collision: Sxx={Sxx}, Sxy={Sxy}, lam={lam}")
            if zero_penalty is not None:
                check(answer != zero_penalty, "ridge answer collides with the zero-penalty (OLS) value")
        except VerificationError:
            continue
        return Sxx, Sxy, lam, answer, candidates
    raise VerificationError("could not draw a collision-free given-directly ridge question")


def _build_easy(rng) -> dict:
    Sxx, Sxy, lam, answer, candidates = _draw_easy_medium(
        rng,
        sxx_hi=10,
        lam_hi=6,
        beta_hi=14,
        force_fraction=False,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(rng, _GIVEN_TEMPLATES, Sxx=Sxx, Sxy=Sxy, lam=lam)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"beta = Sxy / (Sxx + lambda) = {Sxy} / ({Sxx} + {lam}) = {Sxy} / {Sxx + lam} = {answer_text}."],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _build_medium(rng) -> dict:
    Sxx, Sxy, lam, answer, candidates = _draw_easy_medium(
        rng,
        sxx_hi=14,
        lam_hi=9,
        beta_hi=15,
        force_fraction=True,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(rng, _GIVEN_TEMPLATES, Sxx=Sxx, Sxy=Sxy, lam=lam)
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"beta = Sxy / (Sxx + lambda) = {Sxy} / ({Sxx} + {lam}) = {Sxy} / {Sxx + lam} = {answer_text}."],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _draw_centred_points(rng, n, hi):
    xs = [rng.randint(-hi, hi) for _ in range(n - 1)]
    xs.append(-sum(xs))
    ys = [rng.randint(-hi, hi) for _ in range(n - 1)]
    ys.append(-sum(ys))
    return xs, ys


def _build_hard(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        n = rng.choice((3, 4))
        xs, ys = _draw_centred_points(rng, n, hi=6)
        check(sum(xs) == 0 and sum(ys) == 0, "centred data does not sum to zero")
        if all(x == 0 for x in xs):
            continue
        Sxx = sum(x * x for x in xs)
        Sxy = sum(x * y for x, y in zip(xs, ys))
        Syy = sum(y * y for y in ys)
        if Sxx == 0 or Sxy == 0 or Syy == Sxy:
            continue
        lam = rng.randint(2, 8)
        if Sxx == lam:
            continue

        answer = Fraction(Sxy, Sxx + lam)
        # Independent route: recompute Sxx/Sxy via a second, differently
        # ordered accumulation (dot-product identity rather than the
        # elementwise sum used above).
        direct_sxx = sum(x**2 for x in reversed(xs))
        direct_sxy = sum(y * x for x, y in zip(xs, ys))
        check(direct_sxx == Sxx, "Sxx recomputation mismatch")
        check(direct_sxy == Sxy, "Sxy recomputation mismatch")
        check_equivalent(answer, Fraction(direct_sxy, direct_sxx + lam), "hard-tier ridge recomputation mismatch")

        zero_penalty = Fraction(Sxy, Sxx)
        candidates = _beta_candidates(Sxx, Sxy, lam)
        candidates["wrong-sum"] = Fraction(Syy, Sxx + lam)
        candidates["flips-penalty-sign"] = -answer
        values_to_check = [answer, *candidates.values()]
        try:
            check_pairwise_distinct(*values_to_check, msg=f"hard-tier ridge collision: xs={xs}, ys={ys}, lam={lam}")
            check(answer != zero_penalty, "hard-tier ridge answer collides with zero-penalty value")
        except VerificationError:
            continue

        payload, answer_text = _fraction_payload(answer)
        points_text = ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))
        text = phrase(rng, _HARD_TEMPLATES, points=points_text, lam=lam)
        distractor_strs = [rational_text(v) for v in candidates.values()]
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"Sxx = sum(x_i^2) = {Sxx}. Sxy = sum(x_i*y_i) = {Sxy}.",
                f"beta = Sxy / (Sxx + lambda) = {Sxy} / ({Sxx} + {lam}) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=6),
        )
    raise VerificationError("could not draw a collision-free hard-tier ridge question")


def _build_exam(rng) -> dict:
    """Inverse tier: Sxx, Sxy and the resulting beta are stated; the learner
    recovers the penalty lambda from beta = Sxy/(Sxx+lambda), i.e.
    lambda = Sxy/beta - Sxx. This equation is LINEAR in lambda (beta != 0 is
    enforced by construction, since Sxy != 0 and Sxx+lambda != 0), so the
    solution is unique by construction -- there is no second algebraic root
    to rule out, unlike a quadratic inversion. Proved directly below by
    recomputing lambda from the stated beta and checking it lands back on
    the drawn value, the same discipline p12f_markov_hitting_build.py's
    ``_build_exam`` uses for its own (quadratic) inversion."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        Sxx = rng.randint(1, 10)
        lam = rng.randint(2, 20)
        if Sxx == lam:
            continue
        beta_target = rng.randint(-6, 6)
        if beta_target == 0:
            continue
        Sxy = beta_target * (Sxx + lam)
        if Sxy == 0:
            continue
        beta = Fraction(Sxy, Sxx + lam)
        check(beta != 0, "exam-tier beta must be nonzero to invert")

        recovered_lam = Fraction(Sxy, 1) / beta - Sxx
        check_equivalent(recovered_lam, lam, "recovered lambda does not match the drawn lambda")

        candidates = {
            "omits-lambda": Fraction(0),
            # Sxy/beta == Sxx+lam identically (by construction), so this is
            # ALWAYS exactly -lam -- also the same-shape (integer) fallback
            # so the answer is never the only integer-shaped option.
            "subtracts-lambda": Sxx - Fraction(Sxy, 1) / beta,
            "adds-lambda-to-numerator": beta * Sxx - Sxy,
        }
        try:
            check_pairwise_distinct(
                Fraction(lam), *candidates.values(), msg=f"exam-tier ridge collision: Sxx={Sxx}, Sxy={Sxy}, lam={lam}"
            )
        except VerificationError:
            continue

        answer_text = rational_text(Fraction(lam))
        payload = {"p": lam, "q": 1}
        beta_text = rational_text(beta)
        text = phrase(rng, _EXAM_TEMPLATES, Sxx=Sxx, Sxy=Sxy, beta=beta_text)
        distractor_strs = [rational_text(v) for v in candidates.values()]
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"beta = Sxy/(Sxx+lambda), so Sxx+lambda = Sxy/beta = {Sxy}/({beta_text}) = {rational_text(Fraction(Sxy, 1) / beta)}.",
                f"lambda = {rational_text(Fraction(Sxy, 1) / beta)} - {Sxx} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=3),
        )
    raise VerificationError("could not draw a collision-free exam-tier ridge question")


@generator(
    "p13f_ridge_coefficient",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def ridge_coefficient(rng, tier="easy"):
    """Compute the closed-form ridge coefficient for a supplied centred
    one-feature dataset and penalty (the skill's stated
    contract): beta = Sxy / (Sxx + lambda).

    easy/medium: Sxx, Sxy and lambda given directly (medium always produces
    a genuine fraction answer, while easy may produce an integer).
    hard: raw centred data points given instead of the pre-computed sums.
    exam: inverse -- Sxx, Sxy and the resulting beta are stated; recover
    lambda.
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
