"""Normal distribution probabilities and quantiles.

One skill, ``senior-calculate-probabilities-and-quantiles-associated-with-a-given``,
``domain: statistics-probability``, ``answer: number``,
``prerequisites: [simple-event-probability]``, topic
``p13d_normal_dist_prob_quantile``.

Skill name: "Calculate probabilities and quantiles associated with a given normal distribution using
technology, and use these to solve practical problems." This is a CONTINUOUS ``N(mu, sigma^2)`` distribution
computed with ``statistics.NormalDist`` -- never a discrete pmf, never Bernoulli, never a hand-rolled erf
approximation.

Every scenario states a subject that is normally distributed with a stated
mean ``mu`` and standard deviation ``sigma`` (both in the same physical
unit). Four tiers, each a genuine change of DECISION:

- **easy**: ONE tail -- ``P(X < a)`` or ``P(X > b)`` for a stated threshold,
  a direct ``cdf``/``1-cdf`` call, reported to 4 decimal places.
- **medium**: BETWEEN two values -- ``P(a < X < b) = cdf(b) - cdf(a)``, a
  genuinely different computation (two ``cdf`` calls and a subtraction), not
  just a second threshold on the same one-tail shape.
- **hard**: INVERSE -- given a stated cumulative proportion ``q``, find the
  quantile ``x`` with ``inv_cdf(q)``. A genuinely different decision
  (inverting the distribution) from finding a probability.
- **exam**: given a stated event proportion ``p`` and threshold ``a`` but
  WITHOUT one of ``mu``/``sigma`` (the other is stated), solve for the
  missing parameter via ``z = inv_cdf(p)`` and ``a = mu + z*sigma``. This
  composes the inverse standard-normal step with solving a linear equation
  for a distribution PARAMETER -- a genuinely different reasoning move from
  finding ``x`` itself (hard tier).

Every probability is stated to 4 decimal places in both question text and
the (identical) latex; every quantile/parameter answer states its own
unit-appropriate precision (1 decimal place for the quantile, 2 decimal
places for the solved parameter), per ``stated_precision_string``/
``stated_precision_tolerance`` (never hand-rolled rounding).

**Independent in-generator check, a genuinely different route from
construction, every tier:**

- easy: construction calls ``NormalDist(mu, sigma).cdf(a)`` (or
  ``1 - cdf(b)``) directly; the check standardises first --
  ``NormalDist(0, 1).cdf((a - mu) / sigma)`` (or its complementary tail) --
  a different call-path/argument shape for the same identity.
- medium: construction computes ``cdf(b) - cdf(a)``; the check reconstructs
  via the complementary tail, ``(1 - cdf(a)) - (1 - cdf(b))``, a different
  call sequence for the same identity.
- hard: construction calls ``NormalDist(mu, sigma).inv_cdf(q)`` directly;
  the check standardises -- ``z = NormalDist(0, 1).inv_cdf(q)``,
  ``x = mu + z*sigma`` -- and confirms it reproduces the constructed value.
- exam: construction solves the linear equation for the missing parameter;
  the check re-substitutes the solved parameter back into
  ``NormalDist(mu, sigma).cdf(a)`` and confirms it reproduces the STATED
  proportion ``p`` -- a genuinely different check from the algebra that
  solved for it.

All four checks compare at ``abs(...) < 1e-9`` for the probability/quantile
identities (float precision of ``NormalDist`` itself) and ``< 1e-6`` for the
exam tier's re-substituted proportion (a looser bound because the parameter
being re-substituted was itself already rounded to 2 decimal places for
display, so exact float agreement is not expected).

**Misconceptions** (each a genuinely different value, checked unconditionally
every seed via a local ``_gap_ok`` margin at the tier's own stated-precision
tolerance):

- ``wrong_tail`` -- reports ``1 - p`` instead of the asked-for tail/range
  (easy, medium via the complementary total).
- ``reports_z_not_x`` -- reports the standardised ``z`` instead of
  converting back to ``x``/the solved parameter (hard, exam).
- ``variance_as_sd`` -- uses ``sigma**2`` where ``sigma`` belongs in the
  ``NormalDist`` call (easy, medium, hard).
- ``sigma_as_unit_variance`` (easy only, beyond the core list --
  ``sigma`` at all, computing as if ``sigma = 1``. A genuine, common,
  defensible "forgot to standardise by sigma" slip, needed because the
  core list only names two applicable easy-tier misconceptions.
  ``swapped_subtraction``) -- reports ``cdf(b)`` alone, forgetting to
  subtract ``cdf(a)`` (computes only "at most b", dropping the "at least a"
  condition). Replaces the earlier ``cdf(a) - cdf(b)`` formula, which is
  always negative and so instantly recognisable as wrong on units alone (a
  probability cannot be negative) -- ``cdf(b)`` alone is always a genuine
  probability in ``(0, 1)`` and a real, common student slip.
- ``wrong_tail_quantile`` (hard only) -- uses ``inv_cdf(1 - q)`` instead of
  ``inv_cdf(q)``, confusing "smallest p%" with "largest p%".
- ``sign_error`` (exam only, beyond the core list -- documented
  deviation) -- flips the sign in the linear equation. For the ``mu`` branch,
  solves ``mu = a + sigma*z`` instead of ``a - sigma*z`` (unchanged from the
  original build; this value is always in the scenario's own plausible
  x-range and is now asserted, not assumed). For the ``sigma`` branch, this replaces
  the earlier distractor (a negative standard deviation -- physically
  impossible and instantly recognisable as wrong) with ``forgot_to_divide_by_z`` -- reports
  ``abs(a - mu)`` directly, having forgotten to divide by ``z`` at all. This
  is always positive (a genuine, common "forgot the last step" slip) and
  structurally different from the correct ``(a - mu) / z``.

**Domain validity:** every answer and every candidate
distractor is checked against its own physical domain before being offered --
a probability lies in ``[0, 1]`` (easy, medium), a standard deviation is
strictly positive and no more than 3x the scenario's own upper sigma bound
(exam, ``sigma`` branch), and an x-value/solved-mu sits within 5 sigma-highs
of the scenario's own declared mu range (hard, exam ``mu`` branch) -- never
the old global constant range. A candidate outside its domain is resampled
exactly like the existing ``_gap_ok`` collision check, inside the same loop.
"""

from __future__ import annotations

from fractions import Fraction
from statistics import NormalDist

from ..core import (
    check,
    generator,
    numeric_answer,
    paren_negative,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 300
_DP_PROB = 4
_DP_QUANTILE = 1
_DP_PARAM = 2
_TOL_PROB = stated_precision_tolerance(_DP_PROB)
_TOL_QUANTILE = stated_precision_tolerance(_DP_QUANTILE)
_TOL_PARAM = stated_precision_tolerance(_DP_PARAM)

# Each scenario carries its OWN plausible mu/sigma range and its own small-side/large-side wording
# -- replaces the single shared `rng.randint(50, 900)` draw that produced a 659 mm "bolt" and a 134
# cm "adult male" from the same numeric range regardless of unit.
_SCENARIOS = [
    dict(
        subject="the fill weight of a cereal box",
        unit="g",
        item="box",
        plural="boxes",
        mu_lo=300,
        mu_hi=900,
        sigma_lo=5,
        sigma_hi=25,
        small_word="lightest",
        large_word="heaviest",
    ),
    dict(
        subject="the height of an adult male",
        unit="cm",
        item="man",
        plural="men",
        mu_lo=165,
        mu_hi=185,
        sigma_lo=6,
        sigma_hi=9,
        small_word="shortest",
        large_word="tallest",
    ),
    dict(
        subject="the battery life of a phone model",
        unit="hours",
        item="phone",
        plural="phones",
        mu_lo=8,
        mu_hi=20,
        sigma_lo=1,
        sigma_hi=3,
        small_word="shortest-lived",
        large_word="longest-lived",
    ),
    dict(
        subject="the reaction time of a driver",
        unit="ms",
        item="driver",
        plural="drivers",
        mu_lo=150,
        mu_hi=300,
        sigma_lo=15,
        sigma_hi=40,
        small_word="fastest",
        large_word="slowest",
    ),
    dict(
        subject="the diameter of a manufactured bolt",
        unit="mm",
        item="bolt",
        plural="bolts",
        mu_lo=5,
        mu_hi=30,
        sigma_lo=0.1,
        sigma_hi=1.0,
        small_word="narrowest",
        large_word="widest",
    ),
]


def _draw_mu_sigma(rng, scenario: dict) -> tuple[float, float]:
    """Per-scenario mu/sigma draw, replacing the old
    shared `rng.randint(50, 900)` / `rng.uniform(0.05, 0.15) * mu`."""
    mu = round(rng.uniform(scenario["mu_lo"], scenario["mu_hi"]), 1)
    sigma = round(rng.uniform(scenario["sigma_lo"], scenario["sigma_hi"]), 1)
    return mu, sigma


def _prob_in_domain(p: float) -> bool:
    """A probability lies in [0, 1] -- never negative, never above 1."""
    return 0.0 <= p <= 1.0


def _sigma_in_domain(value: float, sigma_hi: float) -> bool:
    """A standard deviation is strictly positive and no more than 3x the
    scenario's own upper sigma bound."""
    return 0.0 < value < sigma_hi * 3


def _x_in_domain(value: float, scenario: dict) -> bool:
    """An x-value (or a solved mu) sits within 5 sigma-highs of the
    scenario's own declared mu range -- wide enough for a genuine tail
    quantile, narrow enough to reject e.g. a 4281 mm bolt. Every scenario in `_SCENARIOS` is a physical
    quantity that cannot be negative (a weight, a height, a duration, a
    diameter), so the lower bound is also floored at 0 -- a negative
    "battery life" or "reaction time" distractor is exactly the class of
    physically-impossible value this predicate exists to reject."""
    lo = max(0.0, scenario["mu_lo"] - 5 * scenario["sigma_hi"])
    hi = scenario["mu_hi"] + 5 * scenario["sigma_hi"]
    return lo <= value <= hi


def _to_fraction(value: float) -> Fraction:
    return Fraction(value).limit_denominator(10**9)


def _round(value: float, dp: int) -> Fraction:
    return round_half_up(_to_fraction(value), dp)


def _fmt(value: Fraction, dp: int) -> str:
    return stated_precision_string(value, dp)


def _near_rounding_boundary(value: float, dp: int, eps: float = 1e-6) -> bool:
    """True when `value` sits within `eps` of the.xx5 halfway point at `dp` decimal places, where construction's
    float arithmetic and a distractor's could legitimately round in opposite directions. One-off local helper."""
    scaled = value * (10**dp)
    frac = scaled - (scaled // 1)
    return abs(frac - 0.5) < eps or abs(frac + 0.5) < eps


def _gap_ok(values: list[Fraction], tol: Fraction) -> bool:
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tol:
                return False
    return True


# =============================================================================
# easy: one-tail probability P(X < a) or P(X > b)
# =============================================================================


def _tail_prob(nd: NormalDist, a: float, tail: str) -> float:
    return nd.cdf(a) if tail == "less" else 1.0 - nd.cdf(a)


def _build_easy(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        scenario = rng.choice(_SCENARIOS)
        mu, sigma = _draw_mu_sigma(rng, scenario)
        if sigma <= 0:
            continue
        tail = rng.choice(["less", "greater"])
        delta = rng.uniform(0.3, 2.2) * rng.choice([-1, 1])
        a = round(mu + delta * sigma, 1)

        nd = NormalDist(mu, sigma)
        p_val = _tail_prob(nd, a, tail)
        if not (1e-4 < p_val < 1 - 1e-4) or _near_rounding_boundary(p_val, _DP_PROB):
            continue
        ans = _round(p_val, _DP_PROB)

        wrong_tail_val = 1.0 - p_val
        variance_as_sd_val = _tail_prob(NormalDist(mu, sigma**2), a, tail)
        sigma_as_one_val = _tail_prob(NormalDist(mu, 1), a, tail)

        cand_floats = [wrong_tail_val, variance_as_sd_val, sigma_as_one_val]
        if any(not _prob_in_domain(v) or _near_rounding_boundary(v, _DP_PROB) for v in cand_floats):
            continue
        cands = [_round(v, _DP_PROB) for v in cand_floats]
        if not _gap_ok([ans] + cands, _TOL_PROB):
            continue
        break
    else:
        raise ValueError("p13d_normal_dist_prob_quantile: no collision-free easy draw found")

    # Independent check: standardise first, a different call path than the
    # construction's direct NormalDist(mu, sigma) call.
    std = NormalDist(0.0, 1.0)
    z = (a - mu) / sigma
    check_val = std.cdf(z) if tail == "less" else 1.0 - std.cdf(z)
    check(abs(check_val - p_val) < 1e-9, "easy: standardised route disagrees with direct cdf call")

    comparator = "less than" if tail == "less" else "greater than"
    text = (
        f"{scenario['subject'].capitalize()} is normally distributed with mean {mu} {scenario['unit']} "
        f"and standard deviation {sigma} {scenario['unit']}. Find the probability that a randomly "
        f"selected {scenario['item']} has a value {comparator} {a} {scenario['unit']}. "
        f"Give your answer correct to 4 decimal places."
    )
    latex = text
    answer_text = _fmt(ans, _DP_PROB)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_PROB) for c in cands], n=3)
    check(len(distractors) >= 1, "easy tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"z = ({a} - {mu}) / {sigma} = {z:.4f}. P = {'Phi(z)' if tail == 'less' else '1 - Phi(z)'} = {answer_text}."
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: between two values P(a < X < b) = cdf(b) - cdf(a)
# =============================================================================


def _build_medium(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        scenario = rng.choice(_SCENARIOS)
        mu, sigma = _draw_mu_sigma(rng, scenario)
        if sigma <= 0:
            continue
        offset_lo = rng.uniform(0.3, 1.8)
        offset_hi = rng.uniform(0.3, 1.8)
        a = round(mu - offset_lo * sigma, 1)
        b = round(mu + offset_hi * sigma, 1)
        if a >= b:
            continue

        nd = NormalDist(mu, sigma)
        p_val = nd.cdf(b) - nd.cdf(a)
        if not (1e-4 < p_val < 1 - 1e-4) or _near_rounding_boundary(p_val, _DP_PROB):
            continue
        ans = _round(p_val, _DP_PROB)

        # forgot_to_subtract: reports cdf(b) alone, forgetting to subtract the lower tail cdf(a) --
        # always a genuine probability in (0, 1).
        forgot_to_subtract_val = nd.cdf(b)
        wrong_tail_val = 1.0 - p_val  # complementary total
        variance_val = NormalDist(mu, sigma**2).cdf(b) - NormalDist(mu, sigma**2).cdf(a)

        cand_floats = [forgot_to_subtract_val, wrong_tail_val, variance_val]
        if any(not _prob_in_domain(v) or _near_rounding_boundary(v, _DP_PROB) for v in cand_floats):
            continue
        cands = [_round(v, _DP_PROB) for v in cand_floats]
        if not _gap_ok([ans] + cands, _TOL_PROB):
            continue
        break
    else:
        raise ValueError("p13d_normal_dist_prob_quantile: no collision-free medium draw found")

    # Independent check: reconstruct via the complementary tail -- a
    # different call sequence than construction's direct cdf(b) - cdf(a).
    check_val = (1.0 - nd.cdf(a)) - (1.0 - nd.cdf(b))
    check(abs(check_val - p_val) < 1e-9, "medium: complementary-tail route disagrees with direct cdf difference")

    text = (
        f"{scenario['subject'].capitalize()} is normally distributed with mean {mu} {scenario['unit']} "
        f"and standard deviation {sigma} {scenario['unit']}. Find the probability that a randomly "
        f"selected {scenario['item']} has a value between {a} {scenario['unit']} and {b} {scenario['unit']}. "
        f"Give your answer correct to 4 decimal places."
    )
    latex = text
    answer_text = _fmt(ans, _DP_PROB)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_PROB) for c in cands], n=3)
    check(len(distractors) >= 1, "medium tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"P({a} < X < {b}) = cdf({b}) - cdf({a}) = {answer_text}."],
        distractors=distractors,
    )


# =============================================================================
# hard: inverse -- given proportion q, find the quantile x
# =============================================================================


def _build_hard(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        scenario = rng.choice(_SCENARIOS)
        mu, sigma = _draw_mu_sigma(rng, scenario)
        if sigma <= 0:
            continue
        q = round(rng.uniform(0.02, 0.20), 2)
        side = rng.choice(["smallest", "largest"])
        q_eff = q if side == "smallest" else 1.0 - q

        nd = NormalDist(mu, sigma)
        x_val = nd.inv_cdf(q_eff)
        if _near_rounding_boundary(x_val, _DP_QUANTILE) or not _x_in_domain(x_val, scenario):
            continue
        ans = _round(x_val, _DP_QUANTILE)

        z_only_val = NormalDist(0.0, 1.0).inv_cdf(q_eff)  # reports z, never converts to x
        wrong_tail_val = nd.inv_cdf(1.0 - q_eff)  # inverted the wrong side
        variance_val = NormalDist(mu, sigma**2).inv_cdf(q_eff)  # uses sigma^2 in place of sigma

        # z_only_val's TYPE is a dimensionless z-score, not an x-value in the
        # scenario's own units -- the x-domain predicate is a category error
        # for it. Its own domain (a standard-normal quantile for
        # q in [0.02, 0.98]) is satisfied by construction; only its
        # rounding-boundary safety is checked here. wrong_tail_val and
        # variance_val ARE genuinely x-typed (both still `inv_cdf`-shaped
        # values in the scenario's own units) and get the full x-domain check.
        if _near_rounding_boundary(z_only_val, _DP_QUANTILE):
            continue
        x_typed_cands = [wrong_tail_val, variance_val]
        if any(_near_rounding_boundary(v, _DP_QUANTILE) or not _x_in_domain(v, scenario) for v in x_typed_cands):
            continue
        cand_floats = [z_only_val, wrong_tail_val, variance_val]
        cands = [_round(v, _DP_QUANTILE) for v in cand_floats]
        if not _gap_ok([ans] + cands, _TOL_QUANTILE):
            continue
        break
    else:
        raise ValueError("p13d_normal_dist_prob_quantile: no collision-free hard draw found")

    # Independent check: standardise, then convert back -- a different route
    # than construction's direct NormalDist(mu, sigma).inv_cdf(q_eff) call.
    z = NormalDist(0.0, 1.0).inv_cdf(q_eff)
    check_val = mu + z * sigma
    check(abs(check_val - x_val) < 1e-9, "hard: standardised route disagrees with direct inv_cdf call")

    # No "rejected"/"accepted" (nonsensical for a person -- "the smallest 12% of men are rejected").
    # Names the scenario's own small-side/large-side word instead; the underlying maths (inv_cdf(q)
    # for the smallest side, inv_cdf(1 - q) for the largest side) is unchanged.
    direction = "below" if side == "smallest" else "above"
    side_word = scenario["small_word"] if side == "smallest" else scenario["large_word"]
    text = (
        f"{scenario['subject'].capitalize()} is normally distributed with mean {mu} {scenario['unit']} "
        f"and standard deviation {sigma} {scenario['unit']}. What is the value {direction} which the "
        f"{side_word} {int(round(q * 100))}% of {scenario['plural']} fall, in {scenario['unit']}? "
        f"Give your answer correct to 1 decimal place."
    )
    latex = text
    answer_text = _fmt(ans, _DP_QUANTILE)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_QUANTILE) for c in cands], n=3)
    check(len(distractors) >= 1, "hard tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"z = InvNorm({q_eff:.4f}) = {z:.4f}. x = mu + z × sigma = {mu} + "
            f"{paren_negative(round(z, 4))} × {sigma} = {answer_text}."
        ],
        distractors=distractors,
    )


# =============================================================================
# exam: given p and a threshold a, solve for the missing parameter mu/sigma
# =============================================================================


def _build_exam(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        scenario = rng.choice(_SCENARIOS)
        solve_for = rng.choice(["mu", "sigma"])
        mu, sigma = _draw_mu_sigma(rng, scenario)
        if sigma <= 0:
            continue
        # p away from 0.5 so z is not near zero (avoids a degenerate/unstable
        # division when solving for sigma).
        p = round(rng.uniform(0.02, 0.35), 2)
        if rng.choice([True, False]):
            p = round(1.0 - p, 2)

        z = NormalDist(0.0, 1.0).inv_cdf(p)
        if abs(z) < 0.3:
            continue
        a = round(mu + z * sigma, 1)  # the STATED threshold consistent with the true mu/sigma

        if solve_for == "sigma":
            solved_val = (a - mu) / z
            if solved_val <= 0 or not _sigma_in_domain(solved_val, scenario["sigma_hi"]):
                continue
            check_nd_kwargs = dict(mu=mu, sigma=solved_val)
        else:  # solve_for == "mu"
            solved_val = a - sigma * z
            if not _x_in_domain(solved_val, scenario):
                continue
            check_nd_kwargs = dict(mu=solved_val, sigma=sigma)

        if _near_rounding_boundary(solved_val, _DP_PARAM):
            continue
        ans = _round(solved_val, _DP_PARAM)

        z_only_val = z  # reports z, not the solved parameter (a named misconception)
        # "wrong tail" (a named misconception): uses inv_cdf(1-p) instead
        # of inv_cdf(p). Standard-normal symmetry makes inv_cdf(1-p) == -z
        # exactly, so algebraically this collapses to a pure sign flip of the
        # solved value for the mu branch -- a genuine mathematical fact, not a coding
        # slip. For the SIGMA branch the sign-flip formula (-solved_val, always a
        # negative standard deviation -- physically impossible) is replaced with
        # forgot_to_divide_by_z: reports abs(a - mu) directly, never dividing
        # by z at all -- always positive, a structurally different (and
        # still principled) slip. A second, STRUCTURALLY different mistake
        # (swapping multiplication for division when isolating the unknown)
        # supplies the third distractor instead of a redundant "sign error"
        # label.
        if solve_for == "sigma":
            wrong_tail_val = abs(a - mu)  # forgot_to_divide_by_z, always positive
            op_flip_val = (a - mu) * z  # multiplied instead of divided
        else:
            wrong_tail_val = -solved_val + 2 * a  # a - sigma*inv_cdf(1-p) == a + sigma*z
            op_flip_val = a - sigma / z  # divided instead of multiplied

        # z_only_val's TYPE is a dimensionless z-score (the named
        # "reports_z_not_x" misconception), not a sigma-typed or x-typed
        # value in the scenario's own units -- the sigma/x-domain predicate
        # is a category error for it, exactly the same reasoning as the hard
        # tier above. Its own domain (a standard-normal quantile, |z| >= 0.3
        # by the earlier `abs(z) < 0.3: continue` guard) is satisfied by
        # construction. wrong_tail_val and op_flip_val ARE genuinely
        # sigma-typed/x-typed (both still solve for the same physical
        # quantity as `solved_val`) and get the full domain check.
        typed_cands = [wrong_tail_val, op_flip_val]
        if solve_for == "sigma":
            domain_ok = all(_sigma_in_domain(v, scenario["sigma_hi"]) for v in typed_cands)
        else:
            domain_ok = all(_x_in_domain(v, scenario) for v in typed_cands)
        cand_floats = [z_only_val, wrong_tail_val, op_flip_val]
        if not domain_ok or any(_near_rounding_boundary(v, _DP_PARAM) for v in cand_floats):
            continue
        cands = [_round(v, _DP_PARAM) for v in cand_floats]
        if not _gap_ok([ans] + cands, _TOL_PARAM):
            continue
        break
    else:
        raise ValueError("p13d_normal_dist_prob_quantile: no collision-free exam draw found")

    # Independent check: re-substitute the solved parameter back into
    # NormalDist(mu, sigma).cdf(a) and confirm it reproduces the STATED
    # proportion p -- a different check from the algebra that solved for it.
    reproduced_p = NormalDist(**check_nd_kwargs).cdf(a)
    check(abs(reproduced_p - p) < 1e-6, "exam: re-substituted parameter does not reproduce the stated proportion p")

    unknown_label = "standard deviation" if solve_for == "sigma" else "mean"
    known_label = "mean" if solve_for == "sigma" else "standard deviation"
    known_val = mu if solve_for == "sigma" else sigma
    text = (
        f"{scenario['subject'].capitalize()} is normally distributed with unknown {unknown_label} and "
        f"{known_label} {known_val} {scenario['unit']}. It is known that {p * 100:g}% of {scenario['plural']} "
        f"have a value below {a} {scenario['unit']}. Find the {unknown_label}, in {scenario['unit']}. "
        f"Give your answer correct to 2 decimal places."
    )
    latex = text
    answer_text = _fmt(ans, _DP_PARAM)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_PARAM) for c in cands], n=3)
    check(len(distractors) >= 1, "exam tier expects at least 1 distinct distractor")

    formula = (
        f"sigma = (a - mu) / z = ({a} - {mu}) / {z:.4f} = {answer_text}."
        if solve_for == "sigma"
        else f"mu = a - sigma × z = {a} - {sigma} × {z:.4f} = {answer_text}."
    )
    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"z = InvNorm({p:.4f}) = {z:.4f}. {formula}"],
        distractors=distractors,
    )


@generator(
    "p13d_normal_dist_prob_quantile",
    level=13,
    presets={
        "easy": dict(mode="one_tail"),
        "medium": dict(mode="between"),
        "hard": dict(mode="quantile"),
        "exam": dict(mode="solve_parameter"),
    },
)
def p13d_normal_dist_prob_quantile(rng, mode="one_tail"):
    """N(mu, sigma^2), computed with statistics.NormalDist. easy: one-tail
    probability P(X<a)/P(X>b). medium: between two values P(a<X<b). hard:
    inverse -- quantile x from a stated cumulative proportion q. exam: given
    p and a threshold a (with one of mu/sigma unstated), solve for the
    missing parameter."""
    if mode == "one_tail":
        return _build_easy(rng)
    if mode == "between":
        return _build_medium(rng)
    if mode == "quantile":
        return _build_hard(rng)
    if mode == "solve_parameter":
        return _build_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
