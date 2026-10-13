"""Generates `fy1ms_z_test_proportion` questions and the exam chain `exd_x_fy1ms_z_test_proportion`.

Skill: `ucv1-proportion-z-test` (statistics). The arithmetic of a one-sample z test for a population proportion:
the standard error under H0, the test statistic, a p-value, and the rejection boundary. The confidence interval for a
proportion is not covered here, because the senior skills on the approximate confidence interval for a proportion and
`ucv1-proportion-confidence-interval` already teach it.

Every question states that the sample is large enough for the normal approximation, and the standard error always
uses the hypothesised proportion p0, never the sample proportion: that is the point of the test.

Tiers (topic `fy1ms_z_test_proportion`):

* easy   -- the standard error sqrt(p0(1 - p0)/n) to 4 places, or the difference p-hat - p0 between the sample
            proportion (from a count) and the hypothesised proportion;
* medium -- the z statistic (p-hat - p0)/sqrt(p0(1 - p0)/n) to 2 places, from a count or from a sample percentage;
* hard   -- one of: the p-value for a stated z and alternative hypothesis (two-sided, greater or less, so the tail
            matters), to 4 places; the sample proportion that marks the edge of the rejection region at the 5% level;
            the smallest number of successes that rejects H0.

Exam (topic `exd_x_fy1ms_z_test_proportion`): (a) the standard error under H0 (1 mark), (b) the z statistic (2 marks,
follow-through from (a)), (c) the two-sided p-value to 3 places (2 marks, follow-through from (b)), (d) the
conclusion at the 5% level as a one-mark multiple-choice part.

Every key is recomputed a second way: the standard error and statistic with mpmath at 40 digits from the raw inputs,
and every normal tail with `math.erfc` (a different function from the `mpmath.ncdf` the first route uses).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import mpmath

from ..core import attempts, check, generator
from ._fy1ms_common import (
    Redraw,
    build_until,
    choice_part,
    clear_of_tie,
    dp_text,
    ft_close,
    num_dist,
    number_part,
)

# ---------------------------------------------------------------------------
# Exact and numeric helpers


def _root(x: Fraction, digits: int = 30) -> Fraction:
    """sqrt(x) to `digits` places as a Fraction (truncated, so the error is below 10**-digits)."""
    scaled = x.numerator * 10 ** (2 * digits) // x.denominator
    return Fraction(math.isqrt(scaled), 10**digits)


def _tail(z: Fraction) -> Fraction:
    """P(Z > z) for the standard normal, as an exact decimal Fraction (mpmath route)."""
    with mpmath.workdps(40):
        value = 1 - mpmath.ncdf(mpmath.mpf(z.numerator) / z.denominator)
        other = mpmath.mpf(math.erfc(float(z) / math.sqrt(2)) / 2)
        check(abs(value - other) < mpmath.mpf("1e-12"), "the two normal-tail routes disagree")
        return Fraction(mpmath.nstr(value, 30))


def _mp_check(actual: Fraction, expected, what: str) -> None:
    a = mpmath.mpf(actual.numerator) / actual.denominator
    check(abs(a - expected) <= mpmath.mpf("1e-12") * max(1, abs(expected)), f"{what}: mpmath route disagrees")


def _places(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def _dec(f: Fraction, dp: int) -> str:
    return dp_text(f, dp)


# ---------------------------------------------------------------------------
# Contexts: a claimed proportion (as a decimal with two places) and a population of sampled things.

_CONTEXTS = [
    dict(
        lead="A factory claims that {pct}% of the components it makes are faulty.",
        p0=("0.04", "0.05", "0.06", "0.08", "0.10", "0.12"),
        sample="components",
        event="were faulty",
        perf="been faulty",
    ),
    dict(
        lead="A council report says that {pct}% of residents support a new bike lane.",
        p0=("0.35", "0.40", "0.45", "0.55", "0.60", "0.65"),
        sample="residents",
        event="supported the bike lane",
        perf="supported the bike lane",
    ),
    dict(
        lead="A seed company states that {pct}% of its tomato seeds germinate.",
        p0=("0.75", "0.80", "0.85", "0.90"),
        sample="seeds",
        event="germinated",
        perf="germinated",
    ),
    dict(
        lead="A courier company promises that {pct}% of its parcels arrive on time.",
        p0=("0.85", "0.88", "0.90", "0.92"),
        sample="parcels",
        event="arrived on time",
        perf="arrived on time",
    ),
    dict(
        lead="A coin is said to land heads with probability {p0}.",
        p0=("0.40", "0.50", "0.60"),
        sample="tosses",
        event="landed heads",
        perf="landed heads",
    ),
    dict(
        lead="A university says that {pct}% of its graduates find work within four months.",
        p0=("0.60", "0.65", "0.70", "0.75", "0.80"),
        sample="graduates",
        event="found work within four months",
        perf="found work within four months",
    ),
]

_SIZES = (100, 120, 125, 150, 200, 250, 300, 400, 500, 600)
_PERCENT_SIZES = (100, 200, 250, 500)  # sizes for which a sample percentage has at most one decimal place


class _Test:
    """A one-sample proportion test: hypothesised proportion, sample size, success count, and what follows."""

    def __init__(self, ctx, p0: Fraction, n: int, x: int):
        self.ctx, self.p0, self.n, self.x = ctx, p0, n, x
        self.q0 = 1 - p0
        self.phat = Fraction(x, n)
        self.var0 = p0 * self.q0 / n
        self.se = _root(self.var0)
        self.diff = self.phat - p0
        self.z = self.diff / self.se
        with mpmath.workdps(40):
            p, nn = mpmath.mpf(p0.numerator) / p0.denominator, mpmath.mpf(n)
            se = mpmath.sqrt(p * (1 - p) / nn)
            _mp_check(self.se, se, "standard error")
            _mp_check(self.z, (mpmath.mpf(x) / nn - p) / se, "z statistic")

    @property
    def p0_text(self) -> str:
        return _dec(self.p0, 2)

    def lead(self) -> str:
        pct = _dec(self.p0 * 100, 0)
        return self.ctx["lead"].format(pct=pct, p0=self.p0_text)

    def sample_sentence(self, *, percent: bool = False) -> str:
        c = self.ctx
        if percent:
            return (
                f"In a random sample of {self.n} {c['sample']}, {_dec(self.phat * 100, 1)}% {c['event']}. "
                f"The sample is large enough for the normal approximation to be used."
            )
        return (
            f"In a random sample of {self.n} {c['sample']}, {self.x} {c['event']}. "
            f"The sample is large enough for the normal approximation to be used."
        )

    def hypotheses(self, alt: str = "two") -> str:
        sign = {"two": "≠", "greater": ">", "less": "<"}[alt]
        return (
            f"Let p be the population proportion and test H0: p = {self.p0_text} against H1: p {sign} {self.p0_text}."
        )


def _draw(rng: random.Random, *, z_range, percent: bool = False, signed: bool = True) -> _Test:
    """A test whose z statistic is about a value in `z_range` (signed at random), with a valid normal approximation."""
    for _ in attempts(500, "a proportion z test"):
        ctx = rng.choice(_CONTEXTS)
        p0 = Fraction(rng.choice(ctx["p0"]))
        n = rng.choice(_PERCENT_SIZES if percent else _SIZES)
        if min(n * p0, n * (1 - p0)) < 10:
            continue
        se = _root(p0 * (1 - p0) / n)
        z = Fraction(rng.randint(int(z_range[0] * 100), int(z_range[1] * 100)), 100)
        if signed and rng.random() < 0.5:
            z = -z
        x = round(n * p0 + z * n * se)
        if not 0 < x < n or x == n * p0:
            continue
        return _Test(ctx, p0, n, x)
    raise AssertionError("unreachable")  # pragma: no cover


def _question(text: str, value, dp: int, cands, working) -> dict:
    part = number_part("", value, dp, cands, working=working)
    return dict(
        text=text,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="number",
        working=part["working"],
        distractors=part["distractors"],
    )


# ---------------------------------------------------------------------------
# Candidate wrong answers


def _se_cands(t: _Test, dp: int):
    p0, q0, n = t.p0, t.q0, t.n
    return [
        num_dist(_root(t.phat * (1 - t.phat) / n), dp, "ztest-se-uses-sample-proportion"),
        num_dist(p0 * q0 / n, dp, "ztest-se-no-root"),
        num_dist(_root(p0 / n), dp, "ztest-se-complement-forgotten"),
        num_dist(_root(p0 * q0) / n, dp, "ztest-se-divides-by-n"),
        num_dist(_root(p0 * q0), dp, "ztest-se-root-n-forgotten"),
    ]


def _z_cands(t: _Test, dp: int, *, percent: bool):
    p0, n = t.p0, t.n
    out = [
        num_dist(-t.z, dp, "ztest-sign-reversed"),
        num_dist(t.diff / _root(t.phat * (1 - t.phat) / n), dp, "ztest-se-uses-sample-proportion"),
        num_dist(t.diff / (p0 * t.q0 / n), dp, "ztest-se-no-root"),
        num_dist(t.diff / _root(p0 * t.q0), dp, "ztest-se-root-n-forgotten"),
        num_dist(t.diff / _root(p0 / n), dp, "ztest-se-complement-forgotten"),
    ]
    if percent:
        out.append(num_dist((t.phat * 100 - p0) / t.se, dp, "ztest-percent-not-converted"))
    else:
        out.append(num_dist((t.x - p0) / t.se, dp, "ztest-count-used-as-proportion"))
    return out


# ---------------------------------------------------------------------------
# Easy


def _easy_se(rng: random.Random) -> dict:
    t = _draw(rng, z_range=(0.5, 3.0))
    dp = 4
    if not clear_of_tie(t.se, dp):
        raise Redraw("standard error on a rounding tie")
    text = (
        f"{t.lead()} {t.sample_sentence()} {t.hypotheses()} "
        f"Find the standard error of the sample proportion when H0 is true, correct to {_places(dp)}."
    )
    return _question(
        text,
        t.se,
        dp,
        _se_cands(t, dp),
        [f"SE = √(p0(1 − p0)/n) = √({t.p0_text} × {_dec(t.q0, 2)}/{t.n}) = {_dec(t.se, dp)}."],
    )


def _easy_diff(rng: random.Random) -> dict:
    t = _draw(rng, z_range=(0.5, 3.0))
    dp = 4
    if t.diff == 0 or not clear_of_tie(t.diff, dp):
        raise Redraw("difference unsuitable")
    cands = [
        num_dist(-t.diff, dp, "ztest-sign-reversed"),
        num_dist(Fraction(t.x) - t.p0, dp, "ztest-count-used-as-proportion"),
        num_dist(t.phat, dp, "ztest-hypothesised-proportion-ignored"),
        num_dist(Fraction(t.x, t.n + t.x) - t.p0, dp, "ztest-sample-proportion-wrong-denominator"),
    ]
    text = (
        f"{t.lead()} {t.sample_sentence()} {t.hypotheses()} "
        f"Find the difference p̂ − p0 between the sample proportion and the hypothesised proportion, "
        f"correct to {_places(dp)}."
    )
    return _question(
        text,
        t.diff,
        dp,
        cands,
        [f"p̂ = {t.x}/{t.n} = {_dec(t.phat, 4)}, so p̂ − p0 = {_dec(t.phat, 4)} − {t.p0_text} = {_dec(t.diff, dp)}."],
    )


def _easy(rng: random.Random) -> dict:
    return _easy_se(rng) if rng.random() < 0.65 else _easy_diff(rng)


# ---------------------------------------------------------------------------
# Medium


def _medium(rng: random.Random) -> dict:
    percent = rng.random() < 0.4
    t = _draw(rng, z_range=(0.5, 3.8), percent=percent)
    dp = 2
    if t.diff == 0 or not clear_of_tie(t.z, dp) or abs(t.z) < Fraction(1, 5):
        raise Redraw("z statistic unsuitable")
    alt = "two" if rng.random() < 0.4 else ("greater" if t.z > 0 else "less")
    text = (
        f"{t.lead()} {t.sample_sentence(percent=percent)} {t.hypotheses(alt)} "
        f"Calculate the value of the test statistic z = (p̂ − p0)/√(p0(1 − p0)/n), correct to {_places(dp)}."
    )
    return _question(
        text,
        t.z,
        dp,
        _z_cands(t, dp, percent=percent),
        [
            f"p̂ = {_dec(t.phat, 4)}, so p̂ − p0 = {_dec(t.diff, 4)}.",
            f"SE = √({t.p0_text} × {_dec(t.q0, 2)}/{t.n}) = {_dec(t.se, 5)}, so z = {_dec(t.z, dp)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Hard


def _hard_p_value(rng: random.Random) -> dict:
    dp = 4
    alt = rng.choice(("two", "greater", "less", "greater", "less"))
    z = Fraction(rng.randint(60, 340), 100) * rng.choice((1, -1))
    if alt == "two":
        value = 2 * _tail(abs(z))
        cands = [
            num_dist(_tail(abs(z)), dp, "ztest-pvalue-doubling-forgotten"),
            num_dist(1 - _tail(abs(z)), dp, "ztest-pvalue-cdf-as-pvalue"),
            num_dist(1 - 2 * _tail(abs(z)), dp, "ztest-pvalue-complement"),
        ]
    else:
        upper = _tail(z)  # P(Z > z), correct for "greater"
        lower = 1 - upper  # P(Z < z), correct for "less"
        value = upper if alt == "greater" else lower
        wrong_tail = lower if alt == "greater" else upper
        cands = [
            num_dist(wrong_tail, dp, "ztest-pvalue-wrong-tail"),
            num_dist(2 * min(upper, lower), dp, "ztest-pvalue-doubled-for-one-sided"),
            num_dist(_tail(abs(z)), dp, "ztest-pvalue-sign-ignored"),
        ]
    if not clear_of_tie(value, dp, Fraction(1, 5)):
        raise Redraw("p-value on a rounding tie")
    sign = {"two": "≠", "greater": ">", "less": "<"}[alt]
    ctx = rng.choice(_CONTEXTS)
    p0 = Fraction(rng.choice(ctx["p0"]))
    text = (
        f"{ctx['lead'].format(pct=_dec(p0 * 100, 0), p0=_dec(p0, 2))} A test of H0: p = {_dec(p0, 2)} against "
        f"H1: p {sign} {_dec(p0, 2)} on a large random sample gives the test statistic z = {_dec(z, 2)}. "
        f"Using the standard normal distribution, find the p-value, correct to {_places(dp)}."
    )
    rule = {
        "two": "P(|Z| > |z|) = 2 × P(Z > |z|)",
        "greater": "P(Z > z)",
        "less": "P(Z < z)",
    }[alt]
    return _question(
        text, value, dp, cands, [f"The p-value is {rule} with z = {_dec(z, 2)}, which is {_dec(value, dp)}."]
    )


def _hard_cutoff(rng: random.Random) -> dict:
    t = _draw(rng, z_range=(0.5, 3.0))
    dp = 4
    zstar = Fraction(196, 100)
    cut = t.p0 + zstar * t.se
    if cut >= 1 or not clear_of_tie(cut, dp):
        raise Redraw("cut-off above 1 or on a rounding tie")
    with mpmath.workdps(40):
        p = mpmath.mpf(t.p0.numerator) / t.p0.denominator
        _mp_check(cut, p + mpmath.mpf("1.96") * mpmath.sqrt(p * (1 - p) / t.n), "rejection boundary")
    cands = [
        num_dist(t.p0 - zstar * t.se, dp, "ztest-cutoff-wrong-side"),
        num_dist(zstar * t.se, dp, "ztest-cutoff-margin-as-bound"),
        num_dist(t.p0 + Fraction(1645, 1000) * t.se, dp, "ztest-wrong-critical-value"),
        num_dist(t.p0 + zstar * _root(t.phat * (1 - t.phat) / t.n), dp, "ztest-se-uses-sample-proportion"),
        num_dist(t.p0 + zstar * t.p0 * t.q0 / t.n, dp, "ztest-se-no-root"),
    ]
    text = (
        f"{t.lead()} {t.hypotheses('greater')} A random sample of {t.n} {t.ctx['sample']} is taken, which is large "
        f"enough for the normal approximation. H0 is rejected at the 5% level of significance when the z statistic "
        f"is greater than 1.96. Find the smallest value of the sample proportion p̂ at which H0 would be rejected, "
        f"correct to {_places(dp)}."
    )
    return _question(
        text,
        cut,
        dp,
        cands,
        [
            f"SE = √({t.p0_text} × {_dec(t.q0, 2)}/{t.n}) = {_dec(t.se, 5)}.",
            f"p̂ = p0 + 1.96 × SE = {t.p0_text} + 1.96 × {_dec(t.se, 5)} = {_dec(cut, dp)}.",
        ],
    )


def _hard_count(rng: random.Random) -> dict:
    t = _draw(rng, z_range=(0.5, 3.0))
    zstar = Fraction(196, 100)
    edge = t.n * (t.p0 + zstar * t.se)
    if edge >= t.n - 1 or not Fraction(1, 5) < edge - math.floor(edge) < Fraction(4, 5):
        raise Redraw("boundary count too close to a whole number")
    answer = math.floor(edge) + 1
    cands = [
        num_dist(math.floor(edge), 0, "ztest-count-rounded-down"),
        num_dist(math.floor(edge) + 2, 0, "ztest-count-rounded-up-twice"),
        num_dist(math.ceil(t.n * (t.p0 - zstar * t.se)), 0, "ztest-cutoff-wrong-side"),
        num_dist(math.floor(t.n * (t.p0 + Fraction(1645, 1000) * t.se)) + 1, 0, "ztest-wrong-critical-value"),
        num_dist(
            math.floor(t.n * (t.p0 + zstar * _root(t.phat * (1 - t.phat) / t.n))) + 1,
            0,
            "ztest-se-uses-sample-proportion",
        ),
    ]
    text = (
        f"{t.lead()} {t.hypotheses('greater')} A random sample of {t.n} {t.ctx['sample']} is taken, which is large "
        f"enough for the normal approximation. H0 is rejected at the 5% level of significance when the z statistic "
        f"is greater than 1.96. What is the smallest number of the {t.n} {t.ctx['sample']} that must have "
        f"{t.ctx['perf']} for H0 to be rejected?"
    )
    question = number_part(
        "",
        answer,
        0,
        cands,
        working=[
            f"SE = √({t.p0_text} × {_dec(t.q0, 2)}/{t.n}) = {_dec(t.se, 5)}, so the count must exceed {t.n} × ({t.p0_text} + 1.96 × {_dec(t.se, 5)}) = {_dec(edge, 2)}.",
            f"The smallest whole number above {_dec(edge, 2)} is {answer}.",
        ],
    )
    return dict(
        text=text,
        answer=question["answer"],
        answer_text=question["answer_text"],
        answer_kind="number",
        working=question["working"],
        distractors=question["distractors"],
    )


@generator(
    "fy1ms_z_test_proportion",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_z_test_proportion(rng: random.Random, tier: str = "easy") -> dict:
    """One-sample z test for a proportion: a standard error (easy), the statistic (medium), a p-value or boundary (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy proportion test question")
    if tier == "medium":
        return build_until(rng, _medium, "a medium proportion test question")
    if tier == "hard":

        def hard(r: random.Random) -> dict:
            pick = r.random()
            return _hard_p_value(r) if pick < 0.4 else _hard_cutoff(r) if pick < 0.7 else _hard_count(r)

        return build_until(rng, hard, "a hard proportion test question")
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain


@generator("exd_x_fy1ms_z_test_proportion", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_z_test_proportion(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: standard error, z statistic, two-sided p-value, conclusion (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a proportion z-test exam question")


def _exam_chain(rng: random.Random) -> dict:
    reject = rng.random() < 0.5
    t = _draw(rng, z_range=(2.1, 3.6) if reject else (0.5, 1.7))
    dp_se, dp_z, dp_p = 4, 2, 3
    p_two = 2 * _tail(abs(t.z))
    for value, dp in ((t.se, dp_se), (t.z, dp_z), (p_two, dp_p)):
        if not clear_of_tie(value, dp):
            raise Redraw("a value sits on a rounding tie")
    if t.diff == 0 or abs(p_two - Fraction(1, 20)) < Fraction(1, 100):
        raise Redraw("p-value too close to 0.05")
    se_text, z_text, p_text = _dec(t.se, dp_se), _dec(t.z, dp_z), _dec(p_two, dp_p)
    ft_close(t.diff / Fraction(se_text), z_text, dp_z)
    ft_close(2 * _tail(abs(Fraction(z_text))), p_text, dp_p)
    se_cands = _se_cands(t, dp_se)
    z_cands = [
        num_dist(-t.z, dp_z, "ztest-sign-reversed"),
        num_dist(t.diff / _root(t.phat * (1 - t.phat) / t.n), dp_z, "ztest-se-uses-sample-proportion"),
        num_dist(t.diff / (t.p0 * t.q0 / t.n), dp_z, "ztest-se-no-root"),
        num_dist(t.diff / _root(t.p0 * t.q0), dp_z, "ztest-se-root-n-forgotten"),
        num_dist((t.x - t.p0) / t.se, dp_z, "ztest-count-used-as-proportion"),
    ]
    zz = abs(Fraction(z_text))
    p_cands = [
        num_dist(_tail(zz), dp_p, "ztest-pvalue-doubling-forgotten"),
        num_dist(1 - _tail(zz), dp_p, "ztest-pvalue-cdf-as-pvalue"),
        num_dist(1 - 2 * _tail(zz), dp_p, "ztest-pvalue-complement"),
    ]
    right = (
        "Reject H0, because the p-value is less than 0.05."
        if reject
        else "Do not reject H0, because the p-value is greater than 0.05."
    )
    reversed_ = (
        "Do not reject H0, because the p-value is less than 0.05."
        if reject
        else "Reject H0, because the p-value is greater than 0.05."
    )
    wrongs = [
        (reversed_, "ztest-conclusion-decision-reversed"),
        (
            "Reject H0, because the sample proportion is not equal to the hypothesised proportion.",
            "ztest-conclusion-sample-differs",
        ),
    ]
    stem = f"{t.lead()} {t.sample_sentence()} Test H0: p = {t.p0_text} against H1: p ≠ {t.p0_text} at the 5% level of significance."
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the standard error of the sample proportion when H0 is true, correct to 4 decimal places.",
                t.se,
                dp_se,
                se_cands,
                marks=1,
                sensible_range=(0, 1),
                working=[f"SE = √({t.p0_text} × {_dec(t.q0, 2)}/{t.n}) = {se_text}."],
            ),
            number_part(
                "Calculate the value of the test statistic z, correct to 2 decimal places.",
                t.z,
                dp_z,
                z_cands,
                marks=2,
                follow_through=f"({t.diff.numerator}/{t.diff.denominator})/part_a",
                working=[f"z = (p̂ − p0)/SE = ({t.x}/{t.n} − {t.p0_text})/{se_text} = {z_text}."],
            ),
            number_part(
                "Hence find the two-sided p-value, correct to 3 decimal places.",
                p_two,
                dp_p,
                p_cands,
                marks=2,
                follow_through="erfc(Abs(part_b)/sqrt(2))",
                sensible_range=(0, 1),
                working=[f"p-value = 2 × P(Z > {_dec(abs(t.z), 2)}) = {p_text}."],
            ),
            choice_part(
                rng,
                "Using the p-value, choose the conclusion of the test at the 5% level.",
                right,
                wrongs,
                marks=1,
                working=[f"The p-value {p_text} is {'less' if reject else 'greater'} than 0.05."],
            ),
        ],
    )
