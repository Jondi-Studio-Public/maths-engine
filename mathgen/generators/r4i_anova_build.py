"""Generates `r4i_anova` questions (easy, medium, hard) and the exam chain `exd_x_r4i_anova`.

``uc2-one-way-anova-table`` (``statistics``, answer ``number``, stated to 2 or 3 decimal places or a whole number): the
pieces of a one-way analysis of variance table, built from group sizes, means and standard deviations or from the raw
observations. A critical value, when one is needed, is stated in the question (and never recalled).

* easy: the between-groups sum of squares SSB for three equal groups; the pooled variance MSW from the group variances
  (unequal sizes); or SSB for unequal sizes (the grand mean is the size-weighted mean).
* medium: the F statistic from group sizes, means and standard deviations (2 dp); the proportion of the total variation
  explained by the groups, SSB / (SSB + SSW) (4 dp); or the mean square between groups MSB from raw group totals.
* hard: F from the raw observations of three small groups (2 dp); the missing SSB in a partly filled table, solved from
  F, the total sum of squares and the degrees of freedom; the half-width of a confidence interval for the difference of two
  group means using the pooled variance and a stated t value (3 dp); or the smallest SSB that would lift F to a stated
  critical value.
* exam: three groups with sizes, means and standard deviations. (a) SSB (2 dp), (b) SSW (2 dp), (c) F (2 dp, follows
  through from (a) and (b)), (d) the half-width of a confidence interval for the difference between the first two means,
  using the pooled variance SSW / (N - 3) (3 dp, follows through from (b)).

Every quantity is computed in exact fractions; the tests recompute them from the numbers in the text, from the raw
observations when given, and by the identity SST = SSB + SSW.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import (
    chain_avoid,
    clean,
    dec,
    need,
    num_distractors,
    num_fields,
    redraw,
    retry,
    stable,
    stored,
)

TOPIC = "r4i_anova"
EXAM = "exd_x_r4i_anova"

T_CRIT = {
    9: "2.262", 10: "2.228", 12: "2.179", 14: "2.145", 15: "2.131", 16: "2.120", 18: "2.101", 20: "2.086",
    21: "2.080", 22: "2.074", 24: "2.064", 25: "2.060", 27: "2.052", 30: "2.042",
}  # fmt: skip

SETTINGS = [
    ("three wheat varieties grown on farms near Esperance", "yield", "t/ha", ("Variety A", "Variety B", "Variety C")),
    ("three training programs used by a Brisbane call centre", "calls handled per hour", "calls", ("Program 1", "Program 2", "Program 3")),
    ("three brands of battery tested in torches", "running time", "hours", ("Brand X", "Brand Y", "Brand Z")),
    ("three teaching methods trialled in Tasmanian schools", "test score", "marks", ("Method 1", "Method 2", "Method 3")),
]  # fmt: skip


def _ft(v) -> str:
    v = Fraction(v)
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _g(v) -> str:
    """A group statistic written plainly: `12`, `12.5`."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{float(v):g}"


def anova(ns, means, vars_):
    """(SSB, SSW, MSB, MSW, F, grand mean) in exact fractions from sizes, means and variances."""
    ns = [Fraction(n) for n in ns]
    total = sum(ns)
    grand = sum(n * m for n, m in zip(ns, means)) / total
    ssb = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ssw = sum((n - 1) * v for n, v in zip(ns, vars_))
    k = len(ns)
    msb, msw = ssb / (k - 1), ssw / (total - k)
    return ssb, ssw, msb, msw, msb / msw, grand


def _group_table(names, ns, means, sds, unit):
    rows = [f"{nm}: n = {n}, mean {_g(m)}, standard deviation {_g(s)}" for nm, n, m, s in zip(names, ns, means, sds)]
    return "; ".join(rows)


def _draw_groups(rng, equal=False, integer_grand=False):
    for _ in attempts(500, "groups with a clean grand mean"):
        k = 3
        n = rng.choice((5, 6, 8, 10))
        ns = [n] * k if equal else [rng.choice((5, 6, 8, 10, 12)) for _ in range(k)]
        base = rng.choice((20, 30, 40, 50, 60))
        means = [base + rng.randint(-8, 8) for _ in range(k)]
        sds = [rng.choice((2, 3, 4, 5, 6)) for _ in range(k)]
        if len(set(means)) < 3:
            continue
        total = sum(ns)
        grand = Fraction(sum(a * b for a, b in zip(ns, means)), total)
        if integer_grand and (grand.denominator != 1):
            continue
        if not integer_grand and grand.denominator > 12:
            continue
        return ns, means, sds
    raise AssertionError("unreachable")


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_ssb_equal(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    ns, means, sds = _draw_groups(rng, equal=True, integer_grand=True)
    n = ns[0]
    grand = Fraction(sum(means), 3)
    ssb = n * sum((m - grand) ** 2 for m in means)
    check(ssb.denominator == 1 and ssb > 0, "SSB not whole")
    sdsum = sum((m - grand) ** 2 for m in means)
    cands = [
        (sdsum, "r4i-av-forgot-group-size"),
        (ssb / 2, "r4i-av-divided-by-df"),
        (ssb / 3, "r4i-av-divided-by-k"),
        (n * sum((m - means[0]) ** 2 for m in means), "r4i-av-wrong-centre"),
        (ssb * n, "r4i-av-size-twice"),
    ]
    dist = need(num_distractors(float(ssb), cands, 0))
    text = (
        f"For each of {setting}, {n} units were measured for {qty} ({unit}). The group means are "
        f"{', '.join(f'{nm} {_g(m)}' for nm, m in zip(names, means))}. Find the between-groups sum of squares SSB."
    )
    working = [
        f"The grand mean is ({' + '.join(str(m) for m in means)})/3 = {_g(grand)}.",
        f"SSB = {n}×Σ(group mean - grand mean)² = {n}×{_g(sdsum)} = {_g(ssb)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(ssb), 0))


def _easy_msw(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(300, "a pooled variance with at most 2 decimals"):
        ns = [rng.choice((5, 6, 8, 11)) for _ in range(3)]
        sds = [rng.choice((2, 3, 4, 5)) for _ in range(3)]
        total = sum(ns)
        msw = Fraction(sum((n - 1) * s * s for n, s in zip(ns, sds)), total - 3)
        if (msw * 100).denominator == 1 and len({*ns}) > 1 and len({*sds}) == 3:
            break
    cands = [
        (Fraction(sum(s * s for s in sds), 3), "r4i-av-unweighted-variances"),
        (Fraction(sum((n - 1) * s * s for n, s in zip(ns, sds)), total), "r4i-av-divided-by-total-n"),
        (Fraction(sum(n * s * s for n, s in zip(ns, sds)), total - 3), "r4i-av-n-for-n-minus-one"),
        (math.sqrt(float(msw)), "r4i-av-sd-for-variance"),
        (Fraction(sum((n - 1) * s * s for n, s in zip(ns, sds)), total - 1), "r4i-av-wrong-df"),
    ]
    dist = need(num_distractors(float(msw), cands, 2))
    text = (
        f"The {qty} ({unit}) was measured for {setting}. "
        f"{'; '.join(f'{nm}: n = {n}, standard deviation {s}' for nm, n, s in zip(names, ns, sds))}. "
        f"Find the pooled estimate of the common variance, the within-groups mean square MSW."
    )
    working = [
        f"SSW = Σ(nᵢ - 1)sᵢ² = {_g(Fraction(sum((n - 1) * s * s for n, s in zip(ns, sds))))}.",
        f"The degrees of freedom are N - k = {total} - 3 = {total - 3}, so MSW = {dec(float(msw), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(msw), 2))


def _easy_ssb_unequal(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(400, "unequal groups with a whole SSB"):
        ns, means, sds = _draw_groups(rng, equal=False)
        if len(set(ns)) < 2:
            continue
        ssb, _, _, _, _, grand = anova(ns, means, [1, 1, 1])
        if (ssb * 100).denominator == 1 and ssb > 0:
            break
    unweighted = Fraction(sum(means), 3)
    cands = [
        (sum(n * (m - unweighted) ** 2 for n, m in zip(ns, means)), "r4i-av-unweighted-grand-mean"),
        (sum((m - grand) ** 2 for m in means), "r4i-av-forgot-group-size"),
        (sum((n - 1) * (m - grand) ** 2 for n, m in zip(ns, means)), "r4i-av-n-minus-one-for-n"),
        (ssb / 2, "r4i-av-divided-by-df"),
        (sum(n * abs(m - grand) for n, m in zip(ns, means)), "r4i-av-absolute-not-squared"),
    ]
    dist = need(num_distractors(float(ssb), cands, 2))
    text = (
        f"The {qty} ({unit}) was measured for {setting}. The group sizes and means are "
        f"{', '.join(f'{nm}: n = {n}, mean {_g(m)}' for nm, n, m in zip(names, ns, means))}. Find the between-groups sum of "
        f"squares SSB, correct to 2 decimal places."
    )
    working = [
        f"The grand mean is the size-weighted mean, {_ft(grand)} = {float(grand):.4f}.",
        f"SSB = Σ nᵢ(group mean - grand mean)² = {dec(float(ssb), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(ssb), 2))


def _easy(rng):
    return rng.choice((_easy_ssb_equal, _easy_msw, _easy_ssb_unequal))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_f(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(500, "an F statistic away from a rounding boundary"):
        ns, means, sds = _draw_groups(rng)
        ssb, ssw, msb, msw, f, grand = anova(ns, means, [s * s for s in sds])
        if stable(float(f), 2, 0.12) and 0.3 < float(f) < 60:
            break
    N = sum(ns)
    cands = [
        (1 / f, "r4i-av-f-inverted"),
        (ssb / ssw, "r4i-av-ss-not-ms"),
        ((ssb / (N - 3)) / (ssw / 2), "r4i-av-df-swapped"),
        (msb / (ssw / N), "r4i-av-wrong-df"),
        (msb / (Fraction(sum(s * s for s in sds), 3)), "r4i-av-unweighted-variances"),
    ]
    dist = need(num_distractors(float(f), cands, 2))
    text = (
        f"The {qty} ({unit}) was measured for {setting}. {_group_table(names, ns, means, sds, unit)}. "
        f"Find the F statistic for testing whether the group means are equal, correct to 2 decimal places."
    )
    working = [
        f"Grand mean {dec(float(grand), 2)}; SSB = {dec(float(ssb), 2)} and SSW = {dec(float(ssw), 2)}.",
        f"MSB = SSB/2 = {dec(float(msb), 2)}, MSW = SSW/{N - 3} = {dec(float(msw), 2)}, so F = {dec(float(f), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(f), 2))


def _medium_explained(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(500, "a proportion away from a rounding boundary"):
        ns, means, sds = _draw_groups(rng)
        ssb, ssw, msb, msw, f, grand = anova(ns, means, [s * s for s in sds])
        p = ssb / (ssb + ssw)
        if stable(float(p), 4, 0.12) and 0.02 < float(p) < 0.95:
            break
    cands = [
        (ssb / ssw, "r4i-av-ratio-not-share"),
        (ssw / (ssb + ssw), "r4i-av-complement"),
        (msb / (msb + msw), "r4i-av-ms-not-ss"),
        (ssb / (ssb + ssw / (sum(ns) - 3)), "r4i-av-mixed-ss-ms"),
    ]
    dist = need(num_distractors(float(p), cands, 4, at_most=1.0))
    text = (
        f"The {qty} ({unit}) was measured for {setting}. {_group_table(names, ns, means, sds, unit)}. "
        f"What proportion of the total variation in {qty} is explained by the differences between the groups? "
        f"Give the answer correct to 4 decimal places."
    )
    working = [
        f"SSB = {dec(float(ssb), 2)} and SSW = {dec(float(ssw), 2)}, so SST = {dec(float(ssb + ssw), 2)}.",
        f"The proportion is SSB/SST = {dec(float(p), 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(p), 4))


def _medium_msb_totals(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(500, "totals with a tidy MSB"):
        n = rng.choice((4, 5, 6, 8))
        means = [rng.randint(10, 40) for _ in range(3)]
        totals = [n * m for m in means]
        grand_total = sum(totals)
        N = 3 * n
        ssb = Fraction(sum(t * t for t in totals), n) - Fraction(grand_total**2, N)
        msb = ssb / 2
        if len(set(means)) == 3 and (msb * 100).denominator == 1 and msb > 0:
            break
    cands = [
        (ssb, "r4i-av-forgot-df"),
        (ssb / 3, "r4i-av-divided-by-k"),
        (Fraction(sum(t * t for t in totals), n) / 2, "r4i-av-forgot-correction-term"),
        (Fraction(sum(t * t for t in totals), N) - Fraction(grand_total**2, N * n), "r4i-av-wrong-divisor-in-sum"),
        (msb * n, "r4i-av-size-twice"),
    ]
    dist = need(num_distractors(float(msb), cands, 2))
    text = (
        f"For each of {setting}, {n} units were recorded for {qty} ({unit}). The group totals are "
        f"{', '.join(f'{nm}: {t}' for nm, t in zip(names, totals))}. Use the totals to find the between-groups mean square "
        f"MSB (the between-groups sum of squares divided by its degrees of freedom)."
    )
    working = [
        f"Grand total {grand_total}, N = {N}. SSB = Σ Tᵢ²/{n} - T²/N = {dec(float(ssb), 2)}.",
        f"MSB = SSB/(3 - 1) = {dec(float(msb), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(msb), 2))


def _medium(rng):
    return rng.choice((_medium_f, _medium_explained, _medium_msb_totals))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_raw_f(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(1000, "raw data with a stable F"):
        m = rng.choice((4, 5))
        base = rng.choice((20, 30, 40))
        groups = [[base + off + rng.randint(-4, 4) for _ in range(m)] for off in (rng.randint(-5, 5) for _ in range(3))]
        ns = [m] * 3
        means = [Fraction(sum(g), m) for g in groups]
        vars_ = []
        for g, mu in zip(groups, means):
            vars_.append(sum((x - mu) ** 2 for x in g) / (m - 1))
        if min(vars_) == 0 or len(set(map(tuple, groups))) < 3:
            continue
        ssb, ssw, msb, msw, f, grand = anova(ns, means, vars_)
        if stable(float(f), 2, 0.12) and 0.2 < float(f) < 60:
            break
    N = 3 * m
    cands = [
        (1 / f, "r4i-av-f-inverted"),
        (ssb / ssw, "r4i-av-ss-not-ms"),
        ((ssb / (N - 3)) / (ssw / 2), "r4i-av-df-swapped"),
        (msb / (ssw / (N - 1)), "r4i-av-wrong-df"),
        (msb / (ssw / N), "r4i-av-wrong-df"),
    ]
    dist = need(num_distractors(float(f), cands, 2))
    rows = "; ".join(f"{nm}: {', '.join(str(x) for x in g)}" for nm, g in zip(names, groups))
    text = (
        f"For each of {setting}, {m} units were recorded for {qty} ({unit}). The observations are {rows}. "
        f"Find the F statistic for testing whether the three group means are equal, correct to 2 decimal places."
    )
    working = [
        f"Group means {', '.join(_ft(x) for x in means)}; grand mean {_ft(grand)}.",
        f"SSB = {dec(float(ssb), 2)}, SSW = {dec(float(ssw), 2)}, MSB = {dec(float(msb), 2)}, MSW = {dec(float(msw), 2)}, F = {dec(float(f), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(f), 2))


def _hard_missing_ssb(rng):
    for _ in attempts(500, "a table with whole sums of squares"):
        k = rng.choice((3, 4))
        N = rng.choice((18, 20, 24, 30))
        ssb = Fraction(rng.randint(40, 400))
        ssw = Fraction(rng.randint(100, 900))
        sst = ssb + ssw
        f = (ssb / (k - 1)) / (ssw / (N - k))
        if (f * 100).denominator == 1:
            break
    cands = [
        (sst * f / (1 + f), "r4i-av-ignored-df"),
        (sst - ssb, "r4i-av-gave-ssw"),
        (f * sst / (k - 1), "r4i-av-f-times-sst"),
        (sst * (f * (k - 1)) / (f * (k - 1) + (N - k)) * Fraction(1, 2), "r4i-av-halved"),
        (sst / (k - 1), "r4i-av-sst-over-df"),
    ]
    dist = need(num_distractors(float(ssb), cands, 0))
    setting = rng.choice(
        (
            "a one-way analysis of variance of weekly sales across the stores",
            "a one-way ANOVA of reaction times across the groups",
        )
    )
    text = (
        f"In {setting}, there are k = {k} groups and N = {N} observations in total. The total sum of squares is "
        f"{_g(sst)} and the F statistic is {float(f):.2f}. Find the between-groups sum of squares SSB."
    )
    working = [
        f"F = (SSB/{k - 1})/((SST - SSB)/{N - k}), so SSB = F({k - 1})SST/({N - k} + F({k - 1})).",
        f"SSB = {_g(ssb)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(ssb), 0))


def _hard_ci(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(500, "a stable half-width"):
        ns, means, sds = _draw_groups(rng)
        ssb, ssw, msb, msw, f, grand = anova(ns, means, [s * s for s in sds])
        N = sum(ns)
        df = N - 3
        if df not in T_CRIT:
            continue
        t = Fraction(T_CRIT[df])
        h = float(t) * math.sqrt(float(msw * (Fraction(1, ns[0]) + Fraction(1, ns[1]))))
        if stable(h, 3, 0.12):
            break
    cands = [
        (float(t) * float(msw * (Fraction(1, ns[0]) + Fraction(1, ns[1]))), "r4i-av-forgot-square-root"),
        (float(t) * math.sqrt(float(msw) * 2 / ns[0]), "r4i-av-equal-sizes-assumed"),
        (1.96 * math.sqrt(float(msw * (Fraction(1, ns[0]) + Fraction(1, ns[1])))), "r4i-av-z-for-t"),
        (float(t) * math.sqrt(float(msw) / ns[0] + float(msw) / ns[1]) * 2, "r4i-av-doubled"),
        (float(t) * math.sqrt(float(ssw / N * (Fraction(1, ns[0]) + Fraction(1, ns[1])))), "r4i-av-divided-by-total-n"),
    ]
    dist = need(num_distractors(h, cands, 3))
    text = (
        f"The {qty} ({unit}) was measured for {setting}. {_group_table(names, ns, means, sds, unit)}. Assuming a common "
        f"variance estimated by the pooled within-groups variance, find the half-width of a 95% confidence interval for the "
        f"difference between the mean of {names[0]} and the mean of {names[1]}, using the t value {T_CRIT[df]} for {df} degrees "
        f"of freedom. Give the answer correct to 3 decimal places."
    )
    working = [
        f"MSW = SSW/{df} = {dec(float(msw), 3)}.",
        f"Half-width = {T_CRIT[df]}×√(MSW(1/{ns[0]} + 1/{ns[1]})) = {dec(h, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(h, 3))


def _hard_critical_ssb(rng):
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(500, "a critical SSB away from a rounding boundary"):
        ns = [rng.choice((6, 8, 10)) for _ in range(3)]
        sds = [rng.choice((2, 3, 4, 5)) for _ in range(3)]
        ssw = Fraction(sum((n - 1) * s * s for n, s in zip(ns, sds)))
        df = sum(ns) - 3
        fcrit = rng.choice(("3.35", "3.40", "3.68", "3.47", "3.55"))
        msw = ssw / df
        need_ssb = Fraction(fcrit) * msw * 2
        if stable(float(need_ssb), 2, 0.12):
            break
    cands = [
        (Fraction(fcrit) * msw, "r4i-av-forgot-df"),
        (Fraction(fcrit) * ssw, "r4i-av-ssw-for-msw"),
        (Fraction(fcrit) * ssw / df * 3, "r4i-av-wrong-df"),
        (msw * 2 / Fraction(fcrit), "r4i-av-f-inverted"),
    ]
    dist = need(num_distractors(float(need_ssb), cands, 2))
    text = (
        f"The {qty} ({unit}) was measured for {setting}, with {', '.join(str(n) for n in ns)} units in the three groups and "
        f"within-group standard deviations {', '.join(str(s) for s in sds)}. The group means differ by an amount that is not "
        f"yet known. How large must the between-groups sum of squares SSB be for the F statistic to reach the critical value "
        f"{fcrit}? Give the answer correct to 2 decimal places."
    )
    working = [
        f"SSW = Σ(nᵢ - 1)sᵢ² = {_g(ssw)}, so MSW = {_g(ssw)}/{df} = {dec(float(msw), 2)}.",
        f"F = (SSB/2)/MSW = {fcrit} gives SSB = 2×{fcrit}×MSW = {dec(float(need_ssb), 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(float(need_ssb), 2))


def _hard(rng):
    return rng.choice((_hard_raw_f, _hard_missing_ssb, _hard_ci, _hard_critical_ssb))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_anova(rng, tier="easy"):
    """One-way analysis of variance pieces; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_anova(rng, tier="exam"):
    """Three groups: SSB, SSW, F, then the half-width of a confidence interval for a difference of two means."""
    check(tier == "exam", f"unknown tier {tier!r}")
    setting, qty, unit, names = rng.choice(SETTINGS)
    for _ in attempts(1000, "an exam draw with stable rounded values"):
        ns, means, sds = _draw_groups(rng)
        ssb, ssw, msb, msw, f, grand = anova(ns, means, [s * s for s in sds])
        N = sum(ns)
        df = N - 3
        if df not in T_CRIT:
            continue
        a = clean(float(ssb), 2)
        b = clean(float(ssw), 2)
        f_ft = (a / 2) / (b / df)
        t = float(T_CRIT[df])
        h_ft = t * math.sqrt(b / df * (1 / ns[0] + 1 / ns[1]))
        h_ex = t * math.sqrt(float(msw) * (1 / ns[0] + 1 / ns[1]))
        if (
            stable(float(f), 2, 0.12)
            and clean(f_ft, 2) == clean(float(f), 2)
            and stable(h_ex, 3, 0.12)
            and clean(h_ft, 3) == clean(h_ex, 3)
            and 0.3 < float(f) < 60
            and len({ns[0], ns[1]}) >= 1
        ):
            break
    c, d = clean(float(f), 2), clean(h_ex, 3)
    order = ["a", "b", "c", "d"]
    canon = {"a": a, "b": b, "c": c, "d": d}
    fns = {
        "c": lambda s: (s["a"] / 2) / (s["b"] / df) if s["b"] else None,
        "d": lambda s: t * math.sqrt(s["b"] / df * (1 / ns[0] + 1 / ns[1])) if s["b"] >= 0 else None,
    }
    rounders = {"c": lambda v: clean(v, 2), "d": lambda v: clean(v, 3)}
    unweighted = sum(means) / 3
    a_dist = need(
        num_distractors(
            a,
            [
                (sum((m - float(grand)) ** 2 for m in means), "r4i-av-forgot-group-size"),
                (sum(n * (m - unweighted) ** 2 for n, m in zip(ns, means)), "r4i-av-unweighted-grand-mean"),
                (sum((n - 1) * (m - float(grand)) ** 2 for n, m in zip(ns, means)), "r4i-av-n-minus-one-for-n"),
                (float(msb), "r4i-av-divided-by-df"),
            ],
            2,
        )
    )
    b_dist = need(
        num_distractors(
            b,
            [
                (sum(n * s * s for n, s in zip(ns, sds)), "r4i-av-n-for-n-minus-one"),
                (sum((n - 1) * s for n, s in zip(ns, sds)), "r4i-av-sd-for-variance"),
                (sum(s * s for s in sds), "r4i-av-forgot-weights"),
                (float(msw), "r4i-av-divided-by-df"),
            ],
            2,
        )
    )
    wrong = {"a": [float(stored(x)) for x in a_dist], "b": [float(stored(x)) for x in b_dist]}
    avoid_c = [v for v in chain_avoid(order, fns, canon, wrong, "c", rounders) if v is not None]
    c_dist = need(
        num_distractors(
            c,
            [
                (1 / c, "r4i-av-f-inverted"),
                (a / b, "r4i-av-ss-not-ms"),
                ((a / df) / (b / 2), "r4i-av-df-swapped"),
                ((a / 2) / (b / (N - 1)), "r4i-av-wrong-df"),
            ],
            2,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [float(v.value) for v in c_dist]
    avoid_d = [v for v in chain_avoid(order, fns, canon, wrong, "d", rounders) if v is not None]
    d_dist = need(
        num_distractors(
            d,
            [
                (t * (b / df) * (1 / ns[0] + 1 / ns[1]), "r4i-av-forgot-square-root"),
                (t * math.sqrt(2 * b / df / ns[0]), "r4i-av-equal-sizes-assumed"),
                (1.96 * math.sqrt(b / df * (1 / ns[0] + 1 / ns[1])), "r4i-av-z-for-t"),
                (t * math.sqrt(b / N * (1 / ns[0] + 1 / ns[1])), "r4i-av-divided-by-total-n"),
            ],
            3,
            avoid=avoid_d,
        )
    )
    table = _group_table(names, ns, means, sds, unit)
    text = rng.choice(
        [
            f"A study compares the {qty} ({unit}) of {setting}. {table}. The observations within each group are independent "
            f"and the groups have a common variance.",
            f"The {qty} ({unit}) was measured for {setting}: {table}. Assume independent observations and a common variance "
            f"in the three groups.",
            f"Researchers recorded the {qty} ({unit}) under {setting}. Summary results: {table}. Treat the groups as independent "
            f"samples with equal variance.",
        ]
    )
    parts = [
        dict(
            prompt="Find the between-groups sum of squares SSB, correct to 2 decimal places.",
            marks=2,
            sensible_range=(0, None),
            working=[f"Grand mean {dec(float(grand), 2)}; SSB = Σ nᵢ(mean - grand mean)² = {dec(a, 2)}."],
            distractors=a_dist,
            **num_fields(a, 2),
        ),
        dict(
            prompt="Find the within-groups sum of squares SSW, correct to 2 decimal places.",
            marks=2,
            sensible_range=(0, None),
            working=[f"SSW = Σ(nᵢ - 1)sᵢ² = {dec(b, 2)}."],
            distractors=b_dist,
            **num_fields(b, 2),
        ),
        dict(
            prompt="Find the F statistic for testing whether the three group means are equal, correct to 2 decimal places.",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"(part_a/2)/(part_b/{df})",
            working=[f"MSB = SSB/2, MSW = SSW/{df}, so F = {dec(c, 2)}."],
            distractors=c_dist,
            **num_fields(c, 2),
        ),
        dict(
            prompt=(
                f"Find the half-width of a 95% confidence interval for the difference between the mean of {names[0]} and the "
                f"mean of {names[1]}, using the pooled variance SSW/{df} and the t value {T_CRIT[df]} for {df} degrees of "
                f"freedom. Give the answer correct to 3 decimal places."
            ),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"{T_CRIT[df]}*sqrt(part_b/{df}*(1/{ns[0]} + 1/{ns[1]}))",
            working=[f"Half-width = {T_CRIT[df]}×√(MSW(1/{ns[0]} + 1/{ns[1]})) = {dec(d, 3)}."],
            distractors=d_dist,
            **num_fields(d, 3),
        ),
    ]
    return dict(text=text, parts=parts)
