"""Generates `judge-stated-conclusion` questions: recompute a claimed figure, then judge whether the data support it.

Every question states data and a claim made from it. The learner first works out the figure the claim rests on (a
percentage, a difference in percentages, a mean, or a value predicted from a line), then picks from five options the
verdict that fits. Each scenario is built with exactly one thing wrong with the claim, or nothing wrong:

  sound            the figures match, the groups were assigned by chance (or the claim is only descriptive);
  wrong figure     the percentage in the claim does not match the table;
  too few          the claim is about everyone but only a handful of cases were recorded;
  cause            the groups chose themselves, but the claim says one thing causes the other;
  outlier          one unusual value drags the mean away from the typical value;
  extrapolation    the claim is about a value far outside the range the line was fitted to.

The six verdicts are written in one sentence shape, so the key cannot be read from the wording.

Ladder (reasoning escalation, not size):
  easy:   one group; sound, wrong figure or too few cases; the number is the percentage from the counts.
  medium: adds two groups compared (cause) and a mean with an outlier; the number is a difference in percentages or
          the mean.
  hard:   all six; the number takes two steps (the difference between two groups' percentages, how far the mean sits
          above the median, or a prediction well outside the data) and the claim is worded as a comparison.
  exam:   a four-part chain on a two-group observational study (the two percentages, their difference, the verdict)
          or a mean with an outlier (the mean, the median, their difference, the verdict).
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._ir2a_kit import NAMES, chain, mcq_part, median_of, number_part, single

_TIERS = ("easy", "medium", "hard")

SOUND, WRONG, FEW, CAUSE, OUTLIER, EXTRAP = "sound", "wrong", "few", "cause", "outlier", "extrapolation"
VERDICT = {
    SOUND: "The data support the claim: the figures match and the study was designed to test it.",
    WRONG: "The data do not support the claim: the figure in the claim does not match the data.",
    FEW: "The data do not support the claim: too few cases were recorded to say anything about everyone.",
    CAUSE: "The data do not support the claim: the groups chose themselves, so the study shows a link but not a cause.",
    OUTLIER: "The data do not support the claim: one unusual value pulls the mean away from a typical value.",
    EXTRAP: "The data do not support the claim: it is about a value well outside the range the data cover.",
}
_ORDER = (SOUND, WRONG, FEW, CAUSE, OUTLIER, EXTRAP)


def _verdict_part(flaw: str, others: tuple[str, ...], marks: int = 1) -> dict:
    wrongs = []
    for f in others:
        if f == flaw:
            continue
        tag = (
            "ir2a-rejects-a-sound-claim"
            if flaw == SOUND
            else ("ir2a-accepts-a-flawed-claim" if f == SOUND else "ir2a-names-the-wrong-flaw")
        )
        wrongs.append((VERDICT[f], tag))
    return mcq_part(
        "Which statement is the best judgement of the claim?",
        VERDICT[flaw],
        wrongs,
        marks=marks,
        working=[
            f"The claim fails for one reason only: {flaw}." if flaw != SOUND else "Nothing is wrong with the claim."
        ],
        keep=4,
    )


_SURVEY = (
    ("residents of the Shire of Marlow", "walk to the shops at least once a week"),
    ("students at Banksia Secondary College", "own a mobile phone"),
    ("members of the Wattle Park Sports Club", "would join a weekend competition"),
    ("customers of the Gum Tree Cafe", "buy a takeaway coffee"),
)
_OBS = (
    (
        "students",
        "ate breakfast on the day of the test",
        "did not eat breakfast",
        "passed the maths test",
        "Eating breakfast makes students more likely to pass a maths test.",
    ),
    ("adults", "walked to work", "drove to work", "reported feeling well", "Walking to work makes adults feel well."),
    (
        "students",
        "played a sport",
        "did not play a sport",
        "scored above 70 on the semester exam",
        "Playing a sport improves students' exam results.",
    ),
)
_EXP = (
    (
        "students",
        "were randomly placed in the tutoring program",
        "were randomly placed in the regular class",
        "passed the maths test",
        "The tutoring program makes students more likely to pass the maths test.",
    ),
    (
        "bean seeds",
        "were randomly planted in fertilised soil",
        "were randomly planted in plain soil",
        "germinated within a week",
        "Fertiliser makes bean seeds more likely to germinate within a week.",
    ),
    (
        "members",
        "were randomly given the new training plan",
        "were randomly given the old plan",
        "improved their 5 km time",
        "The new training plan makes members more likely to improve their 5 km time.",
    ),
)
_PRICE = (
    (
        "the house prices in a suburb, in thousands of dollars",
        "The typical house in the suburb sells for {mean} thousand dollars.",
    ),
    (
        "the weekly wages of the staff in a small business, in dollars",
        "The typical weekly wage in the business is ${mean}.",
    ),
    (
        "the prices of the paintings sold at an auction, in thousands of dollars",
        "The typical painting sold at the auction costs {mean} thousand dollars.",
    ),
)
_LINE = (
    dict(
        what="a child's height (cm) by age (years)",
        a=75,
        b=6,
        lo=2,
        hi=10,
        bad=(25, 40),
        claim="A {x}-year-old will be {y} cm tall.",
        unit="cm",
    ),
    dict(
        what="the number of ice creams sold in a day by the daily maximum temperature (degrees Celsius)",
        a=-60,
        b=8,
        lo=18,
        hi=38,
        bad=(48, 60),
        claim="On a day with a maximum of {x} degrees, {y} ice creams will be sold.",
        unit="ice creams",
    ),
    dict(
        what="the height of a sunflower (cm) by its age (weeks)",
        a=10,
        b=14,
        lo=2,
        hi=9,
        bad=(30, 40),
        claim="A sunflower will be {y} cm tall at {x} weeks.",
        unit="cm",
    ),
)


def _pct(k: int, n: int) -> Fraction:
    return Fraction(k * 100, n)


def _one_group(rng, flaw: str):
    """sound, wrong figure or too few cases, from a survey of one group."""
    name = rng.choice(NAMES)
    who, does = rng.choice(_SURVEY)
    if flaw == FEW:
        n = rng.choice([8, 10, 12, 16, 20])
        pairs = [(k, n) for k in range(2, n - 1) if (k * 100) % n == 0 and 2 * k != n]
    else:
        n = rng.choice([200, 250, 400, 500, 800])
        pairs = [(k, n) for k in range(int(0.2 * n), int(0.8 * n)) if (k * 100) % n == 0 and 2 * k != n]
    k, n = rng.choice(pairs)
    actual = _pct(k, n)
    claimed = actual if flaw != WRONG else actual + rng.choice([-15, -10, 10, 15])
    check(0 < claimed < 100, "claimed percentage must be sensible")
    stem = (
        f"{name} asked {n} randomly chosen {who} and found that {k} of them {does}. "
        f"{name} claims: \u201c{int(claimed)}% of all the {who} {does}.\u201d"
    )
    a = number_part(
        prompt=f"What percentage of the {n} {who} asked {does}?",
        exact=actual,
        places=0,
        marks=2,
        working=[f"{k} / {n} x 100 = {int(actual)}%."],
        wrong=[
            (k, "ir2a-reports-the-count-as-a-percentage"),
            (100 - actual, "ir2a-gives-the-rest-of-the-whole"),
            (Fraction(k * 100, n - k), "ir2a-divides-by-the-rest-of-the-whole"),
        ],
        sensible_range=(0, 100),
    )
    return stem, a


def _two_groups(rng, flaw: str):
    """Two groups' percentages. CAUSE is observational, the others are experiments; WRONG and FEW make a claim
    about the gap, SOUND and CAUSE claim a cause."""
    name = rng.choice(NAMES)
    bank = _OBS if flaw == CAUSE else _EXP
    who, ga, gb, outcome, claim = rng.choice(bank)
    sizes = [10, 20] if flaw == FEW else [60, 80, 100, 120, 150, 200]
    for _ in attempts(500, "two groups with whole percentages"):
        na, nb = rng.choice(sizes), rng.choice(sizes)
        pa, pb = rng.choice(range(40, 95, 5)), rng.choice(range(25, 85, 5))
        if pa - pb >= 15 and na * pa % 100 == 0 and nb * pb % 100 == 0:
            break
    ka, kb = na * pa // 100, nb * pb // 100
    how = (
        f"Each of the {who} chose which group to be in."
        if flaw == CAUSE
        else f"Which group each of the {who} went into was decided by chance."
    )
    gap = pa - pb
    if flaw in (WRONG, FEW):
        said = gap + rng.choice([-10, -5, 5, 10]) if flaw == WRONG else gap
        claim = f"The {who} that {ga} were {said} percentage points more likely to have {outcome} than those that {gb}."
    stem = (
        f"A study followed {na + nb} {who}. {na} {ga}, and {ka} of them {outcome}. {nb} {gb}, and {kb} of them {outcome}. "
        f"{how} {name} claims: \u201c{claim}\u201d"
    )
    return stem, (who, ga, gb, outcome, na, nb, ka, kb, pa, pb)


def _mean_outlier(rng):
    name = rng.choice(NAMES)
    desc, claim = rng.choice(_PRICE)
    n = 9
    for _ in attempts(500, "a list with a whole mean"):
        base = rng.choice([400, 450, 500, 600])
        vals = [base + rng.randint(-80, 120) for _ in range(n - 1)]
        big = base * rng.choice([4, 5, 6])
        total = sum(vals) + big
        adjust = (-total) % n
        big += adjust
        if len(set(vals)) == n - 1:
            break
    vals.append(big)
    rng.shuffle(vals)
    mean = Fraction(sum(vals), n)
    check(mean.denominator == 1, "mean must be whole")
    med = median_of(vals)
    stem = f"{name} recorded {desc}: {', '.join(map(str, vals))}. {name} claims: “{claim.format(mean=int(mean))}”"
    return stem, vals, mean, med, n


def _extrap(rng, level):
    name = rng.choice(NAMES)
    sc = rng.choice(_LINE)
    x = rng.randint(*sc["bad"])
    y = sc["a"] + sc["b"] * x
    stem = (
        f"{name} fitted the line y = {sc['a']} + {sc['b']}x to {sc['what']}. The data used had x-values from {sc['lo']} to {sc['hi']}. "
        f"{name} claims: “{sc['claim'].format(x=x, y=y)}”"
    ).replace("+ -", "- ")
    a = number_part(
        prompt=f"Use the line to find the value of y when x is {x}.",
        exact=y,
        places=0,
        marks=2,
        working=[f"y = {sc['a']} + {sc['b']} x {x} = {y}."],
        wrong=[
            (sc["b"] * x, "ir2a-forgets-the-intercept"),
            (sc["a"] - sc["b"] * x, "ir2a-flips-the-sign-of-the-slope-term"),
            (sc["a"] + sc["b"] + x, "ir2a-adds-instead-of-multiplying"),
        ],
        sensible_range=None,
    )
    return stem, a


def _gap_part(d, marks=3):
    who, ga, gb, outcome, na, nb, ka, kb, pa, pb = d
    return number_part(
        prompt=f"By how many percentage points is the percentage of the first group that {outcome} higher than that of the second group?",
        exact=pa - pb,
        places=0,
        marks=marks,
        working=[f"First group: {ka}/{na} = {pa}%. Second group: {kb}/{nb} = {pb}%.", f"{pa} - {pb} = {pa - pb}."],
        wrong=[
            (pa + pb, "ir2a-adds-instead-of-subtracting"),
            (ka - kb, "ir2a-compares-counts-not-percentages"),
            (Fraction((pa - pb) * 100, pb), "ir2a-gives-a-relative-change-not-points"),
        ],
        sensible_range=(0, 100),
    )


def _build(rng, level):
    pool = {
        "easy": (SOUND, WRONG, FEW),
        "medium": (SOUND, WRONG, FEW, CAUSE, OUTLIER),
        "hard": _ORDER,
    }[level]
    flaw = rng.choice(pool)
    options = (SOUND, WRONG, FEW, CAUSE, OUTLIER) if level == "easy" else _ORDER
    if flaw == OUTLIER:
        stem, vals, mean, med, n = _mean_outlier(rng)
        if level == "medium":
            a = number_part(
                prompt="What is the mean of the values?",
                exact=mean,
                places=0,
                marks=2,
                working=[f"{sum(vals)} / {n} = {int(mean)}."],
                wrong=[
                    (sum(vals), "ir2a-forgets-to-divide-by-the-count"),
                    (med, "ir2a-takes-a-middle-value-for-the-mean"),
                    (Fraction(sum(vals), n - 1), "ir2a-divides-by-one-too-few"),
                ],
                sensible_range=(0, None),
            )
        else:
            a = number_part(
                prompt="By how much does the mean of the values exceed their median?",
                exact=mean - med,
                places=0,
                marks=3,
                working=[
                    f"Mean = {sum(vals)} / {n} = {int(mean)}.",
                    f"Median = the 5th value in order = {int(med)}.",
                    f"{int(mean)} - {int(med)} = {int(mean - med)}.",
                ],
                wrong=[
                    (mean, "ir2a-gives-the-mean-not-the-excess"),
                    (med, "ir2a-gives-the-median-not-the-excess"),
                    (mean + med, "ir2a-adds-instead-of-subtracting"),
                ],
                sensible_range=(0, None),
            )
        return chain(stem, a, _verdict_part(OUTLIER, options))
    if flaw == EXTRAP:
        stem, a = _extrap(rng, level)
        return chain(stem, a, _verdict_part(EXTRAP, options))
    one_group = flaw in (SOUND, WRONG, FEW) and (
        level == "easy" or (level == "medium" and (flaw != SOUND or rng.random() < 0.5))
    )
    if one_group:
        stem, a = _one_group(rng, flaw)
        return chain(stem, a, _verdict_part(flaw, options))
    stem, d = _two_groups(rng, flaw)
    return chain(stem, _gap_part(d), _verdict_part(flaw, options))


@generator("ir2a_judge_stated_conclusion", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_judge_stated_conclusion(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return _build(rng, tier)


@generator("exd_x_ir2a_judge_stated_conclusion", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_judge_stated_conclusion(rng: random.Random, tier: str = "exam") -> dict:
    """Two percentages, their difference and the verdict on a causal claim; or mean, median, excess and the verdict."""
    check(tier == "exam", "only the exam tier exists")
    if rng.random() < 0.5:
        flaw = rng.choice((CAUSE, WRONG))
        stem, d = _two_groups(rng, flaw)
        who, ga, gb, outcome, na, nb, ka, kb, pa, pb = d
        a = number_part(
            prompt=f"What percentage of the first group {outcome}?",
            exact=pa,
            places=0,
            marks=1,
            working=[f"{ka}/{na} x 100 = {pa}%."],
            wrong=[
                (ka, "ir2a-reports-the-count-as-a-percentage"),
                (100 - pa, "ir2a-gives-the-rest-of-the-whole"),
                (Fraction(ka * 100, na - ka), "ir2a-divides-by-the-rest-of-the-whole"),
                (Fraction(ka * 100, na + nb), "ir2a-divides-by-both-groups"),
            ],
            sensible_range=(0, 100),
        )
        b = number_part(
            prompt=f"What percentage of the second group {outcome}?",
            exact=pb,
            places=0,
            marks=1,
            working=[f"{kb}/{nb} x 100 = {pb}%."],
            wrong=[
                (kb, "ir2a-reports-the-count-as-a-percentage"),
                (100 - pb, "ir2a-gives-the-rest-of-the-whole"),
                (Fraction(kb * 100, nb - kb), "ir2a-divides-by-the-rest-of-the-whole"),
                (Fraction(kb * 100, na + nb), "ir2a-divides-by-both-groups"),
            ],
            sensible_range=(0, 100),
        )
        c = number_part(
            prompt="By how many percentage points is the first group's percentage higher than the second group's?",
            exact=pa - pb,
            places=0,
            marks=2,
            working=[f"{pa} - {pb} = {pa - pb}."],
            wrong=[
                (pa + pb, "ir2a-adds-instead-of-subtracting"),
                (ka - kb, "ir2a-compares-counts-not-percentages"),
                (Fraction((pa - pb) * 100, pb), "ir2a-gives-a-relative-change-not-points"),
            ],
            follow_through="part_a - part_b",
            sensible_range=(0, 100),
        )
        return chain(stem, a, b, c, _verdict_part(flaw, _ORDER, 2))
    stem, vals, mean, med, n = _mean_outlier(rng)
    a = number_part(
        prompt="What is the mean of the values?",
        exact=mean,
        places=0,
        marks=1,
        working=[f"{sum(vals)} / {n} = {int(mean)}."],
        wrong=[
            (sum(vals), "ir2a-forgets-to-divide-by-the-count"),
            (med, "ir2a-takes-a-middle-value-for-the-mean"),
            (Fraction(sum(vals), n - 1), "ir2a-divides-by-one-too-few"),
        ],
        sensible_range=(0, None),
    )
    b = number_part(
        prompt="What is the median of the values?",
        exact=med,
        places=0,
        marks=1,
        working=[f"The 5th value in order is {int(med)}."],
        wrong=[
            (mean, "ir2a-gives-the-mean-for-the-median"),
            (vals[n // 2], "ir2a-takes-the-middle-of-the-list-unsorted"),
            (max(vals), "ir2a-reports-the-largest-value"),
        ],
        sensible_range=(0, None),
    )
    c = number_part(
        prompt="By how much does the mean exceed the median?",
        exact=mean - med,
        places=0,
        marks=2,
        working=[f"{int(mean)} - {int(med)} = {int(mean - med)}."],
        wrong=[
            (mean + med, "ir2a-adds-instead-of-subtracting"),
            (med - mean, "ir2a-subtracts-in-the-wrong-order"),
            (mean, "ir2a-gives-the-mean-not-the-excess"),
        ],
        follow_through="part_a - part_b",
        sensible_range=(0, None),
    )
    return chain(stem, a, b, c, _verdict_part(OUTLIER, _ORDER, 2))
