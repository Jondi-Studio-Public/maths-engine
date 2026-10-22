"""Generates `choose-sampling-method` questions: name the sampling method a survey used, then use it.

Every question describes how a survey group was chosen from a stated population and asks the learner to name the
method (a multiple-choice part with five options: census, simple random, systematic, stratified random,
convenience), then to work out one number that only that method makes meaningful:

  census         how many more interviews than a sample needs, the hours of interviewing, or the extra cost;
  simple random  the chance a particular member is in the sample, the expected count from one group, or the chance
                 of being left out;
  systematic     the gap down the list, the place of the fourth person chosen from a random start, or the place of
                 the last person chosen;
  stratified     how many to take from one group, from group sizes or from percentages;
  convenience    the percentage of the population that could be chosen at all.

Ladder (reasoning escalation, not size):
  easy:   the method is described by its mechanism and the number is one direct use of the method.
  medium: the mechanism is described in the situation's own words, and the number takes two steps (find the gap,
          then count along the list; find the sampling fraction, then apply it to a small group).
  hard:   no method is described. The learner is given the AIM of the survey and has to name the method that
          meets it, then use the method on numbers the aim implies (percentages instead of counts, a gap given
          instead of a sample size).
  exam:   a four-part chain on a stratified or systematic sample, each part using the one before.

The method options are the same five sentences every time, so neither wording nor length gives the key away; the
key is whichever method the scenario uses, drawn evenly.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._ir2a_kit import NAMES, chain, fraction_part, mcq_part, number_part, single, tidy

_TIERS = ("easy", "medium", "hard")

CENSUS, SIMPLE, SYSTEMATIC, STRATIFIED, CONVENIENCE = "census", "simple", "systematic", "stratified", "convenience"
METHOD_TEXT = {
    CENSUS: "Census",
    SIMPLE: "Simple random sample",
    SYSTEMATIC: "Systematic sample",
    STRATIFIED: "Stratified random sample",
    CONVENIENCE: "Convenience sample",
}
#: The error a learner makes in naming `wrong` when the method is something else.
WRONG_TAG = {
    CENSUS: "ir2a-calls-a-sample-a-census",
    SIMPLE: "ir2a-calls-any-chance-selection-simple-random",
    SYSTEMATIC: "ir2a-calls-any-regular-pattern-systematic",
    STRATIFIED: "ir2a-calls-any-grouping-stratified",
    CONVENIENCE: "ir2a-calls-any-quick-sample-convenience",
}
#: Which wrong methods are most tempting for each right one, most tempting first.
TEMPTING = {
    CENSUS: (SIMPLE, SYSTEMATIC, STRATIFIED, CONVENIENCE),
    SIMPLE: (SYSTEMATIC, STRATIFIED, CONVENIENCE, CENSUS),
    SYSTEMATIC: (SIMPLE, STRATIFIED, CONVENIENCE, CENSUS),
    STRATIFIED: (SIMPLE, SYSTEMATIC, CONVENIENCE, CENSUS),
    CONVENIENCE: (SIMPLE, SYSTEMATIC, STRATIFIED, CENSUS),
}

_POPS = (
    dict(
        pl="students",
        sg="student",
        org="Banksia Secondary College",
        topic="which lunch option they prefer",
        gate="the front gate",
        groups=("Year 7", "Year 8", "Year 9", "Year 10"),
        group_phrase="year levels",
    ),
    dict(
        pl="members",
        sg="member",
        org="the Wattle Park Sports Club",
        topic="whether the opening hours should change",
        gate="the clubhouse door",
        groups=("junior", "intermediate", "senior", "veteran"),
        group_phrase="membership grades",
    ),
    dict(
        pl="employees",
        sg="employee",
        org="Southern Cross Freight",
        topic="how they travel to work",
        gate="the staff car park",
        groups=("the warehouse", "the depot", "the head office", "the call centre"),
        group_phrase="sites",
    ),
    dict(
        pl="residents",
        sg="resident",
        org="the Shire of Marlow",
        topic="whether a new skate park should be built",
        gate="the main street shops",
        groups=("the northern suburbs", "the central suburbs", "the southern suburbs", "the rural area"),
        group_phrase="areas",
    ),
)


def _group_label(pop: dict, i: int) -> str:
    g = pop["groups"][i]
    return g if g.startswith("the ") else f"the {g} group"


def _stem_for(method: str, pop: dict, N: int, n: int, name: str, level: str, extra: dict) -> str:
    pl, sg, org, topic = pop["pl"], pop["sg"], pop["org"], pop["topic"]
    if method == CENSUS:
        if level == "easy":
            return f"{name} asks every one of the {N} {pl} of {org} {topic}."
        return (
            f"{org} has {N} {pl}. {name} arranges to interview each of them in turn, so that no {sg} is left out "
            f"of the survey on {topic}."
        )
    if method == SIMPLE:
        if level == "easy":
            return (
                f"Each of the {N} {pl} of {org} is given a different number from 1 to {N}. A computer picks {n} "
                f"different numbers at random, and the {pl} with those numbers are asked {topic}."
            )
        return (
            f"{name} puts the names of all {N} {pl} of {org} into a barrel, mixes them well and draws out {n} "
            f"names without looking. Those {n} {pl} are surveyed on {topic}."
        )
    if method == SYSTEMATIC:
        if level == "easy":
            return (
                f"The {N} {pl} of {org} are on one alphabetical list. {name} picks a starting place at random near "
                f"the top, then asks the {sg} at that place and every {sg} the same number of places further down "
                f"the list, so that {n} {pl} are asked {topic}."
            )
        return (
            f"The {N} {pl} of {org} are numbered 1 to {N} by their date of joining. {name} draws one number at "
            f"random from the first few, and then surveys the {pl} whose numbers follow at a fixed step, until {n} "
            f"{pl} have been surveyed on {topic}."
        )
    if method == STRATIFIED:
        parts = extra["group_desc"]
        if level == "easy":
            return (
                f"The {N} {pl} of {org} fall into {extra['k']} {pop['group_phrase']}: {parts}. {name} takes a random "
                f"sample from each group, with the number taken from each group matching its share of the {pl}, "
                f"to give {n} {pl} to ask {topic}."
            )
        return (
            f"{org} has {N} {pl}, split by {pop['group_phrase']} into {parts}. {name} wants {n} {pl} to ask {topic}, "
            f"draws names by chance within each group, and makes sure each group supplies the same fraction of "
            f"its {pl}."
        )
    # convenience
    if level == "easy":
        return f"{name} stands at {pop['gate']} of {org} on Monday morning and asks the first {n} {pl} who come past {topic}."
    return (
        f"{name} has {N} {pl} to choose from at {org}. {name} asks the {n} {pl} who are nearest at lunchtime on one "
        f"day, because that is quickest, about {topic}."
    )


def _split_into_groups(rng, total_units: int, k: int) -> list[int]:
    """`k` whole-number multiples that sum to `total_units`, each at least 2."""
    for _ in attempts(500, "a split into groups"):
        cuts = sorted(rng.sample(range(1, total_units), k - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, total_units])]
        if min(parts) >= 2 and len(set(parts)) == k:
            return parts
    raise AssertionError("unreachable")


def _method_part(rng, method: str, level: str) -> dict:
    wrong_methods = TEMPTING[method]
    return mcq_part(
        "Which sampling method meets these requirements?" if level == "hard" else "Which sampling method was used?",
        METHOD_TEXT[method],
        [(METHOD_TEXT[m], WRONG_TAG[m]) for m in wrong_methods],
        working=[
            f"The way the {'group was' if method != CENSUS else 'people were'} chosen is a {METHOD_TEXT[method].lower()}."
        ],
        keep=4,
    )


# -- numbers for each method ------------------------------------------------------------------------------------


def _build(rng: random.Random, method: str, level: str) -> tuple[str, dict]:
    """(stem, number part) for `method` at `level` (easy / medium / hard)."""
    pop = rng.choice(_POPS)
    name = rng.choice(NAMES)
    pl, sg = pop["pl"], pop["sg"]
    extra: dict = {}

    if method == SYSTEMATIC:
        k = rng.randint(4, 15)
        n = rng.choice([20, 25, 30, 35, 40, 45, 50, 60])
        N = n * k
        s = rng.randint(1, min(k, 6)) if level != "easy" else None
        check(N % n == 0 and N // n == k, "gap does not divide")
        stem = _stem_for(method, pop, N, n, name, "easy" if level == "easy" else "plain", extra)
        if level == "easy":
            part = number_part(
                prompt=f"How many places apart on the list are the {pl} who are asked?",
                exact=k,
                places=0,
                marks=2,
                working=[f"Gap = population / sample = {N} / {n} = {k}."],
                wrong=[
                    (n, "ir2a-uses-sample-size-as-the-gap"),
                    (N - n, "ir2a-gap-from-the-difference"),
                    (k + 1, "ir2a-gap-counts-the-start"),
                ],
                sensible_range=(0, None),
            )
        elif level == "medium":
            j = rng.randint(3, 5)
            stem += f" The first {sg} chosen is number {s}."
            part = number_part(
                prompt=f"What is the number of the {_ordinal(j)} {sg} chosen?",
                exact=s + (j - 1) * k,
                places=0,
                marks=2,
                working=[f"Gap = {N} / {n} = {k}.", f"Number {j} chosen = {s} + {j - 1} × {k} = {s + (j - 1) * k}."],
                wrong=[
                    (s + j * k, "ir2a-counts-one-gap-too-many"),
                    (j * k, "ir2a-forgets-the-random-start"),
                    (s + (j - 1) * n, "ir2a-uses-sample-size-as-the-gap"),
                ],
                sensible_range=(0, None),
            )
        else:
            stem = (
                f"{name} surveys {pl} of {pop['org']} from a numbered list of {N}, choosing number {s} at random "
                f"and then every {k}th number after it, down to the end of the list."
            )
            last = s + (N // k - 1) * k
            part = number_part(
                prompt=f"What is the number of the last {sg} chosen?",
                exact=last,
                places=0,
                marks=3,
                working=[
                    f"Number surveyed = {N} / {k} = {N // k}.",
                    f"Last number = {s} + ({N // k} - 1) × {k} = {last}.",
                ],
                wrong=[
                    (s + (N // k) * k, "ir2a-counts-one-gap-too-many"),
                    (N, "ir2a-takes-the-end-of-the-list"),
                    ((N // k) * k + s - k + 1, "ir2a-forgets-the-random-start"),
                    (s + (N // k - 2) * k, "ir2a-counts-one-gap-too-few"),
                ],
                sensible_range=(0, N + 1),
            )
            check(last <= N, "last selected is off the list")
        return stem, part

    if method == STRATIFIED:
        k = 3 if level != "hard" else 4
        if level == "hard":
            pct = rng.choice([[40, 30, 20, 10], [35, 30, 20, 15], [45, 25, 20, 10], [30, 30, 25, 15], [50, 20, 20, 10]])
            n = rng.choice([40, 60, 80, 100, 120, 140])
            N = n * rng.randint(4, 9)
            rng.shuffle(pct)
            gi = rng.randrange(4)
            counts = [n * p // 100 for p in pct]
            check(sum(counts) == n and all(n * p % 100 == 0 for p in pct), "percentages do not split the sample")
            desc = ", ".join(f"{p}% in {pop['groups'][i]}" for i, p in enumerate(pct))
            stem = (
                f"{name} wants a sample of {n} of the {N} {pl} of {pop['org']} about {pop['topic']}. The {pl} are in "
                f"{pop['group_phrase']}: {desc}. Every group has to be represented in proportion to its size, with "
                f"{pl} chosen at random within each group."
            )
            part = number_part(
                prompt=f"How many {pl} should be chosen from {pop['groups'][gi] if pop['groups'][gi].startswith('the ') else 'the ' + pop['groups'][gi] + ' group'}?",
                exact=counts[gi],
                places=0,
                marks=2,
                working=[f"{pct[gi]}% of {n} = {counts[gi]}."],
                wrong=[
                    (n // 4, "ir2a-equal-allocation-ignores-size"),
                    (N * pct[gi] // 100, "ir2a-applies-the-percentage-to-the-population"),
                    (pct[gi], "ir2a-reports-the-percentage-as-a-count"),
                    (counts[(gi + 1) % 4], "ir2a-reads-another-groups-count"),
                ],
                sensible_range=(0, n),
            )
            return stem, part
        # easy / medium: sizes d*m_i with sampling fraction c/d
        c, d = rng.choice([(1, 5), (1, 8), (1, 10), (1, 20), (2, 5), (3, 20), (1, 6), (1, 12)])
        ms = _split_into_groups(rng, rng.randint(14, 30), k)
        sizes = [d * m for m in ms]
        takes = [c * m for m in ms]
        N, n = sum(sizes), sum(takes)
        gi = rng.randrange(k) if level == "easy" else min(range(k), key=lambda i: sizes[i])
        extra["group_desc"] = ", ".join(f"{sizes[i]} in {pop['groups'][i]}" for i in range(k))
        extra["k"] = k
        stem = _stem_for(method, pop, N, n, name, "easy" if level == "easy" else "plain", extra)
        gname = _group_label(pop, gi)
        frac = Fraction(n, N)
        wrong = [
            (Fraction(n, 3) if k == 3 else Fraction(n, k), "ir2a-equal-allocation-ignores-size"),
            (takes[(gi + 1) % k], "ir2a-reads-another-groups-count"),
            (Fraction(sizes[gi] * 100, N), "ir2a-reports-the-percentage-as-a-count"),
        ]
        if level == "easy":
            prompt = f"How many {pl} should be taken from {gname}?"
            working = [
                f"Sampling fraction = {n} / {N} = {frac.numerator}/{frac.denominator}.",
                f"{frac.numerator}/{frac.denominator} × {sizes[gi]} = {takes[gi]}.",
            ]
        else:
            prompt = f"How many {pl} are taken from the smallest group, {gname}?"
            working = [
                f"Every group supplies the same fraction of its {pl}: {n} / {N} = {frac.numerator}/{frac.denominator}.",
                f"{frac.numerator}/{frac.denominator} × {sizes[gi]} = {takes[gi]}.",
            ]
        part = number_part(
            prompt=prompt,
            exact=takes[gi],
            places=0,
            marks=2,
            working=working,
            wrong=wrong,
            sensible_range=(0, n),
        )
        check(takes[gi] == frac * sizes[gi], "stratum count disagrees with the sampling fraction")
        return stem, part

    if method == SIMPLE:
        n = rng.choice([20, 25, 30, 40, 50, 60])
        N = n * rng.randint(4, 14)
        N = max(N, n + 10)
        if level == "easy":
            stem = _stem_for(method, pop, N, n, name, "easy", extra)
            f = Fraction(n, N)
            part = fraction_part(
                prompt=f"What is the probability that a particular {sg} is in the sample?",
                exact=f,
                marks=2,
                working=[f"Each {sg} is equally likely: {n} chances in {N} = {f.numerator}/{f.denominator}."],
                wrong=[
                    (Fraction(N, n), "inverted-ratio"),
                    (Fraction(1, N), "ir2a-chance-of-one-draw-only"),
                    (Fraction(n, N - n), "part-to-part-instead-of-part-to-whole"),
                ],
                sensible_range=(0, 1),
            )
        elif level == "medium":
            # expected members of one group: want n*g/N whole
            for _ in attempts(500, "a group whose expected count is whole"):
                n = rng.choice([20, 25, 30, 40, 50, 60])
                N = n * rng.randint(4, 14)
                g = rng.randrange(N // 10, N // 2)
                if (n * g) % N == 0:
                    break
            stem = (
                _stem_for(method, pop, N, n, name, "plain", extra) + f" {g} of the {N} {pl} are in the youngest group."
            )
            e = n * g // N
            part = number_part(
                prompt=f"About how many of the {n} {pl} in the sample would you expect to be from the youngest group?",
                exact=e,
                places=0,
                marks=2,
                working=[f"The youngest group is {g}/{N} of the {pl}, so expect {g}/{N} × {n} = {e}."],
                wrong=[
                    (g, "ir2a-reports-the-group-size-not-its-share"),
                    (n // 2, "ir2a-equal-allocation-ignores-size"),
                    (Fraction(g * N, n), "inverted-ratio"),
                ],
                sensible_range=(0, n),
            )
        else:
            stem = (
                f"{name} has a numbered register of all {N} {pl} of {pop['org']} and wants every {sg} to have exactly "
                f"the same chance of being chosen, with {n} {pl} to be asked {pop['topic']}."
            )
            f = Fraction(N - n, N)
            part = fraction_part(
                prompt=f"What is the probability that a particular {sg} is NOT chosen?",
                exact=f,
                marks=3,
                working=[
                    f"Chosen with probability {n}/{N}, so not chosen with probability 1 - {n}/{N} = {f.numerator}/{f.denominator}."
                ],
                wrong=[
                    (Fraction(n, N), "ir2a-gives-the-chance-of-being-chosen"),
                    (Fraction(N - n, n), "part-to-part-instead-of-part-to-whole"),
                    (Fraction(N, n), "inverted-ratio"),
                ],
                sensible_range=(0, 1),
            )
        return stem, part

    if method == CENSUS:
        n = rng.choice([20, 25, 30, 40, 50])
        if level == "easy":
            N = rng.choice([60, 80, 90, 120, 150, 200, 240])
            stem = _stem_for(method, pop, N, n, name, "easy", extra)
            part = number_part(
                prompt=f"A sample of {n} {pl} would need {n} interviews. How many more interviews does this survey need?",
                exact=N - n,
                places=0,
                marks=2,
                working=[f"{N} - {n} = {N - n}."],
                wrong=[
                    (N, "ir2a-reports-the-whole-not-the-extra"),
                    (N + n, "ir2a-adds-instead-of-subtracting"),
                    (n, "ir2a-reports-the-sample-size"),
                ],
                sensible_range=(0, None),
            )
        elif level == "medium":
            t = rng.choice([6, 10, 12, 15, 20, 30])
            for _ in attempts(500, "whole hours"):
                N = rng.randint(40, 300)
                if (N * t) % 60 == 0:
                    break
            stem = _stem_for(method, pop, N, n, name, "plain", extra) + f" Each interview takes {t} minutes."
            hrs = N * t // 60
            part = number_part(
                prompt="How many hours of interviewing does the survey need in total?",
                exact=hrs,
                places=0,
                marks=2,
                working=[f"{N} × {t} = {N * t} minutes = {hrs} hours."],
                wrong=[
                    (N * t, "ir2a-leaves-minutes-unconverted"),
                    (Fraction(N * t, 100), "ir2a-converts-minutes-by-100"),
                    (Fraction(n * t, 60), "ir2a-reports-the-sample-size"),
                ],
                sensible_range=(0, None),
            )
        else:
            cost = rng.choice([4, 5, 8, 12, 15, 20])
            N = rng.choice([150, 180, 240, 300, 360, 450, 600])
            stem = (
                f"{name} wants no sampling error at all for the survey of {pop['org']}: the results must describe "
                f"all {N} {pl}, and every {sg} can be contacted at the next meeting. Each completed form costs "
                f"${cost} to process. A sample of {n} {pl} would have cost the same ${cost} a form."
            )
            part = number_part(
                prompt="How much more does the survey cost than the sample would have?",
                exact=(N - n) * cost,
                places=0,
                marks=3,
                working=[f"Extra forms = {N} - {n} = {N - n}.", f"Extra cost = {N - n} × ${cost} = ${(N - n) * cost}."],
                wrong=[
                    (N * cost, "ir2a-reports-the-whole-not-the-extra"),
                    (n * cost, "ir2a-reports-the-sample-size"),
                    ((N + n) * cost, "ir2a-adds-instead-of-subtracting"),
                ],
                sensible_range=(0, None),
            )
        return stem, part

    # convenience
    pct_choices = [(10, 1), (8, 1), (5, 1), (12.5, 1), (20, 1), (25, 1)]
    pct, _ = rng.choice(pct_choices)
    for _ in attempts(500, "a population where the reachable share is whole"):
        N = rng.choice([200, 240, 320, 400, 480, 600, 800, 1000, 1200])
        m = N * pct / 100
        n = rng.choice([20, 25, 30, 40])
        if m == int(m) and m > n:
            m = int(m)
            break
    stem = _stem_for(method, pop, N, n, name, "easy" if level != "hard" else "plain", extra)
    stem += f" Only {m} of the {N} {pl} ever come past {pop['gate']}."
    pctv = Fraction(m * 100, N)
    part = number_part(
        prompt=f"What percentage of the {N} {pl} could possibly have been chosen?",
        exact=pctv,
        places=1,
        marks=2,
        working=[f"{m} / {N} × 100 = {float(pctv):.1f}%."],
        wrong=[
            (Fraction(n * 100, N), "ir2a-uses-the-sample-not-the-reachable-group"),
            (Fraction(m * 100, n), "ir2a-reachable-over-sample"),
            (Fraction((N - m) * 100, N), "ir2a-gives-the-share-who-cannot-be-chosen"),
        ],
        sensible_range=(0, 100),
    )
    return stem, part


def _ordinal(j: int) -> str:
    return {3: "third", 4: "fourth", 5: "fifth"}[j]


# -- tiers ---------------------------------------------------------------------------------------------------------

_EASY_METHODS = (SIMPLE, SYSTEMATIC, STRATIFIED, CONVENIENCE, CENSUS)
_HARD_METHODS = (SIMPLE, SYSTEMATIC, STRATIFIED, CENSUS)


def _two_part(rng, method: str, level: str) -> dict:
    stem, number_p = _build(rng, method, level)
    return chain(stem, _method_part(rng, method, level), number_p)


@generator("ir2a_choose_sampling_method", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_choose_sampling_method(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    if tier == "hard":
        method = rng.choice(_HARD_METHODS)
    else:
        method = rng.choice(_EASY_METHODS)
    return _two_part(rng, method, tier)


# -- exam chain --------------------------------------------------------------------------------------------------


def _exam_systematic(rng: random.Random) -> dict:
    pop = rng.choice(_POPS)
    name = rng.choice(NAMES)
    pl, sg = pop["pl"], pop["sg"]
    k = rng.randint(5, 16)
    n = rng.choice([25, 30, 35, 40, 45, 50, 60])
    N = n * k
    s = rng.randint(2, min(k, 7))
    j = rng.randint(4, 7)
    stem = (
        f"{name} is surveying the {pl} of {pop['org']} about {pop['topic']}. The {N} {pl} are numbered from 1 to {N}. "
        f"{name} chooses number {s} at random as the first {sg}, then surveys every {sg} a fixed number of places "
        f"further along, until {n} {pl} have been chosen."
    )
    last = s + (n - 1) * k
    a = _method_part(rng, SYSTEMATIC, "exam")
    a["marks"] = 1
    b = number_part(
        prompt="How many places apart are the numbers of the people chosen?",
        exact=k,
        places=0,
        marks=2,
        working=[f"{N} / {n} = {k}."],
        wrong=[
            (n, "ir2a-uses-sample-size-as-the-gap"),
            (N - n, "ir2a-gap-from-the-difference"),
            (k + 1, "ir2a-gap-counts-the-start"),
        ],
        sensible_range=(0, None),
    )
    c = number_part(
        prompt=f"What is the number of the {_ordinal_n(j)} {sg} chosen?",
        exact=s + (j - 1) * k,
        places=0,
        marks=2,
        working=[f"{s} + {j - 1} × {k} = {s + (j - 1) * k}."],
        wrong=[
            (s + j * k, "ir2a-counts-one-gap-too-many"),
            (j * k, "ir2a-forgets-the-random-start"),
            (s + (j - 1) * n, "ir2a-uses-sample-size-as-the-gap"),
        ],
        follow_through=f"{s} + {j - 1}*part_b",
        sensible_range=(0, None),
    )
    d = number_part(
        prompt=f"What is the number of the last {sg} chosen?",
        exact=last,
        places=0,
        marks=2,
        working=[f"{s} + ({n} - 1) × {k} = {last}."],
        wrong=[
            (s + n * k, "ir2a-counts-one-gap-too-many"),
            (N, "ir2a-takes-the-end-of-the-list"),
            (n * k, "ir2a-forgets-the-random-start"),
            (s + (n - 2) * k, "ir2a-counts-one-gap-too-few"),
        ],
        follow_through=f"{s} + {n - 1}*part_b",
        sensible_range=(0, None),
    )
    check(last <= N, "last selected is off the list")
    return chain(stem, a, b, c, d)


def _ordinal_n(j: int) -> str:
    return {4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}[j]


def _exam_stratified(rng: random.Random) -> dict:
    pop = rng.choice(_POPS)
    name = rng.choice(NAMES)
    pl = pop["pl"]
    pct = rng.choice([[45, 35, 20], [50, 30, 20], [40, 35, 25], [30, 45, 25], [55, 25, 20], [60, 25, 15]])
    rng.shuffle(pct)
    n = rng.choice([40, 60, 80, 100, 120, 160, 200])
    N = n * rng.randint(5, 12)
    counts = [n * p // 100 for p in pct]
    check(sum(counts) == n and all(n * p % 100 == 0 for p in pct), "percentages do not split the sample")
    groups = pop["groups"][:3]
    desc = ", ".join(f"{p}% in {g}" for p, g in zip(pct, groups))
    stem = (
        f"{name} is surveying {n} of the {N} {pl} of {pop['org']} about {pop['topic']}. The {pl} are in {pop['group_phrase']}: "
        f"{desc}. Within each group, {pl} are chosen by chance, and each group supplies the same share of the sample "
        f"as it has of all the {pl}."
    )
    a = _method_part(rng, STRATIFIED, "exam")
    labels = [_group_label(pop, i) for i in range(3)]
    b = number_part(
        prompt=f"How many {pl} are chosen from {labels[0]}?",
        exact=counts[0],
        places=0,
        marks=2,
        working=[f"{pct[0]}% of {n} = {counts[0]}."],
        wrong=[
            (n // 3, "ir2a-equal-allocation-ignores-size"),
            (N * pct[0] // 100, "ir2a-applies-the-percentage-to-the-population"),
            (pct[0], "ir2a-reports-the-percentage-as-a-count"),
        ],
        sensible_range=(0, n),
    )
    c = number_part(
        prompt=f"How many {pl} are chosen from {labels[1]}?",
        exact=counts[1],
        places=0,
        marks=2,
        working=[f"{pct[1]}% of {n} = {counts[1]}."],
        wrong=[
            (n // 3, "ir2a-equal-allocation-ignores-size"),
            (N * pct[1] // 100, "ir2a-applies-the-percentage-to-the-population"),
            (pct[1], "ir2a-reports-the-percentage-as-a-count"),
        ],
        sensible_range=(0, n),
    )
    d = number_part(
        prompt=f"How many {pl} are chosen from {labels[2]}, so that the sample has {n} in all?",
        exact=counts[2],
        places=0,
        marks=2,
        working=[f"{n} - {counts[0]} - {counts[1]} = {counts[2]}.", f"Check: {pct[2]}% of {n} = {counts[2]}."],
        wrong=[
            (n - counts[0], "ir2a-leftover-ignores-a-group"),
            (n - counts[1], "ir2a-leftover-ignores-a-group"),
            (n // 3, "ir2a-equal-allocation-ignores-size"),
        ],
        follow_through=f"{n} - part_b - part_c",
        sensible_range=(0, n),
    )
    return chain(stem, a, b, c, d)


@generator("exd_x_ir2a_choose_sampling_method", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_choose_sampling_method(rng: random.Random, tier: str = "exam") -> dict:
    """Name the sampling method, then work through a systematic or a proportional stratified sample."""
    check(tier == "exam", "only the exam tier exists")
    return rng.choice((_exam_systematic, _exam_stratified))(rng)
