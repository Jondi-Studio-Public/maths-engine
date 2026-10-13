"""Generates `ir2a-compare-groups-evidence` questions: is the difference between two groups likely to be real?

Two groups of measurements are compared. Each question states the rule the verdict follows:

    The difference is strong evidence of a real difference only when the difference between the two medians is at
    least as large as the larger of the two interquartile ranges (IQRs). Otherwise it is weak evidence.

and, on the harder tiers, adds: "A group with fewer than 12 values is too small: no judgement can be made."

The learner finds the difference between the medians, the IQRs, and then picks the verdict from five options. Every
option quotes the numbers it relies on, so each is a checkable claim. The wrong options make one named error each:
the right numbers with the opposite verdict, a miscalculated difference, the smaller IQR in place of the larger,
the range in place of the IQR, any difference at all as evidence, or an invented size problem.

Ladder (reasoning escalation, not size):
  easy:   the five-number summaries are given; find the difference in medians, then judge (the IQRs come from Q3 - Q1).
  medium: summaries again, but the learner finds the difference in medians and the larger IQR as two steps, then judges.
  hard:   raw data of 11, 15 or 19 values per group; a group of 11 is too small under the size rule, so the verdict
          may be that no judgement can be made.
  exam:   raw data of 11 values per group (the size rule is 10 values): the difference in medians, each group's IQR,
          then the verdict.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import VerificationError, attempts, check, generator

from ._ir2a_kit import NAMES, chain, mcq_part, median_of, number_part, quartiles

_TIERS = ("easy", "medium", "hard")

STRONG, WEAK, NONE = "strong", "weak", "none"

_CONTEXTS = (
    dict(
        measure="time to finish a puzzle (seconds)",
        a="students who practised first",
        b="students who did not",
        lo=70,
        hi=150,
    ),
    dict(measure="words typed in one minute", a="the morning class", b="the afternoon class", lo=30, hi=75),
    dict(measure="height after eight weeks (cm)", a="plants kept in sunlight", b="plants kept in shade", lo=15, hi=60),
    dict(measure="battery life (hours)", a="brand A batteries", b="brand B batteries", lo=8, hi=30),
    dict(
        measure="time to cycle to school (minutes)",
        a="students using the cycle path",
        b="students using the road",
        lo=10,
        hi=40,
    ),
)


def _summary(vals):
    q1, med, q3 = quartiles(vals)
    return dict(lo=min(vals), q1=q1, med=med, q3=q3, hi=max(vals), iqr=q3 - q1, n=len(vals))


def _verdict(d, L, na, nb, min_n):
    if min(na, nb) < min_n:
        return NONE
    return STRONG if d >= L else WEAK


def _draw(rng, ctx, na, nb, want, min_n):
    """Two raw data sets whose verdict is `want`, with whole-number medians, quartiles and distinct IQRs."""
    lo, hi = ctx["lo"], ctx["hi"]
    for _ in attempts(3000, "two groups with the wanted verdict"):
        shift = rng.randint(2, max(3, (hi - lo) // 3))
        a = [rng.randint(lo, hi - shift) for _ in range(na)]
        b = [rng.randint(lo + shift, hi) for _ in range(nb)]
        sa, sb = _summary(a), _summary(b)
        if any(v.denominator != 1 for v in (sa["med"], sb["med"], sa["iqr"], sb["iqr"])):
            continue
        d = abs(sa["med"] - sb["med"])
        L, S = max(sa["iqr"], sb["iqr"]), min(sa["iqr"], sb["iqr"])
        if d == 0 or sa["iqr"] == sb["iqr"] or d == L:
            continue
        if _verdict(d, L, na, nb, min_n) != want:
            continue
        if want == WEAK and d < S:
            continue  # keep a weak case where the smaller IQR would mislead
        return a, b, sa, sb, int(d), int(L), int(S)
    raise VerificationError("no draw with the wanted verdict")


def _texts(d, L, S, min_n):
    return {
        STRONG: f"Strong evidence of a real difference: the medians differ by {d}, which is at least the larger interquartile range, {L}.",
        WEAK: f"Weak evidence of a real difference: the medians differ by {d}, which is less than the larger interquartile range, {L}.",
        NONE: f"No judgement can be made: one group has fewer than {min_n} values.",
    }


def _verdict_part(rng, key, d, L, S, d_wrong, min_n, marks, allow_none):
    t = _texts(d, L, S, min_n)
    opposite = WEAK if key == STRONG else STRONG
    wrongs = [(t[opposite], "ir2a-misjudges-the-difference-against-the-spread")]
    same = _texts(d_wrong, L, S, min_n)[key] if key != NONE else None
    if same and same != t[key]:
        wrongs.append((same, "ir2a-uses-a-miscalculated-difference"))
    wrongs.append(
        (
            f"Strong evidence of a real difference: the medians differ by {d}, which is at least the smaller interquartile range, {S}."
            if key != STRONG
            else f"Weak evidence of a real difference: the ranges of the two groups overlap.",
            "ir2a-compares-with-the-smaller-spread" if key != STRONG else "ir2a-uses-the-range-not-the-iqr",
        )
    )
    wrongs.append(
        (
            f"Strong evidence of a real difference: the two medians are not equal, they differ by {d}.",
            "ir2a-any-difference-counts-as-evidence",
        )
    )
    if allow_none and key != NONE:
        wrongs.append((t[NONE], "ir2a-invents-a-size-problem"))
    if key == NONE:
        wrongs.append((t[STRONG], "ir2a-ignores-the-size-rule"))
        wrongs.append((t[WEAK], "ir2a-ignores-the-size-rule"))
    return mcq_part(
        "Which statement is the correct verdict?",
        t[key],
        wrongs,
        marks=marks,
        working=[f"Difference in medians = {d}; larger IQR = {L}; smaller IQR = {S}.", t[key]],
        keep=4,
    )


def _wrong_diff(a, b, sa, sb):
    """A plausible miscalculated difference: the difference of means, or of the maxima."""
    cand = [
        abs(round(sum(a) / len(a)) - round(sum(b) / len(b))),
        abs(sa["hi"] - sb["hi"]),
        abs(sa["hi"] - sa["lo"] - (sb["hi"] - sb["lo"])),
    ]
    return cand


def _rule(min_n):
    base = (
        "The rule: the difference is strong evidence of a real difference only when the difference between the two "
        "medians is at least as large as the larger of the two interquartile ranges (IQRs); otherwise it is weak evidence."
    )
    if min_n:
        base += f" A group with fewer than {min_n} values is too small, and no judgement can be made."
    return base


def _the(group: str) -> str:
    return group if group.startswith("the ") else f"the {group}"


def _fmt(vals):
    return ", ".join(str(v) for v in vals)


def _summaries_text(ctx, sa, sb):
    def one(label, s):
        return (
            f"{label}: smallest {s['lo']}, lower quartile {int(s['q1'])}, median {int(s['med'])}, upper quartile {int(s['q3'])}, "
            f"largest {s['hi']} ({s['n']} values)"
        )

    return f"{one(ctx['a'].capitalize(), sa)}. {one(ctx['b'].capitalize(), sb)}."


def _number_parts(sa, sb, ctx):
    d = int(abs(sa["med"] - sb["med"]))
    L = int(max(sa["iqr"], sb["iqr"]))
    pa = number_part(
        prompt="What is the difference between the medians of the two groups? Give the larger minus the smaller.",
        exact=d,
        places=0,
        marks=1,
        working=[f"Medians: {int(sa['med'])} and {int(sb['med'])}.", f"Difference = {d}."],
        wrong=[
            (int(sa["med"] + sb["med"]), "ir2a-adds-instead-of-subtracting"),
            (abs(sa["hi"] - sb["hi"]), "ir2a-compares-the-largest-values"),
            (abs(sa["q3"] - sb["q3"]), "ir2a-compares-upper-quartiles-not-medians"),
            (abs(sa["lo"] - sb["lo"]), "ir2a-compares-the-smallest-values"),
        ],
        sensible_range=(0, None),
    )
    pb = number_part(
        prompt="What is the larger of the two interquartile ranges?",
        exact=L,
        places=0,
        marks=2,
        working=[f"IQRs: {int(sa['iqr'])} and {int(sb['iqr'])}.", f"Larger = {L}."],
        wrong=[
            (int(min(sa["iqr"], sb["iqr"])), "ir2a-takes-the-smaller-spread"),
            (max(sa["hi"] - sa["lo"], sb["hi"] - sb["lo"]), "ir2a-uses-the-range-not-the-iqr"),
            (int(sa["iqr"] + sb["iqr"]), "ir2a-adds-instead-of-subtracting"),
        ],
        sensible_range=(0, None),
    )
    return pa, pb, d, L


def _build_summary(rng, level):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    n = rng.choice([11, 15])
    want = rng.choice([STRONG, WEAK])
    a, b, sa, sb, d, L, S = _draw(rng, ctx, n, n, want, 0)
    stem = f"{name} compared the {ctx['measure']} of two groups. {_summaries_text(ctx, sa, sb)} {_rule(0)}"
    pa, pb, d, L = _number_parts(sa, sb, ctx)
    wrong_d = [w for w in _wrong_diff(a, b, sa, sb) if w != d][:1] or [d + 3]
    v = _verdict_part(rng, want, d, L, S, wrong_d[0], 0, 2, False)
    if level == "easy":
        return chain(stem, pa, v)
    return chain(stem, pa, pb, v)


def _build_raw(rng, level):
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    min_n = 12
    if rng.random() < 0.3:
        na, nb = rng.choice([(11, 15), (15, 11), (11, 19), (19, 11), (11, 11)])
    else:
        na, nb = rng.choice([(15, 15), (15, 19), (19, 15), (19, 19)])
    a, b, sa, sb, d, L, S = _draw(rng, ctx, na, nb, rng.choice([STRONG, WEAK]), 0)
    stem = (
        f"{name} measured the {ctx['measure']} for two groups. {ctx['a'].capitalize()} ({na} values): {_fmt(a)}. "
        f"{ctx['b'].capitalize()} ({nb} values): {_fmt(b)}. {_rule(min_n)}"
    )
    pa, pb, d, L = _number_parts(sa, sb, ctx)
    key = _verdict(d, L, na, nb, min_n)
    wrong_d = [w for w in _wrong_diff(a, b, sa, sb) if w != d][:1] or [d + 3]
    v = _verdict_part(rng, key, d, L, S, wrong_d[0], min_n, 2, True)
    return chain(stem, pa, pb, v)


@generator("ir2a_compare_groups_evidence", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_compare_groups_evidence(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    if tier == "hard":
        return _build_raw(rng, "hard")
    return _build_summary(rng, tier)


@generator("exd_x_ir2a_compare_groups_evidence", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_compare_groups_evidence(rng: random.Random, tier: str = "exam") -> dict:
    """The difference in medians, each group's interquartile range, then the verdict."""
    check(tier == "exam", "only the exam tier exists")
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(NAMES)
    min_n = 10
    want = rng.choice([STRONG, WEAK])
    a, b, sa, sb, d, L, S = _draw(rng, ctx, 11, 11, want, min_n)
    stem = (
        f"{name} measured the {ctx['measure']} for two groups. {ctx['a'].capitalize()} (11 values): {_fmt(a)}. "
        f"{ctx['b'].capitalize()} (11 values): {_fmt(b)}. {_rule(min_n)}"
    )
    pa = number_part(
        prompt="What is the difference between the medians of the two groups? Give the larger minus the smaller.",
        exact=d,
        places=0,
        marks=2,
        working=[f"Medians: {int(sa['med'])} and {int(sb['med'])}; difference {d}."],
        wrong=[
            (int(sa["med"] + sb["med"]), "ir2a-adds-instead-of-subtracting"),
            (abs(sa["hi"] - sb["hi"]), "ir2a-compares-the-largest-values"),
            (abs(sa["q3"] - sb["q3"]), "ir2a-compares-upper-quartiles-not-medians"),
            (abs(sa["lo"] - sb["lo"]), "ir2a-compares-the-smallest-values"),
        ],
        sensible_range=(0, None),
    )
    pb = number_part(
        prompt=f"What is the interquartile range of {_the(ctx['a'])}?",
        exact=int(sa["iqr"]),
        places=0,
        marks=2,
        working=[f"Q1 = {int(sa['q1'])}, Q3 = {int(sa['q3'])}, IQR = {int(sa['iqr'])}."],
        wrong=[
            (sa["hi"] - sa["lo"], "ir2a-uses-the-range-not-the-iqr"),
            (int(sa["q3"]), "ir2a-gives-the-upper-quartile-not-the-iqr"),
            (int(sa["med"] - sa["q1"]), "ir2a-halves-the-iqr"),
        ],
        sensible_range=(0, None),
    )
    pc = number_part(
        prompt=f"What is the interquartile range of {_the(ctx['b'])}?",
        exact=int(sb["iqr"]),
        places=0,
        marks=2,
        working=[f"Q1 = {int(sb['q1'])}, Q3 = {int(sb['q3'])}, IQR = {int(sb['iqr'])}."],
        wrong=[
            (sb["hi"] - sb["lo"], "ir2a-uses-the-range-not-the-iqr"),
            (int(sb["q3"]), "ir2a-gives-the-upper-quartile-not-the-iqr"),
            (int(sb["med"] - sb["q1"]), "ir2a-halves-the-iqr"),
        ],
        sensible_range=(0, None),
    )
    gap = L - d
    pgap = number_part(
        prompt="How much larger is the larger interquartile range than the difference between the medians? Use a negative number if it is smaller.",
        exact=gap,
        places=0,
        marks=2,
        working=[f"Larger IQR = {L}; difference in medians = {d}.", f"{L} - {d} = {gap}."],
        wrong=[
            (-gap, "ir2a-subtracts-in-the-wrong-order"),
            (L + d, "ir2a-adds-instead-of-subtracting"),
            (int(min(sa["iqr"], sb["iqr"])) - d, "ir2a-takes-the-smaller-spread"),
        ],
        follow_through="Max(part_b, part_c) - part_a",
        sensible_range=None,
    )
    wrong_d = [w for w in _wrong_diff(a, b, sa, sb) if w != d][:1] or [d + 3]
    pd = _verdict_part(rng, want, d, L, S, wrong_d[0], min_n, 2, False)
    return chain(stem, pa, pb, pc, pgap, pd)
