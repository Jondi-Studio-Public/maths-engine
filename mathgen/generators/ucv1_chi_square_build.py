"""Generates `ucv1_chi_square_test` questions and the exam chain `exd_x_uc_chi_square`.

``ucv1-chi-square-test`` (``statistics``): the arithmetic of a first-year chi-square test, for goodness of fit
(a fair die, equally likely categories, stated proportions or a stated ratio) and for independence in a
contingency table printed in the text. Critical values are always GIVEN in the question and every expected count
is at least 5, the rule a first-year course teaches. A conclusion is only ever the one-mark last part of the exam
chain.

Tiers (topic ``ucv1_chi_square_test``):

* easy   -- one expected count (equal split, stated percentages, or a cell of a 2 x 2 table), to 2 decimal places;
* medium -- the chi-square statistic for a 3 to 5 category goodness-of-fit or a 2 x 2 table, to 2 decimal places;
  or the degrees of freedom of an r x c table, or of a fitted model with estimated parameters;
* hard   -- the statistic for a 2 x 3 table of independence, or for a goodness-of-fit whose expected counts must
  first be built from a ratio such as 9 : 3 : 3 : 1.

The exam tier (topic ``exd_x_uc_chi_square``) is a four-part chain: (a) one expected count, (b) the statistic
(follow-through from (a): only that cell's term changes), (c) the degrees of freedom, (d) the one-mark conclusion
as a numbered choice, using the critical values printed in the stem for the degrees of freedom the learner found.

Every statistic is built from exact Fractions and re-derived with mpmath at 40 digits; `verify_critical_values()`
re-derives the printed chi-square tables from the gamma function with mpmath (the test file calls it).
"""

from __future__ import annotations

import random
from fractions import Fraction

import mpmath

from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    check,
    generator,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# ---------------------------------------------------------------------------
# Upper-tail critical values of chi-square, 3 decimal places, df 1..12.

CHI_5 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307, 11: 19.675, 12: 21.026}  # fmt: skip
CHI_1 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086, 6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209, 11: 24.725, 12: 26.217}  # fmt: skip


def verify_critical_values() -> None:
    """Re-derive every printed critical value from the chi-square distribution (mpmath) at 3 decimals."""
    with mpmath.workdps(30):
        for table, tail in ((CHI_5, "0.05"), (CHI_1, "0.01")):
            for df, printed in table.items():
                a = mpmath.mpf(tail)
                exact = mpmath.findroot(
                    lambda x, df=df, a=a: mpmath.gammainc(mpmath.mpf(df) / 2, x / 2, mpmath.inf, regularized=True) - a,
                    df + 2,
                )
                if abs(float(exact) - printed) > 0.0005 + 1e-9:
                    raise VerificationError(f"chi-square table wrong at df={df}, tail={tail}: {printed} vs {exact}")


# ---------------------------------------------------------------------------
# Helpers


def _places(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def _dp_text(value: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(value, dp), dp)


def _unambiguous(value: Fraction, dp: int) -> bool:
    scaled = abs(value) * 10**dp
    frac = scaled - (scaled.numerator // scaled.denominator)
    return abs(frac - Fraction(1, 2)) > Fraction(1, 10)


def _num_dist(value: Fraction, dp: int, tag: str) -> Distractor:
    text = _dp_text(value, dp)
    return Distractor(text, tag, float(text))


def _int_dist(value: int, tag: str) -> Distractor:
    return Distractor(str(value), tag, value)


def _pick(answer_text: str, cands: list[Distractor], dp: int, n: int = 4) -> list[Distractor]:
    return pick_distractors(answer_text, cands, n, tolerance=Fraction(1, 10**dp))


def _stat(obs, exp) -> Fraction:
    return sum(((Fraction(o) - e) ** 2 / e for o, e in zip(obs, exp)), Fraction(0))


def _mp_stat(obs, exp) -> float:
    return float(
        sum(
            (mpmath.mpf(o) - mpmath.mpf(e.numerator) / e.denominator) ** 2 / (mpmath.mpf(e.numerator) / e.denominator)
            for o, e in zip(obs, exp)
        )
    )


def _check_stat(stat: Fraction, obs, exp, what: str) -> None:
    other = _mp_stat(obs, exp)
    check(abs(float(stat) - other) <= 1e-9 * max(1.0, abs(other)), f"{what}: independent route disagrees")


def _grid(corner: str, header: list[str], rows: list[tuple[str, list[int]]]) -> str:
    head = " — ".join([corner] + header)
    body = "; ".join(" — ".join([name] + [str(v) for v in values]) for name, values in rows)
    return f"Table ({head}): {body}."


def _expected_table(table: list[list[int]]) -> list[list[Fraction]]:
    n = sum(map(sum, table))
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table)]
    return [[Fraction(r * c, n) for c in cols] for r in rows]


# ---------------------------------------------------------------------------
# Contexts

_UNIFORM = [
    dict(
        key="die",
        ks=[6],
        labels=lambda k: [str(i) for i in range(1, k + 1)],
        corner="Face",
        intro="A six-sided die is rolled {n} times to test whether it is fair.",
        easy_ask="Assuming the die is fair, find the expected number of times each face appears",
        h0="the die is fair",
    ),
    dict(
        key="weekday",
        ks=[5],
        labels=lambda k: ["Mon", "Tue", "Wed", "Thu", "Fri"][:k],
        corner="Weekday",
        intro="A council records {n} road crashes on weekdays and tests the claim that a crash is equally likely on each weekday.",
        easy_ask="Find the expected number of crashes on each weekday if the claim is true",
        h0="a crash is equally likely on each weekday",
    ),
    dict(
        key="spinner",
        ks=[4, 5],
        labels=lambda k: ["Red", "Blue", "Green", "Yellow", "Purple"][:k],
        corner="Colour",
        intro="A game spinner has {k} sectors of different colours, said to be equally likely, and is spun {n} times.",
        easy_ask="Find the expected number of spins landing on each colour if the sectors are equally likely",
        h0="the sectors are equally likely",
    ),
    dict(
        key="lollies",
        ks=[3, 4, 5],
        labels=lambda k: ["Red", "Green", "Orange", "Yellow", "Black"][:k],
        corner="Colour",
        intro="A lolly company claims that its {k} colours are equally common. A tub of {n} lollies is sorted by colour.",
        easy_ask="Find the expected number of lollies of each colour if the claim is true",
        h0="the colours are equally common",
    ),
]

_SHARES = {
    3: [(50, 30, 20), (40, 35, 25), (60, 25, 15), (45, 35, 20), (55, 30, 15)],
    4: [(40, 30, 20, 10), (35, 30, 20, 15), (50, 25, 15, 10), (45, 25, 20, 10), (30, 30, 25, 15)],
    5: [(30, 25, 20, 15, 10), (35, 25, 20, 12, 8), (40, 25, 15, 12, 8), (30, 28, 20, 14, 8)],
}

_PROPORTION = [
    dict(
        key="commute",
        labels=["Car", "Train", "Bus", "Walk", "Bike"],
        corner="Mode",
        intro="A council's last survey found these shares of commuters by main mode: {shares}.",
        noun="commuters",
        h0="the shares have not changed",
        verb="use",
    ),
    dict(
        key="canteen",
        labels=["Sandwich", "Wrap", "Salad", "Hot food", "Fruit cup"],
        corner="Order",
        intro="A school canteen's usual lunch orders are: {shares}.",
        noun="orders",
        h0="the usual shares still apply",
        verb="are for",
    ),
    dict(
        key="cars",
        labels=["White", "Grey", "Black", "Blue", "Red"],
        corner="Colour",
        intro="Last year's new cars sold by a Perth dealer had these colour shares: {shares}.",
        noun="new cars",
        h0="the colour shares are unchanged",
        verb="are",
    ),
]

_RATIOS = [
    dict(
        ratio=(9, 3, 3, 1),
        labels=["Purple long", "Purple round", "Red long", "Red round"],
        corner="Phenotype",
        intro="A plant breeder crosses two sweet-pea plants and predicts offspring phenotypes in the ratio 9 : 3 : 3 : 1 (purple long : purple round : red long : red round). In the next generation {n} offspring are classified.",
        h0="the 9 : 3 : 3 : 1 ratio holds",
        n_lo=150,
        n_hi=320,
    ),
    dict(
        ratio=(1, 2, 1),
        labels=["AA", "Aa", "aa"],
        corner="Genotype",
        intro="A genetics model predicts the genotypes of offspring in the ratio 1 : 2 : 1 (AA : Aa : aa). A sample of {n} offspring is classified.",
        h0="the 1 : 2 : 1 ratio holds",
        n_lo=60,
        n_hi=160,
    ),
    dict(
        ratio=(3, 1),
        labels=["Tall", "Short"],
        corner="Height",
        intro="A garden-pea experiment predicts tall and short plants in the ratio 3 : 1. A sample of {n} plants is classified.",
        h0="the 3 : 1 ratio holds",
        n_lo=40,
        n_hi=140,
    ),
    dict(
        ratio=(2, 1, 1),
        labels=["Brown", "Black", "White"],
        corner="Coat",
        intro="A breeder predicts the coat colours of {n} kelpie pups in the ratio 2 : 1 : 1 (brown : black : white).",
        h0="the 2 : 1 : 1 ratio holds",
        n_lo=60,
        n_hi=160,
    ),
]

_TABLES = [
    dict(
        key="travel",
        row_var="year level",
        rows=["Year 11", "Year 12"],
        col_var="way of travelling to school",
        cols3=["Bus", "Walk", "Car"],
        cols2=["Bus", "Walk"],
        corner="Year / Travel",
        intro="A Brisbane school asks students how they travel to school.",
    ),
    dict(
        key="drink",
        row_var="age group",
        rows=["Under 30", "30 and over"],
        col_var="preferred drink",
        cols3=["Tea", "Coffee", "Juice"],
        cols2=["Tea", "Coffee"],
        corner="Age / Drink",
        intro="A café asks customers for their preferred drink.",
    ),
    dict(
        key="sport",
        row_var="region",
        rows=["Coastal", "Inland"],
        col_var="favourite winter sport",
        cols3=["Rugby league", "AFL", "Netball"],
        cols2=["Rugby league", "AFL"],
        corner="Region / Sport",
        intro="A survey of Queensland adults records their favourite winter sport.",
    ),
    dict(
        key="shift",
        row_var="shift",
        rows=["Day", "Night"],
        col_var="grade of finished panel",
        cols3=["Grade A", "Grade B", "Reject"],
        cols2=["Passed", "Rejected"],
        corner="Shift / Grade",
        intro="A factory inspector grades finished panels from each shift.",
    ),
    dict(
        key="trial",
        row_var="treatment",
        rows=["Vaccine", "Placebo"],
        col_var="outcome after one season",
        cols3=["No illness", "Mild illness", "Severe illness"],
        cols2=["Infected", "Not infected"],
        corner="Treatment / Outcome",
        intro="In a clinical trial volunteers are given either a vaccine or a placebo and followed for a season.",
    ),
]


# ---------------------------------------------------------------------------
# Draws


def _draw_uniform(rng: random.Random, *, k=None, stat_range=None, heat=1.0):
    ctx = rng.choice(_UNIFORM)
    k = k if k in ctx["ks"] else rng.choice(ctx["ks"])
    for _ in attempts(500, "a goodness-of-fit sample"):
        base = rng.randint(14, 38)
        spread = max(3, round(heat * base * rng.choice((0.25, 0.4, 0.6))))
        obs = [base + rng.randint(-spread, spread) for _ in range(k)]
        n = sum(obs)
        exp = [Fraction(n, k)] * k
        stat = _stat(obs, exp)
        if min(exp) < 5 or obs == [obs[0]] * k:
            continue
        if stat_range and not (stat_range[0] <= stat <= stat_range[1]):
            continue
        if not _unambiguous(stat, 2):
            continue
        return ctx, k, obs, n, exp, stat
    raise VerificationError("unreachable")  # pragma: no cover


def _share_text(labels, shares):
    return ", ".join(f"{label} {p}%" for label, p in zip(labels, shares))


def _draw_proportions(rng: random.Random, *, k=None, stat_range=None, heat=1.0):
    ctx = rng.choice(_PROPORTION)
    k = k or rng.choice([3, 4, 5])
    for _ in attempts(500, "a proportions sample"):
        shares = rng.choice(_SHARES[k])
        n0 = rng.randint(90, 240)
        obs = []
        for p in shares:
            mid = n0 * p / 100
            width = max(3, round(heat * rng.choice((0.15, 0.3, 0.5)) * mid))
            obs.append(max(1, round(mid) + rng.randint(-width, width)))
        n = sum(obs)
        exp = [Fraction(n * p, 100) for p in shares]
        if min(exp) < 5:
            continue
        stat = _stat(obs, exp)
        if stat_range and not (stat_range[0] <= stat <= stat_range[1]):
            continue
        if not _unambiguous(stat, 2):
            continue
        return ctx, k, shares, obs, n, exp, stat
    raise VerificationError("unreachable")  # pragma: no cover


def _draw_ratio(rng: random.Random, *, stat_range=None, heat=1.0):
    ctx = rng.choice(_RATIOS)
    ratio = ctx["ratio"]
    total = sum(ratio)
    for _ in attempts(500, "a ratio sample"):
        n0 = rng.randint(ctx["n_lo"], ctx["n_hi"])
        obs = []
        for w in ratio:
            mid = n0 * w / total
            width = max(2, round(heat * rng.choice((0.15, 0.3, 0.5)) * mid))
            obs.append(max(1, round(mid) + rng.randint(-width, width)))
        n = sum(obs)
        exp = [Fraction(n * w, total) for w in ratio]
        if min(exp) < 5:
            continue
        stat = _stat(obs, exp)
        if stat_range and not (stat_range[0] <= stat <= stat_range[1]):
            continue
        if not _unambiguous(stat, 2):
            continue
        return ctx, ratio, obs, n, exp, stat
    raise VerificationError("unreachable")  # pragma: no cover


def _draw_table(rng: random.Random, cols: int, *, stat_range=None, heat=1.0):
    ctx = rng.choice(_TABLES)
    for _ in attempts(1000, "a contingency table"):
        base = rng.randint(14, 42)
        spread = max(4, round(heat * base * rng.choice((0.3, 0.5, 0.75))))
        table = [[max(4, base + rng.randint(-spread, spread)) for _ in range(cols)] for _ in range(2)]
        exp = _expected_table(table)
        if min(min(r) for r in exp) < 5:
            continue
        flat_o = [v for r in table for v in r]
        flat_e = [v for r in exp for v in r]
        stat = _stat(flat_o, flat_e)
        if stat_range and not (stat_range[0] <= stat <= stat_range[1]):
            continue
        if not _unambiguous(stat, 2):
            continue
        return ctx, table, exp, stat
    raise VerificationError("unreachable")  # pragma: no cover


def _table_text(ctx, table, cols):
    names = ctx["cols3"] if len(cols) == 3 else ctx["cols2"]
    return _grid(ctx["corner"], names, list(zip(ctx["rows"], table)))


# ---------------------------------------------------------------------------
# Distractor builders for the statistic


def _stat_candidates(obs, exp, *, equal_exp=None, one_total_exp=None):
    """Tagged wrong statistics for observed `obs` and expected `exp` (flat lists)."""
    pairs = list(zip(obs, exp))
    cands = [
        (sum(((Fraction(o) - e) ** 2 / Fraction(o) for o, e in pairs), Fraction(0)), "chi-divides-by-observed"),
        (sum((abs(Fraction(o) - e) / e for o, e in pairs), Fraction(0)), "chi-no-square"),
        (sum(((Fraction(o) - e) ** 2 for o, e in pairs), Fraction(0)), "chi-no-division-by-expected"),
    ]
    if equal_exp is not None:
        cands.append((_stat(obs, equal_exp), "expected-assumes-equal-split"))
    if one_total_exp is not None:
        cands.append((_stat(obs, one_total_exp), "expected-from-one-total-only"))
    cands.append((sum(((Fraction(o) - e) ** 2 / e for o, e in pairs[:-1]), Fraction(0)), "chi-skips-a-category"))
    return cands


# ---------------------------------------------------------------------------
# Easy: one expected count


def _easy_uniform(rng):
    ctx = rng.choice(_UNIFORM)
    k = rng.choice(ctx["ks"])
    for _ in attempts(500, "an unambiguous expected count"):
        n = rng.randint(60, 480)
        exp = Fraction(n, k)
        if _unambiguous(exp, 2):
            break
    answer_text = _dp_text(exp, 2)
    cands = [
        _num_dist(Fraction(n, k - 1), 2, "expected-divides-by-categories-minus-one"),
        _num_dist(Fraction(1, k), 2, "expected-probability-not-count"),
        _num_dist(Fraction(n * (k - 1), k), 2, "expected-complement-count"),
        _num_dist(Fraction(n, k + 1), 2, "expected-divides-by-categories-minus-one"),
    ]
    distractors = _pick(answer_text, cands, 2)
    check(len(distractors) >= 2, "too few distractors")
    intro = ctx["intro"].format(n=n, k=k)
    return dict(
        text=f"{intro} {ctx['easy_ask']}, correct to 2 decimal places.",
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"E = n/k = {n}/{k} = {answer_text}."],
        distractors=distractors,
    )


def _easy_proportion(rng):
    ctx = rng.choice(_PROPORTION)
    k = rng.choice([3, 4, 5])
    shares = rng.choice(_SHARES[k])
    for _ in attempts(500, "an unambiguous expected count"):
        n = rng.randint(80, 400)
        j = rng.randrange(k)
        exp = Fraction(n * shares[j], 100)
        if _unambiguous(exp, 2):
            break
    answer_text = _dp_text(exp, 2)
    p = shares[j]
    cands = [
        _num_dist(Fraction(n, k), 2, "expected-assumes-equal-split"),
        _num_dist(Fraction(p), 2, "expected-percent-used-as-count"),
        _num_dist(Fraction(n * (100 - p), 100), 2, "expected-complement-count"),
        _num_dist(Fraction(p, 100), 2, "expected-probability-not-count"),
    ]
    distractors = _pick(answer_text, cands, 2)
    check(len(distractors) >= 2, "too few distractors")
    labels = ctx["labels"][:k]
    shares_text = _share_text(labels, shares)
    return dict(
        text=(
            f"{ctx['intro'].format(shares=shares_text)} This year {n} {ctx['noun']} are surveyed. "
            f"If {ctx['h0']}, find the expected number of the {n} {ctx['noun']} in the {labels[j]} category, correct to 2 decimal places."
        ),
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"E = n × p = {n} × {p}/100 = {answer_text}."],
        distractors=distractors,
    )


def _easy_cell(rng):
    ctx, table, exp, _stat_value = _draw_table(rng, 2)
    i, j = rng.randrange(2), rng.randrange(2)
    for _ in attempts(500, "a cell with unambiguous rounding"):
        if _unambiguous(exp[i][j], 2):
            break
        ctx, table, exp, _stat_value = _draw_table(rng, 2)
        i, j = rng.randrange(2), rng.randrange(2)
    n = sum(map(sum, table))
    r, c = sum(table[i]), sum(row[j] for row in table)
    answer_text = _dp_text(exp[i][j], 2)
    cands = [
        _num_dist(Fraction(r * c), 2, "expected-ignores-grand-total"),
        _num_dist(Fraction(c, 2), 2, "expected-from-one-total-only"),
        _num_dist(Fraction(r, 2), 2, "expected-from-one-total-only"),
        _num_dist(Fraction(n, 4), 2, "expected-assumes-equal-split"),
        _num_dist(Fraction(r), 2, "expected-from-one-total-only"),
        _num_dist(Fraction(c), 2, "expected-from-one-total-only"),
    ]
    distractors = _pick(answer_text, cands, 2)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=(
            f"{ctx['intro']} Observed counts by {ctx['row_var']} and {ctx['col_var']}: "
            f"{_table_text(ctx, table, ctx['cols2'])} Assuming {ctx['row_var']} and {ctx['col_var']} are independent, "
            f"find the expected count for the cell ({ctx['rows'][i]}, {ctx['cols2'][j]}), correct to 2 decimal places."
        ),
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"E = (row total × column total)/grand total = {r} × {c}/{n} = {answer_text}."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Medium


def _medium_gof(rng):
    if rng.random() < 0.5:
        ctx, k, obs, n, exp, stat = _draw_uniform(rng, k=rng.choice([3, 4, 5, 6]), stat_range=(Fraction(3, 5), 14))
        header = ctx["labels"](k)
        table = _grid(ctx["corner"], header, [("Observed", obs)])
        intro = ctx["intro"].format(n=n, k=k)
        h0 = f"H0: {ctx['h0']}"
        cands = _stat_candidates(obs, exp)
        equal = None
    else:
        ctx, k, shares, obs, n, exp, stat = _draw_proportions(rng, stat_range=(Fraction(3, 5), 14))
        labels = ctx["labels"][:k]
        table = _grid(ctx["corner"], labels, [("Observed", obs)])
        intro = f"{ctx['intro'].format(shares=_share_text(labels, shares))} This year {n} {ctx['noun']} are surveyed."
        h0 = f"H0: {ctx['h0']}"
        equal = [Fraction(n, k)] * k
        cands = _stat_candidates(obs, exp, equal_exp=equal)
    _check_stat(stat, obs, exp, "goodness-of-fit statistic")
    answer_text = _dp_text(stat, 2)
    distractors = _pick(answer_text, [_num_dist(v, 2, tag) for v, tag in cands], 2)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=f"{intro} {table} Test {h0}. Calculate the chi-square statistic, correct to 2 decimal places.",
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            "Expected counts: " + ", ".join(_dp_text(e, 2) for e in exp) + ".",
            f"χ² = Σ(O - E)²/E = {answer_text}.",
        ],
        distractors=distractors,
    )


def _medium_independence(rng):
    ctx, table, exp, stat = _draw_table(rng, 2, stat_range=(Fraction(3, 5), 14))
    flat_o = [v for r in table for v in r]
    flat_e = [v for r in exp for v in r]
    _check_stat(stat, flat_o, flat_e, "2 x 2 statistic")
    cols = [sum(c) for c in zip(*table)]
    rows = [sum(r) for r in table]
    one_total = [Fraction(cols[j], 2) for _ in range(2) for j in range(2)]
    answer_text = _dp_text(stat, 2)
    cands = _stat_candidates(flat_o, flat_e, one_total_exp=one_total)
    distractors = _pick(answer_text, [_num_dist(v, 2, tag) for v, tag in cands], 2)
    check(len(distractors) >= 2, "too few distractors")
    check(rows and cols, "empty table")
    return dict(
        text=(
            f"{ctx['intro']} Observed counts by {ctx['row_var']} and {ctx['col_var']}: {_table_text(ctx, table, ctx['cols2'])} "
            f"Test H0: {ctx['row_var']} and {ctx['col_var']} are independent. "
            "Calculate the chi-square statistic without Yates' continuity correction, correct to 2 decimal places."
        ),
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=["Expected counts: " + ", ".join(_dp_text(e, 2) for e in flat_e) + f". χ² = {answer_text}."],
        distractors=distractors,
    )


_FITTED = [
    (
        "A Poisson model, with its mean estimated from the data, is fitted to the number of emails received per hour, grouped into {k} categories.",
        1,
    ),
    (
        "A binomial model B(6, p), with p estimated from the data, is fitted to the number of defective bolts in packs of six, grouped into {k} categories.",
        1,
    ),
    (
        "A normal model, with its mean and standard deviation both estimated from the data, is fitted to weekly rainfall totals grouped into {k} classes.",
        2,
    ),
]


def _medium_df(rng):
    if rng.random() < 0.55:
        r, c = rng.choice([(2, 3), (3, 3), (3, 4), (4, 3), (2, 5), (4, 4), (3, 5), (5, 3), (2, 4), (4, 5)])
        df = (r - 1) * (c - 1)
        cands = [
            _int_dist(r * c - 1, "df-cells-minus-one"),
            _int_dist(r + c - 2, "df-adds-dimensions"),
            _int_dist(r * c, "df-counts-cells"),
            _int_dist((r - 1) * c, "df-one-dimension-only"),
        ]
        n = rng.randint(120, 480)
        context = rng.choice(_TABLES)
        text = (
            f"{context['intro']} {n} responses are cross-classified in a table with {r} categories of {context['row_var']} "
            f"and {c} categories of {context['col_var']}, and a chi-square test of independence is carried out. "
            "How many degrees of freedom does the test have?"
        )
        working = [f"df = (r - 1)(c - 1) = {r - 1} × {c - 1} = {df}."]
    else:
        template, m = rng.choice(_FITTED)
        k = rng.randint(m + 3, m + 7)
        df = k - 1 - m
        cands = [
            _int_dist(k - 1, "df-ignores-estimated-parameter"),
            _int_dist(k, "df-counts-cells"),
            _int_dist(k - m, "df-subtracts-too-many"),
            _int_dist(k - 1 - 2 * m, "df-subtracts-too-many"),
        ]
        text = (
            template.format(k=k)
            + " A chi-square goodness-of-fit test is carried out. How many degrees of freedom does it have?"
        )
        working = [f"df = categories - 1 - estimated parameters = {k} - 1 - {m} = {df}."]
    answer_text = str(df)
    distractors = pick_distractors(answer_text, [d for d in cands if d.value >= 1], 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text, answer=df, answer_text=answer_text, answer_kind="number", working=working, distractors=distractors
    )


# ---------------------------------------------------------------------------
# Hard


def _hard_independence(rng):
    ctx, table, exp, stat = _draw_table(rng, 3, stat_range=(Fraction(1), 16))
    flat_o = [v for r in table for v in r]
    flat_e = [v for r in exp for v in r]
    _check_stat(stat, flat_o, flat_e, "2 x 3 statistic")
    cols = [sum(c) for c in zip(*table)]
    one_total = [Fraction(cols[j], 2) for _ in range(2) for j in range(3)]
    answer_text = _dp_text(stat, 2)
    cands = _stat_candidates(flat_o, flat_e, one_total_exp=one_total)
    distractors = _pick(answer_text, [_num_dist(v, 2, tag) for v, tag in cands], 2)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=(
            f"{ctx['intro']} Observed counts by {ctx['row_var']} and {ctx['col_var']}: {_table_text(ctx, table, ctx['cols3'])} "
            f"Test H0: {ctx['row_var']} and {ctx['col_var']} are independent. "
            "Calculate the chi-square statistic, correct to 2 decimal places."
        ),
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=["Expected counts: " + ", ".join(_dp_text(e, 2) for e in flat_e) + f". χ² = {answer_text}."],
        distractors=distractors,
    )


def _hard_ratio(rng):
    ctx, ratio, obs, n, exp, stat = _draw_ratio(rng, stat_range=(Fraction(3, 5), 15))
    _check_stat(stat, obs, exp, "ratio goodness-of-fit statistic")
    k = len(ratio)
    equal = [Fraction(n, k)] * k
    answer_text = _dp_text(stat, 2)
    cands = _stat_candidates(obs, exp, equal_exp=equal)
    distractors = _pick(answer_text, [_num_dist(v, 2, tag) for v, tag in cands], 2)
    check(len(distractors) >= 2, "too few distractors")
    table = _grid(ctx["corner"], ctx["labels"], [("Observed", obs)])
    return dict(
        text=(
            f"{ctx['intro'].format(n=n)} {table} Test H0: {ctx['h0']}. "
            "Calculate the chi-square statistic, correct to 2 decimal places."
        ),
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=["Expected counts: " + ", ".join(_dp_text(e, 2) for e in exp) + f". χ² = {answer_text}."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# The single-answer topic


@generator(
    "ucv1_chi_square_test",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_chi_square_test(rng: random.Random, tier: str = "easy") -> dict:
    """Chi-square test arithmetic: an expected count (easy), a statistic or degrees of freedom (medium), a 2 x 3 table
    or ratio-built statistic (hard)."""
    pick = rng.random()
    if tier == "easy":
        return _easy_uniform(rng) if pick < 0.34 else _easy_proportion(rng) if pick < 0.67 else _easy_cell(rng)
    if tier == "medium":
        return _medium_gof(rng) if pick < 0.4 else _medium_independence(rng) if pick < 0.7 else _medium_df(rng)
    if tier == "hard":
        return _hard_independence(rng) if pick < 0.6 else _hard_ratio(rng)
    raise VerificationError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain


def _decision_part(rng: random.Random, reject: bool, level: str, claim: str, proved: str, any_diff: str) -> dict:
    reject_text = f"Reject H0: there is evidence at the {level} level {claim}."
    keep_text = f"Do not reject H0: there is not enough evidence at the {level} level {claim}."
    proved_text = f"Do not reject H0: the data show {proved}."
    any_text = f"Reject H0: the observed counts are not all equal to the expected counts, so {any_diff}."
    if reject:
        right, wrong = (
            reject_text,
            [(keep_text, "decision-compared-wrong-way"), (proved_text, "null-accepted-as-proved")],
        )
    else:
        right, wrong = (
            keep_text,
            [(reject_text, "decision-compared-wrong-way"), (any_text, "any-difference-is-significant")],
        )
    order = [right] + [w[0] for w in wrong]
    rng.shuffle(order)
    position = {text: i + 1 for i, text in enumerate(order)}
    listed = " ".join(f"{i + 1}. {text}" for i, text in enumerate(order))
    return dict(
        prompt=f"State the conclusion of the test. Enter the number of the correct statement. {listed}",
        answer_kind="number",
        answer=position[right],
        answer_text=str(position[right]),
        marks=1,
        working=[f"The correct statement is number {position[right]}: {right}"],
        distractors=[Distractor(str(position[t]), tag, position[t]) for t, tag in wrong],
    )


def _critical_row(level: str, df: int) -> str:
    table = CHI_5 if level == "5%" else CHI_1
    return ", ".join(f"df = {d}: {table[d]:.3f}" for d in range(1, 6))


def _exam_draw(rng: random.Random):
    want_reject = rng.random() < 0.5  # chosen first, so the two conclusions are equally common
    for _ in attempts(1000, "a chi-square exam chain"):
        shape = rng.choice(["table23", "table23", "table22", "gof_prop", "gof_ratio"])
        level = rng.choice(["5%", "5%", "1%"])
        table_c = CHI_5 if level == "5%" else CHI_1
        heat = 2.0 if want_reject else 0.8
        if shape == "table23":
            ctx, table, exp, stat = _draw_table(rng, 3, heat=heat)
            flat_o = [v for r in table for v in r]
            flat_e = [v for r in exp for v in r]
            df = 2
            i, j = rng.randrange(2), rng.randrange(3)
            cell = (i, j)
            idx = i * 3 + j
        elif shape == "table22":
            ctx, table, exp, stat = _draw_table(rng, 2, heat=heat)
            flat_o = [v for r in table for v in r]
            flat_e = [v for r in exp for v in r]
            df = 1
            i, j = rng.randrange(2), rng.randrange(2)
            cell = (i, j)
            idx = i * 2 + j
        elif shape == "gof_prop":
            ctx, k, shares, flat_o, n, flat_e, stat = _draw_proportions(rng, heat=heat)
            df = k - 1
            table, cell = None, None
            idx = rng.randrange(k)
        else:
            ctx, ratio, flat_o, n, flat_e, stat = _draw_ratio(rng, heat=heat)
            df = len(ratio) - 1
            table, cell = None, None
            idx = rng.randrange(len(ratio))
        crit = Fraction(str(table_c[df]))
        is_reject = stat > crit
        if abs(stat - crit) < Fraction(6, 10) or stat > 22 or not _unambiguous(stat, 2):
            continue
        # both conclusions about equally often: the target was chosen first
        if is_reject != want_reject:
            continue
        e_a = flat_e[idx]
        if not _unambiguous(e_a, 2):
            continue
        rest = stat - (Fraction(flat_o[idx]) - e_a) ** 2 / e_a
        e_a2 = round_half_up(e_a, 2)
        stat_ft = rest + (Fraction(flat_o[idx]) - e_a2) ** 2 / e_a2
        if abs(stat - stat_ft) > Fraction(9, 1000):
            continue
        return shape, level, crit, df, ctx, table, cell, flat_o, flat_e, stat, is_reject, idx, rest
    raise VerificationError("unreachable")  # pragma: no cover


class _Retry(Exception):
    """This draw cannot supply enough distinct distractors; the generator draws again."""


@generator("exd_x_uc_chi_square", level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_chi_square(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: an expected count, the statistic, the degrees of freedom, the conclusion (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "a chi-square exam chain with enough distractors"):
        try:
            return _exam_build(rng)
        except _Retry:
            continue
    raise VerificationError("unreachable")  # pragma: no cover


def _exam_build(rng: random.Random) -> dict:
    shape, level, crit, df, ctx, table, cell, obs, exp, stat, is_reject, idx, rest = _exam_draw(rng)
    _check_stat(stat, obs, exp, "exam statistic")
    k = len(obs)
    e_a, o_a = exp[idx], obs[idx]
    a_text, b_text = _dp_text(e_a, 2), _dp_text(stat, 2)

    if shape.startswith("table"):
        cols_n = 3 if shape == "table23" else 2
        names = ctx["cols3"] if cols_n == 3 else ctx["cols2"]
        i, j = cell
        stem = (
            f"{ctx['intro']} Observed counts by {ctx['row_var']} and {ctx['col_var']}: {_table_text(ctx, table, names)} "
            f"A chi-square test of independence of {ctx['row_var']} and {ctx['col_var']} is carried out at the {level} level. "
            f"Upper-tail critical values of χ²: {_critical_row(level, df)}."
            + (" Calculate the statistic without Yates' continuity correction." if cols_n == 2 else "")
        )
        a_prompt = f"Find the expected count for the cell ({ctx['rows'][i]}, {names[j]}), correct to 2 decimal places."
        n = sum(obs)
        r_tot, c_tot = sum(table[i]), sum(row[j] for row in table)
        a_cands = [
            _num_dist(Fraction(r_tot * c_tot), 2, "expected-ignores-grand-total"),
            _num_dist(Fraction(c_tot, 2), 2, "expected-from-one-total-only"),
            _num_dist(Fraction(r_tot, cols_n), 2, "expected-from-one-total-only"),
            _num_dist(Fraction(n, 2 * cols_n), 2, "expected-assumes-equal-split"),
            _num_dist(Fraction(r_tot), 2, "expected-from-one-total-only"),
            _num_dist(Fraction(c_tot), 2, "expected-from-one-total-only"),
        ]
        cols_tot = [sum(c) for c in zip(*table)]
        one_total = [Fraction(cols_tot[jj], 2) for _ in range(2) for jj in range(cols_n)]
        b_cands = _stat_candidates(obs, exp, one_total_exp=one_total)
        c_cands = [
            _int_dist(2 * cols_n - 1, "df-cells-minus-one"),
            _int_dist(2 + cols_n - 2, "df-adds-dimensions"),
            _int_dist(2 * cols_n, "df-counts-cells"),
        ]
        claim = f"of an association between {ctx['row_var']} and {ctx['col_var']}"
        proved = f"that {ctx['row_var']} and {ctx['col_var']} are independent"
        any_diff = f"there is an association between {ctx['row_var']} and {ctx['col_var']}"
        df_working = f"df = (2 - 1)({cols_n} - 1) = {df}."
    else:
        n = sum(obs)
        if shape == "gof_prop":
            labels = ctx["labels"][:k]
            shares = [Fraction(e * 100, n) for e in exp]
            stem = (
                f"{ctx['intro'].format(shares=_share_text(labels, [int(p) for p in shares]))} This year {n} {ctx['noun']} are surveyed. "
                f"{_grid(ctx['corner'], labels, [('Observed', obs)])} A chi-square goodness-of-fit test of H0: {ctx['h0']} is carried out at the {level} level. "
                f"Upper-tail critical values of χ²: {_critical_row(level, df)}."
            )
            label_a = labels[idx]
            a_prompt = f"Find the expected number of {label_a} under H0, correct to 2 decimal places."
            p_a = int(shares[idx])
            a_cands = [
                _num_dist(Fraction(n, k), 2, "expected-assumes-equal-split"),
                _num_dist(Fraction(p_a), 2, "expected-percent-used-as-count"),
                _num_dist(Fraction(n * (100 - p_a), 100), 2, "expected-complement-count"),
            ]
            equal = [Fraction(n, k)] * k
        else:
            labels = ctx["labels"]
            stem = (
                f"{ctx['intro'].format(n=n)} {_grid(ctx['corner'], labels, [('Observed', obs)])} "
                f"A chi-square goodness-of-fit test of H0: {ctx['h0']} is carried out at the {level} level. "
                f"Upper-tail critical values of χ²: {_critical_row(level, df)}."
            )
            label_a = labels[idx]
            a_prompt = f"Find the expected number of {label_a} under H0, correct to 2 decimal places."
            a_cands = [
                _num_dist(Fraction(n, k), 2, "expected-assumes-equal-split"),
                _num_dist(Fraction(ctx["ratio"][idx]), 2, "expected-percent-used-as-count"),
                _num_dist(
                    Fraction(n * (sum(ctx["ratio"]) - ctx["ratio"][idx]), sum(ctx["ratio"])),
                    2,
                    "expected-complement-count",
                ),
                _num_dist(Fraction(ctx["ratio"][idx], sum(ctx["ratio"])), 2, "expected-probability-not-count"),
            ]
            equal = [Fraction(n, k)] * k
        b_cands = _stat_candidates(obs, exp, equal_exp=equal)
        c_cands = [
            _int_dist(k, "df-counts-cells"),
            _int_dist(k - 2, "df-subtracts-too-many"),
            _int_dist(k + 1, "df-counts-cells"),
        ]
        claim = "that the distribution differs from the one stated in H0"
        proved = "that the stated distribution is exactly correct"
        any_diff = "the distribution differs from the one stated in H0"
        df_working = f"df = categories - 1 = {k} - 1 = {df}."

    a_d = _pick(a_text, a_cands, 2, 3)
    b_d = _pick(b_text, [_num_dist(v, 2, tag) for v, tag in b_cands], 2, 3)
    c_d = pick_distractors(str(df), [d for d in c_cands if d.value >= 1], 3)
    if min(len(a_d), len(b_d), len(c_d)) < 2:
        raise _Retry
    d_part = _decision_part(rng, is_reject, level, claim, proved, any_diff)
    rest_text = repr(float(rest))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=a_prompt,
                answer_kind="number",
                answer=float(a_text),
                answer_text=a_text,
                marks=1,
                sensible_range=(0, None),
                working=[f"E = {a_text}."],
                distractors=a_d,
            ),
            dict(
                prompt="Hence calculate the chi-square statistic, correct to 2 decimal places.",
                answer_kind="number",
                answer=float(b_text),
                answer_text=b_text,
                marks=3,
                follow_through=f"{rest_text} + ({o_a} - part_a)**2/part_a",
                working=[f"χ² = Σ(O - E)²/E = {b_text}."],
                distractors=b_d,
            ),
            dict(
                prompt="Write down the number of degrees of freedom.",
                answer_kind="number",
                answer=df,
                answer_text=str(df),
                marks=1,
                working=[df_working],
                distractors=c_d,
            ),
            d_part,
        ],
    )
