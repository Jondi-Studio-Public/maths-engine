"""Generates `f10-random-versus-convenience-sample` questions (AC9M8ST02): read two samples against their population.

A school of 100 or 200 students has a recorded variable. The stem gives the population as a table (how many students
at each value, or in each category) and two samples of it: a random sample and a convenience sample ("the first ten
students out of the gym"). The learner works out a sample's mean (or its percentage) and how far it is from the
population value. The written report on a sample stays with the rubric skills; nothing here asks for an explanation.

Both samples are real draws without replacement from the expanded population, so the values listed are possible.
The random sample is a uniform draw; the convenience sample is a draw weighted towards one end of the variable (or
towards one category), which is what makes it biased. Every number is exact: the population mean has at most two
decimal places, a sample mean (ten values) one, and a percentage (a sample of twenty from 100 or 200) is a whole number.

Ladder (reasoning escalation, not size):
  easy:   one quantity from the data: a sample's mean or percentage, or the population's from the table.
  medium: the gap between one sample's value and the population's (the sample, then the population, then subtract).
  hard:   two samples at once: how much closer one sample is than the other; how far the two samples pooled sit from the
          population; or how far the whole-school total estimated from the convenience sample is from the true total.
  exam:   a four-part chain on one scenario: the population value, each sample's gap, then the closer-by or the
          whole-school error (follow-through from the earlier parts).

Every key is re-derived from the expanded population and the listed samples with ``statistics.fmean`` as a float check.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator, round_half_up
from ._company_kit import keep_company, numeric_pool
from ._r3j_kit import number_part as _number_part
from ._r3j_kit import sane
from ._r3j_kit import single_number as _single_number

TOPIC = "ds_sample_vs_pop_gap"
EXAM_TOPIC = "ds_sample_vs_pop_exam"
_TIERS = ("easy", "medium", "hard")


# -- wrong answers that keep the key company -------------------------------------------------------------------------
def _company(rng, key, places: int, slips: list) -> list:
    """`slips` as (value, tag) in an order that leaves the key with company in every visible feature (digits, whole or
    not, nearby options), topped up from plain near misses when the slips alone leave it the odd one out."""
    key = round_half_up(Fraction(key), places)
    unit = Fraction(1, 10**places)
    seen, ordered = {key}, []
    for value, tag in sane(slips, key, 15):
        shown = round_half_up(Fraction(value), places)
        if shown in seen or shown <= 0 or abs(shown - key) < unit:
            continue
        seen.add(shown)
        ordered.append((shown, tag))
    step = Fraction(1) if places == 0 else Fraction(1, 2) if places == 1 else Fraction(1, 10)
    top = max([key] + [v for v, _ in ordered]) * 2
    pool = numeric_pool(
        key, step=step, lo=unit, hi=top, ok=lambda v: (v * 10**places).denominator == 1 and abs(v - key) >= unit
    )
    # the key's rank among the options is drawn uniformly: how many of the shown wrong options lie below it
    below = [c for c in ordered + pool if c[0] < key]
    above = [c for c in ordered + pool if c[0] > key]
    low = min(rng.randrange(4), len(below))
    low = max(low, 3 - len(above))
    chosen = below[:low] + above[: 3 - low]
    rest = [c for c in ordered if c not in chosen]
    return keep_company(key, chosen + rest, pool, 3, places=places, max_pool=24)


def single_number(rng, text, value, places, working, slips, **kw):
    return _single_number(text, value, places, working, _company(rng, value, places, slips), **kw)


def number_part(rng, prompt, value, places, marks, working, slips, **kw):
    return _number_part(prompt, value, places, marks, working, _company(rng, value, places, slips), **kw)


# -- contexts ------------------------------------------------------------------------------------------------------
# A numerical variable: (quantity for a question, table heading, lo, hi, step, direction of the bias, the convenience
# sample as a phrase, what the total of the variable is called, the lead sentence).
_MEAN_CTX = (
    dict(
        qty="number of hours of sport played last week",
        head="Hours of sport (number of students)",
        lo=0,
        hi=8,
        step=1,
        bias="high",
        conv="the first {n} students out of the gym at lunchtime",
        total="hours of sport",
        lead="All {N} students at a school recorded how many hours of sport they played last week.",
    ),
    dict(
        qty="number of books read this term",
        head="Books read (number of students)",
        lo=0,
        hi=9,
        step=1,
        bias="high",
        conv="the first {n} students through the library door at lunchtime",
        total="books",
        lead="All {N} students at a school recorded how many books they read this term.",
    ),
    dict(
        qty="number of minutes spent travelling to school",
        head="Minutes travelling (number of students)",
        lo=5,
        hi=40,
        step=5,
        bias="high",
        conv="the first {n} students off the bus from the outer suburbs",
        total="minutes of travel",
        lead="All {N} students at a school recorded how long their trip to school takes.",
    ),
    dict(
        qty="number of hours of screen time yesterday",
        head="Hours of screen time (number of students)",
        lo=0,
        hi=7,
        step=1,
        bias="high",
        conv="the first {n} students in the computer room at lunchtime",
        total="hours of screen time",
        lead="All {N} students at a school recorded their screen time yesterday.",
    ),
    dict(
        qty="number of days a week the student rides a bike",
        head="Days riding a bike (number of students)",
        lo=0,
        hi=6,
        step=1,
        bias="low",
        conv="the first {n} students waiting at the bus stop after school",
        total="days of riding",
        lead="All {N} students at a school recorded how many days a week they ride a bike.",
    ),
)
_RANDOM_HOW = (
    "names drawn from a hat",
    "chosen by a random number generator from the student list",
    "student numbers picked by a random number generator",
)
# A categorical variable: categories and the targets asked about, each with the verb phrase for "the students who ...",
# the convenience sample, and whether that sample over- or under-represents the target.
_PROP_CTX = (
    dict(
        head="How students travel to school (number of students)",
        cats=("Walk", "Bus", "Car", "Bike"),
        lead="A school has {N} students. Each was asked how they usually travel to school.",
        targets={
            "Bike": ("cycle to school", "the first {n} students into the bike shed", "high"),
            "Car": ("arrive by car", "the first {n} students at the drop-off zone", "high"),
            "Walk": ("walk to school", "the first {n} students off the bus", "low"),
        },
    ),
    dict(
        head="Where students get lunch (number of students)",
        cats=("Canteen", "Packed", "Shops", "Home"),
        lead="A school has {N} students. Each was asked where they usually get lunch.",
        targets={
            "Canteen": ("buy lunch at the canteen", "the first {n} students in the canteen queue", "high"),
            "Packed": ("bring a packed lunch", "the first {n} students in the canteen queue", "low"),
        },
    ),
    dict(
        head="Main after-school activity (number of students)",
        cats=("Sport", "Music", "Art", "None"),
        lead="A school has {N} students. Each was asked about their main activity after school.",
        targets={
            "Sport": ("play sport after school", "the first {n} students at the oval after school", "high"),
            "Music": ("have a music lesson", "the first {n} students waiting outside the music room", "high"),
        },
    ),
)


# -- the scenario --------------------------------------------------------------------------------------------------
@dataclass
class Scenario:
    kind: str  # "mean" or "prop"
    N: int
    n: int
    ctx: dict
    table: str
    lead: str
    pop: Fraction  # population mean, or percentage
    rnd: Fraction  # random sample's mean, or percentage
    cnv: Fraction
    rnd_line: str
    cnv_line: str
    rnd_raw: tuple  # values (mean) or the count of the target (prop)
    cnv_raw: tuple
    table_raw: dict  # value or category -> number of students
    who: str = ""
    rows: int = 0

    @property
    def g_rnd(self) -> Fraction:
        return abs(self.rnd - self.pop)

    @property
    def g_cnv(self) -> Fraction:
        return abs(self.cnv - self.pop)

    @property
    def sample_noun(self) -> str:
        return "mean" if self.kind == "mean" else "percentage"


def _pick_weighted(rng: random.Random, items: list, weights: list, count: int) -> list:
    """`count` items drawn without replacement, each draw proportional to the integer weights."""
    items, weights = list(items), list(weights)
    chosen = []
    for _ in range(count):
        point = rng.randrange(sum(weights))
        for i, w in enumerate(weights):
            if point < w:
                chosen.append(items.pop(i))
                weights.pop(i)
                break
            point -= w
    return chosen


def _frequencies(rng: random.Random, rows: int, N: int) -> list[int]:
    """`rows` counts that sum to N, each at least 4% of N, with a single hump so the table looks like real data."""
    peak = rng.randrange(rows)
    raw = [rng.randint(3, 6) + max(0, 8 - 2 * abs(i - peak)) for i in range(rows)]
    total = sum(raw)
    counts = [max(N // 25, N * r // total) for r in raw]
    i = 0
    while sum(counts) != N:
        j = (peak + i) % rows
        counts[j] += 1 if sum(counts) < N else -1
        if counts[j] < N // 25:
            counts[j] = N // 25
        i += 1
        check(i < 10 * N, "frequencies would not sum")
    return counts


def _mean_scenario(rng: random.Random) -> Scenario:
    ctx = rng.choice(_MEAN_CTX)
    N, n = rng.choice((100, 200)), 10
    values = list(range(ctx["lo"], ctx["hi"] + 1, ctx["step"]))
    span = ctx["hi"] - ctx["lo"]
    for _ in attempts(3000, "a population and two samples with exact, well-separated means"):
        counts = _frequencies(rng, len(values), N)
        pop_list = [v for v, c in zip(values, counts) for _ in range(c)]
        pop = Fraction(sum(pop_list), N)
        if (pop * 100).denominator != 1:
            continue
        if ctx["bias"] == "high":
            weights = [1 + ((v - ctx["lo"]) // ctx["step"]) ** 2 for v in pop_list]
        else:
            weights = [1 + ((ctx["hi"] - v) // ctx["step"]) ** 2 for v in pop_list]
        r_vals = rng.sample(pop_list, n)
        c_vals = _pick_weighted(rng, pop_list, weights, n)
        rnd, cnv = Fraction(sum(r_vals), n), Fraction(sum(c_vals), n)
        up = 1 if ctx["bias"] == "high" else -1
        if (cnv - pop) * up < Fraction(15 * span, 100):
            continue
        if rnd == 0 or cnv == 0 or abs(rnd - pop) < Fraction(ctx["step"], 20) or abs(rnd - pop) == abs(cnv - pop):
            continue
        if len({abs(rnd - pop), abs(cnv - pop), abs(rnd - cnv), abs(rnd - pop) + abs(cnv - pop)}) < 4:
            continue
        r_vals, c_vals = sorted(r_vals), sorted(c_vals)
        break
    # a second way: the mean of the expanded population by the standard library
    check(abs(statistics.fmean(pop_list) - float(pop)) < 1e-9, "population mean differs from statistics.fmean")
    check(abs(statistics.fmean(r_vals) - float(rnd)) < 1e-9, "random sample mean differs from statistics.fmean")
    table = ", ".join(f"{v} ({c})" for v, c in zip(values, counts))
    how = rng.choice(_RANDOM_HOW)
    return Scenario(
        kind="mean",
        N=N,
        n=n,
        ctx=ctx,
        table=f"{ctx['head']}: {table}.",
        lead=ctx["lead"].format(N=N),
        pop=pop,
        rnd=rnd,
        cnv=cnv,
        rnd_line=f"Random sample ({n} students, {how}): {', '.join(map(str, r_vals))}.",
        cnv_line=f"Convenience sample ({ctx['conv'].format(n=n)}): {', '.join(map(str, c_vals))}.",
        rnd_raw=tuple(r_vals),
        cnv_raw=tuple(c_vals),
        table_raw=dict(zip(values, counts)),
        rows=len(values),
    )


def _prop_scenario(rng: random.Random) -> Scenario:
    ctx = rng.choice(_PROP_CTX)
    target = rng.choice(sorted(ctx["targets"]))
    who, conv, bias = ctx["targets"][target]
    N, n = rng.choice((100, 200, 300, 400, 500)), 20
    cats = ctx["cats"]
    for _ in attempts(3000, "a population and two samples with whole-number percentages"):
        pct = rng.randint(12, 45)
        target_count = pct * N // 100
        rest = _frequencies(rng, len(cats) - 1, N - target_count)
        counts = {c: k for c, k in zip([c for c in cats if c != target], rest)}
        counts[target] = target_count
        if min(counts.values()) < N // 25:
            continue
        pop_list = [c for c in cats for _ in range(counts[c])]
        hit = 4
        weights = [hit if (c == target) == (bias == "high") else 1 for c in pop_list]
        r_s = rng.sample(pop_list, n)
        c_s = _pick_weighted(rng, pop_list, weights, n)
        kr, kc = r_s.count(target), c_s.count(target)
        rnd, cnv = Fraction(100 * kr, n), Fraction(100 * kc, n)
        up = 1 if bias == "high" else -1
        pop = Fraction(pct)
        if (cnv - pop) * up < 15:
            continue
        if rnd == 0 or cnv == 0 or abs(rnd - pop) < 5 or abs(rnd - pop) == abs(cnv - pop):
            continue
        if len({abs(rnd - pop), abs(cnv - pop), abs(rnd - cnv), abs(rnd - pop) + abs(cnv - pop)}) < 4:
            continue
        break
    check(Fraction(100 * pop_list.count(target), N) == pop, "population percentage differs from the expanded list")

    def line(sample):
        return ", ".join(f"{c} {sample.count(c)}" for c in cats)

    how = rng.choice(_RANDOM_HOW)
    return Scenario(
        kind="prop",
        N=N,
        n=n,
        ctx=ctx,
        table=f"{ctx['head']}: " + ", ".join(f"{c} ({counts[c]})" for c in cats) + ".",
        lead=ctx["lead"].format(N=N),
        pop=pop,
        rnd=rnd,
        cnv=cnv,
        rnd_line=f"Random sample ({n} students, {how}): {line(r_s)}.",
        cnv_line=f"Convenience sample ({conv.format(n=n)}): {line(c_s)}.",
        rnd_raw=(kr,),
        cnv_raw=(kc,),
        table_raw=counts | {"target": target},
        who=who,
        rows=len(cats),
    )


def _scenario(rng: random.Random, kind: str | None = None) -> Scenario:
    kind = kind or rng.choice(("mean", "prop"))
    return _mean_scenario(rng) if kind == "mean" else _prop_scenario(rng)


def _stem(s: Scenario, question: str, *, intro_extra: str = "") -> str:
    return f"{s.lead} {s.table} {s.rnd_line} {s.cnv_line} {intro_extra}{question}".replace("  ", " ")


# -- numbers and wording that depend on the kind --------------------------------------------------------------------
def _places_est(s: Scenario) -> int:
    return 1 if s.kind == "mean" else 0


def _places_pop(s: Scenario) -> int:
    return 2 if s.kind == "mean" else 0


def _places_gap(s: Scenario) -> int:
    return 2 if s.kind == "mean" else 0


def _quantity(s: Scenario) -> str:
    return "mean" if s.kind == "mean" else "percentage of students who " + s.who


def _of_sample(s: Scenario, which: str) -> str:
    return f"the {which} sample's" + (" mean" if s.kind == "mean" else " percentage")


def _unit_gap(s: Scenario) -> str:
    return "" if s.kind == "mean" else " in percentage points"


def _rf(s, which):
    return s.rnd if which == "random" else s.cnv


def _gf(s, which):
    return s.g_rnd if which == "random" else s.g_cnv


def _raw_sum(s, which):
    return sum(s.rnd_raw if which == "random" else s.cnv_raw)


def _pop_name(s):
    return "the population mean" if s.kind == "mean" else "the percentage for the whole school"


# -- the question shapes ---------------------------------------------------------------------------------------------
def _sample_estimate(rng, s: Scenario) -> dict:
    which = rng.choice(("random", "convenience"))
    other = "convenience" if which == "random" else "random"
    est, other_est = _rf(s, which), _rf(s, other)
    if s.kind == "mean":
        q = f"What is the mean of the {which} sample?"
        slips = [
            (Fraction(_raw_sum(s, which)), "ds-svp-total-not-mean"),
            (Fraction(_raw_sum(s, which), s.n - 1), "ds-svp-divides-by-wrong-count"),
            (s.pop, "ds-svp-uses-population-value"),
            (other_est, "ds-svp-other-sample"),
        ]
        working = [
            f"Add the {s.n} values: {_raw_sum(s, which)}.",
            f"Divide by {s.n}: {_raw_sum(s, which)} / {s.n} = {float(est):g}.",
        ]
    else:
        q = f"What percentage of the {which} sample {s.who}?"
        k = (s.rnd_raw if which == "random" else s.cnv_raw)[0]
        slips = [
            (Fraction(k), "ds-svp-count-not-percent"),
            (Fraction(100 * k, s.N), "ds-svp-percent-of-population-size"),
            (s.pop, "ds-svp-uses-population-value"),
            (other_est, "ds-svp-other-sample"),
        ]
        working = [f"{k} of the {s.n} students {s.who}.", f"{k} / {s.n} x 100 = {int(est)}%."]
    return single_number(rng, _stem(s, q), est, _places_est(s), working, slips)


def _population_value(rng, s: Scenario) -> dict:
    if s.kind == "mean":
        total = sum(v * c for v, c in s.table_raw.items())
        plain = Fraction(sum(s.table_raw), s.rows)
        mode_value = max(s.table_raw, key=lambda v: s.table_raw[v])
        q = "What is the mean of the whole school?"
        slips = [
            (plain, "ds-svp-ignores-frequencies"),
            (Fraction(total), "ds-svp-total-not-mean"),
            (Fraction(total, s.rows), "ds-svp-divides-by-number-of-rows"),
            (Fraction(mode_value), "ds-svp-most-common-for-mean"),
        ]
        working = [
            f"Multiply each value by its number of students and add: {total}.",
            f"Divide by the {s.N} students: {total} / {s.N} = {float(s.pop):g}.",
        ]
    else:
        target = s.table_raw["target"]
        k = s.table_raw[target]
        q = f"What percentage of all the students in the school {s.who}?"
        slips = [
            (Fraction(k), "ds-svp-count-not-percent"),
            (Fraction(100 * k, s.n), "ds-svp-percent-of-sample-size"),
            (100 - s.pop, "ds-svp-complement"),
        ]
        working = [f"{k} of the {s.N} students {s.who}.", f"{k} / {s.N} x 100 = {int(s.pop)}%."]
    return single_number(rng, _stem(s, q), s.pop, _places_pop(s), working, slips)


def _gap_slips(s: Scenario, which: str):
    other = "convenience" if which == "random" else "random"
    est = _rf(s, which)
    return [
        (est, "ds-svp-sample-value-as-gap"),
        (s.pop, "ds-svp-population-value-as-gap"),
        (est + s.pop, "ds-svp-adds-instead-of-subtracting"),
        (_gf(s, other), "ds-svp-other-samples-gap"),
        (_gf(s, which) * 100 / s.pop, "ds-svp-gap-as-relative-percent"),
    ]


def _gap_working(s: Scenario, which: str):
    est = _rf(s, which)
    unit = "" if s.kind == "mean" else "%"
    return [
        f"The {which} sample gives {float(est):g}{unit}; the school gives {float(s.pop):g}{unit}.",
        f"The gap is {float(_gf(s, which)):g}{unit if s.kind == 'mean' else ' percentage points'}.",
    ]


def _gap_prompt(s: Scenario, which: str) -> str:
    if s.kind == "mean":
        return f"How far is the mean of the {which} sample from the mean of the whole school?"
    return (
        f"By how many percentage points does the {which} sample's percentage who {s.who} differ from the "
        "percentage for the whole school?"
    )


def _gap(rng, s: Scenario) -> dict:
    which = rng.choice(("random", "convenience"))
    return single_number(
        rng,
        _stem(s, _gap_prompt(s, which)),
        _gf(s, which),
        _places_gap(s),
        _gap_working(s, which),
        _gap_slips(s, which),
    )


def _closer_by(rng, s: Scenario) -> dict:
    diff = abs(s.g_cnv - s.g_rnd)
    lead = (
        "One sample's mean is nearer"
        if s.kind == "mean"
        else f"One sample's percentage of students who {s.who} is nearer"
    )
    where = "the whole-school mean" if s.kind == "mean" else "the whole-school percentage"
    q = f"{lead} to {where} than the other's. By how {'much' if s.kind == 'mean' else 'many percentage points'} is it nearer?"
    slips = [
        (s.g_rnd + s.g_cnv, "ds-svp-sum-of-gaps"),
        (abs(s.cnv - s.rnd), "ds-svp-gap-between-samples"),
        (max(s.g_rnd, s.g_cnv), "ds-svp-one-gap-only"),
        (min(s.g_rnd, s.g_cnv), "ds-svp-one-gap-only"),
    ]
    working = [
        f"Random sample: {float(s.rnd):g} against {float(s.pop):g}, a gap of {float(s.g_rnd):g}.",
        f"Convenience sample: {float(s.cnv):g} against {float(s.pop):g}, a gap of {float(s.g_cnv):g}.",
        f"The gaps differ by {float(diff):g}.",
    ]
    return single_number(rng, _stem(s, q), diff, _places_gap(s), working, slips)


def _pooled(rng, s: Scenario) -> dict:
    pooled = (s.rnd + s.cnv) / 2
    gap = abs(pooled - s.pop)
    places = 2 if s.kind == "mean" else 1
    q = (
        f"The two samples are put together as one group of {2 * s.n} students. How far is the mean of the group "
        "from the mean of the whole school?"
    )
    if s.kind == "prop":
        q = (
            f"The two samples are put together as one group of {2 * s.n} students. By how many percentage points "
            f"does the percentage of the group who {s.who} differ from the percentage for the whole school?"
        )
    slips = [
        ((s.g_rnd + s.g_cnv) / 2, "ds-svp-averages-the-gaps"),
        (s.g_cnv, "ds-svp-one-gap-only"),
        (s.g_rnd, "ds-svp-one-gap-only"),
        (pooled, "ds-svp-sample-value-as-gap"),
        (s.g_rnd + s.g_cnv, "ds-svp-sum-of-gaps"),
    ]
    unit = "" if s.kind == "mean" else "%"
    working = [
        f"Combined {s.sample_noun} = ({float(s.rnd):g} + {float(s.cnv):g}) / 2 = {float(pooled):g}{unit}.",
        f"The school value is {float(s.pop):g}{unit}, so the gap is {float(gap):g}.",
    ]
    return single_number(rng, _stem(s, q), gap, places, working, slips)


def _total_error(rng, s: Scenario) -> dict:
    err = s.g_cnv * s.N
    if s.kind == "mean":
        q = (
            f"A teacher multiplies the convenience sample's mean by {s.N} to estimate the total number of "
            f"{s.ctx['total']} for the whole school. By how many does this estimate differ from the true total?"
        )
        est_total, true_total = s.cnv * s.N, s.pop * s.N
    else:
        q = (
            f"The school applies the convenience sample's percentage to all {s.N} students to estimate how many "
            f"students {s.who}. By how many students does this estimate differ from the true number?"
        )
        est_total, true_total = s.cnv * s.N / 100, s.pop * s.N / 100
        err = s.g_cnv * s.N / 100
    slips = [
        (s.g_cnv * s.n if s.kind == "mean" else s.g_cnv * s.n / 100, "ds-svp-scales-by-sample-size"),
        (s.g_cnv, "ds-svp-gap-not-scaled"),
        (est_total, "ds-svp-estimate-not-error"),
        (true_total, "ds-svp-true-total-not-error"),
        (s.g_rnd * s.N if s.kind == "mean" else s.g_rnd * s.N / 100, "ds-svp-other-samples-gap"),
    ]
    working = [
        f"Gap: {float(s.g_cnv):g}{' percentage points' if s.kind == 'prop' else ''}.",
        f"Scale to the whole school: {float(err):g}.",
    ]
    return single_number(rng, _stem(s, q), err, 0, working, slips, spread=60)


_EASY = (_sample_estimate, _population_value)
_MEDIUM = (_gap,)
_HARD = (_closer_by, _pooled, _total_error)


def _tier_question(rng: random.Random, tier: str) -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    shapes = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}[tier]
    for _ in attempts(200, "a question with enough distinct wrong answers"):
        s = _scenario(rng)
        build = rng.choice(shapes)
        try:
            return build(rng, s)
        except VerificationError as exc:  # a draw with fewer than two usable distractors is redrawn
            if "fewer than two distractors" not in str(exc):
                raise


@generator(TOPIC, level=11, presets={t: dict(tier=t) for t in _TIERS})
def ds_sample_vs_pop_gap(rng: random.Random, tier: str = "easy") -> dict:
    return _tier_question(rng, tier)


# -- exam ------------------------------------------------------------------------------------------------------------
def _exam_chain(rng: random.Random) -> dict:
    s = _scenario(rng)
    mean = s.kind == "mean"
    unit = "" if mean else "%"
    pts = "" if mean else " percentage points"
    places = _places_gap(s)
    stem = _stem(s, "Use the table and the two samples to answer the parts below.")
    a_prompt = "What is the mean of the whole school?" if mean else f"What percentage of the whole school {s.who}?"
    est_r, est_c = (
        Fraction(sum(s.rnd_raw), s.n) if mean else s.rnd,
        Fraction(sum(s.cnv_raw), s.n) if mean else s.cnv,
    )
    ft_r, ft_c = (
        f"Abs({est_r.numerator}/{est_r.denominator} - part_a)",
        f"Abs({est_c.numerator}/{est_c.denominator} - part_a)",
    )
    if mean:
        total = sum(v * c for v, c in s.table_raw.items())
        a_slips = [
            (Fraction(sum(s.table_raw), s.rows), "ds-svp-ignores-frequencies"),
            (Fraction(total), "ds-svp-total-not-mean"),
            (Fraction(total, s.rows), "ds-svp-divides-by-number-of-rows"),
        ]
        a_work = [f"{total} / {s.N} = {float(s.pop):g}."]
    else:
        k = s.table_raw[s.table_raw["target"]]
        a_slips = [
            (Fraction(k), "ds-svp-count-not-percent"),
            (Fraction(100 * k, s.n), "ds-svp-percent-of-sample-size"),
            (100 - s.pop, "ds-svp-complement"),
        ]
        a_work = [f"{k} / {s.N} x 100 = {int(s.pop)}%."]
    parts = [
        number_part(
            rng,
            a_prompt,
            s.pop,
            _places_pop(s),
            1,
            a_work,
            a_slips,
            sensible_range=(0, None if mean else 100),
        )
    ]
    for label, which, ft in (("b", "random", ft_r), ("c", "convenience", ft_c)):
        prompt = (
            f"How far is the mean of the {which} sample from the mean of the whole school?"
            if mean
            else f"By how many percentage points does the {which} sample's percentage who {s.who} differ from the "
            "whole-school percentage?"
        )
        parts.append(
            number_part(
                rng,
                prompt,
                _gf(s, which),
                places,
                2,
                _gap_working(s, which),
                _gap_slips(s, which),
                follow_through=ft,
                sensible_range=(0, None),
            )
        )
    if mean:
        d_prompt = (
            f"A teacher multiplies the convenience sample's mean by {s.N} to estimate the total number of "
            f"{s.ctx['total']} for the whole school. By how many does this estimate differ from the true total?"
        )
        d_value = s.g_cnv * s.N
        d_slips = [
            (s.g_cnv * s.n, "ds-svp-scales-by-sample-size"),
            (s.g_cnv, "ds-svp-gap-not-scaled"),
            (s.cnv * s.N, "ds-svp-estimate-not-error"),
            (s.pop * s.N, "ds-svp-true-total-not-error"),
        ]
        d_ft = f"Abs(part_c) * {s.N}"
        d_places = 0
        d_work = [f"{float(s.g_cnv):g} x {s.N} = {float(d_value):g}."]
    else:
        d_prompt = (
            f"The school applies the convenience sample's percentage to all {s.N} students to estimate how many "
            f"students {s.who}. By how many students does this estimate differ from the true number?"
        )
        d_value = s.g_cnv * s.N / 100
        d_slips = [
            (s.g_cnv, "ds-svp-gap-not-scaled"),
            (s.g_cnv * s.n / 100, "ds-svp-scales-by-sample-size"),
            (s.cnv * s.N / 100, "ds-svp-estimate-not-error"),
            (s.pop * s.N / 100, "ds-svp-true-total-not-error"),
        ]
        d_ft = f"Abs(part_c) * {s.N} / 100"
        d_places = 0
        d_work = [f"{float(s.g_cnv):g} / 100 x {s.N} = {float(d_value):g}."]
    parts.append(
        number_part(
            rng,
            d_prompt,
            d_value,
            d_places,
            2,
            d_work,
            d_slips,
            follow_through=d_ft,
            sensible_range=(0, None),
            spread=60,
        )
    )
    answers = [p["answer_text"] for p in parts]
    check(len(set(answers)) == len(answers), "exam parts share an answer")
    return dict(text=stem, parts=parts)


@generator(EXAM_TOPIC, level=12, presets={"exam": dict(tier="exam")})
def ds_sample_vs_pop_exam(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(100, "an exam chain with distinct part answers"):
        try:
            return _exam_chain(rng)
        except VerificationError as exc:
            if "fewer than two distractors" not in str(exc) and "share an answer" not in str(exc):
                raise
