"""Generates `time-series-features` questions (easy, medium, hard) and the exam chain `sr2a_time_series_chain`.

Features of a time series read from a drawn line graph (ACMGM088; the reading half of ACMGM087), `statistics-probability`,
answer ``number``. The graph is a ``cartesian`` figure with the points joined in time order, every value on a gridline.
The prompt never lists the data.

A series is built from three features, so the learner can be asked about each:

* a seasonal pattern of ``p`` distinct offsets that repeats every ``p`` time points (``p`` is 4, 6 or 7: quarters,
  two-monthly bills, days of the week);
* a trend that lifts the whole pattern by the same step ``D`` each cycle (``D`` is 2, 3 or 4 units, or -2, -3 when
  the series is falling);
* at most one irregular value, a one-off jump of 5 units up or down that breaks the pattern.

``value = base + D * (completed cycles) + offset[season]``, and a value on the graph is that many units of the
vertical gridline step.

* easy: read two values from the graph and find how much higher the later one is;
* medium: the length of the cycle is stated and the level is said to change by the same amount each cycle; predict
  the value at the next time point (read the same season one cycle ago, and the step between that cycle and the one
  before);
* hard: the length of the cycle is NOT stated. Either name the time point of the one value broken by a one-off event
  (found by comparing the same season in each cycle), or predict the next value when the same season in the latest
  cycle is the one broken by the one-off event, so that the last cycle cannot be used;
* exam (``sr2a_time_series_chain``): (a) the length of one cycle, (b) the time point broken by a one-off event,
  (c) the change in level from one cycle to the next, (d) a prediction of the next value using (c).

The registered figure readers re-derive each answer from the plotted points and the prompt alone: the cycle length is
the shift that makes `value[t + shift] - value[t]` most often the same, the step is that common difference, and a
season's usual value is the median over the cycles of its values lifted to the same cycle.
"""

from __future__ import annotations

import re
from statistics import median

from .. import figures
from ..core import attempts, check, generator, paren_negative, phrase
from ._sr2a_kit import number_part, num_text, single

_CONTEXTS = (
    dict(
        p=4,
        cycles=(3, 4),
        unit="quarter",
        units="quarters",
        x_label="Quarter number",
        y_label="Sales ($ thousand)",
        lead="The graph shows the quarterly sales, in thousands of dollars, of a surf shop over {n} consecutive quarters.",
        what="sales",
        steps=(5, 10),
        money=True,
    ),
    dict(
        p=6,
        cycles=(3,),
        unit="bill",
        units="bills",
        x_label="Bill number",
        y_label="Electricity used (kWh)",
        lead="The graph shows the electricity used by a household, in kilowatt-hours, on each of {n} consecutive two-monthly bills.",
        what="electricity used",
        steps=(10, 20),
        money=False,
    ),
    dict(
        p=7,
        cycles=(3,),
        unit="day",
        units="days",
        x_label="Day number",
        y_label="Customers",
        lead="The graph shows the number of customers who visited a cafe on each of {n} consecutive days.",
        what="number of customers",
        steps=(10, 20),
        money=False,
    ),
)


class Series:
    """A drawn series: ``ys`` are the plotted values (units times ``step``), ``t`` runs 1..n."""

    def __init__(self, ctx, step, offsets, base, delta, cycles, irregular=None):
        self.ctx, self.step, self.offsets, self.base, self.delta = ctx, step, offsets, base, delta
        self.p, self.cycles = ctx["p"], cycles
        self.n = self.p * cycles
        self.irregular = irregular  # (t, jump in units) or None
        units = [base + delta * ((t - 1) // self.p) + offsets[(t - 1) % self.p] for t in range(1, self.n + 1)]
        self.regular = list(units)
        if irregular:
            t, jump = irregular
            units[t - 1] += jump
        self.units = units
        self.ys = [u * step for u in units]

    def y(self, t):
        return self.ys[t - 1]

    def figure(self):
        pts = [{"at": [t, self.y(t)]} for t in range(1, self.n + 1)]
        segs = [{"from": [t, self.y(t)], "to": [t + 1, self.y(t + 1)]} for t in range(1, self.n)]
        top = (max(self.ys) // self.step + 1) * self.step
        return figures.cartesian(
            x_range=(0, self.n + 1),
            y_range=(0, top),
            grid=1,
            x_grid=1,
            y_grid=self.step,
            equal_scale=False,
            points=pts,
            segments=segs,
            x_label=self.ctx["x_label"],
            y_label=self.ctx["y_label"],
            alt="a line graph of values recorded at equal time steps",
        )


def _draw(rng, *, irregular: str | None, delta_pool=(-3, -2, 2, 2, 3, 3, 4)):
    """A random Series. ``irregular``: None, 'middle' (a value in the second cycle) or 'last' (the first point of the
    next cycle, one cycle after the last full season's repeat, i.e. the season of time n + 1 in the final cycle)."""
    ctx = rng.choice(_CONTEXTS)
    p = ctx["p"]
    cycles = rng.choice(ctx["cycles"])
    step = rng.choice(ctx["steps"])
    for _ in attempts(500, "a time series with clear features"):
        offsets = rng.sample(range(0, p + 3), p)
        if max(offsets) - min(offsets) < 4:
            continue
        delta = rng.choice(delta_pool)
        base = rng.randint(2, 5) + (abs(delta) * (cycles - 1) if delta < 0 else 0)
        irr = None
        if irregular == "middle":
            t = rng.randint(p + 1, 2 * p)
            irr = (t, rng.choice([-5, 5]))
        elif irregular == "last":
            t = p * cycles + 1 - p  # the season that the next point shares, in the latest full cycle
            irr = (t, rng.choice([-5, 5]))
        s = Series(ctx, step, offsets, base, delta, cycles, irr)
        if min(s.units) < 1 or max(s.units) > 22:
            continue
        # the value a question asks to predict (the next point) must be a plausible, positive amount
        if base + delta * cycles + offsets[(s.n) % p] < 1:
            continue
        # the one-off value must break the pattern: it is not a value its own season takes in the other cycles
        if irr:
            t, _jump = irr
            same = [s.units[k - 1] - s.delta * ((k - 1) // p) for k in range(1, s.n + 1) if (k - t) % p == 0 and k != t]
            if s.units[t - 1] - s.delta * ((t - 1) // p) in same:
                continue
        return s
    raise AssertionError  # pragma: no cover


def _prompt_head(s):
    return s.ctx["lead"].format(n=s.n)


def _label(s, t):
    return f"{s.ctx['unit']} {t}"


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_ASK = (
    "How much higher was the {what} in {b} than in {a}?",
    "By how much did the {what} in {b} exceed the {what} in {a}?",
)


def _easy(rng):
    for _ in attempts(300, "an easy time series"):
        s = _draw(rng, irregular=None)
        pairs = [(a, b) for a in range(1, s.n + 1) for b in range(a + 2, s.n + 1) if s.y(b) > s.y(a) and b - a != s.p]
        if not pairs:
            continue
        a, b = rng.choice(pairs)
        key = s.y(b) - s.y(a)
        what = s.ctx["what"]
        text = f"{_prompt_head(s)} " + phrase(rng, _EASY_ASK, what=what, a=_label(s, a), b=_label(s, b))
        wrong = [
            (s.y(b) + s.y(a), "sr2a-ts-adds-readings"),
            (s.y(b + 1) - s.y(a) if b < s.n else s.y(b - 1) - s.y(a), "sr2a-ts-neighbour-read"),
            (s.y(b) - s.y(a - 1) if a > 1 else s.y(b) - s.y(a + 1), "sr2a-ts-neighbour-read"),
            (key + s.step, "sr2a-ts-gridline-slip"),
            (key - s.step, "sr2a-ts-gridline-slip"),
            (s.y(b), "sr2a-ts-one-reading"),
        ]
        working = [
            f"Read {what} in {_label(s, a)}: {s.y(a)}. Read {what} in {_label(s, b)}: {s.y(b)}.",
            f"{s.y(b)} - {s.y(a)} = {key}.",
        ]
        q = single(text, key, working, wrong, lo=1, figure=s.figure())
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -----------------------------------------------------------------------------------------------------

_MEDIUM_ASK = (
    "The seasonal pattern repeats every {p} {units}, and the level rises or falls by the same amount from one cycle "
    "to the next. Predict the {what} in {t}, continuing both the pattern and the change of level.",
    "Every {p} {units} the pattern repeats, and from one cycle to the next the level changes by a steady amount. "
    "What {what} would you predict for {t}?",
)


def _medium(rng):
    for _ in attempts(300, "a medium time series"):
        s = _draw(rng, irregular=None)
        p, n = s.p, s.n
        prev = s.y(n + 1 - p)
        before = s.y(n + 1 - 2 * p)
        step_up = prev - before
        key = prev + step_up
        # second route: the same answer from the model it was built with
        model = (s.base + s.delta * s.cycles + s.offsets[n % p]) * s.step
        check(key == model, "forecast by reading the graph and from the model differ")
        text = f"{_prompt_head(s)} " + phrase(
            rng, _MEDIUM_ASK, p=p, units=s.ctx["units"], what=s.ctx["what"], t=_label(s, n + 1)
        )
        wrong = [
            (prev, "sr2a-ts-no-trend-added"),
            (s.y(n) + step_up, "sr2a-ts-ignores-season"),
            (prev + step_up * 2, "sr2a-ts-trend-twice"),
            (s.y(n), "sr2a-ts-ignores-season"),
            (before + step_up, "sr2a-ts-no-trend-added"),
            (key + s.step, "sr2a-ts-gridline-slip"),
        ]
        working = [
            f"{_label(s, n + 1)} is in the same season as {_label(s, n + 1 - p)}, which was {prev}, and {_label(s, n + 1 - 2 * p)}, which was {before}.",
            f"The level changed by {prev} - {before} = {step_up} over a cycle, so the prediction is {prev} + {paren_negative(step_up)} = {key}.",
        ]
        q = single(text, key, working, wrong, lo=1, figure=s.figure())
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------

_WHICH_ASK = (
    "One value was changed by a one-off event and does not follow the pattern. Which {unit} number was it?",
    "Exactly one value was affected by an unusual one-off event, so it does not fit the repeating pattern or the "
    "steady change of level. At which {unit} number did this happen?",
)
_HARD_FORECAST = (
    "The seasonal pattern repeats regularly, and the level changes by the same amount from one cycle to the next. "
    "One value was affected by a one-off event. Predict the {what} in {t}, continuing the pattern and the change of "
    "level but ignoring the effect of the one-off event.",
    "Ignoring the effect of the one value that was disturbed by a one-off event, and assuming the pattern repeats and "
    "the level keeps changing by a steady amount from cycle to cycle, predict the {what} in {t}.",
)


def _hard_which(rng):
    for _ in attempts(300, "a time series with a broken value"):
        s = _draw(rng, irregular="middle")
        t, _jump = s.irregular
        p = s.p
        highest = s.ys.index(max(s.ys)) + 1
        lowest = s.ys.index(min(s.ys)) + 1
        wrong = [
            (t + p if t + p <= s.n else t - p, "sr2a-ts-irregular-wrong-cycle"),
            (t - p if t - p >= 1 else t + p, "sr2a-ts-irregular-wrong-cycle"),
            (t + 1, "sr2a-ts-irregular-neighbour"),
            (t - 1, "sr2a-ts-irregular-neighbour"),
            (highest if highest != t else lowest, "sr2a-ts-irregular-as-extreme"),
            (lowest if lowest != t else highest, "sr2a-ts-irregular-as-extreme"),
        ]
        text = f"{_prompt_head(s)} " + phrase(rng, _WHICH_ASK, unit=s.ctx["unit"])
        working = [
            f"The pattern repeats every {p} {s.ctx['units']}, so compare each value with the same season in the other cycles.",
            f"Every other season rises or falls by the same amount from cycle to cycle; only {_label(s, t)} breaks the pattern.",
        ]
        q = single(text, t, working, wrong, lo=1, figure=s.figure())
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard_forecast(rng):
    for _ in attempts(300, "a time series whose latest season is broken"):
        s = _draw(rng, irregular="last")
        p, n = s.p, s.n
        t, _jump = s.irregular
        check(t == n + 1 - p, "the broken value is not the one a naive forecast would use")
        step_up = s.y(n + 2 - p) - s.y(n + 2 - 2 * p)  # the step, from the next season (not broken)
        check(step_up == s.delta * s.step, "the level step read from the graph differs from the model")
        key = s.y(n + 1 - 2 * p) + 2 * step_up
        model = (s.base + s.delta * s.cycles + s.offsets[n % p]) * s.step
        check(key == model, "forecast by reading the graph and from the model differ")
        broken = s.y(t)
        text = f"{_prompt_head(s)} " + phrase(rng, _HARD_FORECAST, what=s.ctx["what"], t=_label(s, n + 1))
        wrong = [
            (broken + step_up, "sr2a-ts-uses-irregular-value"),
            (s.y(n + 1 - 2 * p) + step_up, "sr2a-ts-trend-once-for-two-cycles"),
            (broken, "sr2a-ts-no-trend-added"),
            (s.y(n) + step_up, "sr2a-ts-ignores-season"),
            (key + s.step, "sr2a-ts-gridline-slip"),
            (key - s.step, "sr2a-ts-gridline-slip"),
        ]
        working = [
            f"The cycle is {p} {s.ctx['units']} long. {_label(s, n + 1)} is in the same season as {_label(s, t)}, but {_label(s, t)} was broken by the one-off event.",
            f"Use the cycle before: {_label(s, n + 1 - 2 * p)} was {s.y(n + 1 - 2 * p)}, and the level changes by {step_up} each cycle.",
            f"Two cycles on: {s.y(n + 1 - 2 * p)} + 2 x {paren_negative(step_up)} = {key}.",
        ]
        q = single(text, key, working, wrong, lo=1, figure=s.figure())
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard(rng):
    return _hard_which(rng) if rng.random() < 0.5 else _hard_forecast(rng)


@generator(
    "sr2a_time_series",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_time_series(rng, tier="easy"):
    """Features of a drawn time series; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2a_time_series_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_time_series_chain(rng, tier="exam"):
    """Exam chain on one graph: the cycle length, the broken time point, the step in level, and a prediction."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a time series chain with enough distractors"):
        s = _draw(rng, irregular="middle")
        p, n, units = s.p, s.n, s.ctx["units"]
        t, _jump = s.irregular
        # a regular season for the step in level: the one just after the cycle's start that is not the broken one
        season = next(k for k in range(1, p + 1) if (k - t) % p != 0)
        step_up = s.y(season + p) - s.y(season)
        check(step_up == s.delta * s.step, "the step read from the graph differs from the model")
        prev = s.y(n + 1 - p)
        key = prev + step_up
        check(key == (s.base + s.delta * s.cycles + s.offsets[n % p]) * s.step, "forecast differs from the model")
        stem = f"{_prompt_head(s)} The seasonal pattern repeats regularly and the level changes by the same amount from one cycle to the next, apart from one value that was affected by a one-off event."
        part_a = number_part(
            f"How many {units} are there in one cycle of the seasonal pattern?",
            p,
            1,
            [f"The pattern between one high point and the next takes {p} {units}."],
            [
                (p - 1, "sr2a-ts-period-off-by-one"),
                (p + 1, "sr2a-ts-period-off-by-one"),
                (s.cycles, "sr2a-ts-period-count-points"),
                (n, "sr2a-ts-period-count-points"),
                (p * 2, "sr2a-ts-period-off-by-one"),
            ],
            lo=2,
        )
        part_b = number_part(
            f"Which {s.ctx['unit']} number has the value that was changed by the one-off event?",
            t,
            2,
            [f"Comparing the same season in each cycle, only {_label(s, t)} does not follow the pattern."],
            [
                (t + p if t + p <= n else t - p, "sr2a-ts-irregular-wrong-cycle"),
                (t - p, "sr2a-ts-irregular-wrong-cycle"),
                (t + 1, "sr2a-ts-irregular-neighbour"),
                (t - 1, "sr2a-ts-irregular-neighbour"),
                (s.ys.index(max(s.ys)) + 1 if s.ys.index(max(s.ys)) + 1 != t else None, "sr2a-ts-irregular-as-extreme"),
            ],
            lo=1,
        )
        part_c = number_part(
            f"By how much does the level change from one cycle to the next? Give a positive number if the level rises and a "
            "negative number if it falls.",
            step_up,
            2,
            [
                f"Compare {_label(s, season)} with {_label(s, season + p)}, the same season one cycle later: {s.y(season + p)} - {s.y(season)} = {step_up}."
            ],
            [
                (s.y(n) - s.y(1), "sr2a-ts-trend-as-total"),
                (s.y(season + 1) - s.y(season), "sr2a-ts-trend-one-step"),
                (-step_up, "sr2a-ts-trend-sign"),
                (step_up + s.step, "sr2a-ts-gridline-slip"),
                (step_up - s.step, "sr2a-ts-gridline-slip"),
            ],
            lo=None,
            sensible=(None, None),
        )
        part_d = number_part(
            f"Predict the {s.ctx['what']} in {_label(s, n + 1)}, continuing the pattern and the change of level.",
            key,
            2,
            [
                f"{_label(s, n + 1)} is in the same season as {_label(s, n + 1 - p)}, which was {prev}.",
                f"{prev} + {paren_negative(step_up)} = {key}.",
            ],
            [
                (prev, "sr2a-ts-no-trend-added"),
                (s.y(n) + step_up, "sr2a-ts-ignores-season"),
                (prev + 2 * step_up, "sr2a-ts-trend-twice"),
                (key + s.step, "sr2a-ts-gridline-slip"),
            ],
            follow=f"{prev} + part_c",
            lo=1,
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d], figure=s.figure())
    raise AssertionError  # pragma: no cover


# -- the figure readers -----------------------------------------------------------------------------------------


def _points(question):
    pts = sorted((int(p["at"][0]), p["at"][1]) for p in question.figure["points"])
    return {t: y for t, y in pts}


def _infer(ys: dict):
    """(cycle length, step in level per cycle) from the plotted values alone."""
    n = max(ys)
    best = None
    for p in (4, 6, 7):
        if n < 2 * p:
            continue
        diffs = [ys[t + p] - ys[t] for t in range(1, n - p + 1)]
        mode = max(set(diffs), key=diffs.count)
        score = diffs.count(mode)
        if best is None or score > best[0]:
            best = (score, p, mode)
    return best[1], best[2]


def _usual(ys: dict, p: int, d: int, t: int) -> int:
    """The value a season usually takes in the cycle of time ``t``: the median over cycles lifted to that cycle."""
    cycle = (t - 1) // p
    season = (t - 1) % p
    lifted = [ys[season + 1 + k * p] + (cycle - k) * d for k in range(len(ys) // p) if season + 1 + k * p in ys]
    return int(median(lifted))


def _read_features(question) -> str:
    ys = _points(question)
    p, d = _infer(ys)
    n = len(ys)
    text = question.text
    if "one-off" in text and "Which" in text or "At which" in text:
        flagged = [t for t in ys if ys[t] != _usual(ys, p, d, t)]
        return str(flagged[0])
    if "How much higher" in text or "exceed" in text:
        later, earlier = (int(x) for x in re.findall(r"(?:quarter|bill|day) (\d+)", text)[-2:])
        return str(ys[later] - ys[earlier])
    return str(_usual(ys, p, d, n + 1))


def _read_chain(question) -> str:
    ys = _points(question)
    p, d = _infer(ys)
    n = len(ys)
    flagged = [t for t in ys if ys[t] != _usual(ys, p, d, t)]
    return f"(a) {p}; (b) {flagged[0]}; (c) {d}; (d) {_usual(ys, p, d, n + 1)}"


figures.register_reader("sr2a_time_series", _read_features)
figures.register_reader("sr2a_time_series_chain", _read_chain)
