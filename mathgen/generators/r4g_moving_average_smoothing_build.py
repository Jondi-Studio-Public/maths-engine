"""Smoothing a time series: `r4g_moving_average_smoothing` (easy, medium, hard).

``moving-average-median-smoothing`` (``statistics-probability``, answer ``number``). The series is written in the
question as `day 1: 14; day 2: 18; ...`. A smoothed value is centred on its day: the 3-point value for day k uses
days k-1, k and k+1, and the 5-point value uses days k-2 to k+2.

* easy: the 3-point moving mean at a stated day.
* medium: the 3-point moving median (median smoothing) at a stated day, or the 5-point moving mean.
* hard: how far the 3-point moving mean at a spike lies above the 3-point moving median there, or which day has
  the greatest 5-point moving mean.

The windows at the spike are built so the 3-point mean is a whole number, so every key is exact at the places shown.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import number_text, single_number, retry

CONTEXTS = [
    dict(what="the number of takeaway coffees a cafe sold", unit="coffees", when="day", lo=35, hi=70, plural="days"),
    dict(what="the number of visitors to a school library", unit="visitors", when="day", lo=40, hi=90, plural="days"),
    dict(
        what="the number of new members a gym signed up", unit="new members", when="week", lo=15, hi=40, plural="weeks"
    ),
    dict(
        what="the number of kilolitres of water a town used",
        unit="kilolitres",
        when="month",
        lo=50,
        hi=95,
        plural="months",
    ),
    dict(what="the number of parcels a depot dispatched", unit="parcels", when="day", lo=120, hi=210, plural="days"),
]


def mean_at(series, k, w):
    h = w // 2
    return Fraction(sum(series[k - h : k + h + 1]), w)


def median_at(series, k, w):
    h = w // 2
    return Fraction(sorted(series[k - h : k + h + 1])[h])


def series_text(ctx, series):
    items = "; ".join(f"{ctx['when']} {i + 1}: {v}" for i, v in enumerate(series))
    return f"A series records {ctx['what']} on each of {len(series)} {ctx['plural']}: {items}."


def _walk(rng, ctx, n):
    span = ctx["hi"] - ctx["lo"]
    v = rng.randint(ctx["lo"] + span // 4, ctx["hi"] - span // 4)
    out = []
    for _ in range(n):
        v = min(ctx["hi"], max(ctx["lo"], v + rng.randint(-span // 6, span // 6)))
        out.append(v)
    return out


def draw_series(rng, *, n=None, k=None, need3=True, spike=False):
    """(context, series, k): `k` is a 1-based day with room for a 5-point window, and the 3-point window sum there is
    a multiple of 3 when `need3`. With `spike` the value on day k sits well above its neighbours."""
    ctx = rng.choice(CONTEXTS)
    n = n or rng.choice((9, 10, 11))
    for _ in attempts(2000, "a series with the wanted smoothed values"):
        s = _walk(rng, ctx, n)
        day = k or rng.randint(3, n - 2)
        i = day - 1
        if spike:
            s[i] = max(s[i - 1], s[i + 1]) + rng.randint(12, 24)
        if need3:
            fix = (-sum(s[i - 1 : i + 2])) % 3
            if spike:
                s[i] += fix
            else:
                s[i + 1] += fix
        if need3 and sum(s[i - 1 : i + 2]) % 3:
            continue
        if spike and mean_at(s, i, 3) <= median_at(s, i, 3) + 1:
            continue
        window3 = sorted(s[i - 1 : i + 2])
        window5 = s[i - 2 : i + 3]
        if len(set(window3)) < 3:
            continue
        if spike and window3[1] == s[i]:
            continue
        if len({mean_at(s, i, 3), median_at(s, i, 3), Fraction(s[i])}) < 3:
            continue
        if not all(ctx["lo"] <= v <= ctx["hi"] + 30 for v in s) or len(set(window5)) < 4:
            continue
        return ctx, s, day
    raise AssertionError("unreachable")


def _places(v):
    return 0 if Fraction(v).denominator == 1 else 1


def _easy(rng):
    ctx, s, day = draw_series(rng)
    i = day - 1
    ans = mean_at(s, i, 3)
    window = s[i - 1 : i + 2]
    text = (
        f"{series_text(ctx, s)} Find the 3-point moving mean for {ctx['when']} {day}, "
        f"which uses {ctx['when']}s {day - 1}, {day} and {day + 1}."
    )
    slips = [
        (mean_at(s, i - 1, 3), "r4g-window-shifted-one-step"),
        (mean_at(s, i + 1, 3), "r4g-window-shifted-one-step"),
        (Fraction(sum(window), 4), "r4g-window-sum-divided-by-wrong-count"),
        (Fraction(s[i]), "r4g-raw-value-for-smoothed"),
        (Fraction(sum(window), 1), "r4g-window-sum-not-divided"),
        (mean_at(s, i, 5), "r4g-wrong-window-length"),
    ]
    working = [
        f"Add the values for {ctx['when']}s {day - 1} to {day + 1}: {' + '.join(map(str, window))} = {sum(window)}.",
        f"Divide by 3: {sum(window)} / 3 = {number_text(ans, _places(ans))}.",
    ]
    return single_number(text, ans, _places(ans), working, slips)


def _medium_median(rng):
    ctx, s, day = draw_series(rng, need3=False)
    i = day - 1
    ans = median_at(s, i, 3)
    window = s[i - 1 : i + 2]
    text = (
        f"{series_text(ctx, s)} Smooth the series with 3-point moving medians. "
        f"Find the smoothed value for {ctx['when']} {day}, which uses {ctx['when']}s {day - 1}, {day} and {day + 1}."
    )
    slips = [
        (mean_at(s, i, 3), "r4g-mean-instead-of-median-smoothing"),
        (Fraction(s[i]), "r4g-raw-value-for-smoothed"),
        (Fraction(window[1]), "r4g-middle-in-time-not-in-size"),
        (median_at(s, i - 1, 3), "r4g-window-shifted-one-step"),
        (median_at(s, i + 1, 3), "r4g-window-shifted-one-step"),
        (Fraction(max(window) + min(window), 2), "r4g-midrange-for-median"),
    ]
    ans_places = 0
    mean_gap = {mean_at(s, i, 3)}
    working = [
        f"The values for {ctx['when']}s {day - 1} to {day + 1} are {', '.join(map(str, window))}.",
        f"In order: {', '.join(map(str, sorted(window)))}. The middle one is {number_text(ans, 0)}.",
    ]
    del mean_gap
    return single_number(text, ans, ans_places, working, slips)


def _medium_five(rng):
    ctx, s, day = draw_series(rng, need3=False)
    i = day - 1
    ans = mean_at(s, i, 5)
    window = s[i - 2 : i + 3]
    places = _places(ans)
    text = (
        f"{series_text(ctx, s)} Find the 5-point moving mean for {ctx['when']} {day}, "
        f"which uses {ctx['when']}s {day - 2} to {day + 2}."
    )
    slips = [
        (mean_at(s, i, 3), "r4g-wrong-window-length"),
        (Fraction(sum(window), 4), "r4g-window-sum-divided-by-wrong-count"),
        (Fraction(sum(window), 6), "r4g-window-sum-divided-by-wrong-count"),
        (mean_at(s, i - 1, 5) if i - 3 >= 0 else mean_at(s, i + 1, 5), "r4g-window-shifted-one-step"),
        (Fraction(sorted(window)[2]), "r4g-median-instead-of-mean-smoothing"),
    ]
    working = [
        f"Add {' + '.join(map(str, window))} = {sum(window)}.",
        f"Divide by 5: {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


def _medium(rng):
    return _medium_median(rng) if rng.random() < 0.5 else _medium_five(rng)


def _hard_spike(rng):
    ctx, s, day = draw_series(rng, spike=True)
    i = day - 1
    ans = mean_at(s, i, 3) - median_at(s, i, 3)
    check(ans > 0, "the spike does not pull the mean above the median")
    text = (
        f"{series_text(ctx, s)} For {ctx['when']} {day}, by how much is the 3-point moving mean greater than the "
        f"3-point moving median?"
    )
    slips = [
        (mean_at(s, i, 3) - s[i], "r4g-mean-minus-raw-value"),
        (Fraction(s[i]) - median_at(s, i, 3), "r4g-raw-minus-median"),
        (mean_at(s, i, 3), "r4g-one-smoothed-value-not-the-difference"),
        (mean_at(s, i, 5) - median_at(s, i, 3), "r4g-wrong-window-length"),
        (median_at(s, i, 3), "r4g-one-smoothed-value-not-the-difference"),
    ]
    window = s[i - 1 : i + 2]
    working = [
        f"Mean of {', '.join(map(str, window))} is {number_text(mean_at(s, i, 3), 0)}.",
        f"Median is {number_text(median_at(s, i, 3), 0)}.",
        f"Difference = {number_text(ans, _places(ans))}.",
    ]
    return single_number(text, ans, _places(ans), working, slips)


def _hard_peak(rng):
    for _ in attempts(500, "a series with one clear greatest 5-point mean"):
        ctx, s, day = draw_series(rng, n=rng.choice((10, 11)), need3=False)
        sums = {d: sum(s[d - 3 : d + 2]) for d in range(3, len(s) - 1)}  # days 3 .. n-2 (1-based)
        best = max(sums.values())
        winners = [d for d, v in sums.items() if v == best]
        runner = sorted(sums.values())[-2]
        if len(winners) == 1 and best - runner >= 3:
            break
    d = winners[0]
    raw_peak = 1 + s.index(max(s))
    text = (
        f"{series_text(ctx, s)} Find the 5-point moving means of the series. On which {ctx['when']} is the "
        f"5-point moving mean the greatest?"
    )
    first3 = max(range(2, len(s)), key=lambda day_: sum(s[day_ - 2 : day_ + 1]))  # trailing-window peak
    slips = [
        (raw_peak, "r4g-peak-of-raw-series"),
        (d - 1, "r4g-window-shifted-one-step"),
        (d + 1, "r4g-window-shifted-one-step"),
        (first3 + 1, "r4g-wrong-window-length"),
        (d - 2, "r4g-window-start-day-for-centre"),
    ]
    slips = [x for x in slips if x[0] != d]
    working = [
        "5-point moving means (days 3 to "
        + str(len(s) - 2)
        + "): "
        + ", ".join(number_text(Fraction(sums[x], 5), 1) for x in sorted(sums))
        + ".",
        f"The greatest is {number_text(Fraction(best, 5), 1)}, centred on {ctx['when']} {d}.",
    ]
    return single_number(text, d, 0, working, slips)


def _hard(rng):
    return _hard_spike(rng) if rng.random() < 0.55 else _hard_peak(rng)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_moving_average_smoothing",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_moving_average_smoothing(rng, tier="easy"):
    return retry(_TIER[tier], rng)
