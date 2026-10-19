"""Generates `ucv1_power_series_convergence` questions (easy, medium, hard) and the exam chain `exd_x_uc_power_series`.

``ucv1-power-series-convergence`` (``single-variable-calculus``, answer ``inequality``): where a power series
Σ s^n (c x - b)^n / (n^p k^n), n = 1, 2, 3, …, converges. Its centre is b/c and its radius k/c.

* easy: the open interval for a geometric-type series centred at 0, Σ (x/k)^n or Σ (cx)^n, with or without a factor n in
  the numerator (both endpoints then diverge, because the terms do not tend to 0).
* medium: the open interval of absolute convergence by the ratio test for a shifted, non-geometric series (centre b/c not
  0), with p in {-1, 1, 2} and c in {1, 2, 3}.
* hard: the full interval of convergence, endpoints included, for p in {1, 2} (and sometimes p = -1, where both
  endpoints fail), never the geometric p = 0: at an endpoint the series is Σ 1/n^p or Σ (-1)^n/n^p, so one end can be in and the other out.
* exam (``exd_x_uc_power_series``): f(x) = Σ s^n (x - a)^n / (n k^n), s = ±1. (a) the radius of convergence,
  (b) the open interval, (c) the interval of convergence with endpoints, (d) f'(x0) at a point inside it, by
  differentiating term by term to a geometric series; (d) follows through from (a).

Each answer is derived from the ratio test and the p-series / alternating series facts, then checked numerically from
the series itself: the ratio of consecutive terms at n = 200 is below 1 half a radius inside the interval and above 1
half a radius outside, and the partial sums over [N, 2N] with N = 400 settle at each endpoint exactly where the answer
includes it. The exam sum is checked against the series' partial sum.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Interval, Rational

from ..answers import Distractor, exact_value
from ..core import check, generator, phrase, pick_distractors, rational_text
from ..grading.inequality import set_text

_N = 400
_RATIO_N = 200


def _x_term(c, b):
    """The bracket (cx - b) as text."""
    if c == 1:
        return "x" if b == 0 else (f"(x - {b})" if b > 0 else f"(x + {-b})")
    return f"({c}x)" if b == 0 else (f"({c}x - {b})" if b > 0 else f"({c}x + {-b})")


def _series_text(s, p, c, b, k):
    """Σ s^n (cx - b)^n / (n^p k^n) as text, the n = 1, 2, 3, … range left to the stem."""
    inner = _x_term(c, b)
    sign = "(-1)^n " if s == -1 else ""
    power = f"{inner}^n"
    numerator = f"{sign}{'n' if p == -1 else ''}{power}"
    denominator = []
    if p == 1:
        denominator.append("n")
    elif p == 2:
        denominator.append("n²")
    if k > 1:
        denominator.append(f"{k}^n")
    if not denominator:
        return f"Σ {numerator}"
    if len(denominator) == 1:
        return f"Σ {numerator} / {denominator[0]}"
    return f"Σ {numerator} / ({'·'.join(denominator)})"


def _included(eps, p):
    """Whether Σ eps^n / n^p converges: eps = 1 needs p > 1, eps = -1 needs p > 0."""
    return p > 1 if eps == 1 else p > 0


def _interval(s, p, c, b, k, full):
    """The interval of convergence (`full`) or the open interval of absolute convergence, as a SymPy Interval."""
    lo, hi = Rational(b - k, c), Rational(b + k, c)
    if not full:
        return Interval.open(lo, hi)
    # The right endpoint has bracket +k, so the endpoint series has ratio s; the left has -k, ratio -s.
    return Interval(lo, hi, not _included(-s, p), not _included(s, p))


def _term(s, p, c, b, k, x, n):
    return float(s) ** n * (float(c * x - b) / k) ** n / float(n) ** p


def _verify(s, p, c, b, k, answer: Interval, full: bool):
    """Numerical re-derivation of the interval from the series itself (floats; see the module docstring)."""
    lo, hi = float(answer.start), float(answer.end)
    radius, centre = (hi - lo) / 2, (hi + lo) / 2
    exact = {lo: Fraction(int(answer.start.p), int(answer.start.q)), hi: Fraction(int(answer.end.p), int(answer.end.q))}

    def ratio(x):
        return abs(_term(s, p, c, b, k, x, _RATIO_N + 1) / _term(s, p, c, b, k, x, _RATIO_N))

    for x in (centre - radius / 2, centre + radius / 2):
        check(ratio(x) < 1, "the terms do not shrink inside the interval")
    for x in (centre - 1.5 * radius, centre + 1.5 * radius):
        check(ratio(x) > 1, "the terms do not grow outside the interval")
    if not full:
        return
    for x, is_closed in ((lo, not answer.left_open), (hi, not answer.right_open)):
        block = sum(_term(s, p, c, b, k, exact[x], n) for n in range(_N + 1, 2 * _N + 1))
        settles = abs(block) < 0.05 and abs(_term(s, p, c, b, k, exact[x], 2 * _N)) < 0.05
        check(settles == is_closed, f"endpoint {x}: the numerical test disagrees with the answer")


def _text(interval):
    return set_text(interval)


def _dist(interval, tag):
    text = _text(interval)
    return Distractor(text, tag, text)


def _interval_candidates(s, p, c, b, k, answer, full):
    centre, radius = Rational(b, c), Rational(k, c)
    lo, hi = answer.start, answer.end
    out = []
    if centre != 0:
        zero = Interval(-radius, radius, answer.left_open, answer.right_open)
        out.append((zero, "series-interval-centred-at-zero"))
        flipped = Interval(-centre - radius, -centre + radius, answer.left_open, answer.right_open)
        out.append((flipped, "series-centre-sign-error"))
    if radius != 1:
        recip = Interval(centre - 1 / radius, centre + 1 / radius, answer.left_open, answer.right_open)
        out.append((recip, "series-radius-reciprocal"))
    if c > 1:
        out.append((Interval(b - k, b + k, answer.left_open, answer.right_open), "series-radius-ignores-coefficient"))
    for left_open, right_open, tag in (
        (False, False, "series-endpoints-both-included"),
        (True, True, "series-endpoints-both-excluded"),
        (not answer.left_open, not answer.right_open, "series-endpoints-swapped"),
    ):
        out.append((Interval(lo, hi, left_open, right_open), tag))
    if full:
        order = [t for t in out if t[1].startswith("series-endpoints")] + [
            t for t in out if not t[1].startswith("series-endpoints")
        ]
    else:
        order = [t for t in out if t[1] == "series-endpoints-both-included"] + [
            t for t in out if t[1] not in ("series-endpoints-both-included", "series-endpoints-both-excluded")
        ]
    return [(iv, tag) for iv, tag in order if iv != answer]


def _finish_interval(rng, text, s, p, c, b, k, full, working):
    answer = _interval(s, p, c, b, k, full)
    _verify(s, p, c, b, k, answer, full)
    answer_text = _text(answer)
    candidates = [_dist(iv, tag) for iv, tag in _interval_candidates(s, p, c, b, k, answer, full)]
    distractors = pick_distractors(answer_text, candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="inequality",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=distractors,
    )


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Find all values of x for which the series {series}, summed over n = 1, 2, 3, …, converges.",
    "For which values of x does the power series {series}, with n = 1, 2, 3, …, converge?",
    "Find the set of values of x for which the series {series}, summed over n = 1, 2, 3, …, converges.",
)
_MEDIUM_STEMS = (
    "Use the ratio test to find the open interval of values of x for which the series {series}, summed over "
    "n = 1, 2, 3, …, converges absolutely.",
    "Find the open interval on which the power series {series}, summed over n = 1, 2, 3, …, converges absolutely.",
    "Find the values of x for which the ratio test shows that the series {series}, with n = 1, 2, 3, …, converges.",
)
_HARD_STEMS = (
    "Find the interval of convergence of the power series {series}, summed over n = 1, 2, 3, …",
    "Determine the interval of convergence of the series {series}, with n = 1, 2, 3, …, including the behaviour at "
    "each endpoint.",
    "For which values of x does the series {series}, summed over n = 1, 2, 3, …, converge?",
)


def _working_ratio(p, c, b, k, interval):
    inner = _x_term(c, b)
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]  # |2x + 7|, not |(2x + 7)|
    bound = f"|{inner}|" if k == 1 else f"|{inner}| / {k}"
    return [
        f"The ratio of consecutive terms tends to {bound}, which is below 1 when "
        + (f"|{inner}| < {k}." if k != 1 else f"|{inner}| < 1."),
        f"That is {_text(Interval.open(interval.start, interval.end))}.",
    ]


def _working_endpoints(s, p, c, b, k, interval):
    lo, hi = interval.start, interval.end
    names = {1: "Σ 1/n^p", -1: "Σ (-1)^n/n^p"}

    def describe(eps, x):
        series = names[eps].replace("p", str(p) if p > 1 else "") if p >= 1 else f"terms that do not tend to 0 (n^{-p})"
        if p == 1:
            series = "the harmonic series Σ 1/n" if eps == 1 else "the alternating harmonic series Σ (-1)^n/n"
        elif p == 2:
            series = "Σ 1/n²" if eps == 1 else "Σ (-1)^n/n²"
        elif p <= 0:
            return f"At x = {rational_text(x)} the terms do not tend to 0, so the series diverges."
        verdict = "converges" if _included(eps, p) else "diverges"
        return f"At x = {rational_text(x)} the series is {series}, which {verdict}."

    return [describe(-s, lo), describe(s, hi)]


def _easy(rng):
    p = rng.choice([0, -1])
    s = rng.choice([1, 1, -1])
    if rng.random() < 0.6:
        c, k = 1, rng.randint(2, 12)
    else:
        c, k = rng.randint(2, 9), 1
    series = _series_text(s, p, c, 0, k)
    text = phrase(rng, _EASY_STEMS, series=series)
    answer = _interval(s, p, c, 0, k, True)
    return _finish_interval(rng, text, s, p, c, 0, k, True, _working_ratio(p, c, 0, k, answer))


def _draw_shifted(rng):
    c = rng.choice([1, 1, 2, 3])
    if c == 1:
        k, b = rng.randint(2, 7), rng.choice([v for v in range(-5, 6) if v != 0])
    else:
        k, b = rng.randint(1, 6), rng.choice([v for v in range(-7, 8) if v != 0])
    return c, b, k


def _medium(rng):
    p = rng.choice([-1, 1, 1, 2])
    s = rng.choice([1, -1])
    c, b, k = _draw_shifted(rng)
    series = _series_text(s, p, c, b, k)
    text = phrase(rng, _MEDIUM_STEMS, series=series)
    answer = _interval(s, p, c, b, k, False)
    return _finish_interval(rng, text, s, p, c, b, k, False, _working_ratio(p, c, b, k, answer))


def _hard(rng):
    p = rng.choice([1, 1, 1, 2, 2, 2, -1])
    s = rng.choice([1, -1])
    c, b, k = _draw_shifted(rng)
    series = _series_text(s, p, c, b, k)
    text = phrase(rng, _HARD_STEMS, series=series)
    answer = _interval(s, p, c, b, k, True)
    working = [
        *_working_ratio(p, c, b, k, answer)[:1],
        f"The series converges for {_text(Interval.open(answer.start, answer.end))} and diverges outside it.",
        *_working_endpoints(s, p, c, b, k, answer),
        f"So the interval of convergence is {_text(answer)}.",
    ]
    return _finish_interval(rng, text, s, p, c, b, k, True, working)


@generator(
    "ucv1_power_series_convergence",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_power_series_convergence(rng, tier="easy"):
    """Where a power series converges; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _frac_dist(value, tag):
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


@generator("exd_x_uc_power_series", level=18, presets={"exam": dict(tier="exam")})
def exd_x_uc_power_series(rng, tier="exam"):
    """Exam chain: radius, open interval, interval with endpoints, and f'(x0) by term-by-term differentiation."""
    check(tier == "exam", f"unknown tier {tier!r}")
    s = rng.choice([1, -1])
    k = rng.randint(2, 6)
    a = rng.choice([v for v in range(-4, 5) if v != 0])
    t = rng.choice([v for v in range(-(k - 1), k) if v != 0])
    x0 = a + t
    open_interval = _interval(s, 1, 1, a, k, False)
    full_interval = _interval(s, 1, 1, a, k, True)
    _verify(s, 1, 1, a, k, full_interval, True)
    check(open_interval.start < x0 < open_interval.end, "x0 is not inside the interval")
    # f'(x0): sum of s^n (x0 - a)^(n-1) / k^n, a geometric series with first term s/k and ratio s t / k.
    value = Fraction(s, k - s * t)
    ratio = Fraction(s * t, k)
    check(value == Fraction(s, k) / (1 - ratio), "geometric sum formula disagrees")
    # Independent route: the partial sum of the differentiated series itself (floats; the tail is below 1e-14).
    partial = sum(float(s) ** n * float(t) ** (n - 1) / float(k) ** n for n in range(1, 301))
    check(abs(partial - float(value)) < 1e-9, "the partial sum of f' disagrees with the geometric sum")

    answer_a = k
    a_dist = pick_distractors(
        str(k),
        [
            _frac_dist(Fraction(1, k), "series-radius-reciprocal"),
            _frac_dist(a + k, "series-radius-as-endpoint"),
            _frac_dist(2 * k, "series-radius-as-width"),
            _frac_dist(abs(a), None),
        ],
        3,
    )
    b_dist = pick_distractors(
        _text(open_interval),
        [_dist(iv, tag) for iv, tag in _interval_candidates(s, 1, 1, a, k, open_interval, False)],
        3,
    )
    c_dist = pick_distractors(
        _text(full_interval),
        [_dist(iv, tag) for iv, tag in _interval_candidates(s, 1, 1, a, k, full_interval, True)],
        3,
    )
    d_dist = pick_distractors(
        rational_text(value),
        [
            _frac_dist(Fraction(1, k - t), "series-geometric-sum-sign"),
            _frac_dist(Fraction(s, k + s * t), "series-geometric-sum-sign"),
            _frac_dist(Fraction(s, k), "series-geometric-first-term-only"),
            _frac_dist(Fraction(k, k - s * t), "series-geometric-factor-dropped"),
        ],
        3,
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")

    series = _series_text(s, 1, 1, a, k)
    stem = f"The function f is defined by the power series f(x) = {series}, where the sum is taken over n = 1, 2, 3, …"
    open_text, full_text = _text(open_interval), _text(full_interval)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the radius of convergence of the series.",
                answer_kind="fraction",
                answer=exact_value(answer_a),
                answer_text=str(answer_a),
                marks=2,
                working=[
                    f"The ratio of consecutive terms tends to |x {'-' if a > 0 else '+'} {abs(a)}| / {k}.",
                    f"This is below 1 when |x {'-' if a > 0 else '+'} {abs(a)}| < {k}, so the radius is {k}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence write down the open interval on which the series converges.",
                answer_kind="inequality",
                answer=open_text,
                answer_text=open_text,
                answer_latex="plain:" + open_text,
                marks=1,
                working=[f"The series converges for |x {'-' if a > 0 else '+'} {abs(a)}| < {k}, that is {open_text}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the interval of convergence of the series, testing each endpoint.",
                answer_kind="inequality",
                answer=full_text,
                answer_text=full_text,
                answer_latex="plain:" + full_text,
                marks=3,
                working=_working_endpoints(s, 1, 1, a, k, full_interval) + [f"So the interval is {full_text}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Differentiating term by term, the series for f'(x) is geometric inside the interval. Find f'({x0}).",
                answer_kind="fraction",
                answer=exact_value(value),
                answer_text=rational_text(value),
                marks=2,
                follow_through=f"({s})/(part_a - ({s})*({t}))",
                working=[
                    f"f'(x) = Σ {'(-1)^n ' if s == -1 else ''}(x {'-' if a > 0 else '+'} {abs(a)})^(n-1) / {k}^n, "
                    f"a geometric series with first term {rational_text(Fraction(s, k))} and ratio "
                    f"{rational_text(ratio)} at x = {x0}.",
                    f"Its sum is {rational_text(Fraction(s, k))} / (1 - ({rational_text(ratio)})) = {rational_text(value)}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
