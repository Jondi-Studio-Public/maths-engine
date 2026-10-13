"""Generators for the Years 5-8 experimental-probability skills (`fpm_*`).

Four skills share this module, each with its own topics and an exam chain:

* ``f10-relative-frequency-estimate``: estimate a probability from a table of trial results (``fpm_relative_frequency``).
  Easy: one category. Medium: an event made of two categories, or "not" one category. Hard: the relative frequencies
  are given as percentages with one category unrecorded, so the missing share comes from "they sum to 100%".
* ``f10-expected-versus-observed``: predict a count from a theoretical probability, then compare with what happened
  (``fpm_expected_vs_observed`` and the chain ``fpm_expected_observed_exam``).
* ``f10-sample-size-and-reliability``: pool results, trust the larger experiment, see the estimate settle towards the
  theoretical value (``fpm_sample_size`` and ``fpm_sample_size_exam``). Trial counts are drawn from the probability
  with the seeded rng, so larger experiments really do sit closer to it.
* ``f10-compound-event-frequencies``: P(A and B), P(A or B), "neither" and "exactly one" read from two-way trial data
  (``fpm_compound_frequencies`` and ``fpm_compound_frequencies_exam``).

Every key is derived by construction and then re-derived a second way (a direct count over the table, SymPy-free exact
fractions). Percentages are only ever asked about experiments whose size divides 1000, so every percentage is exact to
one decimal place and the stated precision of the key is its true value.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

# -- small shared helpers ---------------------------------------------------------------------------------------


def _frac_d(value, tag):
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def _int_d(value, tag):
    return Distractor(str(int(value)), tag, int(value))


def _pct_text(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _pct_d(value, tag):
    value = round_half_up(Fraction(value), 1)
    return Distractor(stated_precision_string(value, 1), tag, numeric_answer(value))


def _pct(successes, trials) -> Fraction:
    return Fraction(100 * successes, trials)


def _check_exact_one_dp(value: Fraction):
    check((value * 10).denominator == 1, "a percentage is not exact to one decimal place")


def _join_or(labels) -> str:
    return " or ".join(labels)


class _Short(Exception):
    """A draw that cannot give three distinct wrong options; the generator redraws."""


def _need(*lists):
    if any(len(x) < 3 for x in lists):
        raise _Short


def _redraw(fn):
    """Re-run a generator body on the next draws when a draw could not give enough distinct distractors."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(200, "a draw with enough distinct distractors"):
            try:
                return fn(rng, *args, **kwargs)
            except _Short:
                continue

    return wrapper


# -- 1. relative frequency from a table --------------------------------------------------------------------------

_FREQ_CONTEXTS = (
    dict(
        lead="A spinner with four coloured sections was spun many times and the colour was recorded each time.",
        cats=["red", "blue", "green", "yellow"],
        pos="the next spin lands on {c}",
        neg="the next spin does not land on {c}",
    ),
    dict(
        lead=(
            "Marbles were drawn one at a time from a bag, each marble being put back before the next draw, "
            "and the colour was recorded."
        ),
        cats=["red", "blue", "green", "white"],
        pos="the next marble drawn is {c}",
        neg="the next marble drawn is not {c}",
    ),
    dict(
        lead="A six-sided die was rolled many times and the number rolled was recorded each time.",
        cats=["1", "2", "3", "4", "5", "6"],
        pos="the next number rolled is {c}",
        neg="the next number rolled is not {c}",
    ),
    dict(
        lead="Over one morning, the vehicles that passed a school gate were recorded by type.",
        cats=["car", "bus", "truck", "bicycle"],
        pos="the next vehicle is classified as {c}",
        neg="the next vehicle is not classified as {c}",
    ),
    dict(
        lead="A student flicked a bottle cap many times and recorded how it landed.",
        cats=["top up", "top down", "on its side"],
        pos="the next flick lands {c}",
        neg="the next flick does not land {c}",
    ),
)
_FREQ_ASK = (
    "Use the results to estimate the probability that {event}. Give your answer as a fraction in simplest form.",
    "Estimate, from these results, the probability that {event}. Give your answer as a fraction in simplest form.",
    "Based on the results, what is the estimated probability that {event}? Answer as a fraction in simplest form.",
)


def _event(ctx, cats, negate=False):
    if negate:
        return ctx["neg"].format(c=cats[0])
    return ctx["pos"].format(c=_join_or(cats))


@generator(
    "fpm_relative_frequency",
    level=5,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_relative_frequency(rng, tier: str = "easy"):
    ctx = rng.choice(_FREQ_CONTEXTS)
    cats = ctx["cats"]
    k = len(cats)
    if tier == "hard":
        return _freq_hard(rng, ctx)

    for _ in attempts(2000, "a results table with a plain estimate"):
        counts = [rng.randint(2, 16) for _ in range(k)]
        total = sum(counts)
        if not 20 <= total <= 90:
            continue
        if tier == "easy":
            i = rng.randrange(k)
            chosen = [i]
            negate = False
        else:
            negate = rng.random() < 0.4 and k >= 4
            i = rng.randrange(k)
            chosen = [i] if negate else sorted(rng.sample(range(k), 2))
        favourable = (total - counts[i]) if negate else sum(counts[j] for j in chosen)
        p = Fraction(favourable, total)
        if p in (Fraction(1, k), Fraction(1, 2)) or counts[i] == favourable and not negate and tier == "medium":
            continue
        break

    # Independent route: the favourable count as the total less every category NOT in the event.
    outside = [counts[j] for j in range(k) if (j == i if negate else j not in chosen)]
    check(Fraction(total - sum(outside), total) == p, "relative frequency disagrees with the complement route")
    check(0 < p < 1, "estimate is not a proper probability")

    table = ", ".join(f"{cat} {n}" for cat, n in zip(cats, counts))
    event = _event(ctx, [cats[j] for j in chosen], negate)
    stem = f"{ctx['lead']} The results were: {table}. " + phrase(rng, _FREQ_ASK, event=event)

    named = [counts[j] for j in chosen]
    others = [counts[j] for j in range(k) if j not in chosen]
    if tier == "easy":
        candidates = [
            _int_d(counts[i], "reports-frequency-not-relative-frequency"),
            _frac_d(Fraction(counts[i], max(others)), "divides-by-one-other-count"),
            _frac_d(Fraction(1, k), "assumes-equal-likelihood-ignores-data"),
            _frac_d(1 - p, "gave-complement-for-event"),
        ]
    elif negate:
        candidates = [
            _frac_d(Fraction(counts[i], total), "answered-the-event-not-its-complement"),
            _int_d(favourable, "reports-frequency-not-relative-frequency"),
            _frac_d(Fraction(favourable, counts[i]), "divides-by-one-other-count"),
            _frac_d(Fraction(k - 1, k), "assumes-equal-likelihood-ignores-data"),
        ]
    else:
        candidates = [
            _frac_d(Fraction(named[0], total), "counts-only-one-named-category"),
            _frac_d(Fraction(named[1], total), "counts-only-one-named-category"),
            _frac_d(Fraction(favourable, max(1, total - favourable)), "divides-by-one-other-count"),
            _frac_d(1 - p, "gave-complement-for-event"),
            _frac_d(Fraction(favourable, total - min(others)), "wrong-denominator-for-proportion"),
        ]
    candidates = [
        c for c in candidates if c.misconception == "reports-frequency-not-relative-frequency" or _is_proper(c)
    ]
    distractors = pick_distractors(rational_text(p), candidates, 3)
    _need(distractors)

    if negate:
        working = [
            f"There were {total} results in all, and {counts[i]} were {cats[i]}.",
            f"Not {cats[i]}: {total} - {counts[i]} = {favourable} results.",
            f"Relative frequency = {favourable}/{total} = {rational_text(p)}.",
        ]
    else:
        working = [
            f"There were {' + '.join(map(str, counts))} = {total} results in all.",
            f"The event covers {' + '.join(map(str, named))} = {favourable} of them.",
            f"Relative frequency = {favourable}/{total} = {rational_text(p)}.",
        ]
    return dict(
        text=stem,
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=latex_answer(rational_text(p)),
        working=working,
        distractors=distractors,
    )


def _freq_hard(rng, ctx):
    cats = ctx["cats"]
    k = len(cats)
    missing = rng.randrange(k)
    others = [j for j in range(k) if j != missing]
    other = rng.choice(others)
    for _ in attempts(2000, "relative frequencies with one category unrecorded"):
        miss = rng.randint(8, 40)
        rest = 100 - miss
        cuts = sorted(rng.sample(range(1, rest), k - 2)) if k > 2 else []
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, rest])]
        if len(parts) != k - 1 or min(parts) < 4:
            continue
        shares = {missing: miss}
        for j, share in zip(others, parts):
            shares[j] = share
        check(sum(shares.values()) == 100, "the relative frequencies do not sum to 100%")
        p = Fraction(shares[missing] + shares[other], 100)
        known_sum = Fraction(100 - shares[missing], 100)
        candidates = [
            _frac_d(Fraction(shares[other], 100), "omits-the-missing-category"),
            _frac_d(Fraction(shares[missing], 100), "counts-only-one-named-category"),
            _frac_d(known_sum, "adds-the-given-shares-instead-of-the-remainder"),
            _frac_d(1 - p, "gave-complement-for-event"),
            _frac_d(1 - Fraction(shares[other], 100), "adds-the-given-shares-instead-of-the-remainder"),
        ]
        distractors = pick_distractors(rational_text(p), [c for c in candidates if _is_proper(c)], 3)
        if len(distractors) == 3:
            break
    _need(distractors)
    # second route: the event is everything except the categories outside it
    check(p == 1 - Fraction(sum(shares[j] for j in range(k) if j not in (missing, other)), 100), "second route differs")

    shown = ", ".join(f"{cats[j]} {shares[j]}%" for j in range(k) if j != missing)
    event = ctx["pos"].format(c=_join_or([cats[i] for i in sorted((missing, other))]))
    stem = (
        f"{ctx['lead']} The relative frequencies of the results were: {shown}. "
        f"The relative frequency for {cats[missing]} was not recorded. " + phrase(rng, _FREQ_ASK, event=event)
    )
    return dict(
        text=stem,
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=latex_answer(rational_text(p)),
        working=[
            f"The relative frequencies add to 100%, so {cats[missing]} was 100% - {100 - shares[missing]}% "
            f"= {shares[missing]}%.",
            f"The event is {shares[missing]}% + {shares[other]}% = {shares[missing] + shares[other]}%.",
            f"As a fraction in simplest form: {rational_text(p)}.",
        ],
        distractors=distractors,
    )


# -- 2. expected versus observed --------------------------------------------------------------------------------

_DIE_EVENTS = (
    ("a number greater than 4", lambda v: v > 4),
    ("an even number", lambda v: v % 2 == 0),
    ("a 6", lambda v: v == 6),
    ("a number less than 3", lambda v: v < 3),
    ("a multiple of 3", lambda v: v % 3 == 0),
    ("a prime number", lambda v: v in (2, 3, 5)),
)
_DIE_UNIONS = (
    ("an even number or a number greater than 4", lambda v: v % 2 == 0, lambda v: v > 4),
    ("a multiple of 3 or an even number", lambda v: v % 3 == 0, lambda v: v % 2 == 0),
    ("a prime number or an even number", lambda v: v in (2, 3, 5), lambda v: v % 2 == 0),
    ("an odd number or a number less than 3", lambda v: v % 2 == 1, lambda v: v < 3),
    ("a number greater than 3 or a multiple of 3", lambda v: v > 3, lambda v: v % 3 == 0),
)
_ANGLES = (30, 40, 45, 60, 72, 75, 80, 90, 108, 120, 135, 150)
_EO_SPINNER_M = ((5, 1), (5, 2), (8, 3), (8, 5), (10, 3), (10, 7), (4, 1), (6, 1))
_EO_BAGS = ((3, 5, 2), (4, 3, 5), (6, 4, 5), (5, 5, 5), (7, 3, 2), (2, 6, 7))


def _device_simple(rng):
    """A theoretical probability read straight off the equipment: die, equal spinner or bag."""
    kind = rng.choice(["die", "spinner", "bag"])
    if kind == "die":
        ev, pred = rng.choice(_DIE_EVENTS)
        k = sum(1 for v in range(1, 7) if pred(v))
        return dict(
            m=6,
            k=k,
            setting="A fair six-sided die is rolled {n} times.",
            ev_q=f"{ev} to come up",
            obs=f"{ev} came up {{obs}} times",
            p=Fraction(k, 6),
            p_cands=[],
        )
    if kind == "spinner":
        m, k = rng.choice(_EO_SPINNER_M)
        colour = rng.choice(["blue", "green", "red"])
        return dict(
            m=m,
            k=k,
            setting=f"A fair spinner has {m} equal sections, {k} of them {colour}. It is spun {{n}} times.",
            ev_q=f"the spinner to land on {colour}",
            obs=f"the spinner landed on {colour} {{obs}} times",
            p=Fraction(k, m),
            p_cands=[],
        )
    r, b, g = rng.choice(_EO_BAGS)
    colour, name = rng.choice([(r, "red"), (b, "blue"), (g, "green")])
    m = r + b + g
    return dict(
        m=m,
        k=colour,
        setting=(
            f"A bag holds {r} red, {b} blue and {g} green marbles. "
            "A marble is drawn at random and then put back, {n} times in all."
        ),
        ev_q=f"a {name} marble to be drawn",
        obs=f"a {name} marble was drawn {{obs}} times",
        p=Fraction(colour, m),
        p_cands=[],
    )


def _device_compound(rng):
    """A theoretical probability that takes a second step: a sector angle, a removed marble or an overlapping 'or'."""
    kind = rng.choice(["angle", "removal", "union"])
    if kind == "angle":
        theta = rng.choice(_ANGLES)
        p = Fraction(theta, 360)
        return dict(
            kind=kind,
            setting=f"A spinner is spun {{n}} times. Its blue section has a centre angle of {theta}°.",
            ev_q="the spinner to land on blue",
            obs="the spinner landed on blue {obs} times",
            p=p,
            p_cands=[
                (Fraction(theta, 180), "uses-a-half-turn-as-the-whole-circle"),
                (1 - p, "gave-complement-for-event"),
                (Fraction(theta, 360 - theta), "compares-with-the-rest-not-the-whole"),
            ],
            working=[f"P(blue) = {theta}/360 = {rational_text(p)}."],
        )
    if kind == "removal":
        r, b, g = rng.choice([(6, 4, 5), (7, 5, 4), (5, 6, 4), (8, 4, 3), (9, 3, 4)])
        x = rng.choice([1, 2, 3])
        total = r + b + g
        p = Fraction(r - x, total - x)
        word = "marble is" if x == 1 else "marbles are"
        return dict(
            kind=kind,
            setting=(
                f"A bag holds {r} red, {b} blue and {g} green marbles. {x} red {word} removed and not replaced. "
                "A marble is then drawn at random and put back, {n} times in all."
            ),
            ev_q="a red marble to be drawn",
            obs="a red marble was drawn {obs} times",
            p=p,
            p_cands=[
                (Fraction(r, total), "ignores-the-removed-marbles"),
                (Fraction(r - x, total), "removes-from-the-red-but-not-the-total"),
                (Fraction(r - x, b + g), "compares-with-the-rest-not-the-whole"),
            ],
            working=[f"After removal there are {r - x} red marbles out of {total - x}: P(red) = {rational_text(p)}."],
        )
    ev, pa, pb = rng.choice(_DIE_UNIONS)
    outcomes = [v for v in range(1, 7) if pa(v) or pb(v)]
    na, nb = sum(1 for v in range(1, 7) if pa(v)), sum(1 for v in range(1, 7) if pb(v))
    both = sum(1 for v in range(1, 7) if pa(v) and pb(v))
    check(len(outcomes) == na + nb - both, "inclusion-exclusion disagrees with the listed outcomes")
    p = Fraction(len(outcomes), 6)
    return dict(
        kind=kind,
        setting="A fair six-sided die is rolled {n} times.",
        ev_q=f"{ev} to come up",
        obs=f"{ev} came up {{obs}} times",
        p=p,
        p_cands=[
            (Fraction(na + nb, 6), "overlap-counted-twice-in-or"),
            (Fraction(na, 6), "counts-only-one-named-category"),
            (Fraction(both, 6), "and-instead-of-or"),
        ],
        working=[f"The outcomes are {', '.join(map(str, outcomes))}: P = {len(outcomes)}/6 = {rational_text(p)}."],
    )


def _n_for(p: Fraction, rng, lo, hi) -> int:
    """A trial count in [lo, hi] that makes n * p a whole number."""
    q = p.denominator
    choices = [n for n in range(lo, hi + 1) if n % q == 0]
    check(bool(choices), "no trial count gives a whole expected number")
    return rng.choice(choices)


_OBS_ASK_MEDIUM = (
    "In the experiment, {obs}. By how many does the observed number of times differ from the expected number?",
    "In the experiment, {obs}. Find the difference between the observed number of times and the expected number.",
)


@generator(
    "fpm_expected_vs_observed",
    level=6,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_expected_vs_observed(rng, tier: str = "easy"):
    if tier == "hard":
        dev = _device_compound(rng)
        lo, hi = 60, 600
    else:
        dev = _device_simple(rng)
        lo, hi = 30, 400
    p = dev["p"]
    n = _n_for(p, rng, lo, hi)
    expected = p * n
    check(expected.denominator == 1 and 0 < p < 1, "expected count is not a whole number")
    check(expected == Fraction(n * p.numerator, p.denominator), "expected count differs by the direct route")
    expected = int(expected)
    setting = dev["setting"].format(n=n)

    if tier == "easy":
        m, k = (dev["m"], dev["k"]) if "m" in dev else (p.denominator, p.numerator)
        candidates = [
            _int_d(n // m, "expected-count-uses-one-outcome"),
            _int_d(n - expected, "expected-complement-count"),
            _int_d(n * k, "expected-count-multiplies-by-favourable-only"),
            _int_d(round_half_up(Fraction(n * k, m - k)), "compares-with-the-rest-not-the-whole"),
        ]
        candidates = [c for c in candidates if int(c.text) != expected]
        distractors = pick_distractors(str(expected), candidates, 3, tolerance=GRADER_TOLERANCE)
        _need(distractors)
        return dict(
            text=f"{setting} How many times would you expect {dev['ev_q']}?",
            answer_kind="number",
            answer=expected,
            answer_text=str(expected),
            answer_latex=latex_answer(str(expected)),
            working=[
                f"P = {rational_text(p)}.",
                f"Expected number = {rational_text(p)} × {n} = {expected}.",
            ],
            distractors=distractors,
        )

    delta = rng.randint(3, max(4, expected // 4))
    sign = rng.choice([-1, 1])
    observed = expected + sign * delta
    check(observed > 0 and observed != expected, "observed count is not a plausible different count")
    candidates = [
        _int_d(observed, "reports-observed-not-difference"),
        _int_d(expected, "reports-expected-not-difference"),
        _int_d(observed + expected, "adds-instead-of-subtracting"),
        _int_d(abs(observed - (n - expected)), "expected-complement-count"),
        _int_d(abs(observed - n // p.denominator), "expected-count-uses-one-outcome"),
    ]
    for pr, tag in dev.get("p_cands", []):
        if 0 < pr < 1:
            wrong = pr * n
            candidates.insert(0, _int_d(abs(observed - round_half_up(wrong)), tag))
    candidates = [c for c in candidates if int(c.text) != delta and int(c.text) > 0]
    distractors = pick_distractors(str(delta), candidates, 3, tolerance=GRADER_TOLERANCE)
    _need(distractors)
    stem = f"{setting} " + phrase(rng, _OBS_ASK_MEDIUM, obs=dev["obs"].format(obs=observed))
    return dict(
        text=stem,
        answer_kind="number",
        answer=delta,
        answer_text=str(delta),
        answer_latex=latex_answer(str(delta)),
        working=[
            *dev.get("working", [f"P = {rational_text(p)}."]),
            f"Expected number = {rational_text(p)} × {n} = {expected}.",
            f"Difference = |{observed} - {expected}| = {delta}.",
        ],
        distractors=distractors,
    )


@generator("fpm_expected_observed_exam", level=7, presets={"exam": {}})
@_redraw
def fpm_expected_observed_exam(rng):
    dev = _device_compound(rng)
    p = dev["p"]
    n = _n_for(p, rng, 80, 600)
    expected = int(p * n)
    delta = rng.randint(3, max(4, expected // 4))
    observed = expected + rng.choice([-1, 1]) * delta
    check(observed > 0, "observed count must be positive")
    need = _n_for(p, rng, 150, 1500)
    target = int(p * need)
    check(Fraction(target, need) == p, "target count does not match the probability")
    stem = dev["setting"].format(n=n) + f" In the experiment, {dev['obs'].format(obs=observed)}."

    p_text = rational_text(p)
    a_cands = [_frac_d(pr, tag) for pr, tag in dev["p_cands"] if 0 < pr < 1]
    a_cands.append(_frac_d(Fraction(observed, n), "uses-observed-as-theoretical-probability"))
    a_d = pick_distractors(p_text, a_cands, 3)
    b_cands = [
        _int_d(observed, "reports-observed-not-expected"),
        _int_d(n - expected, "expected-complement-count"),
        _int_d(n // p.denominator, "expected-count-uses-one-outcome"),
        _int_d(n * p.numerator, "expected-count-multiplies-by-favourable-only"),
    ]
    b_d = pick_distractors(str(expected), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_cands = [
        _int_d(observed, "reports-observed-not-difference"),
        _int_d(expected, "reports-expected-not-difference"),
        _int_d(observed + expected, "adds-instead-of-subtracting"),
        _int_d(abs(observed - (n - expected)), "expected-complement-count"),
    ]
    c_d = pick_distractors(str(delta), [c for c in c_cands if int(c.text) != delta], 3, tolerance=GRADER_TOLERANCE)
    d_cands = [
        _int_d(target, "reports-the-target-not-the-trials"),
        _int_d(int(target * p), "multiplies-instead-of-dividing"),
        _int_d(target * p.numerator, "multiplies-instead-of-dividing"),
        _int_d(
            int(target * (1 - p)) if (target * (1 - p)).denominator == 1 else target + 1, "divides-by-the-complement"
        ),
        _int_d(target + need // 2, "adds-instead-of-dividing"),
    ]
    d_d = pick_distractors(str(need), [c for c in d_cands if int(c.text) != need], 3, tolerance=GRADER_TOLERANCE)
    _need(a_d, b_d, c_d, d_d)

    ev_phrase = dev["ev_q"]
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the theoretical probability of the event, as a fraction in simplest form.",
                answer_kind="fraction",
                answer=exact_value(p),
                answer_text=p_text,
                answer_latex=latex_answer(p_text),
                marks=2,
                sensible_range=(0, 1),
                working=dev["working"],
                distractors=a_d,
            ),
            dict(
                prompt=f"Hence find how many times you would expect {ev_phrase} in the {n} trials.",
                answer_kind="number",
                answer=expected,
                answer_text=str(expected),
                answer_latex=latex_answer(str(expected)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{n}",
                working=[f"Expected number = {p_text} × {n} = {expected}."],
                distractors=b_d,
            ),
            dict(
                prompt="By how many does the observed number of times differ from the expected number?",
                answer_kind="number",
                answer=delta,
                answer_text=str(delta),
                answer_latex=latex_answer(str(delta)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"Abs({observed} - part_b)",
                working=[f"|{observed} - {expected}| = {delta}."],
                distractors=c_d,
            ),
            dict(
                prompt=(
                    f"How many trials would be needed to expect {ev_phrase} {target} times? "
                    "Use the theoretical probability from (a)."
                ),
                answer_kind="number",
                answer=need,
                answer_text=str(need),
                answer_latex=latex_answer(str(need)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{target}/part_a",
                working=[f"{target} ÷ {p_text} = {need} trials."],
                distractors=d_d,
            ),
        ],
    )


# -- 3. sample size and reliability ------------------------------------------------------------------------------

# (lead with {n} placeholders handled per use, success phrase, theoretical probability or None, hidden-rate options)
_SS_CONTEXTS = (
    dict(
        action="tossed a fair coin {n} times.",
        result="Heads came up {s} times.",
        unit="toss",
        p=Fraction(1, 2),
        thing="the probability that a toss lands heads",
    ),
    dict(
        action="spun a fair spinner with 5 equal sections, one of them red, {n} times.",
        result="It landed on red {s} times.",
        unit="spin",
        p=Fraction(1, 5),
        thing="the probability that a spin lands on red",
    ),
    dict(
        action="drew a marble {n} times from a bag of 3 red and 7 blue marbles, putting it back each time.",
        result="A red marble was drawn {s} times.",
        unit="draw",
        p=Fraction(3, 10),
        thing="the probability that a marble drawn is red",
    ),
    dict(
        action="spun a fair spinner with 4 equal sections, one of them green, {n} times.",
        result="It landed on green {s} times.",
        unit="spin",
        p=Fraction(1, 4),
        thing="the probability that a spin lands on green",
    ),
)
_SS_UNKNOWN = (
    dict(
        action="took {n} free throws.",
        result="{s} of the throws were scored.",
        unit="throw",
        p=None,
        hidden=Fraction(13, 20),
        thing="the probability that this player scores a free throw",
    ),
    dict(
        action="dropped a drawing pin {n} times.",
        result="It landed point up {s} times.",
        unit="drop",
        p=None,
        hidden=Fraction(2, 5),
        thing="the probability that a dropped drawing pin lands point up",
    ),
    dict(
        action="took {n} kicks at goal from the same spot.",
        result="{s} of the kicks scored.",
        unit="kick",
        p=None,
        hidden=Fraction(7, 20),
        thing="the probability that a kick from that spot scores",
    ),
)
_NAMES = ("Ava", "Ben", "Chloe", "Dev", "Ella", "Finn", "Grace", "Hamid", "Isla", "Jack", "Kiri", "Liam")
_SIZES = (20, 40, 50, 100, 200, 250, 500)  # every size divides 1000, so a percentage is exact to 1 d.p.
_TOTALS = (50, 100, 200, 250, 500, 1000)


def _simulate(rng, n: int, rate: Fraction) -> int:
    """Successes in n trials of the given rate, drawn with the seeded rng."""
    threshold = float(rate)
    return sum(1 for _ in range(n) if rng.random() < threshold)


def _rate(ctx) -> Fraction:
    return ctx["p"] if ctx["p"] is not None else ctx["hidden"]


def _report(name, ctx, n, s) -> str:
    return f"{name} {ctx['action'].format(n=n)} {ctx['result'].format(s=s)}"


@generator(
    "fpm_sample_size",
    level=6,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_sample_size(rng, tier: str = "easy"):
    if tier == "hard" and rng.random() < 0.5:
        return _ss_closer(rng)
    ctx = rng.choice(_SS_CONTEXTS + _SS_UNKNOWN)
    rate = _rate(ctx)
    names = rng.sample(_NAMES, 3)
    if tier == "easy":
        return _ss_pool(rng, ctx, rate, names[:2], two=True)
    if tier == "medium":
        return _ss_pick(rng, ctx, rate, names)
    return _ss_pool(rng, ctx, rate, names, two=False)


def _draw_sizes(rng, total: int, groups: int, multiple: int = 10, minimum: int = 20):
    """Split `total` into `groups` unequal parts, each a multiple of `multiple` and at least `minimum`."""
    for _ in attempts(2000, "an unequal split of the trials"):
        cuts = sorted(rng.sample(range(minimum // multiple, (total - minimum) // multiple + 1), groups - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, total // multiple])]
        parts = [x * multiple for x in parts]
        if len(parts) == groups and min(parts) >= minimum and len(set(parts)) == groups:
            return parts
    raise AssertionError  # pragma: no cover


def _ss_pool(rng, ctx, rate, names, two):
    groups = len(names)
    for _ in attempts(500, "pooled results whose pooled and averaged percentages differ"):
        total = rng.choice([200, 250, 500, 1000] if groups == 3 else [50, 100, 200, 250, 500])
        sizes = _draw_sizes(rng, total, groups, multiple=10, minimum=20)
        # one small group and the rest larger, so a plain average of percentages misleads
        if max(sizes) < 2 * min(sizes):
            continue
        successes = [_simulate(rng, n, rate) for n in sizes]
        pooled = _pct(sum(successes), total)
        _check_exact_one_dp(pooled)
        percents = [Fraction(100 * s, n) for s, n in zip(successes, sizes)]
        mean_of_percents = sum(percents) / groups
        if round_half_up(mean_of_percents, 1) != round_half_up(pooled, 1) and all(
            0 < s < n for s, n in zip(successes, sizes)
        ):
            break
    # second route: weight each group's percentage by its share of the trials
    check(sum(pc * Fraction(n, total) for pc, n in zip(percents, sizes)) == pooled, "pooled percentage differs")
    reports = " ".join(_report(nm, ctx, n, s) for nm, n, s in zip(names, sizes, successes))
    joint = "Their" if two else "All three"
    stem = (
        f"{reports} {joint} results are combined, as if the {total} {ctx['unit']}s were one experiment. "
        f"Use the combined results to estimate {ctx['thing']}. Give your answer as a percentage, correct to 1 decimal place."
    )
    candidates = [_pct_d(percents[j], "uses-one-experiment-only") for j in range(groups)]
    candidates.insert(0, _pct_d(mean_of_percents, "averages-percentages-ignoring-trial-numbers"))
    if not two:
        candidates.append(
            _pct_d(
                _pct(sum(successes) - successes[sizes.index(min(sizes))], total - min(sizes)),
                "leaves-a-group-out-of-the-total",
            )
        )
    candidates.append(_pct_d(_pct(sum(successes), total - min(sizes)), "wrong-denominator-for-proportion"))
    candidates = [c for c in candidates if float(c.value) <= 100]
    distractors = pick_distractors(_pct_text(pooled), candidates, 3, tolerance=GRADER_TOLERANCE)
    _need(distractors)
    text = _pct_text(pooled)
    return dict(
        text=stem,
        answer_kind="number",
        answer=numeric_answer(pooled),
        answer_text=text,
        answer_latex=latex_answer(text),
        working=[
            f"Successes in all: {' + '.join(map(str, successes))} = {sum(successes)}.",
            f"Trials in all: {' + '.join(map(str, sizes))} = {total}.",
            f"Estimate = {sum(successes)} ÷ {total} = {text}%.",
        ],
        distractors=distractors,
    )


def _ss_pick(rng, ctx, rate, names):
    for _ in attempts(500, "three experiments of distinct sizes with distinct percentages"):
        sizes = rng.sample(_SIZES, 3)
        successes = [_simulate(rng, n, rate) for n in sizes]
        percents = [_pct(s, n) for s, n in zip(successes, sizes)]
        if (
            len(set(percents)) == 3
            and max(sizes) >= 4 * min(sizes)
            and all(0 < s < n for s, n in zip(successes, sizes))
        ):
            break
    big = sizes.index(max(sizes))
    for pc in percents:
        _check_exact_one_dp(pc)
    reports = " ".join(_report(nm, ctx, n, s) for nm, n, s in zip(names, sizes, successes))
    stem = (
        f"{reports} {names[0]} will use just one of the three results as the estimate of {ctx['thing']}. "
        "The estimate should be the one most likely to be close to the true probability. "
        "Which estimate should be used? Give your answer as a percentage, correct to 1 decimal place."
    )
    text = _pct_text(percents[big])
    candidates = [_pct_d(percents[j], "trusts-a-small-experiment") for j in range(3) if j != big]
    candidates.append(_pct_d(sum(percents) / 3, "averages-percentages-ignoring-trial-numbers"))
    candidates.append(_pct_d(_pct(sum(successes), sum(sizes)), "pools-when-asked-for-one-result"))
    distractors = pick_distractors(text, candidates, 3, tolerance=GRADER_TOLERANCE)
    _need(distractors)
    return dict(
        text=stem,
        answer_kind="number",
        answer=numeric_answer(percents[big]),
        answer_text=text,
        answer_latex=latex_answer(text),
        working=[
            f"The largest experiment has {sizes[big]} trials, so its result is the most reliable.",
            f"Estimate = {successes[big]} ÷ {sizes[big]} = {text}%.",
        ],
        distractors=distractors,
    )


def _ss_closer(rng):
    ctx = rng.choice(_SS_CONTEXTS)
    rate = ctx["p"]
    ptheory = Fraction(100) * rate
    _check_exact_one_dp(ptheory)
    for _ in attempts(500, "a small experiment far from p and a pooled one close to it"):
        total = rng.choice([250, 500, 1000])
        small = rng.choice([20, 40, 50])
        rest = _draw_sizes(rng, total - small, 2, multiple=10, minimum=50)
        sizes = [small, *rest]
        successes = [_simulate(rng, n, rate) for n in sizes]
        small_pct = _pct(successes[0], small)
        pooled = _pct(sum(successes), total)
        _check_exact_one_dp(pooled)
        dist_small, dist_pool = abs(small_pct - ptheory), abs(pooled - ptheory)
        if dist_small - dist_pool >= 4 and small_pct != pooled:
            break
    answer = dist_small - dist_pool
    names = rng.sample(_NAMES, 3)
    reports = " ".join(_report(nm, ctx, n, s) for nm, n, s in zip(names, sizes, successes))
    stem = (
        f"The theoretical value of {ctx['thing']} is {rational_text(rate)}, or "
        f"{_pct_text(ptheory)}%. {reports} Compare {names[0]}'s result on its own with all three results combined. "
        "By how many percentage points is the combined estimate closer to the theoretical probability than "
        f"{names[0]}'s estimate? Give your answer correct to 1 decimal place."
    )
    text = _pct_text(answer)
    candidates = [
        _pct_d(dist_small, "reports-one-distance-only"),
        _pct_d(dist_pool, "reports-one-distance-only"),
        _pct_d(abs(small_pct - pooled), "compares-the-two-estimates-not-the-distance-to-p"),
        _pct_d(dist_small + dist_pool, "adds-the-distances"),
    ]
    candidates = [c for c in candidates if float(c.value) <= 100]
    distractors = pick_distractors(text, candidates, 3, tolerance=GRADER_TOLERANCE)
    _need(distractors)
    return dict(
        text=stem,
        answer_kind="number",
        answer=numeric_answer(answer),
        answer_text=text,
        answer_latex=latex_answer(text),
        working=[
            f"{names[0]}'s estimate is {_pct_text(small_pct)}%, which is {_pct_text(dist_small)} points from {_pct_text(ptheory)}%.",
            f"The combined estimate is {_pct_text(pooled)}%, which is {_pct_text(dist_pool)} points from {_pct_text(ptheory)}%.",
            f"Closer by {_pct_text(dist_small)} - {_pct_text(dist_pool)} = {text} percentage points.",
        ],
        distractors=distractors,
    )


@generator("fpm_sample_size_exam", level=7, presets={"exam": {}})
@_redraw
def fpm_sample_size_exam(rng):
    ctx = rng.choice(_SS_CONTEXTS)
    rate = ctx["p"]
    ptheory = Fraction(100) * rate
    for _ in attempts(500, "two classes whose pooled estimate is not the theoretical value"):
        total = rng.choice([250, 500, 1000])
        n1 = rng.choice([20, 30, 40, 50])
        n2 = total - n1
        s1, s2 = _simulate(rng, n1, rate), _simulate(rng, n2, rate)
        pooled = _pct(s1 + s2, total)
        _check_exact_one_dp(pooled)
        if pooled != ptheory and 0 < s1 < n1:
            break
    s = s1 + s2
    gap = abs(pooled - ptheory)
    future = rng.choice([1000, 2000])
    predicted = pooled * future / 100
    check(predicted.denominator == 1, "prediction is not a whole number")
    predicted = int(predicted)
    names = rng.sample(_NAMES, 2)
    stem = (
        f"{_report(names[0], ctx, n1, s1)} {_report(names[1], ctx, n2, s2)} "
        f"The theoretical value of {ctx['thing']} is {rational_text(rate)}, or {_pct_text(ptheory)}%. "
        "The two sets of results are combined."
    )
    pct_text, gap_text = _pct_text(pooled), _pct_text(gap)
    a_d = pick_distractors(
        str(s),
        [
            _int_d(s1, "uses-one-experiment-only"),
            _int_d(s2, "uses-one-experiment-only"),
            _int_d(total, "reports-the-trials-not-the-successes"),
            _int_d(s + n1, "adds-a-group-of-trials-to-the-successes"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_d = pick_distractors(
        pct_text,
        [
            _pct_d(_pct(s1, n1), "uses-one-experiment-only"),
            _pct_d(_pct(s2, n2), "uses-one-experiment-only"),
            _pct_d((_pct(s1, n1) + _pct(s2, n2)) / 2, "averages-percentages-ignoring-trial-numbers"),
            _pct_d(_pct(s, total - n1), "wrong-denominator-for-proportion"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(
        gap_text,
        [
            _pct_d(ptheory, "reports-the-theoretical-value-not-the-gap"),
            _pct_d(pooled, "reports-the-estimate-not-the-gap"),
            _pct_d(abs(_pct(s1, n1) - ptheory), "uses-one-experiment-only"),
            _pct_d(abs(pooled - ptheory) / 10, "slips-a-decimal-place"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    d_d = pick_distractors(
        str(predicted),
        [
            _int_d(round_half_up(ptheory * future / 100), "uses-theoretical-not-the-estimate"),
            _int_d(round_half_up(_pct(s1, n1) * future / 100), "uses-one-experiment-only"),
            _int_d(s, "reports-raw-count-not-projection"),
            _int_d(round_half_up((100 - pooled) * future / 100), "projects-the-complement"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    _need(a_d, b_d, c_d, d_d)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many times did the event occur in the combined results?",
                answer_kind="number",
                answer=s,
                answer_text=str(s),
                answer_latex=latex_answer(str(s)),
                marks=1,
                sensible_range=(0, None),
                working=[f"{s1} + {s2} = {s}."],
                distractors=a_d,
            ),
            dict(
                prompt="Use the combined results to estimate the probability, as a percentage correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(pooled),
                answer_text=pct_text,
                answer_latex=latex_answer(pct_text),
                marks=2,
                sensible_range=(0, 100),
                follow_through=f"100*part_a/{total}",
                working=[f"{s} ÷ {total} × 100 = {pct_text}%."],
                distractors=b_d,
            ),
            dict(
                prompt=(
                    "By how many percentage points is your estimate from (b) from the theoretical probability? "
                    "Give your answer correct to 1 decimal place."
                ),
                answer_kind="number",
                answer=numeric_answer(gap),
                answer_text=gap_text,
                answer_latex=latex_answer(gap_text),
                marks=1,
                sensible_range=(0, 100),
                follow_through=f"Abs(part_b - {ptheory})",
                working=[f"|{pct_text} - {_pct_text(ptheory)}| = {gap_text} percentage points."],
                distractors=c_d,
            ),
            dict(
                prompt=f"Use your estimate from (b) to predict how many times the event would occur in {future} further trials.",
                answer_kind="number",
                answer=predicted,
                answer_text=str(predicted),
                answer_latex=latex_answer(str(predicted)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b*{future}/100",
                working=[f"{pct_text}% of {future} = {predicted}."],
                distractors=d_d,
            ),
        ],
    )


# -- 4. compound events from trial data -------------------------------------------------------------------------

# a_pos / a_neg / b_pos / b_neg are clauses for "on a trial, ..."
_CE_CONTEXTS = (
    dict(
        lead="A coin was tossed and a spinner was spun together, {n} times in all.",
        rows=("Heads", "Tails"),
        cols=("Red", "Blue"),
        row_title="coin",
        col_title="spinner",
        a="the coin lands heads",
        b="the spinner lands on red",
        unit="trial",
    ),
    dict(
        lead="A basketball player took two free throws in each of {n} rounds.",
        rows=("Scored", "Missed"),
        cols=("Scored", "Missed"),
        row_title="first throw",
        col_title="second throw",
        a="the first throw is scored",
        b="the second throw is scored",
        unit="round",
    ),
    dict(
        lead="On each of {n} mornings, a student recorded whether it rained and whether the school bus was late.",
        rows=("Rain", "No rain"),
        cols=("Late", "On time"),
        row_title="weather",
        col_title="bus",
        a="it rains",
        b="the bus is late",
        unit="morning",
    ),
    dict(
        lead="{n} seeds were planted, some in sun and some in shade, and each was checked for sprouting.",
        rows=("Sun", "Shade"),
        cols=("Sprouted", "Did not sprout"),
        row_title="position",
        col_title="result",
        a="the seed is planted in the sun",
        b="the seed sprouts",
        unit="seed",
    ),
    dict(
        lead="{n} customers at a canteen were recorded by whether they bought a drink and whether they bought food.",
        rows=("Drink", "No drink"),
        cols=("Food", "No food"),
        row_title="drink",
        col_title="food",
        a="the customer buys a drink",
        b="the customer buys food",
        unit="customer",
    ),
)


def _grid_text(ctx, grid, totals=True, unknown=None):
    """The two-way table as one line of text, `?` in the cell `unknown` (row, col)."""
    (a, ab), (ba, bb) = grid
    r1, r2 = ctx["rows"]
    c1, c2 = ctx["cols"]

    def cell(i, j, v):
        return "?" if unknown == (i, j) else str(v)

    head = f"Table ({ctx['row_title']} / {ctx['col_title']} — {c1} — {c2}"
    if totals:
        head += " — Total"
        body = (
            f"{r1} — {cell(0, 0, a)} — {cell(0, 1, ab)} — {a + ab}; "
            f"{r2} — {cell(1, 0, ba)} — {cell(1, 1, bb)} — {ba + bb}; "
            f"Total — {a + ba} — {ab + bb} — {a + ab + ba + bb}"
        )
    else:
        body = f"{r1} — {cell(0, 0, a)} — {cell(0, 1, ab)}; {r2} — {cell(1, 0, ba)} — {cell(1, 1, bb)}"
    return f"{head}): {body}."


def _draw_grid(rng, lo=3, hi=22, total_lo=40, total_hi=120):
    for _ in attempts(2000, "a two-way table of trial counts"):
        cells = [rng.randint(lo, hi) for _ in range(4)]
        total = sum(cells)
        if total_lo <= total <= total_hi:
            return cells, total
    raise AssertionError  # pragma: no cover


def _is_proper(d: Distractor) -> bool:
    v = d.value
    value = Fraction(v["p"], v["q"]) if isinstance(v, dict) else Fraction(v)
    return 0 < value < 1


_FRACTION_SUFFIX = "Give your answer as a fraction in simplest form."


@generator(
    "fpm_compound_frequencies",
    level=8,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_compound_frequencies(rng, tier: str = "easy"):
    ctx = rng.choice(_CE_CONTEXTS)
    cells, total = _draw_grid(rng, 3, 22) if tier != "hard" else _draw_grid(rng, 4, 24)
    both, a_only, b_only, neither = cells
    n_a, n_b = both + a_only, both + b_only
    lead = ctx["lead"].format(n=total)
    a, b, u = ctx["a"], ctx["b"], ctx["unit"]
    if tier == "easy":
        grid = ((both, a_only), (b_only, neither))
        p = Fraction(both, total)
        check(p == Fraction(total - a_only - b_only - neither, total), "cell differs from the remainder route")
        text = (
            f"{lead} {_grid_text(ctx, grid)} Use the results to estimate the probability that, for one {u} chosen "
            f"at random, {a} and {b}. {_FRACTION_SUFFIX}"
        )
        candidates = [
            _frac_d(Fraction(both, n_a), "divides-by-one-row-total"),
            _frac_d(Fraction(n_a, total), "counts-only-one-named-category"),
            _frac_d(Fraction(both, n_b), "divides-by-one-row-total"),
            _frac_d(Fraction(n_a + n_b - both, total), "or-instead-of-and"),
        ]
        working = [f"{both} of the {total} {u}s had both.", f"P = {both}/{total} = {rational_text(p)}."]
    elif tier == "medium":
        grid = ((both, a_only), (b_only, neither))
        p = Fraction(n_a + n_b - both, total)
        check(p == 1 - Fraction(neither, total), "or-probability disagrees with the complement")
        text = (
            f"{lead} {_grid_text(ctx, grid)} Use the results to estimate the probability that, for one {u} chosen "
            f"at random, {a} or {b} (or both). {_FRACTION_SUFFIX}"
        )
        candidates = [
            _frac_d(Fraction(n_a + n_b, total), "overlap-counted-twice-in-or"),
            _frac_d(Fraction(both, total), "and-instead-of-or"),
            _frac_d(Fraction(a_only + b_only, total), "leaves-out-the-overlap-for-or"),
            _frac_d(Fraction(neither, total), "answered-neither-not-or"),
            _frac_d(Fraction(n_a, total), "counts-only-one-named-category"),
        ]
        working = [
            f"{a_only} + {both} + {b_only} = {n_a + n_b - both} {u}s had at least one.",
            f"P = {n_a + n_b - both}/{total} = {rational_text(p)}.",
        ]
    else:
        shape = rng.choice(["neither", "exactly", "only"])
        intro = (
            f"{lead} Let A be the event that {a}, and B the event that {b}. A happened {n_a} times, "
            f"B happened {n_b} times, and both A and B happened {both} times."
        )
        if shape == "neither":
            p = Fraction(neither, total)
            ask = "Estimate the probability that neither A nor B happens."
            candidates = [
                _frac_d(
                    Fraction(total - n_a - n_b, total) if total > n_a + n_b else Fraction(n_a + n_b - both, total),
                    "subtracts-both-totals-without-adding-back-the-overlap",
                ),
                _frac_d(Fraction(n_a + n_b - both, total), "answered-the-event-not-its-complement"),
                _frac_d(Fraction(both, total), "and-instead-of-or"),
                _frac_d(Fraction(total - n_a, total), "ignores-the-second-event"),
                _frac_d(Fraction(a_only + b_only, total), "leaves-out-the-overlap-for-or"),
            ]
            working = [
                f"At least one of A and B: {n_a} + {n_b} - {both} = {n_a + n_b - both}.",
                f"Neither: {total} - {n_a + n_b - both} = {neither}, so P = {neither}/{total} = {rational_text(p)}.",
            ]
        elif shape == "exactly":
            p = Fraction(a_only + b_only, total)
            ask = "Estimate the probability that exactly one of A and B happens."
            candidates = [
                _frac_d(Fraction(n_a + n_b, total), "overlap-counted-twice-in-or"),
                _frac_d(Fraction(n_a + n_b - both, total), "or-instead-of-exactly-one"),
                _frac_d(Fraction(both, total), "and-instead-of-or"),
                _frac_d(Fraction(a_only, total), "counts-only-one-named-category"),
            ]
            working = [
                f"Only A: {n_a} - {both} = {a_only}. Only B: {n_b} - {both} = {b_only}.",
                f"P = ({a_only} + {b_only})/{total} = {rational_text(p)}.",
            ]
        else:
            p = Fraction(a_only, total)
            ask = "Estimate the probability that A happens but B does not."
            candidates = [
                _frac_d(Fraction(n_a, total), "ignores-the-overlap-for-but-not"),
                _frac_d(Fraction(both, total), "and-instead-of-but-not"),
                _frac_d(Fraction(b_only, total), "swaps-the-two-events"),
                _frac_d(Fraction(a_only, n_a), "wrong-denominator-for-proportion"),
            ]
            working = [
                f"{n_a} - {both} = {a_only} {u}s had A without B.",
                f"P = {a_only}/{total} = {rational_text(p)}.",
            ]
        text = f"{intro} {ask} {_FRACTION_SUFFIX}"
    check(0 < p < 1, "not a proper probability")
    distractors = pick_distractors(rational_text(p), [c for c in candidates if _is_proper(c)], 3)
    _need(distractors)
    return dict(
        text=text,
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=latex_answer(rational_text(p)),
        working=working,
        distractors=distractors,
    )


@generator("fpm_compound_frequencies_exam", level=9, presets={"exam": {}})
@_redraw
def fpm_compound_frequencies_exam(rng):
    ctx = rng.choice(_CE_CONTEXTS)
    cells, total = _draw_grid(rng, 4, 24, 60, 140)
    both, a_only, b_only, neither = cells
    n_a, n_b = both + a_only, both + b_only
    union = n_a + n_b - both
    p_both, p_union = Fraction(both, total), Fraction(union, total)
    check(p_union == Fraction(a_only + b_only + both, total) == 1 - Fraction(neither, total), "union differs")
    projected = Fraction(0)
    future = 100
    for _ in attempts(500, "a future count clear of a rounding boundary"):
        future = rng.choice([100, 120, 150, 200, 250, 300, 400, 500, 600])
        projected = future * p_union
        if abs((projected % 1) - Fraction(1, 2)) > Fraction(1, 10):
            break
    answer_d = int(round_half_up(projected))
    grid = ((both, a_only), (b_only, neither))
    table = _grid_text(ctx, grid, totals=False, unknown=(0, 0))
    lead = ctx["lead"].format(n=total)
    a, b, u = ctx["a"], ctx["b"], ctx["unit"]
    stem = (
        f"{lead} Let A be the event that {a}, and B the event that {b}. One count in the table has been lost. {table}"
    )
    known = a_only + b_only + neither
    a_d = pick_distractors(
        str(both),
        [
            _int_d(known, "reports-known-sum-not-remainder"),
            _int_d(total, "reports-total-as-missing-value"),
            _int_d(total - a_only - b_only, "left-a-class-out-of-the-sum"),
            _int_d(max(1, both - 1), "miscounts-by-one"),
            _int_d(both + 1, "miscounts-by-one"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_cands = [
        _frac_d(Fraction(both, n_a), "divides-by-one-row-total"),
        _frac_d(Fraction(a_only + b_only, total), "leaves-out-the-overlap-for-or"),
        _frac_d(Fraction(n_a, total), "counts-only-one-named-category"),
        _frac_d(p_union, "or-instead-of-and"),
    ]
    c_cands = [
        _frac_d(Fraction(n_a + n_b, total), "overlap-counted-twice-in-or"),
        _frac_d(Fraction(a_only + b_only, total), "leaves-out-the-overlap-for-or"),
        _frac_d(p_both, "and-instead-of-or"),
        _frac_d(Fraction(neither, total), "answered-neither-not-or"),
    ]
    b_d = pick_distractors(rational_text(p_both), [c for c in b_cands if _is_proper(c)], 3)
    c_d = pick_distractors(rational_text(p_union), [c for c in c_cands if _is_proper(c)], 3)
    d_d = pick_distractors(
        str(answer_d),
        [
            _int_d(round_half_up(future * (1 - p_union)), "projects-the-complement"),
            _int_d(round_half_up(future * p_both), "and-instead-of-or"),
            _int_d(union, "reports-raw-count-not-projection"),
            _int_d(round_half_up(future * Fraction(n_a + n_b, total)), "overlap-counted-twice-in-or"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    _need(a_d, b_d, c_d, d_d)
    pb, pu = rational_text(p_both), rational_text(p_union)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the missing count: the number of {u}s in which both A and B happen.",
                answer_kind="number",
                answer=both,
                answer_text=str(both),
                answer_latex=latex_answer(str(both)),
                marks=1,
                sensible_range=(0, total),
                working=[f"{total} - {a_only} - {b_only} - {neither} = {both}."],
                distractors=a_d,
            ),
            dict(
                prompt=(
                    f"Estimate the probability that, for one {u} chosen at random, both A and B happen. "
                    f"{_FRACTION_SUFFIX}"
                ),
                answer_kind="fraction",
                answer=exact_value(p_both),
                answer_text=pb,
                answer_latex=latex_answer(pb),
                marks=1,
                sensible_range=(0, 1),
                follow_through=f"part_a/{total}",
                working=[f"P = {both}/{total} = {pb}."],
                distractors=b_d,
            ),
            dict(
                prompt=(
                    f"Estimate the probability that, for one {u} chosen at random, A or B (or both) happens. "
                    f"{_FRACTION_SUFFIX}"
                ),
                answer_kind="fraction",
                answer=exact_value(p_union),
                answer_text=pu,
                answer_latex=latex_answer(pu),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"({a_only} + {b_only} + part_a)/{total}",
                working=[f"{a_only} + {both} + {b_only} = {union}.", f"P = {union}/{total} = {pu}."],
                distractors=c_d,
            ),
            dict(
                prompt=(
                    f"Using your answer to (c), predict for how many of the next {future} {u}s A or B (or both) will "
                    "happen. Round to the nearest whole number."
                ),
                answer_kind="number",
                answer=answer_d,
                answer_text=str(answer_d),
                answer_latex=latex_answer(str(answer_d)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{future}*part_c",
                working=[f"{future} × {pu} = {float(projected):.2f}, so about {answer_d}."],
                distractors=d_d,
            ),
        ],
    )
