"""Generates `ucv1_reduction_formulae` questions and the exam chain `exd_x_fy_reduction_formulae`.

``ucv1-reduction-formulae`` (``integration``, answer ``number``): evaluating a specific member of a family of integrals
by applying its reduction formula step by step from the base cases. Every answer is a decimal correct to 3 places.

* easy: the formula is given. I_k = ((k - 1)/k) I_(k-2) for the integral of sin^k x or cos^k x over [0, π/2], or of
  sin^k x over [0, π]; find I_n for n from 3 to 8 from the two base cases.
* medium: the formula is given. K_k = e^a/a - (k/a) K_(k-1) for the integral of x^k e^(ax) over [0, 1]; find K_n for
  n from 2 to 4. The recurrence has a boundary term, so a slip in the sign or in the division by a is visible.
* hard: the formula has to be derived. The integral of tan^n x over [0, π/4] (tan^n = tan^(n-2)(sec² - 1)), of
  (ln x)^n over [1, e], or of (1 - x²)^n over [0, 1].
* exam (``exd_x_fy_reduction_formulae``): K_n = ∫ x^n e^(ax) dx from 0 to b with the recurrence stated;
  (a) K_0, (b) K_1, (c) K_2, (d) K_3, each following through from the one before.

Every value is re-derived by numerical quadrature (composite Simpson) of the integrand, which shares nothing with the
recurrence.
"""

from __future__ import annotations

import math

from ..core import attempts, check, generator, pick_distractors
from ._fy1cd_kit import F, decimal_text, pick_decimal, say


class _Retry(Exception):
    """A draw whose values leave too few distinct wrong answers."""


def _need(ds) -> None:
    if len(ds) < 2:
        raise _Retry


def _retrying(build, rng):
    for _ in attempts(100, "a draw with enough distinct wrong answers"):
        try:
            return build(rng)
        except _Retry:
            continue


def _numeric(fn, lo, hi, panels: int = 600) -> float:
    """Composite Simpson's rule: a numerical value of the integral that shares nothing with the recurrence."""
    lo, hi = float(lo), float(hi)
    h = (hi - lo) / panels
    total = fn(lo) + fn(hi)
    for i in range(1, panels):
        total += fn(lo + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def _verify(value, fn, lo, hi, what: str = "the integral") -> None:
    got = _numeric(fn, lo, hi)
    check(abs(got - float(value)) < 1e-7 * max(1.0, abs(got)), f"{what}: quadrature {got}, recurrence {float(value)}")


def _seq(n: int, bases, rec) -> float:
    """The n-th term of a sequence with the given base values and recurrence ``rec(k, values)``."""
    vals = list(bases)
    for k in range(len(vals), n + 1):
        vals.append(rec(k, vals))
    return vals[n]


def _result(stem, value, ds, working, places: int = 3):
    number, text = decimal_text(value, places)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=number,
        answer_text=text,
        answer_latex=text,
        working=working,
        distractors=ds,
    )


def _stem(*items, places: int = 3):
    return say(*items, f", correct to {places} decimal places.")


# -- easy: I_k = ((k - 1)/k) I_(k-2) -----------------------------------------------------------------------------

_PI = math.pi
# (function name, TeX function, python function, upper limit plain, upper limit TeX, upper limit value, I_0, I_1, I_0 text, I_1 text)
_WALLIS = [
    ("sin", r"\sin", math.sin, "π/2", r"\frac{\pi}{2}", _PI / 2, _PI / 2, 1.0, "π/2", "1"),
    ("cos", r"\cos", math.cos, "π/2", r"\frac{\pi}{2}", _PI / 2, _PI / 2, 1.0, "π/2", "1"),
    ("sin", r"\sin", math.sin, "π", r"\pi", _PI, _PI, 2.0, "π", "2"),
]


def _easy(rng):
    name, tex_name, fn, up, up_tex, up_v, i0, i1, i0_text, i1_text = rng.choice(_WALLIS)
    n = rng.randint(3, 10)
    true = lambda k, v: (k - 1) / k * v[k - 2]  # noqa: E731
    value = _seq(n, (i0, i1), true)
    _verify(value, lambda x: fn(x) ** n, 0, up_v, f"I_{n}")
    cands = [
        (_seq(n, (i0, i1), lambda k, v: k / (k - 1) * v[k - 2]), "redform-ratio-inverted"),
        (_seq(n, (i1, i0), true), "redform-wrong-base"),
        (_seq(n, (i0, i1), lambda k, v: (k - 1) / k * v[k - 1]), "redform-order-off-by-one"),
        (_seq(n, (i0, i1), lambda k, v: v[k - 2] / k), "redform-factor-one-over-k"),
        ((n - 1) / n, "redform-stopped-early"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    integral = F(f"∫ {name}^k x dx from x = 0 to x = {up}", rf"\int_{{0}}^{{{up_tex}}}{tex_name}^{{k}}x\,dx")
    stem = _stem(
        "The integrals I_k = ",
        integral,
        " satisfy I_k = ((k - 1)/k) I_(k-2) for k ≥ 2, with ",
        F(f"I_0 = {i0_text} and I_1 = {i1_text}", rf"I_{{0}}={i0_text}\text{{ and }}I_{{1}}={i1_text}"),
        f". Find I_{n}",
    )
    steps = []
    start = n % 2
    for k in range(start + 2, n + 1, 2):
        steps.append(f"I_{k} = {k - 1}/{k} × I_{k - 2} = {decimal_text(_seq(k, (i0, i1), true), 4)[1]}")
    return _result(
        stem,
        value,
        ds,
        [
            f"I_{start} = {i0_text if start == 0 else i1_text}, so apply the formula {len(steps)} time{'s' if len(steps) != 1 else ''}, stepping k up by 2.",
            "; ".join(steps) + ".",
            f"I_{n} = {decimal_text(value, 3)[1]}.",
        ],
    )


# -- the recurrence K_k = b^k e^(ab)/a - (k/a) K_(k-1) for the integral of x^k e^(ax) from 0 to b ------------------


def _kstep(k, a, b, prev, *, plus=False, n_over_a=True, boundary_over_a=True, power_b=True) -> float:
    boundary = (b**k if power_b else 1) * math.exp(a * b) / (a if boundary_over_a else 1)
    n_term = k * prev / (a if n_over_a else 1)
    return boundary + n_term if plus else boundary - n_term


def _k0(a, b) -> float:
    return (math.exp(a * b) - 1) / a


def _kchain(n, a, b, k0=None, **flaws) -> float:
    return _seq(n, (_k0(a, b) if k0 is None else k0,), lambda k, v: _kstep(k, a, b, v[k - 1], **flaws))


_K_FLAWS = [
    (dict(plus=True), "redform-sign-of-recurrence-term"),
    (dict(n_over_a=False), "redform-n-term-not-divided"),
    (dict(boundary_over_a=False), "redform-boundary-not-divided"),
    (dict(power_b=False), "redform-power-of-b-dropped"),
]


def _e(c: int) -> str:
    """e to the power c in plain text, bracketing a negative power."""
    return f"e^({c})" if c < 0 else f"e^{c}"


def _k_formula(a: int, b: int, v: str = "k") -> F:
    """The recurrence as a sentence fragment, in plain text and TeX."""
    c = a * b
    m = abs(a)
    bound = f"{b}^{v} {_e(c)}" if b != 1 else _e(c)
    bound_tex = rf"{b}^{{{v}}}e^{{{c}}}" if b != 1 else f"e^{{{c}}}"
    if a > 0:
        return F(
            f"K_{v} = {bound}/{m} - ({v}/{m}) K_({v}-1)",
            rf"K_{{{v}}}=\frac{{{bound_tex}}}{{{m}}}-\frac{{{v}}}{{{m}}}K_{{{v}-1}}",
        )
    return F(
        f"K_{v} = -{bound}/{m} + ({v}/{m}) K_({v}-1)",
        rf"K_{{{v}}}=-\frac{{{bound_tex}}}{{{m}}}+\frac{{{v}}}{{{m}}}K_{{{v}-1}}",
    )


def _k_integral(a: int, b: int, exponent: str = "k") -> F:
    return F(
        f"∫ x^{exponent} e^({a}x) dx from x = 0 to x = {b}",
        rf"\int_{{0}}^{{{b}}}x^{{{exponent}}}e^{{{a}x}}\,dx",
    )


def _medium(rng):
    a = rng.choice([-3, -2, -1, 2, 3, 4, 5])
    n = rng.randint(2, 5)
    value = _kchain(n, a, 1)
    _verify(value, lambda x: x**n * math.exp(a * x), 0, 1, f"K_{n}")
    cands = [(_kchain(n, a, 1, **flaws), tag) for flaws, tag in _K_FLAWS if "power_b" not in flaws]
    cands.append((_kchain(n, a, 1, k0=math.exp(a) / a), "redform-lower-limit-dropped"))
    cands.append((_kchain(n, a, 1, k0=1.0), "redform-stopped-early"))
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    stem = _stem(
        "The integrals K_k = ",
        _k_integral(a, 1),
        " satisfy ",
        _k_formula(a, 1),
        " for k ≥ 1. Find ",
        F(f"K_{n}", f"K_{{{n}}}"),
    )
    steps = [f"K_0 = ({_e(a)} - 1)/{a} = {decimal_text(_k0(a, 1), 4)[1]}"]
    for k in range(1, n + 1):
        steps.append(f"K_{k} = {decimal_text(_kchain(k, a, 1), 4)[1]}")
    return _result(
        stem,
        value,
        ds,
        [
            f"The base case is K_0, the integral of e^({a}x) from 0 to 1.",
            "; ".join(steps) + ".",
            f"K_{n} = {decimal_text(value, 3)[1]}.",
        ],
    )


# -- hard: the formula has to be derived ------------------------------------------------------------------------


def _hard_tan(rng):
    n = rng.randint(3, 12)
    t0, t1 = _PI / 4, math.log(2) / 2
    true = lambda k, v: 1 / (k - 1) - v[k - 2]  # noqa: E731
    value = _seq(n, (t0, t1), true)
    _verify(value, lambda x: math.tan(x) ** n, 0, _PI / 4, f"T_{n}")
    cands = [
        (_seq(n, (t0, t1), lambda k, v: 1 / (k - 1) + v[k - 2]), "redform-sign-of-recurrence-term"),
        (_seq(n, (t0, t1), lambda k, v: 1 / k - v[k - 2]), "redform-denominator-off-by-one"),
        (_seq(n, (t0, t1), lambda k, v: 1 / (k - 1) - v[k - 1]), "redform-order-off-by-one"),
        (_seq(n, (t1, t0), true), "redform-wrong-base"),
        (_seq(n, (t0, t1), lambda k, v: 1 / (k - 1)), "redform-term-dropped"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    integral = F(f"∫ tan^{n} x dx from x = 0 to x = π/4", rf"\int_{{0}}^{{\pi/4}}\tan^{{{n}}}x\,dx")
    stem = _stem(
        "Writing tan^n x = tan^(n-2) x (sec² x - 1) gives a reduction formula for T_n = ∫ tan^n x dx. Use it to evaluate ",
        integral,
    )
    return _result(
        stem,
        value,
        ds,
        [
            "Integrate tan^(n-2) x sec² x directly and the other term is T_(n-2): T_n = 1/(n - 1) - T_(n-2) over [0, π/4].",
            f"The base cases are T_0 = π/4 and T_1 = (ln 2)/2, and T_{n} = {decimal_text(value, 4)[1]} after stepping up by 2.",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


def _hard_log(rng):
    n = rng.randint(3, 7)
    e = math.e
    true = lambda k, v: e - k * v[k - 1]  # noqa: E731
    value = _seq(n, (e - 1,), true)
    _verify(value, lambda x: math.log(x) ** n, 1, e, f"L_{n}")
    cands = [
        (_seq(n, (e - 1,), lambda k, v: e + k * v[k - 1]), "redform-sign-of-recurrence-term"),
        (_seq(n, (e - 1,), lambda k, v: 1 - k * v[k - 1]), "redform-boundary-value-wrong"),
        (_seq(n, (e - 1,), lambda k, v: -k * v[k - 1]), "redform-boundary-dropped"),
        (_seq(n, (e - 1,), lambda k, v: e - v[k - 1]), "redform-coefficient-dropped"),
        (_seq(n, (e,), true), "redform-lower-limit-dropped"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    integral = F(f"∫ (ln x)^{n} dx from x = 1 to x = e", rf"\int_{{1}}^{{e}}(\ln x)^{{{n}}}\,dx")
    stem = _stem(
        "Integration by parts gives a reduction formula for L_n = ∫ (ln x)^n dx. Use it to evaluate ",
        integral,
    )
    return _result(
        stem,
        value,
        ds,
        [
            "Take u = (ln x)^n and dv = dx. Then L_n = [x (ln x)^n] - n L_(n-1), and the boundary term over [1, e] is e.",
            f"So L_n = e - n L_(n-1), with L_0 = e - 1, and L_{n} = {decimal_text(value, 4)[1]} after {n} steps.",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


def _hard_one_minus_square(rng):
    n = rng.randint(3, 12)
    true = lambda k, v: 2 * k / (2 * k + 1) * v[k - 1]  # noqa: E731
    value = _seq(n, (1.0,), true)
    _verify(value, lambda x: (1 - x * x) ** n, 0, 1, f"W_{n}")
    cands = [
        (_seq(n, (1.0,), lambda k, v: (2 * k + 1) / (2 * k) * v[k - 1]), "redform-ratio-inverted"),
        (_seq(n, (1.0,), lambda k, v: (2 * k - 1) / (2 * k) * v[k - 1]), "redform-ratio-index-shift"),
        (_seq(n - 1, (1.0,), true), "redform-order-off-by-one"),
        (_seq(n + 1, (1.0,), true), "redform-order-off-by-one"),
        (_seq(n, (1.0,), lambda k, v: k / (k + 1) * v[k - 1]), "redform-ratio-simplified-wrongly"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    integral = F(f"∫ (1 - x²)^{n} dx from x = 0 to x = 1", rf"\int_{{0}}^{{1}}(1-x^{{2}})^{{{n}}}\,dx")
    stem = _stem(
        "Integration by parts gives a reduction formula for W_n = ∫ (1 - x²)^n dx. Use it to evaluate ",
        integral,
    )
    return _result(
        stem,
        value,
        ds,
        [
            "Take u = (1 - x²)^n and dv = dx. The boundary term vanishes, and W_n = 2n/(2n + 1) × W_(n-1) over [0, 1].",
            f"With W_0 = 1, applying this {n} time{'s' if n != 1 else ''} gives W_{n} = {decimal_text(value, 4)[1]}.",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


@generator(
    "ucv1_reduction_formulae",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_reduction_formulae(rng, tier="easy"):
    """Evaluate a member of a family of integrals from its reduction formula; the tiers are in the module docstring."""
    if tier == "easy":
        return _retrying(_easy, rng)
    if tier == "medium":
        return _retrying(_medium, rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retrying(rng.choice([_hard_tan, _hard_log, _hard_one_minus_square]), rng)


# -- exam -------------------------------------------------------------------------------------------------------


_EXAM_SHAPES = [(a, b) for a in (-5, -4, -3, -2, 2, 3, 4, 5) for b in (1, 2, 3) if abs(a * b) <= 8]


def _exam(rng):
    a, b = rng.choice(_EXAM_SHAPES)
    m = rng.choice([0, 1])  # the first index asked for; with m = 1 the stem supplies K_0
    c = a * b
    idx = [m + j for j in range(4)]
    ks = [_kchain(i, a, b) for i in range(m + 4)]
    for i in range(m + 4):
        _verify(ks[i], lambda x, i=i: x**i * math.exp(a * x), 0, b, f"K_{i}")
    places = [4, 4, 4, 3]
    texts = [decimal_text(ks[i], places[j]) for j, i in enumerate(idx)]

    if m == 0:
        cands0 = [
            (math.exp(c) - 1, "redform-boundary-not-divided"),
            (math.exp(c) / a, "redform-lower-limit-dropped"),
            ((math.exp(c) + 1) / a, "redform-limit-sign"),
            ((math.exp(a) - 1) / a, "redform-upper-limit-ignored"),
        ]
    else:
        cands0 = [(_kstep(1, a, b, ks[0], **flaws), tag) for flaws, tag in _K_FLAWS]
    ds = [pick_decimal(ks[idx[0]], cands0, 4)]
    for j in range(1, 4):
        prev = ks[idx[j] - 1]
        cands = [(_kstep(idx[j], a, b, prev, **flaws), tag) for flaws, tag in _K_FLAWS]
        ds.append(pick_decimal(ks[idx[j]], cands, places[j]))

    # What a learner's own earlier (wrong) answer leads to is not offered as a wrong option, as it would deny the
    # follow-through marks.
    for j in range(1, 4):
        earlier = [texts[j - 1][0]] + [float(d.text) for d in ds[j - 1]]
        own = {decimal_text(_kstep(idx[j], a, b, v), places[j])[1] for v in earlier} - {texts[j][1]}
        ds[j] = [d for d in ds[j] if d.text not in own]
    for d in ds:
        _need(d)

    fts = [None] + [f"{b}**{idx[j]}*exp({c})/({a}) - {idx[j]}*part_{'abc'[j - 1]}/({a})" for j in range(1, 4)]
    k0 = f"({_e(c)} - 1)/{a}" if a > 0 else f"(1 - {_e(c)})/{-a}"
    k0_tex = rf"\frac{{e^{{{c}}}-1}}{{{a}}}" if a > 0 else rf"\frac{{1-e^{{{c}}}}}{{{-a}}}"
    pieces = [
        "Let ",
        F("K_n = ", "K_{n}="),
        _k_integral(a, b, "n"),
        ". Integration by parts gives ",
        _k_formula(a, b, "n"),
        " for n ≥ 1.",
    ]
    if m == 1:
        pieces += [" Also ", F(f"K_0 = {k0}", rf"K_{{0}}={k0_tex}"), "."]
    stem = say(*pieces)
    prompts = []
    for j in range(4):
        what = "Find" if (j == 0 and m == 0) else "Use the formula to find"
        prompts.append(
            (
                f"{what} K_{idx[j]}, correct to {places[j]} decimal places.",
                rf"\text{{{what} }}K_{{{idx[j]}}}\text{{, correct to {places[j]} decimal places.}}",
            )
        )
    bpow = lambda i: "" if b == 1 else f"{b}^{i} "  # noqa: E731
    workings = []
    for j in range(4):
        i = idx[j]
        if i == 0:
            workings.append([f"K_0 is the integral of e^({a}x) from 0 to {b}, which is {k0} = {texts[j][1]}."])
        else:
            workings.append([f"K_{i} = {bpow(i)}{_e(c)}/{a} - ({i}/{a}) K_{i - 1} = {texts[j][1]}."])
    parts = []
    for j in range(4):
        part = dict(
            prompt=prompts[j][0],
            prompt_latex=prompts[j][1],
            answer_kind="number",
            answer=texts[j][0],
            answer_text=texts[j][1],
            answer_latex=texts[j][1],
            marks=2,
            working=workings[j],
            distractors=ds[j],
        )
        if fts[j]:
            part["follow_through"] = fts[j]
        parts.append(part)
    return dict(text=stem[0], latex=stem[1], parts=parts)


@generator("exd_x_fy_reduction_formulae", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_reduction_formulae(rng, tier="exam"):
    """Exam chain: four consecutive K_n (from K_0 or K_1) for the integral of x^n e^(ax), each from the one before."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retrying(_exam, rng)
