"""Generates `exam_parts_compound_interest`: a four-part exam chain on the effect of the compounding frequency
or the interest rate on a savings balance (senior-use-technology-to-investigate-the-effect-of-the).

One scenario: a principal P is invested at r% p.a., compounded n times a year, for t years.

(a) the balance after t years, to the nearest cent (2 marks);
(b) hence the interest earned (follow-through ``part_a - P``);
(c) the balance when the SAME amount is compounded annually instead, or earns a lower rate, still compounded n
    times a year (independent);
(d) using (a) and (c), how much more the first account earns (follow-through ``part_a - part_c``).

All money is exact ``Fraction`` arithmetic, rounded half up once, at display, to exactly 2 decimal places. A draw
whose exact value sits within 1e-9 of a half-cent boundary is redrawn.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.slots import NAMES

_TOL = stated_precision_tolerance(2)
_FREQ_WORD = {2: "six-monthly", 4: "quarterly", 12: "monthly"}
_FREQ_LABEL = {2: "half-yearly", 4: "quarterly", 12: "monthly"}
_PRODUCTS = ("term deposit", "savings account", "high-interest savings account", "fixed-term investment")
_STEM = (
    "{name} puts ${p} into {art} {product} that pays {r}% p.a. interest, compounded {freq}. "
    "{name} leaves it untouched for {t} years.",
    "{Art} {product} opened by {name} with ${p} earns {r}% p.a., compounded {freq}. No deposits or withdrawals are "
    "made for {t} years.",
    "{name} invests ${p} in {art} {product} at {r}% p.a. with interest compounded {freq}, and does not touch the "
    "money for {t} years.",
)


def _grow(p: Fraction, rate: Fraction, n: int, t: int) -> Fraction:
    return p * (1 + rate / n) ** (n * t)


def _grow_loop(p: Fraction, rate: Fraction, n: int, t: int) -> Fraction:
    """Independent route: period by period."""
    b = p
    for _ in range(n * t):
        b += b * rate / n
    return b


def _cents(v: Fraction) -> Fraction:
    return round_half_up(v, 2)


def _fmt(v: Fraction) -> str:
    return stated_precision_string(_cents(v), 2)


def _safe(v: Fraction) -> bool:
    """False when 100v sits within 1e-9 of a half-cent boundary."""
    x = v * 100
    frac = x - (x.numerator // x.denominator)
    return abs(frac - Fraction(1, 2)) > Fraction(1, 10**9)


def _pct(r: Fraction) -> str:
    v = r * 100
    return str(int(v)) if v.denominator == 1 else str(float(v))


def _money(v: Fraction) -> str:
    return f"{int(v):,}"


def _part(prompt, v, marks, working, candidates, follow_through=None, others=(), sensible_range=None):
    """One number part; every candidate is checked wrong against the answer and the other displayed values."""
    answer = _cents(v)
    cand_vals = [_cents(c) for c in candidates]
    check_distinct(answer, *cand_vals, msg=f"distractor equals the answer in: {prompt}")
    text = stated_precision_string(answer, 2)
    shown = [stated_precision_string(c, 2) for c in cand_vals]
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(answer),
        answer_text=text,
        answer_latex=latex_answer(answer),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, shown, n=3, tolerance=_TOL),
    )
    if follow_through:
        out["follow_through"] = follow_through
    if sensible_range:
        out["sensible_range"] = sensible_range
    return out


@generator("exam_parts_compound_interest", level=13, presets={"exam": {}})
def exam_parts_compound_interest(rng):
    for _ in range(200):
        p = Fraction(rng.randint(4, 80) * 500)
        n = rng.choice((2, 4, 12))
        t = rng.randint(2, 6)
        r1000 = rng.randint(30, 90) * 1  # tenths of a percent: 3.0% .. 9.0%
        rate = Fraction(r1000, 1000)
        mode = rng.choice(("freq", "rate"))
        if mode == "rate":
            r2_tenths = r1000 - rng.choice((5, 10, 15, 20, 25))
            rate2 = Fraction(r2_tenths, 1000)
            n2 = n
        else:
            rate2 = rate
            n2 = 1
        fv_a = _grow(p, rate, n, t)
        fv_c = _grow(p, rate2, n2, t)
        if not (_safe(fv_a) and _safe(fv_c)):
            continue
        # distractor values
        wrong_rate = p * (1 + rate) ** (n * t)  # rate not divided by n
        wrong_periods = p * (1 + rate / n) ** t  # periods not multiplied by n
        simple = p * (1 + rate * t)
        interest_a = fv_a - p
        if mode == "freq":
            c_cands = [fv_a, p * (1 + rate * t), fv_c - p, p * (1 + rate / 2) ** (2 * t)]
        else:
            c_cands = [fv_a, p * (1 + rate2 / n) ** t, p * (1 + rate2 * t), fv_c - p]
        d_ans = fv_a - fv_c
        if _cents(d_ans) < Fraction(1, 2) or any(
            abs(_cents(x) - _cents(y)) < Fraction(2, 100)
            for x, y in [(fv_a, wrong_rate), (fv_a, wrong_periods), (fv_a, simple), (fv_a, interest_a)]
        ):
            continue
        break
    else:  # pragma: no cover
        raise RuntimeError("no safe draw")

    check_equivalent(fv_a, _grow_loop(p, rate, n, t), "balance (a): closed form and period loop disagree")
    check_equivalent(fv_c, _grow_loop(p, rate2, n2, t), "balance (c): closed form and period loop disagree")
    check(d_ans > 0, "the first account must earn more")

    name = rng.choice(NAMES)
    product = rng.choice(_PRODUCTS)
    stem = phrase(
        rng,
        _STEM,
        name=name,
        product=product,
        p=_money(p),
        r=_pct(rate),
        art="an" if product[0] in "aeiou" else "a",
        Art="An" if product[0] in "aeiou" else "A",
        freq=_FREQ_LABEL[n] if rng.random() < 0.5 else _FREQ_WORD[n],
        t=t,
    )
    if mode == "freq":
        c_prompt = (
            f"Suppose the same ${_money(p)} had earned {_pct(rate)}% p.a. compounded annually for the same "
            f"{t} years. Find the balance, to the nearest cent."
        )
        c_work = [f"n = 1: A = {_money(p)} × (1 + {float(rate)})^{t} = ${_fmt(fv_c)}."]
    else:
        c_prompt = (
            f"Suppose the same ${_money(p)} had earned {_pct(rate2)}% p.a. instead, still compounded "
            f"{_FREQ_WORD[n] if _FREQ_WORD[n] in stem else _FREQ_LABEL[n]}, for the same {t} years. Find the "
            f"balance, to the nearest cent."
        )
        c_work = [f"Rate {_pct(rate2)}%: A = {_money(p)} × (1 + {float(rate2 / n)})^{n * t} = ${_fmt(fv_c)}."]

    ans_a, ans_c = _cents(fv_a), _cents(fv_c)
    parts = [
        _part(
            "Find the balance after the time stated, to the nearest cent.",
            fv_a,
            2,
            [
                f"Rate per period = {_pct(rate)}% ÷ {n} = {float(rate / n * 100)}%; periods = {n} × {t} = {n * t}.",
                f"A = {_money(p)} × (1 + {float(rate / n)})^{n * t} = ${_fmt(fv_a)}.",
            ],
            [wrong_rate, wrong_periods, simple, interest_a],
            sensible_range=(int(p), None),  # a balance that grew
        ),
        _part(
            "Hence find the interest earned, to the nearest cent.",
            fv_a - p,
            1,
            [f"Interest = balance - principal = ${_fmt(ans_a)} - ${_money(p)} = ${_fmt(ans_a - p)}."],
            [fv_a, p * rate * t, fv_a + p],
            follow_through=f"part_a - {int(p)}",
        ),
        _part(c_prompt, fv_c, 1, c_work, c_cands, sensible_range=(int(p), None)),
        _part(
            "Using your answers to (a) and (c), find how much more interest the first account earns, to the "
            "nearest cent.",
            ans_a - ans_c,
            1,
            [f"${_fmt(ans_a)} - ${_fmt(ans_c)} = ${_fmt(ans_a - ans_c)}."],
            [ans_c - ans_a, fv_a - p, fv_c - p, (ans_a - ans_c) * t],
            follow_through="part_a - part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
