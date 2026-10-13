"""Exam tier for "estimate counts and check reasonableness": estimate, calculate exactly, compare.

The practice tiers ask for one rounded product. Here a boxes-of-items story asks three connected things, as a
national-assessment paper does: (a) estimate the total by rounding each factor to a stated place, (b) work out the
exact total, (c) say by how much the estimate is out. Part (c) uses both earlier answers, so it carries
`follow_through`: a learner whose estimate or exact total is wrong still earns (c) for subtracting their own two
values correctly.

Built from the two factors alone. Every key is recomputed by a second route (rounding by digit inspection, the
exact product as a sum of partial products), and each distractor is a slip a learner really makes.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, round_half_up

_TOWNS = ["Geelong", "Townsville", "Bunbury", "Dubbo", "Launceston", "Mildura", "Whyalla", "Toowoomba", "Bendigo"]
_BUYERS = [
    "A primary school in {town}",
    "A scout group in {town}",
    "A surf life saving club in {town}",
    "A community library in {town}",
    "A netball association in {town}",
]
_ITEMS = ["pencils", "stickers", "bookmarks", "paper clips", "rubber bands", "raffle tickets", "badges"]
_STEMS = [
    ("{buyer} orders {a} boxes of {item}. Each box holds {b} {item}.", "boxes"),
    ("{buyer} receives a delivery of {a} cartons. Each carton contains {b} {item}.", "cartons"),
    ("{buyer} buys {a} packs of {item}, with {b} {item} in every pack.", "packs"),
]


def _round_to(n: int, place: int) -> int:
    """Round half up to a multiple of `place`, by looking at the digit below it (a different route from `round_half_up`)."""
    kept, rest = divmod(n, place)
    return (kept + (1 if rest * 2 >= place else 0)) * place


def _floor_to(n: int, place: int) -> int:
    return (n // place) * place


def _column_gap(x: int, y: int) -> int:
    """Subtract column by column, always taking the smaller digit from the larger (no borrowing)."""
    hi, lo = max(x, y), min(x, y)
    digits = []
    for dh, dl in zip(str(hi).zfill(len(str(hi)))[::-1], str(lo).zfill(len(str(hi)))[::-1]):
        digits.append(abs(int(dh) - int(dl)))
    return int("".join(str(d) for d in reversed(digits)))


@generator("exd_x_pr7_estimate_chain", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_estimate_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Estimate a product from stated rounding, find it exactly, then measure the estimate's error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "factors whose estimate differs from the exact product"):
        a = rng.randint(13, 98)
        b = rng.randint(106, 996)
        place_b = rng.choice([10, 100])
        if a % 10 == 0 or b % place_b == 0:
            continue
        ra, rb = _round_to(a, 10), _round_to(b, place_b)
        if ra == a or rb == b or ra * rb == a * b:
            continue
        break

    estimate, exact = ra * rb, a * b
    # Second routes: Fraction half-up rounding, and the exact product as a sum of partial products.
    check(ra == 10 * int(round_half_up(Fraction(a, 10))) and rb == place_b * int(round_half_up(Fraction(b, place_b))))
    check(exact == (a // 10) * 10 * b + (a % 10) * b, "exact product disagrees with partial products")
    gap = abs(exact - estimate)

    town = rng.choice(_TOWNS)
    item = rng.choice(_ITEMS)
    buyer = rng.choice(_BUYERS).format(town=town)
    template, unit = rng.choice(_STEMS)
    place_word = {10: "ten", 100: "hundred"}[place_b]
    stem = (
        f"{template.format(buyer=buyer, a=a, b=b, item=item)} To estimate, round the number of {unit} to the "
        f"nearest ten and the number of {item} in each one to the nearest {place_word}."
    )

    est_candidates = [
        Distractor(
            str(_floor_to(a, 10) * _floor_to(b, place_b)),
            "rounded-down-not-to-nearest",
            _floor_to(a, 10) * _floor_to(b, place_b),
        ),
        Distractor(str(exact), "uses-unrounded-operands", exact),
        Distractor(str(ra * b), "rounded-only-one-factor", ra * b),
        Distractor(str(ra * _round_to(b, 10)), "one-place-applied-to-both", ra * _round_to(b, 10)),
    ]
    shifted = ((a // 10) + (a % 10)) * b  # tens partial product written without its zero
    exact_candidates = [
        Distractor(str(estimate), "gave-estimate-for-exact", estimate),
        Distractor(str(shifted), "partial-product-not-shifted", shifted),
        Distractor(str(a + b), "added-instead-of-multiplied", a + b),
    ]
    col = _column_gap(exact, estimate)
    gap_candidates = [
        Distractor(str(exact + estimate), "added-instead-of-subtracted", exact + estimate),
        Distractor(str(col), "subtracted-smaller-from-larger-each-column", col),
        Distractor(str(exact), "stopped-before-final-step", exact),
        Distractor(str(estimate), "stopped-before-final-step", estimate),
    ]
    d_est = pick_distractors(str(estimate), est_candidates, 3)
    d_exact = pick_distractors(str(exact), exact_candidates, 3)
    d_gap = pick_distractors(str(gap), gap_candidates, 3)
    check(len(d_est) >= 2 and len(d_exact) >= 2 and len(d_gap) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Estimate the total number of {item}.",
                answer_kind="number",
                answer=estimate,
                answer_text=str(estimate),
                answer_latex=latex_answer(str(estimate)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{ra} × {rb} = {estimate}."],
                distractors=d_est,
            ),
            dict(
                prompt=f"Work out the exact total number of {item}.",
                answer_kind="number",
                answer=exact,
                answer_text=str(exact),
                answer_latex=latex_answer(str(exact)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{a} × {b} = {exact}."],
                distractors=d_exact,
            ),
            dict(
                prompt="By how much does your estimate differ from the exact total?",
                answer_kind="number",
                answer=gap,
                answer_text=str(gap),
                answer_latex=latex_answer(str(gap)),
                marks=1,
                sensible_range=(0, None),
                follow_through="Abs(part_b - part_a)",
                working=[f"The difference between {exact} and {estimate} is {gap}."],
                distractors=d_gap,
            ),
        ],
    )
