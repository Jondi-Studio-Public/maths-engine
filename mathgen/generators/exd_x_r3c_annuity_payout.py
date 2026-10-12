"""Exam source for `senior-use-a-recurrence-relation-to-model-an-annuity`: balance, term, last balance, final payment.

The stem gives an annuity (opening amount, annual rate compounded monthly or quarterly, a fixed withdrawal at the end
of each period after interest is added). Withdrawals continue while the full amount can be paid; what is left is
paid out with its interest as a smaller final payment.

(a) the balance after the first withdrawal, to the cent: A1 = P(1 + i) - d;
(b) the number of full withdrawals N (iterating the recurrence);
(c) the balance after the last full withdrawal, from (b): P(1 + i)^N - d((1 + i)^N - 1)/i;
(d) the final payment, from (c): the balance with one more period's interest.

Parts (c) and (d) carry `follow_through`. The recurrence is run exactly (Fractions); the closed form of (c) is
checked against it.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, round_half_up

from .exd_x_r3c_parts import number_part
from ._exd_r3_distinct import distinct_part_answers

_PERIODS = (("monthly", 12, "month"), ("quarterly", 4, "quarter"))


def _clear(value):
    """True when `value` (dollars) is at least 0.02 of a cent from the point where rounding to the cent flips."""
    return abs((value * 100) % 1 - Fraction(1, 2)) > Fraction(2, 100)


def _run(principal, i, withdrawal):
    """(number of full withdrawals, balance after each) by running the recurrence A(n+1) = A(n)(1 + i) - d."""
    balances = [Fraction(principal)]
    while balances[-1] * (1 + i) >= withdrawal:
        balances.append(balances[-1] * (1 + i) - withdrawal)
        if len(balances) > 400:
            break
    return len(balances) - 1, balances


_STEMS = (
    "{amount} is invested in an annuity that earns {pct}% per annum, compounded {word}. At the end of each {unit}, "
    "after the interest is added, {w} is withdrawn. A withdrawal is made only while the full {w} can be paid. The "
    "money then left in the annuity, with the next {unit}'s interest, is paid out as a final smaller payment.",
    "A retiree puts {amount} into an account paying {pct}% per annum, compounded {word}. Each {unit}, once the "
    "interest has been credited, {w} is paid to the retiree, for as long as the full {w} is available. What remains "
    "after that, with the next {unit}'s interest, is paid as one last smaller payment.",
    "A scholarship fund starts with {amount} and earns {pct}% per annum, compounded {word}. At the end of every "
    "{unit}, after interest is credited, a grant of {w} is paid out while the fund can cover the full grant. The "
    "remaining money, with the next {unit}'s interest, is then paid as a final smaller grant.",
)


def _money(value):
    return f"${value:,.2f}" if value != int(value) else f"${int(value):,}"


@generator("exd_x_r3c_annuity_payout", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_annuity_payout(rng, tier="exam"):
    """Annuity: first balance, number of full withdrawals, balance after the last one, the final payment."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "an annuity with 8 to 60 full withdrawals and cents clear of a rounding boundary"):
        word, per_year, unit = rng.choice(_PERIODS)
        principal = rng.choice(range(40_000, 300_001, 5_000))
        tenths = rng.randint(30, 70)  # annual rate 3.0% to 7.0%
        rate = Fraction(tenths, 1000)
        i = rate / per_year
        withdrawal = rng.choice(range(500, 6_001, 50))
        n, balances = _run(principal, i, withdrawal)
        if not 8 <= n <= 60:
            continue
        a1, a_n = balances[1], balances[-1]
        last = a_n * (1 + i)
        if last <= 0 or not all(_clear(v) for v in (a1, a_n, last)):
            continue
        break
    # Independent route: the closed form for the balance after n withdrawals.
    growth = (1 + i) ** n
    check(principal * growth - withdrawal * (growth - 1) / i == a_n, "closed form disagrees with the recurrence")
    check(a_n * (1 + i) < withdrawal <= balances[-2] * (1 + i), "N is not the number of full withdrawals")

    pct = f"{tenths / 10:.1f}"
    stem = rng.choice(_STEMS).format(amount=_money(principal), pct=pct, word=word, unit=unit, w=_money(withdrawal))
    first, last_bal, final = round_half_up(a1, 2), round_half_up(a_n, 2), round_half_up(last, 2)
    part_a = number_part(
        "Find the balance of the annuity immediately after the first withdrawal, to the nearest cent.",
        a1,
        dp=2,
        sensible_range=(0, None),
        working=[f"A1 = {principal} × (1 + {pct}% ÷ {per_year}) - {withdrawal} = {first}."],
        wrongs=[
            (principal - withdrawal, "interest-not-credited"),
            (principal * (1 + i) + withdrawal, "withdrawal-added-not-subtracted"),
            (principal * (1 + rate) - withdrawal, "annual-rate-as-periodic"),
        ],
    )
    part_b = number_part(
        "Find the number of full withdrawals that the annuity pays.",
        n,
        marks=2,
        sensible_range=(0, 400),
        working=[f"Iterating A(n+1) = A(n) × (1 + i) - {withdrawal}, the annuity pays {n} full withdrawals."],
        wrongs=[
            (n + 1, "off-by-one-over"),
            (n - 1, "off-by-one-under"),
            (principal // withdrawal, "ignores-compounding-in-count"),
        ],
    )
    part_c = number_part(
        "Using your answer to (b), find the balance of the annuity immediately after its last full withdrawal, to the nearest cent.",
        a_n,
        dp=2,
        marks=2,
        sensible_range=(0, None),
        working=[f"A({n}) = {principal} × (1 + i)^{n} - {withdrawal} × ((1 + i)^{n} - 1) ÷ i = {last_bal}."],
        wrongs=[
            (balances[-2], "balance-before-last-withdrawal"),
            (a_n + withdrawal, "final-withdrawal-not-taken"),
        ],
        follow_through=(
            f"{principal}*(1 + {i.numerator}/{i.denominator})**part_b - "
            f"{withdrawal}*((1 + {i.numerator}/{i.denominator})**part_b - 1)/({i.numerator}/{i.denominator})"
        ),
    )
    part_d = number_part(
        f"Using your answer to (c), find the final payment made at the end of the next {unit}, to the nearest cent.",
        last,
        dp=2,
        sensible_range=(0, None),
        working=[f"{last_bal} × (1 + {pct}% ÷ {per_year}) = {final}."],
        wrongs=[
            (a_n, "interest-not-credited"),
            (a_n + withdrawal, "final-withdrawal-not-taken"),
            (a_n * (1 + rate), "annual-rate-as-periodic"),
        ],
        follow_through=f"part_c*(1 + {i.numerator}/{i.denominator})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
