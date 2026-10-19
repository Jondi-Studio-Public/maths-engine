"""Independent oracle for the `p12b_loan_term` questions: it re-derives each answer from the rendered question
text.

Imports nothing from `mathgen.generators.p12b_loan_term_build`: not the
formula helpers, not the templates, not the construction ranges. Parses
`P`, `r` and the repayment amount(s) straight out of the RENDERED
`question.text` and re-derives the correct answer from scratch via the
plain flat-interest identity `M * 12 * n = P * (1 + r*n/100)`, solved for
whichever quantity the tier asks for. Every tier's answer is re-derived by
a route independent of however the generator's own construction built the
scenario (construct-to-target may work backward from a chosen answer; this
file always works FORWARD from the stated numbers alone).

Run across >= 1000 seeds total, spread over all four tiers.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC = "p12b_loan_term_from_repayment"
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000


def _compile(template: str) -> re.Pattern:
    """Turn one of the generator's own `.format()`-style templates into a
    regex with a named group per `{placeholder}`, escaping every literal
    character around it. Re-authored here, independently, purely to parse
    the rendered text -- this file never imports or calls the generator."""
    parts = re.split(r"(\{\w+\})", template)
    pieces = []
    seen: set[str] = set()
    for part in parts:
        m = re.match(r"^\{(\w+)\}$", part)
        if m:
            name = m.group(1)
            if name in seen:
                pieces.append(rf"(?P={name})")
            else:
                seen.add(name)
                pieces.append(rf"(?P<{name}>\d+)")
        else:
            pieces.append(re.escape(part))
    return re.compile("^" + "".join(pieces) + "$")


_YEARS_PATTERNS = [
    _compile(t)
    for t in [
        "A ${P} flat-interest loan charges {r}% per year in interest. The borrower "
        "repays ${M} per month. How many years will it take to repay the loan in full?",
        "Sam takes out a ${P} loan at a flat interest rate of {r}% p.a. Repayments "
        "are ${M} each month. How many years does it take to pay off the loan?",
        "A flat-interest loan of ${P} is charged at {r}% per annum. Monthly "
        "repayments of ${M} are made until the loan is cleared. Find the number of "
        "years required to repay the loan.",
    ]
]

_MONTHS_PATTERNS = [
    _compile(t)
    for t in [
        "A ${P} flat-interest loan charges {r}% per year in interest. The borrower "
        "repays ${M} per month. How many MONTHS will it take to repay the loan in full?",
        "Priya borrows ${P} at a flat interest rate of {r}% p.a. She repays ${M} "
        "each month. How many months does it take her to clear the loan?",
        "A flat-interest loan of ${P} is charged at {r}% per annum, repaid in "
        "monthly instalments of ${M}. Find the exact number of months needed to "
        "repay the loan in full.",
    ]
]

_DIFF_PATTERNS = [
    _compile(t)
    for t in [
        "A ${P} flat-interest loan charges {r}% per year in interest. Repaying "
        "${M1} per month clears the loan in one plan; repaying ${M2} per month "
        "clears it in another. How many fewer years does the higher repayment "
        "take, compared with the lower one?",
        "Two repayment plans exist for the same ${P} loan at a flat rate of {r}% "
        "p.a.: Plan A repays ${M1} per month, Plan B repays ${M2} per month. Find "
        "the difference, in years, between the two plans' repayment terms.",
        "Deng is comparing two ways to repay a ${P} flat-interest loan charged at "
        "{r}% per annum: ${M1} per month, or ${M2} per month. By how many years is "
        "the loan cleared sooner under the higher monthly repayment?",
    ]
]

_MAX_RATE_PATTERNS = [
    _compile(t)
    for t in [
        "A borrower can afford ${M} per month and wants to clear a ${P} loan within "
        "{n} years. What is the MAXIMUM flat annual interest rate (%) the loan "
        "could charge for this to be possible?",
        "A ${P} loan is to be repaid within {n} years at ${M} per month under a "
        "flat-interest scheme. Find the highest flat annual rate (%) for which the "
        "loan is still fully repaid within {n} years.",
        "Wei can repay ${M} per month towards a ${P} loan and needs it cleared "
        "within {n} years. Assuming flat interest, what is the maximum annual "
        "interest rate (%) the loan can carry?",
    ]
]


def _parse(text: str, patterns: list[re.Pattern]) -> dict:
    plain = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # the stem writes $20,000; the templates here have bare digits
    for pattern in patterns:
        m = pattern.match(plain)
        if m:
            return {k: Fraction(v) for k, v in m.groupdict().items()}
    raise AssertionError(f"unparseable question against all {len(patterns)} known templates: {text!r}")


def _n_years(P: Fraction, r: Fraction, M: Fraction) -> Fraction:
    """Forward solve of `M * 12 * n = P * (1 + r*n/100)` for `n`, in years."""
    denom = 12 * M - P * r / 100
    assert denom != 0
    return P / denom


def _r_percent(P: Fraction, n: Fraction, M: Fraction) -> Fraction:
    """Forward solve of the same identity for `r`."""
    assert P != 0 and n != 0
    return 100 * (12 * M * n - P) / (P * n)


def test_oracle_years_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", mode="years")
        fields = _parse(q.text, _YEARS_PATTERNS)
        n = _n_years(fields["P"], fields["r"], fields["M"])
        assert n.denominator == 1, (seed, q.text, n)
        assert str(int(n)) == q.answer_text, (seed, q.text, q.answer_text, n)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_months_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", mode="months")
        fields = _parse(q.text, _MONTHS_PATTERNS)
        m = _n_years(fields["P"], fields["r"], fields["M"]) * 12
        assert m.denominator == 1, (seed, q.text, m)
        assert str(int(m)) == q.answer_text, (seed, q.text, q.answer_text, m)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_term_difference_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", mode="term_difference")
        fields = _parse(q.text, _DIFF_PATTERNS)
        P, r = fields["P"], fields["r"]
        n_lower_M = _n_years(P, r, fields["M1"])  # lower repayment -> longer term
        n_higher_M = _n_years(P, r, fields["M2"])  # higher repayment -> shorter term
        assert n_lower_M.denominator == 1 and n_higher_M.denominator == 1, (seed, q.text)
        diff = n_lower_M - n_higher_M
        assert diff > 0, (seed, q.text, n_lower_M, n_higher_M)
        assert str(int(diff)) == q.answer_text, (seed, q.text, q.answer_text, n_lower_M, n_higher_M)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_max_rate_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="max_rate")
        fields = _parse(q.text, _MAX_RATE_PATTERNS)
        r = _r_percent(fields["P"], fields["n"], fields["M"])
        assert r.denominator == 1, (seed, q.text, r)
        assert str(int(r)) == q.answer_text, (seed, q.text, q.answer_text, r)
        checked += 1
    assert checked == SEEDS_PER_TIER
