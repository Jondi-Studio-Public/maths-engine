"""Independent oracle for the `pv0b_interest_crossover` questions: it re-derives each answer from the rendered
question text.

Imports nothing from `mathgen.generators.pv0b_interest_crossover_build`.
Parses the stated principal(s), rate(s) or growth data straight out of the
RENDERED `question.text` and re-derives the crossover year from scratch by
scanning year by year with exact `Fraction` arithmetic -- a route
independent of the generator's own float-locate-then-confirm search.

Run across >= 1000 questions total, spread over all four tiers (>= 250
each).
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC = "pv0b_interest_crossover"
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000
HORIZON = 45


def _compile(template: str, names: list[str]) -> re.Pattern:
    """Turn one of the generator's own `.format()`-style templates into a
    regex with a named group per placeholder -- re-authored here,
    independently, purely to parse the rendered text."""
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
                pieces.append(rf"(?P<{name}>\d+(?:\.\d+)?)")
        else:
            pieces.append(re.escape(part))
    return re.compile("^" + "".join(pieces) + "$")


_SAME_PRINCIPAL_PATTERNS = [
    _compile(t, [])
    for t in [
        "An investor deposits ${P} into each of two accounts and makes no further "
        "deposits or withdrawals. Account A earns simple interest at {s}% p.a. "
        "Account B earns compound interest at {c}% p.a., compounded annually. "
        "Within the first 40 years, find the first whole year at the end of which "
        "Account B's balance exceeds Account A's balance.",
        "Nadia invests ${P} in each of two accounts, with no further deposits or "
        "withdrawals. The first pays simple interest at {s}% p.a. The second pays "
        "compound interest at {c}% p.a., compounded annually. Within the first 40 "
        "years, find the first whole year at the end of which the second account's "
        "balance is greater than the first's.",
        "${P} is invested in each of two accounts, with no further deposits or "
        "withdrawals. Account X earns simple interest at {s}% p.a. Account Y earns "
        "compound interest at {c}% p.a., compounded annually. Within the first 40 "
        "years, find the first whole year at the end of which Account Y's balance "
        "exceeds Account X's balance.",
    ]
]

_NOMINAL_PATTERNS = [
    _compile(t, [])
    for t in [
        "An investor deposits ${P} into each of two accounts and makes no further "
        "deposits or withdrawals. Account A earns simple interest at {s}% p.a. "
        "Account B earns compound interest at a nominal rate of {c}% p.a., "
        "compounded {freq}. Within the first 40 years, find the first whole year "
        "at the end of which Account B's balance exceeds Account A's balance.",
        "Priya invests ${P} in each of two accounts, with no further deposits or "
        "withdrawals. The first pays simple interest at {s}% p.a. The second pays "
        "compound interest at a nominal rate of {c}% p.a., compounded {freq}. "
        "Within the first 40 years, find the first whole year at the end of which "
        "the second account's balance is greater than the first's.",
        "${P} is invested in each of two accounts, with no further deposits or "
        "withdrawals. Account X earns simple interest at {s}% p.a. Account Y earns "
        "compound interest at a nominal rate of {c}% p.a., compounded {freq}. "
        "Within the first 40 years, find the first whole year at the end of which "
        "Account Y's balance exceeds Account X's balance.",
    ]
]
# `{freq}` is a word ("monthly"/"quarterly"), not a number -- patch the
# generic `\d+` group for it after compiling.
_NOMINAL_PATTERNS = [
    re.compile(p.pattern.replace(r"(?P<freq>\d+(?:\.\d+)?)", r"(?P<freq>monthly|quarterly)")) for p in _NOMINAL_PATTERNS
]

_DIFFERENT_PRINCIPALS_PATTERNS = [
    _compile(t, [])
    for t in [
        "Yara owes ${Ps} on a simple-interest loan charged at {s}% p.a. Zane owes "
        "${Pc} on a compound-interest loan charged at {c}% p.a., compounded "
        "annually. Neither makes any repayments. Within the first 40 years, find "
        "the first whole year at the end of which Zane owes more than Yara.",
        "A simple-interest loan of ${Ps} is charged at {s}% p.a. A separate "
        "compound-interest loan of ${Pc} is charged at {c}% p.a., compounded "
        "annually. Neither loan is repaid. Within the first 40 years, find the "
        "first whole year at the end of which the amount owed on the "
        "compound-interest loan exceeds the amount owed on the simple-interest "
        "loan.",
        "Deng owes ${Ps} on a simple-interest loan at {s}% p.a. Wei owes ${Pc} on "
        "a compound-interest loan at {c}% p.a., compounded annually. Neither makes "
        "any repayments. Within the first 40 years, find the first whole year at "
        "the end of which Wei owes more than Deng.",
        "Yara invests ${Ps} in an account earning simple interest at {s}% p.a. "
        "Zane invests ${Pc} in a separate account earning compound interest at "
        "{c}% p.a., compounded annually. Neither makes any further deposits or "
        "withdrawals. Within the first 40 years, find the first whole year at the "
        "end of which Zane's balance exceeds Yara's balance.",
        "${Ps} is invested in an account earning simple interest at {s}% p.a. A "
        "separate amount of ${Pc} is invested in an account earning compound "
        "interest at {c}% p.a., compounded annually. Neither account has any "
        "further deposits or withdrawals. Within the first 40 years, find the "
        "first whole year at the end of which the balance of the compound-"
        "interest account exceeds the balance of the simple-interest account.",
        "Deng invests ${Ps} in an account earning simple interest at {s}% p.a. "
        "Wei invests ${Pc} in a separate account earning compound interest at "
        "{c}% p.a., compounded annually. Neither makes any further deposits or "
        "withdrawals. Within the first 40 years, find the first whole year at the "
        "end of which Wei's balance exceeds Deng's balance.",
    ]
]

_RECOVERED_RATES_PATTERNS = [
    _compile(t, [])
    for t in [
        "Priya's simple-interest loan of ${Ps} accrues ${X} in interest every year. "
        "Owen's compound-interest loan of ${Pc}, compounded annually, grows to "
        "${Y} in total owed after 1 year. Neither makes any repayments. Within the "
        "first 40 years, find the first whole year at the end of which Owen owes "
        "more than Priya.",
        "A simple-interest loan of ${Ps} accrues ${X} in interest every year. A "
        "separate compound-interest loan of ${Pc}, compounded annually, grows to a "
        "total of ${Y} owed after 1 year. Neither loan is repaid. Within the first "
        "40 years, find the first whole year at the end of which the amount owed "
        "on the compound-interest loan exceeds the amount owed on the "
        "simple-interest loan.",
        "Wei's simple-interest loan of ${Ps} accrues ${X} in interest every year. "
        "Zara's compound-interest loan of ${Pc}, compounded annually, grows to "
        "${Y} in total owed after 1 year. Neither makes any repayments. Within the "
        "first 40 years, find the first whole year at the end of which Zara owes "
        "more than Wei.",
        "Priya's simple-interest investment of ${Ps} earns ${X} in interest every "
        "year. Owen's compound-interest investment of ${Pc}, compounded annually, "
        "grows to a balance of ${Y} after 1 year. Neither makes any further "
        "deposits or withdrawals. Within the first 40 years, find the first whole "
        "year at the end of which Owen's balance exceeds Priya's balance.",
        "A simple-interest investment of ${Ps} earns ${X} in interest every year. "
        "A separate compound-interest investment of ${Pc}, compounded annually, "
        "grows to a balance of ${Y} after 1 year. Neither investment has any "
        "further deposits or withdrawals. Within the first 40 years, find the "
        "first whole year at the end of which the balance of the "
        "compound-interest investment exceeds the balance of the "
        "simple-interest investment.",
        "Wei's simple-interest investment of ${Ps} earns ${X} in interest every "
        "year. Zara's compound-interest investment of ${Pc}, compounded annually, "
        "grows to a balance of ${Y} after 1 year. Neither makes any further "
        "deposits or withdrawals. Within the first 40 years, find the first whole "
        "year at the end of which Zara's balance exceeds Wei's balance.",
    ]
]


def _parse(text: str, patterns: list[re.Pattern]) -> dict:
    for pattern in patterns:
        m = pattern.match(text)
        if m:
            return m.groupdict()
    raise AssertionError(f"unparseable question against all {len(patterns)} known templates: {text!r}")


def _crossover_year(
    Ps: Fraction, s_pct: Fraction, Pc: Fraction, per_period_pct: Fraction, periods_per_year: int
) -> int:
    for y in range(1, HORIZON + 1):
        compound_val = Pc * (1 + per_period_pct / 100) ** (periods_per_year * y)
        simple_val = Ps * (1 + s_pct * y / 100)
        if compound_val > simple_val:
            return y
    raise AssertionError("no crossover found within the horizon")


def test_oracle_same_principal_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", mode="same_principal")
        f = _parse(q.text, _SAME_PRINCIPAL_PATTERNS)
        P, s, c = Fraction(f["P"]), Fraction(f["s"]), Fraction(f["c"])
        n = _crossover_year(P, s, P, c, 1)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_nominal_compounding_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", mode="nominal_compounding")
        f = _parse(q.text, _NOMINAL_PATTERNS)
        P, s, c_nom = Fraction(f["P"]), Fraction(f["s"]), Fraction(f["c"])
        freq = {"monthly": 12, "quarterly": 4}[f["freq"]]
        per_period = c_nom / freq
        n = _crossover_year(P, s, P, per_period, freq)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_different_principals_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", mode="different_principals")
        f = _parse(q.text, _DIFFERENT_PRINCIPALS_PATTERNS)
        Ps, Pc = Fraction(f["Ps"]), Fraction(f["Pc"])
        s, c = Fraction(f["s"]), Fraction(f["c"])
        assert Pc < Ps, (seed, q.text)
        n = _crossover_year(Ps, s, Pc, c, 1)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_recovered_rates_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="recovered_rates")
        f = _parse(q.text, _RECOVERED_RATES_PATTERNS)
        Ps, Pc = Fraction(f["Ps"]), Fraction(f["Pc"])
        X, Y = Fraction(f["X"]), Fraction(f["Y"])
        s_pct = 100 * X / Ps
        c_pct = 100 * (Y / Pc - 1)
        n = _crossover_year(Ps, s_pct, Pc, c_pct, 1)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        checked += 1
    assert checked == SEEDS_PER_TIER
