"""Independent oracle for `senior-calculate-the-dividend-paid-on-a-portfolio-of`.

Reads only `question.text`: it parses the one-line table, then solves every blank by propagating the three
relationships the question states (dividend paid = shares x dividend per share; yield = dividend per share /
price as a percentage; P/E = price / earnings per share), reading a "3.2% of share price" or "$0.85 per share"
declaration where there is one. It checks every given cell, every blank, the canonical `answer_text` and the real
`table` grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "senior_dividend_portfolio_table"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("SENIOR_DIVIDEND_ORACLE_SEEDS", "120"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")

# header text -> variable name
_VARS = {
    "Shares held": "n",
    "Share price ($)": "price",
    "Earnings per share ($)": "eps",
    "Dividend per share ($)": "dps",
    "Dividend paid ($)": "paid",
    "Dividend yield (%)": "yield",
    "P/E ratio": "pe",
}
_PLACES = {"paid": 2, "price": 2, "eps": 2, "dps": 2, "n": 0, "yield": 1, "pe": 1}


def _round_half_up(value: Fraction, places: int) -> Fraction:
    scaled = value * 10**places
    return Fraction((scaled + Fraction(1, 2)).__floor__(), 10**places)


def _show(value: Fraction, places: int) -> str:
    scaled = int(_round_half_up(value, places) * 10**places)
    if places == 0:
        return str(scaled)
    digits = str(scaled).rjust(places + 1, "0")
    return f"{digits[:-places]}.{digits[-places:]}"


def _parse(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return head, rows


def _solve_row(cells, head):
    """({variable: Fraction}, unknown names) for one company row, unknowns solved by propagation."""
    known, unknown = {}, set()
    for col, cell in zip(head[1:], cells[1:]):
        if col in _VARS:
            name = _VARS[col]
            if cell == "?":
                unknown.add(name)
            else:
                known[name] = Fraction(cell)
        elif col == "Dividend declared":
            if cell.endswith("% of share price"):
                known["declared_pct"] = Fraction(cell.split("%")[0])
            else:
                m = re.fullmatch(r"\$(\d+\.\d\d) per share", cell)
                assert m, cell
                known["dps"] = Fraction(m.group(1))
    # in the medium table the dividend per share is not a column: a percentage declaration gives it directly
    if "declared_pct" in known:
        known["dps"] = known["declared_pct"] / 100 * known["price"]
    relations = [
        ("paid", lambda v: v["n"] * v["dps"], ("n", "dps")),
        ("n", lambda v: v["paid"] / v["dps"], ("paid", "dps")),
        ("dps", lambda v: v["paid"] / v["n"], ("paid", "n")),
        ("dps", lambda v: v["yield"] / 100 * v["price"], ("yield", "price")),
        ("yield", lambda v: v["dps"] / v["price"] * 100, ("dps", "price")),
        ("price", lambda v: v["dps"] / v["yield"] * 100, ("dps", "yield")),
        ("pe", lambda v: v["price"] / v["eps"], ("price", "eps")),
        ("price", lambda v: v["pe"] * v["eps"], ("pe", "eps")),
        ("eps", lambda v: v["price"] / v["pe"], ("price", "pe")),
    ]
    solved = dict(known)
    changed = True
    while changed:
        changed = False
        for target, fn, needs in relations:
            if target in unknown and target not in solved and all(n in solved for n in needs):
                solved[target] = fn(solved)
                changed = True
    return solved, unknown


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_solves_every_blank_and_matches_the_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        head, rows = _parse(q.text)
        body, total = rows[:-1], rows[-1]
        parts, paid_total = [], Fraction(0)
        for cells in body:
            solved, unknown = _solve_row(cells, head)
            assert unknown <= set(solved), (tier, seed, cells, "the blanks are not determined by the given cells")
            paid_total += solved["paid"]
            for col, cell in zip(head[1:], cells[1:]):
                if col not in _VARS:
                    continue
                name = _VARS[col]
                shown = _show(solved[name], _PLACES[name])
                if cell == "?":
                    parts.append(f"{cells[0]} {col} = {shown}")
                else:
                    assert cell == shown, (tier, seed, cells[0], col, cell, shown)
        paid_col = head.index("Dividend paid ($)")
        if total[paid_col] == "?":
            parts.append(f"Total Dividend paid ($) = {_show(paid_total, 2)}")
        else:
            assert total[paid_col] == _show(paid_total, 2)
        # reading order: the oracle collected row by row, columns in header order, then the total
        assert parts and len(parts) == len(q.answer["blanks"])
        assert set(parts) == set(q.answer_text.split(", "))
        assert q.answer_text == ", ".join(parts)
        assert grade_table(q, ", ".join(parts)).correct, (tier, seed)


def test_exam_blanks_need_a_backward_step_in_at_least_one_row():
    """In every exam question some blank can only be found from a relation run backwards."""
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="exam")
        head, rows = _parse(q.text)
        backward = False
        for cells in rows[:-1]:
            blank_cols = {col for col, cell in zip(head[1:], cells[1:]) if cell == "?"}
            if blank_cols & {"Share price ($)", "Shares held", "Dividend per share ($)"}:
                backward = True
        assert backward, seed


def test_hard_and_exam_state_the_formulae_the_oracle_uses():
    for tier in ("hard", "exam"):
        text = make(TOPIC, seed=1, difficulty=tier).text
        assert "dividend yield = dividend per share / share price" in text
        assert "P/E ratio = share price / earnings per share" in text
