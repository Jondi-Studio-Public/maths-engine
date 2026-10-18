"""Tests `boolean-algebra-logic-gates` (topic `w6a_boolean_logic`) and its exam chain `exd_x_w6a_boolean_logic`.

The oracles read the expression, the circuit or the Karnaugh map back out of the question TEXT and re-derive the key
with Python bit arithmetic and SymPy's `SOPform`, never from the generator's own helpers.
"""

from __future__ import annotations

import re

import pytest
from sympy import symbols
from sympy.logic.boolalg import SOPform, And, Or, Not

import mathgen.generators.w6a_boolean_logic_build  # noqa: F401  (registers the topics)
from tests import w6a_support as sup

TOPIC = "w6a_boolean_logic"
EXAM = "exd_x_w6a_boolean_logic"
SKILL = sup.skill_of("boolean-algebra-logic-gates")

_GATE_RE = re.compile(r"Gate ([PQRS]) is an? (AND|OR|NOT|NAND|NOR|XOR) gate whose inputs? (?:is|are) ([^.]*)\.")
_ARG_RE = re.compile(r"the output of gate ([PQRS])|\b([ABCD])\b")
_OPS = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XOR": lambda a, b: a ^ b,
}


def _circuit_ones(text: str, n: int) -> list[int]:
    gates = _GATE_RE.findall(text)
    assert gates, text
    out = []
    for row in range(2**n):
        sig = {"ABCD"[i]: (row >> (n - 1 - i)) & 1 for i in range(n)}
        for name, op, args in gates:
            vals = [sig[g or v] for g, v in _ARG_RE.findall(args)]
            sig[name] = 1 - vals[0] if op == "NOT" else _OPS[op](*vals)
        if sig[gates[-1][0]]:
            out.append(row)
    return out


def _eval_expression(expr: str, a: int, b: int, c: int = 0) -> int:
    py = expr.replace("¬", "~").replace("·", "&").replace("⊕", "^").replace("+", "|")
    return eval(py, {}, {"A": a, "B": b, "C": c}) & 1  # noqa: S307  (our own generated text)


def _ones_of_expression(expr: str) -> list[int]:
    return [r for r in range(8) if _eval_expression(expr, (r >> 2) & 1, (r >> 1) & 1, r & 1)]


def _kmap_from_text(text: str):
    """(n, ones, dcs) from either the row listing or the printed map."""
    m = re.search(
        r"is 1 on the rows? numbered ([\d, ]+?)(?:, and its output is a don't care on rows? ([\d, ]+?))? \(", text
    )
    n = 4 if "four inputs" in text else 3
    if m:
        ones = [int(x) for x in m.group(1).split(",")]
        dcs = [int(x) for x in m.group(2).split(",")] if m.group(2) else []
        return n, ones, dcs
    gray = (0, 1, 3, 2)
    ones, dcs = [], []
    if n == 3:
        for a, body in re.findall(r"A = (\d): ([01d ]+)", text):
            for c, cell in enumerate(body.split()):
                r = int(a) * 4 + gray[c]
                (ones if cell == "1" else dcs if cell == "d" else []).append(r)
    else:
        for ab, body in re.findall(r"AB = (\d\d): ([01d ]+)", text):
            for c, cell in enumerate(body.split()):
                r = int(ab, 2) * 4 + gray[c]
                (ones if cell == "1" else dcs if cell == "d" else []).append(r)
    return n, sorted(ones), sorted(dcs)


def _sympy_literals(n, ones, dcs):
    vs = symbols("A B C D")[:n]
    expr = SOPform(vs, ones, dcs)
    terms = expr.args if isinstance(expr, Or) else (expr,)
    count = 0
    for t in terms:
        count += len(t.args) if isinstance(t, And) else 1
    return count, len(terms)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "kmap-no-grouping" in {m.id for m in SKILL.misconceptions}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_independent_evaluation():
    kinds = set()
    for seed in range(120):
        q = sup.single(TOPIC, "easy", seed)
        if "Gate P" in q.text:
            assert len(_circuit_ones(q.text, 3)) == int(q.answer), (seed, q.text)
            kinds.add("circuit")
        elif re.search(r"AB = 00, 01, 10, 11", q.text):
            expr = re.search(r"F = (.+?)(?: with rows| [,.]|[,.] |\.| List| Find)", q.text)
            expr = re.search(r"F = ([^ ]+(?: [+⊕·] [^ ]+)*)", q.text).group(1) if expr is None else expr.group(1)
            bits = "".join(str(_eval_expression(expr, a, b)) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
            assert int(bits, 2) == int(q.answer), (seed, q.text, bits)
            kinds.add("table")
        else:
            expr = re.search(r"F = (.+?)(?:, with| for| Apply|, where|\. )", q.text).group(1)
            assert len(_ones_of_expression(expr)) == int(q.answer), (seed, q.text)
            kinds.add("expression")
    assert kinds == {"circuit", "table", "expression"}


def test_medium_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(TOPIC, "medium", seed)
        if "G = " in q.text:
            g, h = re.search(r"G = (.+?) and H = (.+?)(?:, both| for the same)", q.text).groups()
            diff = sum(
                _eval_expression(g, r >> 2 & 1, r >> 1 & 1, r & 1) != _eval_expression(h, r >> 2 & 1, r >> 1 & 1, r & 1)
                for r in range(8)
            )
            assert diff == int(q.answer), (seed, q.text)
            seen.add("differ")
        elif "Gate P" in q.text:
            assert len(_circuit_ones(q.text, 3)) == int(q.answer)
            assert re.search(r"NAND|NOR|XOR", q.text)
            seen.add("circuit")
        else:
            n, ones, dcs = _kmap_from_text(q.text)
            lits, terms = _sympy_literals(n, ones, dcs)
            want = terms if "product terms" in q.text else lits
            assert want == int(q.answer), (seed, q.text, lits, terms)
            seen.add("kmap")
    assert seen == {"differ", "circuit", "kmap"}


def test_hard_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(TOPIC, "hard", seed)
        if "Gate P" in q.text:
            assert len(_circuit_ones(q.text, 4)) == int(q.answer), (seed, q.text)
            seen.add("circuit")
        else:
            n, ones, dcs = _kmap_from_text(q.text)
            lits, _ = _sympy_literals(n, ones, dcs)
            assert lits == int(q.answer), (seed, q.text)
            seen.add("dc" if dcs else "kmap4")
    assert seen == {"circuit", "dc", "kmap4"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 2, 1], ["number", "set", "number", "number"])


def test_exam_keys_by_independent_evaluation():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        ones = _circuit_ones(q.stem, 3)
        a, b, c, d = q.parts
        assert int(a.answer) == len(ones)
        assert sorted(b.answer) == ones
        lits, terms = _sympy_literals(3, ones, [])
        assert int(c.answer) == lits, (seed, q.stem)
        assert int(d.answer) == 3 * len(ones) - lits
        assert d.follow_through == "3*part_a - part_c"


def test_exam_follow_through_from_a_wrong_count():
    def wrong(q, canon):
        a, c = int(q.parts[0].answer), int(q.parts[2].answer)
        if 3 * (a + 1) - c <= 0:
            return None
        return {"a": str(a + 1), "d": str(3 * (a + 1) - c)}

    assert sup.check_follow_through(SKILL, EXAM, "d", wrong) >= 30


def test_set_part_accepts_any_order_and_rejects_a_missing_row():
    for seed in range(30):
        q = sup.exam(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        rows = list(q.parts[1].answer)
        reply = ", ".join(str(r) for r in reversed(rows))
        assert sup.grade(SKILL, q, {**canon, "b": reply}).parts["b"].marks == 2
        assert sup.grade(SKILL, q, {**canon, "b": ", ".join(map(str, rows[1:]))}).parts["b"].marks == 0


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 5, (tier, spread)
        assert max(spread.values()) <= 30, (tier, spread.most_common(3))
