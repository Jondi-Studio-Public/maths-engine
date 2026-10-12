"""Independent oracle for the nine `cp1_*_proof` topics: it re-derives what each question claims from the
rendered question text alone.

Imports nothing from `mathgen.generators` -- no helper, no table, no constant. Each prompt is read with this
module's own small text-to-sympy translation, and the checks are:

(a) the working a learner is asked to write is a true chain: the expression the prompt says to start from and
    the one it says to finish at agree at several points (or, for a worked numeric item, the start expression
    equals the value in the answer);
(b) the start and end printed in the prompt are the start and target of the marking dict;
(c) every claim the prompt states as a fact holds: the sum formula in an induction prompt for n = 1..10, the
    divisibility claim for n = 1..40, the inductive-step start and target, the equation of a path along the
    path, each binomial coefficient the prompt quotes, the sum of a stated sequence, and the real or imaginary
    part of a stated complex number.

`N_SEEDS` per tier is 60 by default, overridable with `CP1_ORACLE_SEEDS`. Each checker is also run against a
deliberately wrong prompt (the red-first tests at the end), so a checker that passes everything cannot pass.
"""

from __future__ import annotations

import math
import os
import re

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    factorial_notation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen

TOPICS = (
    "cp1_sum_induction_proof",
    "cp1_divisibility_proof",
    "cp1_pascal_identity_proof",
    "cp1_cartesian_path_proof",
    "cp1_exponential_laws_proof",
    "cp1_arithmetic_sum_proof",
    "cp1_geometric_sum_proof",
    "cp1_factor_remainder_proof",
    "cp1_de_moivre_proof",
)
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("CP1_ORACLE_SEEDS", "60"))

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor, factorial_notation)
_BINOM_RE = re.compile(r"\bC\(([^(),]+),\s*([^(),]+)\)")


def _parse(text: str) -> sp.Expr:
    e = text.strip().replace("×", "*").replace("÷", "/").replace("π", "pi").replace("−", "-")
    e = _BINOM_RE.sub(r"binomial(\1,\2)", e)
    e = re.sub(r"\bi\b", "I", e)
    return parse_expr(e, transformations=_TRANSFORMS, evaluate=True)


# ---------------------------------------------------------------------------
# Points to evaluate at
# ---------------------------------------------------------------------------
_REALS = (0.37, 1.19, 2.53, 0.81, 1.7, 2.2)
_INTS = (3, 4, 5, 6, 7, 8)


def _points(symbols, integer: bool, count: int = 5):
    pool = _INTS if integer else _REALS
    ordered = sorted(symbols, key=str)
    return [{s: pool[(p * 2 + j * 3) % len(pool)] for j, s in enumerate(ordered)} for p in range(count)]


def _value(expr, point):
    try:
        v = complex(sp.N(expr.subs(point), 30))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if any(math.isnan(x) or math.isinf(x) for x in (v.real, v.imag)):
        return None
    return v


def _agree(a, b, integer: bool, need: int = 2):
    """True when a and b agree at `need` or more points; a mismatch at any point is False."""
    a, b = sp.sympify(a), sp.sympify(b)
    symbols = a.free_symbols | b.free_symbols
    good = 0
    for point in _points(symbols, integer):
        va, vb = _value(a, point), _value(b, point)
        if va is None or vb is None:
            continue
        if abs(va - vb) > 1e-8 * max(1.0, abs(va), abs(vb)):
            return False
        good += 1
    return good >= need


# ---------------------------------------------------------------------------
# Reading the prompt
# ---------------------------------------------------------------------------
_ENDS = (
    re.compile(
        r".*(?:Start your working from|Begin from|Start from) (.+?) and (?:finish|end) at "
        r"(?:the exact value|(.+?)(?:, writing one expression per line|, one typed expression per line"
        r"|, one expression per line|\.(?= Put brackets)))"
    ),
    re.compile(r".*starting at (.+?) and finishing at (.+?)\.(?= Put brackets)"),
)


def _ends(text: str):
    for pattern in _ENDS:
        m = pattern.match(text)
        if m:
            return m.group(1), m.group(2)
    return None


def _split_top(text: str, sep: str):
    parts, depth, cur = [], 0, ""
    i = 0
    while i < len(text):
        ch = text[i]
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and text.startswith(sep, i):
            parts.append(cur)
            cur = ""
            i += len(sep)
            continue
        cur += ch
        i += 1
    parts.append(cur)
    return parts


def _is_integer_item(text: str, start: str) -> bool:
    return "integer" in text or "!" in start or "C(" in start


def _answer_value(answer_text: str):
    return _parse(answer_text.rsplit("=", 1)[-1])


# ---------------------------------------------------------------------------
# The checkers: each returns a list of problems (empty means every claim held)
# ---------------------------------------------------------------------------
def check_ends(text: str, answer_text: str, answer: dict) -> list[str]:
    problems: list[str] = []
    ends = _ends(text)
    if ends is None:
        return ["no start and end clause found in the prompt"]
    start_text, end_text = ends
    start = _parse(start_text)
    integer = _is_integer_item(text, start_text)
    marking_start = sp.sympify(answer["start"])
    if not _agree(start, marking_start, integer):
        problems.append(f"printed start {start_text!r} is not the marking start {answer['start']!r}")
    if end_text is None:  # a worked numeric item: the start is an expression whose value the answer states
        if not _agree(start, _answer_value(answer_text), False, need=1):
            problems.append(f"start {start_text!r} does not evaluate to the stated answer {answer_text!r}")
    else:
        end = _parse(end_text)
        if not _agree(end, sp.sympify(answer["target"]), integer):
            problems.append(f"printed end {end_text!r} is not the marking target {answer['target']!r}")
        if not _agree(start, end, integer):
            problems.append(f"{start_text!r} does not equal {end_text!r}")
    return problems


_SUM_CLAIM = re.compile(r"Prove by induction that (.+?) = (.+?) for every positive integer n\.")
_STEP_NOTE = re.compile(r"add the \(k \+ 1\)th term")


def check_sum_claim(text: str, answer: dict) -> list[str]:
    m = _SUM_CLAIM.search(text)
    if not m or not _STEP_NOTE.search(text) or " + ... + " not in m.group(1):
        return []
    lhs, rhs_text = m.group(1), m.group(2)
    head, _, last = lhs.partition(" + ... + ")
    n = sp.Symbol("n", integer=True, positive=True)
    k = sp.Symbol("k", integer=True, positive=True)
    last_expr = _parse(last).subs(sp.Symbol("n"), n)
    rhs = _parse(rhs_text).subs(sp.Symbol("n"), n)
    problems: list[str] = []
    shown_total = _parse(head)
    if not any(sp.simplify(sum(last_expr.subs(n, i) for i in range(1, m + 1)) - shown_total) == 0 for m in range(2, 6)):
        problems.append("the listed leading terms are not the first terms of the general term")
    for upto in range(1, 11):
        total = sum(last_expr.subs(n, i) for i in range(1, upto + 1))
        if sp.simplify(total - rhs.subs(n, upto)) != 0:
            problems.append(f"the stated sum formula is false at n = {upto}")
            break
    start, target = sp.sympify(answer["start"]), sp.sympify(answer["target"])
    step_start = rhs.subs(n, k) + last_expr.subs(n, k + 1)
    if not _agree(start, step_start.subs(k, sp.Symbol("k")), True):
        problems.append("the inductive-step start is not the formula at k plus the next term")
    if not _agree(target, rhs.subs(n, k + 1).subs(k, sp.Symbol("k")), True):
        problems.append("the inductive-step target is not the formula at k + 1")
    return problems


_DIV_CLAIM = re.compile(
    r"Prove by induction that (.+?) is divisible by (\d+) for every positive integer n\. .*?assume that (.+?) is "
    r"divisible by \2 and rewrite (.+?) in terms of it\."
)


def check_divisibility_claim(text: str, answer: dict) -> list[str]:
    m = _DIV_CLAIM.search(text)
    if not m:
        return []
    f_n, divisor, f_k, f_k1 = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    n, k = sp.Symbol("n"), sp.Symbol("k")
    f = _parse(f_n)
    problems: list[str] = []
    for value in range(1, 41):
        v = sp.nsimplify(f.subs(n, value))
        if v != int(v) or int(v) % divisor != 0:
            problems.append(f"the claim is false at n = {value}")
            break
    if not _agree(_parse(f_k), f.subs(n, k), True):
        problems.append("the hypothesis is not the claim at k")
    if not _agree(_parse(f_k1), f.subs(n, k + 1), True):
        problems.append("the expression to rewrite is not the claim at k + 1")
    if not _agree(sp.sympify(answer["start"]), _parse(f_k1), True):
        problems.append("the marking start is not the expression to rewrite")
    return problems


_PATH_CLAIM = re.compile(r"r\((\w)\) = (.+?) i \+ (.+?) j\. Show that its points satisfy (.+?) = (.+?)\. Replace")


def check_path_claim(text: str) -> list[str]:
    m = _PATH_CLAIM.search(text)
    if not m:
        return []
    param = sp.Symbol(m.group(1))
    x_expr, y_expr = _parse(m.group(2)), _parse(m.group(3))
    lhs, rhs = _parse(m.group(4)), _parse(m.group(5))
    x, y = sp.Symbol("x"), sp.Symbol("y")
    diff = (lhs - rhs).subs({x: x_expr, y: y_expr})
    good = 0
    for value in (0.31, 0.77, 1.13, 1.9, 2.6, 3.4):
        v = _value(diff, {param: value})
        if v is None:
            continue
        if abs(v) > 1e-8 * max(1.0, abs(_value(lhs.subs({x: x_expr, y: y_expr}), {param: value}) or 1)):
            return ["a point of the path does not satisfy the stated equation"]
        good += 1
    return [] if good >= 3 else ["too few usable points on the path"]


_COMB_FACT = re.compile(r"C\((\d+), (\d+)\) = (\d+)")
_COMB_FORMULA = re.compile(r"\bC\(n, (\d+)\) = ([^,]+?) and C\(n, (\d+)\) = ([^.]+?)\. ")


def check_binomials(text: str, answer_text: str) -> list[str]:
    problems: list[str] = []
    for top, bottom, value in _COMB_FACT.findall(text):
        if math.comb(int(top), int(bottom)) != int(value):
            problems.append(f"C({top}, {bottom}) is not {value}")
    m = re.match(r"C\((\d+), (\d+)\) = (\d+)$", answer_text)
    if m and math.comb(int(m.group(1)), int(m.group(2))) != int(m.group(3)):
        problems.append(f"the stated answer {answer_text!r} is false")
    n = sp.Symbol("n")
    for r1, f1, r2, f2 in _COMB_FORMULA.findall(text):
        for r, f in ((r1, f1), (r2, f2)):
            for value in (8, 9, 10, 12):
                if sp.simplify(_parse(f).subs(n, value) - math.comb(value, int(r))) != 0:
                    problems.append(f"C(n, {r}) is not {f}")
                    break
    return problems


_SEQUENCE = re.compile(r"(?:first (\d+) terms of the sequence|arithmetic sequence) ([\d, ]+), \.\.\.(?:, (\d+))?")


def check_sequence_sum(text: str, answer_text: str) -> list[str]:
    m = _SEQUENCE.search(text)
    if not m:
        return []
    terms = [int(t) for t in m.group(2).replace(" ", "").split(",") if t]
    if len(terms) < 3:
        return []
    a, b, c = terms[:3]
    if m.group(1):
        count = int(m.group(1))
    else:
        count = (int(m.group(3)) - a) // (b - a) + 1
    if b - a == c - b:
        total = sum(a + i * (b - a) for i in range(count))
        if m.group(3):
            assert a + (count - 1) * (b - a) == int(m.group(3)), "the stated last term is not in the sequence"
    else:
        ratio = sp.Rational(b, a)
        assert sp.Rational(c, b) == ratio, "neither arithmetic nor geometric"
        total = sum(a * ratio**i for i in range(count))
    stated = _answer_value(answer_text)
    return [] if sp.simplify(total - stated) == 0 else [f"the sum of the stated sequence is {total}, not {stated}"]


_PART = re.compile(r"find the (real|imaginary) part of (.+?) exactly\.")


def check_complex_part(text: str, answer_text: str) -> list[str]:
    m = _PART.search(text)
    if not m:
        return []
    z = _parse(m.group(2))
    value = complex(sp.N(z, 30))
    part = value.real if m.group(1) == "real" else value.imag
    stated = complex(sp.N(_answer_value(answer_text), 30)).real
    return [] if abs(part - stated) < 1e-9 else [f"the {m.group(1)} part is {part}, not {stated}"]


def check_item(topic: str, q) -> list[str]:
    text, answer_text, answer = q.text, q.answer_text, q.answer
    problems = check_ends(text, answer_text, answer)
    problems += check_sum_claim(text, answer)
    problems += check_divisibility_claim(text, answer)
    problems += check_path_claim(text)
    problems += check_binomials(text, answer_text)
    problems += check_sequence_sum(text, answer_text)
    problems += check_complex_part(text, answer_text)
    return problems


def _distinct(topic: str, tier: str, n: int = N_SEEDS):
    seen = {}
    for seed in range(n):
        q = mathgen.make(topic, seed=seed, difficulty=tier)
        seen.setdefault(q.text, q)
    return list(seen.values())


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_every_stated_claim_is_true(topic, tier):
    items = _distinct(topic, tier)
    assert len(items) >= 6, (topic, tier, len(items))
    for q in items:
        assert check_item(topic, q) == [], (topic, tier, q.text)


# ---------------------------------------------------------------------------
# Coverage: the topic-specific checkers must actually have run on the topics they belong to.
# ---------------------------------------------------------------------------
def _hits(topic, tier, finder):
    return sum(1 for q in _distinct(topic, tier) if finder(q))


def test_the_specific_checkers_are_exercised():
    assert _hits("cp1_sum_induction_proof", "medium", lambda q: _SUM_CLAIM.search(q.text)) >= 5
    assert _hits("cp1_arithmetic_sum_proof", "easy", lambda q: _SUM_CLAIM.search(q.text)) >= 3
    assert _hits("cp1_geometric_sum_proof", "easy", lambda q: _SUM_CLAIM.search(q.text)) >= 3
    assert _hits("cp1_divisibility_proof", "medium", lambda q: _DIV_CLAIM.search(q.text)) >= 5
    assert _hits("cp1_cartesian_path_proof", "medium", lambda q: _PATH_CLAIM.search(q.text)) >= 5
    assert _hits("cp1_pascal_identity_proof", "easy", lambda q: _COMB_FACT.search(q.text)) >= 3
    assert _hits("cp1_arithmetic_sum_proof", "exam", lambda q: _SEQUENCE.search(q.text)) >= 3
    assert _hits("cp1_geometric_sum_proof", "easy", lambda q: _SEQUENCE.search(q.text)) >= 3
    assert _hits("cp1_de_moivre_proof", "easy", lambda q: _PART.search(q.text)) >= 3


# ---------------------------------------------------------------------------
# Red first: each checker rejects a deliberately wrong prompt.
# ---------------------------------------------------------------------------
class _Fake:
    def __init__(self, q, text=None, answer_text=None, answer=None):
        self.text = text if text is not None else q.text
        self.answer_text = answer_text if answer_text is not None else q.answer_text
        self.answer = answer if answer is not None else q.answer


def _first(topic, tier, finder):
    for q in _distinct(topic, tier):
        if finder(q):
            return q
    raise AssertionError((topic, tier))


def test_red_a_false_sum_formula_is_caught():
    q = _first("cp1_sum_induction_proof", "medium", lambda q: _SUM_CLAIM.search(q.text))
    m = _SUM_CLAIM.search(q.text)
    bad = q.text.replace(f" = {m.group(2)} for", f" = ({m.group(2)}) + 1 for", 1)
    assert check_sum_claim(bad, q.answer)


def test_red_a_wrong_first_listed_term_is_caught():
    q = _first("cp1_sum_induction_proof", "medium", lambda q: _SUM_CLAIM.search(q.text))
    head = _SUM_CLAIM.search(q.text).group(1).partition(" + ... + ")[0]
    bad = q.text.replace(head, "99 + " + head, 1)
    assert check_sum_claim(bad, q.answer)


def test_red_a_start_that_is_not_the_step_is_caught():
    q = _first("cp1_sum_induction_proof", "medium", lambda q: _SUM_CLAIM.search(q.text))
    answer = dict(q.answer, start=f"({q.answer['start']}) + 1")
    assert check_sum_claim(q.text, answer)


def test_red_a_false_divisibility_claim_is_caught():
    q = _first("cp1_divisibility_proof", "medium", lambda q: _DIV_CLAIM.search(q.text))
    d = _DIV_CLAIM.search(q.text).group(2)
    bad = re.sub(rf"divisible by {d}", f"divisible by {int(d) * 7 + 1}", q.text)
    assert check_divisibility_claim(bad, q.answer)


def test_red_a_path_off_the_curve_is_caught():
    q = _first("cp1_cartesian_path_proof", "medium", lambda q: _PATH_CLAIM.search(q.text))
    m = _PATH_CLAIM.search(q.text)
    bad = q.text.replace(f"satisfy {m.group(4)} = {m.group(5)}.", f"satisfy {m.group(4)} = ({m.group(5)}) + 2.", 1)
    assert check_path_claim(bad)


def test_red_a_wrong_binomial_value_is_caught():
    q = _first("cp1_pascal_identity_proof", "easy", lambda q: _COMB_FACT.search(q.text))
    top, bottom, value = _COMB_FACT.search(q.text).groups()
    bad = q.text.replace(f"C({top}, {bottom}) = {value}", f"C({top}, {bottom}) = {int(value) + 1}", 1)
    assert check_binomials(bad, q.answer_text)


def test_red_a_wrong_sequence_sum_is_caught():
    q = _first("cp1_arithmetic_sum_proof", "exam", lambda q: _SEQUENCE.search(q.text))
    assert check_sequence_sum(q.text, "sum = 1")


def test_red_a_wrong_complex_part_is_caught():
    q = _first("cp1_de_moivre_proof", "easy", lambda q: _PART.search(q.text))
    assert check_complex_part(q.text, "real part = 12345")


def test_red_a_start_that_differs_from_the_end_is_caught():
    q = _first("cp1_exponential_laws_proof", "medium", lambda q: _ends(q.text) and _ends(q.text)[1])
    start, end = _ends(q.text)
    bad = q.text.replace(f"and end at {end},", f"and end at ({end}) + 1,").replace(
        f"and finish at {end},", f"and finish at ({end}) + 1,"
    )
    bad = bad.replace(f"finishing at {end}.", f"finishing at ({end}) + 1.")
    assert bad != q.text
    assert check_ends(bad, q.answer_text, q.answer)


def test_red_a_marking_start_that_is_not_the_printed_start_is_caught():
    q = _first("cp1_exponential_laws_proof", "medium", lambda q: _ends(q.text) and _ends(q.text)[1])
    answer = dict(q.answer, start=f"2*({q.answer['start']})")
    assert check_ends(q.text, q.answer_text, answer)
