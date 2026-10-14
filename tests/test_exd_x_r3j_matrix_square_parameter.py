"""Exam source of `senior-recognise-different-types-of-matrices-row-column-square`: a square matrix by parameter."""

from __future__ import annotations

import re

import sympy
from exd_r3j_support import (
    canonical,
    check_answer_space,
    check_deterministic,
    check_distractors,
    check_full_marks,
    check_shapes,
    check_source_wired,
    skill,
)

from mathgen import make
from mathgen.engine import grade

TOPIC = "exd_x_r3j_matrix_square_parameter"
SKILL_ID = "senior-recognise-different-types-of-matrices-row-column-square"
_M, _I, _J = sympy.symbols("m i j")


def _entry_rule(text: str):
    """The rule for a(i,j) read from the stem ("3i - 2j + 1", "2i·j - 4") as a SymPy expression."""
    spaced = re.sub(r"(\d)([ij])", r"\1*\2", text.replace("·", "*"))
    spaced = re.sub(r"(?<=\d)\*([ij])\*([ij])", r"*\1*\2", spaced)
    return sympy.parse_expr(spaced, local_dict={"i": _I, "j": _J})


def _oracle(q):
    rule = re.search(r"a\(i,j\) = (.*?)(?: for | where |, where |, for )", q.stem)
    entry = _entry_rule(rule.group(1))
    rows = sympy.parse_expr(_bound(q, "i"), transformations="all", local_dict={"m": _M})
    cols = sympy.parse_expr(_bound(q, "j"), transformations="all", local_dict={"m": _M})
    m = sympy.solve(sympy.Eq(rows, cols), _M)[0]
    n = int(rows.subs(_M, m))
    matrix = sympy.Matrix(n, n, lambda r, c: entry.subs({_I: r + 1, _J: c + 1}))
    assert matrix.shape == (n, n)
    return {"a": str(m), "b": str(n), "c": str(matrix.trace()), "d": str(sum(matrix))}


def _bound(q, var):
    """The upper limit written for `var` in the stem, as text in m."""
    for pattern in (
        rf"{var} = 1, \.\.\., ([^,]*?)(?: and|,| \()",
        rf"{var} runs from 1 to ([^,]*?)(?: and| for|,)",
    ):
        found = re.search(pattern, q.stem)
        if found:
            return found.group(1).strip()
    raise AssertionError(q.stem)


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p19g_matrix_type_and_size", {"variant": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_order():
    """A wrong number of rows, used in the diagonal and total sums as if A were that size, earns both."""
    sk = skill(SKILL_ID)
    done = 0
    for s in range(100):
        q = make(TOPIC, seed=s)
        canon = canonical(q)
        entry = re.search(r"a\(i,j\) = (.*?)(?: for | where |, where |, for )", q.stem).group(1)
        expr = _entry_rule(entry)
        wrong = int(canon["b"]) + 1
        matrix = sympy.Matrix(wrong, wrong, lambda r, c: expr.subs({_I: r + 1, _J: c + 1}))
        reply = {**canon, "b": str(wrong), "c": str(matrix.trace()), "d": str(sum(matrix))}
        if len({int(v) for v in reply.values()}) < 4 or 0 in (matrix.trace(), sum(matrix)):
            continue
        result = grade(sk, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (s, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (s, reply)
        done += 1
    assert done >= 50


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "99", "b": "98", "c": "97", "d": "96"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
