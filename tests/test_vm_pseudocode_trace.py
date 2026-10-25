"""Tests for `pseudocode-trace-output`: `vm_pseudocode_trace` (easy, medium, hard) and the exam chain.

The oracle is a small interpreter written here that reads the pseudocode back out of the rendered text and runs it in
exact `Fraction` arithmetic, so it shares nothing with the generator, which builds the program and its key
separately. Each hard-tier and exam answer is also re-derived with SymPy's numeric root finders where one applies.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "pseudocode-trace-output"
CHAIN = "vm_pseudocode_trace_exam"


# -- an independent reader of the pseudocode -----------------------------------------------------------------------


def split_statements(code: str) -> list[str]:
    """Statements in order, with a loop or branch header split from the first statement of its body."""
    out = []
    for piece in (p.strip() for p in code.split(";") if p.strip()):
        while True:
            m = re.match(r"((?:for|while|if) [^:]+|otherwise):\s*(.*)$", piece)
            if not m:
                break
            out.append(m.group(1))
            piece = m.group(2)
        out.append(piece)
    return out


def parse_block(stmts: list[str], pos: int, stops: tuple[str, ...]):
    """Statements from `pos` until one starts with a stop word; returns (block, new_pos at the stop word)."""
    block = []
    while pos < len(stmts) and not stmts[pos].startswith(stops):
        s = stmts[pos]
        if s.startswith("for "):
            m = re.match(r"for (\w+) from (.+?) to (.+)", s)
            body, pos = parse_block(stmts, pos + 1, ("end for",))
            block.append(("for", m.group(1), m.group(2), m.group(3), body))
            pos += 1
        elif s.startswith("while "):
            body, pos = parse_block(stmts, pos + 1, ("end while",))
            block.append(("while", s[len("while ") :], body))
            pos += 1
        elif s.startswith("if "):
            then, pos = parse_block(stmts, pos + 1, ("otherwise", "end if"))
            other = []
            if stmts[pos].startswith("otherwise"):
                other, pos = parse_block(stmts, pos + 1, ("end if",))
            block.append(("if", s[len("if ") :], then, other))
            pos += 1
        else:
            block.append(("simple", s))
            pos += 1
    return block, pos


def expr(text: str, env: dict):
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.(])", lambda m: f"Fraction('{m.group(1)}')", text)
    return eval(text, {"Fraction": Fraction}, env)  # noqa: S307  (test-only, strings come from the generator)


def cond(text: str, env: dict) -> bool:
    if text.endswith("is even"):
        return expr(text[: -len(" is even")], env) % 2 == 0
    if " is not " in text:
        a, b = text.split(" is not ")
        return expr(a, env) != expr(b, env)
    text = text.replace("≤", "<=").replace("≥", ">=")
    return bool(expr(text, env))


def run_block(block, env, out):
    for node in block:
        kind = node[0]
        if kind == "simple":
            s = node[1]
            if s.startswith("set "):
                m = re.match(r"set (\w+) to (.+)", s)
                env[m.group(1)] = expr(m.group(2), env)
            elif s.startswith("let "):
                m = re.match(r"let (\w)\(x\) = (.+)", s)
                body = m.group(2)
                env[m.group(1)] = lambda x, body=body: expr(body, {**env, "x": x})
            elif s.startswith("print "):
                out.append(expr(s[len("print ") :], env))
            else:
                raise AssertionError(s)
        elif kind == "for":
            _, var, lo, hi, body = node
            for v in range(int(expr(lo, env)), int(expr(hi, env)) + 1):
                env[var] = Fraction(v)
                run_block(body, env, out)
        elif kind == "while":
            _, c, body = node
            for _ in range(10_000):
                if not cond(c, env):
                    break
                run_block(body, env, out)
            else:
                raise AssertionError("loop did not stop")
        else:
            _, c, then, other = node
            run_block(then if cond(c, env) else other, env, out)


def run_program(code: str):
    stmts = split_statements(code)
    block, pos = parse_block(stmts, 0, ())
    assert pos == len(stmts)
    env, out = {}, []
    run_block(block, env, out)
    return env, out


def program_in(text: str) -> str:
    m = re.search(r": ((?:set|let) .*?print [^.]*?(?:; print [^.]*?)?)\.(?: |$)", text)
    assert m, text
    return m.group(1)


def show(v: Fraction, places: int | None = None) -> str:
    if places is None:
        return str(v.numerator) if v.denominator == 1 else str(v)
    q = sympy.Rational(v.numerator, v.denominator)
    scaled = (q * 10**places + sympy.Rational(1, 2)).floor()
    s = str(abs(scaled)).rjust(places + 1, "0")
    s = f"{s[:-places]}.{s[-places:]}" if places else s
    return ("-" if scaled < 0 else "") + s


def oracle(q):
    out = run_program(program_in(q.text))[1][0]
    places = re.search(r"correct to (\d) decimal", q.text)
    return show(out, int(places.group(1))) if places else show(out)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier,distinct,top", [("easy", 15, 0.15), ("medium", 8, 0.25), ("hard", 25, 0.1)])
def test_answer_space(tier, distinct, top):
    t.assert_answer_space(SKILL, tier, distinct=distinct, top=top)


def test_the_interpreter_agrees_with_known_programs():
    assert run_program("set s to 0; for i from 1 to 4: set s to s + i^2; end for; print s")[1] == [30]
    assert run_program(
        "set s to 6; set n to 0; while s is not 1: if s is even: set s to s ÷ 2; otherwise: set s to 3 × s + 1; "
        "end if; set n to n + 1; end while; print n"
    )[1] == [8]
    env, out = run_program(
        "let f(x) = x^2 - 2; let g(x) = 2 × x; set x to 1; for i from 1 to 2: set x to x - f(x) ÷ g(x); end for; print x"
    )
    assert out == [Fraction(17, 12)]


def test_unknown_tier_is_rejected():
    from mathgen import make

    with pytest.raises(ValueError):
        make("vm_pseudocode_trace", seed=1, tier="nonsense")
    with pytest.raises(ValueError):
        make("vm_pseudocode_trace_exam", seed=1, tier="nonsense")


def test_hard_covers_every_method():
    texts = [t.question(SKILL, "hard", s).text for s in range(80)]
    assert any("g(x)" in x for x in texts), "Newton"
    assert any("(a + b) ÷ 2" in x and "if f(m)" in x for x in texts), "bisection"
    assert any("set h to" in x and "(f(a) + f(b)) ÷ 2" not in x for x in texts), "left rectangles"
    assert any("(f(a) + f(b)) ÷ 2" in x for x in texts), "trapezium"


def test_medium_covers_stopping_conditions():
    texts = [t.question(SKILL, "medium", s).text for s in range(80)]
    assert any("while s <" in x for x in texts)
    assert any("while s ≤" in x for x in texts)
    assert any("is even" in x for x in texts)
    assert any("b > 0" in x for x in texts)


def test_bisection_matches_sympy_root_bracket():
    """The printed midpoint is within one interval width of the true root."""
    x = sympy.Symbol("x")
    for seed in range(80):
        q = t.question(SKILL, "hard", seed)
        m = re.search(r"let f\(x\) = (.+?); set a to (\d+); set b to (\d+); for i from 1 to (\d)", q.text)
        if not m or "g(x)" in q.text:
            continue
        f = sympy.sympify(m.group(1).replace("^", "**"))
        a, b, k = int(m.group(2)), int(m.group(3)), int(m.group(4))
        root = sympy.nsolve(f, x, (a, b), solver="bisect")
        assert abs(float(q.answer_text) - float(root)) <= (b - a) / 2**k + 1e-9, (seed, q.text, q.answer_text)


def test_newton_matches_sympy_iteration():
    x = sympy.Symbol("x")
    for seed in range(80):
        q = t.question(SKILL, "hard", seed)
        m = re.search(r"let f\(x\) = (.+?); let g\(x\) = .+?; set x to (\d+); for i from 1 to (\d)", q.text)
        if not m:
            continue
        f = sympy.sympify(m.group(1).replace("^", "**"))
        v = sympy.Integer(int(m.group(2)))
        for _ in range(int(m.group(3))):
            v = v - f.subs(x, v) / sympy.diff(f, x).subs(x, v)
        places = int(re.search(r"correct to (\d) decimal", q.text).group(1))
        assert abs(float(q.answer_text) - float(v)) <= 0.5 * 10**-places + 1e-12, (seed, q.text)


# -- exam chain ----------------------------------------------------------------------------------------------------


def chain_program(q) -> str:
    return program_in(q.stem + " ")


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["number", "number", "fraction", "number"], [1, 2, 2, 2], [0, 1, 0, 0])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=100, top=0.05)


def test_chain_follow_through_is_declared():
    q = sup.exam_question(CHAIN, 3)
    assert q.parts[1].follow_through and "part_a" in q.parts[1].follow_through


def test_chain_key_matches_independent_interpreter():
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        code = chain_program(q)
        env, out = run_program(code)
        got = {p.label: p.answer_text for p in q.parts}
        assert got["c"] == show(out[0]), (seed, code)
        final = out[1] if "x - f(x)" in code else out[1]
        assert got["d"] == show(final, 4), (seed, code, got)
        if "x - f(x)" in code:
            m = re.search(r"let f\(x\) = (.+?); .*?set x to (\d+);", code)
            x = sympy.Symbol("x")
            f = sympy.sympify(m.group(1).replace("^", "**"))
            v = sympy.Integer(int(m.group(2)))
            steps = []
            for _ in range(2):
                v = v - f.subs(x, v) / sympy.diff(f, x).subs(x, v)
                steps.append(v)
            assert got["a"] == show(Fraction(int(steps[0].p), int(steps[0].q)), 3), (seed, code)
            assert got["b"] == show(Fraction(int(steps[1].p), int(steps[1].q)), 3), (seed, code)
        else:
            width = int(re.search(r"set b to (\d+)", code).group(1)) - int(re.search(r"set a to (\d+)", code).group(1))
            assert got["a"] == show(Fraction(width, 4), 3)
            assert got["b"] == show(Fraction(width, 32), 4)


def _rank(key: str, options) -> int:
    """How many wrong options sort below the key: by value when they are numbers, else by text."""

    def sort_key(text):
        try:
            return (0, float(Fraction(text.replace(" ", ""))), "")
        except (ValueError, ZeroDivisionError):
            return (1, 0.0, text)

    return sum(1 for o in options if sort_key(o) < sort_key(key))


def test_key_rank_varies_pseudocode():
    """The key must not sit in one place among the options: each tier (and each exam part) takes at least three ranks."""
    from collections import Counter

    for tier in ("easy", "medium", "hard"):
        ranks = Counter()
        for seed in range(240):
            q = t.question(SKILL, tier, seed)
            ranks[_rank(q.answer_text, q.distractors)] += 1
        assert len(ranks) >= 3, (tier, ranks)
    parts = {}
    for seed in range(240):
        q = sup.exam_question(CHAIN, seed)
        for p in q.parts:
            parts.setdefault(p.label, Counter())[_rank(p.answer_text, p.distractors)] += 1
    for label, ranks in parts.items():
        assert len(ranks) >= 3, (label, ranks)
