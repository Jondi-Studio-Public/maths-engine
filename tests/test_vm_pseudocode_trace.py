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


def test_key_position_in_the_option_list_varies():
    from mathgen.core import mcq_options

    for tier in ("easy", "medium", "hard"):
        spots = {mcq_options(q).index(q.answer_text) for q in (t.question(SKILL, tier, s) for s in range(60))}
        assert len(spots) >= 2, (tier, spots)


# -- what each stem rules out ---------------------------------------------------------------------------------------
#
# Every part of every tier, with what its stem implies about the answer and so about EVERY option (slips and fillers
# alike; the key is never moved to fit). `lo`/`hi` are inclusive and given to within the last printed place.
#
#   part                              constraint on every option
#   --------------------------------  ------------------------------------------------------------------------------
#   easy sum / squares / product/pair whole number, at least 1
#   medium, print n (a pass count)    whole number, at least 1 (a loop that is asked about runs at least once)
#   medium, print s (while s < T)     whole number, at least T (the loop ends once the test fails); T + 1 for s <= T
#   medium Collatz / withdrawal       whole number, at least 1
#   hard Newton (print x)             positive, at most twice the larger of the start and the first iterate
#   hard bisection (print m)          inside the starting bracket [a, b]
#   hard area (print h * s)           between (b - a) f(a) and (b - a) f(b), f being monotone on [a, b]
#   exam Newton (a), (d)              positive, at most the start x0 (Newton falls from a start above the root)
#   exam Newton (b)                   positive, at most the answer to (a) (and at least the answer's last place)
#   exam bisection (a)                positive, at most the starting width
#   exam bisection (b)                positive, at most the width after two passes
#   exam (c), both families           whole number, at least 1
#   CROSS-PART (earlier stems bound later parts)
#   exam Newton (c)                   at least 2: part (b) has already run the loop a second time
#   exam bisection (c)                at least 5: part (b) has already run the loop five times
#   exam Newton (d)                   at most the answer to (b): the loop ran at least twice and x only falls
#   exam bisection (d)                inside the starting bracket [a0, b0]
#   every part                        no option repeats the key


def _opts(part_or_q):
    return [Fraction(d.replace(" ", "")) for d in part_or_q.distractors]


def _slack(prompt: str) -> Fraction:
    m = re.search(r"correct to (\d) decimal", prompt)
    return Fraction(1, 10 ** int(m.group(1))) if m else Fraction(0)


def _inside(opts, lo, hi, slack, context):
    for v in opts:
        assert lo is None or v >= lo - slack, (context, v, lo, opts)
        assert hi is None or v <= hi + slack, (context, v, hi, opts)


def _first_iterate(code: str) -> Fraction:
    return run_program(re.sub(r"for i from 1 to \d+", "for i from 1 to 1", code))[1][0]


def _constraints_single(q):
    code = program_in(q.text)
    opts = _opts(q)
    slack = _slack(q.text)
    key = Fraction(q.answer_text)
    assert key not in opts, q.text
    if "g(x)" in code:  # Newton
        x0 = Fraction(re.search(r"set x to (\d+)", code).group(1))
        _inside(opts, Fraction(0), 2 * max(x0, _first_iterate(code), key), slack, code)
        assert all(v > 0 for v in opts), (code, opts)
    elif "set a to" in code and "set m to" in code:  # bisection
        a = Fraction(re.search(r"set a to (\d+)", code).group(1))
        b = Fraction(re.search(r"set b to (\d+)", code).group(1))
        _inside(opts, a, b, slack, code)
    elif "set h to" in code:  # area
        a = Fraction(re.search(r"set a to (\d+)", code).group(1))
        b = Fraction(re.search(r"set b to (\d+)", code).group(1))
        ftext = re.search(r"let f\(x\) = ([^;]+);", code).group(1)
        fa, fb = (expr(ftext, {"x": Fraction(v)}) for v in (a, b))
        _inside(opts, (b - a) * min(fa, fb), (b - a) * max(fa, fb), slack, code)
    else:
        assert all(v == int(v) and v >= 1 for v in opts), (code, opts)
        w = re.search(r"while s (<|≤) (\d+)", code)
        if w and "print s" in code:
            assert all(v >= int(w.group(2)) + (w.group(1) == "≤") for v in opts), (code, opts)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_no_option_is_ruled_out_by_the_stem(tier):
    for seed in range(300):
        _constraints_single(t.question(SKILL, tier, seed))


def test_no_option_of_an_exam_part_is_ruled_out_by_the_stem():
    for seed in range(300):
        q = sup.exam_question(CHAIN, seed)
        code = chain_program(q)
        parts = {p.label: p for p in q.parts}
        for p in parts.values():
            assert Fraction(p.answer_text) not in _opts(p), (seed, p.label)
        assert all(v == int(v) and v >= 1 for v in _opts(parts["c"])), (seed, parts["c"].distractors)
        newton = "g(x)" in code
        floor = 2 if newton else 5  # what parts (a) and (b) have already made the loop run
        assert "a second time" in parts["b"].prompt if newton else "five times" in parts["b"].prompt
        assert all(v >= floor for v in _opts(parts["c"])), (seed, floor, parts["c"].distractors)
        assert Fraction(parts["c"].answer_text) >= floor, seed
        if newton:
            slack = _slack(parts["b"].prompt)  # (b) is only shown to its own last place
            assert all(v <= Fraction(parts["b"].answer_text) + slack for v in _opts(parts["d"])), (
                seed,
                parts["d"].distractors,
            )
        if "g(x)" in code:
            x0 = Fraction(re.search(r"set x to (\d+)", code).group(1))
            a_key = Fraction(parts["a"].answer_text)
            for label, hi in (("a", x0), ("b", a_key), ("d", x0)):
                opts = _opts(parts[label])
                _inside(opts, Fraction(0), hi, _slack(parts[label].prompt), (seed, label, code))
                assert all(v > 0 for v in opts), (seed, label, opts)
        else:
            a0 = Fraction(re.search(r"set a to (\d+)", code).group(1))
            b0 = Fraction(re.search(r"set b to (\d+)", code).group(1))
            width = b0 - a0
            two = Fraction(parts["a"].answer_text)
            for label, hi in (("a", width), ("b", two)):
                opts = _opts(parts[label])
                _inside(opts, Fraction(0), hi, _slack(parts[label].prompt), (seed, label, code))
                assert all(v > 0 for v in opts), (seed, label, opts)
            _inside(_opts(parts["d"]), a0, b0, _slack(parts["d"].prompt), (seed, "d", code))


def test_working_text_has_no_double_signs_or_empty_terms():
    """Negative constants are rendered with a minus sign, never `+ (-3)` or `--`; a coefficient of 1 or a constant of 0
    is not written out."""
    pattern = re.compile(r"--|\+ -|- -|\(-\d+\)|\b1 ×|\+ 0\b|× 1\b")
    for seed in range(300):
        qs = [t.question(SKILL, tier, seed) for tier in ("easy", "medium", "hard")]
        texts = [(q.text, *q.working) for q in qs]
        ex = sup.exam_question(CHAIN, seed)
        texts.append((ex.stem, *(w for p in ex.parts for w in p.working)))
        for group in texts:
            for text in group:
                assert not pattern.search(text), (seed, text)


# -- the key is not told apart by its shape ---------------------------------------------------------------------------


def _kinds(text: str):
    """The sign, number of integer digits, decimal places and integer-ness of a numeric option (None if not a number)."""
    text = text.replace(" ", "")
    try:
        v = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return {
        "sign": v < 0,
        "digits": len(str(int(abs(v)))),
        "places": len(text.split(".")[1]) if "." in text else 0,
        "whole": v.denominator == 1,
    }


def _lone_kinds(items):
    """Over (key, options) pairs: how often the key is the ONLY option of its kind, per kind."""
    from collections import Counter

    lone, total = Counter(), 0
    for key, options in items:
        k, others = _kinds(key), [_kinds(o) for o in options]
        if k is None or None in others or len(others) < 3:
            continue
        total += 1
        for dim in k:
            if all(o[dim] != k[dim] for o in others):
                lone[dim] += 1
    return {dim: lone[dim] / max(total, 1) for dim in ("sign", "digits", "places", "whole")}, total


def test_the_key_is_never_told_apart_by_sign_size_or_format():
    """Over 1000 seeds per tier and per exam part, the key is the only option of its sign, integer-digit count, decimal
    places or whole-ness in at most a small share of questions (chance alone gives a few percent; a generator whose
    wrong answers share a shape that the key lacks gives a third or more)."""
    from collections import defaultdict

    groups = {}
    for tier in ("easy", "medium", "hard"):
        qs = [t.question(SKILL, tier, s) for s in range(1000)]
        groups[tier] = [(q.answer_text, q.distractors) for q in qs]
    parts = defaultdict(list)
    for s in range(1000):
        for p in sup.exam_question(CHAIN, s).parts:
            parts[f"exam {p.label}"].append((p.answer_text, p.distractors))
    groups.update(parts)
    for name, items in groups.items():
        shares, total = _lone_kinds(items)
        if total < 100:
            continue
        for dim, share in shares.items():
            assert share <= 0.12, (name, dim, round(share, 3), shares)
