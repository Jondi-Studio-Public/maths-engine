"""Shared checks for the second-year complex-analysis skills (``tests/test_w5a_*.py``).

Two kinds of helper live here.

* The structural checks every skill shares: the canonical answer is marked right and each distractor wrong, the text
  is clean, exam parts chain.
* An independent oracle: ``parse`` turns the learner-facing text of a question (``(z² + 1)/((z - i)(z + 2))``,
  ``e^(3z)``, ``2xcos(y)``) back into a SymPy expression, so a test can recompute the answer from the stem alone, by a
  route that shares no code with the generator.
"""

from __future__ import annotations

import functools
import os
import re

import sympy

import mathgen.generators  # noqa: F401  (registers every generator)
from mathgen import make
from mathgen.grading import grade
from mathgen.grading.parts import follow_through_target, reply_value

SEEDS = range(1, 9)
# the oracle checks read more seeds when W5A_ORACLE_SEEDS is set (a wider sweep before a change is merged)
ORACLE_SEEDS = range(1, 1 + int(os.environ.get("W5A_ORACLE_SEEDS", "8")))
TIERS = ("easy", "medium", "hard")

X, Y, Z = sympy.symbols("x y z")


def tier_topic(name: str, tier: str) -> str:
    return f"exd_x_w5a_{name}" if tier == "exam" else f"w5a_{name}"


@functools.lru_cache(maxsize=None)
def _one(name: str, tier: str, seed: int):
    return make(tier_topic(name, tier), seed=seed, difficulty=tier)


def questions(name: str, tier: str, seeds=SEEDS):
    return [_one(name, tier, s) for s in seeds]


# -- structural checks ---------------------------------------------------------------------------------------------------


def check_grading(q) -> None:
    """The canonical answer is marked right and every distractor wrong, part by part."""
    if q.parts:
        right = {p.label: p.answer_text for p in q.parts}
        result = grade(q.parts[0].answer_kind, q, right)
        assert result.correct, (q.seed, right, result)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.seed, p.label)
            for d in p.distractors:
                wrong = dict(right, **{p.label: str(d)})
                assert not grade(q.parts[0].answer_kind, q, wrong).correct, (q.seed, p.label, str(d))
        return
    assert grade(q.answer_kind, q, q.answer_text).correct, (q.seed, q.answer_text)
    assert len(q.distractors) >= 3, q.seed
    for d in q.distractors:
        assert not grade(q.answer_kind, q, str(d)).correct, (q.seed, str(d), q.answer_text)


def check_follow_through(q) -> None:
    """Each follow-through part reproduces its own canonical answer from the earlier parts' answers."""
    values = {}
    for p in q.parts:
        if p.follow_through:
            target = follow_through_target(p, values)
            assert target is not None, (q.seed, p.label)
            assert grade(p.answer_kind, target, p.answer_text).correct or p.answer_kind == "number", (q.seed, p.label)
        values[p.label] = reply_value(p.answer_kind, p.answer_text)


def check_text(q) -> None:
    from mathgen.text_quality import find_text_defects

    assert not find_text_defects(q, include_working=True), (q.seed, find_text_defects(q, include_working=True))
    texts = [q.text] + [p.prompt for p in q.parts]
    for t in texts:
        assert t and t == t.strip()
        assert "  " not in t and "None" not in t and "{" not in t and "}" not in t, t
        assert "^1 " not in t and "(1t)" not in t, t
        assert t.count("(") == t.count(")"), t
        assert "- -" not in t and "+ -" not in t, t


# -- the oracle: stems back into SymPy ---------------------------------------------------------------------------------


_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_FUNCS = ("sinh", "cosh", "tanh", "sin", "cos", "tan", "cot", "exp", "log", "ln", "sqrt", "conj", "atan")
_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(?P<name>[A-Za-zπ√∞]+)|(?P<op>\*\*|[-+*/^()|,]))"
)


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        assert m, f"cannot read {text[pos:]!r} in {text!r}"
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "name":
            # split a run of letters into functions and single-letter symbols
            while val:
                for f in _FUNCS:
                    if val.startswith(f):
                        out.append(("func", f))
                        val = val[len(f) :]
                        break
                else:
                    ch = val[0]
                    val = val[1:]
                    out.append(("sym", ch))
        elif kind == "sup":
            out.append(("op", "**"))
            out.append(("num", "".join(_SUP[c] for c in val)))
        else:
            out.append((kind, val))
    return out


def _function_powers(text: str) -> str:
    """sin²(2z) becomes (sin(2z))^2, so the power applies to the whole function value."""
    pattern = re.compile(r"(sinh|cosh|tanh|sin|cos|tan|cot)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\(")
    while True:
        m = pattern.search(text)
        if not m:
            return text
        depth, k = 1, m.end()
        while depth:
            depth += text[k] == "("
            depth -= text[k] == ")"
            k += 1
        power = "".join(_SUP[c] for c in m.group(2))
        text = text[: m.start()] + f"({m.group(1)}({text[m.end() : k - 1]}))^{power}" + text[k:]


def parse(text: str):
    """A SymPy expression for learner-style text such as ``(z² - 1)/((z - 2i)(z + 1))`` or ``e^(-3y)cos(3x)``."""
    toks = _tokens(_function_powers(text))
    pieces: list[str] = []
    prev = None  # kind of the previous emitted token: "value-end" or other
    for kind, val in toks:
        if kind == "num":
            if prev == "end":
                pieces.append("*")
            pieces.append(val)
            prev = "end"
        elif kind == "sym":
            name = {"π": "pi", "i": "I", "e": "E"}.get(val, val)
            if val == "√":
                name = "sqrt"
            if val == "∞":
                name = "oo"
            if prev == "end":
                pieces.append("*")
            pieces.append(name)
            prev = "start-func" if val == "√" else "end"
        elif kind == "func":
            if prev == "end":
                pieces.append("*")
            name = {"ln": "log", "conj": "conjugate"}.get(val, val)
            pieces.append(name)
            prev = "func"
        elif kind == "op":
            if val == "(":
                if prev == "end":
                    pieces.append("*")
                pieces.append("(")
                prev = "open"
            elif val == ")":
                pieces.append(")")
                prev = "end"
            elif val == "^":
                pieces.append("**")
                prev = "op"
            else:
                pieces.append(val)
                prev = "op"
    src = "".join(pieces)
    src = re.sub(r"sqrt(\d+)", r"sqrt(\1)", src)
    src = re.sub(r"sqrt\*?(\d+)", r"sqrt(\1)", src)
    src = re.sub(r"sqrt\*?I", "sqrt(I)", src)
    return sympy.sympify(
        src,
        locals={"x": X, "y": Y, "z": Z, "t": sympy.Symbol("t"), "k": sympy.Symbol("k"), "conjugate": sympy.conjugate},
    )


def balanced_after(text: str, marker: str) -> str:
    """The expression that follows `marker`: up to the first sentence end, `is`, `has`, `and`, or closing bracket."""
    start = text.index(marker) + len(marker)
    depth, k = 0, start
    while k < len(text):
        ch = text[k]
        if ch in "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            rest = text[k:]
            if (
                re.match(r"\.(\s|$)", rest)
                or re.match(r",\s", rest)
                or re.match(
                    r"\s+(is|has|and|with|satisfies|that|whose|the|for|maps|sends|under|valid|where|taken|traversed|at|of|Find|Determine|Evaluate|Calculate)\b",
                    rest,
                )
            ):
                break
        k += 1
    return text[start:k].strip()


def numeric(expr, **subs) -> complex:
    return complex(sympy.N(sympy.sympify(expr).subs(subs), 20))


def answer_value(q, **subs) -> complex:
    return numeric(parse(q.answer_text), **subs)


def close(a: complex, b: complex, tol: float = 1e-7) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def coverage(checked: int, total: int, floor: float = 0.6) -> None:
    """The oracle read enough of the stems for the test to mean something."""
    assert total and checked / total >= floor, f"oracle read only {checked} of {total} stems"
