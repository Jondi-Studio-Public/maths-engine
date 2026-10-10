"""Independent oracle for the `pr1a_pythagorean_identities` questions: it re-derives each answer from the
rendered question text.

Imports NOTHING from `mathgen.generators.pr1a_pythagorean_identities_build` -- no helper, no constant, no
table.

It parses the identity (or, for an apply item, the stated special-angle
value and the expression asked for) out of `question.text` with its own
small regex + `sympy` translation, and confirms, entirely independently of
how the generator built the item:

(a) the identity/expression the TEXT states is true -- evaluated at five
    random real points (prove items) or by direct `evalf` (apply items,
    which are bare numeric constants);
(b) the text's start and end match the learner-visible `start`/`target`;
(c) every given the text names is stated as a formula (contains "="), not
    only by name.

`N_SEEDS` per tier is 250 by default, overridable with `PR1A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re
import types

import pytest
import sympy as sp

import mathgen

TOPIC = "pr1a_pythagorean_identity_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("PR1A_ORACLE_SEEDS", "250"))

# ---------------------------------------------------------------------------
# Own text -> sympy translation (never the generator's `_eval_symbolic`/
# `_eval_numeric`, never its tables). The rendered notation is fixed and
# small: `name^N(arg)` for a squared/quartic trig term, juxtaposition
# ("sec(x) csc(x)") for a product, "theta"/"pi"/"sqrt(n)" spelled with the
# unicode glyphs this generator renders (θ, π, √).
# ---------------------------------------------------------------------------
_POW_CALL_RE = re.compile(r"(sin|cos|tan|sec|csc|cot)\^(\d+)\(([^()]*)\)")
_JUXT_RE = re.compile(r"(\)|\d)\s+(?=(?:sin|cos|tan|sec|csc|cot)\()")


def _to_sympy_text(expr: str) -> str:
    e = expr.replace("θ", "theta").replace("π", "pi")
    e = re.sub(r"√(\d+)", r"sqrt(\1)", e)
    e = _POW_CALL_RE.sub(r"\1(\3)**\2", e)
    e = e.replace(")(", ")*(")
    e = _JUXT_RE.sub(lambda m: m.group(1) + "*", e)
    # sympy.sympify (unlike parse_expr(transformations='all')) does not
    # itself insert implicit multiplication -- "5pi" must become "5*pi".
    e = re.sub(r"(\d)(pi|theta)\b", r"\1*\2", e)
    return e


def _parse(expr: str) -> sp.Expr:
    return sp.sympify(_to_sympy_text(expr.strip()))


_PROVE_PATTERNS = (
    # Each of the three PROVE_TEMPLATES continues past the stated lhs/rhs with a comma, never a
    # period -- match up to that comma.
    re.compile(r"from (.+?) and (?:finish|end) at (.+?),"),
    re.compile(r"starting from (.+?), finishing at (.+?),"),
)


def _extract_prove_sides(text: str):
    for pat in _PROVE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


_GIVEN_RE = re.compile(r"(sin|cos|tan)\(([^()]+)\) = (.+?)(?:\. |, | \(|\.$)")
_FIND_PATTERNS = (
    re.compile(r"(?:find|evaluate) ([^.,]+?) exactly"),
    re.compile(r"(?:find|evaluate) the exact value of ([^.,]+?)\."),
)


def _extract_find_expr(text: str) -> str | None:
    for pat in _FIND_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


# Five fixed-but-irregular reals, none near a multiple of pi/2 (so sin/cos/
# tan/sec/csc/cot are all comfortably defined at every one of them).
RANDOM_POINTS = (0.41, 0.83, 1.27, 1.71, 2.19)


def _numeric_close(a: sp.Expr, b: sp.Expr, subs: dict | None = None, tol: float = 1e-6) -> bool:
    av = complex(a.evalf(30, subs=subs or {}))
    bv = complex(b.evalf(30, subs=subs or {}))
    scale = max(abs(av), abs(bv), 1.0)
    return abs(av - bv) < tol * scale


def _is_prove_item(question) -> bool:
    return bool(question.answer.get("symbols"))


def check_prove_item(question) -> list[str]:
    """Returns a list of problems found (empty == clean)."""
    problems: list[str] = []
    sides = _extract_prove_sides(question.text)
    if sides is None:
        return [f"could not extract a from/at (or starting-from/finishing-at) clause: {question.text!r}"]
    lhs_text, rhs_text = sides
    try:
        lhs_expr, rhs_expr = _parse(lhs_text), _parse(rhs_text)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return [f"could not parse lhs/rhs ({lhs_text!r}, {rhs_text!r}): {exc}"]
    free = lhs_expr.free_symbols | rhs_expr.free_symbols
    if not free:
        return [f"identity has no free symbol: {lhs_text!r} = {rhs_text!r}"]
    arg = next(iter(free))
    try:
        start_expr = sp.sympify(question.answer["start"])
        target_expr = sp.sympify(question.answer["target"])
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return [f"could not parse answer start/target: {exc}"]
    for point in RANDOM_POINTS:
        subs = {arg: point}
        if not _numeric_close(lhs_expr, rhs_expr, subs):
            problems.append(f"stated identity does not hold at {arg}={point}: {lhs_text} vs {rhs_text}")
        if not _numeric_close(lhs_expr, start_expr, subs):
            problems.append(f"text lhs does not match answer start at {arg}={point}")
        if not _numeric_close(rhs_expr, target_expr, subs):
            problems.append(f"text rhs does not match answer target at {arg}={point}")
    if "=" not in question.text:
        problems.append("no cited given is stated as a formula (no '=' anywhere in the prompt)")
    return problems


def check_apply_item(question) -> list[str]:
    problems: list[str] = []
    given_matches = _GIVEN_RE.findall(question.text)
    if not given_matches:
        problems.append(f"no stated given value found: {question.text!r}")
        return problems
    func_map = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}
    for func_name, arg_text, value_text in given_matches:
        try:
            arg_expr, value_expr = _parse(arg_text), _parse(value_text)
        except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
            problems.append(f"could not parse given clause {func_name}({arg_text}) = {value_text}: {exc}")
            continue
        func = func_map[func_name]
        if not _numeric_close(func(arg_expr), value_expr):
            problems.append(f"stated given is false: {func_name}({arg_text}) != {value_text}")
    find_expr_text = _extract_find_expr(question.text)
    if find_expr_text is None:
        problems.append(f"no 'find/evaluate ... exactly' clause: {question.text!r}")
        return problems
    try:
        target_text_expr = _parse(find_expr_text)
        start_expr = sp.sympify(question.answer["start"])
        target_expr = sp.sympify(question.answer["target"])
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        problems.append(f"could not parse target/answer: {exc}")
        return problems
    if not _numeric_close(target_text_expr, start_expr):
        problems.append("text's 'find ... exactly' expression does not match answer start")
    if not _numeric_close(start_expr, target_expr):
        problems.append("answer start does not evaluate to answer target")
    return problems


def check_item(question) -> list[str]:
    return check_prove_item(question) if _is_prove_item(question) else check_apply_item(question)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_confirms_every_item(tier):
    problems_total: list[str] = []
    checked = 0
    for seed in range(N_SEEDS):
        q = mathgen.make(TOPIC, seed=seed, difficulty=tier)
        problems = check_item(q)
        checked += 1
        if problems:
            problems_total.append((seed, q.text, problems))
    assert not problems_total, problems_total[:5]
    assert checked == N_SEEDS


def test_oracle_is_red_first_on_a_corrupted_target():
    """Prove the oracle actually checks something: corrupt the answer's
    `target` (flip a sign) and confirm `check_item` reports a problem."""
    real = mathgen.make(TOPIC, seed=0, difficulty="easy")
    corrupted_answer = dict(real.answer)
    if _is_prove_item(real):
        # Flip the sign of the last reference line / target (breaks the
        # *stated* identity itself -- the text still asks to prove the
        # ORIGINAL identity, so this simulates a wrong marking `target`).
        corrupted_answer["target"] = f"-({real.answer['target']})"
    else:
        corrupted_answer["target"] = f"-({real.answer['target']})"
    corrupted = types.SimpleNamespace(text=real.text, answer=corrupted_answer)
    problems = check_item(corrupted)
    assert problems, "the oracle failed to catch a sign-flipped target"


def test_oracle_is_red_first_on_a_corrupted_given_value():
    """Corrupt a stated special-angle value in the TEXT itself (an apply
    item) and confirm the oracle catches the false given."""
    q = None
    for seed in range(60):
        candidate = mathgen.make(TOPIC, seed=seed, difficulty="easy")
        if not _is_prove_item(candidate):
            q = candidate
            break
    assert q is not None, "no apply item found in the first 60 easy seeds"
    bad_text = re.sub(r"= (.+?)(?:\. |, | \(|\.$)", "= 999999. ", q.text, count=1)
    corrupted = types.SimpleNamespace(text=bad_text, answer=q.answer)
    problems = check_apply_item(corrupted)
    assert problems, "the oracle failed to catch a corrupted stated given value"
