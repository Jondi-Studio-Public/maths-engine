"""Independent oracle for the `pr1a_double_angle` questions: it re-derives each answer from the rendered question
text.

Imports NOTHING from `mathgen.generators.pr1a_double_angle_build`: every value here is re-derived from the
rendered TEXT with this file's own small regex/translation and the `math` module (never the generator's own
identity/pool tables).

The generator's free variables are Latin only (`x, t, u, v`) and every printed start/end line is typeable
ASCII (brackets round every function argument, `^` for powers, no `**`/`*`/unicode superscript), so this
file's own translation no longer needs any Greek or bare-function-argument handling -- it only needs to (a)
strip whitespace, (b) insert the implicit multiplication a digit or `)` immediately followed by a letter, `π`
or `(` implies, and (c) map `π` -> `pi`, `^` -> `**`, and the one remaining unicode form this pool ever emits,
`√3` -> `sqrt(3)`, from the apply items' `target_readable` hint.

`N_SEEDS` (default 250 per tier) is overridable with `PR1A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import math
import os
import re

import pytest
import sympy

import mathgen
from mathgen.generators import pr1a_double_angle_build  # noqa: F401  (registers the topic)

TOPIC = "pr1a_double_angle_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("PR1A_ORACLE_SEEDS", "250"))

PROBE_ANGLES = (0.31, 0.77, 1.19, -0.41, -0.89)

_SQRT_RE = re.compile(r"√(\d+)")
_IMPLICIT_MULT_RE = re.compile(r"(?<=[0-9)])(?=[A-Za-zπ(])")


def to_python_expr(readable: str) -> str:
    """This file's own translation from the learner-facing readable maths
    (never the generator's pool/identity tables) to a `math`-evaluable
    python expression string. Every symbol in this pool is Latin-only and
    every function argument is already bracketed, so this is
    just: strip whitespace (purely decorative here), turn `√N` into
    `sqrt(N)`, insert the `*` a digit/`)` immediately followed by a letter,
    `π` or `(` implies (`2sin(` -> `2*sin(`, `)cos(` -> `)*cos(`,
    `5π` -> `5*π`), then map `π` -> `pi` and `^` -> `**`."""
    s = re.sub(r"\s+", "", readable)
    s = _SQRT_RE.sub(r"sqrt(\1)", s)
    s = _IMPLICIT_MULT_RE.sub("*", s)
    s = s.replace("π", "pi")
    s = s.replace("^", "**")
    return s


SAFE_NAMESPACE = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "pi": math.pi,
}


def eval_expr(expr_text: str, var_name: str | None, value: float | None) -> float:
    ns = dict(SAFE_NAMESPACE)
    if var_name is not None:
        ns[var_name] = value
    return eval(expr_text, {"__builtins__": {}}, ns)  # noqa: S307 -- fixed namespace, own translation


# Each pattern's `var` group captures the free variable directly from the prompt's own "for all
# [real] X" / "for every [real] value of X" clause -- no separate variable-guessing heuristic needed
# now every variable is Latin. The optional "where both sides are defined" clause is the
# restricted-domain wording; both variants are matched by the same pattern. No trailing
# `$` anchor: every template now ends with the shared notation sentence, which these patterns simply
# don't capture.
_PROVE_PATTERNS = (
    re.compile(
        r"Prove that (?P<identity>.+?), for all(?: real)? (?P<var>\S+?)(?: where both sides are defined)?\. "
        r"You may cite, as given: (?P<givens>.+?)\. "
        r"Start from the left-hand side, (?P<start>.+?), and finish at (?P<target>.+?)\. "
    ),
    re.compile(
        r"Show that (?P<identity>.+?) for every(?: real)? value of (?P<var>\S+?)(?: where both sides are defined)?\. "
        r"Given results you may use: (?P<givens>.+?)\. "
        r"Begin your proof with (?P<start>.+?) and end at (?P<target>.+?)\. "
    ),
    re.compile(
        r"Using (?P<givens>.+?) as given results, prove the identity (?P<identity>.+?), "
        r"true for all(?: real)? (?P<var>\S+?)(?: where both sides are defined)?\. "
        r"Work from (?P<start>.+?) through to (?P<target>.+?)\. "
    ),
)

_APPLY_PATTERNS = (
    re.compile(r"Evaluate (?P<expr>.+?) \(that is, [\d.]+°\) exactly, by recognising it as a double angle\."),
    re.compile(
        r"Show that (?P<expr>.+?) \(that is, [\d.]+°\) equals (?P<value>.+?) by recognising the double-angle form"
    ),
    re.compile(r"^(?P<expr>.+?) \(that is, [\d.]+°\) is a double-angle expression\."),
)


def _parse_prove(text: str) -> dict:
    for pattern in _PROVE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.groupdict()
    raise AssertionError(f"no PROVE pattern matched: {text!r}")


def _parse_apply(text: str) -> dict:
    for pattern in _APPLY_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.groupdict()
    raise AssertionError(f"no APPLY pattern matched: {text!r}")


def _is_prove(text: str) -> bool:
    return (
        text.startswith("Prove that")
        or text.startswith("Show that")
        and " for " in text.split(".")[0]
        or text.startswith("Using ")
    )


def _check_prove(text: str, start_field: str, target_field: str) -> None:
    parsed = _parse_prove(text)
    var_name = parsed["var"]
    lhs_text, _, rhs_text = parsed["identity"].partition("=")
    assert rhs_text, f"identity has no '=' : {parsed['identity']!r}"
    lhs_expr = to_python_expr(lhs_text.strip())
    rhs_expr = to_python_expr(rhs_text.strip())
    start_expr = to_python_expr(parsed["start"].strip())
    target_expr = to_python_expr(parsed["target"].strip())

    # (c) every given is stated as a formula, never only by name.
    for given in parsed["givens"].split(";"):
        assert "=" in given, f"given not stated as a formula: {given!r}"

    for angle in PROBE_ANGLES:
        lhs_v = eval_expr(lhs_expr, var_name, angle)
        rhs_v = eval_expr(rhs_expr, var_name, angle)
        assert math.isclose(lhs_v, rhs_v, abs_tol=1e-6), (text, angle, lhs_v, rhs_v)
        start_v = eval_expr(start_expr, var_name, angle)
        target_v = eval_expr(target_expr, var_name, angle)
        assert math.isclose(start_v, target_v, abs_tol=1e-6), (text, angle, start_v, target_v)
        # (b) the text's stated start/end match the identity's own LHS/RHS.
        assert math.isclose(lhs_v, start_v, abs_tol=1e-6), (text, angle, lhs_v, start_v)
        assert math.isclose(rhs_v, target_v, abs_tol=1e-6), (text, angle, rhs_v, target_v)

    # (b) cross-check against the server-sent `start`/`target` fields
    # directly (sympy, not this module's own translation): same value.
    server_start = sympy.sympify(start_field)
    server_target = sympy.sympify(target_field)
    free = sorted(server_start.free_symbols | server_target.free_symbols, key=str)
    assert len(free) <= 1
    for angle in PROBE_ANGLES:
        subs = {s: angle for s in free}
        s_val = complex(server_start.subs(subs).evalf(30))
        t_val = complex(server_target.subs(subs).evalf(30))
        start_v = eval_expr(start_expr, var_name, angle)
        target_v = eval_expr(target_expr, var_name, angle)
        assert abs(s_val.real - start_v) < 1e-6, (text, angle, s_val, start_v)
        assert abs(t_val.real - target_v) < 1e-6, (text, angle, t_val, target_v)


def _check_apply(text: str, start_field: str, target_field: str) -> None:
    parsed = _parse_apply(text)
    expr_text = to_python_expr(parsed["expr"].strip())
    computed = eval_expr(expr_text, None, None)

    # The angle is stated in radians (never degrees alone) -- the
    # captured expr itself must contain the radian symbol.
    assert "π" in parsed["expr"], text

    server_start = sympy.sympify(start_field)
    server_target = sympy.sympify(target_field)
    assert not server_start.free_symbols and not server_target.free_symbols
    start_v = float(server_start.evalf(30))
    target_v = float(server_target.evalf(30))
    assert math.isclose(computed, start_v, abs_tol=1e-9), (text, computed, start_v)
    assert math.isclose(start_v, target_v, abs_tol=1e-9), (text, start_v, target_v)

    if "value" in parsed and parsed["value"]:
        value_expr = to_python_expr(parsed["value"].strip())
        stated_v = eval_expr(value_expr, None, None)
        assert math.isclose(stated_v, target_v, abs_tol=1e-9), (text, stated_v, target_v)


def _rubric_question_fields(q):
    """`text` and the marking dict's own `start`/`target` -- never `reference_lines`, `key_steps` or `traps`."""
    marking = q.answer
    return q.text, marking["start"], marking["target"]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_item_is_true_and_matches_start_target(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = mathgen.make(TOPIC, seed=seed, difficulty=tier)
        text, start_field, target_field = _rubric_question_fields(q)
        if _is_prove(text):
            _check_prove(text, start_field, target_field)
        else:
            _check_apply(text, start_field, target_field)
        checked += 1
    assert checked == N_SEEDS


def test_oracle_is_red_first_a_corrupted_target_is_caught():
    """Prove the oracle actually checks something: corrupt a rendered
    item's `target` field (as `rubric_practice_question` would send a
    generator's real output) and confirm the oracle's own cross-check
    fails."""
    q = mathgen.make(TOPIC, seed=2, difficulty="easy")  # a PROVE item (seed 2 above)
    text, start_field, target_field = _rubric_question_fields(q)
    assert _is_prove(text)
    corrupted_target = str(sympy.sympify(target_field) + 1)
    with pytest.raises(AssertionError):
        _check_prove(text, start_field, corrupted_target)


if __name__ == "__main__":
    for tier in TIERS:
        for seed in range(3):
            q = mathgen.make(TOPIC, seed=seed, difficulty=tier)
            text, start_field, target_field = _rubric_question_fields(q)
            print(tier, seed, text, "|| start=", start_field, "target=", target_field)
