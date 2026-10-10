"""Independent oracle for the `pr1a_angle_sum_difference` questions: it re-derives each answer from the rendered
question text.

Imports NOTHING from `mathgen.generators.pr1a_angle_sum_difference_build`: it reads only `question.text` and
the learner-visible `start`/`target`, parses the identity/expression the TEXT states with its own small regex
+ translator, and checks it independently with the `math` module.

Four checks
list, made mechanical:
  (a) the identity/expression the TEXT states is true (5 random real points,
      or the stated exact value for an apply item);
  (b) the text's start and end match the learner-visible start/target;
  (c) every given the text names is stated as a formula (contains "="), not
      only by name;
  (d) (documented, not re-checked here -- the CAS-confirmability table
      already covers "correct proof the chain rejects" in the build test).
"""

from __future__ import annotations

import math
import os
import re

import pytest

from mathgen.core import make

TOPIC = "pr1a_angle_sum_difference_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("PR1A_ORACLE_SEEDS", "250"))

_EVAL_NS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "sqrt": math.sqrt}

# Fixed-but-irregular probe points for A, B (radians) -- clear of every
# singularity this pool's identities can divide by (checked directly, see
# the generator's own `_PROBE_AB`; chosen independently here, not imported).
_PROBE_AB = ((0.4, 0.7), (1.0, -0.5), (2.3, 0.9), (0.15, 1.3), (-0.8, 0.55))


def _translate(expr: str) -> str:
    """Readable trig text -> a python-evaluable expression. Handles exactly
    the shapes this skill's own prompts use: `sin A`/`sin^2 A` juxtaposition,
    implicit multiplication between adjacent terms, and `sqrt6`/`5pi`-style
    bare numeric-letter runs. Written without reading the generator's
    tables."""
    s = expr.strip()
    s = re.sub(r"sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"(\d)(?=[A-Za-z])", r"\1*", s)
    s = re.sub(r"(sin|cos|tan)\^2\s*([A-Za-z])(?!\()", r"\1(\2)**2", s)
    s = re.sub(r"(sin|cos|tan)\s+([A-Za-z])(?!\()", r"\1(\2)", s)
    s = re.sub(r"\)\^(\d+)", r")**\1", s)  # cos(A)^2 -> cos(A)**2 (bracket form)
    prev = None
    while prev != s:
        prev = s
        # A closing bracket immediately followed (NO space) by another
        # function call or `(` -- the rule-11 notation `cos(A)cos(B)` -- is
        # always implicit multiplication. Anchored on `)` (never part of an
        # identifier), so this can never split a bare word like `sqrt`.
        s = re.sub(r"\)(?=(sin|cos|tan)\(|\()", ")*", s)
        # Otherwise only insert `*` across an explicit space -- never
        # inside a bare identifier, which has no spaces.
        s = re.sub(r"(\)|[A-Za-z0-9])\s+(?=(sin|cos|tan)\(|[A-Za-z]\b|\()", r"\1*", s)
    return s


# The oracle never imports the generator's pair list; it recovers the two symbol names actually used
# by scanning the learner-visible start/target text itself (reads only what the learner is sent).
_RESERVED_NAMES = {"sin", "cos", "tan", "pi", "sqrt"}


def _free_symbol_names(expr: str) -> list[str]:
    names: list[str] = []
    for tok in re.findall(r"[A-Za-z_]\w*", expr):
        if tok not in _RESERVED_NAMES and tok not in names:
            names.append(tok)
    return names


def _eval_at(expr: str, values: dict) -> float:
    ns = dict(_EVAL_NS, **values)
    return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 -- oracle's own fixed strings only


def _isclose(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-7)


# --- prove-item text patterns (the 4 template shapes) -----------------------

_PROVE_PATTERNS = [
    re.compile(
        r"^Prove that (?P<identity>.+?)\. You are given (?P<given>.+?)\. "
        r"Start from the left-hand side, (?P<start>.+?), and finish at (?P<target>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Using (?P<given>.+?), prove that (?P<identity>.+?)\. "
        r"Begin your proof at (?P<start>.+?) and end at (?P<target>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Show that (?P<identity>.+?) follows from (?P<given>.+?)\. "
        r"Write a chain of equalities starting at (?P<start>.+?) and ending at (?P<target>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Give a complete proof of (?P<identity>.+?), citing (?P<given>.+?) "
        r"at the step where each is used\. Start at (?P<start>.+?); end at (?P<target>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Establish that (?P<identity>.+?) by proving it from (?P<given>.+?)\. "
        r"The chain must run from (?P<start>.+?) to (?P<target>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Demonstrate that (?P<identity>.+?), reasoning from (?P<given>.+?), "
        r"with your working beginning at (?P<start>.+?) and concluding at (?P<target>.+?)\.(?:\s.*)?$"
    ),
]

# --- apply-item text patterns (the 3 template shapes) -----------------------

_APPLY_PATTERNS = [
    re.compile(
        r"^Find the exact value of (?P<start>.+?) \((?P<deg>\d+) degrees\)\. "
        r"Write the angle as a sum or difference of standard angles and use the given "
        r"sum/difference formula: (?P<given>.+?)\.(?:\s.*)?$"
    ),
    re.compile(
        r"^Evaluate (?P<start>.+?) exactly by writing the angle as a sum or difference of "
        r"standard angles \((?P<given>.+?)\)\.(?:\s.*)?$"
    ),
    re.compile(r"^Give the exact value of (?P<start>.+?), using (?P<given>.+?)\.(?:\s.*)?$"),
]


def _match_any(patterns, text):
    for pattern in patterns:
        m = pattern.match(text)
        if m:
            return m
    return None


def _check_prove_item(text: str, start_learner: str, target_learner: str):
    m = _match_any(_PROVE_PATTERNS, text)
    assert m is not None, f"prove text matched no known template shape: {text!r}"
    identity = m.group("identity")
    given = m.group("given")
    start_text = m.group("start")
    target_text = m.group("target")

    assert " = " in identity, identity
    lhs_text, rhs_text = identity.split(" = ", 1)

    lhs_expr, rhs_expr = _translate(lhs_text), _translate(rhs_text)
    start_expr, target_expr = _translate(start_text), _translate(target_text)
    learner_start_expr, learner_target_expr = _translate(start_learner), _translate(target_learner)

    names = _free_symbol_names(start_learner + " " + target_learner)
    assert len(names) == 2, (text, "expected exactly 2 free symbols", names)
    l1, l2 = names

    for a, b in _PROBE_AB:
        values = {l1: a, l2: b}
        lhs_val, rhs_val = _eval_at(lhs_expr, values), _eval_at(rhs_expr, values)
        assert _isclose(lhs_val, rhs_val), (text, values, lhs_val, rhs_val)  # (a) identity is true

        start_val, target_val = _eval_at(start_expr, values), _eval_at(target_expr, values)
        assert _isclose(start_val, lhs_val), (text, "start != lhs", start_val, lhs_val)
        assert _isclose(target_val, rhs_val), (text, "target != rhs", target_val, rhs_val)

        learner_start_val = _eval_at(learner_start_expr, values)
        learner_target_val = _eval_at(learner_target_expr, values)
        assert _isclose(learner_start_val, start_val), (text, "learner start mismatch")  # (b)
        assert _isclose(learner_target_val, target_val), (text, "learner target mismatch")  # (b)

    assert "=" in given, f"given cites a result without stating it as a formula: {given!r}"  # (c)


def _check_apply_item(text: str, start_learner: str, target_learner: str):
    m = _match_any(_APPLY_PATTERNS, text)
    assert m is not None, f"apply text matched no known template shape: {text!r}"
    start_text = m.group("start")
    given = m.group("given")

    func = "sin" if start_text.startswith("sin") else "cos"
    assert start_text.startswith(f"{func}(")

    start_expr = _translate(start_text)
    learner_start_expr = _translate(start_learner)
    learner_target_expr = _translate(target_learner)

    stated_value = _eval_at(start_expr, {})  # no free symbols in an apply item
    learner_start_value = _eval_at(learner_start_expr, {})
    learner_target_value = _eval_at(learner_target_expr, {})

    assert _isclose(stated_value, learner_start_value), (text, "learner start mismatch")  # (b)
    # (a): the stated angle really does have this exact value -- recomputed
    # via math.sin/cos of the same angle, independent of the marking dict.
    assert _isclose(learner_start_value, learner_target_value), (text, "stated value is wrong")

    if "deg" in m.groupdict() and m.group("deg"):
        deg_val = math.radians(int(m.group("deg")))
        deg_value = math.sin(deg_val) if func == "sin" else math.cos(deg_val)
        assert _isclose(deg_value, learner_target_value), (text, "degree value disagrees")

    assert "=" in given, f"given cites a formula without stating it: {given!r}"  # (c)


def _check_one(q) -> None:
    marking = q.answer
    start_learner, target_learner = marking["start"], marking["target"]
    if marking["symbols"]:
        _check_prove_item(q.text, start_learner, target_learner)
    else:
        _check_apply_item(q.text, start_learner, target_learner)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_agrees_with_every_rendered_item(tier):
    checked = 0
    for s in range(N_SEEDS):
        q = make(TOPIC, seed=s, difficulty=tier)
        _check_one(q)
        checked += 1
    assert checked == N_SEEDS


# --- red-first: the oracle must actually catch a corrupted target ----------


def test_oracle_catches_a_corrupted_prove_target(monkeypatch):
    q = make(TOPIC, seed=0, difficulty="easy")
    assert q.answer["symbols"]  # the seed-0 easy item is the cos_sum prove item
    bad_marking = dict(q.answer)
    bad_marking["target"] = "cos(A)*cos(B) + sin(A)*sin(B)"  # sign-flipped, no longer the true target
    bad_text = q.text  # text still states the TRUE target -- exactly what should be caught
    with pytest.raises(AssertionError):
        _check_prove_item(bad_text, bad_marking["start"], bad_marking["target"])


def test_oracle_catches_a_corrupted_apply_target():
    q = None
    for s in range(30):
        candidate = make(TOPIC, seed=s, difficulty="easy")
        if not candidate.answer["symbols"]:
            q = candidate
            break
    assert q is not None, "no apply-tier item found in the first 30 easy seeds"
    with pytest.raises(AssertionError):
        _check_apply_item(q.text, q.answer["start"], "0")  # target replaced with a wrong value


if __name__ == "__main__":
    # Marking dump helper -- not collected by
    # pytest as a test.
    for tier in TIERS:
        print(f"=== {tier} ===")
        for s in range(3):
            q = make(TOPIC, seed=s, difficulty=tier)
            print(q.text)
            print(q.answer)
            print()
