"""Independent oracle for the `pr1a_product_to_sum` questions: it re-derives each answer from the rendered
question text.

Imports NOTHING from `mathgen.generators.pr1a_product_to_sum_build`: it reads only `question.text` and the
learner-visible `start`/`target`, parses the identity/expression the TEXT states with its own small regex +
translator, and checks it independently with the `math` module.

Three of the four checks
render" list, made mechanical, PLUS a fourth for the apply-style items
(medium/hard's numeric-instance pool and exam's special-angle pool), which
never state the target in the prompt (the learner must derive it -- the
same choice the easy-tier apply items made):
  (a) PROVE items: the identity the TEXT states is true (5 random real
      points); apply items: every numeric equation named in the "given"
      clause is arithmetically true;
  (b) the text's start (and, for prove items, end) match the learner-
      visible start/target;
  (c) every given the text names is stated as a formula (contains "="),
      not only by name.
"""

from __future__ import annotations

import math
import os
import re

import pytest

from mathgen.core import make

TOPIC = "pr1a_product_to_sum_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("PR1A_ORACLE_SEEDS", "250"))

_EVAL_NS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "sqrt": math.sqrt}

# Fixed-but-irregular probe points, clear of every singularity this pool's
# identities can divide by (cos(x), cos(2x), cos(4x); checked directly).
_PROBE_AB = ((0.4, 0.7), (1.0, -0.5), (2.3, 0.9), (0.15, 1.3), (-0.8, 0.55))
_PROBE_X = (0.4, 1.1, 2.3, 0.7, 1.9)

_RESERVED_NAMES = {"sin", "cos", "tan", "pi", "sqrt"}


def _translate(expr: str) -> str:
    """Readable, fully-bracketed trig text -> a python-evaluable
    expression. This skill's prompts never write an unbracketed function,
    so the only translation needed is implicit multiplication:
    a digit run glued to a letter (`7pi`, `2x`), a bare `sqrt6`-style run,
    and juxtaposed brackets/functions (`cos(2x)cos(5x)`)."""
    s = expr.strip()
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"(\d)(?=[A-Za-z])", r"\1*", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\)(?=(sin|cos|tan)\(|\()", ")*", s)
        s = re.sub(r"(\)|[A-Za-z0-9])\s+(?=(sin|cos|tan)\(|[A-Za-z]\b|\()", r"\1*", s)
    return s


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
]

# --- apply-item text patterns (symbolic, no target quoted) ------------------

_APPLY_PATTERNS = [
    re.compile(r"^Express (?P<start>.+?) as a sum or difference, using (?P<given>.+?)\.(?:\s.*)?$"),
    re.compile(
        r"^Write (?P<start>.+?) as a sum or difference of two terms, citing (?P<given>.+?) "
        r"at the step where it is used\.(?:\s.*)?$"
    ),
    re.compile(r"^Show how (?P<start>.+?) can be rewritten as a sum or difference, using (?P<given>.+?)\.(?:\s.*)?$"),
]

# --- exam apply-item text patterns (numeric, no target quoted) -------------

_EXAM_APPLY_PATTERNS = [
    re.compile(r"^Find the exact value of (?P<start>.+?)\. Use (?P<given>.+?)\.(?:\s.*)?$"),
    re.compile(r"^Evaluate (?P<start>.+?) exactly, citing (?P<given>.+?) at the step where it is used\.(?:\s.*)?$"),
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
    assert len(names) in (1, 2), (text, "expected 1 or 2 free symbols", names)

    for a, b in _PROBE_AB:
        if len(names) == 1:
            values = {names[0]: a}
        else:
            values = {names[0]: a, names[1]: b}
        try:
            lhs_val, rhs_val = _eval_at(lhs_expr, values), _eval_at(rhs_expr, values)
        except ZeroDivisionError:
            continue
        assert _isclose(lhs_val, rhs_val), (text, values, lhs_val, rhs_val)  # (a) identity is true

        start_val, target_val = _eval_at(start_expr, values), _eval_at(target_expr, values)
        assert _isclose(start_val, lhs_val), (text, "start != lhs", start_val, lhs_val)
        assert _isclose(target_val, rhs_val), (text, "target != rhs", target_val, rhs_val)

        learner_start_val = _eval_at(learner_start_expr, values)
        learner_target_val = _eval_at(learner_target_expr, values)
        assert _isclose(learner_start_val, start_val), (text, "learner start mismatch")  # (b)
        assert _isclose(learner_target_val, target_val), (text, "learner target mismatch")  # (b)

    assert "=" in given, f"given cites a result without stating it as a formula: {given!r}"  # (c)


def _check_apply_item(text: str, start_learner: str):
    """Medium/hard(ii) apply items: only the START is quoted (the learner
    must derive the sum/difference form, never told it -- the same design
    the easy-tier apply items used). Checks (b) the quoted start
    matches the learner-visible start, and (c) the given identity is a
    formula and is itself true at random points."""
    m = _match_any(_APPLY_PATTERNS, text)
    assert m is not None, f"apply text matched no known template shape: {text!r}"
    start_text = m.group("start")
    given = m.group("given")

    assert "=" in given, f"given cites a result without stating it as a formula: {given!r}"  # (c)
    given_lhs, given_rhs = given.split(" = ", 1)
    given_names = _free_symbol_names(given_lhs + " " + given_rhs)
    assert len(given_names) == 2, (text, given_names)
    for a, b in _PROBE_AB:
        values = {given_names[0]: a, given_names[1]: b}
        lhs_val = _eval_at(_translate(given_lhs), values)
        rhs_val = _eval_at(_translate(given_rhs), values)
        assert _isclose(lhs_val, rhs_val), (text, "given identity is false", values)  # (a)

    start_expr = _translate(start_text)
    learner_start_expr = _translate(start_learner)
    names = _free_symbol_names(start_learner)
    assert len(names) == 1, (text, "expected exactly 1 free symbol", names)
    for x in _PROBE_X:
        start_val = _eval_at(start_expr, {names[0]: x})
        learner_start_val = _eval_at(learner_start_expr, {names[0]: x})
        assert _isclose(start_val, learner_start_val), (text, "learner start mismatch")  # (b)


def _check_exam_apply_item(text: str, start_learner: str, target_learner: str):
    """Exam special-angle apply items: numeric, no free symbols. The
    target is not quoted in the prompt, but its correctness is checked
    directly (the learner-visible start and target must agree)."""
    m = _match_any(_EXAM_APPLY_PATTERNS, text)
    assert m is not None, f"exam apply text matched no known template shape: {text!r}"
    start_text = m.group("start")
    given = m.group("given")

    assert "=" in given, f"given cites a result without stating it as a formula: {given!r}"  # (c)
    for clause in given.split(", "):
        if "=" not in clause:
            continue
        lhs, rhs = clause.split("=", 1)
        lhs, rhs = _translate(lhs), _translate(rhs)
        try:
            lhs_val, rhs_val = _eval_at(lhs, {}), _eval_at(rhs, {})
        except (SyntaxError, NameError, ZeroDivisionError):
            continue
        assert _isclose(lhs_val, rhs_val), (text, "given clause is false", clause)  # (a)

    start_expr = _translate(start_text)
    learner_start_expr = _translate(start_learner)
    learner_target_expr = _translate(target_learner)
    start_val = _eval_at(start_expr, {})
    learner_start_val = _eval_at(learner_start_expr, {})
    learner_target_val = _eval_at(learner_target_expr, {})
    assert _isclose(start_val, learner_start_val), (text, "learner start mismatch")  # (b)
    assert _isclose(learner_start_val, learner_target_val), (text, "stated exact value is wrong")  # (a)


def _check_one(q) -> None:
    marking = q.answer
    start_learner, target_learner = marking["start"], marking["target"]
    if _match_any(_PROVE_PATTERNS, q.text) is not None:
        _check_prove_item(q.text, start_learner, target_learner)
    elif marking["symbols"]:
        _check_apply_item(q.text, start_learner)
    else:
        _check_exam_apply_item(q.text, start_learner, target_learner)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_agrees_with_every_rendered_item(tier):
    checked = 0
    for s in range(N_SEEDS):
        q = make(TOPIC, seed=s, difficulty=tier)
        _check_one(q)
        checked += 1
    assert checked == N_SEEDS


# --- red-first: the oracle must actually catch a corrupted target ----------


def test_oracle_catches_a_corrupted_prove_target():
    q = make(TOPIC, seed=0, difficulty="easy")
    assert q.answer["symbols"]
    bad_text = q.text  # text still states the TRUE target -- exactly what should be caught
    with pytest.raises(AssertionError):
        _check_prove_item(bad_text, q.answer["start"], "cos(A)*cos(B)")  # wrong target


def test_oracle_catches_a_corrupted_apply_start():
    q = None
    for s in range(30):
        candidate = make(TOPIC, seed=s, difficulty="medium")
        if _match_any(_APPLY_PATTERNS, candidate.text) is not None:
            q = candidate
            break
    assert q is not None, "no medium apply item found in the first 30 seeds"
    with pytest.raises(AssertionError):
        _check_apply_item(q.text, "0")  # start replaced with a wrong value


def test_oracle_catches_a_corrupted_exam_apply_value():
    q = None
    for s in range(30):
        candidate = make(TOPIC, seed=s, difficulty="exam")
        if not candidate.answer["symbols"]:
            q = candidate
            break
    assert q is not None, "no exam apply item found in the first 30 seeds"
    with pytest.raises(AssertionError):
        _check_exam_apply_item(q.text, q.answer["start"], "0")  # target replaced with a wrong value


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
