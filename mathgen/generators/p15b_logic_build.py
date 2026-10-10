"""Generates a `logic-computability-and-information` question: classify a supplied propositional string under a
given interpretation as ill-formed, true, false or derivable.

Sits in `logic-computability-and-information`, prerequisite
`propositional-truth-evaluation`. The skill's contract: "For a supplied
formal expression and interpretation, identify whether the statement is
well-formed, true or merely derivable." This widens to the genuine 3-4-way
kernel the words point at: given a supplied propositional string and
interpretation, classify it as (A) ill-formed, (B) well-formed and true,
(C) well-formed but false, or (D) syntactically derivable via a stated
inference rule from stated premises, yet false under the interpretation --
the point being that syntactic derivability (a fact about the SHAPE of the
premises, matching an inference-rule schema) is independent of whether the
conclusion happens to be true under any particular interpretation.

To keep the presence of a derivation from being a free tell for option D,
derivation context is also attached to some TRUE-valued draws (still
correctly classified B, "well-formed and true"), so a learner must actually
evaluate the formula rather than pattern-match on "premises means D".
"""

from __future__ import annotations

import random
import re

from ..core import check, generator

_NAMES = ["p", "q", "r", "s"]
_PREMISE_ATOM = "a"
_INTERMEDIATE_ATOM = "b"

# Mutually exclusive by their OWN wording, so exactly one is ever true of a given draw. Fixed by
# making the false-and-not-derivable option say so explicitly ("with no derivation... shown"), which
# is false whenever a derivation IS shown (the derivable_false class), and true whenever one is not
# (the plain false class) -- these two presentation shapes are exhaustive and disjoint by
# construction.
OPTIONS = [
    "Ill-formed (not a valid formula).",
    "Well-formed and true under the interpretation.",
    "Well-formed and false under the interpretation, with no derivation from stated premises shown.",
    "Well-formed and false under the interpretation, but derivable via modus ponens from stated premises shown alongside it.",
]


# ---------------------------------------------------------------------------
# Formula trees: ("atom", name) | ("not", t) | ("and"|"or"|"implies", l, r)


def _eval(tree, values: dict[str, bool]) -> bool:
    kind = tree[0]
    if kind == "atom":
        return values[tree[1]]
    if kind == "not":
        return not _eval(tree[1], values)
    left, right = _eval(tree[1], values), _eval(tree[2], values)
    if kind == "and":
        return left and right
    if kind == "or":
        return left or right
    if kind == "implies":
        return (not left) or right
    raise ValueError(kind)


def _render(tree) -> str:
    kind = tree[0]
    if kind == "atom":
        return tree[1]
    if kind == "not":
        return f"¬{_render(tree[1])}"
    symbol = {"and": "∧", "or": "∨", "implies": "→"}[kind]
    return f"({_render(tree[1])} {symbol} {_render(tree[2])})"


def _render_latex(tree) -> str:
    kind = tree[0]
    if kind == "atom":
        return tree[1]
    if kind == "not":
        return rf"\neg {_render_latex(tree[1])}"
    symbol = {"and": r"\land", "or": r"\lor", "implies": r"\to"}[kind]
    return f"({_render_latex(tree[1])} {symbol} {_render_latex(tree[2])})"


def _python_expr(tree) -> str:
    kind = tree[0]
    if kind == "atom":
        return tree[1]
    if kind == "not":
        return f"(not {_python_expr(tree[1])})"
    left, right = _python_expr(tree[1]), _python_expr(tree[2])
    if kind == "and":
        return f"({left} and {right})"
    if kind == "or":
        return f"({left} or {right})"
    if kind == "implies":
        return f"((not {left}) or {right})"
    raise ValueError(kind)


def _draw_tree(rng: random.Random, atoms: list, complexity: int):
    if complexity <= 1:
        return rng.choice(
            [
                ("not", atoms[0]),
                ("and", atoms[0], atoms[1]),
                ("or", atoms[0], atoms[1]),
                ("implies", atoms[0], atoms[1]),
            ]
        )
    if complexity == 2:
        return rng.choice(
            [
                ("and", ("not", atoms[0]), ("or", atoms[1], atoms[2])),
                ("or", ("and", atoms[0], atoms[1]), ("not", atoms[2])),
                ("implies", ("or", atoms[0], atoms[1]), atoms[2]),
                ("and", ("implies", atoms[0], atoms[1]), ("not", atoms[2])),
            ]
        )
    return rng.choice(
        [
            ("and", ("or", ("not", atoms[0]), atoms[1]), ("implies", atoms[2], atoms[3])),
            ("implies", ("and", atoms[0], ("not", atoms[1])), ("or", atoms[2], atoms[3])),
            ("or", ("implies", atoms[0], atoms[1]), ("and", ("not", atoms[2]), atoms[3])),
        ]
    )


def _n_atoms(complexity: int) -> int:
    return 2 if complexity <= 1 else (3 if complexity == 2 else 4)


def _build_formula(rng: random.Random, complexity: int, target_bool: bool | None):
    """Construct-to-target: draw a random tree and assignment, evaluate
    INDEPENDENTLY through a fresh `eval()` of the tree's own Python
    expression, and keep only a draw whose (re-verified) value matches
    `target_bool` (or any value, when `target_bool` is None)."""
    n_atoms = _n_atoms(complexity)
    names = _NAMES[:n_atoms]
    for _attempt in range(500):
        values = {name: bool(rng.getrandbits(1)) for name in names}
        atoms = [("atom", name) for name in names]
        tree = _draw_tree(rng, atoms, complexity)
        value = _eval(tree, values)
        alt = bool(eval(_python_expr(tree), {"__builtins__": {}}, values))
        check(value == alt, "syntactic-semantic-separation: formula evaluation mismatch")
        if target_bool is None or value == target_bool:
            return tree, values, value
    check(False, "syntactic-semantic-separation: could not construct a formula matching the target truth value")


# ---------------------------------------------------------------------------
# Well-formedness: a from-scratch recursive-descent check over the grammar
# Formula := Atom | "not" Formula | "(" Formula Connective Formula ")"
# used BOTH to prove a genuine WFF renders as well-formed and to prove a
# deliberately mutated string genuinely fails to parse.

_TOKEN_RE = re.compile(r"[pqrsab]|¬|∧|∨|→|\(|\)")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _parse_formula(tokens: list[str], i: int) -> int | None:
    if i >= len(tokens):
        return None
    tok = tokens[i]
    if tok in ("p", "q", "r", "s", "a", "b"):
        return i + 1
    if tok == "¬":
        return _parse_formula(tokens, i + 1)
    if tok == "(":
        j = _parse_formula(tokens, i + 1)
        if j is None or j >= len(tokens) or tokens[j] not in ("∧", "∨", "→"):
            return None
        k = _parse_formula(tokens, j + 1)
        if k is None or k >= len(tokens) or tokens[k] != ")":
            return None
        return k + 1
    return None


def is_well_formed(text: str) -> bool:
    tokens = _tokenize(text)
    if not tokens:
        return False
    end = _parse_formula(tokens, 0)
    return end is not None and end == len(tokens)


def _mutate_to_malformed(rng: random.Random, text: str) -> str:
    for _attempt in range(50):
        kinds = []
        if "(" in text:
            kinds.append("drop_open")
        if ")" in text:
            kinds.append("drop_close")
        if re.search(r"[∧∨→]", text):
            kinds += ["remove_connective", "duplicate_connective"]
        if re.search(r"[pqrs]", text):
            kinds.append("remove_atom")
        check(kinds, "syntactic-semantic-separation: no mutation applicable to formula text")
        kind = rng.choice(kinds)
        if kind == "drop_open":
            positions = [m.start() for m in re.finditer(r"\(", text)]
            pos = rng.choice(positions)
            candidate = text[:pos] + text[pos + 1 :]
        elif kind == "drop_close":
            positions = [m.start() for m in re.finditer(r"\)", text)]
            pos = rng.choice(positions)
            candidate = text[:pos] + text[pos + 1 :]
        elif kind == "remove_connective":
            matches = list(re.finditer(r"[∧∨→]", text))
            m = rng.choice(matches)
            candidate = text[: m.start()] + text[m.end() :]
        elif kind == "duplicate_connective":
            matches = list(re.finditer(r"[∧∨→]", text))
            m = rng.choice(matches)
            candidate = text[: m.end()] + f" {m.group()}" + text[m.end() :]
        else:  # remove_atom
            matches = list(re.finditer(r"[pqrs]", text))
            m = rng.choice(matches)
            candidate = text[: m.start()] + text[m.end() :]
        if not is_well_formed(candidate):
            return candidate
    check(False, "syntactic-semantic-separation: could not construct a genuinely malformed string")


# ---------------------------------------------------------------------------
# Derivation context (modus ponens)


def _mp_context(phi_tree, two_step: bool) -> tuple[str, str]:
    """Premises + inference-rule prose that syntactically license `phi_tree`
    as a derived conclusion, independent of any interpretation."""
    phi_text = _render(phi_tree)
    phi_latex = _render_latex(phi_tree)
    if not two_step:
        premise2 = f"({_PREMISE_ATOM} → {phi_text})"
        premise2_latex = rf"({_PREMISE_ATOM} \to {phi_latex})"
        text = (
            f"Premise 1: {_PREMISE_ATOM}. Premise 2: {premise2}. "
            f"Applying modus ponens to Premise 1 and Premise 2 yields {phi_text} "
            f"as syntactically derivable, independent of any interpretation."
        )
        latex = (
            rf"\text{{Premise 1: }}{_PREMISE_ATOM}\text{{. Premise 2: }}{premise2_latex}"
            rf"\text{{. Modus ponens yields }}{phi_latex}\text{{, independent of any interpretation.}}"
        )
        return text, latex

    premise2 = f"({_PREMISE_ATOM} → {_INTERMEDIATE_ATOM})"
    premise3 = f"({_INTERMEDIATE_ATOM} → {phi_text})"
    premise2_latex = rf"({_PREMISE_ATOM} \to {_INTERMEDIATE_ATOM})"
    premise3_latex = rf"({_INTERMEDIATE_ATOM} \to {phi_latex})"
    text = (
        f"Premise 1: {_PREMISE_ATOM}. Premise 2: {premise2}. Premise 3: {premise3}. "
        f"Applying modus ponens to Premise 1 and Premise 2 yields {_INTERMEDIATE_ATOM}; "
        f"applying modus ponens to {_INTERMEDIATE_ATOM} and Premise 3 then yields {phi_text} "
        f"as syntactically derivable, independent of any interpretation."
    )
    latex = (
        rf"\text{{Premise 1: }}{_PREMISE_ATOM}\text{{. Premise 2: }}{premise2_latex}"
        rf"\text{{. Premise 3: }}{premise3_latex}\text{{. Two modus-ponens steps yield }}"
        rf"{phi_latex}\text{{, independent of any interpretation.}}"
    )
    return text, latex


# ---------------------------------------------------------------------------


@generator(
    "p15b_syntactic_semantic_separation",
    level=20,
    presets={
        "easy": dict(complexity=1, two_step=False, context_prob=0.5),
        "medium": dict(complexity=2, two_step=False, context_prob=0.5),
        "hard": dict(complexity=3, two_step=False, context_prob=0.5),
        "exam": dict(complexity=3, two_step=True, context_prob=0.5),
    },
)
def p15b_syntactic_semantic_separation(rng, complexity=2, two_step=False, context_prob=0.5):
    """Classify a supplied propositional string (and, for the well-formed
    classes, an interpretation) as ill-formed, well-formed and true,
    well-formed but false, or -- via a stated modus-ponens derivation from
    stated premises -- syntactically derivable yet false under the
    interpretation (the skill's contract, widened
    into this genuine 4-way kernel).

    easy/medium/hard escalate by formula complexity (2/3/4 atoms, deeper
    nesting). exam escalates by METHOD: the derivable class now uses a
    genuine TWO-STEP modus-ponens chain (three premises) rather than one.
    Derivation context is attached to some TRUE-valued draws too (still
    correctly answered "well-formed and true"), so seeing premises never by
    itself tells a learner the answer is D.
    """
    target = rng.choice(["ill_formed", "true", "false", "derivable_false"])

    if target == "ill_formed":
        tree, values, _value = _build_formula(rng, complexity, target_bool=None)
        clean_text = _render(tree)
        malformed_text = _mutate_to_malformed(rng, clean_text)
        check(
            not is_well_formed(malformed_text), "syntactic-semantic-separation: 'ill_formed' draw parses as well-formed"
        )
        expr_text = malformed_text
        expr_latex = expr_text
        assignment_text = ", ".join(f"{k} = {v}" for k, v in sorted(values.items()))
        answer_text = OPTIONS[0]
        context_text = ""
        context_latex = ""
        working = [
            f"'{expr_text}' cannot be parsed by the well-formed-formula grammar "
            f"(atom | not Formula | (Formula connective Formula)): it is ill-formed, "
            f"regardless of any interpretation."
        ]
    else:
        if target == "true":
            phi_bool = True
        elif target == "false":
            phi_bool = False
        else:  # derivable_false
            phi_bool = False

        tree, values, value = _build_formula(rng, complexity, target_bool=phi_bool)
        check(value == phi_bool, "syntactic-semantic-separation: constructed formula truth mismatch")
        expr_text = _render(tree)
        expr_latex = _render_latex(tree)
        check(
            is_well_formed(expr_text),
            "syntactic-semantic-separation: constructed well-formed draw fails its own parser",
        )

        show_context = (target == "derivable_false") or (target == "true" and rng.random() < context_prob)
        assignment_text = ", ".join(f"{k} = {v}" for k, v in sorted(values.items()))

        if show_context:
            context_text, context_latex = _mp_context(tree, two_step)
        else:
            context_text, context_latex = "", ""

        if target == "derivable_false":
            answer_text = OPTIONS[3]
            working = [
                f"Under {assignment_text}, {expr_text} evaluates to False. The premises' "
                f"syntactic shape licenses deriving {expr_text} via modus ponens regardless "
                f"of this interpretation, but the interpretation itself makes it false: "
                f"well-formed and false, and derivable via the premises shown alongside it."
            ]
        elif target == "true":
            answer_text = OPTIONS[1]
            working = [
                f"Under {assignment_text}, {expr_text} evaluates to True: well-formed and true, "
                f"whether or not a derivation is also shown."
            ]
        else:
            answer_text = OPTIONS[2]
            working = [
                f"Under {assignment_text}, {expr_text} evaluates to False, and no derivation is "
                f"shown alongside it: well-formed and false, not derivable from any stated premises."
            ]

    instruction = (
        "Classify the expression above: is it ill-formed, well-formed and true under "
        "the interpretation, well-formed and false with no derivation shown, or "
        "well-formed and false but derivable via modus ponens from premises stated "
        "alongside it?"
    )

    if target == "ill_formed":
        body_text = f"Expression: {expr_text}. Interpretation: {assignment_text}. {instruction}"
        body_latex = (
            rf"\text{{Expression: }}{expr_latex}\text{{. Interpretation: {assignment_text}. }}"
            rf"\text{{{instruction}}}"
        )
    elif context_text:
        body_text = f"{context_text} Expression: {expr_text}. Interpretation: {assignment_text}. {instruction}"
        body_latex = (
            rf"\text{{{context_text} Expression: }}{expr_latex}"
            rf"\text{{. Interpretation: {assignment_text}. }}\text{{{instruction}}}"
        )
    else:
        body_text = f"Expression: {expr_text}. Interpretation: {assignment_text}. {instruction}"
        body_latex = (
            rf"\text{{Expression: }}{expr_latex}\text{{. Interpretation: {assignment_text}. }}"
            rf"\text{{{instruction}}}"
        )

    check(answer_text in OPTIONS, "syntactic-semantic-separation: answer not in fixed options")
    distractors = [o for o in OPTIONS if o != answer_text]
    check(len(distractors) == 3, "syntactic-semantic-separation: expected three distractors")
    check(len(set(OPTIONS)) == len(OPTIONS), "syntactic-semantic-separation: option collision")

    return dict(
        text=body_text,
        latex=body_latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=rf"\text{{{answer_text}}}",
        working=working,
        distractors=distractors,
    )
