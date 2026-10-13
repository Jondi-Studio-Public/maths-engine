"""Implication/converse/contrapositive and implication/equivalence/equality symbols.

- ``implication_transform_identify``
  (``senior-use-implication-converse-equivalence-negation-contrapositive``,
  ACMSM024, domain senior-mathematics, answer: mcq) -- given a bounded
  numeric implication "If n <predicate>, then n <predicate>.", identify its
  converse, judge whether a named transform (converse/contrapositive/
  negation) is ALWAYS logically equivalent to the original, or evaluate the
  negation's truth value at the stated n.

- ``implication_equivalence_symbol_classify``
  (``senior-use-the-symbols-for-implication-rightarrow-equivalence-longleftrightarrow``,
  ACMSM026, domain senior-mathematics, answer: mcq) -- given two stated
  things (a numeric equality, or a curated bounded-domain predicate pair),
  pick the ONE symbol among ``⇒``, ``⇔``, ``=`` that correctly relates them,
  or (hard tier only) recognise that none of the three does.

Both topics use real Unicode logic/arrow symbols (``⇒``, ``⇔``, ``×``) in
`text`, confirmed safe against every pattern in `mathgen/text_quality.py`
(the ``star_operator`` pattern only ever flags a literal ASCII ``*``). Every
question's `latex` field mirrors `text`'s logical content using real LaTeX
macros -- ``\\Rightarrow`` for implication, ``\\Longleftrightarrow`` for
equivalence, ordinary ``=`` for equality -- for consistency with the rest of
the package, even though neither skill's instruction-phrase pair is one of
the ones `tests/test_question_latex_instructions.py` enforces.
"""

from __future__ import annotations

import itertools
import math

from ..core import check, generator, latex_answer, pick_distractors


def _latex_text(s: str) -> str:
    """Wrap a plain English clause in a LaTeX ``\\text{}`` block."""
    return rf"\text{{{s}}}"


# ============================================================================
# Skill 1: implication_transform_identify (ACMSM024)
# ============================================================================

# Every name is phrased "is ..." so a clause is always f"{n} {suffix}" and its
# negation is always f"{n} is not {suffix[3:]}". Asserted directly (not just
# eyeballed) in tests/test_p9a_implication_symbols_build.py.
_PREDICATES = [
    ("is even", lambda n: n % 2 == 0),
    ("is odd", lambda n: n % 2 == 1),
    ("is a multiple of 3", lambda n: n % 3 == 0),
    ("is a multiple of 5", lambda n: n % 5 == 0),
    ("is greater than 20", lambda n: n > 20),
    ("is a perfect square", lambda n: n >= 0 and math.isqrt(n) ** 2 == n),
]

# The abstract implication identity, over the 4 boolean truth combinations of
# (P, Q). Deliberately independent of any concrete predicate -- whether a
# transform is ALWAYS equivalent to P => Q is a fact about propositional
# logic, not about what P and Q happen to mean. Factored out so a test can
# corrupt one entry and confirm the check would actually fire.
_TRUTH_ROWS = list(itertools.product([False, True], repeat=2))
_TRANSFORMS = {
    # (not-Q) -> (not-P), written as a material implication: NOT(NOT-Q) OR NOT-P = Q OR NOT-P.
    "contrapositive": lambda p, q: q or (not p),
    "converse": lambda p, q: (not q) or p,  # Q -> P
    "negation": lambda p, q: p and (not q),  # NOT(P -> Q)
}


def _transform_always_equivalent(s_kind: str, transforms: dict = _TRANSFORMS) -> bool:
    """Whether `transforms[s_kind]` matches the implication on every truth row."""
    transform = transforms[s_kind]
    implication = lambda p, q: (not p) or q
    return all(transform(p, q) == implication(p, q) for p, q in _TRUTH_ROWS)


def _negation_via_identity(p_val: bool, q_val: bool) -> bool:
    """The negation's truth value, recomputed via the identity
    ¬(P -> Q) ≡ P ∧ ¬Q from the IMPLICATION's own truth value -- a genuinely
    different route from evaluating `P and not Q` directly a second time."""
    implication_truth = (not p_val) or q_val
    return not implication_truth


@generator(
    "implication_transform_identify",
    level=16,
    presets={
        "easy": dict(variant="converse"),
        "medium": dict(variant="equivalence_check"),
        "hard": dict(variant="negation_truth"),
    },
)
def implication_transform_identify(rng, variant="converse"):
    """Converse / equivalence-of-transform / negation-truth over a bounded
    numeric implication "If n <P>, then n <Q>."."""
    i, j = rng.sample(range(len(_PREDICATES)), 2)
    p_name, p_fn = _PREDICATES[i]
    q_name, q_fn = _PREDICATES[j]

    if variant == "negation_truth":
        negation_target = rng.choice([True, False])
        if negation_target:
            candidates = [n for n in range(2, 41) if p_fn(n) and not q_fn(n)]
        else:
            candidates = [n for n in range(2, 41) if (not p_fn(n)) and q_fn(n)]
        check(len(candidates) > 0, f"no n found realising negation-truth target {negation_target} for this pair")
    else:
        candidates = [n for n in range(2, 41) if p_fn(n) != q_fn(n)]
        check(len(candidates) > 0, "no n found with P(n) != Q(n) for this pair")
    n = rng.choice(candidates)

    clause_p = f"{n} {p_name}"
    clause_q = f"{n} {q_name}"
    not_clause_p = f"{n} is not {p_name[3:]}"
    not_clause_q = f"{n} is not {q_name[3:]}"

    original_text = f"If {clause_p}, then {clause_q}."
    converse_text = f"If {clause_q}, then {clause_p}."
    contrapositive_text = f"If {not_clause_q}, then {not_clause_p}."
    negation_text = f"{clause_p} and {not_clause_q}."
    check(
        len({original_text, converse_text, contrapositive_text, negation_text}) == 4,
        "the four transform texts are not all distinct",
    )

    original_latex = rf"{_latex_text(clause_p)} \Rightarrow {_latex_text(clause_q)}"
    converse_latex = rf"{_latex_text(clause_q)} \Rightarrow {_latex_text(clause_p)}"
    contrapositive_latex = rf"{_latex_text(not_clause_q)} \Rightarrow {_latex_text(not_clause_p)}"
    negation_latex = rf"{_latex_text(clause_p)} \land {_latex_text(not_clause_q)}"

    p_val, q_val = p_fn(n), q_fn(n)

    if variant == "converse":
        text = f"{original_text} What is the converse of the implication above?"
        latex = rf"{original_latex}.\ \text{{What is the converse of the implication above?}}"
        answer_text = converse_text
        distractors = pick_distractors(answer_text, [contrapositive_text, negation_text, original_text])
        working = [
            'The converse of an implication "If A, then B" is "If B, then A": swap the hypothesis and the conclusion.',
        ]

    elif variant == "equivalence_check":
        target = rng.choice([True, False])
        if target:
            s_kind = "contrapositive"
        else:
            s_kind = rng.choice(["converse", "negation"])
        s_text = {"converse": converse_text, "contrapositive": contrapositive_text, "negation": negation_text}[s_kind]
        s_latex = {"converse": converse_latex, "contrapositive": contrapositive_latex, "negation": negation_latex}[
            s_kind
        ]

        # Independent check, exhaustive over the 4 boolean truth combinations
        # of (P(n), Q(n)), keyed to the SPECIFIC s_kind drawn this seed.
        always_equal = _transform_always_equivalent(s_kind)
        check(
            always_equal == (s_kind == "contrapositive"),
            f"{s_kind}'s always-equivalent status does not match the exhaustive truth table",
        )
        check(always_equal == target, "constructed target not realised by the chosen s_kind")

        text = (
            f'{original_text} Consider its {s_kind}: "{s_text}" '
            f"Is this ALWAYS logically equivalent to the original implication, for every value of n?"
        )
        latex = (
            rf"{original_latex}.\ \text{{Consider its {s_kind}: }} {s_latex}.\ "
            rf"\text{{Is this ALWAYS logically equivalent to the original implication, for every value of }} n\text{{?}}"
        )
        # Identify the transform, then say whether that kind is always equivalent. The classic slip is the
        # converse taken as equivalent, so it appears with both verdicts.
        kind_options = {
            "contrapositive": "True: it is the contrapositive, which always has the same truth value as the original",
            "converse": "False: it is the converse, which can have a different truth value from the original",
            "negation": "False: it is the negation, which always has the opposite truth value to the original",
        }
        answer_text = kind_options[s_kind]
        distractors = [t for k, t in kind_options.items() if k != s_kind]
        distractors.append("True: it is the converse, which always has the same truth value as the original")
        working = [
            "A transformed statement is ALWAYS equivalent to the original implication only "
            "when it matches the original on all four rows of the truth table for P and Q.",
            "The contrapositive (¬Q ⇒ ¬P) always matches the original P ⇒ Q; the converse "
            "and the negation do not, in general.",
        ]

    elif variant == "negation_truth":
        negation_truth = p_val and not q_val
        # Independent check: recompute via the ¬(P->Q) ≡ P∧¬Q identity from
        # the implication's own truth value, a genuinely different route.
        check(
            negation_truth == _negation_via_identity(p_val, q_val),
            "negation truth value does not match the implication identity",
        )
        check(negation_truth == negation_target, "constructed negation-truth target not realised by the chosen n")

        text = f"{original_text} Its negation is: {negation_text} Given n = {n}, is this negation True or False?"
        latex = (
            rf"{original_latex}.\ \text{{Its negation is: }} {negation_latex}.\ "
            rf"\text{{Given }} n = {n}\text{{, is this negation True or False?}}"
        )
        # The negation is "P and not Q", so its truth value is fixed by the truth values of P and Q at n.
        states = {
            (True, False): "True, because the hypothesis is true and the conclusion is false",
            (True, True): "False, because the hypothesis and the conclusion are both true",
            (False, True): "False, because the hypothesis is false and the conclusion is true",
            (False, False): "False, because the hypothesis and the conclusion are both false",
        }
        answer_text = states[(p_val, q_val)]
        distractors = [t for state, t in states.items() if state != (p_val, q_val)]
        working = [
            'The negation of "If P, then Q" is "P and not Q" (from ¬(P ⇒ Q) ≡ P ∧ ¬Q).',
            f"Evaluate P and not Q directly at n = {n} using the two given statements.",
        ]

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: implication_equivalence_symbol_classify (ACMSM026)
# ============================================================================

# Curated pairs. Each entry is (A_desc, A_fn, B_desc, B_fn).
FORWARD_ONLY = [
    ("n is a multiple of 4", lambda n: n % 4 == 0, "n is even", lambda n: n % 2 == 0),
    ("n is a multiple of 9", lambda n: n % 9 == 0, "n is a multiple of 3", lambda n: n % 3 == 0),
    ("n is a multiple of 10", lambda n: n % 10 == 0, "n is a multiple of 5", lambda n: n % 5 == 0),
]
BICONDITIONAL = [
    ("n is even", lambda n: n % 2 == 0, "n is not odd", lambda n: n % 2 == 0),
    (
        "n is a multiple of 6",
        lambda n: n % 6 == 0,
        "n is a multiple of 2 and a multiple of 3",
        lambda n: n % 2 == 0 and n % 3 == 0,
    ),
    (
        "n is a multiple of 12",
        lambda n: n % 12 == 0,
        "n is a multiple of 3 and a multiple of 4",
        lambda n: n % 3 == 0 and n % 4 == 0,
    ),
]
UNRELATED = [
    ("n is even", lambda n: n % 2 == 0, "n is a multiple of 3", lambda n: n % 3 == 0),
    (
        "n is a multiple of 5",
        lambda n: n % 5 == 0,
        "n is a perfect square",
        lambda n: n >= 0 and math.isqrt(n) ** 2 == n,
    ),
]

_SYMBOL_TAIL_TEXT = "Which symbol correctly relates A and B: ⇒, ⇔, or =?"
_SYMBOL_TAIL_LATEX = r"\text{Which symbol correctly relates A and B: } \Rightarrow, \Longleftrightarrow, \text{ or } =?"


def _forward_backward(a_fn, b_fn, n_hi: int) -> tuple[bool, bool]:
    """Whether A => B (forward) and B => A (backward) hold over every whole
    number 1..n_hi -- factored so a test can pass a swapped/corrupted pair
    and confirm the check would fire."""
    forward_holds = all((not a_fn(n)) or b_fn(n) for n in range(1, n_hi + 1))
    backward_holds = all((not b_fn(n)) or a_fn(n) for n in range(1, n_hi + 1))
    return forward_holds, backward_holds


def _statement_pair_block(a_desc: str, b_desc: str, n_hi: int) -> tuple[str, str]:
    """(text, latex) for the "Statement A / Statement B / bounded-N" preamble
    shared by the ⇒/⇔/unrelated branches."""
    text = (
        f"Statement A: {a_desc}. Statement B: {b_desc}. "
        f"This is considered for every whole number n from 1 to {n_hi}. {_SYMBOL_TAIL_TEXT}"
    )
    latex = (
        rf"\text{{Statement A: {a_desc}. Statement B: {b_desc}. This is considered for every "
        rf"whole number }} n \text{{ from 1 to }} {n_hi}. {_SYMBOL_TAIL_LATEX}"
    )
    return text, latex


@generator(
    "implication_equivalence_symbol_classify",
    level=16,
    presets={
        "easy": dict(classes=("=", "⇔")),
        "medium": dict(classes=("⇒", "⇔")),
        "hard": dict(classes=("⇒", "⇔", "=", "unrelated")),
    },
)
def implication_equivalence_symbol_classify(rng, classes=("=", "⇔")):
    """Pick the one symbol (⇒, ⇔, =) that relates two stated things, or (hard
    tier) recognise that none of the three does."""
    symbol_class = rng.choice(classes)

    if symbol_class == "=":
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        product = a * b
        # Independent check: repeated addition, a genuinely different route
        # from the `*` operator.
        check(sum(a for _ in range(b)) == product, "repeated-addition check failed for a*b")
        text = f"Statement A: the value of {a} × {b}. Statement B: the value of {product}. {_SYMBOL_TAIL_TEXT}"
        latex = (
            rf"\text{{Statement A: the value of }} {a} \times {b}. "
            rf"\text{{Statement B: the value of }} {product}. {_SYMBOL_TAIL_LATEX}"
        )
        answer_text = "="
        distractors = pick_distractors(answer_text, ["⇒", "⇔", "No single symbol connects these"])
        working = [f'{a} × {b} and {product} name the same value, so "=" relates A and B.']

    elif symbol_class == "⇔":
        a_desc, a_fn, b_desc, b_fn = rng.choice(BICONDITIONAL)
        n_hi = rng.randint(24, 40)
        forward_holds, backward_holds = _forward_backward(a_fn, b_fn, n_hi)
        check(forward_holds and backward_holds, "biconditional pair is not mutually implying over this domain")
        text, latex = _statement_pair_block(a_desc, b_desc, n_hi)
        answer_text = "⇔"
        distractors = pick_distractors(answer_text, ["⇒", "=", "No single symbol connects these"])
        working = [
            f'Check both directions for n = 1 to {n_hi}: A implies B and B implies A both hold, so "⇔" relates A and B.'
        ]

    elif symbol_class == "⇒":
        a_desc, a_fn, b_desc, b_fn = rng.choice(FORWARD_ONLY)
        n_hi = rng.randint(24, 40)
        forward_holds, backward_holds = _forward_backward(a_fn, b_fn, n_hi)
        check(forward_holds and not backward_holds, "curated pair is not forward-only over this domain")
        text, latex = _statement_pair_block(a_desc, b_desc, n_hi)
        answer_text = "⇒"
        distractors = pick_distractors(answer_text, ["⇔", "=", "No single symbol connects these"])
        working = [
            f"Check both directions for n = 1 to {n_hi}: A implies B, but B does not imply "
            'A (a counterexample exists), so only "⇒" holds.'
        ]

    elif symbol_class == "unrelated":
        a_desc, a_fn, b_desc, b_fn = rng.choice(UNRELATED)
        n_hi = rng.randint(24, 40)
        forward_holds, backward_holds = _forward_backward(a_fn, b_fn, n_hi)
        check(not forward_holds and not backward_holds, "curated pair is not unrelated over this domain")
        text, latex = _statement_pair_block(a_desc, b_desc, n_hi)
        answer_text = "No single symbol connects these"
        distractors = pick_distractors(answer_text, ["⇒", "⇔", "="])
        working = [
            f"Check both directions for n = 1 to {n_hi}: neither A implies B nor B implies "
            "A holds (a counterexample exists each way), so none of ⇒, ⇔, = connects them."
        ]

    else:
        raise ValueError(f"unknown symbol class {symbol_class!r}")

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )
