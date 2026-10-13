"""Exam question on semantic consequence: counting the interpretations behind a claimed derivation.

The stem gives two premises and a conclusion over four or five atoms. Whether the conclusion follows is a question
about all 2^n interpretations, not about how the formulas look. Four parts:

(a) the number of interpretations making both premises true                     number (whole), 2 marks
(b) the number making both premises and the conclusion true                     number (whole), 2 marks
(c) the number making the premises true and the conclusion false, a - b         number (whole), 1 mark, follow_through part_a - part_b
(d) the fraction of the premise-satisfying interpretations that satisfy the
    conclusion, b / a (it would be 1 exactly if the conclusion followed)        fraction, 1 mark, follow_through part_b / part_a

Each formula is a small tree over distinct atoms (so there are no vacuous pieces such as "s and s"). Every count is
made by evaluating the formula tree on every interpretation; the tests count again from the printed formulas with a
separate parser, so the printing and the meaning cannot drift apart.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .exd_x_r2h_support import build_part

_ATOMS = "pqrst"

_STEM = (
    "Premise 1: {p1}. Premise 2: {p2}. A student concludes {c} from these two premises. Consider all {rows} "
    "interpretations that assign True or False to each of {atoms}.",
    "The premises are {p1} and {p2}, and the student's conclusion is {c}. There are {rows} interpretations, "
    "assigning True or False to each of {atoms}.",
    "From the premises {p1} and {p2}, a student draws the conclusion {c}. Look at all {rows} interpretations of "
    "{atoms}, each giving every atom the value True or False.",
)
_PROMPT_A = (
    "Under how many of the interpretations are both premises true?",
    "Count the interpretations that make both premises true.",
)
_PROMPT_B = (
    "Under how many interpretations are both premises and the conclusion all true?",
    "Count the interpretations that make both premises and the conclusion true.",
)
_PROMPT_C = (
    "Hence find the number of interpretations that make both premises true and the conclusion false.",
    "Use your counts to find how many interpretations make both premises true but the conclusion false.",
)
_PROMPT_D = (
    "Of the interpretations that make both premises true, what fraction also make the conclusion true? Give your "
    "answer as a fraction.",
    "Hence find the fraction of the interpretations satisfying both premises that also satisfy the conclusion, as "
    "a fraction.",
)


def _random_formula(rng: random.Random, atoms: str, size: int):
    """A formula tree over `size` distinct atoms of `atoms`, each negated with probability 0.3, joined by random
    connectives into a left-leaning or balanced tree."""
    chosen = rng.sample(atoms, size)
    leaves = [("not", ("atom", x)) if rng.random() < 0.3 else ("atom", x) for x in chosen]
    node = leaves[0]
    for leaf in leaves[1:]:
        node = (
            (rng.choice(("and", "or", "imp")), node, leaf)
            if rng.random() < 0.5
            else (rng.choice(("and", "or", "imp")), leaf, node)
        )
    return node


def _evaluate(tree, valuation: dict) -> bool:
    kind = tree[0]
    if kind == "atom":
        return valuation[tree[1]]
    if kind == "not":
        return not _evaluate(tree[1], valuation)
    left, right = _evaluate(tree[1], valuation), _evaluate(tree[2], valuation)
    if kind == "and":
        return left and right
    if kind == "or":
        return left or right
    return (not left) or right


_SYMBOL = {"and": "∧", "or": "∨", "imp": "→"}


def _render(tree) -> str:
    kind = tree[0]
    if kind == "atom":
        return tree[1]
    if kind == "not":
        return "¬" + _render(tree[1])
    return f"({_render(tree[1])} {_SYMBOL[kind]} {_render(tree[2])})"


def _count(atoms: str, *trees) -> int:
    total = 0
    for values in itertools.product((False, True), repeat=len(atoms)):
        valuation = dict(zip(atoms, values))
        if all(_evaluate(t, valuation) for t in trees):
            total += 1
    return total


def _argument(rng: random.Random, atoms: str):
    """Two premises and a conclusion: formulas of three, two and two atoms."""
    return (
        _random_formula(rng, atoms, rng.choice((3, 4))),
        _random_formula(rng, atoms, rng.choice((2, 3))),
        _random_formula(rng, atoms, rng.choice((2, 3))),
    )


def _wrongs(atoms, p1, p2, concl, a, b):
    rows = 2 ** len(atoms)
    c = a - b
    a_wrong = [
        (_count(atoms, p1), "drops-a-premise"),
        (_count(atoms, p2), "drops-a-premise"),
        (rows - a, "counts-the-complement"),
        (_count(atoms, ("or", p1, p2)), "or-for-and"),
    ]
    b_wrong = [
        (_count(atoms, concl), "counts-conclusion-alone"),
        (a, "ignores-the-conclusion"),
        (_count(atoms, p1, concl), "drops-a-premise"),
        (_count(atoms, p2, concl), "drops-a-premise"),
        (rows - b, "counts-the-complement"),
    ]
    c_wrong = [
        (b, "counts-supporting-rows"),
        (_count(atoms, ("not", concl)), "counts-conclusion-alone"),
        (a, "ignores-the-conclusion"),
        (_count(atoms, p1, ("not", concl)), "drops-a-premise"),
        (_count(atoms, p2, ("not", concl)), "drops-a-premise"),
    ]
    d_wrong = [
        (Fraction(b, rows), "divides-by-all-rows"),
        (Fraction(c, a), "gave-complement-for-event"),
        (Fraction(c, b) if c <= b else None, "odds-as-probability"),
        (Fraction(b, rows - a) if b <= rows - a else None, "divides-by-wrong-rows"),
        (Fraction(a, rows), "divides-by-all-rows"),
        (Fraction(c, rows), "divides-by-all-rows"),
    ]
    return a_wrong, b_wrong, c_wrong, d_wrong


def _enough(key, wrong) -> bool:
    return len({Fraction(v) for v, _ in wrong if v is not None and 0 <= v and v != key}) >= 3


def _build(rng: random.Random) -> dict:
    atoms = _ATOMS[: rng.choice((5, 5, 5, 4))]
    for _ in attempts(5000, "an argument with a spread of counts"):
        p1, p2, concl = _argument(rng, atoms)
        rows = 2 ** len(atoms)
        a = _count(atoms, p1, p2)
        b = _count(atoms, p1, p2, concl)
        c = a - b
        if not (4 <= a <= rows - 6 and 1 <= b < a):
            continue
        a_wrong, b_wrong, c_wrong, d_wrong = _wrongs(atoms, p1, p2, concl, a, b)
        key_d = Fraction(b, a)
        if all(_enough(k, w) for k, w in ((a, a_wrong), (b, b_wrong), (c, c_wrong), (key_d, d_wrong))):
            break
    check(c >= 1 and b <= a, "counts are inconsistent")
    check(
        a
        == sum(
            1
            for v in itertools.product((False, True), repeat=len(atoms))
            if all(_evaluate(t, dict(zip(atoms, v))) for t in (p1, p2))
        ),
        "premise count disagrees",
    )
    check(
        b + _count(atoms, p1, p2, ("not", concl)) == a, "the conclusion and its negation do not split the premise rows"
    )

    stem = phrase(
        rng,
        _STEM,
        p1=_render(p1),
        p2=_render(p2),
        c=_render(concl),
        rows=rows,
        atoms=", ".join(atoms),
    )
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            Fraction(a),
            kind="number",
            marks=2,
            working=[f"Check each of the {rows} rows of the truth table: {a} make both premises true."],
            wrong=a_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B),
            Fraction(b),
            kind="number",
            marks=2,
            working=[f"Among those {a} rows, the conclusion is also true in {b}."],
            wrong=b_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_C),
            Fraction(c),
            kind="number",
            marks=1,
            working=[
                f"{a} - {b} = {c}. "
                + (
                    "There are none, so the conclusion follows."
                    if c == 0
                    else "These are counterexamples, so the step is not valid."
                )
            ],
            wrong=c_wrong,
            follow_through="part_a - part_b",
        ),
        build_part(
            phrase(rng, _PROMPT_D),
            key_d,
            kind="fraction",
            marks=1,
            working=[f"{b} / {a} = {key_d}."],
            wrong=d_wrong,
            follow_through="part_b/part_a",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_entailment_count_chain", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r2h_entailment_count_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Count the interpretations behind a claimed derivation: premises true, conclusion true, counterexamples, fraction."""
    check(tier == "exam", "exd_x_r2h_entailment_count_chain: only the exam tier exists")
    return _build(rng)
