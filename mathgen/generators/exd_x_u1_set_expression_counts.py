"""Exam-tier source for `finite-set-operations`: a compound set expression, then counting from the result.

The stem gives a universal set U = {1, ..., N} and three subsets A, B and C, and names a set R by a two-operation
expression such as (A ∪ B) ∖ C, A ∩ (B ∪ C)′ or (A △ B) ∖ C, where ′ is the complement in U.

* (a) list the elements of R;
* (b) say how many elements R has;
* (c) a count that depends on that number: the number of subsets of R, the number of its subsets with exactly two
  elements, or the number of its non-empty proper subsets. This part follows through from (b).

Each expression carries its own table of believable wrong routes (a union read as an intersection, a difference
taken the wrong way round, a complement left out, De Morgan applied wrongly). The draw keeps R between two and
six elements and keeps every wrong route different from R, so none of them is the answer in disguise.
"""

from __future__ import annotations

import random
from itertools import combinations
from math import comb

from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors

# (text, function of (A, B, C, U), [(wrong-route function, misconception id or None), ...])
_EXPRESSIONS = [
    (
        "(A ∪ B) ∖ C",
        lambda A, B, C, U: (A | B) - C,
        [
            (lambda A, B, C, U: (A & B) - C, "union-intersection-confusion"),
            (lambda A, B, C, U: C - (A | B), "reversed-difference"),
            (lambda A, B, C, U: (A | B) & C, None),
        ],
    ),
    (
        "(A ∖ B) ∪ (B ∖ C)",
        lambda A, B, C, U: (A - B) | (B - C),
        [
            (lambda A, B, C, U: (B - A) | (C - B), "reversed-difference"),
            (lambda A, B, C, U: (A - B) & (B - C), "union-intersection-confusion"),
            (lambda A, B, C, U: A ^ B, None),
        ],
    ),
    (
        "A ∩ (B ∪ C)′",
        lambda A, B, C, U: A & (U - (B | C)),
        [
            (lambda A, B, C, U: A & (B | C), "complement-left-out"),
            (lambda A, B, C, U: A & ((U - B) | (U - C)), "complement-of-union-as-union-of-complements"),
            (lambda A, B, C, U: (U - A) & (B | C), None),
        ],
    ),
    (
        "(A △ B) ∖ C",
        lambda A, B, C, U: (A ^ B) - C,
        [
            (lambda A, B, C, U: (A | B) - C, "symmetric-difference-as-union"),
            (lambda A, B, C, U: (A & B) - C, "symmetric-difference-as-intersection"),
            (lambda A, B, C, U: (A ^ B) | C, None),
        ],
    ),
    (
        "(A ∪ B)′ ∩ C",
        lambda A, B, C, U: (U - (A | B)) & C,
        [
            (lambda A, B, C, U: (A | B) & C, "complement-left-out"),
            (lambda A, B, C, U: ((U - A) | (U - B)) & C, "complement-of-union-as-union-of-complements"),
            (lambda A, B, C, U: (U - (A & B)) & C, None),
        ],
    ),
    (
        "A ∩ (B △ C)",
        lambda A, B, C, U: A & (B ^ C),
        [
            (lambda A, B, C, U: A & (B | C), "symmetric-difference-as-union"),
            (lambda A, B, C, U: A & (B & C), "symmetric-difference-as-intersection"),
            (lambda A, B, C, U: A | (B ^ C), "union-intersection-confusion"),
        ],
    ),
]

# (stem with a note on the complement, stem without one, LaTeX of the same stem built from pieces)
_STEMS = [
    (
        "Let U = {u}, A = {a}, B = {b} and C = {c}{note}. Let R = {expr}.",
        ", where ′ denotes the complement in U",
    ),
    (
        "The universal set is U = {u}. Given A = {a}, B = {b} and C = {c}, define R = {expr}{note}.",
        " (the complement ′ is taken in U)",
    ),
]
_LATEX_SYMBOLS = {"∪": r"\cup ", "∩": r"\cap ", "∖": r"\setminus ", "△": r"\triangle ", "′": "'"}
_A_PROMPTS = ["List the elements of R.", "Write down the elements of R."]
_B_PROMPTS = ["How many elements does R have?", "Find the number of elements in R."]
# (prompt, function of k, follow-through expression in part_b, label of the slip family)
_C_KINDS = [
    ("How many subsets does R have?", lambda k: 2**k, "2**part_b", "count"),
    ("How many of the subsets of R have exactly two elements?", lambda k: comb(k, 2), "part_b*(part_b-1)/2", "pairs"),
    ("How many non-empty proper subsets does R have?", lambda k: 2**k - 2, "2**part_b-2", "proper"),
]


def _text(values) -> str:
    return "{" + ", ".join(map(str, sorted(values))) + "}"


def _set_latex(values) -> str:
    return r"\{" + ", ".join(map(str, sorted(values))) + r"\}"


def _expr_latex(expr_text: str) -> str:
    return "".join(_LATEX_SYMBOLS.get(ch, ch) for ch in expr_text.replace(" ", ""))


def _stem_latex(variant: int, universe, a, b, c, expr_text: str, note: bool) -> str:
    """The stem as real LaTeX: sets and the expression in maths mode, prose in `\\text{}`."""
    u, sa, sb, sc = _set_latex(universe), _set_latex(a), _set_latex(b), _set_latex(c)
    expr = _expr_latex(expr_text)
    if variant == 0:
        tail = r"\text{, where }{}'\text{ denotes the complement in }U" if note else ""
        return rf"\text{{Let }}U={u},\ A={sa},\ B={sb}\text{{ and }}C={sc}{tail}\text{{. Let }}R={expr}\text{{.}}"
    tail = r"\text{ (the complement }{}'\text{ is taken in }U\text{)}" if note else ""
    return (
        rf"\text{{The universal set is }}U={u}\text{{. Given }}A={sa},\ B={sb}\text{{ and }}C={sc}"
        rf"\text{{, define }}R={expr}{tail}\text{{.}}"
    )


def _draw(rng: random.Random):
    for _ in range(2000):
        n = rng.randint(9, 14)
        universe = set(range(1, n + 1))
        a, b, c = ({v for v in universe if rng.random() < 0.5} for _ in range(3))
        expr_text, fn, slips = rng.choice(_EXPRESSIONS)
        result = fn(a, b, c, universe)
        wrong = [s(a, b, c, universe) for s, _tag in slips]
        if not 2 <= len(result) <= 6 or any(w == result for w in wrong) or len({frozenset(w) for w in wrong}) < 3:
            continue
        if min(len(a), len(b), len(c)) < 3 or max(len(a), len(b), len(c)) > n - 3:
            continue
        return n, universe, a, b, c, expr_text, fn, slips, result, wrong
    raise RuntimeError("no suitable sets in 2000 draws")


def _build(rng: random.Random) -> dict | None:
    """One candidate question, or None when a part ends up with fewer than two distinct distractors."""
    n, universe, a, b, c, expr_text, fn, slips, result, wrong = _draw(rng)
    c_prompt, c_fn, c_ft, c_family = rng.choice(_C_KINDS)
    k = len(result)

    # Independent route for the count in (c): enumerate subsets of R.
    members = sorted(result)
    enumerated = {
        "count": sum(1 for r in range(k + 1) for _ in combinations(members, r)),
        "pairs": sum(1 for _ in combinations(members, 2)),
        "proper": sum(1 for r in range(1, k) for _ in combinations(members, r)),
    }[c_family]
    key_c = c_fn(k)
    check(enumerated == key_c, "subset count disagrees with enumeration")
    check(all(1 <= v <= n for v in result), "result leaves the universe")

    answer_text = _text(result)
    a_cands = [Distractor(_text(w), tag, sorted(w)) for w, (_s, tag) in zip(wrong, slips)] + [
        Distractor(_text(universe - result), "complement-given-for-set", sorted(universe - result))
    ]
    # Count slips: the number of elements treated as the count (b); doubling, squaring and off-by-one (c).
    b_cands = [Distractor(str(len(w)), tag, len(w)) for w, (_s, tag) in zip(wrong, slips)] + [
        Distractor(str(n - k), "complement-given-for-set", n - k)
    ]
    c_slip = {
        "count": [
            (2 * k, "subsets-counted-as-doubling"),
            (k * k, "subsets-counted-as-square"),
            (2**k - 1, "nonempty-subsets-only"),
        ],
        "pairs": [
            (k * (k - 1), "ordered-pairs-for-subsets"),
            (2 * k, "subsets-counted-as-doubling"),
            (comb(k + 1, 2), None),
        ],
        "proper": [
            (2**k, "improper-subsets-included"),
            (2**k - 1, "improper-subsets-included"),
            (2 * k - 2, "subsets-counted-as-doubling"),
        ],
    }[c_family]
    c_cands = [Distractor(str(v), tag, v) for v, tag in c_slip]
    a_dist = pick_distractors(answer_text, a_cands, 3, unordered=True)
    b_dist = pick_distractors(str(k), b_cands, 3)
    c_dist = pick_distractors(str(key_c), c_cands, 3)
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        return None

    variant = rng.randrange(len(_STEMS))
    template, note_text = _STEMS[variant]
    note = "′" in expr_text
    return dict(
        text=template.format(
            u=_text(universe), a=_text(a), b=_text(b), c=_text(c), expr=expr_text, note=note_text if note else ""
        ),
        latex=_stem_latex(variant, universe, a, b, c, expr_text, note),
        parts=[
            dict(
                prompt=rng.choice(_A_PROMPTS),
                answer_kind="list",
                answer=sorted(result),
                answer_text=answer_text,
                marks=3,
                working=[f"Work through {expr_text} one operation at a time to get {answer_text}."],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS),
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=1,
                sensible_range=(0, None),
                working=[f"R has {k} elements."],
                distractors=b_dist,
            ),
            dict(
                prompt=c_prompt,
                answer_kind="number",
                answer=key_c,
                answer_text=str(key_c),
                marks=2,
                sensible_range=(0, None),
                follow_through=c_ft,
                working=[f"With {k} elements in R, the count is {key_c}."],
                distractors=c_dist,
            ),
        ],
    )


@generator("exd_x_u1_set_expression_counts", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u1_set_expression_counts(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: evaluate a compound set expression, count it, and count subsets of it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a set question with enough distractors"):
        question = _build(rng)
        if question is not None:
            return question
