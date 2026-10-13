"""Exam question: the length of a shortest formal derivation (`ucv1-formal-derivation-check`).

Premises: one conjunction x0 & z, a chain of implications x0 -> x1, x1 -> x2, ..., x(K-1) -> xK, and a few irrelevant
premises over fresh atoms, all listed in a shuffled order. The target is a conjunction of m atoms taken from the chain
and possibly z (grouped from the left), optionally followed by "| w" for a fresh atom w. The rules are Premise,
Conjunction Elimination (either side), Conjunction Introduction, Disjunction Introduction and Modus Ponens, and a
line may cite any earlier lines. The parts chain:

* (a) the number of Modus Ponens steps in a shortest derivation, J, the largest chain index in the target;
* (b) how many of the given premises it does not use, N - 1 - J (follow-through from (a));
* (c) how many other inference steps it needs: Conjunction Elimination steps (one for x0, one more when z is a
  conjunct), Conjunction Introduction steps (m - 1) and Disjunction Introduction (0 or 1);
* (d) the number of lines, premise lines plus inference steps: (1 + J) + J + (c) (follow-through from (a) and (c)).

Every chain implication up to J is needed, every conjunct has to appear on its own line, and no step can be shared
or skipped, so the count is exact.
"""

from __future__ import annotations

import random

from ..core import check, generator
from .exd_x_r2g_support import build_part, redraws

_ATOMS = "abcdefghijkmnpqrstuvwxyz"

_INTRO = [
    "Using only the rules Premise, Conjunction Elimination (left or right), Conjunction Introduction, Disjunction "
    "Introduction and Modus Ponens, derive {target} from the premises {premises}. Count a premise as one line each "
    "time it is written, and a line may cite any earlier lines.",
    "A formal derivation may use only Premise, Conjunction Elimination, Conjunction Introduction, Disjunction "
    "Introduction and Modus Ponens. From the premises {premises}, a shortest derivation of {target} is wanted. Each "
    "premise written down is one line, and each line may cite any earlier lines.",
]


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        j = rng.randint(2, 6)
        k = j + rng.randint(0, 2)
        letters = rng.sample(_ATOMS, k + 1 + 1 + 1 + 4)
        x = letters[: k + 1]
        z, w = letters[k + 1], letters[k + 2]
        fresh = letters[k + 3 :]
        m = rng.randint(2, 4)
        use_z = rng.random() < 0.5
        need = m - 1 if use_z else m
        pool = list(range(0, j))
        if need - 1 > len(pool):
            continue
        chosen = [j, *rng.sample(pool, need - 1)]
        rng.shuffle(chosen)
        conjuncts = [x[i] for i in chosen]
        if use_z:
            conjuncts.insert(rng.randint(0, len(conjuncts)), z)
        use_or = rng.random() < 0.5
        extras = rng.randint(0, 2)
        premises = [f"{x[0]} & {z}", *(f"{x[i]} -> {x[i + 1]}" for i in range(k))]
        for e in range(extras):
            a, b = fresh[2 * e], fresh[2 * e + 1]
            premises.append(rng.choice([f"{a} -> {b}", f"{a} & {b}"]))
        rng.shuffle(premises)
        body = " & ".join(conjuncts)
        target = f"({body}) | {w}" if use_or else body
        return dict(j=j, k=k, m=len(conjuncts), use_z=use_z, use_or=use_or, premises=premises, target=target)
    check(False, "no acceptable derivation draw")


@generator("exd_x_r2g_derivation_length", level=15, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_derivation_length(rng: random.Random, tier: str = "exam") -> dict:
    """Modus Ponens steps, unused premises, other steps and total lines of a shortest derivation."""
    check(tier == "exam", "exd_x_r2g_derivation_length: only the exam tier exists")
    d = _draw(rng)
    j, k, m, use_z, use_or, premises = d["j"], d["k"], d["m"], d["use_z"], d["use_or"], d["premises"]
    n = len(premises)
    elim = 1 + (1 if use_z else 0)
    intro = m - 1
    other = elim + intro + (1 if use_or else 0)
    used = 1 + j
    unused = n - used
    lines = used + j + other
    stem = rng.choice(_INTRO).format(target=d["target"], premises="; ".join(premises))
    stem += " Grouping of a conjunction of several atoms is from the left."
    parts = [
        build_part(
            "How many Modus Ponens steps does a shortest derivation contain?",
            j,
            kind="number",
            marks=1,
            pad_counts=True,
            working=[
                f"The deepest atom needed sits {j} implications along the chain, and each implication is used once."
            ],
            wrong=[
                (k, "counts-every-given-implication"),
                (j + 1, "off-by-one-count"),
                (m, "counts-conjuncts-as-steps"),
                (j - 1, "off-by-one-count"),
            ],
        ),
        build_part(
            "Hence find how many of the given premises a shortest derivation does not use.",
            unused,
            kind="number",
            marks=1,
            pad_counts=True,
            follow_through=f"{n} - 1 - part_a",
            working=[f"It uses the conjunction and {j} implications, so {n} - {used} = {unused} are unused."],
            wrong=[
                (n - j, "forgets-the-conjunction-premise"),
                (n - 1 - k, "counts-only-some-unused-premises"),
                (k - j, "counts-only-some-unused-premises"),
                (j, "reports-steps-as-unused-premises"),
            ],
        ),
        build_part(
            "How many steps other than Premise and Modus Ponens (Conjunction Elimination, Conjunction Introduction "
            "and Disjunction Introduction together) does a shortest derivation need?",
            other,
            kind="number",
            marks=2,
            pad_counts=True,
            working=[
                f"Conjunction Elimination: {elim}; Conjunction Introduction: {intro}; Disjunction Introduction: "
                f"{1 if use_or else 0}.",
                f"Total {other}.",
            ],
            wrong=[
                (intro, "leaves-out-a-kind-of-inference-step"),
                (other - elim, "leaves-out-a-kind-of-inference-step"),
                (other - 1 if use_or else None, "leaves-out-a-kind-of-inference-step"),
                (elim + m, "miscounts-inference-steps"),
            ],
        ),
        build_part(
            "Hence find the total number of lines in a shortest derivation.",
            lines,
            kind="number",
            marks=2,
            pad_counts=True,
            follow_through="(1 + part_a) + part_a + part_c",
            working=[f"Premise lines {used} + Modus Ponens {j} + other steps {other} = {lines}."],
            wrong=[
                (j + other, "counts-steps-not-lines"),
                (n + j + other, "counts-unused-premises-as-lines"),
                (2 * j + other, "forgets-the-conjunction-premise"),
                (lines + 1, "off-by-one-count"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
