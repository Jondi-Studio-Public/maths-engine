"""Shared pieces for the rubric-marked critique, audit, proof and analysis generators (`crit_*_build.py`).

A written-answer question is marked against a rubric, not against one expected answer, so a generator returns a
prompt plus a per-item marking dict (`Question.answer`):

    {"start": None, "target": None, "reference_lines": [...], "key_steps": [...],
     "traps": [...], "checks": []}

`reference_lines` is one model answer in plain sentences, `key_steps` says which rubric criterion each key point
of a correct answer earns, and `traps` names the usual wrong answers, each with the question a marker should ask
about it. `checks` is empty: nothing here can be decided by a computer algebra system alone, so a person or a
model marks the written answer.

The prompt and formula helpers (`Prompt`, `Fm`, `tidy`, ...) are the proof kit's; this module adds the marking
dict for rubrics other than the proof one, polynomial text, and a few small number helpers.

Underscore-prefixed, so the generator registry does not import it as a generator module.
"""

from __future__ import annotations

from ..core import check
from ._free_proof_kit import Fm, Prompt, TAILS, attempts, finish, frac_plain, frac_tex, tidy  # noqa: F401

PROOF_CRITERIA = ("correctness", "logical_completeness", "definitions_assumptions", "communication")
PROOF_REQUIRED = ("correctness", "logical_completeness")

#: Criterion ids of the short-written-answer rubric.
SHORT_CRITERIA = ("correct_idea", "completeness", "use_of_context", "communication")
SHORT_REQUIRED = ("correct_idea", "completeness")


def marking(
    *,
    reference: list[str],
    steps: list[tuple[str, str]],
    traps: list[tuple[str, str]],
    criteria: tuple[str, ...] = PROOF_CRITERIA,
    required: tuple[str, ...] = PROOF_REQUIRED,
) -> dict:
    """The per-item marking dict. `steps` are (criterion id, what earns it); `traps` are (name, question)."""
    check(len(reference) >= 2, "a reference answer needs at least two lines")
    ids = {cid for cid, _ in steps}
    check(ids <= set(criteria), f"unknown criterion id in {ids - set(criteria)}")
    check(set(required) <= ids, f"key steps must cover {required}")
    check(len(traps) >= 2, "an item needs at least two named traps")
    check(len({name for name, _ in traps}) == len(traps), "trap names must be distinct")
    check(all("?" in q for _, q in traps), "every trap is asked as a question")
    return {
        "start": None,
        "target": None,
        "reference_lines": [tidy(line) for line in reference],
        "key_steps": [{"criterion_id": cid, "description": tidy(desc)} for cid, desc in steps],
        "traps": [{"name": name, "question": tidy(question)} for name, question in traps],
        "checks": [],
    }


def short_marking(*, reference, steps, traps) -> dict:
    return marking(reference=reference, steps=steps, traps=traps, criteria=SHORT_CRITERIA, required=SHORT_REQUIRED)


SHORT_TAILS = (
    "Answer in a few clear sentences and give a reason for each point.",
    "Write a short answer, and back each point with something from the description.",
    "Explain your answer in two to five sentences.",
    "Keep your answer short, but say why each point holds.",
)


def poly_fm(coeffs: list[int], var: str = "n") -> Fm:
    """The polynomial with ascending integer `coeffs` (constant term first), written highest power first."""
    tex_parts: list[str] = []
    txt_parts: list[str] = []
    for power in range(len(coeffs) - 1, -1, -1):
        c = coeffs[power]
        if c == 0:
            continue
        mag = abs(c)
        if power == 0:
            body_t = body_x = str(mag)
        else:
            lead = "" if mag == 1 else str(mag)
            body_t = lead + (var if power == 1 else f"{var}^{{{power}}}")
            body_x = lead + (var if power == 1 else f"{var}^{power}")
        if not txt_parts:
            sign_t = sign_x = "-" if c < 0 else ""
        else:
            sign_t = "-" if c < 0 else "+"
            sign_x = " - " if c < 0 else " + "
        tex_parts.append(sign_t + body_t)
        txt_parts.append(sign_x + body_x)
    if not txt_parts:
        return Fm("0", "0")
    return Fm("".join(tex_parts), "".join(txt_parts))


def poly_eval(coeffs: list[int], x: int) -> int:
    return sum(c * x**i for i, c in enumerate(coeffs))


def ordinal_list(items: list[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
