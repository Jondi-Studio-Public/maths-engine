"""Independent oracle for the `p11e_number_discrete` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (`question.text`) and `question.answer_text`, and re-derives the expected
answer from scratch by a route that shares no code with the generator.

Imports nothing from `mathgen.generators.p11e_number_discrete_build` -- not a
helper, not a constant, not a formula. Both routes below are independent of
the generator's own construction:

- the CRT oracle brute-force-enumerates every integer in [0, m1*m2), never
  using a modular inverse (the generator's own construction route);
- the inclusion-exclusion oracle SEARCHES over every candidate intersection
  size and rebuilds explicit Python sets to test each one, never assuming
  the arithmetic formula |A union B| = |A| + |B| - |A intersect B| the
  generator itself applies.
"""

from __future__ import annotations

import re

from mathgen import make

CRT_TOPIC = "chinese_remainder_two_congruences"
IE_TOPIC = "inclusion_exclusion_two_sets"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)  # 60 seeds, matching the shared variety sweep


# --- Skill 1: chinese_remainder_two_congruences -----------------------------

_CRT_RE = re.compile(r"x ≡ (-?\d+) \(mod (\d+)\).*?x ≡ (-?\d+) \(mod (\d+)\)")


def _oracle_crt(text: str) -> int:
    m = _CRT_RE.search(text)
    assert m, f"unparseable chinese_remainder_two_congruences question: {text!r}"
    r1_raw, m1, r2_raw, m2 = (int(g) for g in m.groups())
    solutions = [x for x in range(m1 * m2) if x % m1 == r1_raw % m1 and x % m2 == r2_raw % m2]
    assert len(solutions) == 1, f"expected exactly one CRT solution, found {solutions} for {text!r}"
    return solutions[0]


def test_chinese_remainder_two_congruences_oracle_matches_every_seed_every_tier():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(CRT_TOPIC, seed=seed, difficulty=tier)
            expected = _oracle_crt(q.text)
            assert str(expected) == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


# --- Skill 2: inclusion_exclusion_two_sets ----------------------------------


def _set_union_len(a_size: int, b_size: int, inter: int) -> int:
    """Explicit finite sets of consecutive integers realising the given
    sizes, overlapped by exactly `inter` elements."""
    a = set(range(a_size))
    b = set(range(a_size - inter, a_size - inter + b_size))
    assert len(a & b) == inter
    return len(a | b)


def _oracle_ie(text: str) -> int:
    numbers = [int(tok) for tok in re.findall(r"\d+", text)]
    if text.startswith("In a survey of"):
        total, a_size, b_size, neither = numbers
        union = total - neither
    elif text.rstrip().endswith("or both?"):
        # easy: |A|, |B|, |A intersect B| given directly; find |A union B|.
        a_size, b_size, inter = numbers
        return _set_union_len(a_size, b_size, inter)
    else:
        # medium: |A|, |B|, |A union B| given directly; find |A intersect B|.
        a_size, b_size, union = numbers

    # Search every candidate intersection size and rebuild the explicit sets
    # for each -- never assumes the arithmetic rearrangement.
    for inter in range(min(a_size, b_size) + 1):
        if _set_union_len(a_size, b_size, inter) == union:
            return inter
    raise AssertionError(f"no intersection size in range reproduces union {union} for {text!r}")


def test_inclusion_exclusion_two_sets_oracle_matches_every_seed_every_tier():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(IE_TOPIC, seed=seed, difficulty=tier)
            expected = _oracle_ie(q.text)
            assert str(expected) == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)
