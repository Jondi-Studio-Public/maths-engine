"""Generates `p19a_add_sub_fact_family` questions: from an addition fact through 20 and one addend, state the
related subtraction fact.

Scope: "Given a known addition fact through 20 and one of its two
addends, state the value of the related subtraction fact formed by
subtracting that addend from the sum."

## BOUNDARY with `add-within-20` / `subtract-within-20`

Those two prerequisites each compute one bare operation from scratch. This
skill GIVES the addition fact outright and asks for a subtraction value that
the fact already contains -- the graded step is reading which addend the
fact leaves over once the stated one is subtracted, never computing a fresh
sum or difference from raw numbers. It never poses a missing-value equation
(`box + 5 = 13`); that belongs to the planned
`f10-unknown-add-sub-number-equations`.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the fact is always written addend-first ("a + b = s"), and the
  subtraction always removes the addend written SECOND. DECISION: read off
  the addend that is not being subtracted.
- medium: the subtracted addend is either one (so which of the two is left
  over varies), and the fact is sometimes written sum-first ("s = a + b").
  DECISION: the same addend-left-over reading as easy, now unavailable via a
  fixed "subtract the second one" rule -- the learner must identify which
  named addend is the subtrahend and work out what remains, regardless of
  which side of the "=" the sum sits on.
- hard: two known facts sharing the same sum are given, and the subtraction
  asks for `sum - x` where `x` is an addend from the SECOND fact only.
  DECISION: medium's addend-left-over reading, PLUS picking the specific fact
  that actually contains the number being subtracted out of two candidate
  facts, rather than assuming there is only one fact to read from.

## Independent check for every draw

`_check_fact` recomputes each stated addition fact's own arithmetic (a + b
== s) as a sanity check on construction, and `check_equivalent` verifies the
key equals `sum - subtrahend` computed directly by subtraction -- a different
route from simply reading off "the other addend" that construction used to
pick it. `check_pairwise_distinct` covers the key against every distractor
candidate before formatting.

The oracle parses
`question.text` with its own regexes over the same fixed "a + b = s" / "s = a
+ b" vocabulary, re-derives the minuend/subtrahend from the final "m - n"
expression, confirms that minuend is some stated fact's sum and subtrahend is
one of that fact's addends, and recomputes the key as `minuend - subtrahend`
-- never importing this module.
"""

from __future__ import annotations

from ..core import check, check_equivalent, check_pairwise_distinct, generator, phrase

TOPIC = "p19a_add_sub_fact_family"
_TIERS = ("easy", "medium", "hard")


def _check_fact(a: int, b: int, s: int, msg: str) -> None:
    """Independent route: confirm the stated fact's own arithmetic (never
    re-using whatever produced `s` in the first place -- construction always
    computes `s = a + b` directly, so this re-adds and compares)."""
    check(a + b == s, msg)


def _candidates(x: int, s: int, key: int) -> list[int]:
    """Misconception candidates, in priority order:
    - `x` itself: names the subtracted addend instead of the one left over.
    - `s`: restates the sum, as if the subtraction did nothing.
    - `s + x`: adds instead of subtracting.
    - `key - 1`, `key + 1`: an off-by-one slip on the correct value.
    """
    return [x, s, s + x, key - 1, key + 1]


def _finalise(text: str, key: int, raw_candidates: list[int]) -> dict:
    seen = {key}
    distinct_ints: list[int] = []
    distractors: list[str] = []
    for c in raw_candidates:
        if c in seen:
            continue
        seen.add(c)
        distinct_ints.append(c)
        distractors.append(str(c))
    check(len(distractors) >= 1, "p19a_add_sub_fact_family: no distractor survived dedup")
    check_pairwise_distinct(
        key, *distinct_ints, msg="p19a_add_sub_fact_family: a distractor collides with the key or another distractor"
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        distractors=distractors,
        working=[f"answer: {key}"],
    )


_EASY = (
    "{a} + {b} = {s}. So {s} - {b} = ?",
    "We know {a} + {b} = {s}. What is {s} - {b}?",
    "{a} + {b} = {s}. What does {s} - {b} equal?",
)

_MEDIUM_NORMAL = (
    "{a} + {b} = {s}. What is {s} - {x}?",
    "We know {a} + {b} = {s}. Find {s} - {x}.",
)
_MEDIUM_SUMFIRST = (
    "{s} = {a} + {b}. What is {s} - {x}?",
    "{s} = {a} + {b}. Find {s} - {x}.",
)

_HARD = (
    "You know {a1} + {b1} = {s} and {a2} + {b2} = {s}. Use one of them to find {s} - {x}.",
    "{a1} + {b1} = {s} and {a2} + {b2} = {s}. What is {s} - {x}?",
    "Given {a1} + {b1} = {s} and {a2} + {b2} = {s}, work out {s} - {x}.",
)


def _draw_simple_fact(rng) -> tuple[int, int, int]:
    """a in 1-15, b in 1-(20-a), so a + b <= 20 always. Wide enough spread
    on `a` (the easy-tier key) that no single value dominates 200 draws."""
    a = rng.randint(1, 15)
    b = rng.randint(1, 20 - a)
    s = a + b
    _check_fact(a, b, s, "p19a_add_sub_fact_family: easy/medium fact does not add up")
    return a, b, s


def _draw_two_facts(rng) -> tuple[int, int, int, int, int]:
    """Sum `s` (6-20) with two DISTINCT decompositions (a1, b1) and
    (a2, b2). Rejection-sampled: small sums have few decompositions, so a
    short retry loop is needed to guarantee two different ones exist."""
    s = rng.randint(6, 20)
    for _ in range(200):
        a1 = rng.randint(1, s - 1)
        b1 = s - a1
        a2 = rng.randint(1, s - 1)
        b2 = s - a2
        if {a1, b1} != {a2, b2}:
            _check_fact(a1, b1, s, "p19a_add_sub_fact_family: hard fact 1 does not add up")
            _check_fact(a2, b2, s, "p19a_add_sub_fact_family: hard fact 2 does not add up")
            return a1, b1, a2, b2, s
    check(False, "p19a_add_sub_fact_family: could not draw two distinct decompositions")


def _build_easy(rng) -> dict:
    a, b, s = _draw_simple_fact(rng)
    key = a
    check_equivalent(key, s - b, "p19a_add_sub_fact_family easy: key does not equal sum - subtrahend")
    text = phrase(rng, _EASY, a=a, b=b, s=s)
    return _finalise(text, key, _candidates(b, s, key))


def _build_medium(rng) -> dict:
    a, b, s = _draw_simple_fact(rng)
    subtract_a = rng.random() < 0.5
    x = a if subtract_a else b
    key = b if subtract_a else a
    check_equivalent(key, s - x, "p19a_add_sub_fact_family medium: key does not equal sum - subtrahend")
    sum_first = rng.random() < 0.5
    templates = _MEDIUM_SUMFIRST if sum_first else _MEDIUM_NORMAL
    text = phrase(rng, templates, a=a, b=b, s=s, x=x)
    return _finalise(text, key, _candidates(x, s, key))


def _build_hard(rng) -> dict:
    a1, b1, a2, b2, s = _draw_two_facts(rng)
    x = b2
    key = a2
    check_equivalent(key, s - x, "p19a_add_sub_fact_family hard: key does not equal sum - subtrahend")
    text = phrase(rng, _HARD, a1=a1, b1=b1, a2=a2, b2=b2, s=s, x=x)
    return _finalise(text, key, _candidates(x, s, key))


def _build(rng, tier: str = "easy") -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    return _build_hard(rng)


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_add_sub_fact_family(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
