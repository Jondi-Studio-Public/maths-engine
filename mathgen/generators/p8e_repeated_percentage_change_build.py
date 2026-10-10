"""Repeated percentage change, senior ``acara-senior-v8-4``, domain ``arithmetic``, ``answer: number``.

One generator, ``repeated_percentage_change``
(``senior-determine-the-overall-change-in-a-quantity-following``, ACMEM063) --
a starting quantity undergoes two (easy/medium) or three (hard) successive
percentage changes, applied IN ORDER, and the question asks for the FINAL
quantity, never an abstract "overall percentage change" (which would raise a
sign-convention ambiguity the final-quantity framing avoids entirely, since
the final quantity is always a plain positive number).

THE MISCONCEPTION THIS SKILL EXISTS TO CATCH: successive percentage changes
do not add. A 10% increase followed by a 10% decrease is a net 1% decrease
(1.10 * 0.90 = 0.99), not 0%. Every question offers "sum the percentages and
apply once to the original quantity" (``adds-the-percentages``) as its
leading distractor, engineered to be a genuinely different value from the
true compounded answer on every single seed -- see the module-level proof
notes on ``_draw`` below for exactly how.

Difficulty (genuine reasoning escalation, not span):
- easy: two changes, SAME direction (both increases or both decreases). The
  naive-additive answer is directionally right (same sign of net change) but
  always the wrong magnitude.
- medium: two changes, OPPOSITE directions -- the naive-additive answer can
  now get the DIRECTION of the net change wrong, most starkly in the classic
  equal-and-opposite case (a rise of p% then a fall of the same p%), where
  "add the percentages" says "no net change" but the true compounded value is
  always a net decrease.
- hard: THREE successive changes with MIXED directions (never all three the
  same sign) -- a genuinely harder reasoning load, tracking three
  multiplicative factors instead of two.

Exactness: every quantity is drawn so that EVERY PREFIX of the compounding
chain lands on an exact integer -- not merely the final answer. A population,
an attendance figure, a subscriber count or a stock level cannot legitimately
sit at a fractional value at any intermediate step, so "the population after
the first change" must itself be a whole number a learner could sensibly
round from without ambiguity; if only the final product were constrained,
some seeds would present an intermediate step (e.g. "137.5 people") that a
careful learner could read two ways. This can't be solved by picking Q as a
fixed multiple of some N (see ``_draw``'s docstring for why), so ``_draw``
draws Q and the percentages together and RESAMPLES whenever any prefix product
is not an exact integer or the final answer is >= 1000, the same resampling
discipline ``mathgen/generators/p7d_rates_build.py``'s ``_compare_hard`` uses.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.core import (
    check,
    check_equivalent,
    generator,
    pick_distractors,
    round_half_up,
)

# Nonzero multiples of 10, bounded to +-40. Bounding the magnitude keeps every
# per-step factor in [0.6, 1.4], so Q in a modest 100-400 range can compound
# two or three of them and still land under 1000 (worst case, three +40%
# changes: 400 * 1.4**3 = 1097.6 -- close enough to the bound that resampling
# away from it is routine, not exceptional).
_CHANGE_VALUES = (-40, -30, -20, -10, 10, 20, 30, 40)
_POS_CHANGES = tuple(v for v in _CHANGE_VALUES if v > 0)
_NEG_CHANGES = tuple(v for v in _CHANGE_VALUES if v < 0)

_Q_LO, _Q_HI = 100, 400

# Bounded resample attempts (mirrors p7d_rates_build.py's `_RESAMPLE_ATTEMPTS` pattern). Measured:
# two-change draws need at most a few hundred attempts; three-change draws (rarer, since three
# independent prefix-integrality constraints plus the mixed-direction requirement plus the collision
# guard all have to clear at once) were observed up to ~3800 attempts in a 20000-draw simulation, so
# the cap here carries a large safety margin over the observed maximum.
_RESAMPLE_ATTEMPTS_2 = 3000
_RESAMPLE_ATTEMPTS_3 = 12000

_CONTEXTS = (
    ("A town's population", "population"),
    ("The attendance at a concert", "attendance"),
    ("The number of subscribers to a streaming service", "number of subscribers"),
    ("The stock level in a warehouse", "stock level"),
)


def _phrase(p: int) -> str:
    return f"it increases by {p}%" if p > 0 else f"it decreases by {-p}%"


def _factor_str(p: int) -> str:
    # Every p is a multiple of 10, so (100 + p) / 100 always has exactly one
    # decimal digit (0.6 .. 1.4) -- ":.1f" never hides or invents precision.
    return f"{(100 + p) / 100:.1f}"


def _compound_prefixes(q: int, percentages: tuple[int, ...]) -> list[Fraction] | None:
    """[Q, Q*f1, Q*f1*f2, ...] as exact Fractions, or None the first time a
    prefix is not an exact integer (the caller resamples in that case)."""
    running = Fraction(q)
    prefixes = [running]
    for p in percentages:
        running = running * Fraction(100 + p, 100)
        if running.denominator != 1:
            return None
        prefixes.append(running)
    return prefixes


def _draw(rng, pick_changes, attempts: int) -> tuple[int, tuple[int, ...], list[int], int, int, int]:
    """Draw (Q, percentages, integer prefix trace) satisfying, together, in one
    resample loop (never a closed-form divisibility rule -- see the module
    docstring for why a fixed multiple of Q cannot work for the three-change
    case):

    1. EVERY prefix of the compounding chain (including the final answer) is
       an exact integer -- Fraction, never float, and never a fixed multiple
       of Q, because a Q large enough to guarantee that for three changes
       (a multiple of 1000) makes the largest three-increase combination
       exceed 1000 (1000 * 1.4**3 = 2744).
    2. The final answer is < 1000 (so a typed answer never needs a thousands
       separator; `mathgen/grading/typed.py`'s typed-number path does not yet accept
       one -- confirmed by reading the module, not assumed).
    3. `adds_frac` (Q compounded via the SUMMED percentage, applied once to
       the original Q -- the "adds-the-percentages" misconception) differs
       from the true compounded answer by at least a whole unit. For a
       two-change draw this is ALWAYS true by construction (see the proof
       below) and this condition never actually resamples anything; for a
       three-change draw it is not structurally guaranteed (the xy+yz+zx+xyz
       gap can be arbitrarily small, even zero, for some (p1,p2,p3) -- see
       the module docstring) and this is the genuine runtime guard.
    4. The four values {true answer, "applies only the first change",
       rounded "adds-the-percentages", rounded "applies only the last
       change"} are pairwise distinct, so all three distractors are usable
       and none silently collides with another distractor or the answer.

    PROOF that condition 3 never fires for a two-change draw: writing the
    naive-additive value as Q*(1 + (p1+p2)/100), it differs from the true
    compounded value Q*(1+p1/100)*(1+p2/100) by exactly Q*p1*p2/10000. Since
    Q >= 100 and |p1|, |p2| >= 10 here, |Q*p1*p2/10000| >= 100*10*10/10000 =
    1 whenever p1 != 0 and p2 != 0 -- which every draw here satisfies by
    construction (every value in _CHANGE_VALUES is nonzero). So the gap is
    always at least 1 for two changes; condition 3 is a redundant safety net
    at that tier, not a real constraint.
    """
    for _ in range(attempts):
        q = rng.randint(_Q_LO, _Q_HI)
        percentages = pick_changes(rng)
        if percentages is None:
            continue
        prefixes = _compound_prefixes(q, percentages)
        if prefixes is None:
            continue
        true_answer = prefixes[-1]
        if true_answer.denominator != 1 or true_answer >= 1000:
            continue
        true_answer = int(true_answer)

        adds_frac = q * Fraction(100 + sum(percentages), 100)
        if abs(adds_frac - true_answer) < 1:
            continue

        first_only = int(prefixes[1])  # exact by construction (a genuine prefix)
        last_only_frac = q * Fraction(100 + percentages[-1], 100)
        check(adds_frac >= 0, "adds-the-percentages distractor value must not be negative")
        adds_rounded = int(round_half_up(adds_frac, 0))
        check(last_only_frac >= 0, "applies-only-the-last-change distractor value must not be negative")
        last_rounded = int(round_half_up(last_only_frac, 0))

        if len({true_answer, first_only, adds_rounded, last_rounded}) != 4:
            continue

        check(
            all(v > 0 for v in (true_answer, first_only, adds_rounded, last_rounded)),
            "every displayed quantity must be a plain positive number",
        )

        return q, percentages, [int(p) for p in prefixes], adds_rounded, first_only, last_rounded
    raise ValueError("repeated_percentage_change: no collision-free draw found")


def _pick_easy(rng) -> tuple[int, ...] | None:
    pool = _POS_CHANGES if rng.choice((True, False)) else _NEG_CHANGES
    return (rng.choice(pool), rng.choice(pool))


def _pick_medium(rng) -> tuple[int, ...] | None:
    p1 = rng.choice(_CHANGE_VALUES)
    opposite_pool = _NEG_CHANGES if p1 > 0 else _POS_CHANGES
    p2 = rng.choice(opposite_pool)
    return (p1, p2)


def _pick_hard(rng) -> tuple[int, ...] | None:
    ps = tuple(rng.choice(_CHANGE_VALUES) for _ in range(3))
    if len({1 if p > 0 else -1 for p in ps}) == 1:
        return None  # all three the same direction -- not "mixed", resample
    return ps


# `text` must independently reproduce Q and the signed percentages so the
# module's own self-check (below) never trusts the construction's local
# variables -- only what the rendered sentence actually says.
_Q_RE = re.compile(r"starts at (\d+)\.")
_CHANGE_RE = re.compile(r"(?:First|Then) it (increases|decreases) by (\d+)%\.")


def _verify_from_text(text: str, expected_answer: int) -> None:
    """Independent in-generator check: parse Q and every signed percentage
    back out of the rendered sentence (never the generator's own local `q`/
    `percentages`), recompute the compounded final value as an exact
    Fraction product, and assert it equals the constructed answer. This is a
    genuinely different implementation from the test file's own oracle, which never
    imports this module at all."""
    q_match = _Q_RE.search(text)
    check(q_match is not None, "self-check: could not parse starting quantity from text")
    q = int(q_match.group(1))
    changes = _CHANGE_RE.findall(text)
    check(len(changes) in (2, 3), "self-check: could not parse 2 or 3 percentage changes from text")
    value = Fraction(q)
    for direction, pct in changes:
        signed = int(pct) if direction == "increases" else -int(pct)
        value *= Fraction(100 + signed, 100)
    check_equivalent(value, expected_answer, "self-check: recomputed compounded value from text must match the answer")


def _build(rng, pick_changes, attempts: int) -> dict:
    q, percentages, prefixes, adds_rounded, first_only, last_rounded = _draw(rng, pick_changes, attempts)
    true_answer = prefixes[-1]

    subject, noun = rng.choice(_CONTEXTS)
    sentences = [f"{subject} starts at {q}."]
    for i, p in enumerate(percentages):
        lead = "First" if i == 0 else "Then"
        sentences.append(f"{lead} {_phrase(p)}.")
    sentences.append(f"What is the final {noun}?")
    text = " ".join(sentences)

    _verify_from_text(text, true_answer)

    working = []
    for i, p in enumerate(percentages):
        prev, nxt = prefixes[i], prefixes[i + 1]
        verb = "increases" if p > 0 else "decreases"
        working.append(f"{prev} {verb} by {abs(p)}%: {prev} x {_factor_str(p)} = {nxt}.")

    answer_text = str(true_answer)
    candidates = [str(adds_rounded), str(first_only), str(last_rounded)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "expects exactly 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=true_answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "repeated_percentage_change",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def repeated_percentage_change(rng, mode="easy"):
    """A starting quantity undergoes two (easy/medium) or three (hard)
    successive percentage changes, applied in order; the question asks for
    the final quantity. easy: two changes, same direction. medium: two
    changes, opposite directions (includes the classic equal-and-opposite
    case, p1 = -p2, on about 1 in 4 draws). hard: three changes, mixed
    directions, tracking three multiplicative factors."""
    if mode == "easy":
        return _build(rng, _pick_easy, _RESAMPLE_ATTEMPTS_2)
    if mode == "medium":
        return _build(rng, _pick_medium, _RESAMPLE_ATTEMPTS_2)
    if mode == "hard":
        return _build(rng, _pick_hard, _RESAMPLE_ATTEMPTS_3)
    raise ValueError(f"unknown mode: {mode!r}")
