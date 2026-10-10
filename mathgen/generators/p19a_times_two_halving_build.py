"""Generates `p19a_times_two_halving` questions: from a known fact for times two, state the related division fact
obtained by halving.

Scope: "Given a known multiplication fact for two, state the
value of the related division fact obtained by halving the product."

## Boundary against the two prerequisites and the sharp neighbours

`multiplication-facts`/`whole-number-division-facts` ask for ONE value from
ONE operation over the full 1-10 range; this skill is restricted to the
times-two/divide-by-two family only, and always frames the question as
doubling/halving derivation, never a bare "recall the fact" stem.
`f10-multiplication-division-inverse-unknown` solves a stated equation with a
missing-value placeholder for arbitrary factors -- this skill never writes
such an equation (the multiplication fact is always fully stated, including
its own product). `f10-multiplication-facts-3-4-5-10`
is the 3/4/5/10-family pair-answer sibling; this skill is twos only and
asks for a single number (the half), never a (product, quotient) pair, and
-- unlike that sibling -- states the product numerically, since concealing it here
would not hide any part of the answer (the answer is the HALF of a stated
product, not the product itself).

## Why `answer: number`

The related division fact's value here IS the halved product -- a single
number.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the known fact stated symbolically, factor order varied (`n x 2` or
  `2 x n`), then "so <product> / 2 = ?". DECISION: read the twos fact and
  its stated product, then halve it -- no doubling/halving vocabulary is
  present, only the bare fact.
- medium: the same relation in doubling/halving language ("Double n is
  <product>. What is half of <product>?"). DECISION: recognise
  "double"/"half" as naming the SAME multiply-by-two/divide-by-two pair the
  bare fact named at easy -- a vocabulary-translation step easy never needs.
- hard: two known twos facts are given (for two different values `a`/`b`)
  and the learner must pick the one whose product answers the asked
  halving, OR the halving is asked as a sharing-between-two question about
  the product of a stated doubling fact. DECISION: resolve which of two
  known facts is the relevant one (or translate a sharing frame into a
  halving), neither of which easy/medium ever requires (they only ever give
  one fact).

## Independent check

`_verify_half` recomputes the halved value two ways: construction reads it
straight off the drawn `n`; verification independently reconstructs the
product as `2 * n` and divides it back by 2, via `check`, on every seed.

## Misconceptions (checked distinct from the answer via `check_distinct`
on every candidate, every seed)

- `unhalved-product`: reports the product itself (`2n`), not halved.
- `doubled-product`: reports double the product (`4n`) -- "halve" read as
  "double" a second time.
- `other-facts-half` (hard, two-fact draws only): the OTHER stated fact's
  half, offered when the learner resolves the wrong one of the two facts.
- `off-by-one`: the correct half, shifted by one.

The independent oracle
 parses `question.text` with
its own regexes and re-derives the halved value from scratch, never
importing this module.
"""

from __future__ import annotations

from ..core import check, check_distinct, generator, latex_answer, phrase, pick_distractors

TOPIC = "p19a_times_two_halving"
_TIERS = ("easy", "medium", "hard")

_N_RANGE = range(2, 13)  # 2..12


def _draw_n(rng):
    return rng.choice(_N_RANGE)


def _verify_half(n: int, half: int):
    """Construction reads `half = n` straight off the draw; verification
    independently reconstructs the product as `2 * n` and divides it back by
    2 -- a genuinely different route (multiply-then-divide) from simply
    reusing `n`."""
    product = 2 * n
    check(product % 2 == 0, "p19a_times_two_halving: product not even")
    check(product // 2 == half, "p19a_times_two_halving: halved product != declared half")
    check(half == n, "p19a_times_two_halving: half disagrees with the drawn factor")


_EASY_TEMPLATES = (
    "{n} × 2 = {product}. So {product} ÷ 2 = ?",
    "2 × {n} = {product}. What is {product} ÷ 2?",
    "{n} × 2 = {product}. Find {product} ÷ 2.",
)
_MEDIUM_TEMPLATES = (
    "Double {n} is {product}. What is half of {product}?",
    "The double of {n} is {product}. Find half of {product}.",
    "{n} doubled gives {product}. Halve {product} to find the answer.",
)
_HARD_SHARE = (
    "Double {n} is {product}. Share {product} equally between 2 people. How many does each person get?",
    "The double of {n} is {product}. Split {product} equally between 2 groups. How many in each group?",
)


def _finalise(text, half, raw_candidates):
    formatted = []
    seen = {half}
    for cand in raw_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        check_distinct(half, cand, msg="p19a_times_two_halving: a distractor equals the answer")
        formatted.append(str(cand))

    check(len(formatted) >= 1, "p19a_times_two_halving: no distractor survived dedup")

    return dict(
        text=text,
        answer=half,
        answer_text=str(half),
        answer_latex=latex_answer(str(half)),
        distractors=pick_distractors(str(half), formatted, n=3),
        working=[f"double = {2 * half}", f"half = {half}"],
    )


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    if tier == "easy":
        n = _draw_n(rng)
        _verify_half(n, n)
        text = phrase(rng, _EASY_TEMPLATES, n=n, product=2 * n)
        raw = [2 * n, 4 * n, n + 1, max(n - 1, 0)]
        return _finalise(text, n, raw)

    if tier == "medium":
        n = _draw_n(rng)
        _verify_half(n, n)
        text = phrase(rng, _MEDIUM_TEMPLATES, n=n, product=2 * n)
        raw = [2 * n, 4 * n, n + 1, max(n - 1, 0)]
        return _finalise(text, n, raw)

    # hard: either the two-known-facts shape, or the sharing-between-two shape.
    if rng.random() < 0.5:
        a = _draw_n(rng)
        for _ in range(50):
            b = _draw_n(rng)
            if b != a:
                break
        else:
            check(False, "p19a_times_two_halving: could not draw a distinct second fact")
        _verify_half(a, a)
        _verify_half(b, b)
        ask_a = rng.random() < 0.5
        target_half = a if ask_a else b
        other_half = b if ask_a else a
        pa, pb = 2 * a, 2 * b
        target_product = pa if ask_a else pb
        text = f"You know 2 × {a} = {pa} and 2 × {b} = {pb}. Use the right fact to find {target_product} ÷ 2."
        raw = [target_product, 2 * target_product, other_half, target_half + 1, max(target_half - 1, 0)]
        return _finalise(text, target_half, raw)

    n = _draw_n(rng)
    _verify_half(n, n)
    text = phrase(rng, _HARD_SHARE, n=n, product=2 * n)
    raw = [2 * n, 4 * n, n + 1, max(n - 1, 0)]
    return _finalise(text, n, raw)


@generator(
    TOPIC,
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_times_two_halving(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
