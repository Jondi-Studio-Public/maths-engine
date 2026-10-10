"""Generates `p19a_mult_fact_family` questions: from a multiplication fact in the 3, 4, 5 or 10 times tables,
state the product and a related division fact.

Scope: "Given a multiplication fact within the 3, 4, 5 or 10
times tables, state the fact's product and the quotient of one specified
related division fact derived from it."

## Boundary against the two prerequisites and the sharp neighbour

`multiplication-facts`/`whole-number-division-facts` each ask for ONE value
from ONE operation, over the full 1-10 range. This skill instead gives a KNOWN
multiplication fact restricted to the 3/4/5/10 families and asks for BOTH
the product AND the quotient of one specified related division fact derived
from that same fact -- the family-derivation LINK is what is graded, never
bare recall of either half alone. `f10-multiplication-division-inverse-unknown`
solves a stated equation with a missing-value placeholder for arbitrary
factors; this skill never writes a missing-value equation -- the multiplication
fact is always fully named (both factors given), and the learner derives a
SECOND fact (the related division) from it, never solving for an unknown
factor of the first fact itself.

## Why `answer: pair`, never two separate `number` skills

The skill's own name is "Derive multiplication and division fact families":
the pair (product, quotient) IS the family-derivation link in one question. A
bare-number product-only or quotient-only skill would re-derive one of the
two prerequisites exactly, which is the overlap to avoid.

## The product is never stated as a number in the prompt

Stating the product would hand the learner half of their own answer before
they have derived it. Every prompt instead presents the multiplication fact
with `?` standing for the (unstated) product, and asks about a division of
THAT `?` by a named divisor -- "{a} x {b} = ?. Use it to find ? / {b}."; the
answer is always `{"product": a*b, "quotient": <the specified quotient>}`.

## Family construction

One factor `f` is always drawn from {3, 4, 5, 10} (the skill's own families);
the other factor `k` is drawn from 2..10. The two are presented in a random
order (`first`, `second`) each seed, so which one is `f` and which is `k`
varies across the render -- the "3/4/5/10 family" constraint is on the SET
of factors, not on writing position.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the division undoes the fact exactly as written -- divide by
  `second` (the factor written second), so the quotient is `first`. DECISION:
  read which value the fact makes available as the divisor from its own
  written order.
- medium: the division uses the OTHER factor on 50% of draws (turnaround),
  mixed with easy's form on the other 50% so position alone never predicts
  the answer. DECISION: recognise that either factor of a known fact can be
  the divisor of a related division fact (the turnaround), not only the one
  written last.
- hard: the related division fact is stated in words -- quotitive ("how many
  groups of `d` make this product?") or partitive ("share this product
  equally into `d` groups; how many in each group?") -- naming a divisor `d`
  without ever writing a division symbol. DECISION: translate the wording
  into a division of the (still-unstated) product by `d`, a decoding step
  neither easy nor medium requires.

## Independent check

`_verify_pair` recomputes the product as `first * second` (construction's own
route) and independently re-derives it as `divisor * quotient` (multiplying
the OTHER way around) -- confirming the two routes agree via `check_equivalent`
-- for every seed, not just the ones under test.

## Misconceptions (every candidate a genuinely different (product, quotient)
pair, checked by `check_pairwise_distinct` including the answer)

- `wrong-factor`: the quotient taken as the OTHER family member instead of
  the one the fact/wording actually specifies (the classic
  partitive/quotitive-divisor confusion) -- same product, swapped quotient.
- `adjacent-fact`: one more group than asked for -- a product and quotient
  from the neighbouring fact `first x (second + 1)` (or the divisor-based
  equivalent), reported as if it answered the question asked.
- `swapped`: product and quotient exchanged, offered only when that pair is
  genuinely distinct from the answer.
- `quotient-off-by-one`: the correct product, quotient shifted by one.

The independent oracle
 parses `question.text` with its
own regexes and re-derives (product, quotient) from scratch, never importing
this module.
"""

from __future__ import annotations

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

TOPIC = "p19a_mult_fact_family"
_TIERS = ("easy", "medium", "hard")

_FAMILIES = (3, 4, 5, 10)
_OTHER_RANGE = range(2, 11)  # 2..10

_FORMAT_NOTE = "Give your answer as product = ..., quotient = ..."

_EASY_TEMPLATES = (
    "{first} × {second} = ?. Use it to find ? ÷ {second}. " + _FORMAT_NOTE,
    "{first} × {second} = ?. What is ? ÷ {second}? " + _FORMAT_NOTE,
    "Given {first} × {second} = ?, find the value of ? ÷ {second}. " + _FORMAT_NOTE,
)
_MEDIUM_SAME_AS_EASY = (
    "{first} × {second} = ?. Use it to find ? ÷ {second}. " + _FORMAT_NOTE,
    "{first} × {second} = ?. What is ? ÷ {second}? " + _FORMAT_NOTE,
    "Given {first} × {second} = ?, find the value of ? ÷ {second}. " + _FORMAT_NOTE,
)
_MEDIUM_TURNAROUND = (
    "{first} × {second} = ?. Use it to find ? ÷ {first}. " + _FORMAT_NOTE,
    "{first} × {second} = ?. What is ? ÷ {first}? " + _FORMAT_NOTE,
    "Given {first} × {second} = ?, find the value of ? ÷ {first}. " + _FORMAT_NOTE,
)
_HARD_QUOTITIVE = (
    "{first} × {second} = ?. How many groups of {divisor} make this product? " + _FORMAT_NOTE,
    "{first} × {second} = ?. How many groups of {divisor} does the product split into? " + _FORMAT_NOTE,
)
_HARD_PARTITIVE = (
    "{first} × {second} = ?. Share this product equally into {divisor} groups. How many in each group? " + _FORMAT_NOTE,
    "{first} × {second} = ?. Split this product equally between {divisor} groups. How many in each group? "
    + _FORMAT_NOTE,
)


def _draw_fact(rng):
    f = rng.choice(_FAMILIES)
    k = rng.choice(_OTHER_RANGE)
    if rng.random() < 0.5:
        first, second = f, k
    else:
        first, second = k, f
    return first, second


def _verify_pair(first: int, second: int, divisor: int, product: int, quotient: int):
    """Construction route: product = first * second, quotient = product //
    the OTHER factor. Independent route: divisor * quotient reconstructed
    and compared to `product` -- multiplication, never the same division
    that produced the quotient."""
    check(divisor * quotient == product, "p19a_mult_fact_family: divisor * quotient != product")
    check_equivalent(product, first * second, "p19a_mult_fact_family: product disagrees with first * second")


def _labelled(product: int, quotient: int) -> str:
    return f"product = {product}, quotient = {quotient}"


def _candidates(first, second, divisor, product, quotient):
    """Since `product == first * second` and `quotient == product //
    divisor`, `divisor` is always the OTHER family member than `quotient`
    (the two facts have only two factors between them) -- offering `divisor`
    itself as if it were the quotient is exactly the wrong-factor
    misconception."""
    raw = []
    # wrong-factor: same product, quotient swapped to the divisor itself
    # (the OTHER family member than the true quotient).
    raw.append((product, divisor))
    # adjacent-fact: one more group of `divisor`, i.e. the neighbouring fact.
    adj_quotient = quotient + 1
    adj_product = divisor * adj_quotient
    raw.append((adj_product, adj_quotient))
    # swapped: product and quotient exchanged.
    raw.append((quotient, product))
    # quotient off-by-one on the correct product.
    raw.append((product, quotient + 1))
    raw.append((product, max(quotient - 1, 0)))
    return raw


def _finalise(text, product, quotient, raw_candidates):
    answer_pair = (product, quotient)
    answer_text = _labelled(product, quotient)

    seen_pairs = [answer_pair]
    formatted = []
    for cand in raw_candidates:
        if cand in seen_pairs:
            continue
        seen_pairs.append(cand)
        formatted.append(_labelled(*cand))

    check(len(formatted) >= 1, "p19a_mult_fact_family: no distractor survived dedup")
    check_pairwise_distinct(
        *seen_pairs,
        msg="p19a_mult_fact_family: two options share the same (product, quotient) pair",
    )

    return dict(
        text=text,
        answer={"product": product, "quotient": quotient},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, formatted, n=3),
        working=[f"product = {product}", answer_text],
    )


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    first, second = _draw_fact(rng)
    product = first * second

    if tier == "easy":
        divisor, quotient = second, first
        _verify_pair(first, second, divisor, product, quotient)
        text = phrase(rng, _EASY_TEMPLATES, first=first, second=second)
        raw = _candidates(first, second, divisor, product, quotient)
        return _finalise(text, product, quotient, raw)

    if tier == "medium":
        turnaround = rng.random() < 0.5
        divisor = first if turnaround else second
        quotient = product // divisor
        _verify_pair(first, second, divisor, product, quotient)
        templates = _MEDIUM_TURNAROUND if turnaround else _MEDIUM_SAME_AS_EASY
        text = phrase(rng, templates, first=first, second=second, divisor=divisor)
        raw = _candidates(first, second, divisor, product, quotient)
        return _finalise(text, product, quotient, raw)

    # hard: worded quotitive/partitive division, divisor drawn independently.
    quotitive = rng.random() < 0.5
    divisor = first if rng.random() < 0.5 else second
    quotient = product // divisor
    _verify_pair(first, second, divisor, product, quotient)
    templates = _HARD_QUOTITIVE if quotitive else _HARD_PARTITIVE
    text = phrase(rng, templates, first=first, second=second, divisor=divisor)
    raw = _candidates(first, second, divisor, product, quotient)
    return _finalise(text, product, quotient, raw)


@generator(
    TOPIC,
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_mult_fact_family(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
