"""Generates `counterexample_identify` questions: pick the counterexample to a stated claim from displayed
candidate values.

Examples and counter-examples.

``counterexample_identify`` (senior-use-examples-and-counter-examples, ACMSM028) --
strictly deterministic by construction: the learner is NEVER asked to "think of" or
"supply" an example or counterexample. Every candidate value is drawn by the
generator and displayed in full; the learner only classifies or selects among the
values the question itself supplies.

Two variants, over a fixed predicate pool and a fixed domain ``range(2, 50)``:

- ``classify`` (easy/medium): a claim "for every n in D, n <P>" is made about a
  5-value set D that always contains one particular value ``c``. The learner
  decides whether ``c`` is an Example or a Counterexample to the claim. ``easy``
  draws its predicate from the eyeball-verifiable slice (even/odd); ``medium``
  draws from the slice requiring an actual computation (multiple-of-3/5, >20,
  perfect-square) -- selected by the ``pool`` parameter, since the
  variant name ("classify") is shared by both tiers and the predicate slice is
  the genuine escalation between them.

  The truth class (Example vs Counterexample) is CONSTRUCTED TO TARGET, never
  left to a uniform random pick of ``c`` from the domain: a predicate like "is a
  perfect square" is true for only ~12% of ``range(2, 50)``, so a uniform draw
  would make "Example" occur by luck far less than half the time. Instead the
  target label is chosen first (uniformly), and ``c`` is then sampled from
  whichever half of the domain satisfies that target.

- ``find_counterexample`` (hard): a claim "every one of the values ... <P>" lists
  four concrete values, three of which genuinely satisfy the predicate and one
  of which genuinely fails it. The learner selects the one value that disproves
  the claim.

Every predicate's own lambda is re-checked, on every draw, against an
independently coded route for that SAME predicate (a different algorithm, not
the same formula copy-pasted) -- ``_INDEPENDENT_CHECKS``, parallel to
``_PREDICATES`` by index.
"""

from __future__ import annotations

import math

from ..core import check, generator, latex_answer, pick_distractors

# ---------------------------------------------------------------------------
# Predicate pool. The third element of each tuple is prose naming the
# independent-check route implemented for that predicate below in
# ``_INDEPENDENT_CHECKS`` (same index) -- genuinely different code from the
# lambda's own comparison, never the same formula copy-pasted.

_PREDICATES = [
    ("is even", lambda n: n % 2 == 0, "check via the last binary digit: bin(n)[-1] == '0'"),
    ("is odd", lambda n: n % 2 == 1, "check via the last binary digit: bin(n)[-1] == '1'"),
    (
        "is a multiple of 3",
        lambda n: n % 3 == 0,
        "check via repeated subtraction of 3 down to 0 or a negative remainder",
    ),
    ("is a multiple of 5", lambda n: n % 5 == 0, "check via repeated subtraction of 5"),
    ("is greater than 20", lambda n: n > 20, "check via `not (n <= 20)`"),
    ("is a perfect square", lambda n: n >= 0 and math.isqrt(n) ** 2 == n, "check via `round(n ** 0.5) ** 2 == n`"),
]

_DOMAIN = range(2, 50)

_EASY_INDICES = (0, 1)
_MEDIUM_INDICES = (2, 3, 4, 5)


def _check_even_via_binary(n: int) -> bool:
    return bin(n)[-1] == "0"


def _check_odd_via_binary(n: int) -> bool:
    return bin(n)[-1] == "1"


def _check_multiple_of_3_via_subtraction(n: int) -> bool:
    remainder = n
    while remainder > 0:
        remainder -= 3
    return remainder == 0


def _check_multiple_of_5_via_subtraction(n: int) -> bool:
    remainder = n
    while remainder > 0:
        remainder -= 5
    return remainder == 0


def _check_greater_than_20_via_negation(n: int) -> bool:
    return not (n <= 20)


def _check_perfect_square_via_rounded_sqrt(n: int) -> bool:
    return round(n**0.5) ** 2 == n


# Parallel to `_PREDICATES` by index: `_INDEPENDENT_CHECKS[i]` is a genuinely
# different route to the same boolean as `_PREDICATES[i]`'s own lambda.
_INDEPENDENT_CHECKS = [
    _check_even_via_binary,
    _check_odd_via_binary,
    _check_multiple_of_3_via_subtraction,
    _check_multiple_of_5_via_subtraction,
    _check_greater_than_20_via_negation,
    _check_perfect_square_via_rounded_sqrt,
]

_CLASSIFY_OPTIONS = ["Example", "Counterexample", "Both", "Neither"]


def _predicate_pool_sizes() -> dict[str, tuple[int, int]]:
    """Diagnostic: (passing, failing) pool sizes for every predicate over
    `_DOMAIN`, for a 6-predicate pool-size report."""
    return {
        name: (len([x for x in _DOMAIN if fn(x)]), len([x for x in _DOMAIN if not fn(x)]))
        for name, fn, _route in _PREDICATES
    }


def _set_text(values) -> str:
    """Explicit set notation, never Python's `[1, 2, 3]` list rendering."""
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def _pad_to_5(c: int, rng) -> list[int]:
    """`{c}` unioned with 4 more distinct values drawn from `_DOMAIN`,
    resampling the padding set whenever the union collapses below 5 unique
    values (i.e. the padding draw happened to repeat `c` or itself)."""
    domain_list = list(_DOMAIN)
    while True:
        values = {c} | set(rng.sample(domain_list, 4))
        if len(values) == 5:
            return sorted(values)


def _classify(rng, indices):
    target = rng.choice(["Example", "Counterexample"])
    idx = rng.choice(indices)
    p_name, p_fn, _route = _PREDICATES[idx]
    want_true = target == "Example"

    pool = [x for x in _DOMAIN if p_fn(x) == want_true]
    check(pool, f"no candidates for predicate {p_name!r} target {target!r}")
    c = rng.choice(pool)

    independent = _INDEPENDENT_CHECKS[idx]
    check(
        independent(c) == want_true, f"independent check disagrees with construction for predicate {p_name!r} at n={c}"
    )

    D = _pad_to_5(c, rng)
    check(c in D, "constructed value c is missing from the displayed set D")

    text = f"Claim: for every n in {_set_text(D)}, n {p_name}. Is n = {c} an example or a counterexample to this claim?"
    answer_text = target
    check(
        answer_text != "Both" and answer_text != "Neither",
        "'Both'/'Neither' must never be the correct answer for a single value under a total predicate",
    )
    distractors = pick_distractors(answer_text, [o for o in _CLASSIFY_OPTIONS if o != answer_text])
    working = [
        f"Check n {p_name} for n = {c}: this is {'true' if want_true else 'false'}.",
        (
            f"A single value satisfying the claim's property is an example of it, so n = {c} is an example."
            if want_true
            else f"A single value failing the claim's property disproves it, so n = {c} is a counterexample."
        ),
    ]
    return text, answer_text, distractors, working


def _find_counterexample(rng):
    idx = rng.randrange(len(_PREDICATES))
    p_name, p_fn, _route = _PREDICATES[idx]
    independent = _INDEPENDENT_CHECKS[idx]

    passing_pool = [x for x in _DOMAIN if p_fn(x)]
    failing_pool = [x for x in _DOMAIN if not p_fn(x)]
    check(
        len(passing_pool) >= 3 and len(failing_pool) >= 1,
        f"predicate {p_name!r} pool too small: {len(passing_pool)} passing, {len(failing_pool)} failing",
    )

    passing_selected = rng.sample(passing_pool, 3)
    failing_selected = rng.choice(failing_pool)

    four_values = passing_selected + [failing_selected]
    rng.shuffle(four_values)

    text = (
        f"Claim: every one of the values {', '.join(map(str, four_values))} {p_name}. "
        "Which of these values disproves the claim?"
    )
    answer_text = str(failing_selected)

    check(
        independent(failing_selected) is False,
        f"independent check disagrees: {failing_selected} should fail predicate {p_name!r}",
    )
    for value in passing_selected:
        check(independent(value) is True, f"independent check disagrees: {value} should satisfy predicate {p_name!r}")

    distractor_candidates = [str(v) for v in passing_selected]
    distractors = pick_distractors(answer_text, distractor_candidates)
    working = [
        f"Check n {p_name} for each of {', '.join(map(str, four_values))}.",
        f"n = {failing_selected} is the only value that fails it, so it disproves the claim.",
    ]
    return text, answer_text, distractors, working


@generator(
    "counterexample_identify",
    level=12,
    presets={
        "easy": dict(variant="classify", pool="easy"),
        "medium": dict(variant="classify", pool="medium"),
        "hard": dict(variant="find_counterexample", pool=None),
    },
)
def counterexample_identify(rng, variant="classify", pool="easy"):
    """Classify a supplied value as Example/Counterexample to a claim (easy,
    medium), or select the one supplied value that disproves a claim among
    four (hard). Every candidate value is supplied by the question; the
    learner never constructs one.
    """
    if variant not in ("classify", "find_counterexample"):
        raise ValueError(f"variant must be 'classify' or 'find_counterexample'; got {variant!r}")

    if variant == "classify":
        if pool not in ("easy", "medium"):
            raise ValueError(f"pool must be 'easy' or 'medium' for variant='classify'; got {pool!r}")
        indices = _EASY_INDICES if pool == "easy" else _MEDIUM_INDICES
        text, answer_text, distractors, working = _classify(rng, indices)
    else:
        text, answer_text, distractors, working = _find_counterexample(rng)

    latex = f"plain:{text}"
    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )
