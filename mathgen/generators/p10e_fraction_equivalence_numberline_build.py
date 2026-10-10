"""Builds the `f10-fraction-equivalence-numberline` skill (AC9M6N03, AC9M7N04): ordering fractions and
recognising equivalent fractions on a number line.

``p10e_fraction_equivalence_numberline`` -- order a SUPPLIED SET of halves, thirds and quarters from smallest
to largest (``answer: ordered_list``, ``grade_ordered_list`` -- exact position, no partial credit, matching
the sibling ``f10-decimal-order-numberline``'s own shape). Not a duplicate of the catalogue skill
``compare-common-fractions``: that skill is a PAIRWISE ``mcq`` comparison of two fractions; this one orders a
whole supplied set of three or four values.

The skill restricts every value's denominator to {2, 3, 4}, so the only proper-fraction VALUES available
at all are ``{1/4, 1/3, 1/2, 2/3, 3/4}`` -- exactly five.

The hard tier additionally displays exactly one of the four drawn values in a
non-canonical, unreduced EQUIVALENT form (e.g. "1/3" shown as "2/6"),
operationalising the "use equivalent representations to
compare" clause: the learner must recognise the hidden equivalence before
they can place it. Every option (the answer and all three distractors)
redisplays that one altered item in the SAME unreduced form it was shown in
the question -- never silently reduced back -- so no option is identifiable
by its own formatting, and the fraction the learner is asked to place is
literally the string they read.

Every value is held as an exact ``fractions.Fraction``; the independent check
never reuses ``sorted()`` for the ordering claim itself -- it walks every
ordered pair ``(i, j)`` with ``i < j`` in the claimed answer and asserts
strictly ascending ``Fraction`` order directly, and separately confirms the
answer is a permutation of the exact drawn multiset of VALUES (compared by
value, not by displayed string) -- the same two-part check
``p9e_decimal_accuracy_order_build.py``'s own ``_verify_ascending_permutation``
uses, this module's worked template.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors, reversed_and_adjacent_swap

_RESAMPLE_ATTEMPTS = 400

_VALUES = [Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4)]


def _verify_ascending_permutation(drawn: list[Fraction], ordered: list[Fraction]) -> None:
    """Independent check, worked template
    ``p9e_decimal_accuracy_order_build.py::_verify_ascending_permutation``:
    never reuses ``sorted()`` for the ordering claim -- walks every pair
    directly -- and separately confirms ``ordered`` is a permutation of the
    exact drawn VALUE multiset."""
    check(sorted(drawn) == sorted(ordered), "ordered list is not a permutation of the drawn values")
    check(len(ordered) == len(set(ordered)), "drawn values must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            check(ordered[i] < ordered[j], f"{ordered[i]} does not precede {ordered[j]}")


def _build(rng, tier: str) -> dict:
    subset_size = 3 if tier == "easy" else 4
    for _ in range(_RESAMPLE_ATTEMPTS):
        drawn = rng.sample(_VALUES, subset_size)
        display: dict[Fraction, str] = {v: f"{v.numerator}/{v.denominator}" for v in drawn}

        if tier == "hard":
            altered = rng.choice(drawn)
            k = rng.choice([2, 3])
            display[altered] = f"{altered.numerator * k}/{altered.denominator * k}"

        def text_of(v: Fraction) -> str:
            return display[v]

        ordered_values = sorted(drawn)
        _verify_ascending_permutation(drawn, ordered_values)

        reversed_values, adjacent_values = reversed_and_adjacent_swap(ordered_values)

        # denominator-then-numerator-order: sorts by the DISPLAYED
        # denominator ascending, then the displayed numerator ascending --
        # ignoring the actual fraction VALUE. Uses the DISPLAY (num, den)
        # pair (post hard-tier alteration), the digits a learner misreading
        # the fraction would actually see. Resample the whole draw on a
        # collision with the true order, rather than ship two "different"
        # options that are secretly the same sequence (the same pattern
        # p9e_decimal_order_numberline uses for its own
        # longer_is_bigger/shorter_is_bigger collision).
        def _display_pair(v: Fraction) -> tuple[int, int]:
            n, d = display[v].split("/")
            return int(d), int(n)

        dn_order = sorted(drawn, key=_display_pair)
        if dn_order == ordered_values:
            continue

        answer_text = ", ".join(text_of(v) for v in ordered_values)
        reversed_text = ", ".join(text_of(v) for v in reversed_values)
        adjacent_text = ", ".join(text_of(v) for v in adjacent_values)
        dn_text = ", ".join(text_of(v) for v in dn_order)

        candidates = [reversed_text, adjacent_text, dn_text]
        distractors = pick_distractors(answer_text, candidates, n=3)
        if len(distractors) < 2:
            continue

        display_order = list(drawn)
        rng.shuffle(display_order)
        question_text = ", ".join(text_of(v) for v in display_order)

        working = [
            "Rewrite any fraction that is not already a half, third or quarter in its reduced form, then compare.",
            f"In order: {answer_text}.",
        ]
        return dict(
            text=f"Arrange these fractions from smallest to largest: {question_text}.",
            latex=r"\text{Arrange from smallest to largest: }" + question_text,
            answer=list(ordered_values),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=working,
            distractors=distractors,
        )

    raise RuntimeError(
        f"could not build a valid fraction_equivalence_numberline question for tier={tier!r} "
        f"in {_RESAMPLE_ATTEMPTS} attempts"
    )


@generator(
    "p10e_fraction_equivalence_numberline",
    level=6,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10e_fraction_equivalence_numberline(rng, tier="easy"):
    """Order a supplied set of halves/thirds/quarters, smallest to largest.

    easy: 3 of the 5 possible values, each shown in reduced canonical form.
    medium: 4 of the 5 values (one more than easy -- with only 5 values total,
    this makes it near-certain all three denominators are present at once, a
    genuine escalation in how many denominators must be reconciled). hard:
    4 values, same as medium, but exactly one is displayed in a non-canonical
    unreduced equivalent form -- the learner must recognise the hidden
    equivalence before ordering it.
    """
    return _build(rng, tier)
