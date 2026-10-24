"""Exam-tier source for `probability-representations`: probabilities given in three different forms.

A group of students is described by the probability that a randomly chosen student falls in each of three
categories, one given as a decimal, one as a fraction and one as a percentage. Everyone else is in a fourth
category. The learner must put the three on a common footing before anything else can be done:

(a) the probability of the fourth category, as a percentage (convert, add, take from 100);
(b) how many students that is in this group (a percentage of the group size, follow-through from (a));
(c) a conditional share: of the students NOT in the fourth category, the percentage in the first
    (follow-through from (a)).

Built backwards from whole-student counts, so every figure shown is exact and every key is a whole percentage.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    rounded_decimal_string,
    stated_precision_string,
)
from ._exd_distinct_parts import distinct_part_answers

# Not 100: with a group of 100 the count in (b) is the percentage in (a), and the follow-through only restates it.
_SIZES = (25, 40, 50, 80, 200)

# (setting, first, second, third, fourth) -- each category phrase completes "a student who ...".
_CONTEXTS = (
    (
        "In a Year 8 group at a school in Geelong, {n} students were asked how they usually get to school.",
        "walks",
        "rides a bike",
        "catches the bus",
        "is driven",
    ),
    (
        "{n} students at a Brisbane high school each chose one summer sport.",
        "plays cricket",
        "plays tennis",
        "swims",
        "plays touch football",
    ),
    (
        "{n} students in a Perth Year 7 cohort each named their favourite canteen drink.",
        "chooses water",
        "chooses juice",
        "chooses milk",
        "chooses flavoured milk",
    ),
    (
        "{n} students in a surf life saving program at a Wollongong school each named their main job on patrol day.",
        "runs the beach flags",
        "paddles a board",
        "swims in the rescue squad",
        "helps in first aid",
    ),
)

_ASK_A = (
    "Find the probability that the student {fourth}. Give your answer as a percentage.",
    "Find, as a percentage, the probability that the student {fourth}.",
)
_IRREGULAR = {"is": "are", "has": "have"}


def _plural(phrase: str) -> str:
    """'plays tennis' -> 'play tennis', 'is driven' -> 'are driven' (the verb for several students)."""
    verb, _, rest = phrase.partition(" ")
    verb = _IRREGULAR.get(verb, verb[:-1] if verb.endswith("s") else verb)
    return f"{verb} {rest}".strip()


def _negative(phrase: str) -> str:
    """'plays tennis' -> 'does not play tennis', 'is driven' -> 'is not driven'."""
    verb, _, rest = phrase.partition(" ")
    if verb == "is":
        return f"is not {rest}"
    return f"does not {_plural(phrase)}"


def _dis(value: Fraction, tag: str | None, places: int = 2) -> Distractor:
    """A tagged distractor showing `value` to at most `places` decimal places (the value is the displayed one)."""
    return Distractor(rounded_decimal_string(value, places), tag, exact_value(round_half_up(value, places)))


def _pct(value: Fraction) -> str:
    return exact_decimal_string(value, 2)


@generator("exd_x_sc8_prob_rep_chain", level=9, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_sc8_prob_rep_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: convert a decimal, a fraction and a percentage probability, then find the remainder."""
    check(tier == "exam", f"unknown tier {tier!r}")
    setting, first, second, third, fourth = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a group whose figures are all whole percentages"):
        n = rng.choice(_SIZES)
        w, c, b = (rng.randint(2, n // 2) for _ in range(3))
        d = n - w - c - b
        if d < 2 or len({w, c, b, d}) < 4:
            continue
        # Every conversion must be a whole percentage, the fraction must not reduce to a unit fraction of a
        # silly size, and the conditional share in (c) must be a whole percentage too.
        if any((Fraction(x, n) * 100).denominator != 1 for x in (w, c, b, d)):
            continue
        if (Fraction(100 * w, n - d) * 10) % 1 in (0, Fraction(1, 2)):
            continue  # (c) is never a whole number or a rounding tie at 1 decimal place
        pw, pc = Fraction(w, n), Fraction(c, n)
        if pc.numerator == 1 and pc.denominator in (2, 4, 5, 10):
            continue  # too easy to recognise without converting
        key_a = Fraction(100 * d, n)
        key_c = Fraction(100 * w, n - d)
        shown = {str(n), exact_decimal_string(pw, 3), f"{pc.numerator}/{pc.denominator}", _pct(Fraction(100 * b, n))}
        if _pct(key_a) in shown or rounded_decimal_string(key_c, 1) in shown or str(d) in shown:
            continue
        if key_c == pw * 100:
            continue
        break

    check(Fraction(w + c + b + d) == n and (pw + pc + Fraction(b, n)) == 1 - Fraction(d, n), "counts do not sum")
    check(key_a == 100 - 100 * (pw + pc + Fraction(b, n)), "percentage of the fourth category disagrees")

    pw_text = exact_decimal_string(pw, 3)
    pc_text = f"{pc.numerator}/{pc.denominator}"
    pb_text = _pct(Fraction(100 * b, n))
    stem = (
        f"{setting.format(n=n)} A student is chosen at random. The probability that the student {first} is "
        f"{pw_text}. The probability that the student {second} is {pc_text}. The probability that the student "
        f"{third} is {pb_text}%. Every student is in exactly one of four groups: the student {first}, {second}, "
        f"{third} or {fourth}."
    )

    # Distractors, each a slip a learner makes while converting or combining.
    shown_pct = Fraction(100) * (pw + pc + Fraction(b, n))
    a_cands = [
        _dis(shown_pct, "found-complement-not-event"),
        _dis(100 - (pw + 100 * pc + Fraction(b * 100, n)), "decimal-percent-no-scaling"),
        _dis(100 - (100 * pw + pc.numerator + Fraction(b * 100, n)), "fraction-numerator-as-percent"),
        _dis(Fraction(b * 100, n), None),
    ]
    b_cands = [
        _dis(key_a, "percent-as-absolute"),
        _dis(n * key_a, "percent-not-divided-by-100"),
        _dis(Fraction(n - d), "found-complement-not-event"),
        _dis(Fraction(b), None),
        _dis(Fraction(c), None),
    ]
    c_cands = [
        _dis(pw * 100, "used-whole-group-as-denominator", 1),
        _dis(Fraction(100 * w, d), "used-wrong-subgroup-as-denominator", 1),
        _dis(Fraction(100 * w, n - b), None, 1),
        _dis(Fraction(100 * w, n - c), None, 1),
        _dis(Fraction(100 * (w + c), n - d), None, 1),
    ]
    key_a_text, key_b_text, key_c_text = _pct(key_a), str(d), stated_precision_string(round_half_up(key_c, 1), 1)
    a_dis = pick_distractors(key_a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    # A later distractor must not be the value a learner reaches by carrying a wrong (a) forward correctly: that
    # reply earns follow-through marks, so it cannot also be offered as a wrong answer.
    carried_b = [n * Fraction(x.text) / 100 for x in a_dis]
    carried_c = [pw * 10000 / (100 - Fraction(x.text)) for x in a_dis if Fraction(x.text) != 100]
    b_cands = [x for x in b_cands if all(abs(Fraction(x.text) - v) > GRADER_TOLERANCE for v in carried_b)]
    c_cands = [x for x in c_cands if all(abs(Fraction(x.text) - v) > GRADER_TOLERANCE for v in carried_c)]
    b_dis = pick_distractors(key_b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_dis = pick_distractors(key_c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    ask_a = rng.choice(_ASK_A).format(fourth=fourth)
    ask_c = (
        f"A student who {_negative(fourth)} is chosen at random. Find, as a percentage, the probability that "
        f"this student {first}."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=ask_a,
                answer_kind="number",
                answer=numeric_answer(key_a),
                answer_text=key_a_text,
                answer_latex=latex_answer(key_a_text + "%"),
                marks=3,
                sensible_range=(0, 100),
                working=[
                    f"{pw_text} = {_pct(pw * 100)}% and {pc_text} = {_pct(pc * 100)}%.",
                    f"P({first}, {second} or {third}) = {_pct(pw * 100)}% + {_pct(pc * 100)}% + {pb_text}% "
                    f"= {_pct(shown_pct)}%.",
                    f"P({fourth}) = 100% - {_pct(shown_pct)}% = {key_a_text}%.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=f"Hence find how many of the {n} students {_plural(fourth)}.",
                answer_kind="number",
                answer=d,
                answer_text=key_b_text,
                answer_latex=latex_answer(key_b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{n}*part_a/100",
                working=[f"{key_a_text}% of {n} = {key_a_text} / 100 x {n} = {d}."],
                distractors=b_dis,
            ),
            dict(
                prompt=ask_c,
                answer_kind="number",
                answer=numeric_answer(round_half_up(key_c, 1)),
                answer_text=key_c_text,
                answer_latex=latex_answer(key_c_text + "%"),
                marks=2,
                sensible_range=(0, 100),
                follow_through=f"{pw_text}*10000/(100-part_a)",
                working=[
                    f"P(a student who {_negative(fourth)}) = 100% - {key_a_text}% = {_pct(100 - key_a)}%.",
                    f"Share who {first}: {pw_text} / {_pct((100 - key_a) / 100)} = {key_c_text}% (1 d.p.).",
                ],
                distractors=c_dis,
            ),
        ],
    )
