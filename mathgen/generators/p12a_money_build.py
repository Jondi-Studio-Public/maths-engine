"""Money representation + operation-choice model, domain ``financial-discrete``, both ``answer: number``.

Two generators/skills:

- ``money_dollars_cents_representation`` (``f10-money-dollars-cents-representation``,
  AC9M3M06, virgin) -- pure unit conversion between dollars-and-cents notation and
  total cents, never a payment/change shape (that is ``money_change``'s own job).
  Two directions: dollars-and-cents -> cents (``d2c``) and cents -> dollars-and-cents
  (``c2d``). Amounts always stay under $10.00 (cents < 1000).

- ``additive_multiplicative_money_model`` (``f10-additive-multiplicative-money-model``,
  AC9M2N06, non-virgin -- ``money_change`` already maps this area) -- given a stated
  money situation, choose which of the four operations (add/subtract/multiply/divide)
  models it and compute the result. All four operations are live at every difficulty;
  the difficulty label never predicts the operation. The subtraction branch is always a
  balance-reduction ("has $X, spends $Y, how much left"), never a payment/change
  transaction -- that shape belongs to ``money_change`` on this same target area.

Both modules keep every value on an exact ``Fraction`` grid and use
``mathgen.core.check``/``check_pairwise_distinct``/``check_equivalent`` for the
in-generator independent checks (a route genuinely different from however construction
built the string/chose the operation -- see each ``_build_*`` function's own comment).
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    fixed_point_string,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from ..slots import NAMES

# ---------------------------------------------------------------------------
# f10-money-dollars-cents-representation
# ---------------------------------------------------------------------------

# Dollars-and-cents -> cents templates. `{amt}` is a "$D.cc"/"$D" string.
_D2C_TEMPLATES = [
    "Write {amt} in cents.",
    "Convert {amt} to cents.",
    # hard-only: never drawn at easy/medium, guarantees a new prompt_shape().
    "A price tag shows {amt}. What is this price in cents?",
]

# Cents -> dollars-and-cents templates. `{amt}` is a bare cents integer,
# `{unit}` is "cent"/"cents" (the hard tier's minimum cents value is 1, so
# the singular must be handled -- see `_build_c2d`).
_C2D_TEMPLATES = [
    "Write {amt} {unit} in dollars.",
    "Convert {amt} {unit} to dollars.",
    # hard-only: never drawn at easy/medium, guarantees a new prompt_shape().
    "A coin jar contains {amt} {unit}. What is this in dollars?",
]


def _d2c_value(rng, tier: str) -> tuple[int, int]:
    """Draw (dollars, frac_cents) for a dollars-and-cents -> cents question.

    easy: frac_cents always 0 (whole-dollar amounts only).
    medium: frac_cents in [10, 99] -- always two digits, never the
      dollars-to-cents regrouping trap (a single-digit fractional part).
    hard: frac_cents in [1, 9] -- ALWAYS the regrouping trap (the cents
      part, if naively concatenated without zero-padding, reads as one
      digit short: $3.05 -> naive "35" instead of the correct 305).
    """
    dollars = rng.randint(1, 9)
    if tier == "easy":
        frac = 0
    elif tier == "medium":
        frac = rng.randint(10, 99)
    else:
        frac = rng.randint(1, 9)
    return dollars, frac


def _c2d_cents(rng, tier: str) -> int:
    """Draw a cents amount for a cents -> dollars-and-cents question.

    easy: a whole-dollar multiple of 100 (100..900).
    medium: dollars_part*100 + frac_part with frac_part in [1, 99] -- always
      >= 101, so the leading-zero trap (cents < 100) never fires.
    hard: cents in [1, 99] -- ALWAYS the leading-zero-in-cents trap: the
      result is always under one dollar (e.g. 45 cents -> $0.45), where the
      naive error drops the required tenths-place zero and writes "4.5".
    """
    if tier == "easy":
        return rng.randint(1, 9) * 100
    if tier == "medium":
        return rng.randint(1, 9) * 100 + rng.randint(1, 99)
    return rng.randint(1, 99)


def _build_d2c(rng, tier: str) -> dict:
    dollars, frac = _d2c_value(rng, tier)
    dollars_value = Fraction(dollars) + Fraction(frac, 100)
    amt_text = stated_precision_string(dollars_value, 2)
    display_amt = f"${amt_text}"

    # Construction assembles the answer via place-value string manipulation:
    # the dollars part contributes whole hundreds of cents, the fractional
    # part contributes its own two digits of cents directly.
    constructed_cents = dollars * 100 + frac
    answer_text = fixed_point_string(constructed_cents, 0)

    # Independent check: recompute purely via Fraction arithmetic from the
    # STATED input value (`dollars_value`, the same string the learner
    # reads), a route that never touches the place-value string assembly
    # construction used above.
    recomputed = dollars_value * 100
    check(recomputed.denominator == 1, f"d2c independent check: {recomputed} is not an exact cent count")
    check_equivalent(int(recomputed), constructed_cents, "d2c independent check failed")

    templates = _D2C_TEMPLATES[:2] if tier != "hard" else _D2C_TEMPLATES
    template = rng.choice(templates)
    text = template.format(amt=display_amt)
    latex = text

    # Misconceptions -- always Fraction/int arithmetic on the STATED cents
    # value (dollars>=1 always, proven pairwise-distinct in every regime,
    #  - regrouping_failure: drops the leading zero of a single-digit
    #    fractional part (piecewise, see below) -- the named hard-tier trap.
    #  - forgot_the_dollars: reports only the cents remainder, ignoring the
    #    whole-dollar part entirely.
    #  - off_by_one_dollar: an extra/missing whole dollar (100 cents off).
    if frac < 10:
        regrouping_failure = dollars * 10 + frac
    else:
        regrouping_failure = dollars * 100 + (frac % 10)
    forgot_the_dollars = frac
    off_by_one_dollar = constructed_cents + 100

    check_pairwise_distinct(
        constructed_cents,
        regrouping_failure,
        forgot_the_dollars,
        off_by_one_dollar,
        msg="d2c misconception collides with the answer or another option",
    )

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(Fraction(constructed_cents)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{display_amt} = {dollars} {'dollar' if dollars == 1 else 'dollars'} and {frac:02d} cents "
            f"= {dollars} × 100{f' + {frac}' if frac else ''} = {answer_text} cents.",
        ],
        distractors=pick_distractors(
            answer_text,
            [
                fixed_point_string(regrouping_failure, 0),
                fixed_point_string(forgot_the_dollars, 0),
                fixed_point_string(off_by_one_dollar, 0),
            ],
        ),
    )


def _build_c2d(rng, tier: str) -> dict:
    cents = _c2d_cents(rng, tier)
    display_amt = f"{cents}"

    # Construction assembles the answer via place-value string manipulation:
    # dollars = cents // 100, remaining two digits are the cents part.
    dollars_part, frac_part = divmod(cents, 100)
    constructed_value = Fraction(dollars_part) + Fraction(frac_part, 100)
    answer_text = stated_precision_string(constructed_value, 2)

    # Independent check: recompute purely via Fraction arithmetic from the
    # STATED input cents value, dividing by 100 directly -- a route that
    # never touches the divmod/place-value split construction used above.
    recomputed = Fraction(cents, 100)
    check_equivalent(recomputed, constructed_value, "c2d independent check failed")

    templates = _C2D_TEMPLATES[:2] if tier != "hard" else _C2D_TEMPLATES
    template = rng.choice(templates)
    unit = "cent" if cents == 1 else "cents"
    text = template.format(amt=display_amt, unit=unit)
    latex = text

    # Misconceptions -- all Fraction arithmetic on V = constructed_value > 0,
    # algebraically proven pairwise distinct for every V > 0 (see result
    # file): leading_zero_drop = V*10 (the named hard-tier trap -- drops the
    # required tenths-place zero, e.g. 0.45 -> 4.50); bare_digits_as_dollars
    # = the raw cents count read directly as a dollar figure with no
    # conversion at all; div_by_10 = V/10 (moved the decimal the wrong way).
    leading_zero_drop = round_half_up(constructed_value * 10, 2)
    bare_digits_as_dollars = round_half_up(Fraction(cents), 2)
    div_by_10 = round_half_up(constructed_value / 10, 2)

    check_pairwise_distinct(
        constructed_value,
        leading_zero_drop,
        bare_digits_as_dollars,
        div_by_10,
        msg="c2d misconception collides with the answer or another option",
    )

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(constructed_value),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{cents} {unit} = {cents} ÷ 100 = ${answer_text}.",
        ],
        distractors=pick_distractors(
            answer_text,
            [
                stated_precision_string(leading_zero_drop, 2),
                stated_precision_string(bare_digits_as_dollars, 2),
                stated_precision_string(div_by_10, 2),
            ],
            tolerance=stated_precision_tolerance(2),
        ),
    )


@generator(
    "money_dollars_cents_representation",
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def money_dollars_cents_representation(rng, tier="easy"):
    """Convert a stated dollars/cents money amount into its single equivalent
    numeral value in the other unit -- the core of
    ``f10-money-dollars-cents-representation`` (AC9M3M06 narrowed). Never a
    payment/change shape; see the module docstring."""
    direction = rng.choice(["d2c", "c2d"])
    if direction == "d2c":
        return _build_d2c(rng, tier)
    return _build_c2d(rng, tier)


# ---------------------------------------------------------------------------
# f10-additive-multiplicative-money-model
# ---------------------------------------------------------------------------

_ITEMS = ["comic", "toy car", "book", "puzzle", "sticker pack", "yo-yo", "board game", "cap"]


def _pick_two(rng):
    a, b = rng.sample(NAMES, 2)
    return a, b


# In-generator independent check: read the operation and its operands back OUT OF THE RENDERED TEXT
# (never from construction's own a/b/n/p variables) via a structural regex per template, and
# recompute the value from those extracted quantities alone. This is the "two in-generator checks"
# discipline's `check_equivalent` route, applied to the artefact the learner actually receives --
# genuinely separate from however `_build_*` tagged its own operation internally ("verify the
# artefact the learner receives, not the one your code holds"). One (regex, formula) pair per
# template; the dispatcher tries each in turn and returns the first structural match.
_REPARSE_RULES: list[tuple[re.Pattern, callable]] = [
    # easy
    (
        re.compile(
            r"^(\w+) has \$(\d+)\. \1 earns another \$(\d+) from doing chores\. "
            r"How much money does \1 have now\?$"
        ),
        lambda m: int(m.group(2)) + int(m.group(3)),
    ),
    (
        re.compile(
            r"^(\w+) has \$(\d+)\. \1 spends \$(\d+) on a ([\w -]+)\. "
            r"How much money does \1 have left\?$"
        ),
        lambda m: int(m.group(2)) - int(m.group(3)),
    ),
    (
        re.compile(r"^(\w+) buys (\d+) of the same ([\w -]+) at \$(\d+) each\. What is the total cost\?$"),
        lambda m: int(m.group(2)) * int(m.group(4)),
    ),
    (
        re.compile(r"^(\w+) shares \$(\d+) equally among (\d+) people\. How much does each person get\?$"),
        lambda m: Fraction(int(m.group(2)), int(m.group(3))),
    ),
    # medium
    (
        re.compile(
            r"^(\w+) has \$(\d+)\. On Saturday \1 earns \$(\d+) from a garage sale, and on Sunday \1 "
            r"earns \$(\d+) more from babysitting\. How much money does \1 have by Sunday evening\?$"
        ),
        lambda m: int(m.group(2)) + int(m.group(3)) + int(m.group(4)),
    ),
    (
        re.compile(
            r"^(\w+) has \$(\d+)\. \1 earns \$(\d+) from chores, then spends \$(\d+) on a ([\w -]+)\. "
            r"How much money does \1 have now\?$"
        ),
        lambda m: int(m.group(2)) + int(m.group(3)) - int(m.group(4)),
    ),
    (
        re.compile(
            r"^(\w+) buys (\d+) ([\w -]+)s at \$(\d+) each, and also buys a ([\w -]+) for \$(\d+)\. "
            r"What is the total amount \1 spends\?$"
        ),
        lambda m: int(m.group(2)) * int(m.group(4)) + int(m.group(6)),
    ),
    (
        re.compile(
            r"^\$?(\d+) dollars of prize money is shared equally among (\d+) winners\. "
            r"Each winner then spends \$(\d+) on a ([\w -]+)\. How much does each winner have left\?$"
        ),
        lambda m: Fraction(int(m.group(1)), int(m.group(2))) - int(m.group(3)),
    ),
    # hard
    (
        re.compile(
            r"^(\w+) and (\w+) shared equally in washing cars for the fundraiser, and each earned "
            r"\$(\d+) from Mum and \$(\d+) from Dad for helping\. "
            r"How much money did \1 earn in total\?$"
        ),
        lambda m: int(m.group(3)) + int(m.group(4)),
    ),
    (
        re.compile(
            r"^(\w+) is saving twice as fast as before and now has \$(\d+) saved towards a "
            r"\$(\d+) skateboard\. How much more money does \1 need\?$"
        ),
        lambda m: int(m.group(3)) - int(m.group(2)),
    ),
    (
        re.compile(
            r"^(\w+) buys (\d+) raffle tickets\. Each ticket costs \$(\d+) more than last year's "
            r"\$(\d+) price\. What is the total cost of \1's tickets this year\?$"
        ),
        lambda m: int(m.group(2)) * (int(m.group(4)) + int(m.group(3))),
    ),
    (
        re.compile(r"^\$(\d+) is shared equally into envelopes of \$(\d+) each\. How many envelopes are there\?$"),
        lambda m: Fraction(int(m.group(1)), int(m.group(2))),
    ),
]


def _reparse_from_text(text: str):
    """Independently classify and recompute `text` per `_REPARSE_RULES`
    above. Raises via `check()` if no rule's structural pattern matches --
    an under-specified question is itself a finding, per the conventions
    file."""
    for pattern, formula in _REPARSE_RULES:
        m = pattern.match(text)
        if m:
            return formula(m)
    check(False, f"no independent-reading rule matched rendered text: {text!r}")


def _build_add_easy(rng) -> dict:
    name = rng.choice(NAMES)
    a = rng.randint(10, 60)
    b = rng.randint(5, 30)
    answer = a + b
    text = f"{name} has ${a}. {name} earns another ${b} from doing chores. How much money does {name} have now?"
    return _finish_add(rng, text, name, a, b, answer, tier="easy")


def _build_subtract_easy(rng) -> dict:
    name = rng.choice(NAMES)
    a = rng.randint(30, 80)
    b = rng.randint(5, a - 5)
    item = rng.choice(_ITEMS)
    answer = a - b
    text = f"{name} has ${a}. {name} spends ${b} on a {item}. How much money does {name} have left?"
    return _finish_subtract(rng, text, name, a, b, answer, tier="easy")


def _build_multiply_easy(rng) -> dict:
    name = rng.choice(NAMES)
    n = rng.randint(3, 9)
    p = rng.randint(2, 12)
    item = rng.choice(_ITEMS)
    answer = n * p
    text = f"{name} buys {n} of the same {item} at ${p} each. What is the total cost?"
    return _finish_multiply(rng, text, name, n, p, answer, tier="easy")


def _build_divide_easy(rng) -> dict:
    name = rng.choice(NAMES)
    n = rng.randint(2, 8)
    each = rng.randint(3, 15)
    total = n * each
    answer = each
    text = f"{name} shares ${total} equally among {n} people. How much does each person get?"
    return _finish_divide(rng, text, name, total, n, answer, tier="easy")


def _build_add_medium(rng) -> dict:
    """Two-step composition: two separate additions (repeated addition), the
    decisive operation stays addition but the learner must combine two named
    amounts before reaching the final value."""
    name = rng.choice(NAMES)
    a = rng.randint(10, 40)
    b = rng.randint(5, 25)
    c = rng.randint(5, 25)
    answer = a + b + c
    text = (
        f"{name} has ${a}. On Saturday {name} earns ${b} from a garage sale, "
        f"and on Sunday {name} earns ${c} more from babysitting. "
        f"How much money does {name} have by Sunday evening?"
    )
    return _finish_add(rng, text, name, a, b + c, answer, tier="medium")


def _build_subtract_medium(rng) -> dict:
    """Two-step composition: earn then spend (add then subtract), the
    a standard worked example shape."""
    name = rng.choice(NAMES)
    a = rng.randint(10, 30)
    b = rng.randint(5, 25)
    item = rng.choice(_ITEMS)
    c_hi = min(20, a + b - 1)
    c = rng.randint(3, c_hi)
    # Avoid c == b (wrong-order misconception a-b+c would then equal the
    # true answer a+b-c) and c == 2*b (off-by-one-step a+b would then equal
    # the wrong-order misconception a-b+c) -- both real algebraic collisions
    # `check_pairwise_distinct` below would otherwise catch and reject.
    while c == b or c == 2 * b:
        c = rng.randint(3, c_hi)
    answer = a + b - c
    text = (
        f"{name} has ${a}. {name} earns ${b} from chores, then spends ${c} on a {item}. "
        f"How much money does {name} have now?"
    )
    return _finish_subtract_two_step(rng, text, name, a, b, c, answer, tier="medium")


def _build_multiply_medium(rng) -> dict:
    """Two-step composition: buy several of one item, then one more of a
    different item (multiply then add) -- the decisive operation stays
    multiplication, but the learner must combine two named amounts (the
    multiplied group cost and the single extra item) before reaching the
    final value."""
    name = rng.choice(NAMES)
    n = rng.randint(3, 8)
    p = rng.randint(2, 10)
    bonus = rng.randint(2, 15)
    item, item2 = rng.sample(_ITEMS, 2)
    answer = n * p + bonus
    text = (
        f"{name} buys {n} {item}s at ${p} each, and also buys a {item2} for ${bonus}. "
        f"What is the total amount {name} spends?"
    )
    return _finish_multiply_two_step(rng, text, name, n, p, bonus, answer, tier="medium")


def _build_divide_medium(rng) -> dict:
    """Two-step composition: share equally, then each spends more
    (divide then subtract) -- the decisive operation stays division."""
    name = rng.choice(NAMES)
    n = rng.randint(2, 6)
    each = rng.randint(10, 25)
    total = n * each
    item = rng.choice(_ITEMS)
    c = rng.randint(2, each - 2)
    answer = each - c
    text = (
        f"{total} dollars of prize money is shared equally among {n} winners. "
        f"Each winner then spends ${c} on a {item}. How much does each winner have left?"
    )
    return _finish_divide_two_step(rng, text, total, n, c, answer, tier="medium")


def _build_add_hard(rng) -> dict:
    """Trap: 'shared equally' normally cues division, but the actual
    question (how much did NAME earn) is addition of two named amounts."""
    a_name, b_name = _pick_two(rng)
    a = rng.randint(8, 25)
    b = rng.randint(8, 25)
    answer = a + b
    text = (
        f"{a_name} and {b_name} shared equally in washing cars for the fundraiser, and each earned "
        f"${a} from Mum and ${b} from Dad for helping. How much money did {a_name} earn in total?"
    )
    return _finish_add(
        rng,
        text,
        a_name,
        a,
        b,
        answer,
        tier="hard",
        trap="'shared equally' cues division; "
        f"the question only asks {a_name}'s own total, which is addition (${a} + ${b}).",
    )


def _build_subtract_hard(rng) -> dict:
    """Trap: 'twice' normally cues multiplication, but the actual question
    (how much MORE is needed) is subtraction. `saved` is drawn independently
    of `full` (never a fixed fraction of it), so the answer is never a value
    already printed in the question -- a fixed half/double relationship
    would let a learner copy a displayed number without subtracting at all."""
    name = rng.choice(NAMES)
    full = rng.randint(40, 90)
    saved = rng.randint(5, full - 5)
    answer = full - saved
    text = (
        f"{name} is saving twice as fast as before and now has ${saved} saved towards a "
        f"${full} skateboard. How much more money does {name} need?"
    )
    return _finish_subtract(
        rng,
        text,
        name,
        full,
        saved,
        answer,
        tier="hard",
        trap=f"'twice' cues multiplication; the question asks how much MORE is "
        f"needed, which is subtraction (${full} - ${saved}).",
    )


def _build_multiply_hard(rng) -> dict:
    """Trap: the additive word 'more' appears prominently, but the final
    decisive operation is multiplication (after a small addition step)."""
    name = rng.choice(NAMES)
    base = rng.randint(5, 15)
    extra = rng.randint(2, 5)
    n = rng.randint(3, 7)
    per_ticket = base + extra
    answer = n * per_ticket
    text = (
        f"{name} buys {n} raffle tickets. Each ticket costs ${extra} more than last year's ${base} price. "
        f"What is the total cost of {name}'s tickets this year?"
    )
    return _finish_multiply_hard(
        rng,
        text,
        name,
        n,
        base,
        extra,
        per_ticket,
        answer,
        trap=f"'more' cues addition; the final decisive operation is "
        f"multiplication ({n} × ${per_ticket}) once the per-ticket price is found.",
    )


def _build_divide_hard(rng) -> dict:
    """Trap: 'each' normally cues multiplication, but the actual op is
    division -- a standard worked example, in money form."""
    price = rng.randint(3, 9)
    n = rng.randint(4, 12)
    total = price * n
    answer = n
    text = f"${total} is shared equally into envelopes of ${price} each. How many envelopes are there?"
    return _finish_divide(
        rng,
        text,
        None,
        total,
        price,
        answer,
        tier="hard",
        trap=f"'each' cues multiplication; the actual operation is division (${total} ÷ ${price}).",
    )


def _finish_add(rng, text, name, a, b, answer, tier, trap=None) -> dict:
    # Independent check: read the two dollar quantities back out of the
    # RENDERED text with a structural (not keyword) rule -- "combine every
    # named dollar amount into one running total" -- and confirm it matches
    # construction's own a + b, a route genuinely separate from however
    # construction picked "add" as its branch.
    check_equivalent(_reparse_from_text(text), answer, "add independent check (reparsed from rendered text) failed")
    # Algebraically proven pairwise distinct for every a >= 10, b >= 5 (the
    # misconception_subtract = |a-b| < a+b always (a, b > 0); misconception_
    # multiply = a*b, equal to a+b only at (a-1)(b-1)=1, impossible for
    # a >= 10; misconception_added_twice_b = a+2b, equal to a*b only at
    # a = 2b/(b-1) <= 2.5 for any b >= 5, impossible for a >= 10.
    misconception_subtract = abs(a - b)
    misconception_multiply = a * b
    misconception_added_twice_b = a + 2 * b
    check_pairwise_distinct(
        answer,
        misconception_subtract,
        misconception_multiply,
        misconception_added_twice_b,
        msg="add misconception collides with the answer or another option",
    )
    working = [f"${a} + ${b} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_subtract), str(misconception_multiply), str(misconception_added_twice_b)],
        ),
    )


def _finish_subtract(rng, text, name, a, b, answer, tier, trap=None) -> dict:
    check_equivalent(
        _reparse_from_text(text), answer, "subtract independent check (reparsed from rendered text) failed"
    )
    misconception_add = a + b
    misconception_reversed = b - a  # operands swapped
    misconception_multiply = a * b if a * b != answer else a + b + 1
    check_pairwise_distinct(
        answer,
        misconception_add,
        misconception_reversed,
        misconception_multiply,
        msg="subtract misconception collides with the answer or another option",
    )
    working = [f"${a} - ${b} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_add), str(misconception_reversed), str(misconception_multiply)],
        ),
    )


def _finish_subtract_two_step(rng, text, name, a, b, c, answer, tier, trap=None) -> dict:
    check_equivalent(
        _reparse_from_text(text), answer, "subtract (two-step) independent check (reparsed from rendered text) failed"
    )
    misconception_off_by_one_step = a + b  # forgot the second step (the spend)
    misconception_wrong_order = a - b + c  # subtracted the earn, added the spend
    misconception_add_all = a + b + c
    check_pairwise_distinct(
        answer,
        misconception_off_by_one_step,
        misconception_wrong_order,
        misconception_add_all,
        msg="subtract (two-step) misconception collides with the answer or another option",
    )
    working = [f"${a} + ${b} - ${c} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_off_by_one_step), str(misconception_wrong_order), str(misconception_add_all)],
        ),
    )


def _finish_multiply(rng, text, name, n, p, answer, tier, trap=None) -> dict:
    check_equivalent(
        _reparse_from_text(text), answer, "multiply independent check (reparsed from rendered text) failed"
    )
    # Algebraically proven pairwise distinct for every n >= 3, p >= 2: misconception_add = n+p,
    # misconception_price_only = p (forgot to multiply by the count), misconception_extra_item =
    # (n+1)*p (counted one item too many).
    misconception_add = n + p
    misconception_price_only = p
    misconception_extra_item = (n + 1) * p
    check_pairwise_distinct(
        answer,
        misconception_add,
        misconception_price_only,
        misconception_extra_item,
        msg="multiply misconception collides with the answer or another option",
    )
    working = [f"{n} × ${p} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_add), str(misconception_price_only), str(misconception_extra_item)],
        ),
    )


def _finish_multiply_two_step(rng, text, name, n, p, bonus, answer, tier, trap=None) -> dict:
    check_equivalent(
        _reparse_from_text(text), answer, "multiply (two-step) independent check (reparsed from rendered text) failed"
    )
    misconception_off_by_one_step = n * p  # forgot the bonus
    misconception_add_first = (n + bonus) * p
    misconception_multiply_bonus = n * p * bonus if n * p * bonus != answer else n * (p + bonus)
    check_pairwise_distinct(
        answer,
        misconception_off_by_one_step,
        misconception_add_first,
        misconception_multiply_bonus,
        msg="multiply (two-step) misconception collides with the answer or another option",
    )
    working = [f"({n} × ${p}) + ${bonus} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_off_by_one_step), str(misconception_add_first), str(misconception_multiply_bonus)],
        ),
    )


def _finish_multiply_hard(rng, text, name, n, base, extra, per_ticket, answer, trap=None) -> dict:
    check_equivalent(
        _reparse_from_text(text), answer, "multiply (hard) independent check (reparsed from rendered text) failed"
    )
    misconception_add_extra_once = n * base + extra  # added the trap word's "more" only once
    misconception_used_base_price = n * base
    # An UNDER-count `(n-1)*per_ticket` (an earlier draft) can equal `misconception_add_extra_once`
    # or `misconception_used_base_price` for particular draws. An OVER-count `(n+1)*per_ticket` is
    # algebraically proven distinct from all three of {answer, add_extra_once, used_base_price} for
    # every n >= 3, base/extra > 0: each equality reduces to a positive quantity equalling zero.
    misconception_off_by_one_ticket = (n + 1) * per_ticket
    check_pairwise_distinct(
        answer,
        misconception_add_extra_once,
        misconception_used_base_price,
        misconception_off_by_one_ticket,
        msg="multiply (hard) misconception collides with the answer or another option",
    )
    working = [
        f"Per-ticket price = ${base} + ${extra} = ${per_ticket}; {n} × ${per_ticket} = ${answer}."
        + (f" Trap: {trap}" if trap else "")
    ]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [
                str(misconception_add_extra_once),
                str(misconception_used_base_price),
                str(misconception_off_by_one_ticket),
            ],
        ),
    )


def _finish_divide(rng, text, name, total, n, answer, tier, trap=None) -> dict:
    check(total % n == 0, "divide construction did not draw an exact multiple")
    check_equivalent(_reparse_from_text(text), answer, "divide independent check (reparsed from rendered text) failed")
    misconception_multiply = total * n
    misconception_subtract = total - n
    misconception_reversed = n  # reported the divisor instead of the quotient (only when n != answer)
    if misconception_reversed == answer:
        misconception_reversed = total
    check_pairwise_distinct(
        answer,
        misconception_multiply,
        misconception_subtract,
        misconception_reversed,
        msg="divide misconception collides with the answer or another option",
    )
    working = [f"${total} ÷ {n} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_multiply), str(misconception_subtract), str(misconception_reversed)],
        ),
    )


def _finish_divide_two_step(rng, text, total, n, c, answer, tier, trap=None) -> dict:
    check(total % n == 0, "divide (two-step) construction did not draw an exact multiple")
    each = total // n
    check_equivalent(
        _reparse_from_text(text), answer, "divide (two-step) independent check (reparsed from rendered text) failed"
    )
    misconception_off_by_one_step = each  # forgot the second step (the spend)
    # subtracted the (undivided) spend from the total BEFORE dividing, then
    # forgot to divide again -- always an exact integer (total, c both
    # integers), unlike `Fraction(total - c, n)` which need not divide
    # evenly and would raise `stated_precision_string`-style off-grid errors.
    misconception_wrong_order = total - c
    # `n*c` (used in an earlier draft) can equal `each - c` for particular draws (c*(n+1) == each).
    # Reporting the un-divided total instead is algebraically proven distinct from all three of
    # {answer, off_by_one, wrong_order} for every n >= 2, c >= 1: total == each requires n == 1;
    # total == total - c requires c == 0; total == each - c would need n == 1 too, all impossible in
    # this generator's ranges.
    misconception_reported_total = total
    check_pairwise_distinct(
        answer,
        misconception_off_by_one_step,
        misconception_wrong_order,
        misconception_reported_total,
        msg="divide (two-step) misconception collides with the answer or another option",
    )
    working = [f"${total} ÷ {n} = ${each}; ${each} - ${c} = ${answer}." + (f" Trap: {trap}" if trap else "")]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(
            str(answer),
            [str(misconception_off_by_one_step), str(misconception_wrong_order), str(misconception_reported_total)],
        ),
    )


_EASY_BUILDERS = [_build_add_easy, _build_subtract_easy, _build_multiply_easy, _build_divide_easy]
_MEDIUM_BUILDERS = [_build_add_medium, _build_subtract_medium, _build_multiply_medium, _build_divide_medium]
_HARD_BUILDERS = [_build_add_hard, _build_subtract_hard, _build_multiply_hard, _build_divide_hard]

_TIER_BUILDERS = {"easy": _EASY_BUILDERS, "medium": _MEDIUM_BUILDERS, "hard": _HARD_BUILDERS}


@generator(
    "additive_multiplicative_money_model",
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def additive_multiplicative_money_model(rng, tier="easy"):
    """Given a stated additive-or-multiplicative money situation, determine
    which of addition, subtraction, multiplication or division correctly
    models it and compute the resulting value -- the core of
    ``f10-additive-multiplicative-money-model`` (AC9M2N06 narrowed). All four
    operations are live at every tier; the operation is drawn independently
    of the difficulty label, never predicted by it. See the module docstring
    for the subtraction/``money_change`` boundary."""
    builders = _TIER_BUILDERS[tier]
    builder = rng.choice(builders)
    return builder(rng)
