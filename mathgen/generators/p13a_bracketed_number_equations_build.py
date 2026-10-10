"""Generates `p13a_bracketed_number` questions for `f10-bracketed-number-equations`: find the value of a single
numeric blank `?` in a numerical equation containing brackets and multiple arithmetic operations, using order
of operations and inverse operations.

Three tiers, each a genuinely different equation SHAPE, not a bigger number:

- easy (`shape="isolate"`): `a * (? + b) = c` -- one bracket, one outer
  multiplication. Two inverse steps: divide by `a`, subtract `b`.
- medium (`shape="outer_add"`): `a * (? - b) + d = c` -- an extra additive
  term OUTSIDE the bracket. Three inverse steps: subtract `d`, divide by
  `a`, add `b`.
- hard (`shape="leading_subtract"`): `d - a * (? + b) = e` -- the bracket
  term is SUBTRACTED from a leading constant, so the coefficient of `?` is
  effectively negative. The learner must isolate the bracket as
  `a*(?+b) = d - e` (note the bracket's own value is `(d-e)`, not `(e-d)`)
  before dividing and subtracting -- a genuinely different decision (sign
  handling) from easy/medium's left-to-right shape.

Construct-to-target throughout: `?`'s target value and every other literal
are drawn FIRST, and the equation's stated constant (`c`/`e`) is computed
from them, so every equation is exact by construction -- never solve-then-
round. The independent check is a genuinely different route from that
construction: it builds the same equation as a `sympy.Eq` with a real
`Symbol` standing in for `?` and calls `sympy.solve`, confirming the lone
root equals the constructed target.

Misconceptions (each a different final number, proven distinct from the
answer and from each other, unconditionally, every seed):

- "isolates the bracket but stops" -- gets to the bracket's own value and
  reports THAT as the answer, forgetting the final add/subtract step
  (`easy`/`medium` m1, `hard` m3).
- a sign/operation error -- adds where it should subtract, or (on the `hard`
  shape specifically) computes the bracket's value as `(e - d)` instead of
  `(d - e)` (`easy` m2, `medium` m3, `hard` m1).
- an order-of-operations slip -- re-applies the outer multiplier to a term
  that was already isolated, divides before subtracting the outer additive
  term, or drops the leading constant entirely instead of moving it to the
  other side (`easy` m3, `medium` m2, `hard` m2).

`d` is constructed as a multiple of `a` on `medium`/`hard` purely so the "divide before handling the
additive/leading term" misconception itself stays an exact integer -- this is invisible in the rendered
equation (the learner sees only the two numbers `a` and `d`, never their ratio) and changes nothing about the
difficulty or the shape of the question.
"""

from __future__ import annotations

from sympy import Eq, Symbol, solve

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    check_equivalent,
    paren_negative,
    phrase,
    pick_distractors,
    generator,
)

_Q_LO, _Q_HI = -20, 40
_A_LO, _A_HI = 2, 9
_B_LO, _B_HI = 1, 12
_RETRY_LIMIT = 50

_QSYM = Symbol("q_blank", real=True)

_EASY_PATTERNS = (
    "Solve: {a} × (? + {b}) = {c}",
    "Find the value of ? in {a} × (? + {b}) = {c}",
    "What number makes {a} × (? + {b}) = {c} true?",
)
_MEDIUM_PATTERNS = (
    "Solve: {a} × (? - {b}) + {d} = {c}",
    "Find the value of ? in {a} × (? - {b}) + {d} = {c}",
    "What number makes {a} × (? - {b}) + {d} = {c} true?",
)
_HARD_PATTERNS = (
    "Solve: {d} - {a} × (? + {b}) = {e}",
    "Find the value of ? in {d} - {a} × (? + {b}) = {e}",
    "What number makes {d} - {a} × (? + {b}) = {e} true?",
)


def _draw_common(rng):
    a = rng.randint(_A_LO, _A_HI)
    b = rng.randint(_B_LO, _B_HI)
    q = rng.randint(_Q_LO, _Q_HI)
    return a, b, q


def _distinct(*values) -> bool:
    return len(values) == len(set(values))


def _verify_blank(equation: Eq, expected: int, msg: str) -> None:
    """The in-generator independent check, factored out so a test can exercise
    it directly (per `test_ordered_sequence_build.py`'s `_verify_*` pattern):
    solve `equation` for `_QSYM` with sympy and confirm the lone root equals
    `expected`. This is the SAME check every builder below calls after its own
    Python-integer construction -- a genuinely different route (sympy's linear
    solver on a symbolic equation), never a replay of the construction
    arithmetic."""
    solutions = solve(equation, _QSYM)
    check(len(solutions) == 1, "expected exactly one root for a linear equation")
    check_equivalent(solutions[0], expected, msg)


def _build_easy(rng):
    for _ in range(_RETRY_LIMIT):
        a, b, q = _draw_common(rng)
        c = a * (q + b)
        m1 = q + b  # isolates the bracket, forgets to subtract b
        m2 = q + 2 * b  # adds b instead of subtracting it
        m3 = q + b * (1 - a)  # re-multiplies b by a (order-of-operations slip)
        if _distinct(q, m1, m2, m3):
            break
    else:
        raise RuntimeError("could not draw a distinct-distractor easy question")

    check(c % a == 0, "constructed constant is not an exact multiple of a")

    _verify_blank(Eq(a * (_QSYM + b), c), q, "easy bracket equation: sympy solve disagrees with construction")

    check_distinct(q, m1, m2, m3, msg="an easy-tier distractor equals the answer")
    check_pairwise_distinct([m1, m2, m3], msg="two easy-tier distractors coincide")

    text = phrase(rng, _EASY_PATTERNS, a=a, b=b, c=c)
    working = [
        f"Isolate the bracket: divide both sides by {a} to get ? + {b} = {c // a}.",
        f"Subtract {b} from both sides: ? = {q}.",
    ]
    candidates = [str(m1), str(m2), str(m3)]
    return dict(
        text=text,
        answer=int(q),
        answer_text=str(q),
        working=working,
        distractors=pick_distractors(str(q), candidates),
    )


def _build_medium(rng):
    for _ in range(_RETRY_LIMIT):
        a, b, q = _draw_common(rng)
        d_mult = rng.randint(1, 8)
        d = a * d_mult
        c = a * (q - b) + d
        m1 = q - b  # isolates the bracket, forgets to add b back
        m2 = q + d_mult  # divides before subtracting d
        m3 = q - 2 * b  # subtracts b instead of adding it back
        if _distinct(q, m1, m2, m3):
            break
    else:
        raise RuntimeError("could not draw a distinct-distractor medium question")

    check((c - d) % a == 0, "constructed constant does not isolate a whole bracket value")

    _verify_blank(Eq(a * (_QSYM - b) + d, c), q, "medium bracket equation: sympy solve disagrees with construction")

    check_distinct(q, m1, m2, m3, msg="a medium-tier distractor equals the answer")
    check_pairwise_distinct([m1, m2, m3], msg="two medium-tier distractors coincide")

    text = phrase(rng, _MEDIUM_PATTERNS, a=a, b=b, c=c, d=d)
    working = [
        f"Subtract {d} from both sides: {a} × (? - {b}) = {c - d}.",
        f"Divide both sides by {a}: ? - {b} = {(c - d) // a}.",
        f"Add {b} to both sides: ? = {q}.",
    ]
    candidates = [str(m1), str(m2), str(m3)]
    return dict(
        text=text,
        answer=int(q),
        answer_text=str(q),
        working=working,
        distractors=pick_distractors(str(q), candidates),
    )


def _build_hard(rng):
    for _ in range(_RETRY_LIMIT):
        a, b, q = _draw_common(rng)
        if q + b == 0:
            continue  # degenerate: a zero bracket collapses the sign-error distractor onto the answer
        d_mult = rng.randint(3, 12)
        d = a * d_mult
        e = d - a * (q + b)
        m1 = -(q + b) - b  # computes (e - d)/a instead of (d - e)/a (sign error)
        m2 = q - d_mult  # drops the leading constant d entirely
        m3 = q + b  # isolates the bracket, forgets to subtract b
        if _distinct(q, m1, m2, m3):
            break
    else:
        raise RuntimeError("could not draw a distinct-distractor hard question")

    check((d - e) % a == 0, "constructed constant does not isolate a whole bracket value")

    _verify_blank(Eq(d - a * (_QSYM + b), e), q, "hard bracket equation: sympy solve disagrees with construction")

    check_distinct(q, m1, m2, m3, msg="a hard-tier distractor equals the answer")
    check_pairwise_distinct([m1, m2, m3], msg="two hard-tier distractors coincide")

    text = phrase(rng, _HARD_PATTERNS, a=a, b=b, e=e, d=d)
    working = [
        f"Isolate the bracket: {a} × (? + {b}) = {d} - {paren_negative(e)} = {d - e}.",
        f"Divide both sides by {a}: ? + {b} = {(d - e) // a}.",
        f"Subtract {b} from both sides: ? = {q}.",
    ]
    candidates = [str(m1), str(m2), str(m3)]
    return dict(
        text=text,
        answer=int(q),
        answer_text=str(q),
        working=working,
        distractors=pick_distractors(str(q), candidates),
    )


_BUILDERS = {
    "isolate": _build_easy,
    "outer_add": _build_medium,
    "leading_subtract": _build_hard,
}


@generator(
    "p13a_bracketed_number",
    level=6,
    presets={
        "easy": dict(shape="isolate"),
        "medium": dict(shape="outer_add"),
        "hard": dict(shape="leading_subtract"),
    },
)
def p13a_bracketed_number(rng, shape="isolate"):
    """Find the value of the numeric blank `?` in a bracketed numerical
    equation. `shape` selects one of the three genuinely different equation
    forms (see module docstring); never a letter unknown, never a bigger
    number standing in for a harder question."""
    builder = _BUILDERS.get(shape)
    if builder is None:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(_BUILDERS)}")
    return builder(rng)
