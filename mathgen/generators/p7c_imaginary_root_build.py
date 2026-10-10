"""- ``imaginary_number_i_root`` (``senior-define-the-imaginary-number-i-as-a-root``,
  ACMSM067, domain senior-mathematics, answer: expression) -- given bounded,
  fully specified inputs, define the imaginary number i as a root of the
  equation x^2 = -1. Return the resulting value, expression, relation or
  classification.

Load-bearing note: ``grade_expression`` (``mathgen/grading/typed.py``) parses both the learner's typed
answer and the canonical ``answer_text`` with sympy's ``parse_expr``, where a lowercase ``i`` always parses to
a plain ``Symbol('i')`` -- never the true imaginary unit ``sympy.I``. So every canonical answer here is a
plain string ("i", "-5i", "-i"...), and ``question.answer`` is set to that exact same string -- never
``sympy.I``, Python's ``1j``, or a ``complex``. All genuine complex arithmetic lives only in the independent
checks below, built with Python's native ``complex`` type -- a structurally different route from the
string/table construction, and (being real arithmetic rather than a symbolic route) faster too. Inside the
generator, distractor distinctness is proved with sympy objects built from a local ``Symbol('i')``
(``check_distinct`` subtracts its arguments), never by comparing the display strings directly.
"""

from __future__ import annotations

from ..core import check, check_distinct, fmt_signed, generator, paren_negative, pick_distractors
from sympy import Symbol

i_sym = Symbol("i")

# n % 4 -> the reduced value of i**n, as both the display string the learner
# is shown and the sympy object used for check_distinct.
_POWER_TABLE = {0: "1", 1: "i", 2: "-1", 3: "-i"}
_POWER_TABLE_COMPLEX = {0: 1 + 0j, 1: 1j, 2: -1 + 0j, 3: -1j}
_POWER_TABLE_SYM = {"1": 1, "-1": -1, "i": i_sym, "-i": -i_sym}
#: display string -> complex value, so a claimed answer (whichever of the
#: four strings it is) can be checked against the raw-exponentiation route.
_STR_TO_COMPLEX = {"1": 1 + 0j, "i": 1j, "-1": -1 + 0j, "-i": -1j}


def _verify_power_cycle(n: int, answer_str: str) -> None:
    """Independent check for the medium/hard tiers, callable on its own so a
    test can corrupt ``answer_str`` and confirm ``VerificationError`` fires.

    Raw complex exponentiation -- a structurally different route from the
    remainder-table lookup that built ``answer_str`` -- must land on the same
    point of the four-cycle as the claimed answer. A small tolerance is
    needed only because complex ** int accumulates float noise for large
    |n| (e.g. n=500), never because the identity is approximate.
    """
    r = n % 4
    actual = complex(0, 1) ** n
    check(abs(actual - _POWER_TABLE_COMPLEX[r]) < 1e-6, f"power cycle check failed for n={n}")
    check(abs(actual - _STR_TO_COMPLEX[answer_str]) < 1e-6, f"claimed answer {answer_str!r} does not match i**{n}")
    if n < 0:
        # The negative-exponent identity i**-1 == 1/i == -i, verified
        # directly with complex arithmetic rather than assumed.
        j = complex(0, 1)
        check(abs(j**-1 - (1 / j)) < 1e-9, "i**-1 != 1/i")
        check(abs(j**-1 - (-j)) < 1e-9, "i**-1 != -i")


def _verify_sqrt(k: int, claimed: complex) -> None:
    """Independent check for the easy tier, callable on its own so a test can
    pass a corrupted ``claimed`` value and confirm ``VerificationError``
    fires. Squaring the claimed answer (Python's native complex type) must
    reproduce the radicand -- a structurally different route from the
    table-driven construction of the answer text."""
    check(claimed**2 == complex(-(k * k), 0), f"sqrt(-k^2) check failed for k={k}")


def _power_cycle_question(n: int) -> dict:
    """Shared construction for the medium and hard tiers: evaluate i**n by
    reducing the exponent mod 4.

    Distractors are the other three members of {1, -1, i, -i}: off-by-one in the mod-4 reduction in each
    direction, and the sign-flip.
    """
    r = n % 4
    answer_str = _POWER_TABLE[r]
    _verify_power_cycle(n, answer_str)

    off_by_plus = _POWER_TABLE[(r + 1) % 4]
    off_by_minus = _POWER_TABLE[(r + 3) % 4]
    sign_flip = _POWER_TABLE[(r + 2) % 4]
    check_distinct(
        _POWER_TABLE_SYM[answer_str],
        _POWER_TABLE_SYM[off_by_plus],
        _POWER_TABLE_SYM[off_by_minus],
        _POWER_TABLE_SYM[sign_flip],
    )
    distractors = pick_distractors(answer_str, [off_by_plus, off_by_minus, sign_flip])

    text = f"Evaluate i^{paren_negative(n)}. Give your answer as one of 1, -1, i, -i."
    return dict(
        text=text,
        latex=rf"i^{{{n}}} \ \text{{Give your answer as one of }} 1, -1, i, -i",
        answer=answer_str,
        answer_text=answer_str,
        answer_latex=answer_str,
        distractors=distractors,
    )


@generator(
    "imaginary_number_i_root",
    level=8,
    presets={
        "easy": dict(difficulty="easy"),
        "medium": dict(difficulty="medium"),
        "hard": dict(difficulty="hard"),
    },
)
def imaginary_number_i_root(rng, difficulty="easy"):
    """
    Define i as a root of x^2 = -1: three genuinely different reasoning acts.

    easy: apply the definition directly -- simplify sqrt(-k^2) for a
    perfect-square radicand (i.e. i = sqrt(-1) applied literally).
    medium: apply the power cycle -- evaluate i^n for a small positive n via
    n % 4 (recognising and applying periodicity, not direct square-root
    evaluation).
    hard: the same mod-4 reduction but with exponents too large to reduce by
    counting, PLUS negative exponents, which need the i**-1 == -i identity --
    a genuinely different move from either easier tier, not just a bigger n.
    """
    if difficulty == "medium":
        n = rng.randint(5, 30)
        return _power_cycle_question(n)

    if difficulty == "hard":
        if rng.random() < 0.5:
            n = rng.randint(25, 500)
        else:
            n = rng.randint(-60, -1)
        return _power_cycle_question(n)

    # easy: k = 1..20, radicand = -(k**2), answer = k*i (i.e. "i" for k=1,
    # never "1i").
    k = rng.randint(1, 20)
    radicand = -(k * k)

    # Independent check: squaring the claimed answer (k*i, built with
    # Python's native complex type) must reproduce the radicand -- a
    # structurally different route from the table-driven construction above.
    _verify_sqrt(k, complex(0, k))

    answer_text = fmt_signed(k, "i")
    check_distinct(k * i_sym, k, -k, -k * i_sym)
    distractors = pick_distractors(answer_text, [fmt_signed(k, ""), fmt_signed(-k, ""), fmt_signed(-k, "i")])

    text = f"Simplify sqrt({radicand})."
    return dict(
        text=text,
        latex=rf"\sqrt{{{radicand}}}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )
