"""Exam-tier 2 by 2 matrix multiplication questions (multiple choice).

The practice tier multiplies two numeric matrices once. The exam tier composes the product with other matrix
operations, or runs it backwards:

* ``commutator``: ``AB - BA``. Two products are needed, and the learner has to respect that the order of the
  factors matters.
* ``polynomial``: ``A^2 - kA + mI``. A product, a scalar multiple and the identity combined in one expression.
* ``unknowns``: ``AB = C`` where one entry of ``A`` and one entry of ``B`` are unknown. The learner reads off
  equations from the entries of the product and solves them in the right order.

Options follow the practice tier's style: a matrix is written ``[[a, b], [c, d]]`` and an answer for the
unknowns is written ``x = ..., y = ...``. Every wrong option is the result of a named mistake.
"""

from __future__ import annotations

import random

import sympy

from ..core import check, generator, paren_negative, pick_distractors


def _mul(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def _add(a, b, sign=1):
    return [[a[i][j] + sign * b[i][j] for j in range(2)] for i in range(2)]


def _scale(a, k):
    return [[k * v for v in row] for row in a]


def _hadamard(a, b):
    return [[a[i][j] * b[i][j] for j in range(2)] for i in range(2)]


def _text(m) -> str:
    return str(m)


def _rand_matrix(rng: random.Random, lo: int = -3, hi: int = 4):
    return [[rng.randint(lo, hi) for _ in range(2)] for _ in range(2)]


def _finish(text: str, answer: str, candidates: list[str], working: list[str]) -> dict:
    distractors = pick_distractors(answer, candidates)
    check(len(distractors) >= 1, "no usable distractor")
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        working=working,
        distractors=distractors,
    )


def _commutator(rng: random.Random) -> dict:
    while True:
        a, b = _rand_matrix(rng), _rand_matrix(rng)
        ab, ba = _mul(a, b), _mul(b, a)
        diff = _add(ab, ba, -1)
        if diff != [[0, 0], [0, 0]]:
            break
    # Independent route: the same difference from sympy matrices.
    check(sympy.Matrix(diff) == sympy.Matrix(a) * sympy.Matrix(b) - sympy.Matrix(b) * sympy.Matrix(a), "AB - BA")
    text = (
        f"Let A={_text(a)} and B={_text(b)}. Find AB - BA."
        if rng.random() < 0.5
        else f"For A={_text(a)} and B={_text(b)}, calculate the matrix AB - BA."
    )
    cands = [
        _text(ab),  # only the first product
        _text(_add(ba, ab, -1)),  # the subtraction taken the wrong way round
        _text(_add(ab, ba)),  # added instead of subtracted
        _text(_add(_hadamard(a, b), ba, -1)),  # the first product taken entry by entry
    ]
    working = [f"AB = {_text(ab)} and BA = {_text(ba)}, so AB - BA = {_text(diff)}."]
    return _finish(text, _text(diff), cands, working)


def _polynomial(rng: random.Random) -> dict:
    a = _rand_matrix(rng, -3, 3)
    k, m = rng.randint(2, 4), rng.randint(2, 5)
    a2 = _mul(a, a)
    result = _add(_add(a2, _scale(a, k), -1), _scale([[1, 0], [0, 1]], m))
    check(
        sympy.Matrix(result) == sympy.Matrix(a) ** 2 - k * sympy.Matrix(a) + m * sympy.eye(2),
        "A^2 - kA + mI",
    )
    text = (
        f"Let A={_text(a)}. Find the matrix A^2 - {k}A + {m}I, where I is the 2 by 2 identity matrix."
        if rng.random() < 0.5
        else f"Given A={_text(a)} and the 2 by 2 identity matrix I, calculate A^2 - {k}A + {m}I."
    )
    cands = [
        _text(_add(_add(_hadamard(a, a), _scale(a, k), -1), _scale([[1, 0], [0, 1]], m))),  # A^2 squared entry by entry
        _text(_add(a2, _scale(a, k), -1)),  # the + mI term left out
        _text(_add(_add(a2, _scale(a, k), -1), [[m, m], [m, m]])),  # m added to every entry instead of the diagonal
        _text(_add(_add(a2, _scale(a, k)), _scale([[1, 0], [0, 1]], m))),  # the sign of kA reversed
    ]
    working = [
        f"A^2 = {_text(a2)}; {k}A = {_text(_scale(a, k))}; {m}I = {_text(_scale([[1, 0], [0, 1]], m))}. "
        f"So A^2 - {k}A + {m}I = {_text(result)}."
    ]
    return _finish(text, _text(result), cands, working)


def _nonzero(rng: random.Random, lo: int = -4, hi: int = 4) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _unknowns(rng: random.Random) -> dict:
    x0, y0 = _nonzero(rng), rng.randint(-4, 4)
    b, c, d = _nonzero(rng, -3, 3), rng.randint(-3, 3), rng.randint(-3, 3)
    e, g, h = _nonzero(rng, -3, 3), rng.randint(-3, 3), rng.randint(-3, 3)
    a_known = [[x0, b], [c, d]]
    b_known = [[e, y0], [g, h]]
    product = _mul(a_known, b_known)
    x, y = sympy.symbols("x y")
    a_sym = sympy.Matrix([[x, b], [c, d]])
    b_sym = sympy.Matrix([[e, y], [g, h]])
    equations = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(a_sym * b_sym, sympy.Matrix(product))]
    solutions = sympy.solve(equations, [x, y], dict=True)
    check(solutions == [{x: x0, y: y0}], "the matrix equation must have exactly the chosen solution")
    answer = f"x = {x0}, y = {y0}"
    text = phrase_unknowns(rng, b, c, d, e, g, h, product)
    # Mistake: the product read entry by entry, so xe = (AB)11 and xy = (AB)12.
    x_entrywise = product[0][0] / e
    cands = [
        f"x = {y0}, y = {x0}",  # the two unknowns swapped
        f"x = {x0}, y = {-y0}",  # sign lost on y
        f"x = {-x0}, y = {y0}",  # sign lost on x
    ]
    if x_entrywise == int(x_entrywise) and x_entrywise != 0 and product[0][1] % int(x_entrywise) == 0:
        x_wrong = int(x_entrywise)
        cands.append(f"x = {x_wrong}, y = {product[0][1] // x_wrong}")
    working = [
        f"Entry (1,1): {e}x + {paren_negative(b)} × {paren_negative(g)} = {product[0][0]}, so x = {x0}. "
        f"Entry (1,2): xy + {paren_negative(b)} × {paren_negative(h)} = {product[0][1]}, so y = {y0}."
    ]
    return _finish(text, answer, cands, working)


def phrase_unknowns(rng, b, c, d, e, g, h, product) -> str:
    a_text = f"[[x, {b}], [{c}, {d}]]"
    b_text = f"[[{e}, y], [{g}, {h}]]"
    if rng.random() < 0.5:
        return f"Let A={a_text} and B={b_text}. Given that AB={_text(product)}, find x and y."
    return f"The matrices A={a_text} and B={b_text} satisfy AB={_text(product)}. Find the values of x and y."


@generator(
    "exd_x_h2_matrix_multiplication",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_matrix_multiplication(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier matrix multiplication: AB - BA, a matrix polynomial, or solving AB = C for unknown entries."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_commutator, _polynomial, _unknowns))(rng)
