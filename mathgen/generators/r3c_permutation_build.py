"""Generates `permutation-binary-matrices` questions (easy, medium, hard) and the exam chain `r3c_permutation_chain`.

Two uses of a matrix of 0s and 1s.

A permutation matrix ``P`` has one 1 in each row and each column; ``P B`` rearranges the entries of a block ``B`` (a
column), so a message written in blocks as the columns of ``M`` is scrambled by ``P M`` and unscrambled by
``P`` transposed (the inverse of a permutation matrix is its transpose). A binary matrix records a relation on a small
set: entry ``(i, j)`` is 1 when the ``i``th number is related to the ``j``th.

* easy: either scramble a message with a given ``P`` (the product ``P M``), or write the binary matrix of a relation
  such as "a divides b" on five numbers;
* medium: unscramble a received message with a given ``P`` (a 3-cycle or 4-cycle, so ``P`` is not its own inverse);
* hard: ``P`` is not given; one block and what it was scrambled to are given (all entries different), and a received
  scrambled message is to be unscrambled, which needs ``P`` found and then inverted;
* exam (``r3c_permutation_chain``): the matrix ``P`` from one block before and after, a second message scrambled,
  how many scrambles return a block to itself, and the fewest scrambles that match a long run (carried forward).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3c_kit import mat_mul, matrix_part, matrix_single, number_part, transpose


def _perm_matrix(sigma):
    n = len(sigma)
    return [[1 if j == sigma[i] else 0 for j in range(n)] for i in range(n)]


def _cycle_lengths(sigma):
    seen, out = set(), []
    for i in range(len(sigma)):
        if i in seen:
            continue
        j, length = i, 0
        while j not in seen:
            seen.add(j)
            j = sigma[j]
            length += 1
        out.append(length)
    return out


def _lcm(values):
    out = 1
    for v in values:
        a, b = out, v
        while b:
            a, b = b, a % b
        out = out * v // a
    return out


def _order(sigma):
    return _lcm(_cycle_lengths(sigma))


def _random_sigma(rng, n, min_order=3):
    for _ in attempts(200, "a permutation of the right order"):
        sigma = list(range(n))
        rng.shuffle(sigma)
        if _order(sigma) >= min_order:
            return sigma
    raise AssertionError  # pragma: no cover


def _ints(m):
    return [[int(v) for v in row] for row in m]


def _mul(a, b):
    return _ints(mat_mul(a, b))


def _message(rng, rows, cols):
    """A message matrix of distinct-looking code numbers between 1 and 26."""
    for _ in attempts(200, "a message with varied entries"):
        m = [[rng.randint(1, 26) for _ in range(cols)] for _ in range(rows)]
        if all(len(set(m[i][j] for i in range(rows))) == rows for j in range(cols)):
            return m
    raise AssertionError  # pragma: no cover


def _show(m) -> str:
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in m) + "]"


def _entry_slip(rng, m):
    out = [list(r) for r in m]
    i, j = rng.randrange(len(m)), rng.randrange(len(m[0]))
    out[i][j] += rng.choice((-1, 1))
    return out


def _scramble_slips(rng, P, M):
    n = len(P)
    Pt = transpose(P)
    out = [
        (_mul(Pt, M), "r3c-perm-inverse-used"),
        (M, "r3c-perm-unchanged"),
    ]
    if len(M[0]) == n:
        out.append((_mul(M, P), "r3c-perm-wrong-order"))
    else:
        out.append((_mul(P, _mul(P, M)), "r3c-perm-applied-twice"))
    out.append((_entry_slip(rng, _mul(P, M)), "r3c-perm-entry-slip"))
    out.append(([list(r) for r in zip(*_mul(P, M))], "r3c-perm-transposed-result"))
    return out


def _easy_scramble(rng):
    for _ in attempts(300, "a scrambled message"):
        n = rng.randint(3, 4)
        sigma = _random_sigma(rng, n, 3)
        P = _perm_matrix(sigma)
        cols = rng.randint(2, 3)
        M = _message(rng, n, cols)
        key = _mul(P, M)
        text = (
            f"A message is turned into numbers and written in blocks of {n}, one block in each column of the matrix "
            f"M = {_show(M)}. A scrambler rearranges each block by multiplying it on the left by the matrix "
            f"P = {_show(P)}. Write the scrambled message PM as a matrix, in the same column layout."
        )
        working = [
            "Each column of PM is P times the matching column of M.",
            f"PM = {_show(key)}.",
        ]
        q = matrix_single(text, key, working, _scramble_slips(rng, P, M))
        if q:
            return q
    raise AssertionError  # pragma: no cover


_RELATIONS = (
    ("a divides b", lambda a, b: b % a == 0, "r3c-perm-relation-reversed"),
    ("b is more than 2 greater than a", lambda a, b: b - a > 2, "r3c-perm-relation-reversed"),
    ("b is exactly 3 greater than a, or exactly 4 greater", lambda a, b: b - a in (3, 4), "r3c-perm-relation-reversed"),
)


def _easy_relation(rng):
    for _ in attempts(500, "a relation matrix"):
        label, rule, _tag = rng.choice(_RELATIONS)
        elements = sorted(rng.sample(range(1, 13), 5))
        R = [[int(rule(a, b)) for b in elements] for a in elements]
        total = sum(map(sum, R))
        if not 6 <= total <= 16 or R == _ints(transpose(R)):
            continue
        n = len(elements)
        diag = [[R[i][j] if i != j else 1 - R[i][j] for j in range(n)] for i in range(n)]
        complement = [[1 - v for v in row] for row in R]
        text = (
            f"A relation on the numbers {', '.join(map(str, elements[:-1]))} and {elements[-1]} is defined by: the number a is related to the "
            f"number b when {label}. Write the binary matrix of the relation, with a row and a column for each number "
            f"in the order {', '.join(map(str, elements))}, and 1 in row a and column b when a is related to b, 0 when it is not. "
            "Write it as rows, for example [[1, 0], [0, 1]]."
        )
        slips = [
            (_ints(transpose(R)), "r3c-perm-relation-reversed"),
            (complement, "r3c-perm-ones-zeros-swapped"),
            (diag, "r3c-perm-diagonal-slip"),
            (_entry_slip(rng, R), "r3c-perm-entry-slip"),
            (_entry_slip(rng, R), "r3c-perm-entry-slip"),
        ]
        working = [
            "Row a, column b is 1 exactly when the rule holds for the pair (a, b), taken in that order.",
            *[f"{a}: {R[i]}" for i, a in enumerate(elements)],
        ]
        q = matrix_single(text, R, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(300, "a message to unscramble"):
        n = rng.randint(3, 4)
        sigma = _random_sigma(rng, n, 3)
        P = _perm_matrix(sigma)
        cols = rng.randint(2, 3)
        M = _message(rng, n, cols)
        E = _mul(P, M)
        Pt = transpose(P)
        text = (
            f"A message in numbers was scrambled by multiplying each block (column) on the left by "
            f"P = {_show(P)}. The scrambled message received is E = {_show(E)}. "
            "Write the original message as a matrix, in the same column layout."
        )
        slips = [
            (_mul(P, E), "r3c-perm-inverse-is-itself"),
            (E, "r3c-perm-unchanged"),
            (_ints(transpose(M)), "r3c-perm-transposed-result"),
            (_mul(Pt, _mul(Pt, E)), "r3c-perm-applied-twice"),
            (_entry_slip(rng, M), "r3c-perm-entry-slip"),
        ]
        working = [
            "The inverse of a permutation matrix is its transpose, so the original is P transposed times E.",
            f"P transposed = {_show(Pt)}.",
            f"Original = {_show(M)}.",
        ]
        q = matrix_single(text, M, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _p_slips(rng, P, sigma):
    n = len(P)
    out = [(_ints(transpose(P)), "r3c-perm-inverse-used")]
    for _ in range(2):
        rows = [list(r) for r in P]
        a, b = rng.sample(range(n), 2)
        rows[a], rows[b] = rows[b], rows[a]
        out.append((rows, "r3c-perm-rows-swapped"))
    for _ in range(2):
        a, b = rng.sample(range(n), 2)
        cols = [[row[b] if j == a else row[a] if j == b else row[j] for j in range(n)] for row in P]
        out.append((cols, "r3c-perm-wrong-positions"))
    return out


def _block_pair_text(rng, B, C):
    return f"the block {tuple(B)} came out as {tuple(C)}"


def _hard(rng):
    for _ in attempts(300, "a permutation to recover"):
        n = rng.randint(4, 5)
        sigma = _random_sigma(rng, n, 3)
        P = _perm_matrix(sigma)
        B = rng.sample(range(1, 27), n)
        C = [B[sigma[i]] for i in range(n)]
        check(_mul(P, [[b] for b in B]) == [[c] for c in C], "the permutation matrix does not map the block")
        M = _message(rng, n, 2)
        E = _mul(P, M)
        Pt = transpose(P)
        bad = [list(r) for r in P]
        a, b = rng.sample(range(n), 2)
        bad[a], bad[b] = bad[b], bad[a]
        text = phrase(
            rng,
            [
                "A scrambler rearranges every block of {n} numbers by multiplying it on the left by a permutation "
                "matrix P, which is not given. It is seen that {seen}. A message written in blocks as the columns "
                "of a matrix M is scrambled in the same way and received as E = {E}. Write the original message M as "
                "a matrix, in the same column layout.",
                "Every block of {n} numbers in a message is multiplied on the left by a permutation matrix P that "
                "is not given. It is seen that {seen}. The blocks of the message M, written as its columns, are "
                "scrambled in the same way, and the result received is E = {E}. Recover M as a matrix with the same "
                "column layout.",
                "A secret code rearranges blocks of {n} numbers using an unknown permutation matrix P, applied on "
                "the left. It is seen that {seen}. A message M, written in blocks as its columns, is sent through "
                "the same code and arrives as E = {E}. Write M as a matrix in the same column layout.",
            ],
            n=n,
            seen=_block_pair_text(rng, B, C),
            E=_show(E),
        )
        slips = [
            (_mul(P, E), "r3c-perm-inverse-is-itself"),
            (_mul(transpose(bad), E), "r3c-perm-rows-swapped"),
            (E, "r3c-perm-unchanged"),
            (_entry_slip(rng, M), "r3c-perm-entry-slip"),
            ([list(r) for r in zip(*M)], "r3c-perm-transposed-result"),
        ]
        working = [
            "Find P from the block: position i of the result is the entry of the original block that P moves to position i.",
            *[
                f"Position {i + 1} of the result is {C[i]}, which was position {sigma[i] + 1} of the block."
                for i in range(n)
            ],
            "The inverse of a permutation matrix is its transpose, so the original message is P transposed times E.",
            f"M = {_show(M)}.",
        ]
        q = matrix_single(text, M, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_permutation",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_permutation(rng, tier="easy"):
    """Permutation and binary matrices; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy_scramble(rng) if rng.random() < 0.5 else _easy_relation(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("r3c_permutation_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_permutation_chain(rng, tier="exam"):
    """Exam chain: P from a block, a message scrambled, the order of P, and the fewest scrambles matching a long run."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(400, "a permutation chain"):
        n = 4
        sigma = _random_sigma(rng, n, 3)
        P = _perm_matrix(sigma)
        order = _order(sigma)
        B = rng.sample(range(1, 27), n)
        C = [B[sigma[i]] for i in range(n)]
        M2 = _message(rng, n, 2)
        runs = rng.randint(60, 120)
        r = runs % order
        if r == 0:
            continue
        stem = phrase(
            rng,
            [
                "A scrambler rearranges every block of {n} numbers by multiplying it on the left by a permutation "
                "matrix P. It is seen that {seen}.",
                "To hide a message, every block of {n} numbers is multiplied on the left by a permutation matrix P. "
                "It is seen that {seen}.",
                "A code sends numbers in blocks of {n}, and each block is rearranged by multiplying it on the left "
                "by a permutation matrix P. It is seen that {seen}.",
            ],
            n=n,
            seen=_block_pair_text(rng, B, C),
        )
        part_a = matrix_part(
            "Write the matrix P.",
            P,
            3,
            [
                f"Position {i + 1} of the result is {C[i]}, which was position {sigma[i] + 1} of the block."
                for i in range(n)
            ],
            _p_slips(rng, P, sigma),
        )
        key_b = _mul(P, M2)
        part_b = matrix_part(
            f"A message is written in blocks as the columns of M = {_show(M2)}. Write the scrambled message PM.",
            key_b,
            2,
            [f"PM = {_show(key_b)}."],
            _scramble_slips(rng, P, M2),
        )
        lengths = _cycle_lengths(sigma)
        part_c = number_part(
            "A block is scrambled again and again. What is the smallest number of scrambles that returns every block to its original arrangement?",
            order,
            2,
            [
                f"The positions move in cycles of lengths {', '.join(map(str, sorted(lengths)))}, so the smallest number is {order}."
            ],
            [
                (n, "r3c-perm-order-is-size"),
                (sum(lengths) if sum(lengths) != order else None, "r3c-perm-order-is-size"),
                (order + 1, "r3c-perm-count-slip"),
                (order - 1, "r3c-perm-count-slip"),
                (order * 2, "r3c-perm-order-is-size"),
            ],
            lo=2,
            sensible=(1, None),
        )
        part_d = number_part(
            f"A block is scrambled {runs} times in a row. What is the smallest number of scrambles that would have left it in the same arrangement?",
            r,
            2,
            [
                f"{runs} = {runs // order} x {order} + {r}, and each full {order} scrambles changes nothing, so {'1 scramble gives' if r == 1 else str(r) + ' scrambles give'} the same arrangement."
            ],
            [
                (runs // order, "r3c-perm-quotient-not-remainder"),
                (order - r, "r3c-perm-inverse-used"),
                (runs % n if runs % n != r else None, "r3c-perm-order-is-size"),
                (runs - order * (runs // order - 1), "r3c-perm-count-slip"),
                (r + 1, "r3c-perm-count-slip"),
            ],
            follow=f"{runs} - part_c*floor({runs}/part_c)",
            lo=1,
            sensible=(0, None),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
