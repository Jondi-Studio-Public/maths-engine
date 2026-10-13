"""Narrow, mechanically gradable university algebra and logic generators.

This module intentionally stops at finite/computational slices.  The skills
which consume it do not claim proof competence or mastery of abstract
structures; they expose small calculations that can be independently checked
by enumeration, exact arithmetic, or polynomial identities.
"""

from __future__ import annotations

from itertools import product
from math import gcd

import sympy

from ..core import check, generator, latex_answer, pick_distractors


# ---------------------------------------------------------------------------
# Permutations


def _compose_permutations(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return left after right, using one-based image notation."""
    return tuple(left[right[i] - 1] for i in range(len(left)))


def _apply_permutation(permutation: tuple[int, ...], value: int) -> int:
    return permutation[value - 1]


def _perm_text(permutation: tuple[int, ...]) -> str:
    return "(" + ", ".join(map(str, permutation)) + ")"


@generator(
    "permutation_composition",
    level=16,
    presets={
        "easy": dict(size=3),
        "medium": dict(size=4),
        "hard": dict(size=5),
    },
)
def permutation_composition(rng, size=4):
    """Compose two explicitly listed permutations on {1,...,n}."""
    p = tuple(rng.sample(range(1, size + 1), size))
    q = tuple(rng.sample(range(1, size + 1), size))
    answer = _compose_permutations(p, q)  # p after q

    # Verify the mapping in a second representation, element by element.
    independent = tuple(_apply_permutation(p, _apply_permutation(q, i)) for i in range(1, size + 1))
    check(answer == independent, "permutation composition mismatch")
    check(sorted(answer) == list(range(1, size + 1)), "composition is not a permutation")

    reverse = _compose_permutations(q, p)
    p_first_identity = tuple(_apply_permutation(q, _apply_permutation(p, i)) for i in range(1, size + 1))
    candidates = [_perm_text(reverse), _perm_text(p_first_identity), _perm_text(p), _perm_text(q)]
    # Identity inputs or commuting pairs can collapse the most obvious
    # misconceptions.  Rotations and an adjacent swap supply stable,
    # permutation-shaped alternatives without ever copying the answer.
    for offset in range(1, size):
        candidates.append(_perm_text(answer[offset:] + answer[:offset]))
    candidates.append(
        _perm_text(tuple(answer[1] if i == 0 else answer[0] if i == 1 else answer[i] for i in range(size)))
    )
    answer_text = _perm_text(answer)
    return dict(
        text=(
            f"Let p = {_perm_text(p)} and q = {_perm_text(q)}, where each tuple "
            "lists the images of 1, 2, ... . Find the composition p after q."
        ),
        latex=(
            rf"p={_perm_text(p)},\;q={_perm_text(q)};\quad "
            r"(p\circ q)(i)=p(q(i)).\;\text{Find }p\circ q."
        ),
        answer=list(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Apply q, then p, to each i = 1,...,{size}.", f"The image tuple is {answer_text}."],
        distractors=pick_distractors(answer_text, candidates),
    )


# ---------------------------------------------------------------------------
# Modular arithmetic and cyclic groups


@generator(
    "modular_multiplicative_inverse",
    level=16,
    presets={
        "easy": dict(modulus_hi=11),
        "medium": dict(modulus_hi=19),
        "hard": dict(modulus_hi=31),
    },
)
def modular_multiplicative_inverse(rng, modulus_hi=19, kind=None):
    """Find a in [1,m-1] inverse modulo m, restricted to gcd(a,m)=1.

    ``kind="solve_congruence"`` generalises the fixed "≡ 1"
    target to a GENERAL ``b`` drawn independently of ``value``/``modulus``:
    "Find x such that {value}x ≡ {b} (mod {modulus})". Solving it requires
    finding the inverse of ``value`` AND THEN multiplying by ``b``
    (``x = (inverse_of_value * b) % modulus``) -- an extra reasoning step
    beyond "find the inverse" alone, drawing on both of the skill's own
    prerequisites (modular-arithmetic-congruence, euclidean-algorithm-gcd).
    ``b`` ranges over ``range(1, modulus)``, excluding ``b=0``: when ``b=0``
    the answer ``x = (inverse * 0) % modulus`` is always ``0`` regardless of
    ``inverse``, so the answer would be invariant to the very "find the
    inverse" step the escalation exists to test. ``value == 1`` draws are
    also excluded (redrawn until ``value != 1``), since 1's own inverse is
    trivially 1 and the "find the inverse" half of the two-step reasoning
    would be vacuous. The default ``kind=None`` path below (the original
    "find the inverse" question, fixed at ``b=1``) is untouched byte-for-byte.
    """
    modulus = rng.randint(5, modulus_hi)
    value = rng.randint(1, modulus - 1)
    while gcd(value, modulus) != 1:
        modulus = rng.randint(5, modulus_hi)
        value = rng.randint(1, modulus - 1)

    inverse = pow(value, -1, modulus)
    enumerated = next(candidate for candidate in range(1, modulus) if value * candidate % modulus == 1)
    check((value * inverse) % modulus == 1, "inverse does not satisfy congruence")
    check(inverse == enumerated, "inverse disagrees with exhaustive verification")

    if kind == "solve_congruence":
        # value=1's own inverse is trivially 1, so the "find the inverse"
        # half of this escalation would be vacuous for that draw -- redraw a
        # fresh coprime modulus/value pair, the same way the shared block
        # above does, looping until value != 1 too, and recompute inverse
        # (and re-verify it) against the NEW pair rather than reusing the
        # old one.
        if value == 1:
            while value == 1 or gcd(value, modulus) != 1:
                modulus = rng.randint(5, modulus_hi)
                value = rng.randint(1, modulus - 1)
            inverse = pow(value, -1, modulus)
            enumerated = next(candidate for candidate in range(1, modulus) if value * candidate % modulus == 1)
            check((value * inverse) % modulus == 1, "inverse does not satisfy congruence")
            check(inverse == enumerated, "inverse disagrees with exhaustive verification")

        b = rng.randint(1, modulus - 1)
        x = (inverse * b) % modulus
        # Independent check route 1: x satisfies the congruence directly.
        check((value * x) % modulus == b % modulus, "solve_congruence: x does not satisfy value*x ≡ b (mod modulus)")
        # Independent check route 2: recompute the inverse of `value` from
        # scratch under its own name and re-derive x, cross-checked equal to
        # the constructed x -- mirrors the base branch's own
        # inverse-then-exhaustive-search independent pair.
        inverse_of_value = pow(value, -1, modulus)
        x_expected = (inverse_of_value * b) % modulus
        check(x == x_expected, "solve_congruence: constructed x disagrees with from-scratch recomputation")

        answer_text = str(x)
        # Misconceptions: (1) found the inverse of `value` correctly but
        # forgot to multiply by `b` -- reports the inverse itself, the
        # direct sibling of the base skill's own answer when b=1; (2)
        # multiplied `value` by `b` directly instead of by the inverse,
        # confusing "multiply by the inverse" with "multiply by the
        # coefficient itself"; (3) used the wrong residue for `b` (reduced
        # `b` incorrectly, here `b - 1`, before multiplying by the inverse).
        wrong_forgot_multiply = str(inverse_of_value)
        wrong_multiplied_by_coefficient = str((value * b) % modulus)
        wrong_bad_residue_for_b = str((inverse_of_value * ((b - 1) % modulus)) % modulus)
        candidates = [
            wrong_forgot_multiply,
            wrong_multiplied_by_coefficient,
            wrong_bad_residue_for_b,
            str((x + 1) % modulus),
        ]
        coefficient_text = "x" if value == 1 else f"{value}x"
        return dict(
            text=(f"Find x in {{0,...,{modulus - 1}}} such that {coefficient_text} ≡ {b} (mod {modulus})."),
            latex=rf"\text{{Find }}x:\; {coefficient_text}\equiv {b}\pmod{{{modulus}}}.",
            answer=x,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"The inverse of {value} modulo {modulus} is {inverse_of_value} "
                f"(since {value}·{inverse_of_value} ≡ 1 (mod {modulus})).",
                f"x = {inverse_of_value}·{b} mod {modulus} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    answer_text = str(inverse)
    candidates = [str((modulus - inverse) % modulus), str(value), str((inverse + 1) % modulus), str(modulus - 1)]
    coefficient_text = "x" if value == 1 else f"{value}x"
    return dict(
        text=(
            f"Find the multiplicative inverse of {value} modulo {modulus}; "
            f"that is, find x in {{1,...,{modulus - 1}}} with "
            f"{coefficient_text} ≡ 1 (mod {modulus})."
        ),
        latex=rf"\text{{Find }}x:\; {coefficient_text}\equiv 1\pmod{{{modulus}}}.",
        answer=inverse,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"gcd({value}, {modulus}) = 1, so an inverse exists.",
            f"Check: {value}·{inverse} ≡ 1 (mod {modulus}).",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


def _cyclic_order_of_sum(rng, modulus_hi):
    """Hard tier: order of [a] + [b] in Z/nZ, where a + b wraps past n."""
    for _ in range(500):
        modulus = rng.randint(8, max(8, modulus_hi))
        a = rng.randint(1, modulus - 1)
        b = rng.randint(1, modulus - 1)
        total = a + b
        if total < modulus or total % modulus == 0:
            continue
        residue = total % modulus
        expected = modulus // gcd(residue, modulus)
        order_a = modulus // gcd(a, modulus)
        order_b = modulus // gcd(b, modulus)
        lcm_ab = order_a * order_b // gcd(order_a, order_b)
        rivals = {lcm_ab, order_a + order_b, order_a * order_b, max(order_a, order_b)}
        if expected in rivals:
            continue
        break
    else:
        raise AssertionError("no admissible cyclic-sum draw")
    enumerated = None
    current = 0
    for count in range(1, modulus + 1):
        current = (current + residue) % modulus
        if current == 0:
            enumerated = count
            break
    check(enumerated == expected, "cyclic-sum order mismatch")
    check((expected * total) % modulus == 0, "cyclic-sum order fails closure check")
    check(expected not in rivals, "cyclic-sum answer collides with a misconception")
    answer_text = str(expected)
    candidates = [
        str(v) for v in (lcm_ab, order_a + order_b, order_a * order_b, max(order_a, order_b), modulus, order_a, order_b)
    ]
    candidates.extend(str(candidate) for candidate in range(1, 2 * modulus + 1))
    distractors = pick_distractors(answer_text, candidates)
    check(all(d != answer_text for d in distractors), "cyclic-sum distractor equals answer")
    return dict(
        text=f"In the additive cyclic group Z/{modulus}Z, find the order of the element [{a}] + [{b}].",
        latex=rf"\text{{In }}\mathbb{{Z}}/{modulus}\mathbb{{Z}},\; "
        rf"\text{{find }}\operatorname{{ord}}([{a}]+[{b}]).",
        answer=expected,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Add first: [{a}] + [{b}] = [{total}] = [{residue}] in Z/{modulus}Z.",
            f"The order of [{residue}] is {modulus}/gcd({residue}, {modulus}) = {expected}.",
        ],
        distractors=distractors,
    )


@generator(
    "cyclic_group_element_order",
    level=16,
    presets={
        "easy": dict(modulus_hi=9),
        "medium": dict(modulus_hi=15),
        "hard": dict(modulus_hi=24),
    },
)
def cyclic_group_element_order(rng, modulus_hi=15, kind=None):
    """Find the additive order of [a] in the finite cyclic group Z/nZ."""
    if kind == "sum":
        return _cyclic_order_of_sum(rng, modulus_hi)
    modulus = rng.randint(4, modulus_hi)
    value = rng.randint(0, modulus - 1)
    expected = 1 if value == 0 else modulus // gcd(value, modulus)
    current = 0
    enumerated = None
    for count in range(1, modulus + 1):
        current = (current + value) % modulus
        if current == 0:
            enumerated = count
            break
    check(enumerated == expected, "cyclic-group order mismatch")
    check((expected * value) % modulus == 0, "order fails closure check")
    answer_text = str(expected)
    candidates = [str(modulus), str(value or modulus), str(max(1, expected - 1)), str(gcd(value, modulus))]
    candidates.extend(str(candidate) for candidate in range(1, 2 * modulus + 1))
    return dict(
        text=(f"In the additive cyclic group Z/{modulus}Z, find the order of the element [{value}]."),
        latex=rf"\text{{In }}\mathbb{{Z}}/{modulus}\mathbb{{Z}},\; "
        rf"\text{{find }}\operatorname{{ord}}([{value}]).",
        answer=expected,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Repeatedly add [{value}] modulo {modulus} until [0] returns.",
            f"The first return occurs after {expected} additions.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# ---------------------------------------------------------------------------
# Polynomial quotient and remainder


def _poly_display(expr) -> str:
    text = str(sympy.expand(expr)).replace("**", "^")
    return text.replace("*", "")


def _qr_display(quotient, remainder) -> str:
    return f"Q(x) = {_poly_display(quotient)}, R(x) = {_poly_display(remainder)}"


@generator(
    "integer_polynomial_quotient_remainder",
    level=16,
    presets={
        "easy": dict(divisor_degree=1, quotient_degree=1, coefficient_hi=4),
        "medium": dict(divisor_degree=2, quotient_degree=1, coefficient_hi=5),
        "hard": dict(divisor_degree=2, quotient_degree=2, coefficient_hi=7),
    },
)
def integer_polynomial_quotient_remainder(rng, divisor_degree=2, quotient_degree=1, coefficient_hi=5):
    """Find Q and R for f = gQ + R over Z[x], using a monic divisor."""
    x = sympy.symbols("x")
    divisor_coefficients = [rng.randint(-coefficient_hi, coefficient_hi) for _ in range(divisor_degree)] + [1]
    quotient_coefficients = [rng.randint(-coefficient_hi, coefficient_hi) for _ in range(quotient_degree + 1)]
    remainder_coefficients = [rng.randint(-coefficient_hi, coefficient_hi) for _ in range(divisor_degree)]
    divisor = sum(coefficient * x**power for power, coefficient in enumerate(divisor_coefficients))
    quotient = sum(coefficient * x**power for power, coefficient in enumerate(quotient_coefficients))
    remainder = sum(coefficient * x**power for power, coefficient in enumerate(remainder_coefficients))
    dividend = sympy.expand(divisor * quotient + remainder)

    q_check, r_check = sympy.div(sympy.Poly(dividend, x, domain=sympy.ZZ), sympy.Poly(divisor, x, domain=sympy.ZZ))
    check(sympy.expand(q_check.as_expr() - quotient) == 0, "polynomial quotient mismatch")
    check(sympy.expand(r_check.as_expr() - remainder) == 0, "polynomial remainder mismatch")
    check(
        sympy.degree(remainder, x) < divisor_degree or remainder == 0,
        "remainder degree is not smaller than divisor degree",
    )

    answer_text = _qr_display(quotient, remainder)
    alternatives = [
        _qr_display(quotient + 1, remainder),
        _qr_display(quotient, sympy.expand(remainder + divisor)),
        _qr_display(-quotient, remainder),
        _qr_display(quotient, sympy.expand(remainder - divisor)),
    ]
    return dict(
        text=(
            f"Divide f(x) = {_poly_display(dividend)} by "
            f"g(x) = {_poly_display(divisor)} over the integers. "
            "Give the quotient Q(x) and remainder R(x)."
        ),
        latex=(
            rf"f(x)={sympy.latex(dividend)},\quad g(x)={sympy.latex(divisor)}; "
            r"\text{find }Q,R\text{ with }f=gQ+R."
        ),
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=rf"Q={sympy.latex(quotient)},\;R={sympy.latex(remainder)}",
        working=[
            f"Use exact division in Z[x]: f = gQ + R.",
            f"Q(x) = {_poly_display(quotient)} and R(x) = {_poly_display(remainder)}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


# ---------------------------------------------------------------------------
# Propositional equivalence


def _prop_eval(tree: tuple, values: dict[str, bool]) -> bool:
    op = tree[0]
    if op == "atom":
        return values[tree[1]]
    if op == "not":
        return not _prop_eval(tree[1], values)
    left, right = _prop_eval(tree[1], values), _prop_eval(tree[2], values)
    if op == "and":
        return left and right
    if op == "or":
        return left or right
    if op == "implies":
        return (not left) or right
    raise ValueError(op)


def _prop_render(tree: tuple) -> str:
    if tree[0] == "atom":
        return tree[1]
    if tree[0] == "not":
        return f"¬{_prop_render(tree[1])}"
    symbol = {"and": "∧", "or": "∨", "implies": "→"}[tree[0]]
    return f"({_prop_render(tree[1])} {symbol} {_prop_render(tree[2])})"


def _prop_atoms(tree: tuple) -> set[str]:
    if tree[0] == "atom":
        return {tree[1]}
    if tree[0] == "not":
        return _prop_atoms(tree[1])
    return _prop_atoms(tree[1]) | _prop_atoms(tree[2])


@generator(
    "propositional_formula_equivalence",
    level=16,
    presets={
        "easy": dict(complexity=1),
        "medium": dict(complexity=2),
        "hard": dict(complexity=3),
    },
)
def propositional_formula_equivalence(rng, complexity=2):
    """Decide equivalence of two small formulas by exhaustive truth tables."""
    p, q, r = ("atom", "p"), ("atom", "q"), ("atom", "r")
    equivalent_pairs = [
        (p, ("or", ("and", p, q), ("and", p, ("not", q)))),
        (("or", p, q), ("or", q, p)),
        (("and", p, ("or", q, r)), ("or", ("and", p, q), ("and", p, r))),
        (("implies", p, q), ("or", ("not", p), q)),
    ]
    non_equivalent_pairs = [
        (("and", p, q), ("or", p, q)),
        (("implies", p, q), ("implies", q, p)),
        (("and", p, ("or", q, r)), ("or", ("and", p, q), r)),
        (("not", ("and", p, q)), ("or", ("not", p), q)),
    ]
    if complexity >= 3:
        # The hard tier draws its own, deeper pairs (nested implication,
        # contrapositive, De Morgan, distribution), so it is a different
        # question pool from the medium tier rather than the same one relabelled.
        equivalent_pairs = [
            (("implies", p, q), ("implies", ("not", q), ("not", p))),
            (("not", ("and", p, q)), ("or", ("not", p), ("not", q))),
            (("or", p, ("and", q, r)), ("and", ("or", p, q), ("or", p, r))),
            (("implies", p, ("implies", q, r)), ("implies", ("and", p, q), r)),
        ]
        non_equivalent_pairs = [
            (("implies", p, ("implies", q, r)), ("implies", ("implies", p, q), r)),
            (("not", ("or", p, q)), ("or", ("not", p), ("not", q))),
            (("implies", p, q), ("implies", ("not", p), ("not", q))),
            (("or", p, ("and", q, r)), ("and", ("or", p, q), r)),
        ]
    if complexity <= 1:
        # Easy: short two-variable pairs, with the relation drawn first so each of the four is the key on a fair
        # share of seeds (equivalent, first implies second only, second implies first only, neither).
        easy_pairs = [
            (p, ("or", ("and", p, q), ("and", p, ("not", q)))),
            (("or", p, q), ("or", q, p)),
            (("implies", p, q), ("or", ("not", p), q)),
            (("and", p, q), ("or", p, q)),
            (("and", p, q), p),
            (("not", ("or", p, q)), ("and", ("not", p), ("not", q))),
            (("or", p, q), ("and", p, q)),
            (p, ("and", p, q)),
            (("not", ("and", p, q)), ("and", ("not", p), ("not", q))),
            (("implies", p, q), ("implies", q, p)),
            (("and", p, ("not", q)), ("or", ("not", p), q)),
            (("or", p, q), ("and", p, ("not", q))),
        ]
        groups: dict = {}
        for pair in easy_pairs:
            rows = list(product([False, True], repeat=2))
            lv = [_prop_eval(pair[0], dict(zip("pq", row))) for row in rows]
            rv = [_prop_eval(pair[1], dict(zip("pq", row))) for row in rows]
            kind = (all(not a or b for a, b in zip(lv, rv)), all(not b or a for a, b in zip(lv, rv)))
            groups.setdefault(kind, []).append(pair)
        left, right = rng.choice(groups[rng.choice(sorted(groups))])
    else:
        left, right = rng.choice(equivalent_pairs if rng.random() < 0.5 else non_equivalent_pairs)
    atoms = sorted(_prop_atoms(left) | _prop_atoms(right))
    rows = list(product([False, True], repeat=len(atoms)))
    left_values = [_prop_eval(left, dict(zip(atoms, row))) for row in rows]
    right_values = [_prop_eval(right, dict(zip(atoms, row))) for row in rows]
    equivalent = left_values == right_values
    # Independent row-by-row witness search, deliberately separate from the
    # vector comparison used above.
    witnesses = [
        row for row in rows if _prop_eval(left, dict(zip(atoms, row))) != _prop_eval(right, dict(zip(atoms, row)))
    ]
    check(equivalent == (not witnesses), "truth-table equivalence mismatch")

    # The four relations a pair of formulas can have, read off the truth table. Exactly one holds, and each is
    # the key for some pairs, so every option is a genuine answer. The slips they catch: declaring equivalence from
    # the rows that agree, and getting the direction of implication wrong (the converse slip).
    left_implies_right = all((not lv) or rv for lv, rv in zip(left_values, right_values))
    right_implies_left = all((not rv) or lv for lv, rv in zip(left_values, right_values))
    relation_options = {
        "equivalent": "Equivalent",
        "left": "Not equivalent: the first formula implies the second, but not the other way round",
        "right": "Not equivalent: the second formula implies the first, but not the other way round",
        "neither": "Not equivalent: neither formula implies the other",
    }
    if left_implies_right and right_implies_left:
        relation = "equivalent"
    elif left_implies_right:
        relation = "left"
    elif right_implies_left:
        relation = "right"
    else:
        relation = "neither"
    check((relation == "equivalent") == equivalent, "implication directions disagree with the equivalence check")
    answer_text = relation_options[relation]
    alternatives = [t for k, t in relation_options.items() if k != relation]
    return dict(
        text=(
            f"Are the propositional formulas {_prop_render(left)} and "
            f"{_prop_render(right)} logically equivalent? Check every truth assignment."
        ),
        latex=(
            rf"\text{{Are }} {_prop_render(left)} \text{{ and }} "
            rf"{_prop_render(right)} \text{{ logically equivalent?}}"
        ),
        answer=equivalent,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Compare both formulas on all {len(rows)} truth assignments.",
            (
                "No counterexample row exists."
                if equivalent
                else f"A counterexample exists (for example, row {witnesses[0]})."
            ),
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )
