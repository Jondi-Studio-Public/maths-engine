"""Small, mechanically checked generators for the university foundation pilot.

These topics deliberately stay narrow: they produce typed or multiple-choice
answers whose truth can be checked independently of the wording.  They are
not intended to stand in for a proof or abstract-algebra assessment yet; the
corresponding skills therefore describe only the observable computational
slice covered by this pilot.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import GRADER_TOLERANCE, check, draw_until, generator, paren_negative, pick_distractors, rational_text
from .primary_math import _aligned


# ---------------------------------------------------------------------------
# Propositional logic


def _truth_eval(tree: tuple, values: dict[str, bool]) -> bool:
    """Evaluate a small propositional AST using explicit Boolean semantics."""
    op = tree[0]
    if op == "atom":
        return values[tree[1]]
    if op == "not":
        return not _truth_eval(tree[1], values)
    left, right = _truth_eval(tree[1], values), _truth_eval(tree[2], values)
    if op == "and":
        return left and right
    if op == "or":
        return left or right
    if op == "implies":
        return (not left) or right
    raise ValueError(f"unknown proposition operator: {op}")


def _truth_render(tree: tuple) -> str:
    op = tree[0]
    if op == "atom":
        return tree[1]
    if op == "not":
        child = _truth_render(tree[1])
        return f"¬{child}" if tree[1][0] == "atom" else f"¬({child})"
    symbol = {"and": "∧", "or": "∨", "implies": "→"}[op]
    return f"({_truth_render(tree[1])} {symbol} {_truth_render(tree[2])})"


def _truth_python(tree: tuple) -> str:
    """A second, independent representation used for verification."""
    op = tree[0]
    if op == "atom":
        return tree[1]
    if op == "not":
        return f"(not {_truth_python(tree[1])})"
    left, right = _truth_python(tree[1]), _truth_python(tree[2])
    if op == "and":
        return f"({left} and {right})"
    if op == "or":
        return f"({left} or {right})"
    if op == "implies":
        return f"((not {left}) or {right})"
    raise ValueError(f"unknown proposition operator: {op}")


def _atom(name: str) -> tuple:
    return ("atom", name)


@generator(
    "propositional_truth_value",
    level=14,
    presets={
        "easy": dict(complexity=1),
        "medium": dict(complexity=2),
        "hard": dict(complexity=3),
    },
)
def propositional_truth_value(rng, complexity=2):
    """Evaluate a proposition after assigning Boolean values to its atoms."""
    names = ["p", "q", "r", "s"]
    n_atoms = 2 if complexity <= 1 else (3 if complexity == 2 else 4)
    values = {name: bool(rng.getrandbits(1)) for name in names[:n_atoms]}
    atoms = [_atom(name) for name in names[:n_atoms]]

    if complexity <= 1:
        tree = rng.choice(
            [
                ("not", atoms[0]),
                ("and", atoms[0], atoms[1]),
                ("or", atoms[0], atoms[1]),
                ("implies", atoms[0], atoms[1]),
            ]
        )
    elif complexity == 2:
        tree = rng.choice(
            [
                ("and", ("not", atoms[0]), ("or", atoms[1], atoms[2])),
                ("or", ("and", atoms[0], atoms[1]), ("not", atoms[2])),
                ("implies", ("or", atoms[0], atoms[1]), atoms[2]),
                ("and", ("implies", atoms[0], atoms[1]), ("not", atoms[2])),
            ]
        )
    else:
        tree = rng.choice(
            [
                ("and", ("or", ("not", atoms[0]), atoms[1]), ("implies", atoms[2], atoms[3])),
                ("implies", ("and", atoms[0], ("not", atoms[1])), ("or", atoms[2], atoms[3])),
                ("or", ("implies", atoms[0], atoms[1]), ("and", ("not", atoms[2]), atoms[3])),
            ]
        )

    expression = _truth_render(tree)
    answer = _truth_eval(tree, values)
    # Independent check through a restricted Python Boolean expression.  The
    # expression is generated from a closed AST, not student input.
    alternate = bool(eval(_truth_python(tree), {"__builtins__": {}}, values))
    check(answer == alternate, "propositional truth evaluation mismatch")

    assignment = ", ".join(f"{key} = {str(value)}" for key, value in values.items())
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(not answer), "Undetermined", "Both true and false"])
    return dict(
        text=f"Given {assignment}, evaluate {expression}.",
        latex=expression.replace("¬", r"\neg ").replace("∧", r"\land ").replace("∨", r"\lor ").replace("→", r"\to "),
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"\mathrm{{{answer_text}}}",
        working=[f"Substitute the stated Boolean values into {expression}: {answer_text}."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Finite sets


def _set_text(values: set[int] | list[int]) -> str:
    ordered = sorted(values)
    return "{" + ", ".join(map(str, ordered)) + "}"


@generator(
    "finite_set_operations",
    level=14,
    presets={
        "easy": dict(elements_hi=6),
        "medium": dict(elements_hi=9),
        "hard": dict(elements_hi=14),
    },
)
def finite_set_operations(rng, elements_hi=9):
    """Compute one binary operation on two finite sets of integers."""
    universe = set(range(1, elements_hi + 1))
    operations = ("union", "intersection", "difference", "symmetric difference")
    operation = rng.choice(operations)
    # Redraw so the answer is non-empty and the two sets are meaningfully
    # different; this keeps the question useful without changing its contract.
    for _ in range(100):
        a = {x for x in universe if rng.random() < 0.45}
        b = {x for x in universe if rng.random() < 0.45}
        if a != b:
            builtin = {
                "union": a | b,
                "intersection": a & b,
                "difference": a - b,
                "symmetric difference": a ^ b,
            }[operation]
            if builtin:
                break
    else:  # pragma: no cover - defensive fallback for pathological RNGs
        a, b, builtin = {1, 2}, {2, 3}, {2}

    manual = {
        x
        for x in universe
        if (operation == "union" and (x in a or x in b))
        or (operation == "intersection" and (x in a and x in b))
        or (operation == "difference" and (x in a and x not in b))
        or (operation == "symmetric difference" and ((x in a) != (x in b)))
    }
    check(builtin == manual, "finite-set operation mismatch")
    answer = sorted(builtin)
    answer_text = _set_text(answer)

    alternatives = [
        _set_text(a & b),
        _set_text(a | b),
        _set_text(b - a),
        _set_text(a),
        _set_text(b),
        _set_text(universe - builtin),
    ]
    text_symbol = {
        "union": "∪",
        "intersection": "∩",
        "difference": "∖",
        "symmetric difference": "△",
    }[operation]
    latex_symbol = {
        "union": r"\cup",
        "intersection": r"\cap",
        "difference": r"\setminus",
        "symmetric difference": r"\triangle",
    }[operation]
    return dict(
        text=f"Let A = {_set_text(a)} and B = {_set_text(b)}. Find A {text_symbol} B.",
        latex=rf"A = {_set_text(a)},\; B = {_set_text(b)};\quad A\, {latex_symbol}\, B",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Apply {operation} element-by-element to obtain {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


# ---------------------------------------------------------------------------
# Modular arithmetic


@generator(
    "modular_arithmetic",
    level=15,
    presets={
        "easy": dict(modulus_hi=7, magnitude_hi=20),
        "medium": dict(modulus_hi=12, magnitude_hi=80),
        "hard": dict(modulus_hi=19, magnitude_hi=250),
    },
)
def modular_arithmetic(rng, modulus_hi=12, magnitude_hi=80, kind=None):
    """Find the least non-negative residue in a congruence class.

    ``kind=None`` (the default) preserves the original single-integer
    behaviour byte-for-byte: draw ONE integer and reduce it to its least
    non-negative residue. ``kind="sum_congruence"`` draws TWO integers
    ``a``/``b`` -- each from roughly HALF of ``magnitude_hi`` so their sum
    lands in a comparable magnitude band to the single-value case's own
    ``[-magnitude_hi, magnitude_hi]`` draw -- and asks for the residue of
    their SUM, requiring the student to add first and only then reduce,
    rather than reduce a single given integer directly.
    """
    # The drawn number must reach the modulus in size, whichever modulus up to
    # modulus_hi is drawn, so the largest number that can be drawn must reach modulus_hi.
    if kind == "sum_congruence":
        half_hi = max(1, magnitude_hi // 2)
        reach, needed = 2 * half_hi, 2 * -(-modulus_hi // 2)
        if reach < modulus_hi:
            raise ValueError(
                f"modular_arithmetic needs magnitude_hi >= {needed} for modulus_hi={modulus_hi} "
                "when kind='sum_congruence': the sum of the two integers must be able to reach the modulus"
            )
    elif magnitude_hi < modulus_hi:
        raise ValueError(
            f"modular_arithmetic needs magnitude_hi >= {modulus_hi} for modulus_hi={modulus_hi}: "
            "the integer must be able to reach the modulus in size"
        )
    modulus = rng.randint(2, modulus_hi)
    if kind == "sum_congruence":
        # Same discipline as the single-value draw below: redraw until the
        # question actually exercises reduction (the SUM's magnitude reaches
        # the modulus), including negative representatives. The odds of a
        # draw reaching the modulus can be about 1 in (2 * half_hi + 1) ** 2,
        # so the cap is scaled to stay far beyond any realistic run of misses.
        a, b = draw_until(
            rng,
            lambda r: (r.randint(-half_hi, half_hi), r.randint(-half_hi, half_hi)),
            lambda pair: abs(pair[0] + pair[1]) >= modulus,
            tries=100 * (2 * half_hi + 1) ** 2,
            what="two integers whose sum reaches the modulus",
        )
        total = a + b
        answer = total % modulus
        independent = total - modulus * math.floor(total / modulus)
        enumerated = next(r for r in range(modulus) if (total - r) % modulus == 0)
        check(answer == independent == enumerated, "modular sum-congruence residue mismatch")

        answer_text = str(answer)
        # Misconceptions: reduced a and b SEPARATELY then added the two
        # residues without re-reducing the sum (can legitimately exceed the
        # modulus, so is often -- not always -- a genuinely different value;
        # `pick_distractors` below drops it whenever it happens to coincide
        # with the answer); reduced only ONE of a/b, leaving the other
        # unreduced, then added; forgot to add at all and reduced only `a`
        # (dropped `b` entirely).
        reduced_separately = (a % modulus) + (b % modulus)
        one_unreduced = (a % modulus) + b
        forgot_b = a % modulus
        alternatives = [
            str(reduced_separately),
            str(one_unreduced),
            str(forgot_b),
            str(total),
            str(answer + modulus),
            str((answer + 1) % modulus),
            str((answer - 1) % modulus),
        ]
        return dict(
            text=(f"Find the least non-negative residue r such that ({a} + {paren_negative(b)}) ≡ r (mod {modulus})."),
            latex=rf"({a} + {paren_negative(b)}) \equiv r \pmod{{{modulus}}},\quad 0 \le r < {modulus}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"{a} + {paren_negative(b)} = {total}.",
                f"{total} = {total // modulus}·{modulus} + {answer}, so r = {answer}.",
            ],
            distractors=pick_distractors(answer_text, alternatives),
        )
    # Ensure the question actually exercises reduction, including negative
    # representatives, rather than mostly returning the input unchanged.
    value = draw_until(
        rng,
        lambda r: r.randint(-magnitude_hi, magnitude_hi),
        lambda candidate: abs(candidate) >= modulus,
        tries=100 * (2 * magnitude_hi + 1),
        what="an integer that reaches the modulus",
    )
    answer = value % modulus
    independent = value - modulus * math.floor(value / modulus)
    enumerated = next(r for r in range(modulus) if (value - r) % modulus == 0)
    check(answer == independent == enumerated, "modular residue mismatch")

    answer_text = str(answer)
    alternatives = [
        str(value),
        str(abs(value)),
        str(value % max(2, modulus - 1)),
        str(answer + modulus),
        str((answer + 1) % modulus),
        str((answer - 1) % modulus),
    ]
    return dict(
        text=(f"Find the least non-negative residue r such that {value} ≡ r (mod {modulus})."),
        latex=rf"{value} \equiv r \pmod{{{modulus}}},\quad 0 \le r < {modulus}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{value} = {value // modulus}·{modulus} + {answer}, so r = {answer}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


# ---------------------------------------------------------------------------
# University-core Batch 2


def _pair_text(x: int, y: int) -> str:
    return f"({x}, {y})"


@generator(
    "vector_projection",
    level=12,
    presets={
        "easy": dict(component_hi=3, scale_hi=2, orthogonal_hi=1),
        "medium": dict(component_hi=5, scale_hi=3, orthogonal_hi=2),
        "hard": dict(component_hi=8, scale_hi=4, orthogonal_hi=4),
    },
)
def vector_projection(rng, component_hi=5, scale_hi=3, orthogonal_hi=2, kind=None):
    """Find the vector projection of u onto v, with an exact integer answer.

    ``kind="line_through_points"`` names the line by two points A and B, so the
    direction v = B - A has to be formed first."""
    if component_hi < 1:
        raise ValueError("vector_projection needs component_hi >= 1: the direction vector v must not be (0, 0)")
    vx, vy = draw_until(
        rng,
        lambda r: (r.randint(-component_hi, component_hi), r.randint(-component_hi, component_hi)),
        lambda v: v != (0, 0),
        what="a nonzero direction vector",
    )
    scale = rng.choice([x for x in range(-scale_hi, scale_hi + 1) if x])
    orthogonal_scale = rng.randint(-orthogonal_hi, orthogonal_hi)
    ux = scale * vx - orthogonal_scale * vy
    uy = scale * vy + orthogonal_scale * vx
    norm_sq = vx * vx + vy * vy
    dot = ux * vx + uy * vy
    projected = (Fraction(dot, norm_sq) * vx, Fraction(dot, norm_sq) * vy)
    expected = (scale * vx, scale * vy)
    # The dot-product formula and the constructed orthogonal decomposition
    # independently agree; the residual also certifies perpendicularity.
    check(projected == expected, "vector projection formula mismatch")
    residual = (Fraction(ux) - projected[0], Fraction(uy) - projected[1])
    check(residual[0] * vx + residual[1] * vy == 0, "projection residual is not orthogonal")

    answer = {"x": expected[0], "y": expected[1]}
    answer_text = _pair_text(*expected)
    if kind == "line_through_points":
        return _vector_projection_line(rng, component_hi, (ux, uy), (vx, vy), expected)
    alternatives = [
        _pair_text(ux, uy),
        _pair_text(expected[1], expected[0]),
        _pair_text(-expected[0], expected[1]),
        _pair_text(expected[0], -expected[1]),
        _pair_text(orthogonal_scale * vx, orthogonal_scale * vy),
    ]
    return dict(
        text=f"Project u = {_pair_text(ux, uy)} onto v = {_pair_text(vx, vy)}.",
        latex=rf"\operatorname{{proj}}_{{v}}u,\quad u={_pair_text(ux, uy)},\;v={_pair_text(vx, vy)}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"projᵥu = ((u·v)/(v·v))v = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


def _project_pair_text(u, w) -> str | None:
    """Text of the projection of u onto the vector w, or None when w is zero."""
    norm_sq = w[0] * w[0] + w[1] * w[1]
    if norm_sq == 0:
        return None
    k = Fraction(u[0] * w[0] + u[1] * w[1], norm_sq)
    return f"({rational_text(k * w[0])}, {rational_text(k * w[1])})"


def _vector_projection_line(rng, component_hi, u, v, expected):
    """Project u onto the line through A and B = A + v (v is not stated)."""
    ax = rng.randint(-component_hi, component_hi)
    ay = rng.randint(-component_hi, component_hi)
    bx, by = ax + v[0], ay + v[1]
    # Second route: rebuild the direction from the stated points and project.
    dx, dy = bx - ax, by - ay
    k = Fraction(u[0] * dx + u[1] * dy, dx * dx + dy * dy)
    check((k * dx, k * dy) == (Fraction(expected[0]), Fraction(expected[1])), "line projection mismatch")
    answer_text = _pair_text(*expected)
    candidates = [
        _project_pair_text(u, (bx, by)),
        _project_pair_text(u, (ax, ay)),
        _pair_text(*u),
        _pair_text(expected[1], expected[0]),
        _pair_text(-expected[0], expected[1]),
        _pair_text(expected[0], -expected[1]),
    ]
    alternatives = [c for c in candidates if c is not None]
    a_text, b_text = _pair_text(ax, ay), _pair_text(bx, by)
    return dict(
        text=f"Project u = {_pair_text(*u)} onto the line through A = {a_text} and B = {b_text}.",
        latex=(rf"\operatorname{{proj}}_{{AB}}u,\quad u={_pair_text(*u)},\;A={a_text},\;B={b_text}"),
        answer={"x": expected[0], "y": expected[1]},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"v = B - A = ({dx}, {dy}).",
            f"projᵥu = ((u·v)/(v·v))v = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives, tolerance=GRADER_TOLERANCE),
    )


def _poly_text(trace: int, determinant: int) -> str:
    if trace == 0:
        linear = ""
    elif trace > 0:
        linear = f" - {trace}λ"
    else:
        linear = f" + {-trace}λ"
    if determinant == 0:
        constant = ""
    elif determinant > 0:
        constant = f" + {determinant}"
    else:
        constant = f" - {-determinant}"
    return f"λ²{linear}{constant}"


@generator(
    "two_by_two_eigenvalues",
    level=13,
    presets={
        "easy": dict(eigenvalue_hi=4, off_diagonal_hi=3),
        "medium": dict(eigenvalue_hi=8, off_diagonal_hi=6),
        "hard": dict(eigenvalue_hi=14, off_diagonal_hi=10),
    },
)
def two_by_two_eigenvalues(rng, eigenvalue_hi=8, off_diagonal_hi=6):
    """Find eigenvalues of an upper-triangular 2x2 matrix."""
    if eigenvalue_hi < 1:
        raise ValueError("two_by_two_eigenvalues needs eigenvalue_hi >= 1: the two eigenvalues must be different")
    first = rng.randint(-eigenvalue_hi, eigenvalue_hi)
    second = draw_until(
        rng,
        lambda r: r.randint(-eigenvalue_hi, eigenvalue_hi),
        lambda candidate: candidate != first,
        what="a second eigenvalue different from the first",
    )
    off_diagonal = rng.randint(-off_diagonal_hi, off_diagonal_hi)
    matrix = sympy.Matrix([[first, off_diagonal], [0, second]])
    lam = sympy.Symbol("λ")
    characteristic = sympy.expand(matrix.charpoly(lam).as_expr())
    expected_polynomial = sympy.expand((lam - first) * (lam - second))
    roots = sorted([first, second])
    check(characteristic == expected_polynomial, "characteristic polynomial mismatch")
    check(all(characteristic.subs(lam, root) == 0 for root in roots), "eigenvalue substitution failed")
    check(
        sum(roots) == int(matrix.trace()) and roots[0] * roots[1] == int(matrix.det()),
        "trace/determinant eigenvalue invariants failed",
    )

    answer_text = ", ".join(map(str, roots))
    alternatives = [
        str(first + second),
        ", ".join(map(str, [first, -second])),
        ", ".join(map(str, [first + off_diagonal, second])),
        ", ".join(map(str, [first, second + off_diagonal])),
    ]
    return dict(
        text=f"For A = [[{first}, {off_diagonal}], [0, {second}]], use det(λI − A) = 0 to find the eigenvalues.",
        latex=rf"A=\begin{{pmatrix}}{first}&{off_diagonal}\\0&{second}\end{{pmatrix}},\quad \det(\lambda I-A)=0",
        answer=roots,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"χ_A(λ) = {_poly_text(first + second, first * second)} = 0; eigenvalues are {answer_text}."],
        # unordered: eigenvalues are a SET and grade_list accepts them in any
        # order, so a transposed pair was the right answer, not a wrong one.
        distractors=pick_distractors(answer_text, alternatives, unordered=True),
    )


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


@generator(
    "discrete_expected_value",
    level=12,
    presets={
        "easy": dict(outcome_hi=5, weight_hi=3, outcomes=3),
        "medium": dict(outcome_hi=9, weight_hi=5, outcomes=4),
        "hard": dict(outcome_hi=14, weight_hi=8, outcomes=5),
    },
)
def discrete_expected_value(rng, outcome_hi=9, weight_hi=5, outcomes=4):
    """Compute the exact expectation of a finite discrete distribution."""
    values = sorted(rng.sample(range(-outcome_hi, outcome_hi + 1), outcomes))
    weights = [rng.randint(1, weight_hi) for _ in values]
    total = sum(weights)
    weighted = sum(value * weight for value, weight in zip(values, weights))
    expected = Fraction(weighted, total)
    expanded = [value for value, weight in zip(values, weights) for _ in range(weight)]
    enumerated = Fraction(sum(expanded), len(expanded))
    check(expected == enumerated, "discrete expectation enumeration mismatch")
    answer, answer_text = _fraction_payload(expected)

    unweighted = Fraction(sum(values), len(values))
    candidates = [unweighted, Fraction(min(values)), Fraction(max(values)), expected + 1]
    distractors = [_fraction_payload(Fraction(candidate))[1] for candidate in candidates]
    probabilities = ", ".join(f"P(X={value})={weight}/{total}" for value, weight in zip(values, weights))
    return dict(
        text=f"A discrete random variable X has {probabilities}. Find E[X].",
        latex=rf"E[X],\quad "
        + r",\; ".join(rf"P(X={value})=\frac{{{weight}}}{{{total}}}" for value, weight in zip(values, weights)),
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"E[X] = ({weighted})/{total} = {answer_text}."],
        distractors=pick_distractors(answer_text, distractors),
    )


@generator(
    "conditional_probability_counts",
    level=12,
    presets={
        "easy": dict(cell_hi=6),
        "medium": dict(cell_hi=12),
        "hard": dict(cell_hi=25),
    },
)
def conditional_probability_counts(rng, cell_hi=12, kind=None):
    """Compute P(A | B) from four finite contingency-table counts.

    ``kind=None`` (the default) preserves the original four-cell behaviour
    byte-for-byte. ``kind="totals"`` gives the row/column TOTALS (|A|, |B|,
    the population size n, and the "neither" count) instead of the four raw
    cells -- so the "both" cell (P(A|B)'s numerator) must be derived first
    via inclusion-exclusion (|A∩B| = |A| + |B| − n + |Aᶜ∩Bᶜ|) rather than
    read off directly. The question asked stays P(A|B) in both cases.
    """
    ab = rng.randint(1, cell_hi)
    a_not_b = rng.randint(1, cell_hi)
    not_a_b = rng.randint(1, cell_hi)
    not_a_not_b = rng.randint(1, cell_hi)
    b_total = ab + not_a_b
    a_total = ab + a_not_b
    total = b_total + a_not_b + not_a_not_b
    expected = Fraction(ab, b_total)
    population = (
        ["A and B"] * ab + ["A and not B"] * a_not_b + ["not A and B"] * not_a_b + ["not A and not B"] * not_a_not_b
    )
    b_entries = [label for label in population if label in ("A and B", "not A and B")]
    enumerated = Fraction(sum(label == "A and B" for label in b_entries), len(b_entries))
    check(expected == enumerated, "conditional probability enumeration mismatch")
    answer, answer_text = _fraction_payload(expected)
    if kind == "totals":
        derived_ab = a_total + b_total - total + not_a_not_b
        check(derived_ab == ab, "inclusion-exclusion cell derivation mismatch")
        candidates = [
            Fraction(a_total, b_total),
            Fraction(derived_ab, a_total),
            Fraction(derived_ab, total),
            Fraction(b_total, total),
        ]
        distractors = [_fraction_payload(Fraction(candidate))[1] for candidate in candidates]
        return dict(
            text=(
                f"Among a population of {total}, {a_total} have A, {b_total} have B, "
                f"and the number with neither is {not_a_not_b}. Find P(A | B)."
            ),
            latex=(
                rf"P(A\mid B)=\frac{{|A\cap B|}}{{|B|}},\quad "
                rf"|A|={a_total},\,|B|={b_total},\,n={total},\,|A^c\cap B^c|={not_a_not_b}"
            ),
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"|A∩B| = |A| + |B| − n + |Aᶜ∩Bᶜ| = {a_total} + {b_total} − {total} + {not_a_not_b} = {derived_ab}.",
                f"P(A|B) = {derived_ab}/{b_total} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractors),
        )
    candidates = [
        Fraction(ab, a_total),
        Fraction(b_total, total),
        Fraction(ab, total),
        Fraction(not_a_b, b_total),
    ]
    distractors = [_fraction_payload(Fraction(candidate))[1] for candidate in candidates]
    return dict(
        text=(
            f"A table records {ab} with both A and B, {a_not_b} with A only, "
            f"{not_a_b} with B only, and {not_a_not_b} with neither. Find P(A | B)."
        ),
        latex=rf"P(A\mid B)=\frac{{|A\cap B|}}{{|B|}}=\frac{{{ab}}}{{{b_total}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Among the {b_total} cases with B, {ab} also have A: P(A|B) = {answer_text}."],
        distractors=pick_distractors(answer_text, distractors),
    )


@generator(
    "euclidean_algorithm_gcd",
    level=10,
    presets={
        "easy": dict(number_hi=40),
        "medium": dict(number_hi=120),
        "hard": dict(number_hi=500),
    },
)
def euclidean_algorithm_gcd(rng, number_hi=120, count: int = 2):
    """Find the gcd of two (or three) positive integers using Euclidean reduction.

    ``count=2`` (the default) preserves the original two-number behaviour
    byte-for-byte. ``count=3`` plants a shared factor ``g`` (2-6) into the
    first two draws so their pair gcd is never forced to 1 by chance, then
    draws an independent third positive integer (unbiased -- drawn the same
    way the two-number tier draws its numbers, with no factor planted) and
    computes ``gcd(gcd(first, second), third)``, requiring the Euclidean
    algorithm applied twice -- a genuine extra step, not a bigger pair.
    """
    first = rng.randint(12, number_hi)
    second = rng.randint(6, number_hi)
    while first == second:
        second = rng.randint(6, number_hi)
    a, b = first, second
    while b:
        a, b = b, a % b
    pair_gcd = a
    if count == 3:
        g = rng.randint(2, 6)
        first = g * rng.randint(2, number_hi // g)
        second = g * rng.randint(2, number_hi // g)
        while second == first:
            second = g * rng.randint(2, number_hi // g)
        a, b = first, second
        while b:
            a, b = b, a % b
        pair_gcd = a
        third = rng.randint(6, number_hi)
        while third == first or third == second:
            third = rng.randint(6, number_hi)
        a, b = pair_gcd, third
        while b:
            a, b = b, a % b
        answer = a
        sympy_answer = int(sympy.gcd(sympy.gcd(first, second), third))
        enumerated = max(
            d for d in range(1, min(first, second, third) + 1) if first % d == 0 and second % d == 0 and third % d == 0
        )
        check(answer == sympy_answer == enumerated, "Euclidean gcd verification mismatch")
        answer_text = str(answer)
        first_two_only = pair_gcd
        last_two = second
        b_last = third
        while b_last:
            last_two, b_last = b_last, last_two % b_last
        candidates = [
            first_two_only,
            last_two,
            first,
            second,
            third,
            first + second + third,
            max(1, answer // 2),
            answer * 2,
        ]
        return dict(
            text=f"Use the Euclidean algorithm to find gcd({first}, {second}, {third}).",
            latex=rf"\gcd({first},{second},{third})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"gcd({first}, {second}) = {pair_gcd} by repeated remainders.",
                f"gcd({pair_gcd}, {third}) = {answer} by repeated remainders, "
                f"so gcd({first}, {second}, {third}) = {answer}.",
            ],
            distractors=pick_distractors(answer_text, [str(candidate) for candidate in candidates]),
        )
    answer = pair_gcd
    sympy_answer = int(sympy.gcd(first, second))
    enumerated = max(d for d in range(1, min(first, second) + 1) if first % d == 0 and second % d == 0)
    check(answer == sympy_answer == enumerated, "Euclidean gcd verification mismatch")
    answer_text = str(answer)
    candidates = [first, second, first + second, max(1, answer // 2), answer * 2]
    return dict(
        text=f"Use the Euclidean algorithm to find gcd({first}, {second}).",
        latex=rf"\gcd({first},{second})",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Repeated remainders end at {answer}, so gcd({first}, {second}) = {answer}."],
        distractors=pick_distractors(answer_text, [str(candidate) for candidate in candidates]),
    )


# ---------------------------------------------------------------------------
# University-core Batch 3


def _manual_matrix_rank(rows: list[list[int]]) -> int:
    """Small exact Gaussian elimination used as a second rank calculation."""
    matrix = [[Fraction(value) for value in row] for row in rows]
    n_rows, n_cols = len(matrix), len(matrix[0])
    pivot_row = 0
    for column in range(n_cols):
        pivot = next((row for row in range(pivot_row, n_rows) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [entry / scale for entry in matrix[pivot_row]]
        for row in range(n_rows):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [left - factor * right for left, right in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def _matrix_text(rows: list[list[int]]) -> str:
    return "[" + ", ".join("[" + ", ".join(map(str, row)) + "]" for row in rows) + "]"


@generator(
    "matrix_rank_small",
    level=13,
    presets={
        "easy": dict(rows=2, cols=2, entry_hi=3),
        "medium": dict(rows=2, cols=3, entry_hi=4),
        "hard": dict(rows=3, cols=3, entry_hi=5),
    },
)
def matrix_rank_small(rng, rows=2, cols=2, entry_hi=3, target_rank=None):
    """Find the rank of a small integer matrix.

    The matrix is deliberately factored as ``A @ B`` with a visible identity
    block, so ``target_rank`` is guaranteed rather than left to chance.  The
    returned answer is still checked independently by exact elimination and
    SymPy's rank implementation.
    """
    max_rank = min(rows, cols)
    if target_rank is None:
        target_rank = rng.randint(1, max_rank)
    if not 1 <= target_rank <= max_rank:
        raise ValueError(f"target_rank must be between 1 and {max_rank}")

    # A has an identity block in its first target_rank rows and B has one in
    # its first target_rank columns.  Therefore A and B each have rank
    # target_rank, and their product has exactly that rank.  Random entries in
    # the remaining rows/columns provide varied integer examples without
    # risking accidental rank collapse.
    left = [[0 for _ in range(target_rank)] for _ in range(rows)]
    right = [[0 for _ in range(cols)] for _ in range(target_rank)]
    for index in range(target_rank):
        left[index][index] = 1
        right[index][index] = 1
    for row in range(target_rank, rows):
        left[row] = [rng.randint(-entry_hi, entry_hi) for _ in range(target_rank)]
    for column in range(target_rank, cols):
        for row in range(target_rank):
            right[row][column] = rng.randint(-entry_hi, entry_hi)
    data = [
        [sum(left[row][inner] * right[inner][column] for inner in range(target_rank)) for column in range(cols)]
        for row in range(rows)
    ]
    matrix = sympy.Matrix(data)
    expected = int(matrix.rank())
    manual = _manual_matrix_rank(data)
    check(expected == manual, "matrix rank elimination mismatch")
    check(expected == target_rank, "constructed matrix did not realise target rank")
    check(expected <= min(rows, cols), "matrix rank exceeded its dimension bound")

    answer_text = str(expected)
    alternatives = [expected - 1, expected + 1, min(rows, cols), rows, cols]
    return dict(
        text=f"Find the rank of the integer matrix {_matrix_text(data)}.",
        latex=rf"\operatorname{{rank}}\left({_matrix_text(data)}\right)",
        answer=expected,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Exact row reduction leaves {expected} pivot row(s), so the rank is {expected}."],
        distractors=pick_distractors(answer_text, [str(value) for value in alternatives]),
    )


def _poly_display(expr) -> str:
    return str(expr).replace("**", "^").replace("*", "")


def _fraction_float_text(value: Fraction) -> str:
    return f"{float(value):.6g}"


@generator(
    "directional_derivative",
    level=18,
    presets={
        "easy": dict(coefficient_hi=3, point_hi=2),
        "medium": dict(coefficient_hi=5, point_hi=4),
        "hard": dict(coefficient_hi=8, point_hi=6),
    },
)
def directional_derivative(rng, coefficient_hi=5, point_hi=4):
    """Compute a scalar directional derivative of a polynomial in two variables."""
    x, y, h = sympy.symbols("x y h")
    a = rng.randint(1, coefficient_hi)
    b = rng.randint(-coefficient_hi, coefficient_hi)
    c = rng.randint(-coefficient_hi, coefficient_hi)
    d = rng.randint(-coefficient_hi, coefficient_hi)
    if b == c == d == 0:
        d = 1
    function = a * x**2 + b * x * y + c * y**2 + d * x
    x0, y0 = rng.randint(-point_hi, point_hi), rng.randint(-point_hi, point_hi)
    ux, uy = rng.choice(
        (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            (Fraction(3, 5), Fraction(4, 5)),
            (Fraction(-3, 5), Fraction(4, 5)),
            (Fraction(5, 13), Fraction(12, 13)),
        )
    )
    gradient = (sympy.diff(function, x).subs({x: x0, y: y0}), sympy.diff(function, y).subs({x: x0, y: y0}))
    expected = Fraction(int(gradient[0]), 1) * ux + Fraction(int(gradient[1]), 1) * uy
    path = sympy.expand(
        function.subs(
            {
                x: x0 + h * sympy.Rational(ux.numerator, ux.denominator),
                y: y0 + h * sympy.Rational(uy.numerator, uy.denominator),
            }
        )
    )
    alternate = sympy.diff(path, h).subs(h, 0)
    check(
        sympy.Rational(expected.numerator, expected.denominator) == alternate,
        "directional derivative substitution mismatch",
    )
    check(ux * ux + uy * uy == 1, "direction vector is not unit length")

    answer = float(expected)
    answer_text = _fraction_float_text(expected)
    candidates = [
        float(gradient[0]),
        float(gradient[1]),
        float(-expected),
        float(expected + 1),
        float(expected - 1),
    ]
    # Every option at the answer's own number of decimal places, so the key
    # cannot be spotted by its shape.
    shown_places = len(answer_text.partition(".")[2])
    _, shown_candidates = _aligned(Fraction(answer_text), [Fraction(value) for value in candidates], shown_places)
    rounding = " Give your answer correct to 6 significant figures." if Fraction(answer_text) != expected else ""
    return dict(
        text=(
            f"For f(x,y) = {_poly_display(function)}, find the directional derivative "
            f"at ({x0}, {y0}) in the unit direction ({ux}, {uy}).{rounding}"
        ),
        latex=rf"D_{{({ux},{uy})}}f({x0},{y0}),\quad f={sympy.latex(function)}"
        + (r",\quad \text{Give your answer correct to 6 significant figures}" if rounding else ""),
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"∇f({x0}, {y0}) · u = {answer_text}."],
        distractors=pick_distractors(answer_text, shown_candidates),
    )


@generator(
    "jacobian_two_variable",
    level=18,
    presets={
        "easy": dict(coefficient_hi=3, point_hi=2),
        "medium": dict(coefficient_hi=5, point_hi=4),
        "hard": dict(coefficient_hi=8, point_hi=6),
    },
)
def jacobian_two_variable(rng, coefficient_hi=5, point_hi=4):
    """Evaluate the Jacobian of a polynomial map R² -> R² at a point."""
    x, y, h = sympy.symbols("x y h")
    coefficients = [rng.randint(-coefficient_hi, coefficient_hi) for _ in range(8)]
    if all(value == 0 for value in coefficients):
        coefficients[0] = 1
    a, b, c, d, e, f, g, k = coefficients
    first = a * x**2 + b * x * y + c * y + d
    second = e * x + f * y**2 + g * x * y + k
    x0, y0 = rng.randint(-point_hi, point_hi), rng.randint(-point_hi, point_hi)
    functions = (first, second)
    jacobian = [[sympy.diff(expr, variable).subs({x: x0, y: y0}) for variable in (x, y)] for expr in functions]
    substitutions = []
    for expr in functions:
        dx = sympy.expand(expr.subs({x: x0 + h, y: y0}))
        dy = sympy.expand(expr.subs({x: x0, y: y0 + h}))
        substitutions.append([sympy.diff(dx, h).subs(h, 0), sympy.diff(dy, h).subs(h, 0)])
    check(jacobian == substitutions, "Jacobian substitution mismatch")
    check(len(jacobian) == 2 and all(len(row) == 2 for row in jacobian), "Jacobian shape mismatch")
    answer = [[int(value) for value in row] for row in jacobian]
    answer_text = _matrix_text(answer)
    transpose = [[answer[0][0], answer[1][0]], [answer[0][1], answer[1][1]]]
    swapped_rows = [answer[1], answer[0]]
    origin = [[int(sympy.diff(expr, variable).subs({x: 0, y: 0})) for variable in (x, y)] for expr in functions]
    perturbed = []
    for row_index, column_index in ((0, 0), (0, 1), (1, 0), (1, 1)):
        candidate = [row[:] for row in answer]
        candidate[row_index][column_index] += 1
        perturbed.append(candidate)
    alternatives = [_matrix_text(candidate) for candidate in (transpose, swapped_rows, origin, *perturbed)]
    return dict(
        text=(
            f"For F(x,y) = ({_poly_display(first)}, {_poly_display(second)}), evaluate the Jacobian at ({x0}, {y0})."
        ),
        latex=rf"J_F({x0},{y0}),\quad F=({sympy.latex(first)},{sympy.latex(second)})",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Differentiate each component with respect to x and y, then substitute ({x0}, {y0})."],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "newton_method_iteration",
    level=16,
    presets={
        "easy": dict(coefficient_hi=3, point_hi=4),
        "medium": dict(coefficient_hi=6, point_hi=6),
        "hard": dict(coefficient_hi=10, point_hi=10),
    },
)
def newton_method_iteration(rng, coefficient_hi=6, point_hi=6, kind=None):
    """Perform one exact Newton iteration.

    ``kind=None`` (the default) preserves the original monic-quadratic
    behaviour byte-for-byte: ``f(x) = x**2 + coefficient*x + constant``.
    ``kind="cubic"`` uses a monic cubic ``f(x) = x**3 + a*x**2 + b*x + c``
    instead, with BOTH ``a`` and ``b`` forced nonzero so the rendered prompt
    always shows all three non-leading terms (a zero ``a`` would let sympy's
    string simplification silently drop the x**2 term and collapse the
    prompt back to the quadratic's own 3-term shape). This is a genuine
    escalation in kind: f'(x) = 3x^2 + 2ax + b is a three-term derivative
    the student must compute, not the quadratic case's one-step
    f'(x) = 2x + coefficient.
    """
    x = sympy.Symbol("x")
    if kind == "cubic":
        a = rng.randint(-coefficient_hi, coefficient_hi) or 1
        b = rng.randint(-coefficient_hi, coefficient_hi) or 1
        c = rng.randint(-coefficient_hi, coefficient_hi)
        x0 = rng.randint(-point_hi, point_hi) or 1
        function = x**3 + a * x**2 + b * x + c
    else:
        coefficient = rng.randint(-coefficient_hi, coefficient_hi) or 1
        constant = rng.randint(-coefficient_hi, coefficient_hi)
        x0 = rng.randint(-point_hi, point_hi) or 1
        function = x**2 + coefficient * x + constant
    derivative = sympy.diff(function, x)
    derivative_at = derivative.subs(x, x0)
    while derivative_at == 0:
        x0 = rng.randint(-point_hi, point_hi) or 1
        derivative_at = derivative.subs(x, x0)
    function_at = function.subs(x, x0)
    next_value = sympy.Rational(x0) - function_at / derivative_at
    answer = Fraction(int(next_value.p), int(next_value.q))
    check(sympy.expand(function_at + derivative_at * (next_value - x0)) == 0, "Newton linearisation did not vanish")
    check(next_value == x0 - function.subs(x, x0) / derivative.subs(x, x0), "Newton iteration formula mismatch")
    answer_payload, answer_text = _fraction_payload(answer)
    candidates = [
        Fraction(x0),
        Fraction(int(x0) - int(function_at), 1),
        Fraction(int(x0) + int(function_at), 1),
        Fraction(int(derivative_at), 1),
        answer + 1,
        answer - 1,
        answer + 2,
    ]
    return dict(
        text=f"For f(x) = {_poly_display(function)}, take one Newton step from x₀ = {x0}. Find x₁.",
        latex=rf"x_1=x_0-\frac{{f(x_0)}}{{f'(x_0)}},\quad f(x)={sympy.latex(function)},\;x_0={x0}",
        answer=answer_payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"x₁ = {x0} − f({x0})/f′({x0}) = {answer_text}."],
        distractors=pick_distractors(answer_text, [_fraction_payload(candidate)[1] for candidate in candidates]),
    )


@generator(
    "composite_trapezoidal_polynomial",
    level=16,
    presets={
        "easy": dict(coefficient_hi=3, interval_hi=4, n_hi=3),
        "medium": dict(coefficient_hi=6, interval_hi=7, n_hi=5),
        "hard": dict(coefficient_hi=10, interval_hi=10, n_hi=8),
    },
)
def composite_trapezoidal_polynomial(rng, coefficient_hi=6, interval_hi=7, n_hi=5):
    """Approximate a polynomial integral with the composite trapezoidal rule."""
    coefficient = rng.randint(1, coefficient_hi)
    linear = rng.randint(-coefficient_hi, coefficient_hi)
    constant = rng.randint(-coefficient_hi, coefficient_hi)
    left = rng.randint(0, max(0, interval_hi - 2))
    width = rng.randint(2, max(2, interval_hi - left))
    right = left + width
    n = rng.randint(2, n_hi)
    step = Fraction(width, n)

    def value(t: Fraction) -> Fraction:
        return coefficient * t * t + linear * t + constant

    points = [Fraction(left) + index * step for index in range(n + 1)]
    values = [value(point) for point in points]
    weighted = step * (values[0] / 2 + sum(values[1:-1], Fraction(0)) + values[-1] / 2)
    pairwise = sum((step * (values[index] + values[index + 1]) / 2 for index in range(n)), Fraction(0))
    check(weighted == pairwise, "trapezoidal pairwise summation mismatch")
    answer, answer_text = _fraction_payload(weighted)
    left_sum = step * sum(values[:-1], Fraction(0))
    right_sum = step * sum(values[1:], Fraction(0))
    exact_integral = (
        Fraction(coefficient, 3) * (Fraction(right) ** 3 - Fraction(left) ** 3)
        + Fraction(linear, 2) * (Fraction(right) ** 2 - Fraction(left) ** 2)
        + constant * width
    )
    candidates = [_fraction_payload(candidate)[1] for candidate in (left_sum, right_sum, exact_integral, weighted + 1)]
    x_symbol = sympy.Symbol("x")
    function_expr = coefficient * x_symbol**2 + linear * x_symbol + constant
    function_text = _poly_display(function_expr)
    return dict(
        text=f"Approximate ∫ from {left} to {right} of f(x) = {function_text} using {n} composite trapezoids.",
        latex=rf"T_{{{n}}}[f],\quad f(x)={sympy.latex(function_expr)},\;[{left},{right}]",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"h = {step}; applying the trapezoidal weights gives Tₙ = {answer_text}."],
        distractors=pick_distractors(answer_text, candidates),
    )
