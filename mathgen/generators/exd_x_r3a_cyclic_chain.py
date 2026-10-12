"""Exam source of `senior-the-opposite-angles-of-a-cyclic-quadrilateral-are`: two unknowns, then two uses of the theorem.

The practice tiers solve for one unknown. An exam question gives all four angles of a cyclic quadrilateral ABCD as
expressions in two unknowns x and y and builds the solution in stages:

(a) the value of x, from the two pairs of opposite angles (a 2 x 2 system);
(b) the value of y, from (a);
(c) the size of one named angle of the quadrilateral, from (a) and (b);
(d) an exterior angle, formed by extending one side, which needs the theorem again.

Parts (b) to (d) follow through from the earlier parts.

## Construction

x, y and the four angle sizes (each pair of opposite angles supplementary) are drawn first; the constant term of each
expression is then set to make the expressions equal those sizes. The system's determinant is checked to be non-zero.
The solution is checked by substitution (both pair-sums are 180) and is unique because the determinant is non-zero.
The four keys are all different, so no part's key can be mistaken for another.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, join_terms, latex_answer, pick_distractors
from ._exd_r3_distinct import distinct_part_answers

_NEXT = {"A": "B", "B": "C", "C": "D", "D": "A"}
_PREV = {v: k for k, v in _NEXT.items()}


def _expr(cx: int, cy: int, k: int) -> str:
    return join_terms([(cx, "x"), (cy, "y"), (k, "")])


def _draw(rng):
    for _ in attempts(3000, "a cyclic quadrilateral with distinct keys"):
        x, y = rng.randint(5, 30), rng.randint(5, 30)
        if x == y:
            continue
        a_val, b_val = rng.randint(50, 130), rng.randint(50, 130)
        vals = {"A": a_val, "C": 180 - a_val, "B": b_val, "D": 180 - b_val}
        coef = {n: (rng.randint(0, 4), rng.randint(0, 4)) for n in "ABCD"}
        if any(coef[n] == (0, 0) for n in "ABCD") or len({coef[n] for n in "ABCD"}) < 4:
            continue
        const = {n: vals[n] - coef[n][0] * x - coef[n][1] * y for n in "ABCD"}
        if any(abs(const[n]) > 70 for n in "ABCD"):
            continue
        det = (coef["A"][0] + coef["C"][0]) * (coef["B"][1] + coef["D"][1]) - (coef["A"][1] + coef["C"][1]) * (
            coef["B"][0] + coef["D"][0]
        )
        if det == 0:
            continue
        both = [
            coef["A"][0] + coef["C"][0] and coef["A"][1] + coef["C"][1],
            coef["B"][0] + coef["D"][0] and coef["B"][1] + coef["D"][1],
        ]
        if not any(both):
            continue  # y must depend on x in at least one of the two equations
        vertex = rng.choice("ABCD")
        exterior = 180 - vals[vertex]
        return x, y, vals, coef, const, vertex, exterior
    raise RuntimeError("no draw")


def _equation(coef, const, pair, rhs):
    """`sum of the angles in pair = rhs` as (coefficient of x, coefficient of y, right-hand side)."""
    n1, n2 = pair
    return (
        coef[n1][0] + coef[n2][0],
        coef[n1][1] + coef[n2][1],
        rhs - const[n1] - const[n2],
    )


def _solve_wrong(e1, e2):
    """The solution of two linear equations when it is a pair of positive whole numbers, else None."""
    (a1, b1, c1), (a2, b2, c2) = e1, e2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    x, y = Fraction(c1 * b2 - c2 * b1, det), Fraction(a1 * c2 - a2 * c1, det)
    if x.denominator != 1 or y.denominator != 1 or x <= 0 or y <= 0:
        return None
    return int(x), int(y)


@generator("exd_x_r3a_cyclic_chain", level=11, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_cyclic_chain(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a draw with enough distractors in every part"):
        x, y, vals, coef, const, vertex, exterior = _draw(rng)
        # Substituting the solution back must satisfy both supplementary equations and give the drawn angles; the
        # determinant checked in `_draw` makes it the only solution.
        angle = {n: coef[n][0] * x + coef[n][1] * y + const[n] for n in "ABCD"}
        check(angle == vals, "angle sizes disagree")
        check(
            angle["A"] + angle["C"] == 180 and angle["B"] + angle["D"] == 180, "opposite angles are not supplementary"
        )
        target = rng.choice([n for n in "ABCD" if n != vertex])
        top = vals[target]
        if len({x, y, top, exterior}) < 4:
            continue
        opposite = {"A": "C", "B": "D", "C": "A", "D": "B"}

        negated = {n: (-coef[n][0], -coef[n][1]) for n in "ABCD"}
        slips = {
            "supplement-rule-as-full-turn": (_equation(coef, const, "AC", 360), _equation(coef, const, "BD", 360)),
            "opposite-angles-taken-as-equal": (
                (coef["A"][0] - coef["C"][0], coef["A"][1] - coef["C"][1], const["C"] - const["A"]),
                (coef["B"][0] - coef["D"][0], coef["B"][1] - coef["D"][1], const["D"] - const["B"]),
            ),
            "adjacent-angles-taken-as-supplementary": (
                _equation(coef, const, "AB", 180),
                _equation(coef, const, "CD", 180),
            ),
        }
        del negated
        wrong = [(tag, _solve_wrong(*eqs)) for tag, eqs in slips.items()]
        wrong = [(tag, w) for tag, w in wrong if w is not None]
        x_c = [Distractor(str(w[0]), tag, w[0]) for tag, w in wrong] + [Distractor(str(y), "unknowns-swapped", y)]
        y_c = [Distractor(str(w[1]), tag, w[1]) for tag, w in wrong] + [Distractor(str(x), "unknowns-swapped", x)]
        a_d = pick_distractors(str(x), x_c, 3)
        b_d = pick_distractors(str(y), y_c, 3)
        c_d = pick_distractors(
            str(top),
            [
                Distractor(str(vals[opposite[target]]), "opposite-angle-reported", vals[opposite[target]]),
                Distractor(str(360 - top), "supplement-rule-as-full-turn", 360 - top),
                Distractor(str(vals[_NEXT[target]]), "adjacent-angle-reported", vals[_NEXT[target]]),
                Distractor(str(vals[_PREV[target]]), "adjacent-angle-reported", vals[_PREV[target]]),
            ],
            3,
        )
        v_val = vals[vertex]
        neighbour = vals[_NEXT[vertex]]
        d_d = pick_distractors(
            str(exterior),
            [
                Distractor(str(v_val), "interior-angle-for-exterior-angle", v_val),
                Distractor(str(360 - v_val), "supplement-rule-as-full-turn", 360 - v_val),
                Distractor(str(neighbour), "adjacent-angle-reported", neighbour),
                Distractor(str(180 - neighbour), "adjacent-angle-reported", 180 - neighbour),
            ],
            3,
        )
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break

    shown = ", ".join(f"angle {n} = ({_expr(*coef[n], const[n])}) degrees" for n in "ABC")
    stem = (
        f"ABCD is a cyclic quadrilateral with {shown} and angle D = ({_expr(*coef['D'], const['D'])}) degrees, "
        "where x and y are constants."
    )
    prev, nxt = _PREV[vertex], _NEXT[vertex]
    # y from the first pair-sum equation that has both unknowns
    eq_ac, eq_bd = _equation(coef, const, "AC", 180), _equation(coef, const, "BD", 180)
    cx, cy, rhs = eq_ac if (eq_ac[0] and eq_ac[1]) else eq_bd
    ft_b = f"({rhs} - {cx}*part_a)/{cy}"

    def angle_ft(n):
        return f"({coef[n][0]}*part_a + {coef[n][1]}*part_b + ({const[n]}))"

    ft_c = angle_ft(target)
    ft_d = f"180 - {angle_ft(vertex)}"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of x.",
                answer_kind="number",
                answer=x,
                answer_text=str(x),
                answer_latex=latex_answer(str(x)),
                marks=3,
                sensible_range=(0, None),
                working=[
                    "Opposite angles of a cyclic quadrilateral add to 180: A + C = 180 and B + D = 180.",
                    f"Solving the two equations gives x = {x}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Using your value of x, find the value of y.",
                answer_kind="number",
                answer=y,
                answer_text=str(y),
                answer_latex=latex_answer(str(y)),
                marks=1,
                sensible_range=(0, None),
                follow_through=ft_b,
                working=[f"Substituting x = {x} into an equation gives y = {y}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your values of x and y, find the size of angle {target}, in degrees.",
                answer_kind="number",
                answer=top,
                answer_text=str(top),
                answer_latex=latex_answer(str(top)),
                marks=2,
                sensible_range=(0, 360),
                follow_through=ft_c,
                working=[
                    f"Angle {target} = {coef[target][0]}x + {coef[target][1]}y + ({const[target]}) = {top}°.",
                ],
                distractors=c_d,
            ),
            dict(
                prompt=f"The side {prev}{vertex} is extended beyond {vertex} to a point E. Using your values of x and y, "
                f"find the size of angle {nxt}{vertex}E, in degrees.",
                answer_kind="number",
                answer=exterior,
                answer_text=str(exterior),
                answer_latex=latex_answer(str(exterior)),
                marks=2,
                sensible_range=(0, 180),
                follow_through=ft_d,
                working=[
                    f"Angle {prev}{vertex}{nxt} = {v_val}°, and angles on a straight line add to 180°.",
                    f"Angle {nxt}{vertex}E = 180° - {v_val}° = {exterior}°.",
                ],
                distractors=d_d,
            ),
        ],
    )
