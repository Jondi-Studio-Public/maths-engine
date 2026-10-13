"""Parallel-line angle theorems and angle-name ordering.

- ``p11d_parallel_line_angle_theorems`` (``f10-parallel-line-angle-theorems``,
  AC9M7M04, domain ``geometry``, answer: ``number``) -- given a stated angle
  and a NAMED relationship (corresponding, alternate or co-interior) formed
  by two parallel lines cut by a transversal, determine the unknown angle in
  degrees. Deliberately NEVER an angles-at-a-point or vertically-opposite
  relationship -- that is the prerequisite ``f10-angle-relationships``'s own
  graded operation (``mathgen/generators/p10e_angle_relationships_build.py``).

- ``p11d_compare_angle_names`` (``f10-estimate-compare-angle-names``,
  AC9M4M04, domain ``geometry``, answer: ``ordered_list``) -- order a stated
  set of DISTINCT angle-type names (acute, right, obtuse, straight, reflex,
  revolution) from smallest to largest by their fixed, conventional size.
  Never receives a degree value (that is ``angle-types``'s own graded
  operation) and never mixes in the turn-fraction vocabulary of
  ``f10-angle-turn-right-reference``.

With ``figure=True`` (what the skill's sources use) ``p11d_parallel_line_angle_theorems`` draws the lines
instead of describing them: two arrow-marked parallels cut by a transversal, the stated angle labelled in
degrees (or as an expression in x at the hard tier) and the asked angle marked ``x``. The picture is marked
"not drawn to scale" and drawn with the asked angle visibly different from its true size, so measuring does
not give the answer; the prompt still names the relationship. The figure form re-derives its answer from the
picture alone (``_read_parallel``), not from the text. The words-only form (``figure=False``, the default) is
unchanged.

Both generators re-derive their answer from the RENDERED question text
(a regex parse of the final ``text`` string, not the internal variables
construction used), independently of however construction built the
question -- the "verify the artefact the learner receives" discipline.
"""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Eq, Symbol, solve
from sympy import latex as sympy_latex

from .. import figures
from ..core import (
    VerificationError,
    attempts,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    reversed_and_adjacent_swap,
)
from ..figure_kinds import space
from . import _space_kit as kit

_X = Symbol("x")

# -- Skill 1: p11d_parallel_line_angle_theorems ---------------------------------

_EQUAL_TYPES = ("corresponding", "alternate")
_ALL_RELATIONSHIPS = ("corresponding", "alternate", "co-interior")

_PREP = {
    "corresponding": "corresponding to",
    "alternate": "alternate to",
    "co-interior": "co-interior with",
}

_RULE_STATEMENT = {
    "corresponding": "Corresponding angles are always equal.",
    "alternate": "Alternate angles are always equal.",
    "co-interior": "Co-interior angles always sum to 180 degrees.",
}

# Figure plausibility floor: every equal-type angle value and every
# co-interior pair lands in [20, 160] -- 180 - 160 = 20 and 180 - 20 = 160,
# so the co-interior complement lands in the same band automatically. 90 is
# excluded outright: it is the exact midpoint of [20, 160], and is the ONLY
# value at which the "wrong rule" misconception distractor (180 - X for an
# equal-type seed, or X for a co-interior seed) would collide with the
# correct answer. Excluding it removes the collision at its source rather
# than resampling or filtering after the fact.
_ANGLE_FLOOR = 20
_ANGLE_CEIL = 160
_MIDPOINT = 90


def _draw_easy_medium_angle(rng):
    while True:
        x = rng.randint(_ANGLE_FLOOR, _ANGLE_CEIL)
        if x != _MIDPOINT:
            return x


def _apply_rule(relationship: str, x):
    """The rule table -- looked up by NAME, never called during construction
    (construction computes the same value inline, a separate code path).
    Used only by ``_check_easy_medium`` below, re-deriving the answer from
    the relationship/angle actually PARSED out of the rendered text."""
    if relationship in _EQUAL_TYPES:
        return x
    if relationship == "co-interior":
        return 180 - x
    raise ValueError(f"unknown relationship {relationship!r}")


_REL_RE = re.compile(r"it is (corresponding|alternate|co-interior) (?:to|with) another angle")
_X_RE = re.compile(r"One angle measures (\d+) degrees")


def _check_easy_medium(text: str, answer_text: str):
    m_rel = _REL_RE.search(text)
    m_x = _X_RE.search(text)
    check(
        m_rel is not None and m_x is not None,
        "p11d_parallel_line_angle_theorems: could not parse relationship/angle from text",
    )
    relationship = m_rel.group(1)
    x = int(m_x.group(1))
    expected = _apply_rule(relationship, x)
    check(
        expected == int(answer_text), "p11d_parallel_line_angle_theorems: rule-table check disagrees with answer_text"
    )


# -- the figure form ------------------------------------------------------------

# (intersection, quadrant) positions as `space.parallel_transversal` numbers them: quadrants 0 and 2 are theta
# wide, 1 and 3 are 180 - theta wide. Each list is the position pairs that stand in that relationship.
_PAIRS = {
    "corresponding": [((0, k), (1, k)) for k in range(4)],
    "alternate": [((0, 2), (1, 0)), ((0, 3), (1, 1))],
    "co-interior": [((0, 3), (1, 0)), ((0, 2), (1, 1))],
}

_ALT_PARALLEL = "Two parallel lines, marked with arrows, cut by a transversal."


def _width(position, theta):
    """The true size of the angle at ``position`` when quadrants 0 and 2 are ``theta`` wide."""
    return theta if position[1] % 2 == 0 else 180 - theta


def _parallel_figure(rng, relationship, value, label_first, label_second, asked):
    """(figure, first_pos, second_pos) for two angles in ``relationship``, the first being ``value`` degrees.

    The pair of positions is drawn at random and either end can carry the first angle. The picture is drawn
    not to scale: theta (50-130) is of the same kind, acute or obtuse, as the true one, and the angle the
    question ``asked`` for (``"first"`` or ``"second"``) is drawn at least 10 degrees from its true size, so
    measuring the picture does not give the answer."""
    pair = rng.choice(_PAIRS[relationship])
    first_pos, second_pos = (pair[0], pair[1]) if rng.random() < 0.5 else (pair[1], pair[0])
    theta_true = value if first_pos[1] % 2 == 0 else 180 - value
    asked_pos = first_pos if asked == "first" else second_pos
    zone = (50.0, 84.0) if theta_true < 90 else (96.0, 130.0)
    for _ in attempts(200, "a drawn transversal angle"):
        theta = round(rng.uniform(*zone), 2)
        if abs(_width(asked_pos, theta) - _width(asked_pos, theta_true)) >= 10 and abs(theta - theta_true) >= 3:
            break
    fig = space.parallel_transversal(
        theta=theta,
        labels={first_pos: label_first, second_pos: label_second},
        rotation=rng.choice((-1, 1)) * rng.uniform(0, 20) + rng.choice((0, 90)),
        alt=_ALT_PARALLEL,
    )
    return fig, first_pos, second_pos


_CLOSERS_EASY_MEDIUM = (
    "Find the size of that other angle, in degrees.",
    "What is the size of that other angle, in degrees?",
    "Determine the size of that other angle, in degrees.",
)


def _easy_medium_question(rng, relationship_pool, figure=False):
    relationship = rng.choice(relationship_pool)
    fig = None
    if figure:
        for _ in attempts(80, "a transversal picture that does not show its own answer"):
            x = _draw_easy_medium_angle(rng)
            shown_answer = x if relationship in _EQUAL_TYPES else 180 - x
            fig, _first, _second = _parallel_figure(rng, relationship, x, "a", "x", asked="second")
            if not kit.leaks(fig, str(shown_answer)):
                break
    else:
        x = _draw_easy_medium_angle(rng)
    closer = rng.choice(_CLOSERS_EASY_MEDIUM)

    if relationship in _EQUAL_TYPES:
        answer_val = x
        d_wrong_rule = 180 - x  # misconception (a): applies the supplementary rule to an equal-type pair
    else:
        answer_val = 180 - x
        d_wrong_rule = x  # misconception (b): applies the equal rule to a co-interior pair
    d_offset = x + 7  # generic distractor, not a named misconception

    text = (
        f"Two parallel lines are cut by a transversal. One angle measures {x} degrees, "
        f"and it is {_PREP[relationship]} another angle formed by the same transversal. "
        f"{closer}"
    )
    if fig is not None:
        # The given size is stated here, not drawn: for corresponding and alternate angles the answer IS the
        # given size, so a drawn "65°" would show the answer.
        text = (
            f"The two lines marked with arrows are parallel, and angle a is {x}°. Angles a and x are "
            f"{relationship} angles. Find the size of angle x, in degrees."
        )
    latex = rf"\text{{{text}}}"
    answer_text = str(answer_val)

    if fig is None:
        _check_easy_medium(text, answer_text)
    else:
        check(
            str(_read_parallel(fig, given=x)) == answer_text,
            "p11d_parallel_line_angle_theorems: picture disagrees with answer",
        )

    check_pairwise_distinct(
        answer_val,
        d_wrong_rule,
        d_offset,
        msg="p11d_parallel_line_angle_theorems: easy/medium distractor collision",
    )
    distractors = pick_distractors(answer_text, [str(d_wrong_rule), str(d_offset)])

    working = [_RULE_STATEMENT[relationship], f"Answer: {answer_val} degrees."]

    return text, latex, answer_text, distractors, working, answer_val, fig


_HARD_REL_RE = re.compile(r"forming two (corresponding|alternate|co-interior) angles")
_HARD_EXPR_RE = re.compile(r"One angle measures \(([^)]+)\) degrees and the other measures \(([^)]+)\) degrees")
_LINEAR_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def _expr_text(coef: int, const: int) -> str:
    return join_terms([(coef, "x"), (const, "")])


def _parse_linear_local(text: str):
    """Independent re-parse of a rendered ``"6x - 110"``-style string --
    written separately from the oracle test file's own parser (a genuinely
    different implementation reading the same rendered text)."""
    m = _LINEAR_RE.match(text.strip())
    check(m is not None, f"p11d_parallel_line_angle_theorems: could not parse linear expression {text!r}")
    coef_str, sign, mag = m.group(1), m.group(2), m.group(3)
    if coef_str in ("", "+"):
        coef = 1
    elif coef_str == "-":
        coef = -1
    else:
        coef = int(coef_str)
    const = 0 if sign is None else (int(mag) if sign == "+" else -int(mag))
    return coef, const


def _check_hard(text: str, answer_text: str):
    m_rel = _HARD_REL_RE.search(text)
    m_expr = _HARD_EXPR_RE.search(text)
    check(
        m_rel is not None and m_expr is not None,
        "p11d_parallel_line_angle_theorems: could not parse relationship/expressions from text",
    )
    relationship = m_rel.group(1)
    a1, a2 = _parse_linear_local(m_expr.group(1))
    c1, c2 = _parse_linear_local(m_expr.group(2))
    e1 = a1 * _X + a2
    e2 = c1 * _X + c2
    eq = Eq(e1, e2) if relationship in _EQUAL_TYPES else Eq(e1 + e2, 180)
    sol = solve(eq, _X)
    check(len(sol) == 1, "p11d_parallel_line_angle_theorems: hard-tier equation did not have exactly one root")
    angle = e1.subs(_X, sol[0])
    check_equivalent(
        angle, int(answer_text), "p11d_parallel_line_angle_theorems: hard-tier rule check disagrees with answer_text"
    )


def _wrong_rule_angle_hard(relationship: str, a1: int, a2: int, c1: int, c2: int):
    """Misconceptions (a)/(b) at the hard tier: misapply the OTHER rule's equation to this seed's pair, then
    substitute back into e1."""
    e1 = a1 * _X + a2
    e2 = c1 * _X + c2
    eq = Eq(e1 + e2, 180) if relationship in _EQUAL_TYPES else Eq(e1, e2)
    sol = solve(eq, _X)
    if len(sol) != 1:
        return None
    x_w = sol[0]
    angle_w = e1.subs(_X, x_w)
    if not angle_w.is_Rational or angle_w.q != 1:
        return None
    return int(angle_w)


_CLOSERS_HARD = (
    "Find the value of the first angle, in degrees.",
    "What is the value of the first angle, in degrees?",
    "Determine the value of the first angle, in degrees.",
)


def _hard_question(rng, figure=False):
    found = False
    fig = None
    for _ in range(50):
        relationship = rng.choice(_ALL_RELATIONSHIPS)
        x = rng.randint(2, 30)
        target_angle = rng.randint(_ANGLE_FLOOR, _ANGLE_CEIL)
        a1 = rng.randint(2, 9)
        c1 = rng.randint(2, 9)
        if a1 == c1:
            continue
        if x == target_angle:
            # guards the reports-x-not-angle distractor's own degeneracy
            continue
        a2 = target_angle - a1 * x
        other_target = target_angle if relationship in _EQUAL_TYPES else 180 - target_angle
        c2 = other_target - c1 * x
        if figure:
            fig, _first, _second = _parallel_figure(
                rng, relationship, target_angle, f"({_expr_text(a1, a2)})°", f"({_expr_text(c1, c2)})°", asked="first"
            )
            if kit.leaks(fig, str(target_angle)):
                continue
        found = True
        break
    check(found, "p11d_parallel_line_angle_theorems: hard tier could not find a valid draw")
    closer = rng.choice(_CLOSERS_HARD)

    expr1 = _expr_text(a1, a2)
    expr2 = _expr_text(c1, c2)
    rule_statement = _RULE_STATEMENT[relationship]
    text = (
        f"Two parallel lines are cut by a transversal, forming two {relationship} angles. "
        f"One angle measures ({expr1}) degrees and the other measures ({expr2}) degrees. "
        f"{rule_statement} {closer}"
    )
    latex = (
        rf"\text{{Two parallel lines are cut by a transversal, forming two {relationship} "
        rf"angles. One angle measures }} "
        f"({sympy_latex(a1 * _X + a2)})"
        rf" \text{{ degrees and the other measures }} "
        f"({sympy_latex(c1 * _X + c2)})"
        rf" \text{{ degrees. {rule_statement} {closer}}}"
    )
    if fig is not None:
        text = (
            "The two lines marked with arrows are parallel. The two marked angles are "
            f"{relationship} angles. {rule_statement} Find the size of the angle marked ({expr1})°, in degrees."
        )
        latex = (
            rf"\text{{The two lines marked with arrows are parallel. The two marked angles are {relationship} "
            rf"angles. {rule_statement} Find the size of the angle marked }} ({sympy_latex(a1 * _X + a2)})^\circ"
            r"\text{, in degrees.}"
        )
    answer_text = str(target_angle)

    if fig is None:
        _check_hard(text, answer_text)
    else:
        asked = space_label_expr = f"({expr1})°"
        check(
            str(_read_parallel(fig, asked)) == answer_text,
            "p11d_parallel_line_angle_theorems: hard-tier picture disagrees with answer",
        )

    d_x = x  # reports-x-not-angle (misconception c)
    d_wrong_rule = _wrong_rule_angle_hard(relationship, a1, a2, c1, c2)  # misconceptions (a)/(b)
    d_offset = target_angle + 11  # generic distractor, not a named misconception

    raw_candidates = [d_x, d_offset] + ([d_wrong_rule] if d_wrong_rule is not None else [])
    # The supplement of the angle (used 180 where the relationship needed no subtraction, or the reverse) and a
    # slip of eleven back: further positive options, so dropping a negative candidate never leaves too few.
    raw_candidates += [180 - target_angle, target_angle - 11]
    # A negative angle is trivially eliminated, so only positive sizes are offered.
    raw_candidates = [c for c in raw_candidates if c > 0]
    # Defensive filter, not a resample: drop any candidate that already equals
    # the answer (the honestly-disclosed degeneracy the misconception can hit)
    # or an already-kept candidate (measured: 1/400 hard-tier seeds has
    # d_wrong_rule land on the same value as d_offset by coincidence) BEFORE
    # proving pairwise distinctness of what remains.
    survivors = []
    for c in raw_candidates:
        if c == target_angle or c in survivors:
            continue
        survivors.append(c)
    check_pairwise_distinct(
        target_angle, *survivors, msg="p11d_parallel_line_angle_theorems: hard-tier distractor collision"
    )
    distractors = pick_distractors(answer_text, [str(c) for c in survivors])

    working = [
        rule_statement,
        f"({expr1}) {'=' if relationship in _EQUAL_TYPES else '+'} ({expr2})"
        + ("" if relationship in _EQUAL_TYPES else " = 180"),
        f"x = {x}",
        f"angle = {target_angle} degrees.",
    ]

    return text, latex, answer_text, distractors, working, target_angle, fig


@generator(
    "p11d_parallel_line_angle_theorems",
    level=7,
    presets={
        "easy": dict(variant="equal_type"),
        "medium": dict(variant="all_types"),
        "hard": dict(variant="algebraic"),
    },
)
def p11d_parallel_line_angle_theorems(rng, variant="equal_type", figure=False):
    if variant == "equal_type":
        text, latex, answer_text, distractors, working, answer, fig = _easy_medium_question(rng, _EQUAL_TYPES, figure)
    elif variant == "all_types":
        text, latex, answer_text, distractors, working, answer, fig = _easy_medium_question(
            rng, _ALL_RELATIONSHIPS, figure
        )
    elif variant == "algebraic":
        text, latex, answer_text, distractors, working, answer, fig = _hard_question(rng, figure)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    out = {
        "text": text,
        "latex": latex,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }
    if fig is not None:
        out["figure"] = fig
    return out


def _read_parallel(fig: dict, asked: str | None = None, given: int | None = None) -> int:
    """The asked angle, re-derived from the picture alone.

    Two angles are marked at the two crossings of a transversal with arrow-marked parallels, so in the drawing
    they are either the same size (corresponding, alternate) or total 180 (co-interior); that, not the
    relationship's name, says which rule applies. ``asked`` is the label of the asked angle (hard tier);
    without it the asked angle is the one marked x and ``given`` is the size of the angle marked a (the
    prompt states it, because drawing it would show the answer).
    """
    arcs = kit.arcs_of(fig)
    check(len(arcs) == 2 and arcs[0]["at"] != arcs[1]["at"], "expected one marked angle at each crossing")
    s1, s2 = arcs[0]["sweep"], arcs[1]["sweep"]
    if abs(s1 - s2) < 0.5:
        equal = True
    elif abs(s1 + s2 - 180) < 0.5:
        equal = False
    else:
        raise VerificationError("the two marked angles are neither equal nor supplementary in the picture")
    if asked is None:
        check(sorted(a["label"] for a in arcs) == ["a", "x"] and given is not None, "expected angles a and x")
        return given if equal else 180 - given
    first = [a for a in arcs if a["label"] == asked]
    check(len(first) == 1, "the asked angle is not marked in the picture")
    second = arcs[1] if first[0] is arcs[0] else arcs[0]
    (a1, a2), (c1, c2) = kit.parse_linear_label(first[0]["label"]), kit.parse_linear_label(second["label"])
    # equal: a1 x + a2 = c1 x + c2;  supplementary: (a1 + c1) x + a2 + c2 = 180
    x = Fraction(c2 - a2, a1 - c1) if equal else Fraction(180 - a2 - c2, a1 + c1)
    angle = a1 * x + a2
    check(angle.denominator == 1, "the angle is not a whole number")
    return int(angle)


_ASKED_RE = re.compile(r"angle marked (\(.+?\)°)")
_GIVEN_RE = re.compile(r"angle a is (\d+)°")


def _read_parallel_question(question) -> str:
    asked, given = _ASKED_RE.search(question.text), _GIVEN_RE.search(question.text)
    return str(
        _read_parallel(question.figure, asked.group(1) if asked else None, int(given.group(1)) if given else None)
    )


figures.register_reader("p11d_parallel_line_angle_theorems", _read_parallel_question)


# -- Skill 2: p11d_compare_angle_names ------------------------------------------

# Fixed canonical order, smallest to largest, matching the everyday degree
# ranges: acute (0-90), right (=90), obtuse (90-180), straight (=180),
# reflex (180-360), revolution (=360).
_ORDER = ["acute", "right", "obtuse", "straight", "reflex", "revolution"]
_RANK = {name: i for i, name in enumerate(_ORDER)}

_ORDER_TEXT_RE = re.compile(r": (.+)\.$")


def _check_order(text: str, answer_text: str):
    """Independent check: recompute the expected order by looking up each
    NAME drawn from the rendered text in the fixed canonical-rank table and
    sorting by it -- a plain, direct check, not a re-sort of construction's
    own ``order_asc``."""
    m = _ORDER_TEXT_RE.search(text)
    check(m is not None, "p11d_compare_angle_names: could not parse drawn names from text")
    names = [n.strip() for n in m.group(1).split(",")]
    expected = sorted(names, key=lambda n: _RANK[n])
    check(
        ", ".join(expected) == answer_text, "p11d_compare_angle_names: canonical-rank check disagrees with answer_text"
    )


def _order_question(rng, n: int, hard_shuffle: bool):
    sample = rng.sample(_ORDER, n)
    order_asc = sorted(sample, key=lambda name: _RANK[name])
    positions = {name: i for i, name in enumerate(order_asc)}

    display = list(sample)
    if hard_shuffle:
        for _ in range(200):
            rng.shuffle(display)
            if all(abs(positions[display[i]] - positions[display[i + 1]]) != 1 for i in range(len(display) - 1)):
                break
        else:
            raise RuntimeError("p11d_compare_angle_names: hard tier could not find a valid derangement")
    else:
        # Reject-and-reshuffle against the IDENTITY display only (a plain
        # `rng.shuffle` alone lands the answer verbatim as the prompt on
        # 1/n! of seeds -- measured 78/400 at N=3, 21/400 at N=4 before this
        # guard -- a question that tests nothing). Unlike the hard tier's own
        # adjacency guard, a full or partial reversal, or any other
        # non-identity permutation, is left alone: those still require the
        # learner to actually reorder the names.
        for _ in range(200):
            rng.shuffle(display)
            if display != order_asc:
                break
        else:
            raise RuntimeError("p11d_compare_angle_names: could not avoid an identity shuffle")

    text = f"Arrange these angle types in order from smallest to largest: {', '.join(display)}."
    answer_text = ", ".join(order_asc)

    _check_order(text, answer_text)

    reversed_list, adjacent_swap = reversed_and_adjacent_swap(order_asc)
    candidates = [", ".join(reversed_list), ", ".join(adjacent_swap)]
    if n >= 4:
        # First-last-swap is a THIRD distractor only at N>=4: at N=3, swapping
        # the first and last item while keeping the middle unchanged is the
        # mathematically identical operation to full reversal, not a distinct
        # wrong answer.
        first_last = list(order_asc)
        first_last[0], first_last[-1] = first_last[-1], first_last[0]
        candidates.append(", ".join(first_last))

    check_pairwise_distinct(answer_text, *candidates, msg="p11d_compare_angle_names: distractor collision")

    working = [
        f"Canonical order, smallest to largest: {', '.join(_ORDER)}.",
        f"Answer: {answer_text}.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=list(order_asc),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "p11d_compare_angle_names",
    level=4,
    presets={
        "easy": dict(n=3, hard_shuffle=False),
        "medium": dict(n=4, hard_shuffle=False),
        "hard": dict(n=5, hard_shuffle=True),
    },
)
def p11d_compare_angle_names(rng, n=3, hard_shuffle=False):
    if n < 2 or n > len(_ORDER):
        raise ValueError(f"n must be between 2 and {len(_ORDER)}; got {n}")
    return _order_question(rng, n, hard_shuffle)
