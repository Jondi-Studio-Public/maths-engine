"""Senior specialist-mathematics skill ACMSM040 (Varignon's theorem): the
midpoints of the sides of a quadrilateral join to form a parallelogram.

**The trap this generator avoids:**
a question that gives two ADJACENT quadrilateral vertices and asks for the
midpoint between them is the existing `midpoint_two_points`
(mathgen/generators/coordinate_geometry_build.py, answer kind ``pair``) or
`midpoint_coordinate_scalar` (mathgen/generators/p5c_midpoint_build.py, answer
kind ``number``) skill's own graded operation, never this one. Every question
this generator builds withholds at least one quadrilateral vertex so that at
least one of the four midpoints P (mid AB), Q (mid BC), R (mid CD), S (mid DA)
can never be obtained from a single given adjacent vertex pair -- the ANSWER
this generator asks for is always one of those un-obtainable midpoints, and it
is only recoverable via the theorem's own vector consequence.

**The vector kernel.** Quadrilateral ABCD has vertices in order; P, Q, R, S
are the midpoints of sides AB, BC, CD, DA respectively. As position vectors,
P=(A+B)/2, Q=(B+C)/2, R=(C+D)/2, S=(D+A)/2, so P+R = Q+S = (A+B+C+D)/2 --
PQRS's diagonals PR and QS bisect each other at the same point, which is
exactly the parallelogram property the theorem asserts. Every tier below
solves this single vector equation for whichever one of P, Q, R, S the
question withholds, given the other three (each either computed from a
stated adjacent vertex pair, or handed directly as a fact the question
states because its own vertex pair is not fully known).

**Difficulty design, disclosed deviation from a literal reading.** A literal
medium-tier design (give A, C, and P, Q, S all directly, ask for R) requires *zero* midpoint-formula
computations at all -- strictly *less* work than easy, not more, since A and C then play no role in the
computation. Implemented here instead: EASY is handed the theorem's vector relation already solved for the
target (S = P + R - Q, a direct substitution after two ordinary midpoint computations); MEDIUM and HARD are
handed only the general relation (P + R = Q + S) and must derive which rearrangement isolates their own target
themselves -- a genuine algebra step easy does not require. HARD then also reduces the computational
scaffolding to a single midpoint (P only, from A and B) plus two directly-given midpoints (R and S), forcing
that self-derived rearrangement with the least support of the three tiers.
"""

from __future__ import annotations

from sympy import Rational

from ..core import (
    GRADER_TOLERANCE,
    check_equivalent,
    generator,
    pick_distractors,
    rational_text,
)


def _parity_group(rng, lo, hi, n):
    """n integers in [lo, hi], all sharing one randomly chosen parity.

    Used so that every plain midpoint the learner is asked to COMPUTE
    themselves (never the withheld one, which is always handed directly or
    solved via the vector relation) lands on an exact integer, never a
    half-integer -- the "exact integer pair, no fractions"
    requirement.
    """
    parity = rng.randint(0, 1)
    out = []
    for _ in range(n):
        while True:
            v = rng.randint(lo, hi)
            if v % 2 == parity:
                out.append(v)
                break
    return out


def _verify_relation(P, Q, R, S):
    """Independent check: confirm the fully reconstructed parallelogram
    PQRS (all four vertices, including the one just solved for) satisfies
    the vector-equality relation under three of its four cyclic rotations
    (Q-P==R-S, R-Q==S-P, S-R==P-Q -- the fourth, P-S==Q-R, is the negation of
    R-Q==S-P and adds nothing). These three are algebraically equivalent
    rearrangements of the SAME single vector identity (a parallelogram's
    four vertices satisfy exactly one independent vector constraint,
    P+R=Q+S) -- there is no second, genuinely independent relation to check
    here -- but each rotation is computed via different terms (a difference
    of two different vertex pairs) than however the unknown vertex was
    actually constructed, so a sign slip or wrong-vertex substitution in the
    construction still shows up as a failure here."""
    for label, lhs, rhs in (
        ("Q - P vs R - S", (Q, P), (R, S)),
        ("R - Q vs S - P", (R, Q), (S, P)),
        ("S - R vs P - Q", (S, R), (P, Q)),
    ):
        (la, lb), (ra, rb) = lhs, rhs
        check_equivalent(la[0] - lb[0], ra[0] - rb[0], f"{label} (x) inconsistent")
        check_equivalent(la[1] - lb[1], ra[1] - rb[1], f"{label} (y) inconsistent")


def _fmt_pair(pt) -> str:
    return f"x = {rational_text(pt[0])}, y = {rational_text(pt[1])}"


def _mid(u, v):
    return ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)


def _easy(rng, coord_lo, coord_hi):
    """A, B, C given (D withheld); R (mid CD) handed directly since D is
    unknown; ask for S (mid DA), also unobtainable directly. Learner computes
    P = mid(A,B) and Q = mid(B,C) themselves, then substitutes into the
    theorem's relation already solved for S: S = P + R - Q."""
    while True:
        ax, bx, cx = _parity_group(rng, coord_lo, coord_hi, 3)
        ay, by, cy = _parity_group(rng, coord_lo, coord_hi, 3)
        A, B, C = (Rational(ax), Rational(ay)), (Rational(bx), Rational(by)), (Rational(cx), Rational(cy))
        if A == B or B == C or A == C:
            continue
        rx = rng.randint(coord_lo, coord_hi)
        ry = rng.randint(coord_lo, coord_hi)
        R = (Rational(rx), Rational(ry))
        P = _mid(A, B)
        Q = _mid(B, C)
        S = (P[0] + R[0] - Q[0], P[1] + R[1] - Q[1])
        if Q == R:  # wrong-sign distractor (P+Q-R) would collide with S iff Q == R
            continue
        if S[0] == S[1]:  # swap-xy distractor would collide with S iff Sx == Sy
            continue
        break

    _verify_relation(P, Q, R, S)

    answer_text = _fmt_pair(S)
    wrong_sign = (P[0] + Q[0] - R[0], P[1] + Q[1] - R[1])
    swapped = (S[1], S[0])
    candidates = [_fmt_pair(wrong_sign), _fmt_pair(swapped), _fmt_pair(R)]

    text = (
        f"Quadrilateral ABCD has vertices A({ax}, {ay}), B({bx}, {by}) and "
        f"C({cx}, {cy}); the fourth vertex D is not given. P, Q, R and S are "
        "the midpoints of sides AB, BC, CD and DA respectively. By the "
        "midpoint theorem, PQRS (in that order) is itself a parallelogram, "
        f"so S = P + R - Q. Given that R = ({rx}, {ry}), find the "
        "coordinates of S."
    )
    working = [
        f"P = midpoint of AB = ({rational_text(P[0])}, {rational_text(P[1])}).",
        f"Q = midpoint of BC = ({rational_text(Q[0])}, {rational_text(Q[1])}).",
        f"S = P + R - Q = ({rational_text(S[0])}, {rational_text(S[1])}).",
    ]
    return dict(
        text=text,
        latex=rf"\text{{Midpoints of quadrilateral }}ABCD: \text{{ find }}S",
        answer={"x": float(S[0]), "y": float(S[1])},
        answer_text=answer_text,
        answer_latex=rf"x = {rational_text(S[0])},\; y = {rational_text(S[1])}",
        working=working,
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _medium(rng, coord_lo, coord_hi):
    """A, B, D given (C withheld); Q (mid BC) handed directly since C is
    unknown; ask for R (mid CD), also unobtainable directly. Learner computes
    P = mid(A,B) and S = mid(D,A) themselves, then must derive -- not
    substitute into a pre-solved formula -- the rearrangement of P + R = Q +
    S that isolates R: R = Q + S - P."""
    while True:
        ax, bx, dx = _parity_group(rng, coord_lo, coord_hi, 3)
        ay, by, dy = _parity_group(rng, coord_lo, coord_hi, 3)
        A, B, D = (Rational(ax), Rational(ay)), (Rational(bx), Rational(by)), (Rational(dx), Rational(dy))
        if A == B or B == D or A == D:
            continue
        qx = rng.randint(coord_lo, coord_hi)
        qy = rng.randint(coord_lo, coord_hi)
        Qg = (Rational(qx), Rational(qy))
        P = _mid(A, B)
        S = _mid(D, A)
        Rt = (Qg[0] + S[0] - P[0], Qg[1] + S[1] - P[1])
        if P == Qg:  # wrong-sign distractor (S+P-Q) collides with R iff P == Q
            continue
        if Qg == Rt:  # identity distractor (given Q) collides with R
            continue
        if Rt[0] == Rt[1]:  # swap-xy distractor collides with R
            continue
        break

    _verify_relation(P, Qg, Rt, S)

    answer_text = _fmt_pair(Rt)
    wrong_sign = (S[0] + P[0] - Qg[0], S[1] + P[1] - Qg[1])
    swapped = (Rt[1], Rt[0])
    candidates = [_fmt_pair(wrong_sign), _fmt_pair(swapped), _fmt_pair(Qg)]

    text = (
        f"Quadrilateral ABCD has vertices A({ax}, {ay}), B({bx}, {by}) and "
        f"D({dx}, {dy}); the fourth vertex C is not given. P, Q, R and S are "
        "the midpoints of sides AB, BC, CD and DA respectively. By the "
        "midpoint theorem, PQRS (in that order) is itself a parallelogram, "
        "so its diagonals bisect each other: P + R = Q + S. Given that Q = "
        f"({qx}, {qy}), find the coordinates of R."
    )
    working = [
        f"P = midpoint of AB = ({rational_text(P[0])}, {rational_text(P[1])}).",
        f"S = midpoint of DA = ({rational_text(S[0])}, {rational_text(S[1])}).",
        "Rearranging P + R = Q + S for R gives R = Q + S - P.",
        f"R = ({rational_text(Rt[0])}, {rational_text(Rt[1])}).",
    ]
    return dict(
        text=text,
        latex=rf"\text{{Midpoints of quadrilateral }}ABCD: \text{{ find }}R",
        answer={"x": float(Rt[0]), "y": float(Rt[1])},
        answer_text=answer_text,
        answer_latex=rf"x = {rational_text(Rt[0])},\; y = {rational_text(Rt[1])}",
        working=working,
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _hard(rng, coord_lo, coord_hi):
    """A, B given (C and D both withheld); R (mid CD) and S (mid DA) both
    handed directly since neither C nor D is known. Learner computes P = mid
    (A,B) -- the ONE midpoint directly computable here -- then derives the
    rearrangement of P + R = Q + S that isolates Q: Q = P + R - S. Two
    genuinely different reasoning steps (the plain midpoint formula, once;
    the parallelogram relation, rearranged) chained in the same question,
    with the least scaffolding of the three tiers (only one of the four
    midpoints is ever computed from a vertex pair)."""
    while True:
        ax, bx = _parity_group(rng, coord_lo, coord_hi, 2)
        ay, by = _parity_group(rng, coord_lo, coord_hi, 2)
        A, B = (Rational(ax), Rational(ay)), (Rational(bx), Rational(by))
        if A == B:
            continue
        rx = rng.randint(coord_lo, coord_hi)
        ry = rng.randint(coord_lo, coord_hi)
        sx = rng.randint(coord_lo, coord_hi)
        sy = rng.randint(coord_lo, coord_hi)
        Rg = (Rational(rx), Rational(ry))
        Sg = (Rational(sx), Rational(sy))
        if Rg == Sg:  # wrong-sign distractor (P+S-R) collides with Q iff R == S
            continue
        P = _mid(A, B)
        Qt = (P[0] + Rg[0] - Sg[0], P[1] + Rg[1] - Sg[1])
        given_choice = rng.choice(["R", "S"])
        identity = Rg if given_choice == "R" else Sg
        if identity == Qt:  # identity distractor collides with Q
            continue
        if Qt[0] == Qt[1]:  # swap-xy distractor collides with Q
            continue
        break

    _verify_relation(P, Qt, Rg, Sg)

    answer_text = _fmt_pair(Qt)
    wrong_sign = (P[0] + Sg[0] - Rg[0], P[1] + Sg[1] - Rg[1])
    swapped = (Qt[1], Qt[0])
    candidates = [_fmt_pair(wrong_sign), _fmt_pair(swapped), _fmt_pair(identity)]

    text = (
        f"Quadrilateral ABCD has vertices A({ax}, {ay}) and B({bx}, {by}); "
        "the other two vertices C and D are not given. P, Q, R and S are "
        "the midpoints of sides AB, BC, CD and DA respectively. By the "
        "midpoint theorem, PQRS (in that order) is itself a parallelogram, "
        "so its diagonals bisect each other: P + R = Q + S. Given that R = "
        f"({rx}, {ry}) and S = ({sx}, {sy}), find the coordinates of Q."
    )
    working = [
        f"P = midpoint of AB = ({rational_text(P[0])}, {rational_text(P[1])}).",
        "Rearranging P + R = Q + S for Q gives Q = P + R - S.",
        f"Q = ({rational_text(Qt[0])}, {rational_text(Qt[1])}).",
    ]
    return dict(
        text=text,
        latex=rf"\text{{Midpoints of quadrilateral }}ABCD: \text{{ find }}Q",
        answer={"x": float(Qt[0]), "y": float(Qt[1])},
        answer_text=answer_text,
        answer_latex=rf"x = {rational_text(Qt[0])},\; y = {rational_text(Qt[1])}",
        working=working,
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


@generator(
    "midpoints_of_the_sides_of_a_quadrilateral_join",
    level=11,
    presets={
        "easy": dict(variant="easy", coord_lo=0, coord_hi=6),
        "medium": dict(variant="medium", coord_lo=-8, coord_hi=8),
        "hard": dict(variant="hard", coord_lo=-14, coord_hi=14),
    },
)
def midpoints_of_the_sides_of_a_quadrilateral_join(rng, variant="easy", coord_lo=-8, coord_hi=8):
    """Compute a withheld midpoint of a quadrilateral's sides via the
    Varignon parallelogram relation, from three already-known midpoints/
    vertices -- never from a single given adjacent vertex pair alone."""
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")
    if variant == "easy":
        return _easy(rng, coord_lo, coord_hi)
    if variant == "medium":
        return _medium(rng, coord_lo, coord_hi)
    if variant == "hard":
        return _hard(rng, coord_lo, coord_hi)
    raise ValueError(f"unknown variant: {variant!r}")
