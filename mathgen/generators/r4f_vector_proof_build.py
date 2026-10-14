"""Generates `r4f_vector_proof` questions and the exam chain `exd_x_r4f_vector_proof`.

Skill `vector-proof-geometry` (``vectors``): the typed steps of a vector proof of a geometric theorem, in the basis of two
vectors a and b. The diagonals of a parallelogram bisect each other (and, in a trapezium, cut in the ratio 1 : k),
the midpoint (or line-joining-points-in-a-ratio) theorem, the midline of a trapezium, and the angle in a semicircle
(the dot product (b - a).(b + a) = |b|² - |a|², which vanishes exactly when |b| = |a|).

Every answer is either an expression in a and b (a vector, written like ``b - a`` or ``a/2 + b/2``), a fraction (the
scalar found by equating coefficients), or a number (a dot product or a length). No answer is a statement about which
line of a proof is right.

Positions are written as pairs (p, q) meaning p a + q b, measured from the first vertex. A vector between two named
points is the difference of their pairs.

Tiers (topic ``r4f_vector_proof``):

* easy   -- a named vector between vertices and midpoints of a parallelogram (or in the semicircle figure), in a and b;
* medium -- the scalar λ in OP = λ OB for the point P where the diagonals of a trapezium (CB = ka) meet, found by
  equating coefficients; k = 1 is the parallelogram, where λ = 1/2;
* hard   -- the midline MN of a trapezium as a vector, PQ for points dividing OA and OB in the same ratio, or the dot
  product that must vanish for the angle in a semicircle, the diagonals of a rhombus or a chord and its midpoint.

The exam chain (three shapes) has four parts: a trapezium's diagonals (AC, OB, λ, then the coefficient of a in OP,
follow-through from λ); the points dividing two sides in a ratio (AB, PQ, the ratio PQ : AB, the length of PQ,
follow-through from the ratio); and the semicircle (AP, BP, the dot product, the change in |OP|² that makes the angle
right, follow-through from the dot product).

Every vector is also checked by an independent route: coordinates in the plane for the positions, and the Gram
expansion of a dot product (the a·b terms must cancel) for the dot-product keys.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4f_ft import follow_through_clean
from .exd_x_r3c_parts import fraction_part, number_part

Pair = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------------------------------------------
# Vectors as pairs (p, q) = p a + q b


def _vec_text(vec: Pair) -> str:
    p, q = Fraction(vec[0]), Fraction(vec[1])
    check(p != 0 or q != 0, "the zero vector is never an answer")
    terms = []
    for coef, sym in ((p, "a"), (q, "b")):
        if coef == 0:
            continue
        mag = abs(coef)
        if mag.denominator == 1:
            body = sym if mag == 1 else f"{mag.numerator}{sym}"
        else:
            body = f"{sym}/{mag.denominator}" if mag.numerator == 1 else f"{mag.numerator}{sym}/{mag.denominator}"
        terms.append(("-" if coef < 0 else "+", body))
    sign, body = terms[0]
    text = ("-" if sign == "-" else "") + body
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return text


def _sub(u: Pair, v: Pair) -> Pair:
    return (u[0] - v[0], u[1] - v[1])


def _add(u: Pair, v: Pair) -> Pair:
    return (u[0] + v[0], u[1] + v[1])


def _scale(c, u: Pair) -> Pair:
    return (Fraction(c) * u[0], Fraction(c) * u[1])


def _neg(u: Pair) -> Pair:
    return (-u[0], -u[1])


def _frac_text(k) -> str:
    k = Fraction(k)
    return str(k.numerator) if k.denominator == 1 else f"{k.numerator}/{k.denominator}"


def _times_a(k) -> str:
    """'ka' as the stem writes it: 2a, a/2, 3a/2."""
    return _vec_text((Fraction(k), Fraction(0)))


def _expr_distractors(answer: Pair, cands) -> list:
    out, seen = [], {tuple(answer)}
    extras = [((answer[0], 0), "one-term-left-out"), ((0, answer[1]), "one-term-left-out")]
    for vec, tag in [*cands, *extras]:
        vec = (Fraction(vec[0]), Fraction(vec[1]))
        if vec == (0, 0) or vec in seen:
            continue
        seen.add(vec)
        text = _vec_text(vec)
        out.append(Distractor(text, tag, text))
    return out


def _expr_single(stem: str, answer: Pair, cands, working: list) -> dict:
    distractors = _expr_distractors(answer, cands)
    check(len(distractors) >= 3, "too few vector distractors")
    text = _vec_text(answer)
    return dict(
        text=stem,
        answer=text,
        answer_text=text,
        answer_kind="expression",
        working=working,
        distractors=distractors[:4],
    )


def _expr_part(prompt: str, answer: Pair, cands, working: list, marks: int = 1) -> dict:
    distractors = _expr_distractors(answer, cands)
    check(len(distractors) >= 3, "too few vector distractors in a part")
    text = _vec_text(answer)
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        marks=marks,
        working=working,
        distractors=distractors[:3],
    )


def _num_single(stem: str, part: dict) -> dict:
    """A single-answer question from a `number_part` result, graded exactly like the skill's other expression answers."""
    text = part["answer_text"]
    return dict(
        text=stem,
        answer=text,
        answer_text=text,
        answer_kind="expression",
        working=part["working"],
        distractors=[Distractor(d.text, d.misconception, d.text) for d in part["distractors"]],
    )


def _gram(u: Pair, v: Pair) -> tuple[Fraction, Fraction, Fraction]:
    """u · v = c_aa |a|² + c_ab a·b + c_bb |b|², by bilinearity."""
    return (u[0] * v[0], u[0] * v[1] + u[1] * v[0], u[1] * v[1])


def _dot_value(u: Pair, v: Pair, r: int, s: int, ab: int = 0) -> Fraction:
    c_aa, c_ab, c_bb = _gram(u, v)
    return c_aa * r * r + c_ab * ab + c_bb * s * s


# ---------------------------------------------------------------------------------------------------------------
# Easy: a named vector in a parallelogram figure or the semicircle figure

_FIGURES = (("O", "A", "B", "C"), ("P", "Q", "R", "S"), ("A", "B", "C", "D"))
_POINT_NAMES = ("M", "N", "X")


def _easy_parallelogram(rng):
    v0, v1, v2, v3 = names = rng.choice(_FIGURES)
    pos = {
        v0: (Fraction(0), Fraction(0)),
        v1: (Fraction(1), Fraction(0)),
        v2: (Fraction(1), Fraction(1)),
        v3: (Fraction(0), Fraction(1)),
    }
    if rng.random() < 0.25:
        # no extra point: a diagonal, or a side-to-opposite-vertex vector
        start, end = rng.choice([(v1, v3), (v3, v1), (v0, v2), (v2, v0)])
        intro = ""
    else:
        side = rng.choice([(v0, v1), (v1, v2), (v2, v3), (v3, v0), (v0, v2), (v1, v3)])
        x_name = rng.choice(_POINT_NAMES)
        m, n = (1, 1) if rng.random() < 0.6 else rng.choice(((1, 2), (2, 1), (1, 3), (3, 1)))
        t = Fraction(m, m + n)
        pos[x_name] = _add(pos[side[0]], _scale(t, _sub(pos[side[1]], pos[side[0]])))
        if m == n:
            intro = f" The point {x_name} is the midpoint of {side[0]}{side[1]}."
        else:
            intro = (
                f" The point {x_name} is on {side[0]}{side[1]} with {side[0]}{x_name} : {x_name}{side[1]} = {m} : {n}."
            )
        for _ in attempts(100, "a start and end that give a two-term vector"):
            start, end = rng.sample(sorted(pos), 2)
            if x_name not in (start, end):
                continue
            ans = _sub(pos[end], pos[start])
            if ans[0] != 0 and ans[1] != 0:
                break
    answer = _sub(pos[end], pos[start])
    # second route: coordinates for a = (3, 1), b = (-1, 2)
    a_vec, b_vec = (Fraction(3), Fraction(1)), (Fraction(-1), Fraction(2))
    xy = lambda pair: (pair[0] * a_vec[0] + pair[1] * b_vec[0], pair[0] * a_vec[1] + pair[1] * b_vec[1])  # noqa: E731
    check(
        xy(answer) == (xy(pos[end])[0] - xy(pos[start])[0], xy(pos[end])[1] - xy(pos[start])[1]),
        "coordinates disagree with the pair arithmetic",
    )
    cands = [
        (_neg(answer), "direction-of-vector-reversed"),
        (pos[end], "position-vector-not-displacement"),
        (_add(pos[end], pos[start]), "adds-position-vectors"),
        ((-answer[0], answer[1]), "sign-slip-on-one-vector"),
        ((answer[0], -answer[1]), "sign-slip-on-one-vector"),
        ((answer[1], answer[0]), "a-and-b-coefficients-swapped"),
        (_scale(2, answer), "midpoint-halving-missed"),
    ]
    stem = (
        f"{v0}{v1}{v2}{v3} is a parallelogram with vector {v0}{v1} = a and vector {v0}{v3} = b.{intro} "
        f"Express the vector {start}{end} in terms of a and b."
    )
    work = [
        f"Vector {start}{end} = (position of {end}) - (position of {start}) = {_vec_text(pos[end]) if pos[end] != (0, 0) else '0'} - ({_vec_text(pos[start]) if pos[start] != (0, 0) else '0'}) = {_vec_text(answer)}."
    ]
    return _expr_single(stem, answer, cands, work)


def _easy_semicircle(rng):
    target = rng.choice(("AP", "BP", "PA", "PB"))
    oa, ob, op = (Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(1))
    pos = {"A": oa, "B": ob, "P": op}
    answer = _sub(pos[target[1]], pos[target[0]])
    cands = [
        (_neg(answer), "direction-of-vector-reversed"),
        (pos[target[1]], "position-vector-not-displacement"),
        (_add(pos[target[1]], pos[target[0]]), "adds-position-vectors"),
        ((answer[1], answer[0]), "a-and-b-coefficients-swapped"),
        ((-answer[0], answer[1]), "sign-slip-on-one-vector"),
    ]
    stem = (
        "O is the centre of a circle and AB is a diameter, with vector OA = a and vector OB = -a. "
        f"The point P is on the circle, with vector OP = b. Express the vector {target} in terms of a and b."
    )
    work = [f"Vector {target} = vector O{target[1]} - vector O{target[0]} = {_vec_text(answer)}."]
    return _expr_single(stem, answer, cands, work)


# ---------------------------------------------------------------------------------------------------------------
# Medium and the first exam chain: the diagonals of a trapezium

_K_VALUES = (
    Fraction(1),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(5),
    Fraction(1, 2),
    Fraction(3, 2),
    Fraction(5, 2),
    Fraction(2, 3),
    Fraction(3, 4),
)


def _trap_check(k: Fraction) -> Fraction:
    """λ with OX = λ OB lying on AC; second route: P = λ(ka + b) satisfies P - A parallel to AC = b - a."""
    lam = 1 / (k + 1)
    p_vec = (lam * k, lam)
    on_ac = _sub(p_vec, (Fraction(1), Fraction(0)))
    ac = (Fraction(-1), Fraction(1))
    check(on_ac[0] * ac[1] - on_ac[1] * ac[0] == 0, "P is not on AC")
    return lam


def _trap_names(rng):
    return rng.choice((("O", "A", "B", "C"), ("P", "Q", "R", "S")))


def _trap_stem(names, k: Fraction) -> str:
    o, a_, b_, c_ = names
    if k == 1:
        shape = f"{o}{a_}{b_}{c_} is a parallelogram with vector {o}{a_} = a and vector {o}{c_} = b."
    else:
        shape = (
            f"{o}{a_}{b_}{c_} is a trapezium in which {c_}{b_} is parallel to {o}{a_}, with vector {o}{a_} = a, "
            f"vector {o}{c_} = b and vector {c_}{b_} = {_times_a(k)}."
        )
    return shape + f" The diagonals {o}{b_} and {a_}{c_} meet at X."


def _lambda_wrongs(k: Fraction):
    return [
        (1 / k, "ratio-op-to-pb-given-as-lambda"),
        (k / (k + 1), "coefficient-of-a-taken-as-lambda"),
        (Fraction(1, 2), "assumes-the-diagonals-bisect"),
        (1 / (k + 2), "sign-slip-in-one-minus-mu"),
        (k + 1, "reciprocal-of-lambda"),
    ]


def _medium(rng):
    k = rng.choice(_K_VALUES)
    names = _trap_names(rng)
    o, a_, b_, c_ = names
    lam = _trap_check(k)
    part = fraction_part(
        "",
        lam,
        working=[
            f"OX = λ({_vec_text((k, Fraction(1)))}) and OX = a + μ(b - a) = (1 - μ)a + μb. "
            f"Comparing b: λ = μ. Comparing a: {_frac_text(k)}λ = 1 - μ. So ({_frac_text(k)} + 1)λ = 1 and λ = {lam}."
        ],
        wrongs=[w for w in _lambda_wrongs(k) if w[0] != lam],
    )
    stem = (
        _trap_stem(names, k) + f" Let vector {o}X = λ × vector {o}{b_} and vector {a_}X = μ × vector {a_}{c_}. "
        f"Equating the coefficients of a and b in the two expressions for the vector {o}X, find λ as a fraction."
    )
    return _num_single(stem, part)


# ---------------------------------------------------------------------------------------------------------------
# Hard

_RATIOS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (3, 4))


def _hard_midline(rng):
    k = rng.choice([x for x in _K_VALUES if x != 1])
    o, a_, b_, c_ = names = _trap_names(rng)
    pos_c, pos_b, pos_a = (Fraction(0), Fraction(1)), (k, Fraction(1)), (Fraction(1), Fraction(0))
    m = _scale(Fraction(1, 2), pos_c)
    n = _scale(Fraction(1, 2), _add(pos_a, pos_b))
    answer = _sub(n, m)
    check(answer == ((k + 1) / 2, Fraction(0)), "midline is not (k + 1)a/2")
    # second route: coordinates a = (4, 0), b = (1, 3): MN is horizontal with length (OA + CB)/2
    cands = [
        (_neg(answer), "direction-of-vector-reversed"),
        (((k - 1) / 2, Fraction(0)), "parallel-sides-subtracted"),
        ((k + 1, Fraction(0)), "midline-not-halved"),
        (n, "position-vector-not-displacement"),
        ((k / 2, Fraction(0)), "one-parallel-side-left-out"),
    ]
    stem = (
        f"{o}{a_}{b_}{c_} is a trapezium in which {c_}{b_} is parallel to {o}{a_}, with vector {o}{a_} = a, vector {o}{c_} = b and "
        f"vector {c_}{b_} = {_times_a(k)}. The point M is the midpoint of {o}{c_} and N is the midpoint of {a_}{b_}. "
        "Express the vector MN in terms of a and b."
    )
    work = [
        f"{b_} = {_vec_text(pos_b)}, so N = ½(a + ({_vec_text(pos_b)})) = {_vec_text(n)}, M = ½b, and MN = N - M = {_vec_text(answer)}."
    ]
    return _expr_single(stem, answer, cands, work)


def _hard_division(rng):
    m, n = rng.choice(_RATIOS)
    o, a_, b_ = rng.choice((("O", "A", "B"), ("P", "Q", "R")))
    t = Fraction(m, m + n)
    pos_p, pos_q = (t, Fraction(0)), (Fraction(0), t)
    answer = _sub(pos_q, pos_p)
    check(answer == (-t, t), "PQ is not t(b - a)")
    if m == n:
        stem_extra = f"X is the midpoint of {o}{a_} and Y is the midpoint of {o}{b_}"
        px, qx = "X", "Y"
    else:
        px, qx = "X", "Y"
        stem_extra = f"X is the point on {o}{a_} with {o}X : X{a_} = {m} : {n}, and Y is the point on {o}{b_} with {o}Y : Y{b_} = {m} : {n}"
    cands = [
        (_neg(answer), "direction-of-vector-reversed"),
        (_scale(Fraction(m, n), (Fraction(-1), Fraction(1))), "ratio-part-to-part-used-as-scale"),
        ((Fraction(-1), Fraction(1)), "scale-factor-left-out"),
        (_scale(Fraction(n, m + n), (Fraction(-1), Fraction(1))), "wrong-part-of-the-ratio"),
        ((t, t), "sum-instead-of-difference"),
    ]
    stem = (
        f"In triangle {o}{a_}{b_}, vector {o}{a_} = a and vector {o}{b_} = b. {stem_extra}. "
        f"Express the vector {px}{qx} in terms of a and b."
    )
    work = [
        f"vector {o}X = {_vec_text((t, Fraction(0)))}, vector {o}Y = {_vec_text((Fraction(0), t))}, so XY = {_vec_text(answer)} = {_frac_text(t)}({_vec_text((Fraction(-1), Fraction(1)))})."
    ]
    return _expr_single(stem, answer, cands, work)


def _hard_dot(rng):
    kind = rng.choice(("semicircle", "rhombus", "chord"))
    for _ in attempts(200, "unequal lengths with a whole answer"):
        r, s = rng.randint(3, 12), rng.randint(3, 12)
        if r != s and (kind != "chord" or (s * s - r * r) % 2 == 0):
            break
    if kind == "semicircle":
        u, v = (Fraction(-1), Fraction(1)), (Fraction(1), Fraction(1))  # AP = b - a, BP = b + a
        value = _dot_value(u, v, r, s)
        stem = (
            f"O is the centre of a circle and AB is a diameter, with vector OA = a and vector OB = -a, where |a| = {r}. "
            f"The point P has vector OP = b with |b| = {s}. Find the dot product of the vectors AP and BP."
        )
        work = [f"AP = b - a and BP = b + a, so AP·BP = |b|² - |a|² = {s * s} - {r * r} = {value}."]
    elif kind == "rhombus":
        u, v = (Fraction(1), Fraction(1)), (Fraction(-1), Fraction(1))  # OB = a + b, AC = b - a
        value = _dot_value(u, v, r, s)
        stem = (
            f"OABC is a parallelogram with vector OA = a and vector OC = b, where |a| = {r} and |b| = {s}. "
            "Find the dot product of the diagonals, vector OB and vector AC."
        )
        work = [f"OB = a + b and AC = b - a, so OB·AC = |b|² - |a|² = {s * s} - {r * r} = {value}."]
    else:
        u, v = (Fraction(1, 2), Fraction(1, 2)), (Fraction(-1), Fraction(1))  # OM = (a + b)/2, AB = b - a
        value = _dot_value(u, v, r, s)
        stem = (
            f"In triangle OAB, vector OA = a and vector OB = b, where |a| = {r} and |b| = {s}. The point M is the "
            "midpoint of AB. Find the dot product of vector OM and vector AB."
        )
        work = [f"OM = ½(a + b) and AB = b - a, so OM·AB = ½(|b|² - |a|²) = ½({s * s} - {r * r}) = {value}."]
    # second route: the a·b terms cancel, and the value does not depend on a·b
    c_aa, c_ab, c_bb = _gram(u, v)
    check(c_ab == 0, "the a·b terms do not cancel")
    check(_dot_value(u, v, r, s, ab=7) == value, "the dot product depends on a·b")
    check(value.denominator == 1, "the dot product is not whole")
    half = Fraction(1, 2) if kind == "chord" else Fraction(1)
    part = number_part(
        "",
        value,
        working=work,
        wrongs=[
            (-value, "dot-product-sign-reversed"),
            (half * (s * s + r * r), "dot-product-sums-the-squares"),
            (half * (s - r) ** 2, "difference-of-squares-expanded-as-a-square"),
            (half * (s - r) if kind != "chord" else s - r, "lengths-subtracted-not-squared"),
            (
                value * 2 if kind == "chord" else half * (s * s),
                "half-of-the-dot-product-dropped" if kind == "chord" else "second-length-left-out",
            ),
        ],
    )
    return _num_single(stem, part)


@generator(
    "r4f_vector_proof",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4f_vector_proof(rng: random.Random, tier: str = "easy") -> dict:
    """A typed vector-proof step: a named vector (easy), the scalar from equating coefficients (medium), a midline, a ratio segment or a dot product (hard)."""
    pick = rng.random()
    if tier == "easy":
        return _easy_parallelogram(rng) if pick < 0.75 else _easy_semicircle(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard_midline(rng) if pick < 0.3 else _hard_division(rng) if pick < 0.6 else _hard_dot(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------------------------------------------
# Exam chains


def _exam_trapezium(rng):
    k = rng.choice([x for x in _K_VALUES if x != 1])
    names = _trap_names(rng)
    o, a_, b_, c_ = names
    lam = _trap_check(k)
    x_coef = k * lam
    forward = rng.random() < 0.5
    start, end = (a_, c_) if forward else (c_, a_)
    ac = (Fraction(-1), Fraction(1)) if forward else (Fraction(1), Fraction(-1))
    ob = (k, Fraction(1))
    part_a = _expr_part(
        f"Find the vector {start}{end} in terms of a and b.",
        ac,
        [
            (_neg(ac), "direction-of-vector-reversed"),
            ((Fraction(1), Fraction(1)), "adds-position-vectors"),
            ((ac[1], ac[0]), "a-and-b-coefficients-swapped"),
            ((ac[0], -ac[1]), "sign-slip-on-one-vector"),
        ],
        [f"Vector {start}{end} = vector O{end} - vector O{start} = {_vec_text(ac)}."],
    )
    part_b = _expr_part(
        f"Find the vector {o}{b_} in terms of a and b.",
        ob,
        [
            ((Fraction(1), Fraction(1)), "parallel-side-taken-as-equal"),
            ((-k, Fraction(1)), "sign-slip-on-one-vector"),
            ((Fraction(1), k), "a-and-b-coefficients-swapped"),
            ((k + 1, Fraction(1)), "sides-added-twice"),
        ],
        [f"Vector {o}{b_} = vector {o}{c_} + vector {c_}{b_} = {_vec_text(ob)}."],
        marks=1,
    )
    part_c = fraction_part(
        (
            f"The point X is on {o}{b_} with vector {o}X = λ × vector {o}{b_}, and on {a_}{c_} with vector {a_}X = μ × vector {a_}{c_}. "
            f"Equating the coefficients of a and b in the two expressions for vector {o}X, find λ as a fraction."
        ),
        lam,
        marks=2,
        sensible_range=(0, 1),
        working=[
            f"{o}X = λ({_vec_text(ob)}) = (1 - μ)a + μb. Comparing b: λ = μ; comparing a: {_frac_text(k)}λ = 1 - μ. So λ = {lam}."
        ],
        wrongs=[w for w in _lambda_wrongs(k) if w[0] != lam and 0 < w[0] < 1]
        + [(Fraction(1, 3) if lam != Fraction(1, 3) else Fraction(1, 4), "assumes-the-diagonals-trisect")],
    )
    part_d = fraction_part(
        f"Hence vector {o}X = x a + y b. Find x, the coefficient of a, as a fraction.",
        x_coef,
        marks=2,
        sensible_range=(0, None),
        working=[f"{o}X = λ({_vec_text(ob)}), so x = {_frac_text(k)}λ = {x_coef}."],
        wrongs=[
            (lam, "gives-lambda-for-x"),
            (k, "scale-factor-left-out"),
            (Fraction(1, 2), "assumes-the-diagonals-bisect"),
            (1 - x_coef, "complement-of-x-given"),
        ],
        follow_through=f"({_frac_text(k)})*part_c",
    )
    return dict(text=_trap_stem(names, k), parts=[part_a, part_b, part_c, part_d])


def _exam_division(rng):
    m, n = rng.choice([x for x in _RATIOS if x[0] != x[1]])
    o, a_, b_ = rng.choice((("O", "A", "B"), ("P", "Q", "R")))
    t = Fraction(m, m + n)
    length = (m + n) * rng.randint(2, 6)
    forward = rng.random() < 0.5
    ab = (Fraction(-1), Fraction(1)) if forward else (Fraction(1), Fraction(-1))
    s0, e0 = (a_, b_) if forward else (b_, a_)
    xy = _scale(t, ab)
    part_a = _expr_part(
        f"Find the vector {s0}{e0} in terms of a and b.",
        ab,
        [
            (_neg(ab), "direction-of-vector-reversed"),
            ((Fraction(1), Fraction(1)), "adds-position-vectors"),
            ((ab[1], ab[0]), "a-and-b-coefficients-swapped"),
            ((ab[0], -ab[1]), "sign-slip-on-one-vector"),
        ],
        [f"Vector {s0}{e0} = vector O{e0} - vector O{s0} = {_vec_text(ab)}."],
    )
    part_b = _expr_part(
        f"Find the vector {'XY' if forward else 'YX'} in terms of a and b.",
        xy,
        [
            (_neg(xy), "direction-of-vector-reversed"),
            (_scale(Fraction(m, n), ab), "ratio-part-to-part-used-as-scale"),
            (ab, "scale-factor-left-out"),
            (_scale(Fraction(n, m + n), ab), "wrong-part-of-the-ratio"),
        ],
        [
            f"vector OX = {_vec_text((t, Fraction(0)))}, vector OY = {_vec_text((Fraction(0), t))}, so the vector is {_vec_text(xy)}."
        ],
        marks=2,
    )
    part_c = fraction_part(
        f"Write vector XY = λ × vector {a_}{b_}. Find λ as a positive fraction.",
        t,
        marks=1,
        sensible_range=(0, 1),
        working=[
            f"XY = {_frac_text(t)}({_vec_text((Fraction(-1), Fraction(1)))}) = {_frac_text(t)} × vector {a_}{b_}, so λ = {t}."
        ],
        wrongs=[
            (Fraction(m, n), "ratio-part-to-part-used-as-scale"),
            (Fraction(n, m + n), "wrong-part-of-the-ratio"),
            (Fraction(1, 2), "assumes-the-midpoint-theorem"),
            (Fraction(m + n, m), "reciprocal-of-lambda"),
        ],
    )
    answer = t * length
    part_d = number_part(
        f"The length {a_}{b_} is {length} cm. Find the length of XY, in cm.",
        answer,
        marks=1,
        sensible_range=(0, None),
        working=[f"XY = {_frac_text(t)} × {length} = {answer} cm."],
        wrongs=[
            (length, "scale-factor-left-out"),
            (Fraction(length, m + n), "one-part-of-the-ratio-only"),
            (2 * answer, "length-doubled"),
            (length - m, "ratio-part-subtracted-from-the-length"),
        ],
        follow_through=f"part_c*{length}",
    )
    return dict(
        text=(
            f"In triangle {o}{a_}{b_}, vector {o}{a_} = a and vector {o}{b_} = b. X is the point on {o}{a_} with "
            f"{o}X : X{a_} = {m} : {n}, and Y is the point on {o}{b_} with {o}Y : Y{b_} = {m} : {n}."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


def _exam_semicircle(rng):
    for _ in attempts(200, "unequal lengths"):
        r, s = rng.randint(3, 12), rng.randint(3, 12)
        if r != s:
            break
    u, v = (Fraction(-1), Fraction(1)), (Fraction(1), Fraction(1))
    value = _dot_value(u, v, r, s)
    check(_gram(u, v)[1] == 0 and _dot_value(u, v, r, s, ab=5) == value, "the dot product depends on a·b")
    part_a = _expr_part(
        "Find the vector AP in terms of a and b.",
        u,
        [
            (_neg(u), "direction-of-vector-reversed"),
            ((Fraction(1), Fraction(1)), "adds-position-vectors"),
            ((Fraction(1), Fraction(-1)), "a-and-b-coefficients-swapped"),
            ((Fraction(-1), Fraction(-1)), "sign-slip-on-one-vector"),
        ],
        ["Vector AP = vector OP - vector OA = b - a."],
    )
    part_b = _expr_part(
        "Find the vector BP in terms of a and b.",
        v,
        [
            (_neg(v), "direction-of-vector-reversed"),
            ((Fraction(-1), Fraction(1)), "a-and-b-coefficients-swapped"),
            ((Fraction(1), Fraction(-1)), "sign-slip-on-one-vector"),
            ((Fraction(0), Fraction(1)), "position-vector-not-displacement"),
        ],
        ["Vector BP = vector OP - vector OB = b + a."],
    )
    part_c = number_part(
        f"Given |a| = {r} and |b| = {s}, find the dot product AP·BP.",
        value,
        marks=2,
        working=[f"AP·BP = (b - a)·(b + a) = |b|² - |a|² = {s * s} - {r * r} = {value}."],
        wrongs=[
            (-value, "dot-product-sign-reversed"),
            (s * s + r * r, "dot-product-sums-the-squares"),
            ((s - r) ** 2, "difference-of-squares-expanded-as-a-square"),
            (s - r, "lengths-subtracted-not-squared"),
        ],
    )
    part_d = number_part(
        "The angle APB is a right angle exactly when AP·BP = 0. By how much would |OP|² have to change, from its "
        "present value, to make that happen? Give an increase as a positive number and a decrease as a negative number.",
        -value,
        marks=2,
        working=[f"AP·BP = |OP|² - {r * r}, so |OP|² must become {r * r}: a change of {r * r} - {s * s} = {-value}."],
        wrongs=[
            (r * r, "new-value-given-not-the-change"),
            (s * s, "present-value-given-not-the-change"),
            (2 * (r * r - s * s), "change-doubled"),
            (s * s - r * r + r, "arithmetic-slip"),
        ],
        follow_through="-part_c",
    )
    return dict(
        text=(
            "O is the centre of a circle and AB is a diameter, with vector OA = a and vector OB = -a. "
            "The point P has vector OP = b."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


@generator("exd_x_r4f_vector_proof", level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@follow_through_clean
def exd_x_r4f_vector_proof(rng: random.Random, tier: str = "exam") -> dict:
    """Exam chain: diagonals of a trapezium, a segment joining two points in a ratio, or the angle in a semicircle."""
    check(tier == "exam", f"unknown tier {tier!r}")
    pick = rng.random()
    return _exam_trapezium(rng) if pick < 0.4 else _exam_division(rng) if pick < 0.7 else _exam_semicircle(rng)
