"""Half-plane and strip loci from Re and Im conditions: `sr2l_re_im` (easy, medium, hard) and `sr2l_re_im_exam`.

Skill: locus-re-im-half-plane. The answer to every practice question is a one-variable condition on x or y (with z = x + iy),
graded by the `inequality` kind: `x <= -2`, `y > 1`, `-2 < x <= 1`. The condition is on a transformed number
w = s u + t, where u is z or its conjugate, s is a real number or a real multiple of i, and t is a complex constant; which
of x and y survives, and which way the inequality points, depend on that transformation.

* easy: w = z + t. Only the constant moves the boundary.
* medium: w = s z + t with s a real whole number, sometimes negative, so the inequality may have to reverse.
* hard: w also involves the conjugate or a multiple of i (so Re and Im swap roles, and Im of a conjugate changes sign),
  and about half the questions give a double inequality that describes a strip.
* exam: S = {z : Re(z - d1) >= Im(z + d2 i)} or its mirror. The boundary line, where it meets the real axis, the condition on
  the real numbers in S (inequality), and the least |z| over S (follow-through from the intercept).

Every key is checked against the printed condition by evaluating both as text at a grid of points of the plane,
boundary points included; they must agree at every one.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, latex_answer, phrase

from . import _sr2l_kit as kit

_OPS = {"<": "<", "≤": r"\le", ">": ">", "≥": r"\ge"}
_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_STRICT = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}
_ASK = "Write the condition on z as an inequality in x or y alone, where z = x + iy."

_IMPLICIT = re.compile(r"(?<=[\d)bzj])\s*(?=[zxy(])")


@functools.lru_cache(maxsize=4096)
def _compile(condition: str):
    t = condition.replace("≤", "<=").replace("≥", ">=").replace("z̄", "zb").replace("−", "-")
    t = re.sub(r"(\d)i(?=zb?\b)", r"\1*1j*", t)
    t = re.sub(r"\bi(?=zb?\b)", "1j*", t)
    t = re.sub(r"(\d)i\b", r"\1j", t)
    t = re.sub(r"\bi\b", "1j", t)
    t = _IMPLICIT.sub("*", t)
    check(re.fullmatch(r"[\d\s+\-*/().jxyzbRmIe<>=]+", t) is not None, f"unreadable condition {condition!r}")
    return compile(t, "<condition>", "eval")


_ENV = {
    "Re": lambda w: w.real if isinstance(w, complex) else w,
    "Im": lambda w: w.imag if isinstance(w, complex) else 0,
}


def holds(condition: str, x, y) -> bool:
    """Evaluate a printed condition (`Re(2z + 3) > 5`, `-2 < x ≤ 1`) at z = x + iy, reading only its text."""
    z = complex(float(x), float(y))
    return bool(
        eval(
            _compile(condition),
            {"__builtins__": {}},
            {"z": z, "zb": z.conjugate(), "x": float(x), "y": float(y), **_ENV},
        )
    )  # noqa: S307


@functools.lru_cache(maxsize=1)
def _points():
    pts = [(i / 2, j / 2) for i in range(-24, 25) for j in range(-24, 25) if (i * 7 + j * 3) % 5 == 0]
    pts += [(float(i), float(j)) for i in range(-12, 13) for j in range(-12, 13)]
    return tuple(pts)


def _agree(printed: str, key: str) -> None:
    for x, y in _points():
        check(holds(printed, x, y) == holds(key, x, y), f"{printed!r} and {key!r} disagree at ({x}, {y})")


class _Transform:
    """w = s u + t with u = z or z-bar; s = k (real) or k i (imaginary)."""

    def __init__(self, k: int, imaginary: bool, conjugate: bool, t: tuple[int, int]):
        self.k, self.imaginary, self.conjugate, self.t = k, imaginary, conjugate, t

    def tex(self) -> str:
        u = r"\bar{z}" if self.conjugate else "z"
        var = f"i{u}" if self.imaginary else u
        return join_terms([(self.k, var), (self.t[0], ""), (self.t[1], "i")])

    def linear(self, part: str):
        """(variable, a, b) with part(w) = a * variable + b."""
        sigma = -1 if self.conjugate else 1
        sr, si = (0, self.k) if self.imaginary else (self.k, 0)
        if part == "Re":
            ax, ay, b = sr, -si * sigma, self.t[0]
        else:
            ax, ay, b = si, sr * sigma, self.t[1]
        check((ax == 0) != (ay == 0), "the transformation leaves both variables")
        return ("x", Fraction(ax), Fraction(b)) if ax else ("y", Fraction(ay), Fraction(b))


def _solve(var, a, b, op, k):
    """The condition a*var + b op k as (var, op', bound)."""
    bound = (k - b) / a
    return var, (_FLIP[op] if a < 0 else op), bound


def _meaning_single(var, op, bound):
    """The variable and the interval `var op bound` describes, as a hashable value."""
    b = Fraction(bound)
    return {
        "<": (var, None, False, b, False),
        "≤": (var, None, False, b, True),
        ">": (var, b, False, None, False),
        "≥": (var, b, True, None, False),
    }[op]


def _meaning_band(var, lo, lo_op, hi, hi_op):
    lo, hi = Fraction(lo), Fraction(hi)
    if lo > hi or (lo == hi and (lo_op == "<" or hi_op == "<")):
        return (var, "empty")
    return (var, lo, lo_op == "≤", hi, hi_op == "≤")


def _text(var, op, bound) -> str:
    return f"{var} {op} {kit.rt(bound)}"


def _band_text(var, lo, lo_op, hi, hi_op) -> str:
    return f"{kit.rt(lo)} {lo_op} {var} {hi_op} {kit.rt(hi)}"


def _single(var, op, bound, tag):
    return _text(var, op, bound), tag, _meaning_single(var, op, bound)


def _band(var, lo, lo_op, hi, hi_op, tag):
    return _band_text(var, lo, lo_op, hi, hi_op), tag, _meaning_band(var, lo, lo_op, hi, hi_op)


def _choose(rng, tier):
    for _ in attempts(500, "a transformation with a single surviving variable"):
        part = rng.choice(("Re", "Im"))
        t = (rng.randint(-6, 6), rng.randint(-6, 6))
        if tier == "easy":
            tr = _Transform(1, False, False, t)
        elif tier == "medium":
            tr = _Transform(rng.choice((2, 3, 4, -1, -2, -3)), False, False, t)
        else:
            imaginary = rng.random() < 0.6
            tr = _Transform(rng.choice((1, -1, 2, -2, 3)), imaginary, rng.random() < 0.55 or not imaginary, t)
        if tier == "hard" and not (tr.imaginary or tr.conjugate):
            continue
        var, a, b = tr.linear(part)
        if tr.t != (0, 0) and (t[0] if part == "Re" else t[1]) != 0:
            return part, tr, var, a, b
    raise AssertionError("unreachable")  # pragma: no cover


def _band_bounds(a, b, lo_k, lo_op, hi_k, hi_op):
    """(lower, lower_op, upper, upper_op) for lo_k lo_op (a*v + b) hi_op hi_k solved for v."""
    l1, l2 = (Fraction(lo_k) - b) / a, (Fraction(hi_k) - b) / a
    return (l1, lo_op, l2, hi_op) if a > 0 else (l2, hi_op, l1, lo_op)


def _build(rng, tier: str):
    part, tr, var, a, b = _choose(rng, tier)
    other = "y" if var == "x" else "x"
    double = tier == "hard" and rng.random() < 0.5
    w = tr.tex()
    head = rf"\operatorname{{{part}}}({w})"
    tag_sign = "sr2l-conjugate-sign-ignored" if tr.conjugate else "sr2l-multiplication-by-i-sign"
    if double:
        lo_k = rng.randint(-5, 2)
        hi_k = lo_k + rng.randint(2, 7)
        lo_op, hi_op = rng.choice(("<", "≤")), rng.choice(("<", "≤"))
        tex = rf"{lo_k} {_OPS[lo_op]} {head} {_OPS[hi_op]} {hi_k}"
        lo, o1, hi, o2 = _band_bounds(a, b, lo_k, lo_op, hi_k, hi_op)
        key = _band_text(var, lo, o1, hi, o2)
        key_meaning = _meaning_band(var, lo, o1, hi, o2)
        cands = []
        if a < 0:  # the inequality signs were left pointing the way they started
            cands.append(
                _band(
                    var,
                    (Fraction(lo_k) - b) / a,
                    lo_op,
                    (Fraction(hi_k) - b) / a,
                    hi_op,
                    "sr2l-inequality-not-reversed",
                )
            )
        up = _band_bounds(a, -b, lo_k, lo_op, hi_k, hi_op)
        cands += [
            _band(var, *up, "sr2l-constant-sign-error"),
            _band(var, Fraction(lo_k) - b, lo_op, Fraction(hi_k) - b, hi_op, "sr2l-coefficient-not-divided"),
            _band(other, lo, o1, hi, o2, "sr2l-re-im-confused"),
            _band(var, lo, _STRICT[o1], hi, o2, "sr2l-boundary-included-or-excluded"),
            _band(var, lo, o1, hi, _STRICT[o2], "sr2l-boundary-included-or-excluded"),
        ]
        if tr.conjugate or tr.imaginary:
            cands.append(_band(var, *_band_bounds(-a, b, lo_k, lo_op, hi_k, hi_op), tag_sign))
        working_core = f"{lo_k} {lo_op} {part}(w) {hi_op} {hi_k}"
    else:
        op = rng.choice(list(_OPS))
        k = rng.randint(-6, 8)
        tex = rf"{head} {_OPS[op]} {k}"
        _, op2, bound = _solve(var, a, b, op, Fraction(k))
        key = _text(var, op2, bound)
        key_meaning = _meaning_single(var, op2, bound)
        cands = []
        if a < 0:
            cands.append(_single(var, op, bound, "sr2l-inequality-not-reversed"))
        if tr.conjugate or tr.imaginary:
            _, op3, bound3 = _solve(var, -a, b, op, Fraction(k))
            cands.append(_single(var, op3, bound3, tag_sign))
        cands += [
            _single(var, op2, (k + b) / a, "sr2l-constant-sign-error"),
            _single(var, op, Fraction(k) - b, "sr2l-coefficient-not-divided")
            if a != 1
            else _single(var, _STRICT[op2], bound, "sr2l-boundary-included-or-excluded"),
            _single(other, op2, bound, "sr2l-re-im-confused"),
            _single(var, _STRICT[op2], bound, "sr2l-boundary-included-or-excluded"),
        ]
        working_core = f"{part}(w) {op} {k}"
    _agree(kit.plain(tex), key)
    working = [
        f"Write w = {kit.plain(w)} in terms of x and y with z = x + iy: {part}(w) = {kit.coef_term(a, var)} {'+' if b >= 0 else '-'} {kit.rt(abs(b))}.",
        f"The condition {working_core} then gives {key}"
        + (" (dividing by a negative number reverses the inequality)." if a < 0 and not double else "."),
    ]
    fields = dict(answer=key, answer_text=key, answer_latex=latex_answer(key, plain=True), answer_kind="inequality")
    return tex, fields, cands, working, key_meaning


_STEMS = (
    "The complex number z = x + iy satisfies {rel}. " + _ASK,
    "The locus of the complex number z = x + iy is given by {rel}. " + _ASK,
    "Let z = x + iy, where {rel}. " + _ASK,
)


def _question(rng, tier: str):
    tex, fields, cands, working, meaning = _build(rng, tier)
    stem = phrase(rng, _STEMS, rel=kit.maths(tex))
    return dict(
        **kit.prompt(stem),
        working=working,
        distractors=kit.tagged_semantic(meaning, cands),
        **fields,
    )


_TIERS = ("easy", "medium", "hard")


@generator("sr2l_re_im", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_re_im(rng, tier: str = "easy"):
    """A condition on Re or Im of a transformation of z, solved to a condition on x or y alone."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, lambda r: _question(r, tier))


# -- exam ------------------------------------------------------------------------------------------------------------


@generator("sr2l_re_im_exam", level=18, presets={"exam": {}})
def sr2l_re_im_exam(rng):
    """The half-plane Re(z - d1) >= Im(z + d2 i) (or its mirror): boundary, intercept, real points, least modulus."""
    return kit.redraw(rng, _exam)


def _exam(rng):
    upper = rng.random() < 0.5
    d1, d2 = rng.randint(-3, 5), rng.randint(-3, 5)
    d = d1 + d2
    if d < 2:
        d1, d2 = rng.randint(1, 5), rng.randint(1, 5)
        d = d1 + d2
    re_side = join_terms([(1, "z"), (-d1, "")])
    im_side = join_terms([(1, "z"), (d2, "i")])
    if upper:
        # Im(z - d1 i) >= Re(z + d2): y - d1 >= x + d2, so y - x >= d
        left = rf"\operatorname{{Im}}({join_terms([(1, 'z'), (-d1, 'i')])})"
        right = rf"\operatorname{{Re}}({join_terms([(1, 'z'), (d2, '')])})"
        line_m, line_c, x_int = 1, d, -d
        key_c = f"x ≤ {-d}"
    else:
        left = rf"\operatorname{{Re}}({re_side})"
        right = rf"\operatorname{{Im}}({im_side})"
        line_m, line_c, x_int = 1, -d, d
        key_c = f"x ≥ {d}"
    cond = rf"{left} \ge {right}"
    printed = kit.plain(cond)
    # the boundary is y = x + line_c, and S lies below it (right form) or above it (upper form)
    for x in range(-8, 9):
        for y in range(-8, 9):
            inside = holds(printed, x, y)
            check(inside == ((y - x >= d) if upper else (x - y >= d)), "the printed set and its description disagree")
    check(
        holds(printed, x_int, 0) and not holds(printed, x_int + (1 if upper else -1), 0),
        "the real-axis boundary point is wrong",
    )
    real_points = {x for x in range(-12, 13) if holds(printed, x, 0)}
    check(
        real_points == {x for x in range(-12, 13) if (x <= -d if upper else x >= d)},
        "the real points of S disagree with the key",
    )
    least = d / math.sqrt(2)
    part_a = kit.line_part(
        "Find the equation of the line that forms the boundary of S, in the form y = mx + c.",
        line_m,
        line_c,
        1,
        ["Write each side in x and y: the boundary is where the two sides are equal."],
        [
            (line_m, -line_c, "sr2l-constant-sign-error"),
            (-line_m, line_c, "sr2l-gradient-sign-wrong"),
            (line_m, d1 - d2 if d1 - d2 not in (line_c, -line_c) else line_c + 1, "sr2l-re-im-constant-mixed"),
            (line_m, 0, "sr2l-vertex-ignored"),
        ],
    )
    part_b = kit.number_part(
        "Find the value of x where the boundary meets the real axis.",
        x_int,
        0,
        1,
        [f"Put y = 0 in the boundary: x = {x_int}."],
        [
            (-x_int, "sr2l-sign-lost-solving"),
            (d1, "sr2l-part-of-constant"),
            (d2, "sr2l-part-of-constant"),
            (d * d, "sr2l-constant-squared"),
        ],
    )
    cond_c = key_c
    fields_c = dict(
        answer=cond_c, answer_text=cond_c, answer_latex=latex_answer(cond_c, plain=True), answer_kind="inequality"
    )
    if upper:
        key_meaning = _meaning_single("x", "≤", -d)
        wrong_c = [
            _single("x", "≥", -d, "sr2l-inequality-not-reversed"),
            _single("x", "<", -d, "sr2l-boundary-included-or-excluded"),
            _single("y", "≤", -d, "sr2l-re-im-confused"),
            _single("x", "≤", d, "sr2l-constant-sign-error"),
        ]
    else:
        key_meaning = _meaning_single("x", "≥", d)
        wrong_c = [
            _single("x", "≤", d, "sr2l-inequality-not-reversed"),
            _single("x", ">", d, "sr2l-boundary-included-or-excluded"),
            _single("y", "≥", d, "sr2l-re-im-confused"),
            _single("x", "≥", -d, "sr2l-constant-sign-error"),
        ]
    part_c = dict(
        **kit.part_prompt(
            "The real numbers in S lie on the real axis, where y = 0. Write the condition on x that these numbers satisfy."
        ),
        marks=2,
        working=[f"Put y = 0 in the condition for S: {key_c}."],
        distractors=kit.tagged_semantic(key_meaning, wrong_c, want=2),
        **fields_c,
    )
    part_d = kit.number_part(
        "The origin is not in S. Find the least value of |z| for z in S, correct to 2 decimal places.",
        least,
        2,
        2,
        [
            f"The nearest point of S to the origin is the foot of the perpendicular to the boundary: |{x_int}| x sin 45° = {kit.number_text(least, 2)}."
        ],
        [
            (abs(x_int), "sr2l-intercept-for-distance"),
            (abs(x_int) * math.sqrt(2), "sr2l-hypotenuse-for-distance"),
            (abs(x_int) / 2, "sr2l-root-two-omitted"),
            (abs(x_int) / math.sqrt(2) / 2, "sr2l-factor-two-omitted"),
        ],
        follow="Abs(part_b)/sqrt(2)",
        sensible=(0, None),
    )
    stem = rf"S is the set of complex numbers z = x + iy that satisfy \({cond}\)."
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])
