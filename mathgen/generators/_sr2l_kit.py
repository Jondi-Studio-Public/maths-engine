"""Shared helpers for the complex-plane locus generators (`sr2l_*_build.py`).

A locus question states a condition on z = x + iy and asks for something the existing graders can mark: a centre or a
point as a `pair`, a boundary line as a single-equation `pair` ("y = 2x + 1", "x = -3"), a one-variable condition as an
`inequality`, a length as a `number`, a classification as an `mcq`. A full Cartesian equation of a circle cannot be typed
and marked, so no question here asks for one.

Prompts are authored once, as prose with inline maths (``\\(\\lvert z - 2 + 3i\\rvert = 4\\)``), and `prompt` derives the
plain text from the same string, so the two can never disagree.

Every wrong option is read by the real grader (`rejects`) before it is kept, so none can be accepted by construction.
"""

from __future__ import annotations

import ast
import cmath
import math
import re
from functools import lru_cache
from fractions import Fraction
from types import SimpleNamespace

from mathgen import figures
from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    VerificationError,
    attempts,
    check,
    join_terms,
    latex_from_inline,
    pick_distractors,
    rational_text,
)
from mathgen.grading.typed import GRADERS

# -- prompts -----------------------------------------------------------------------------------------------------

_PLAIN = (
    (r"\lvert ", "|"),
    (r"\lvert", "|"),
    (r"\rvert", "|"),
    (r"\operatorname{Re}", "Re"),
    (r"\operatorname{Im}", "Im"),
    (r"\operatorname{Arg}", "Arg"),
    (r"\bar{z}", "z̄"),
    (r"\pi", "π"),
    (r"\theta", "θ"),
    (r"\le", "≤"),
    (r"\ge", "≥"),
    (r"\,", " "),
)


def plain(tex_text: str) -> str:
    """The plain-text twin of a prompt written with ``\\( ... \\)`` inline maths."""
    s = tex_text.replace(r"\(", "").replace(r"\)", "")
    s = re.sub(r"\\frac\{(\d*)\\pi\}\{(\d+)\}", r"\1π/\2", s)
    s = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", s)
    s = s.replace("^2", "²").replace("^3", "³")
    for tex, txt in _PLAIN:
        s = s.replace(tex, txt)
    check("\\" not in s, f"an unconverted macro is left in {s!r}")
    return s


def prompt(tex_text: str) -> dict:
    """``text`` and ``latex`` for a prompt written with inline maths."""
    return dict(text=plain(tex_text), latex=latex_from_inline(tex_text))


_IMPLICIT = re.compile(r"(?<=[\d)bzjxy])\s*(?=[zxya(])")


def residual(relation: str, z: complex) -> float:
    """left side minus right side of a stated relation (plain text) at the complex number z.

    The relation is read back from its own printed text, a different route from the one that built it: |...| is the
    modulus, `i` the imaginary unit, `z̄` the conjugate, and x and y the parts of z, so a printed locus can be tested
    at points without trusting the generator's centre and radius."""
    check("=" in relation, f"not a relation: {relation!r}")
    left, right = relation.split("=", 1)
    out = []
    for side in (left, right):
        t = side.replace("^", "**").replace("²", "**2").replace("z̄", "zb").replace("−", "-")
        t = re.sub(r"\|([^|]+)\|", r"abs(\1)", t)
        t = re.sub(r"(\d)i\b", r"\1j", t)
        t = re.sub(r"\bi\b", "1j", t)
        t = _IMPLICIT.sub("*", t)
        check(re.fullmatch(r"[\d\s+\-*/().jxyzabs]+", t.replace("zb", "z")) is not None, f"unreadable side {side!r}")
        out.append(eval(t, {"__builtins__": {}}, {"z": z, "zb": z.conjugate(), "x": z.real, "y": z.imag, "abs": abs}))  # noqa: S307
    return abs(out[0] - out[1])


def check_locus(relation: str, c, r) -> None:
    """The printed `relation` holds at points of the claimed circle (centre c, radius r) and fails just off it."""
    centre = complex(float(Fraction(c[0])), float(Fraction(c[1])))
    for k in range(12):
        direction = cmath.exp(1j * (0.3 + 2 * math.pi * k / 12))
        scale = max(1.0, abs(centre) + r) ** 2
        check(residual(relation, centre + r * direction) < 1e-9 * scale, f"{relation!r} fails on the claimed circle")
        for off in (0.9, 1.1):
            check(
                residual(relation, centre + off * r * direction) > 1e-4,
                f"{relation!r} also holds off the claimed circle",
            )


def part_prompt(tex_text: str) -> dict:
    """``prompt`` and ``prompt_latex`` of an exam part written with inline maths."""
    return dict(prompt=plain(tex_text), prompt_latex=latex_from_inline(tex_text))


def maths(tex: str) -> str:
    """Wrap a formula as inline maths inside a prompt."""
    return rf"\({tex}\)"


# -- complex numbers and their locus expressions ----------------------------------------------------------------


def cplx(p, q) -> str:
    """p + qi as a student writes it: 2 - 3i, 4i, -i, 5."""
    return join_terms([(p, ""), (q, "i")])


def z_minus(p, q, var: str = "z") -> str:
    """The expression z - (p + qi) with the sign of each part resolved: z + 2 - 3i for the centre (-2, 3)."""
    return join_terms([(1, var), (-p, ""), (-q, "i")])


def mod(inner: str) -> str:
    return rf"\lvert {inner}\rvert"


def sq(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


def circle_rel(p, q, r, op: str = "=") -> str:
    """|z - (p + qi)| op r, with the centre's signs resolved."""
    return rf"{mod(z_minus(p, q))} {op} {r}"


def circle_plain(p, q, r, op: str = "=") -> str:
    return plain(circle_rel(p, q, r, op))


def rt(v) -> str:
    """An int or Fraction as a student writes it: 3, -1/2."""
    return rational_text(Fraction(v))


def shifted(var: str, c) -> str:
    """`x - 3`, `x + 2` or `x` for var minus the number c, never `x - -2`."""
    c = Fraction(c)
    if c == 0:
        return var
    return f"{var} - {rt(c)}" if c > 0 else f"{var} + {rt(-c)}"


def coef_term(a, var: str) -> str:
    """`x`, `-x` or `3x` for the coefficient a of var, never `1x`."""
    a = Fraction(a)
    if a == 1:
        return var
    if a == -1:
        return f"-{var}"
    return f"{rt(a)}{var}" if a.denominator == 1 else f"({rt(a)}){var}"


def pair_text(x, y) -> str:
    return f"({rt(x)}, {rt(y)})"


def num(v) -> int | float:
    """A Fraction as the structured number a number/pair answer stores."""
    v = Fraction(v)
    return int(v) if v.denominator == 1 else float(v)


def pair_fields(x, y) -> dict:
    x, y = Fraction(x), Fraction(y)
    text = pair_text(x, y)
    return dict(
        answer={"x": num(x), "y": num(y)},
        answer_text=text,
        answer_latex=text,
        answer_kind="pair",
    )


def line_text(m, c) -> str:
    """The line y = mx + c as typed: y = 2x + 1, y = -x/2... never a vertical line (see `vertical_text`)."""
    m, c = Fraction(m), Fraction(c)
    if m == 0:
        return f"y = {rt(c)}"
    if m == 1:
        lead = "x"
    elif m == -1:
        lead = "-x"
    elif m.denominator == 1:
        lead = f"{m.numerator}x"
    else:
        lead = f"({rt(m)})x"
    if c == 0:
        return f"y = {lead}"
    return f"y = {lead} {'+' if c > 0 else '-'} {rt(abs(c))}"


def vertical_text(k) -> str:
    return f"x = {rt(Fraction(k))}"


def line_fields(m, c) -> dict:
    text = line_text(m, c)
    return dict(answer={"m": num(m), "b": num(c)}, answer_text=text, answer_latex=text, answer_kind="pair")


def vertical_fields(k) -> dict:
    text = vertical_text(k)
    return dict(answer={"x": num(k)}, answer_text=text, answer_latex=text, answer_kind="pair")


def line_through(p, d) -> tuple[Fraction | None, Fraction]:
    """(m, c) of the line through the point p with direction d, or (None, k) for the vertical line x = k."""
    px, py = Fraction(p[0]), Fraction(p[1])
    dx, dy = Fraction(d[0]), Fraction(d[1])
    check(dx != 0 or dy != 0, "a line needs a direction")
    if dx == 0:
        return None, px
    m = dy / dx
    return m, py - m * px


def line_option(m, c) -> str:
    return vertical_text(c) if m is None else line_text(m, c)


# -- the real graders ---------------------------------------------------------------------------------------------


class TooFewOptions(VerificationError):
    """A draw left too few usable wrong options; `redraw` draws again."""


def redraw(rng, build, what: str = "a draw with enough distinct wrong options", tries: int = 60):
    """`build(rng)` again whenever it ran short of usable distractors. Any other failure still propagates."""
    for _ in attempts(tries, what):
        try:
            return build(rng)
        except TooFewOptions:
            continue
    raise AssertionError("unreachable")  # pragma: no cover


def accepts(kind: str, answer_text: str, reply: str, answer=None) -> bool:
    """Whether the real `kind` grader marks `reply` correct against a question whose key is `answer_text`."""
    return _accepts(kind, answer_text, repr(answer if answer is not None else answer_text), reply)


@lru_cache(maxsize=None)
def _accepts(kind: str, answer_text: str, answer_repr: str, reply: str) -> bool:
    answer = answer_text if answer_repr == repr(answer_text) else ast.literal_eval(answer_repr)
    q = SimpleNamespace(
        answer=answer,
        answer_text=answer_text,
        answer_form=None,
        answer_alt=None,
        answer_kind=kind,
        text="",
        parts=None,
    )
    return bool(GRADERS[kind](q, reply).correct)


def tagged(kind: str, key_fields: dict, cands, *, want: int = 3, limit: int = 3) -> list[Distractor]:
    """Wrong options from `cands`, a list of `(text, misconception[, value])` in priority order: no repeats, none the
    key's text and none that the real `kind` grader accepts. Fewer than `want` left is a failed draw."""
    key = key_fields["answer_text"]
    out, seen = [], {key}
    for cand in cands:
        text, tag = cand[0], cand[1]
        value = cand[2] if len(cand) > 2 else None
        if not text or text in seen or accepts(kind, key, text, key_fields["answer"]):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value) if value is not None else Distractor(text, tag))
        if len(out) == limit:
            break
    if len(out) < want:
        raise TooFewOptions(f"only {len(out)} usable distractors for {key}")
    return out


def tagged_lines(key_line, cands, *, want: int = 3, limit: int = 3) -> list[Distractor]:
    """Wrong line options from `cands`, a list of `(m, c, tag)` (m None for the vertical line x = c).

    Two different lines are never the same answer to the single-equation grader (it compares the equations
    symbolically), so a candidate is kept exactly when its (m, c) differs from the key's and from the options already
    kept; no grader call is needed. The tests grade every kept option through the real grader."""
    seen = {key_line}
    out = []
    for m, c, tag in cands:
        mc = (None if m is None else Fraction(m), Fraction(c))
        if mc in seen:
            continue
        seen.add(mc)
        out.append(Distractor(line_option(m, c), tag))
        if len(out) == limit:
            break
    if len(out) < want:
        raise TooFewOptions(f"only {len(out)} usable line options")
    return out


def tagged_semantic(key_meaning, cands, *, want: int = 3, limit: int = 3) -> list[Distractor]:
    """Wrong options from `cands`, a list of `(text, tag, meaning)`, where `meaning` is a hashable value that is equal
    exactly when two texts mean the same answer (for an inequality, the variable and the interval it describes).

    A candidate is kept when its meaning differs from the key's and its text is new, so no grader call is needed."""
    seen_text = set()
    out = []
    for text, tag, meaning in cands:
        if meaning == key_meaning or text in seen_text:
            continue
        seen_text.add(text)
        out.append(Distractor(text, tag))
        if len(out) == limit:
            break
    if len(out) < want:
        raise TooFewOptions(f"only {len(out)} usable options")
    return out


def pair_cands(cands):
    """`(x, y, tag)` triples as pair options."""
    return [(pair_text(x, y), tag, {"x": num(x), "y": num(y)}) for x, y, tag in cands]


def number_text(v, dp: int) -> str:
    """`v` rounded half-up and shown at exactly `dp` decimal places (an integer when dp is 0)."""
    from mathgen.core import round_half_up, stated_precision_string

    shown = round_half_up(Fraction(v).limit_denominator(10**9), dp)
    return stated_precision_string(shown, dp) if dp else str(int(shown))


def number_fields(v, dp: int) -> dict:
    text = number_text(v, dp)
    f = Fraction(text)
    return dict(
        answer=int(f) if dp == 0 else float(f),
        answer_text=text,
        answer_latex=text,
        answer_kind="number",
    )


def number_cands(cands, dp: int):
    return [(number_text(v, dp), tag, exact_value(Fraction(number_text(v, dp)))) for v, tag in cands]


def number_part(prompt_text, value, dp, marks, working, wrong, *, follow=None, sensible=None, ask=None) -> dict:
    """One `number` part of an exam chain (distractors are the part's own errors only)."""
    fields = number_fields(value, dp)
    part = dict(
        **part_prompt(prompt_text),
        marks=marks,
        working=working,
        distractors=tagged("number", fields, number_cands(wrong, dp), want=2),
        **fields,
    )
    if follow:
        part["follow_through"] = follow
    if sensible is not None:
        part["sensible_range"] = sensible
    return part


def pair_part(prompt_text, x, y, marks, working, wrong) -> dict:
    fields = pair_fields(x, y)
    return dict(
        **part_prompt(prompt_text),
        marks=marks,
        working=working,
        distractors=tagged("pair", fields, pair_cands(wrong), want=2),
        **fields,
    )


def line_part(prompt_text, m, c, marks, working, wrong) -> dict:
    """`wrong` is a list of `(m, c, tag)`; m None means a vertical line x = c."""
    fields = vertical_fields(c) if m is None else line_fields(m, c)
    key = (None if m is None else Fraction(m), Fraction(c))
    return dict(
        **part_prompt(prompt_text),
        marks=marks,
        working=working,
        distractors=tagged_lines(key, wrong, want=2),
        **fields,
    )


def list_part(prompt_text, values, marks, working, wrong) -> dict:
    """One `list` part (an unordered list of numbers); `wrong` is a list of lists of numbers, each tagged `(values, tag)`."""
    text = ", ".join(rt(v) for v in values)
    fields = dict(answer=[num(v) for v in values], answer_text=text, answer_latex=text, answer_kind="list")
    cands = [(", ".join(rt(v) for v in vals), tag, [num(v) for v in vals]) for vals, tag in wrong]
    return dict(
        **part_prompt(prompt_text),
        marks=marks,
        working=working,
        distractors=tagged("list", fields, cands, want=2),
        **fields,
    )


def mcq_fields(key: str, wrong, *, key_latex: str | None = None) -> dict:
    """The fields of an mcq question or part: `wrong` is a list of `(text, misconception)`."""
    seen, out = {key}, []
    for text, tag in wrong:
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag))
    if len(out) < 3:
        raise TooFewOptions(f"only {len(out)} distractors for {key}")
    return dict(
        answer=key,
        answer_text=key,
        answer_latex=key_latex or key,
        answer_kind="mcq",
        distractors=out[:3],
    )


# -- figures ------------------------------------------------------------------------------------------------------

RE_LABEL, IM_LABEL = "Re", "Im"


def plane(*, lo: int, hi: int, alt: str, points=(), segments=(), lines=(), curves=(), title: str = "") -> dict:
    """An Argand diagram: a square coordinate plane with the real axis across and the imaginary axis up."""
    return figures.cartesian(
        x_range=(lo, hi),
        y_range=(lo, hi),
        grid=1,
        x_label=RE_LABEL,
        y_label=IM_LABEL,
        title=title,
        alt=alt,
        points=points,
        segments=segments,
        lines=lines,
        curves=curves,
    )


def circle_curve(cx, cy, r, n: int = 72) -> dict:
    """A circle as one closed sampled curve."""
    pts = [
        [round(cx + r * math.cos(2 * math.pi * k / n), 4), round(cy + r * math.sin(2 * math.pi * k / n), 4)]
        for k in range(n)
    ]
    pts.append(pts[0])
    return {"pieces": [pts]}


def pythagorean_steps(limit: int = 13):
    """Integer vectors (a, b), a >= 0 or b >= 0, with an integer length: (3, 4), (0, 5), (6, 8) ... both signs."""
    out = []
    for a in range(-limit, limit + 1):
        for b in range(-limit, limit + 1):
            n = a * a + b * b
            if n and math.isqrt(n) ** 2 == n:
                out.append((a, b))
    return out


PYTH = tuple(v for v in pythagorean_steps(13) if math.hypot(*v) >= 5)
"""Integer vectors of integer length at least 5 (3-4-5, 5-12-13, 6-8-10 and their reflections)."""
