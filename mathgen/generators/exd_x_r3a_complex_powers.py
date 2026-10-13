"""Exam source of `senior-define-the-imaginary-number-i-as-a-root`: powers of a complex number built from i^2 = -1.

The practice tiers evaluate a single power of i. An exam question uses the defining property i^2 = -1 to build up a
higher power of a complex number z = a + bi in stages, each stage using the one before:

* shape ``fourth``: (a) z^2, (b) z^4 as the square of (a), (c) z^5 as the product of (b) and z;
* shape ``cube``: (a) z^2, (b) z^3 as the product of (a) and z, (c) z^4 as the product of (b) and z.

Every part is a complex number in the form p + qi. Parts (b) and (c) follow through from the earlier parts, so a
learner whose z^2 is wrong still earns the later parts for squaring or multiplying it correctly.

## Construction

The real and imaginary parts a and b are drawn (never zero, so the cross term is always present). Each stage is
computed with exact integer arithmetic on (real, imaginary) pairs; an independent route recomputes the keys with
SymPy's ``expand`` and ``I``. Distractors are the slips a learner makes with i^2: taken as +1, the cross term
dropped or not doubled, parts multiplied componentwise.
"""

from __future__ import annotations

from sympy import I, expand

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, join_terms, latex_answer, pick_distractors
from ._exd_distinct_parts import distinct_part_answers

_RANGE = [v for v in range(-5, 6) if v != 0]
_NAMES = ["z", "w", "u"]


def _value(label: str) -> tuple[str, str]:
    """Real and imaginary parts of an earlier part's reply, for a follow-through expression. A learner's `i` is a
    plain symbol in a reply, so it is turned into the imaginary unit first."""
    z = f"Subs({label}, (i,), (I,))"
    return f"re({z})", f"im({z})"


def _ft_square(label: str) -> str:
    x, y = _value(label)
    return f"{x}**2 - {y}**2 + 2*{x}*{y}*I"


def _ft_times(label: str, a: int, b: int) -> str:
    x, y = _value(label)
    return f"({x}*({a}) - {y}*({b})) + ({x}*({b}) + {y}*({a}))*I"


def _ctext(re: int, im: int) -> str:
    """`re + im i` the way a learner writes it."""
    if im == 0:
        return str(re)
    mag = "i" if abs(im) == 1 else f"{abs(im)}i"
    imag = mag if im > 0 else f"-{mag}"
    if re == 0:
        return imag
    return f"{re} {'+' if im > 0 else '-'} {mag}"


def _mul(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    return p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]


def _mul_i_squared_plus(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    """The product with i^2 taken as +1."""
    return p[0] * q[0] + p[1] * q[1], p[0] * q[1] + p[1] * q[0]


def _cands(correct, wrong: list[tuple[tuple[int, int], str]]) -> list[Distractor]:
    out = []
    for (re, im), tag in wrong:
        text = _ctext(re, im)
        out.append(Distractor(text, tag, text))
    return pick_distractors(_ctext(*correct), out, 3)


def _part(prompt, key, marks, distractors, working, follow_through=None):
    text = _ctext(*key)
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_r3a_complex_powers", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_complex_powers(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b = rng.choice(_RANGE), rng.choice(_RANGE)
    shape = rng.choice(["fourth", "cube"])
    name = rng.choice(_NAMES)
    z = (a, b)
    z2 = _mul(z, z)
    if shape == "fourth":
        second = _mul(z2, z2)  # z^4
        third = _mul(second, z)  # z^5
        n2, n3 = 4, 5
        ft_b, ft_c = _ft_square("part_a"), _ft_times("part_b", a, b)
        wrong_b = [
            (_mul_i_squared_plus(z2, z2), "i-squared-as-plus-one"),
            ((z2[0] ** 2, z2[1] ** 2), "squared-each-part-separately"),
            ((z2[0] * z2[0], 0), "cross-term-dropped"),
            ((2 * z2[0], 2 * z2[1]), "power-as-multiplication"),
        ]
        prev_b = second
    else:
        second = _mul(z2, z)  # z^3
        third = _mul(second, z)  # z^4
        n2, n3 = 3, 4
        ft_b, ft_c = _ft_times("part_a", a, b), _ft_times("part_b", a, b)
        wrong_b = [
            (_mul_i_squared_plus(z2, z), "i-squared-as-plus-one"),
            ((z2[0] * a, z2[1] * b), "complex-product-componentwise"),
            ((z2[0] * a - z2[1] * b, z2[0] * b - z2[1] * a), "complex-product-sign-slip"),
            ((3 * a, 3 * b), "power-as-multiplication"),
        ]
        prev_b = second

    # Independent route.
    zs = a + b * I
    for key, n in ((z2, 2), (second, n2), (third, n3)):
        want = expand(zs**n)
        check(want == key[0] + key[1] * I, f"z^{n} disagrees with SymPy")

    wrong_a = [
        ((a * a + b * b, 2 * a * b), "i-squared-as-plus-one"),
        ((a * a - b * b, 0), "cross-term-dropped"),
        ((a * a - b * b, a * b), "cross-term-not-doubled"),
        ((a * a, b * b), "squared-each-part-separately"),
    ]
    wrong_c = [
        (_mul_i_squared_plus(prev_b, z), "i-squared-as-plus-one"),
        ((prev_b[0] * a, prev_b[1] * b), "complex-product-componentwise"),
        (
            (prev_b[0] * a - prev_b[1] * b, prev_b[0] * b - prev_b[1] * a),
            "complex-product-sign-slip",
        ),
        ((prev_b[0] + a, prev_b[1] + b), "power-as-multiplication"),
    ]
    a_d, b_d, c_d = _cands(z2, wrong_a), _cands(second, wrong_b), _cands(third, wrong_c)
    for label, ds in (("a", a_d), ("b", b_d), ("c", c_d)):
        check(len(ds) >= 2, f"too few ({label}) distractors: {ds}")

    base = _ctext(a, b)
    stem = f"Let {name} = {base}, where i² = -1."
    return dict(
        text=stem,
        parts=[
            _part(
                f"Write {name}^2 in the form p + qi.",
                z2,
                2,
                a_d,
                [
                    f"{name}^2 = ({base})² = {join_terms([(a * a, ''), (2 * a * b, 'i'), (b * b, 'i²')])} = {_ctext(*z2)}."
                ],
            ),
            _part(
                f"Using your answer to (a), write {name}^{n2} in the form p + qi.",
                second,
                3,
                b_d,
                [
                    f"{name}^{n2} = {'(' + name + '^2)^2' if shape == 'fourth' else name + '^2 × ' + name} = {_ctext(*second)}."
                ],
                follow_through=ft_b,
            ),
            _part(
                f"Using your answer to (b), write {name}^{n3} in the form p + qi.",
                third,
                3,
                c_d,
                [f"{name}^{n3} = {name}^{n2} × {name} = {_ctext(*third)}."],
                follow_through=ft_c,
            ),
        ],
    )
