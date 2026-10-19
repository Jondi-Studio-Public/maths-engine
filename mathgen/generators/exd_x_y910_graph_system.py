"""Exam tier for `f10-graphical-inequalities-systems`: read one line, then solve it against a second.

Two lines are drawn. Line B's equation is given in the stem; line A's is not, and has to be read off the grid. The
learner reads A's gradient and y-intercept, then solves the pair of simultaneous equations for the crossing point,
which is at a whole or half number:

(a) the gradient of A (1 mark);
(b) the y-intercept of A (1 mark);
(c) the x-coordinate where A and B cross (2 marks, follow-through from (a) and (b));
(d) the y-coordinate where they cross (2 marks, follow-through from (a), (b) and (c)).

Both lines have whole-number gradients and intercepts. The crossing is exact in fractions and checked by
substituting into both equations.
"""

from __future__ import annotations

from fractions import Fraction as F

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, pick_distractors
from mathgen.generators.fg1_line_reading_build import eq_text
from mathgen.generators.pv1a_line_figure_build import line_figure, read_line
from mathgen.generators.y910_graph_inequality_build import _ALT, _as_value, _two_lines, _two_points


def _text(v: F) -> str:
    """A whole or half number the way a student writes it: 3, -2, 2.5."""
    v = F(v)
    return str(v.numerator) if v.denominator == 1 else f"{float(v):.1f}"


def _part(prompt, key: F, wrong, marks, working, low, high, follow_through=None):
    cands = []
    for value, note in wrong:
        value = F(value)
        if value.denominator in (1, 2):
            cands.append(Distractor(_text(value), note, _as_value(value)))
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=_as_value(key),
        answer_text=_text(key),
        answer_latex=latex_answer(_text(key)),
        marks=marks,
        sensible_range=(low, high),
        working=working,
        distractors=pick_distractors(_text(key), cands, 3, tolerance=GRADER_TOLERANCE),
    )
    if follow_through:
        out["follow_through"] = follow_through
    return out


@generator("exd_x_y910_graph_system", level=10, presets={"exam": dict(tier="exam")})
def exd_x_y910_graph_system(rng, tier="exam"):
    """Read line A, then find where it crosses line B (whose equation is given)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "two lines with non-zero intercepts"):
        win, (ma, ca), (mb, cb), (x, y) = _two_lines(rng, "half")
        if ca != 0 and cb != 0:
            break
    check(ma * x + ca == y == mb * x + cb, "the crossing is not on both lines")
    fig = line_figure(win, [_two_points(win, ma, ca)[0], _two_points(win, mb, cb)[0]], _ALT, labels=["A", "B"])
    rm, rc = read_line(fig, 0)
    check((rm, rc) == (ma, ca), "the drawn line does not match the keyed equation")
    text = (
        f"The lines A and B are drawn on the grid. The equation of line B is {eq_text(mb, cb)}. "
        f"Line A continues in both directions."
    )
    cb_i = int(cb)
    ft_x = f"({cb_i} - part_b)/(part_a - ({int(mb)}))"
    parts = [
        _part(
            "Find the gradient of line A.",
            ma,
            [
                (-ma, "gradient-sign-misread"),
                (1 / ma, "inverted-gradient"),
                (-1 / ma, "inverted-gradient"),
                (ma + 1, "misread-the-crossing-by-one-square"),
            ],
            1,
            [f"Line A rises {ma} for each 1 across, so the gradient is {ma}."],
            -10,
            10,
        ),
        _part(
            "Find the y-intercept of line A.",
            ca,
            [
                (-ca, "intercept-sign-error"),
                (-ca / ma, "gave-x-intercept-for-y-intercept"),
                (ma, "gradient-and-intercept-swapped"),
            ],
            1,
            [f"Line A meets the y-axis at {ca}, so its equation is {eq_text(ma, ca)}."],
            -10,
            10,
        ),
        _part(
            "Find the x-coordinate of the point where A and B cross.",
            x,
            [
                ((cb + ca) / (ma - mb), "sign-error-when-setting-equal"),
                ((ca - cb) / (ma - mb), "sign-error-when-setting-equal"),
                ((cb - ca) / (ma + mb) if ma + mb else 99, "sign-error-when-setting-equal"),
                (y, "x-and-y-swapped"),
            ],
            2,
            [
                f"At the crossing, {eq_text(ma, ca)[4:]} = {eq_text(mb, cb)[4:]}.",
                f"{ma - mb} x = {cb - ca}, so x = {x}.",
            ],
            -20,
            20,
            follow_through=ft_x,
        ),
        _part(
            "Find the y-coordinate of the point where A and B cross.",
            y,
            [
                (x, "x-and-y-swapped"),
                (ca, "y-replaced-by-the-intercept"),
                (ma * x, "intercept-left-out-of-the-substitution"),
                (mb * x + ca, "mixed-the-two-equations"),
            ],
            2,
            [f"Putting x = {x} into line A gives y = {y}. Check in line B: it also gives y = {mb * x + cb}."],
            -30,
            30,
            follow_through="part_a*part_c + part_b",
        ),
    ]
    figures.assert_no_leak(fig, [p["answer_text"] for p in parts])
    return dict(text=text, parts=parts, figure=fig)
