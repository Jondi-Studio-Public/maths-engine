"""Generates `r3h_euler_nonlinear` questions (easy, medium, hard) and the exam chain `exd_x_r3h_euler_nonlinear`.

``euler-method-nonlinear-table`` (``differential-equations``, answer ``table``; the exam chain's parts are numbers): Euler's method
``y(n+1) = y(n) + h f(x(n), y(n))`` applied to a first-order equation whose slope depends on ``y`` non-linearly
(``dy/dx = xy``, ``y^2``, ``x/y``, ``xy^2``, ``y(1 - y)``), so each slope has to be recomputed at the new estimate.
The linear one-step skill is `explicit-euler-linear-ode`.

* easy: two steps, completing the column of estimates in a table.
* medium: three steps, completing the estimates and the slopes in a table.
* hard: four steps, then the exact value and the error against the exact solution, which is given (table, 4 decimal
  places).
* exam (``exd_x_r3h_euler_nonlinear``): (a) the estimate after one step, (b) the slope at the new point (follows
  through from (a)), (c) the estimate after two steps (follows through), (d) the error against the exact solution
  at that point (follows through).

All Euler arithmetic is exact (``Fraction``); table values are chosen so that every entry has at most four decimal
places and is stated exactly. The exact solutions are checked by differentiating them back into the equation.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, rounds_unambiguously
from ._f10_table_kit import build_answer, grid_text
from ._r3h_kit import Redraw, build_with_redraws, number_part, plain, say_marked
from ._senior_table_kit import cell_text, finish_wrong, rounds_clear

_F = Fraction
_x, _y = sympy.symbols("x y")


class _Family:
    def __init__(self, text, slope, exact, exact_text, ode):
        self.text, self.slope, self.exact, self.exact_text, self.ode = text, slope, exact, exact_text, ode


def _exact_xy(x0, y0):
    return y0 * sympy.exp((_x**2 - x0**2) / 2)


def _exact_y2(x0, y0):
    return 1 / (1 / y0 - (_x - x0))


def _exact_xy_over(x0, y0):
    return sympy.sqrt(y0**2 + _x**2 - x0**2)


def _exact_xy2(x0, y0):
    return 1 / (1 / y0 - (_x**2 - x0**2) / 2)


def _exact_logistic(x0, y0):
    return 1 / (1 + (1 / y0 - 1) * sympy.exp(-(_x - x0)))


def _exact_2xy(x0, y0):
    return y0 * sympy.exp(_x**2 - x0**2)


def _exact_x2y(x0, y0):
    return y0 * sympy.exp((_x**3 - x0**3) / 3)


def _exact_decay(x0, y0):
    return y0 * sympy.exp(-(_x**2 - x0**2) / 2)


_FAMILIES = {
    "2xy": _Family("2x y", lambda x, y: 2 * x * y, _exact_2xy, True, 2 * _x * _y),
    "x2y": _Family("x^2 y", lambda x, y: x * x * y, _exact_x2y, True, _x**2 * _y),
    "-xy": _Family("-x y", lambda x, y: -x * y, _exact_decay, True, -_x * _y),
    "xy": _Family("x y", lambda x, y: x * y, _exact_xy, True, _x * _y),
    "y2": _Family("y^2", lambda x, y: y * y, _exact_y2, True, _y**2),
    "x/y": _Family("x/y", lambda x, y: x / y, _exact_xy_over, True, _x / _y),
    "xy2": _Family("x y^2", lambda x, y: x * y * y, _exact_xy2, True, _x * _y**2),
    "logistic": _Family("y(1 - y)", lambda x, y: y * (1 - y), _exact_logistic, True, _y * (1 - _y)),
    "x+y2": _Family("x + y^2", lambda x, y: x + y * y, None, False, _x + _y**2),
}


def _euler(fam, x0, y0, h, n, variant="correct"):
    """[(x, y, slope)] for steps 0..n; `variant` is a named wrong method."""
    rows = []
    x, y = _F(x0), _F(y0)
    first_slope = None
    for i in range(n + 1):
        s = fam.slope(x, y)
        if first_slope is None:
            first_slope = s
        rows.append((x, y, s))
        if i == n:
            break
        if variant == "correct":
            y = y + h * s
        elif variant == "slope_at_next_x":
            y = y + h * fam.slope(x + h, y)
        elif variant == "no_step":
            y = y + s
        elif variant == "restart":
            y = _F(y0) + h * s
        elif variant == "slope_frozen":
            y = y + h * first_slope
        x = x + h
    return rows


def _places(value: Fraction) -> int:
    """Decimal places needed to write `value` exactly (more than 4 means 'not clean')."""
    d = value.denominator
    for places in range(0, 7):
        if (10**places) % d == 0:
            return places
    return 9


def _draw(rng, tier):
    fam_names = {
        "easy": ["xy", "y2", "x+y2"],
        "medium": ["xy", "y2", "x/y", "xy2", "x+y2", "logistic"],
        "hard": ["xy", "y2", "x/y", "xy2", "logistic", "2xy", "x2y", "-xy"],
        "exam": ["xy", "y2", "x/y", "xy2", "logistic"],
    }[tier]
    steps = {"easy": 2, "medium": 3, "hard": 4, "exam": 2}[tier]
    for _ in attempts(3000, "an Euler example with clean arithmetic"):
        name = rng.choice(fam_names)
        fam = _FAMILIES[name]
        x0 = _F(rng.choice([0, 1, 2] if tier == "hard" else [0, 1]))
        h = rng.choice(
            [_F(1, 2), _F(1, 4), _F(1, 5), _F(1, 10)] if tier == "hard" else [_F(1, 2), _F(1, 4), _F(1, 5), _F(3, 10)]
        )
        y0 = rng.choice(
            [_F(1), _F(2), _F(1, 2), _F(1, 4), _F(3), _F(3, 2), _F(2, 5), _F(4, 5)]
            + ([_F(1, 5), _F(3, 5), _F(5, 2)] if tier == "hard" else [])
        )
        if name == "logistic":
            y0 = rng.choice([_F(1, 2), _F(1, 4), _F(1, 5), _F(3, 4)])
        rows = _euler(fam, x0, y0, h, steps)
        values = [v for r in rows for v in r]
        if any(abs(v) > 60 or v.denominator > 10**7 for v in values) or any(r[2] == 0 for r in rows[1:]):
            continue
        if tier != "hard" and not all(rounds_clear(v, 3 if tier != "exam" else 4) for r in rows[1:] for v in r[1:]):
            continue
        # the first estimate must actually move (a zero opening slope makes the first step a giveaway)
        if rows[0][2] == 0:
            continue
        # keep the exact solution defined and well behaved on the interval
        if fam.exact is not None:
            exact = fam.exact(
                sympy.Rational(x0.numerator, x0.denominator), sympy.Rational(y0.numerator, y0.denominator)
            )
            xe = float(rows[-1][0])
            try:
                ev = complex(sympy.N(exact.subs(_x, xe), 20))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if abs(ev.imag) > 1e-12 or not (0.05 < abs(ev.real) < 60):
                continue
            # a blow-up (y' = y^2) before the last x makes the exact solution meaningless
            if name in ("y2",) and float(1 / y0 - (rows[-1][0] - x0)) < 0.2:
                continue
            if name == "xy2" and float(1 / y0 - (rows[-1][0] ** 2 - x0**2) / 2) < 0.2:
                continue
        return name, fam, x0, y0, h, steps, rows
    raise AssertionError("unreachable")


def _check_exact(name, fam, x0, y0):
    """The exact solution satisfies the equation and the starting value (differentiated back, at two points)."""
    exact = fam.exact(sympy.Rational(x0.numerator, x0.denominator), sympy.Rational(y0.numerator, y0.denominator))
    lhs = sympy.diff(exact, _x)
    rhs = fam.ode.subs(_y, exact)
    for p in (0.15 + float(x0), 0.3 + float(x0)):
        a, b = complex(sympy.N(lhs.subs(_x, p), 20)), complex(sympy.N(rhs.subs(_x, p), 20))
        check(abs(a - b) < 1e-9 * max(1, abs(a)), f"exact solution of {name} does not satisfy the equation")
    check(abs(complex(sympy.N(exact.subs(_x, float(x0)), 20)) - float(y0)) < 1e-12, "exact solution misses y(x0)")
    return exact


def _xtext(x: Fraction) -> str:
    return f"{float(x):g}"


def _ytext(y: Fraction) -> str:
    return f"{y.numerator}" if y.denominator == 1 else f"{float(y):g}"


def _start(fam, x0, y0, h) -> str:
    return f"Euler's method with step size h = {_xtext(h)} is used to estimate the solution of dy/dx = {fam.text} with y({_xtext(x0)}) = {_ytext(y0)}."


def _table(rows, with_slope: bool, dp: int, blank_slopes: set[int], blank_ys: set[int]):
    columns = ["Step", "x", "y"] + (["Slope"] if with_slope else [])
    full = []
    n = len(rows) - 1
    for i, (x, y, s) in enumerate(rows):
        row = [f"Step {i}", _xtext(x), cell_text(y, dp)]
        if with_slope:
            row.append(cell_text(s, dp) if i < n else "-")
        full.append(row)
    blanks = {(i, 2) for i in blank_ys}
    if with_slope:
        blanks |= {(i, 3) for i in blank_slopes}
    return build_answer(columns, full, blanks)


def _table_question(rng, tier):
    name, fam, x0, y0, h, n, rows = _draw(rng, tier)
    with_slope = tier == "medium"
    dp = 3
    blank_ys = set(range(1, n + 1))
    blank_slopes = set(range(1, n)) if with_slope else set()
    answer = _table(rows, with_slope, dp, blank_slopes, blank_ys)
    # build wrong tables: the y column is index 2 of the row, the slope is index 3 -> map to (x, y, s)
    edits = []
    for tag, variant in (
        ("r3h-euler-slope-at-next-point", "slope_at_next_x"),
        ("r3h-euler-step-size-not-applied", "no_step"),
        ("r3h-euler-restarts-from-initial-value", "restart"),
        ("r3h-euler-slope-not-updated", "slope_frozen"),
    ):
        wrong_rows = _euler(fam, x0, y0, h, n, variant)
        change = {}
        for key in answer["blanks"]:
            r, c = map(int, key.split(","))
            change[key] = wrong_rows[r][1] if c == 2 else wrong_rows[r][2]
        edits.append((tag, change))
    # a table where the learner uses y(x0) in every slope (the starting value never updated)
    answer_text, distractors, notes = finish_wrong(rng, answer, {}, want=3, whole=edits)
    keep = "Keep full accuracy in your working and round only the entries you write."
    if with_slope:
        ask = f"Complete the table, giving every entry to {dp} decimal places. {keep}"
    else:
        ask = f"Complete the table of estimates, giving each value of y to {dp} decimal places. {keep}"
    text = f"{_start(fam, x0, y0, h)} {ask} Fill in each ? with a number. {grid_text(answer)}"
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        working=[
            f"Step {i}: slope {cell_text(s, dp)}, so y moves to {cell_text(rows[i + 1][1], dp)}."
            for i, (_x_, _y_, s) in enumerate(rows[:-1])
        ],
        distractors=distractors,
        option_notes=notes,
        answer_kind="table",
    )


def _hard_question(rng):
    for _ in attempts(60, "a hard Euler table with enough distinct wrong tables"):
        try:
            return _hard_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")


def _hard_once(rng):
    for _ in attempts(500, "a hard Euler question with a clean error"):
        name, fam, x0, y0, h, n, rows = _draw(rng, "hard")
        exact = _check_exact(name, fam, x0, y0)
        xn = rows[-1][0]
        ex = float(sympy.N(exact.subs(_x, float(xn)), 30))
        err = ex - float(rows[-1][1])
        if (
            all(rounds_clear(v, 4) for r in rows[1:] for v in r[1:2])
            and rounds_unambiguously(ex, 4, 0.1)
            and rounds_unambiguously(err, 4, 0.1)
            and abs(err) > 0.004
        ):
            break
    dp = 4
    full = []
    for i, (x, y, _s) in enumerate(rows):
        last = i == n
        full.append(
            [
                f"Step {i}",
                _xtext(x),
                cell_text(y, dp),
                cell_text(Fraction(ex), dp) if last else "-",
                cell_text(Fraction(err), dp) if last else "-",
            ]
        )
    blanks = {(i, 2) for i in range(1, n + 1)} | {(n, 3), (n, 4)}
    answer = build_answer(["Step", "x", "Euler y", "Exact y", "Error"], full, blanks)
    edits = []
    for tag, variant in (
        ("r3h-euler-slope-at-next-point", "slope_at_next_x"),
        ("r3h-euler-restarts-from-initial-value", "restart"),
        ("r3h-euler-slope-not-updated", "slope_frozen"),
    ):
        wrong_rows = _euler(fam, x0, y0, h, n, variant)
        change = {f"{r},2": wrong_rows[r][1] for r in range(1, n + 1)}
        change[f"{n},4"] = Fraction(ex) - wrong_rows[n][1]
        edits.append((tag, change))
    reversed_error = {f"{n},4": -Fraction(err)}
    edits.append(("r3h-euler-error-reversed", reversed_error))
    edits.append(("r3h-euler-error-uses-previous-step", {f"{n},4": Fraction(ex) - rows[n - 1][1]}))
    answer_text, distractors, notes = finish_wrong(rng, answer, {}, want=3, whole=edits)
    exact_text = plain(sympy.simplify(exact))
    keep = "Keep full accuracy in your working and round only the entries you write."
    text = (
        f"{_start(fam, x0, y0, h)} The exact solution is y = {exact_text}. Complete the table of {n} steps, giving every "
        f"entry to {dp} decimal places; the error is the exact value minus the Euler estimate. {keep} "
        f"Fill in each ? with a number. {grid_text(answer)}"
    )
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_kind="table",
        distractors=distractors,
        option_notes=notes,
        working=[
            f"Euler estimate at x = {_xtext(xn)}: {float(rows[-1][1]):.4f}; exact value {ex:.4f}; error {err:.4f}."
        ],
    )


@generator(
    "r3h_euler_nonlinear",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_euler_nonlinear(rng: random.Random, tier="easy"):
    if tier == "hard":
        return _hard_question(rng)
    return _table_question(rng, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    name, fam, x0, y0, h, n, rows = _draw(rng, "exam")
    exact = _check_exact(name, fam, x0, y0)
    (xa, ya, sa), (xb, yb, sb), (xc, yc, _sc) = rows
    x2 = xc
    ex = float(sympy.N(exact.subs(_x, float(x2)), 30))
    err = ex - float(yc)
    if not (rounds_unambiguously(err, 4, 0.1) and abs(err) > 0.003):
        raise Redraw
    dp = 4
    stem, latex = say_marked(
        f"Euler's method with step size «h = {_xtext(h)}» is used on «dy/dx = {fam.text}» with «y({_xtext(x0)}) = {_ytext(y0)}». "
        f"The exact solution is «y = {plain(sympy.simplify(exact))}»."
    )
    x1 = _xtext(xb)
    h_t = _xtext(h)
    ft_slope = {
        "xy": f"{_F(xb)}*part_a",
        "y2": "part_a**2",
        "x/y": f"{_F(xb)}/part_a",
        "xy2": f"{_F(xb)}*part_a**2",
        "logistic": "part_a*(1 - part_a)",
    }[name]
    wrong_next = _euler(fam, x0, y0, h, 2, "slope_at_next_x")
    parts = [
        number_part(
            "Find the Euler estimate of «y({x1})», correct to 4 decimal places.",
            float(yb),
            dp,
            2,
            [f"y({x1}) = {_ytext(y0)} + {h_t} x {_ytext(sa)} = {float(yb):.4f}."],
            [
                ("r3h-euler-step-size-not-applied", float(ya + sa)),
                ("r3h-euler-slope-at-next-point", float(wrong_next[1][1])),
                ("r3h-euler-increment-reported-as-estimate", float(h * sa)),
                ("r3h-euler-step-size-not-applied", float(ya + h)),
            ],
            x1=x1,
        ),
        number_part(
            "Find the slope «dy/dx» given by the equation at «x = {x1}» and your estimate from part (a), correct to 4 decimal places.",
            float(sb),
            dp,
            1,
            [f"dy/dx = {fam.text} at ({x1}, {float(yb):.4f}) = {float(sb):.4f}."],
            [
                ("r3h-euler-slope-not-updated", float(sa)),
                ("r3h-euler-slope-uses-old-point", float(fam.slope(xb, ya))),
                ("r3h-euler-slope-uses-old-point", float(fam.slope(xa, yb))),
                ("r3h-euler-slope-uses-x-only", float(xb)),
            ],
            follow_through=ft_slope,
            x1=x1,
        ),
        number_part(
            "Hence find the Euler estimate of «y({x2})», correct to 4 decimal places.",
            float(yc),
            dp,
            2,
            [f"y({_xtext(x2)}) = {float(yb):.4f} + {h_t} x {float(sb):.4f} = {float(yc):.4f}."],
            [
                ("r3h-euler-restarts-from-initial-value", float(y0 + h * sb)),
                ("r3h-euler-slope-not-updated", float(yb + h * sa)),
                ("r3h-euler-step-size-not-applied", float(yb + sb)),
                ("r3h-euler-increment-reported-as-estimate", float(h * sb)),
            ],
            follow_through=f"part_a + {_F(h)}*part_b",
            x2=_xtext(x2),
        ),
        number_part(
            "Find the error in this estimate, the exact value of «y({x2})» minus the estimate, correct to 4 decimal places.",
            err,
            dp,
            1,
            [f"Exact y({_xtext(x2)}) = {ex:.4f}; error = {ex:.4f} - {float(yc):.4f} = {err:.4f}."],
            [
                ("r3h-euler-error-reversed", -err),
                ("r3h-euler-error-relative", err / ex * 100),
                ("r3h-euler-error-uses-previous-step", ex - float(yb)),
                ("r3h-euler-error-estimate-only", float(yc)),
            ],
            follow_through=f"{ex:.8f} - part_c",
            x2=_xtext(x2),
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_euler_nonlinear", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r3h_euler_nonlinear(rng, tier="exam"):
    """Exam tier: two Euler steps for a non-linear equation, the slope between them, and the error."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
