"""Generates figure questions on graphs that model a practical situation (Year 11-12).

* ``ey_context_line`` (ACMGM042, ``senior-graph-slope-intercept-meaning``): a drawn straight-line graph of a total
  cost, fare or amount against a count of hours, kilometres, GB or minutes. Easy reads the slope as a rate with its
  units, medium says what the vertical intercept means, hard picks the statement that gets both the rate and the fixed
  amount right, and the exam chain (``ey_context_line_exam``) reads the slope, then the intercept, then uses the model
  to predict a value and to find how many units give a stated amount. A decreasing line (a tank emptying, a battery
  draining) appears from the medium tier.
* ``ey_position_tangent`` (ACMMM094, ``senior-position-time-tangent-velocity``): a smooth position-time curve of a cart
  on a straight track. Easy and medium read the velocity as the slope of the drawn tangent (forwards, then backwards:
  a negative slope), hard asks when the cart is at rest or for how long it moves backwards, and the exam chain
  (``ey_position_tangent_exam``) reads the velocity at the tangent, the two times at rest, the time moving backwards
  and the average velocity over that time.

Every value read sits on a gridline crossing, and each reader recovers the numbers from the drawn line or curve (the
line's two points, or the curve's samples) rather than from the stored answer.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction as F

from .. import figures
from ..core import attempts, check, generator
from ._ey_senior_kit import curve_map, frac_text, mcq, near, num_part, retry_short, sample

_TAILS = (
    "Read the values from the grid.",
    "Every value you need sits on a grid crossing.",
    "Work through the parts in order.",
)

# =============================================================================================== a modelled line (ACMGM042)

CONTEXT = "ey_context_line"
CONTEXT_EXAM = "ey_context_line_exam"

_GROW = [
    dict(
        id="plumber",
        intro="A plumber charges a call-out fee plus an hourly rate. The graph shows the total cost of a job against the hours worked.",
        x="Time (hours)",
        y="Cost ($)",
        xs="hour",
        xp="hours",
        yname="cost",
        money=True,
        fixed="call-out fee",
        verb="adds",
    ),
    dict(
        id="taxi",
        intro="A taxi fare is a flat fee plus a charge for each kilometre. The graph shows the fare against the distance travelled.",
        x="Distance (km)",
        y="Fare ($)",
        xs="kilometre",
        xp="kilometres",
        yname="fare",
        money=True,
        fixed="flat fee",
        verb="adds",
    ),
    dict(
        id="phone",
        intro="A phone plan has a monthly fee plus a charge for each GB of extra data. The graph shows the monthly bill against the data used.",
        x="Data used (GB)",
        y="Bill ($)",
        xs="GB",
        xp="GB",
        yname="bill",
        money=True,
        fixed="monthly fee",
        verb="adds",
    ),
    dict(
        id="hire",
        intro="A hire company charges a booking fee plus a daily rate. The graph shows the total cost against the days hired.",
        x="Time (days)",
        y="Hire cost ($)",
        xs="day",
        xp="days",
        yname="hire cost",
        money=True,
        fixed="booking fee",
        verb="adds",
    ),
]
_DECAY = [
    dict(
        id="tank",
        intro="Water is pumped out of a tank at a steady rate. The graph shows the water left in the tank against the time.",
        x="Time (minutes)",
        y="Water (litres)",
        xs="minute",
        xp="minutes",
        yname="amount of water",
        money=False,
        unit="litres",
        fixed="amount of water in the tank at the start",
        verb="falls by",
    ),
    dict(
        id="battery",
        intro="A phone battery drains at a steady rate. The graph shows the charge left against the time.",
        x="Time (hours)",
        y="Charge (%)",
        xs="hour",
        xp="hours",
        yname="charge",
        money=False,
        unit="%",
        fixed="charge at the start",
        verb="falls by",
    ),
]
_BY_ID = {c["id"]: c for c in _GROW + _DECAY}
_BY_YLABEL = {c["y"]: c for c in _GROW + _DECAY}


def _val(ctx: dict, v) -> str:
    """A quantity of this situation: $12, 12 litres or 12%."""
    one = F(v) in (1, -1)
    v = frac_text(v)
    if ctx["money"]:
        return f"${v}" if not v.startswith("-") else f"-${v[1:]}"
    if ctx["unit"] == "%":
        return f"{v}%"
    return f"{v} {ctx['unit'][:-1] if one else ctx['unit']}"


def _rate(ctx: dict, m) -> str:
    return f"{_val(ctx, m)} per {ctx['xs']}"


def _rate_phrase(ctx: dict, v, up: bool) -> str:
    """The rate as the options say it: 'rises by $4 per hour' or 'falls by 5 litres per minute'."""
    return f"{'rises' if up else 'falls'} by {_rate(ctx, v)}"


def _draw_line(rng, tier: str):
    """(ctx, c, m, X, Y, s): intercept c, slope m (negative for a falling line), the window and the grid step s."""
    for _ in attempts(500, "a context line whose points sit on the gridlines"):
        decay = tier != "easy" and rng.random() < 0.4
        ctx = rng.choice(_DECAY if decay else _GROW)
        if ctx["id"] == "battery":
            s, c, mi = 10, 100, rng.choice((1, 2))
            if tier == "exam":
                # the exam battery drains at any of 5, 10, 15 or 20 percent an hour (the grid steps with it)
                s = rng.choice((5, 10))
                mi = rng.randint(1, 2 if s == 10 else 4)
            m = -s * mi
            X, Y = 10, 100
        elif decay:
            s = rng.choice((1, 2, 5))
            mi = rng.randint(1, 4)
            T = rng.randint(3, 8)
            m, c = -s * mi, s * mi * T
            X, Y = min(10, T + rng.randint(0, 2)), c
        else:
            s = rng.choice((1, 2, 5, 10))
            ci, mi = rng.randint(1, 5), rng.randint(1, 4)
            c, m = s * ci, s * mi
            X = rng.choice((5, 6, 8, 10))
            Y = c + m * X
            if ci == mi:
                continue
        if Y // s > 20 or (decay and c // s > 20):
            continue
        x1, x2 = _marks(rng, X, m, c)
        if x1 is None:
            continue
        return ctx, c, m, X, Y, s, (x1, x2)
    raise AssertionError("unreachable")


def _marks(rng, X, m, c):
    """Two integer x-values on the line, away from the vertical axis, with the line still above zero."""
    xs = [x for x in range(2, X + 1) if c + m * x >= 0]
    if len(xs) < 2:
        return None, None
    a, b = sorted(rng.sample(xs, 2))
    return a, b


def _line_figure(ctx, c, m, X, Y, s, marks):
    x1 = marks[0]
    pts = [{"at": [x, c + m * x]} for x in marks]
    return figures.cartesian(
        x_range=(0, X),
        y_range=(0, Y),
        alt=f"a straight-line graph of the {ctx['yname']} against the {ctx['xp']}",
        x_grid=1,
        y_grid=s,
        equal_scale=False,
        x_label=ctx["x"],
        y_label=ctx["y"],
        points=pts,
        lines=[{"through": [[0, c], [x1, c + m * x1]]}],
    )


def _intercept_sentence(ctx, c) -> str:
    return f"The {ctx['fixed']} is {_val(ctx, c)}"


def _intercept_wrong(ctx, c, m, y_end) -> list[str]:
    if m > 0:
        return [
            f"Each {ctx['xs']} adds {_val(ctx, c)}",
            _intercept_sentence(ctx, abs(m)),
            f"The {ctx['yname']} after the first {ctx['xs']} is {_val(ctx, c)}",
        ]
    return [
        f"The {ctx['yname']} falls by {_val(ctx, c)} each {ctx['xs']}",
        _intercept_sentence(ctx, abs(m)),
        f"The {ctx['yname']} left after the first {ctx['xs']} is {_val(ctx, c)}",
    ]


def _statement(ctx, c, m, fixed=None, step=None, rising=None) -> str:
    fixed = c if fixed is None else fixed
    step = abs(m) if step is None else step
    up = (m > 0) if rising is None else rising
    verb = "adds" if up else "falls by"
    tail = (
        f"and each extra {ctx['xs']} {verb} {_val(ctx, step)}"
        if up
        else f"and it falls by {_val(ctx, step)} each {ctx['xs']}"
    )
    return f"{_intercept_sentence(ctx, fixed)}, {tail}"


@generator(
    CONTEXT,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_context_line(rng, tier="easy"):
    """Slope with units, meaning of the intercept, or a statement that gets both right, from a drawn line."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    ctx, c, m, X, Y, s, marks = _draw_line(rng, tier)
    fig = _line_figure(ctx, c, m, X, Y, s, marks)
    y_end = c + m * marks[-1]
    if tier == "easy":
        text = f"{ctx['intro']} {rng.choice(('What does the slope of the line tell you?', 'What rate does the slope of the line show?'))}"
        up = m > 0
        key = _rate_phrase(ctx, abs(m), up)
        wrong = [
            _rate_phrase(ctx, abs(m), not up),
            _rate_phrase(ctx, c, up),
            _rate_phrase(ctx, y_end, up),
            _rate_phrase(ctx, abs(c + m), up),
        ]
        working = [
            f"The line passes through (0, {c}) and ({marks[0]}, {c + m * marks[0]}).",
            f"Slope = ({c + m * marks[0]} - {c}) / ({marks[0]} - 0) = {m}, so it {key}.",
        ]
    elif tier == "medium":
        text = f"{ctx['intro']} {rng.choice(('What does the point where the line meets the vertical axis tell you?', 'What does the vertical intercept of the line tell you?'))}"
        key = _intercept_sentence(ctx, c)
        wrong = _intercept_wrong(ctx, c, m, y_end)
        working = [
            f"The line meets the vertical axis at (0, {c}): the value when no {ctx['xp']} have passed.",
            f"{key}.",
        ]
    else:
        text = f"{ctx['intro']} {rng.choice(('Which statement about the situation is correct?', 'Which statement describes the line correctly?'))}"
        key = _statement(ctx, c, m)
        wrong = [
            _statement(ctx, c, m, fixed=abs(m), step=c),
            _statement(ctx, c, m, rising=not (m > 0)),
            _statement(ctx, c, m, fixed=c + m),
            _statement(ctx, c, m, step=c + abs(m)),
        ]
        working = [
            f"The line meets the vertical axis at {c}, so the {ctx['fixed']} is {_val(ctx, c)}.",
            f"It changes by {m} for each {ctx['xs']}, so the {ctx['yname']} {_rate_phrase(ctx, abs(m), m > 0)}.",
        ]
    distractors = mcq(key, wrong)
    q = type("Q", (), {"figure": fig, "text": text})
    check(_read_context(q) == key, "the context reader disagrees with the key")
    figures.assert_no_leak(fig, [key, *distractors])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _line_cm(fig: dict) -> tuple[int, int]:
    """Intercept and slope read from the drawn line, checked against the two marked points."""
    (x0, y0), (x1, y1) = fig["lines"][0]["through"]
    check(x0 == 0 and x1 > 0, "the line must start on the vertical axis")
    slope = F(y1 - y0, x1 - x0)
    check(slope.denominator == 1, "the slope is not a whole number")
    m, c = int(slope), int(y0)
    for p in fig["points"]:
        check(p["at"][1] == c + m * p["at"][0], "a marked point is off the line")
    return c, m


def _read_context(question) -> str:
    fig, text = question.figure, question.text
    ctx = _BY_YLABEL[fig["y_label"]]
    c, m = _line_cm(fig)
    if "slope of the line" in text:
        return _rate_phrase(ctx, abs(m), m > 0)
    if "meets the vertical axis" in text or "vertical intercept" in text:
        return _intercept_sentence(ctx, c)
    return _statement(ctx, c, m)


figures.register_reader(CONTEXT, _read_context)


@generator(CONTEXT_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_context_line_exam(rng, tier="exam"):
    """Exam chain: the slope, the intercept, a predicted value and the count that gives a stated amount."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a line whose prediction and inverse are whole numbers"):
        ctx, c, m, X, Y, s, marks = _draw_line(rng, "exam")
        if m > 0:
            n_pred = X + rng.randint(2, 6)
            n_goal = n_pred + rng.randint(2, 5)
        else:
            t_zero = c // -m
            options = [n for n in range(2, t_zero) if n not in marks]
            if len(options) < 2:
                continue
            n_pred, n_goal = sorted(rng.sample(options, 2))
        target = c + m * n_goal
        if target > 1 and n_pred != n_goal:
            break
    fig = _line_figure(ctx, c, m, X, Y, s, marks)
    value = c + m * n_pred
    text = f"{ctx['intro']} The graph is a straight line."
    word = ctx["yname"]
    parts = [
        num_part(
            f"Find the slope of the line: the change in {_unit_of(ctx)} for each extra {ctx['xs']}.",
            m,
            [-m, c, c + m],
            working=[f"The line passes through (0, {c}) and ({marks[0]}, {c + m * marks[0]}), so the slope is {m}."],
        ),
        num_part(
            f"Find the vertical intercept of the line, in {_unit_of(ctx)}.",
            c,
            [m, c + m, abs(m)],
            working=[f"The line meets the vertical axis at {c}."],
        ),
        num_part(
            f"Use the line to predict the {word} after {n_pred} {ctx['xp']}, in {_unit_of(ctx)}.",
            value,
            [m * n_pred, c * n_pred, c + m * (n_pred + 1)],
            follow=f"part_b+part_a*{n_pred}",
            working=[f"{word} = {c} + ({m}) x {n_pred} = {value}."],
        ),
        num_part(
            f"After how many {ctx['xp']} is the {word} {target} {_unit_of(ctx)}?",
            n_goal,
            [F(target, m) if m else 0, F(target - c, abs(m)) + 1, n_goal - 1],
            follow=f"({target}-part_b)/part_a",
            working=[f"{target} = {c} + ({m}) x n, so n = ({target} - {c}) / {m} = {n_goal}."],
        ),
    ]
    figures.assert_no_leak(fig, [str(m), str(c)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _unit_of(ctx: dict) -> str:
    return "dollars" if ctx["money"] else ("percent" if ctx["unit"] == "%" else ctx["unit"])


def _read_context_exam(question) -> str:
    c, m = _line_cm(question.figure)
    ctx = _BY_YLABEL[question.figure["y_label"]]
    del ctx
    prompts = _prompts(question)
    n_pred = int(re.search(r"after (\d+) \w+, in", prompts).group(1))
    target = int(re.search(r"(\d+) (?:dollars|percent|litres)\?", prompts).group(1))
    return f"(a) {m}; (b) {c}; (c) {c + m * n_pred}; (d) {(target - c) // m}"


def _prompts(question) -> str:
    return " ".join(p.prompt for p in question.parts)


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)

# =============================================================================================== position-time tangent (ACMMM094)

TANGENT = "ey_position_tangent"
TANGENT_EXAM = "ey_position_tangent_exam"

# s(t) = a (t^3/3 - (t1 + t2) t^2/2 + t1 t2 t) + D with a = +-3 and t1 + t2 even: whole-number positions at whole
# times, s'(t) = a (t - t1)(t - t2) whole numbers, and the cart is at rest at t1 and t2.
_RESTS = ((1, 3), (1, 5), (2, 4), (3, 5))


def _pos(a: int, t1: int, t2: int, d: int):
    return lambda t: a * (t**3 / 3 - (t1 + t2) * t**2 / 2 + t1 * t2 * t) + d


def _vel(a: int, t1: int, t2: int, t: int) -> int:
    return a * (t - t1) * (t - t2)


def _draw_curve(rng, a_choices=(3, -3)):
    for _ in attempts(500, "a position-time curve that fits a 34-line window"):
        t1, t2 = rng.choice(_RESTS)
        a = rng.choice(a_choices)
        d = rng.randint(-3, 3)
        T = t2 + 1
        s = _pos(a, t1, t2, d)
        vals = [round(s(t)) for t in range(T + 1)]
        # the turning points are whole times, so the extremes are among the whole-time values
        lo, hi = min(vals) - 1, max(vals) + 1
        if hi - lo <= 34:
            return a, t1, t2, d, T, lo, hi
    raise AssertionError("unreachable")


_EXAM_GRIDS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25)
_EXAM_RESTS = tuple((u, v) for u in range(1, 8) for v in range(u + 1, 9) if v - u <= 6)


@functools.cache
def _exam_curve_options():
    """Every (t1, t2, g, k, T) whose scaled curve fits a 34-square window and has a whole average velocity."""
    out = []
    for t1, t2 in _EXAM_RESTS:
        for g in _EXAM_GRIDS:
            for k in (1, 2, 3, 4):
                if (t1 + t2) % 2 and k % 2:
                    continue
                s = _pos(3 * g * k, t1, t2, 0)
                if (round(s(t2)) - round(s(t1))) % (t2 - t1):
                    continue
                for T in (t2 + 1, t2 + 2):
                    vals = [round(s(t)) for t in range(T + 1)]
                    if (-(-max(vals) // g) - min(vals) // g) + 2 <= 34:
                        out.append((t1, t2, g, k, T))
    return tuple(out)


def _draw_curve_exam(rng):
    """(a, t1, t2, d, T, lo, hi, g): the exam curve is the tier-1 shape scaled by a whole factor, drawn on a grid of g.

    a = 3 g k keeps every position a multiple of g at whole times when t1 + t2 is even, and so does an even k for any
    pair of rests (a = 6 g j gives 2 j g (t^3 - ...) integers); the window is a whole number of grid squares (at most
    34) and may run one or two seconds past the second rest. Drawn uniformly from every shape that fits."""
    t1, t2, g, k, T = rng.choice(_exam_curve_options())
    a, d = 3 * g * k, g * rng.randint(-3, 3)
    s = _pos(a, t1, t2, d)
    vals = [round(s(t)) for t in range(T + 1)]
    lo, hi = (min(vals) // g - 1) * g, (-(-max(vals) // g) + 1) * g
    return a, t1, t2, d, T, lo, hi, g


def _curve_figure(a, t1, t2, d, T, lo, hi, tangent_at=None, g=1):
    s = _pos(a, t1, t2, d)
    pts, lines = [], []
    if tangent_at is not None:
        t0 = tangent_at
        s0, v = round(s(t0)), _vel(a, t1, t2, t0)
        t_other = t0 + 1 if t0 + 1 <= T and lo <= s0 + v <= hi else t0 - 1
        pts.append({"at": [t0, s0], "label": "T"})
        lines.append({"through": [[t0, s0], [t_other, s0 + (t_other - t0) * v]], "label": "tangent"})
    return figures.cartesian(
        x_range=(0, T),
        y_range=(lo, hi),
        alt="a position-time curve of a cart on a straight track",
        x_grid=1,
        y_grid=g,
        equal_scale=False,
        x_label="Time t (s)",
        y_label="Position s (m)",
        points=pts,
        lines=lines,
        curves=[sample(s, 0, T, lo, hi)],
    )


def _tangent_ok(a, t1, t2, d, T, lo, hi, t0, want_sign) -> bool:
    s = _pos(a, t1, t2, d)
    s0, v = round(s(t0)), _vel(a, t1, t2, t0)
    if v == 0 or (v > 0) != (want_sign > 0):
        return False
    inside = [t for t in (t0 - 1, t0 + 1) if 0 <= t <= T and lo <= s0 + (t - t0) * v <= hi]
    return bool(inside)


_CART = "The graph shows the position s metres of a cart on a straight track, t seconds after it starts."


def _reader_curve(fig):
    """The sampled curve as ``{Fraction(t): s}`` and its local extremes (the times of rest)."""
    cm = curve_map(fig["curves"][0])
    ts = sorted(cm)
    rests = [
        t
        for t0, t, t2 in zip(ts, ts[1:], ts[2:])
        if (cm[t] > cm[t0] and cm[t] > cm[t2]) or (cm[t] < cm[t0] and cm[t] < cm[t2])
    ]
    return cm, rests


def _read_tangent_velocity(fig) -> int:
    (x0, y0), (x1, y1) = fig["lines"][0]["through"]
    v = F(y1 - y0, x1 - x0)
    check(v.denominator == 1, "the tangent's slope is not a whole number")
    t0, s0 = fig["points"][0]["at"]
    cm = curve_map(fig["curves"][0])
    h = F(1, 12)
    # the drawn line must be the tangent: the central difference of the curve at the point has the same slope
    est = (cm[F(t0) + h] - cm[F(t0) - h]) / float(2 * h) if F(t0) - h in cm and F(t0) + h in cm else None
    if est is not None:
        # (the central difference of a cubic with leading coefficient a/3 is out by a h^2 / 3: a steep scaled curve
        # needs a relative allowance, and |v| >= a at any whole time that is not a rest)
        check(
            abs(est - float(v)) < 0.1 + abs(float(v)) / 100,
            "the drawn line is not the tangent to the curve at the marked point",
        )
    return int(v)


@generator(
    TANGENT,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_position_tangent(rng, tier="easy"):
    """Velocity from a tangent (forwards, then backwards), or the times and length of backwards movement."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    if tier in ("easy", "medium"):
        want = 1 if tier == "easy" else -1
        for _ in attempts(500, "a curve with a tangent of the wanted direction"):
            a, t1, t2, d, T, lo, hi = _draw_curve(rng)
            t0s = [t for t in range(0, T + 1) if _tangent_ok(a, t1, t2, d, T, lo, hi, t, want)]
            if t0s:
                break
        t0 = rng.choice(t0s)
        s = _pos(a, t1, t2, d)
        s0, v = round(s(t0)), _vel(a, t1, t2, t0)
        fig = _curve_figure(a, t1, t2, d, T, lo, hi, tangent_at=t0)
        extra = "" if tier == "easy" else " A negative velocity means the cart is moving backwards."
        text = f"{_CART} The line touches the curve at T. {rng.choice(('What is the velocity of the cart at T, in metres per second?', 'Find the velocity of the cart at T, in metres per second.', 'How fast, and in which direction, is the cart moving at T? Give the velocity in metres per second.'))}{extra}"
        sec = round(s(t0 + 1)) - s0 if t0 < T else s0 - round(s(t0 - 1))
        wrong = [-v, F(s0, t0) if t0 else s0, sec, s0]
        key = str(v)
        working = [
            f"The tangent at T passes through ({t0}, {s0}) and ({t0 + 1}, {s0 + v}).",
            f"Velocity = slope of the tangent = {v} metres per second"
            + (", so the cart is moving backwards." if v < 0 else "."),
        ]
        distractors = mcq_number(key, wrong)
        figures.assert_no_leak(fig, [key, *distractors])
        check(str(_read_tangent_velocity(fig)) == key, "the tangent reader disagrees with the key")
        return _number_question(text, key, distractors, working, fig)
    mode = rng.choice(("rest", "backwards"))
    a, t1, t2, d, T, lo, hi = _draw_curve(rng)
    fig = _curve_figure(a, t1, t2, d, T, lo, hi)
    if mode == "rest":
        text = f"{_CART} {rng.choice(('The cart is at rest twice. At what time, in seconds, is it at rest the second time?', 'The cart stops twice. At what time, in seconds, does it stop the second time?', 'At what time, in seconds, is the cart at rest for the second time?'))}"
        key, wrong = str(t2), [t1, T, (t1 + t2) // 2 if (t1 + t2) % 2 == 0 else t1 + 1, round(_pos(a, t1, t2, d)(t2))]
        working = [
            f"At rest means the slope of the curve is zero: the curve turns at t = {t1} and t = {t2}.",
            f"The second time is {t2} seconds.",
        ]
    else:
        text = f"{_CART} For how many of the seconds from t = 0 to t = {T} is the cart moving backwards (s getting smaller)?"
        back = (t2 - t1) if a > 0 else (t1 + (T - t2))
        key, wrong = str(back), [T - back, t2, t1, t2 - t1 + 1]
        working = [
            f"The cart turns at t = {t1} and t = {t2}.",
            f"s is getting smaller {'between' if a > 0 else 'before the first turn and after the second turn, '}"
            + (f"t = {t1} and t = {t2}: {back} seconds." if a > 0 else f"so {t1} + {T - t2} = {back} seconds."),
        ]
    distractors = mcq_number(key, wrong)
    figures.assert_no_leak(fig, [key, *distractors])
    q = type("Q", (), {"figure": fig, "text": text})
    check(_read_tangent(q) == key, "the position-time reader disagrees with the key")
    return _number_question(text, key, distractors, working, fig)


def mcq_number(key: str, wrong) -> list[str]:
    """Whole-number distractors (rounded) for a number answer."""
    from ..core import round_half_up

    return mcq(key, [str(round_half_up(F(w), 0)) for w in wrong])


def _number_question(text, key, distractors, working, fig):
    return dict(
        text=text,
        answer=int(key),
        answer_text=key,
        answer_latex=key,
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_tangent(question) -> str:
    fig, text = question.figure, question.text
    if fig.get("lines"):
        return str(_read_tangent_velocity(fig))
    cm, rests = _reader_curve(fig)
    check(len(rests) == 2, "the curve must turn exactly twice")
    r1, r2 = (int(t) for t in rests)
    T = fig["x_range"][1]
    if "second time" in text:
        return str(r2)
    descending = sum(1 for t0, t in zip(sorted(cm), sorted(cm)[1:]) if cm[t] < cm[t0])
    return str(round(descending / 12))


figures.register_reader(TANGENT, _read_tangent)


@generator(TANGENT_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_position_tangent_exam(rng, tier="exam"):
    """Exam chain: the velocity at T, both times at rest, the time moving backwards, the average velocity then."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a curve with a tangent and whole average velocity"):
        a, t1, t2, d, T, lo, hi, g = _draw_curve_exam(rng)
        s = _pos(a, t1, t2, d)
        t0s = [
            t
            for t in range(0, T + 1)
            if t not in (t1, t2) and _tangent_ok(a, t1, t2, d, T, lo, hi, t, 1 if _vel(a, t1, t2, t) > 0 else -1)
        ]
        s1, s2 = round(s(t1)), round(s(t2))
        if t0s and (s2 - s1) % (t2 - t1) == 0:
            break
    t0 = rng.choice(t0s)
    v0 = _vel(a, t1, t2, t0)
    s0 = round(s(t0))
    fig = _curve_figure(a, t1, t2, d, T, lo, hi, tangent_at=t0, g=g)
    avg = F(s2 - s1, t2 - t1)
    text = f"{_CART} The line touches the curve at T. The cart is at rest twice between t = 0 and t = {T}."
    parts = [
        num_part(
            "Find the velocity of the cart at T, in metres per second.",
            v0,
            [-v0, s0, F(s0, t0) if t0 else s0],
            working=[f"The tangent at T has slope {v0}, so the velocity is {v0} metres per second."],
        ),
        num_part(
            "At what time is the cart first at rest?",
            t1,
            [t2, T, s1],
            working=[f"The curve turns where the slope is zero: first at t = {t1}."],
        ),
        num_part(
            "At what time is the cart at rest the second time?",
            t2,
            [t1, T, s2],
            working=[f"The curve turns again at t = {t2}."],
        ),
        num_part(
            "Hence find the number of seconds the cart moves backwards.",
            t2 - t1,
            [t2 + t1, T - (t2 - t1), t2],
            follow="part_c-part_b",
            working=[f"The cart moves backwards between the two rests: {t2} - {t1} = {_secs(t2 - t1)}."],
        ),
        num_part(
            "Hence find the average velocity of the cart over that time, in metres per second.",
            avg,
            [-avg, s2 - s1, F(s2, t2)],
            follow=f"({s2}-({s1}))/part_d",
            working=[f"s goes from {s1} to {s2} in {_secs(t2 - t1)}: ({s2} - ({s1})) / {t2 - t1} = {frac_text(avg)}."],
        ),
    ]
    figures.assert_no_leak(fig, [str(v0), str(t1), str(t2)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _secs(n: int) -> str:
    return f"{n} second" if n == 1 else f"{n} seconds"


def _read_tangent_exam(question) -> str:
    fig = question.figure
    cm, rests = _reader_curve(fig)
    check(len(rests) == 2, "the curve must turn exactly twice")
    r1, r2 = (int(t) for t in rests)
    v = _read_tangent_velocity(fig)
    avg = F(round(cm[F(r2)]) - round(cm[F(r1)]), r2 - r1)
    check(avg.denominator == 1, "the average velocity is not whole")
    return f"(a) {v}; (b) {r1}; (c) {r2}; (d) {r2 - r1}; (e) {int(avg)}"


figures.register_reader(TANGENT_EXAM, _read_tangent_exam)

_ = near
