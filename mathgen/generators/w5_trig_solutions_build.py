"""All solutions of f(a(x - b)) = c for f one of sin, cos, tan, typed as a list of angles in degrees.

Topic `w5_trig_all_solutions` (skill `senior-find-all-solutions-of-mathrm-f-left-mathrm`, answer `list`) with the exam
chain `exd_x_w5_trig_all_solutions`.

* easy: f(x) = c over a stated 360 degree window (0 to 360, or -180 to 180), exact values at 30, 45 and 60 degrees.
* medium: f(x - b) = c or f(x + b) = c, so the quadrant solutions are shifted and then wrapped back into the window.
* hard: f(a(x - b)) = c with a = 2 or 3, where the window holds 2a solutions from several periods.
* exam: the solutions of f(x) = c, the smallest solution of f(2x) = c, and the solutions of f(2(x - b)) = c.

The solutions are built from the CAST/reference-angle rule, then re-derived a second way by scanning the window every
half degree for angles where f(a(x - b)) equals c (a scan that does not use the reference angle at all).
"""

from __future__ import annotations

import math

from ..core import check, generator, phrase
from ._w5_kit import TooFewDistractors, enough, list_cands, list_fields, number_cands, number_fields, part, retry_draw

TOPIC = "w5_trig_all_solutions"

_VALUE_TEXT = {
    "sin": {30: "1/2", 45: "√2/2", 60: "√3/2"},
    "cos": {30: "√3/2", 45: "√2/2", 60: "1/2"},
    "tan": {30: "1/√3", 45: "1", 60: "√3"},
}
_VALUE = {
    "sin": {30: 0.5, 45: math.sqrt(2) / 2, 60: math.sqrt(3) / 2},
    "cos": {30: math.sqrt(3) / 2, 45: math.sqrt(2) / 2, 60: 0.5},
    "tan": {30: 1 / math.sqrt(3), 45: 1.0, 60: math.sqrt(3)},
}
_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}


def _principal(func: str, theta: int, sign: int) -> list[int]:
    """The angles u in [0, 360) with f(u) = sign * f(theta) by the CAST rule (tan has period 180)."""
    if func == "sin":
        return [theta, 180 - theta] if sign > 0 else [180 + theta, 360 - theta]
    if func == "cos":
        return [theta, 360 - theta] if sign > 0 else [180 - theta, 180 + theta]
    return [theta, theta + 180] if sign > 0 else [180 - theta, 360 - theta]


def _solutions(func: str, theta: int, sign: int, a: int, b: int, lo: int, hi: int) -> list[int]:
    """Every x in (lo, hi) with f(a(x - b)) = sign * f(theta): u = a(x - b) takes the principal values plus whole
    periods, and x = b + u / a."""
    out = set()
    for k in range(-4 * a - 2, 4 * a + 3):
        for u in _principal(func, theta, sign):
            x2 = a * b + u + 360 * k  # = a * x
            if x2 % a:
                raise TooFewDistractors("a half-degree solution")
            x = x2 // a
            if lo < x < hi:
                out.add(x)
            if x == lo or x == hi:
                raise TooFewDistractors("a solution on the end of the window")
    return sorted(out)


def _scan(func: str, c: float, a: int, b: int, lo: int, hi: int) -> list[int]:
    """The same solutions by brute force: every half degree of the window, tested against f(a(x - b)) = c."""
    hits = []
    for half in range(2 * lo + 1, 2 * hi):
        x = half / 2
        angle = a * (x - b)
        if func == "tan" and abs(math.cos(math.radians(angle))) < 1e-9:
            continue
        if abs(_FN[func](math.radians(angle)) - c) < 1e-9:
            hits.append(x)
    return hits


def _window(rng):
    return rng.choice(((0, 360), (0, 360), (-180, 180)))


def _inner(a: int, b: int) -> str:
    """The argument a(x - b) as a learner reads it."""
    shifted = f"x {'-' if b > 0 else '+'} {abs(b)}"
    if a == 1:
        return shifted if b else "x"
    return f"{a}({shifted})" if b else f"{a}x"


def _eq_text(func: str, a: int, b: int, c_text: str, neg: bool) -> str:
    return f"{func}({_inner(a, b)}) = {'-' if neg else ''}{c_text}"


def _window_text(lo: int, hi: int) -> str:
    return f"{lo} ≤ x ≤ {hi}"


def _cands(func, theta, sign, a, b, lo, hi, answer):
    """Tagged wrong lists: opposite sign, one solution dropped, wrong function's quadrants, one period only, wrong shift."""
    other_funcs = [f for f in ("sin", "cos", "tan") if f != func]
    cands = [(_solutions(func, theta, -sign, a, b, lo, hi), "w5t-opposite-quadrants")]
    for g in other_funcs:
        try:
            cands.append((_solutions(g, theta, sign, a, b, lo, hi), "w5t-wrong-function-quadrants"))
        except TooFewDistractors:
            continue
    if b:
        cands.append((_solutions(func, theta, sign, a, -b, lo, hi), "w5t-shift-wrong-way"))
    if a > 1:
        first_period = sorted(
            (a * b + u) // a
            for u in _principal(func, theta, sign)
            if (a * b + u) % a == 0 and lo < (a * b + u) // a < hi
        )
        cands.append((first_period, "w5t-one-period-only"))
        cands.append((_solutions(func, theta, sign, 1, b, lo, hi), "w5t-forgets-the-coefficient"))
    cands.append((answer[:1], "w5t-drops-solutions"))
    cands.append((answer[:-1], "w5t-drops-solutions"))
    return [(v, t) for v, t in cands if v]


def _build(rng, tier: str):
    func = rng.choice(("sin", "cos", "tan"))
    sign = rng.choice((1, -1))
    lo, hi = _window(rng)
    if tier == "easy":
        a, b, theta = 1, 0, rng.choice((30, 45, 60))
    elif tier == "medium":
        a, b, theta = 1, rng.choice((-1, 1)) * rng.randrange(10, 85, 5), rng.choice((30, 45, 60))
    else:
        a = rng.choice((2, 3))
        b = rng.choice((0, rng.choice((-1, 1)) * rng.randrange(5, 45, 5)))
        theta = rng.choice((30, 60) if a == 2 else (30, 45, 60))
    solutions = _solutions(func, theta, sign, a, b, lo, hi)
    c = sign * _VALUE[func][theta]
    check([float(v) for v in solutions] == _scan(func, c, a, b, lo, hi), "reference-angle and scan solutions differ")
    check(len(solutions) == 2 * a, "unexpected number of solutions")
    fields = list_fields(solutions)
    cands = list_cands(fields, _cands(func, theta, sign, a, b, lo, hi, solutions))
    enough(cands, 2)
    equation = _eq_text(func, a, b, _VALUE_TEXT[func][theta], sign < 0)
    stem = phrase(
        rng,
        (
            "Solve {e} for {w}, where x is in degrees. List all the solutions.",
            "Find every solution of {e} with {w}, giving x in degrees.",
        ),
        e=equation,
        w=_window_text(lo, hi),
    )
    work = [
        f"Let u = {_inner(a, b)}. The reference angle is {theta} degrees.",
        f"Use the CAST rule for the {'negative' if sign < 0 else 'positive'} value to find u in one period, then add whole periods until the window is full.",
        f"Solving for x and keeping those with {_window_text(lo, hi)} gives {fields['answer_text']}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    TOPIC,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_trig_all_solutions(rng, tier="easy"):
    """All solutions of f(a(x - b)) = c in a stated window; the tiers are described in the module docstring."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, lambda r: _build(r, tier))


def _exam(rng):
    func = rng.choice(("sin", "cos", "tan"))
    sign = rng.choice((1, -1))
    theta = rng.choice((30, 60))
    b = rng.choice((-1, 1)) * rng.randrange(5, 45, 5)
    lo, hi = _window(rng)
    c_text = ("-" if sign < 0 else "") + _VALUE_TEXT[func][theta]
    c = sign * _VALUE[func][theta]
    s1 = _solutions(func, theta, sign, 1, 0, lo, hi)
    s2 = _solutions(func, theta, sign, 2, 0, lo, hi)
    s3 = _solutions(func, theta, sign, 2, b, lo, hi)
    for sol, (a, bb) in ((s1, (1, 0)), (s2, (2, 0)), (s3, (2, b))):
        check([float(v) for v in sol] == _scan(func, c, a, bb, lo, hi), "reference-angle and scan solutions differ")
    smallest_nonneg = min(v for v in s2 if v >= 0)
    check(smallest_nonneg == min(v for v in _solutions(func, theta, sign, 2, 0, 0, 360)), "smallest differs")
    fa = list_fields(s1)
    da = list_cands(fa, _cands(func, theta, sign, 1, 0, lo, hi, s1))
    fb = number_fields(smallest_nonneg)
    first_u = min(_principal(func, theta, sign))
    db = number_cands(
        smallest_nonneg,
        [
            (first_u, "w5t-forgets-the-coefficient"),
            (theta, "w5t-reference-angle-as-answer"),
            (smallest_nonneg * 2, "w5t-multiplies-for-the-coefficient"),
            (max(s2), "w5t-wrong-end-of-the-list"),
        ],
    )
    fc = list_fields(s3)
    dc = list_cands(fc, _cands(func, theta, sign, 2, b, lo, hi, s3))
    for dd in (da, db, dc):
        enough(dd)
    eq1 = _eq_text(func, 1, 0, _VALUE_TEXT[func][theta], sign < 0)
    eq2 = _eq_text(func, 2, 0, _VALUE_TEXT[func][theta], sign < 0)
    eq3 = _eq_text(func, 2, b, _VALUE_TEXT[func][theta], sign < 0)
    sub = lambda f: {k: v for k, v in f.items() if k != "answer_kind"}  # noqa: E731
    return dict(
        text=f"Throughout, x is in degrees and {_window_text(lo, hi)}. The value {c_text} is used in each part.",
        parts=[
            part(
                f"List all the solutions of {eq1}.",
                "list",
                sub(fa),
                2,
                [f"The reference angle is {theta} degrees; use CAST."],
                da,
            ),
            part(
                f"What is the smallest solution of {eq2} that is at least 0?",
                fb["answer_kind"],
                sub(fb),
                1,
                ["Solve for 2x using CAST, then halve."],
                db,
                sensible_range=(0, None),
            ),
            part(
                f"List all the solutions of {eq3}.",
                "list",
                sub(fc),
                3,
                ["Solve for u = 2(x shifted), then divide by 2 and add the shift; keep those in the window."],
                dc,
            ),
        ],
    )


@generator("exd_x_w5_trig_all_solutions", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_trig_all_solutions(rng, tier="exam"):
    """Exam chain: f(x) = c, the smallest solution of f(2x) = c, and f(2(x - b)) = c."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)
