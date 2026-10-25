"""Tracing pseudocode by hand: `vm_pseudocode_trace` (easy, medium, hard) and the exam chain `vm_pseudocode_trace_exam`.

Skill `pseudocode-trace-output` (VCE Mathematical Methods, algorithms and pseudocode). The program is shown as
plain text on one line: statements are separated by semicolons and a loop or branch is closed by `end for`,
`end while` or `end if`, so that nothing depends on a line break or on indentation.

* easy: a for-loop that accumulates a sum, a sum of squares, a repeated product or a pair of running values;
* medium: a while-loop with a condition (count the passes, or give the final value), a branch inside a loop (the
  3x + 1 rule) and a withdrawal that ends part-way through a pass;
* hard: a numerical method written as pseudocode, with the value after k passes to a stated precision: bisection on
  f, Newton's method on a polynomial, and the left-rectangle or trapezium sum with n strips;
* exam (`vm_pseudocode_trace_exam`): a loop with a stopping rule. (a) the value after one pass (or the width after
  two), (b) the value after the next passes by the same rule, following through from (a), (c) how many passes the
  loop makes before it stops, (d) the value printed at the end.

Every key comes from running the algorithm in exact `Fraction` arithmetic; the tests re-read the program text with a
separate interpreter and recompute each answer.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator, phrase
from ..answers import Distractor, exact_value
from ..core import latex_answer, round_half_up, stated_precision_string
from .exd_x_r2b_support import Redraw, _finish, _first_wrong, _number_value, clear_of_boundary, draw
from ._company_kit import keep_company, nudge_rank, numeric_pool
from ._spread_kit import mixed, uses_rng
from .r4e_common import fraction_part as _fraction_part
from .r4e_common import single

F = Fraction


PLAIN_SHIFT_COST = 1.5  # ... and elsewhere only where one swap clears a feature the key holds alone
COMPANY_SHIFT_COST = 0.9  # a companion may move the key down a rank ...
NUDGE = 0.3  # ... which `nudge_rank` pays back by spreading the ranks of other draws


def _spread(value, wrong, step, lo, hi, *, nudge=False, places=None):
    """The wrong-answer candidates reordered so that the first three hold a random number below the key and the rest above
    it (`_spread_kit.mixed`), with plain near misses a multiple of `step` away to make up a side that is short.

    `lo` and `hi` are what the stem allows (a count is at least 1, a value printed after a loop is at least the loop's
    limit, a bisection value lies inside the bracket, ...). They bind every option, slips and fillers alike: a slip outside
    them is dropped. The key is never moved, and it must lie inside them."""
    key = F(value)
    check(
        (lo is None or key >= lo) and (hi is None or key <= hi), f"the key {key} lies outside its own bounds {lo}, {hi}"
    )
    near = [
        (key + d * step, "near-miss-value")
        for d in (-4, -3, -2, -1, 1, 2, 3, 4)
        if (lo is None or key + d * step >= lo) and (hi is None or key + d * step <= hi)
    ]
    ordered = mixed(
        key, list(wrong), fillers=near, lo=lo, hi=hi, unbounded="open" if lo is None and hi is None else None
    )
    pool = numeric_pool(key, step, lo, hi)
    if not nudge:
        ordered = keep_company(key, ordered, pool, shift_cost=PLAIN_SHIFT_COST, places=places)
        if places is not None and key.denominator == 1 and not any(F(v).denominator == 1 for v, _ in ordered[:3] if v):
            raise Redraw("a whole key beside decimals only: the stem leaves no whole-number option")
        return ordered
    ordered = keep_company(key, ordered, pool, shift_cost=COMPANY_SHIFT_COST, places=places)
    return nudge_rank(key, ordered, pool, p=NUDGE) if key.denominator == 1 else ordered


def fraction_part(prompt, value, marks, working, wrong, *, lo=1, hi=None, nudge=False, **kw):
    """A whole-number answer. Counts and printed values are at least 1 unless the part says otherwise."""
    return _fraction_part(prompt, value, marks, working, _spread(value, wrong, 1, lo, hi, nudge=nudge), **kw)


def number_part(
    prompt, value, places, marks, working, wrong, *, lo=None, hi=None, step=None, follow_through=None
) -> dict:
    """A decimal answer rounded to `places` places, graded as a `number`.

    Like `exd_x_r2b_support.number_part`, except that a wrong answer is dropped only when it rounds to the key. The
    shared builder also drops any within the spoken-transcript tolerance of 0.051, which would remove every slip that
    lands close to the key (a numerical method that has nearly converged); a typed answer is held to half a unit of
    its last place, so a value a whole unit away is always wrong."""
    value = F(value)
    rounded = round_half_up(value, places)
    key_text = stated_precision_string(rounded, places)
    unit = "decimal place" if places == 1 else "decimal places"
    if "decimal place" not in prompt:
        prompt = f"{prompt.rstrip('.?')}, correct to {places} {unit}."

    def build(v, tag):
        r = round_half_up(F(v), places)
        return Distractor(stated_precision_string(r, places), tag, exact_value(r))

    def same(v):
        return round_half_up(F(v), places) == rounded

    unit_in_last_place = F(1, 10**places)
    lo = unit_in_last_place if lo is None else lo - unit_in_last_place  # a rounded key may sit one unit outside
    hi = None if hi is None else hi + unit_in_last_place
    wrong = [(v, t) for v, t in wrong if v is not None and not same(v)]  # before the split, so it is not skewed
    options = _first_wrong(
        key_text, _spread(rounded, wrong, step or F(1, 10 ** max(places - 1, 0)), lo, hi, places=places), same, build
    )
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="number",
            answer=_number_value(rounded),
            answer_text=key_text,
            answer_latex=latex_answer(key_text),
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        None,
    )


def dec(v) -> str:
    """A rational as the decimal a learner reads in code: `7`, `2.5`, `0.125` (the value must terminate)."""
    v = F(v)
    if v.denominator == 1:
        return str(v.numerator)
    for places in range(1, 10):
        if (v * 10**places).denominator == 1:
            return f"{float(v):.{places}f}"
    raise VerificationError(f"{v} is not a short terminating decimal")


# -- program text --------------------------------------------------------------------------------------------------


def _set(var: str, expr) -> str:
    return f"set {var} to {expr}"


def _for(var: str, lo, hi, body) -> str:
    return f"for {var} from {lo} to {hi}: " + "; ".join(body) + "; end for"


def _while(cond: str, body) -> str:
    return f"while {cond}: " + "; ".join(body) + "; end while"


def _if(cond: str, then, other=None) -> str:
    text = f"if {cond}: " + "; ".join(then)
    if other:
        text += "; otherwise: " + "; ".join(other)
    return text + "; end if"


def _program(*statements) -> str:
    return "; ".join(statements)


def _stem(rng, code: str) -> str:
    return phrase(
        rng,
        (
            "Here is a short program written in pseudocode: {code}.",
            "A program is written in pseudocode as follows: {code}.",
            "Trace this pseudocode by hand: {code}.",
            "The following pseudocode is run: {code}.",
        ),
        code=code,
    )


_PRINTED = (
    "What value is printed?",
    "What does the program print?",
    "Find the value that is printed.",
)

_NAMES = (("s", "i"), ("t", "k"), ("total", "j"), ("p", "i"))

_NEAR = "near-miss-value"


def _loop_wrongs(base, passes, run, extra=()):
    """Wrong answers from running a loop one pass too many or too few: `run(n)` is the answer after n passes."""
    return [(run(passes + 1), "vm-loop-extra-pass"), (run(passes - 1), "vm-loop-missing-pass"), *extra, (base, _NEAR)]


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy_sum(rng):
    s0 = rng.choice((0, 0, 1, 2, 5, 10))
    n = rng.randint(4, 8)
    c, d = rng.randint(1, 5), rng.randint(0, 4)
    acc, i = rng.choice(_NAMES)
    term = (f"{c} × {i}" if c != 1 else i) + (f" + {d}" if d else "")
    if c == 1 and not d:
        term = f"{i} + {rng.randint(1, 3)}"
        d = int(term.split("+ ")[1])
    code = _program(_set(acc, s0), _for(i, 1, n, [_set(acc, f"{acc} + {term}")]), f"print {acc}")

    def run(passes):
        return s0 + sum(c * k + d for k in range(1, passes + 1))

    key = run(n)
    wrong = [
        (run(n + 1), "vm-loop-extra-pass"),
        (run(n - 1), "vm-loop-missing-pass"),
        (key - s0, "vm-forgets-initial-value") if s0 else (None, ""),
        (c * n + d, "vm-last-term-only"),
        (key + 1, _NEAR),
    ]
    working = [f"Start with {acc} = {s0}. Each pass adds {term} for {i} = 1, 2, ..., {n}, so {acc} = {key}."]
    return single(_stem(rng, code), fraction_part(rng.choice(_PRINTED), key, 2, working, wrong))


def _easy_squares(rng):
    n = rng.randint(3, 7)
    k = rng.choice((0, 0, 1, 2, 3))
    acc, i = rng.choice(_NAMES)
    code = _program(_set(acc, k), _for(i, 1, n, [_set(acc, f"{acc} + {i}^2")]), f"print {acc}")

    def run(passes):
        return k + sum(v * v for v in range(1, passes + 1))

    key = run(n)
    wrong = [
        (k + sum(range(1, n + 1)) ** 2, "vm-squares-sum-confused"),
        (run(n + 1), "vm-loop-extra-pass"),
        (run(n - 1), "vm-loop-missing-pass"),
        (n * n, "vm-last-term-only"),
        (key - k, "vm-forgets-initial-value") if k else (None, ""),
    ]
    working = [f"The loop adds {', '.join(f'{v}^2' for v in range(1, n + 1))} to {k}, so {acc} = {key}."]
    return single(_stem(rng, code), fraction_part(rng.choice(_PRINTED), key, 2, working, wrong))


def _easy_product(rng):
    p0 = rng.choice((1, 2, 3, 5))
    m = rng.choice((2, 3, 4))
    n = rng.randint(3, 6) if m < 4 else rng.randint(3, 5)
    acc, i = rng.choice(_NAMES)
    code = _program(_set(acc, p0), _for(i, 1, n, [_set(acc, f"{acc} × {m}")]), f"print {acc}")
    key = p0 * m**n
    wrong = [
        (p0 * m ** (n - 1), "vm-loop-missing-pass"),
        (p0 * m ** (n + 1), "vm-loop-extra-pass"),
        (p0 + m * n, "vm-power-as-sum"),
        (p0 * m * n, "vm-power-as-product"),
    ]
    working = [
        f"{acc} is multiplied by {m} on each of {n} passes, so {acc} = {(str(p0) + ' × ') if p0 != 1 else ''}{m}^{n} = {key}."
    ]
    return single(_stem(rng, code), fraction_part(rng.choice(_PRINTED), key, 2, working, wrong))


def _easy_pair(rng):
    a0, b0 = rng.choice(((1, 1), (0, 1), (1, 3), (2, 1), (1, 2), (3, 1)))
    n = rng.randint(3, 6)
    code = _program(
        _set("a", a0),
        _set("b", b0),
        _for("i", 1, n, [_set("c", "a + b"), _set("a", "b"), _set("b", "c")]),
        "print b",
    )

    def run(passes):
        a, b = a0, b0
        for _ in range(passes):
            a, b = b, a + b
        return a, b

    key = run(n)[1]
    wrong = [
        (run(n)[0], "vm-prints-wrong-variable"),
        (run(n + 1)[1], "vm-loop-extra-pass"),
        (run(1)[1], "vm-stops-after-first-pass"),
        (key - 1, _NEAR),
    ]
    working = [
        f"After each pass a takes the old b and b takes a + b. Starting from {a0}, {b0} the value of b after {n} passes is {key}."
    ]
    return single(_stem(rng, code), fraction_part("What value of b is printed?", key, 2, working, wrong))


_EASY = (_easy_sum, _easy_sum, _easy_squares, _easy_product, _easy_pair)


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_additive(rng):
    s0, d = rng.randint(0, 10), rng.randint(3, 9)
    m = rng.randint(3, 8)
    exact = rng.random() < 0.6
    target = s0 + m * d if exact else s0 + m * d - rng.randint(1, d - 1)
    strict = rng.choice((True, False))
    cond = f"s {'<' if strict else '≤'} {target}"
    code = _program(
        _set("s", s0), _set("n", 0), _while(cond, [_set("s", f"s + {d}"), _set("n", "n + 1")]), "print " + "{out}"
    )

    def run(strict_):
        s, n = s0, 0
        for _ in attempts(100, "the loop test failing"):
            if not (s < target if strict_ else s <= target):
                break
            s += d
            n += 1
        return s, n

    s_end, n_end = run(strict)
    s_other, n_other = run(not strict)
    ask_n = rng.random() < 0.5
    out = "n" if ask_n else "s"
    key = n_end if ask_n else s_end
    other = n_other if ask_n else s_other
    wrong = [
        (other, "vm-while-boundary-misread"),
        (key + 1, "vm-counts-wrong-passes" if ask_n else _NEAR),
        (key - 1, "vm-counts-wrong-passes" if ask_n else _NEAR),
        (key - 2, _NEAR),
        (key + 3, _NEAR),
        (target, "vm-prints-threshold") if not ask_n else (n_end + 2, _NEAR),
    ]
    working = [
        f"s goes {', '.join(str(s0 + d * k) for k in range(n_end + 1))}; the test s {'<' if strict else '≤'} {target} "
        f"first fails when s = {s_end}, after {n_end} passes."
    ]
    return single(
        _stem(rng, code.format(out=out)),
        fraction_part(
            "How many times does the loop run?" if ask_n and rng.random() < 0.4 else rng.choice(_PRINTED),
            key,
            2,
            working,
            wrong,
            lo=1 if ask_n else (target if strict else target + 1),  # the loop ends once the test fails
        ),
    )


def _medium_multiplicative(rng):
    s0 = rng.choice((1, 2, 3, 5))
    m = rng.choice((2, 3))
    steps = rng.randint(3, 6) if m == 2 else rng.randint(3, 4)
    exact = rng.random() < 0.5
    target = s0 * m**steps if exact else s0 * m**steps - rng.randint(1, max(1, s0 * m ** (steps - 1) - 1))
    strict = rng.choice((True, False))
    cond = f"s {'<' if strict else '≤'} {target}"
    out = rng.choice(("n", "s"))
    code = _program(
        _set("s", s0), _set("n", 0), _while(cond, [_set("s", f"s × {m}"), _set("n", "n + 1")]), f"print {out}"
    )

    def run(strict_):
        s, n = s0, 0
        for _ in attempts(100, "the loop test failing"):
            if not (s < target if strict_ else s <= target):
                break
            s *= m
            n += 1
        return s, n

    s_end, n_end = run(strict)
    s_other, n_other = run(not strict)
    ask_n = out == "n"
    key = n_end if ask_n else s_end
    other = n_other if ask_n else s_other
    wrong = [
        (other, "vm-while-boundary-misread"),
        (key + 1 if ask_n else s_end * m, "vm-counts-wrong-passes" if ask_n else "vm-loop-extra-pass"),
        (key - 1, "vm-counts-wrong-passes") if ask_n else (s_end * m * m, _NEAR),
        (target if not ask_n else n_end + 2, "vm-prints-threshold" if not ask_n else _NEAR),
    ]
    working = [
        f"s goes {', '.join(str(s0 * m**k) for k in range(n_end + 1))}; the test s {'<' if strict else '≤'} {target} "
        f"first fails when s = {s_end}, after {n_end} passes."
    ]
    return single(
        _stem(rng, code),
        fraction_part(
            rng.choice(_PRINTED), key, 2, working, wrong, lo=1 if ask_n else (target if strict else target + 1)
        ),
    )


def _medium_collatz(rng):
    start = rng.choice((3, 5, 6, 7, 9, 10, 11, 12))
    code = _program(
        _set("s", start),
        _set("n", 0),
        _while(
            "s is not 1",
            [_if("s is even", [_set("s", "s ÷ 2")], [_set("s", "3 × s + 1")]), _set("n", "n + 1")],
        ),
        "print n",
    )
    s, n, seq = start, 0, [start]
    for _ in attempts(60, "the 3x + 1 rule reaching 1"):
        if s == 1:
            break
        s = s // 2 if s % 2 == 0 else 3 * s + 1
        n += 1
        seq.append(s)
    check(s == 1 and n >= 5, "the sequence must reach 1 in at least five passes")
    wrong = [
        (n - 1, "vm-counts-wrong-passes"),
        (n + 1, "vm-counts-wrong-passes"),
        (len([v for v in seq[:-1] if v % 2 == 0]), "vm-counts-one-branch"),
        (start, "vm-reads-initial-value"),
    ]
    working = [f"s takes the values {', '.join(map(str, seq))}, which is {n} passes."]
    return single(
        _stem(rng, code),
        fraction_part(
            rng.choice(("What value of n is printed?", "What does the program print?")), n, 2, working, wrong
        ),
    )


def _medium_withdrawal(rng):
    r = rng.randint(4, 9)
    full = rng.randint(3, 7)
    rest = rng.randint(1, r - 1)
    bal = full * r + rest
    code = _program(
        _set("b", bal),
        _set("y", 0),
        _while(
            "b > 0",
            [_if(f"b > {r}", [_set("b", f"b - {r}")], [_set("b", 0)]), _set("y", "y + 1")],
        ),
        "print y",
    )
    key = full + 1
    wrong = [
        (full, "vm-ignores-final-partial-pass"),
        (key + 1, "vm-counts-wrong-passes"),
        (full - 1, "vm-counts-wrong-passes"),
        (key + 2, _NEAR),
    ]
    working = [f"b falls by {r} on {full} passes to {rest}, and one more pass takes it to 0, so y = {key}."]
    return single(_stem(rng, code), fraction_part("What value of y is printed?", key, 2, working, wrong))


_MEDIUM = (_medium_additive, _medium_multiplicative, _medium_collatz, _medium_withdrawal, _medium_additive)


# -- hard: numerical methods --------------------------------------------------------------------------------------


def _fx(f, x):
    return f(x)


_QUAD = ("x^2 - {c}", lambda c: lambda x: x * x - c, "2 × x", lambda x: 2 * x)
_CUBE = ("x^3 - {c}", lambda c: lambda x: x**3 - c, "3 × x^2", lambda x: 3 * x * x)
_CUBE_X = ("x^3 + x - {c}", lambda c: lambda x: x**3 + x - c, "3 × x^2 + 1", lambda x: 3 * x * x + 1)


def _root_estimate(f, lo=F(0), hi=F(40)):
    """The root of an increasing f on [lo, hi], by 60 bisection halvings (only to choose a starting value)."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lo


def _newton_run(f, g, x0, passes):
    x = F(x0)
    for _ in range(passes):
        x = x - f(x) / g(x)
    return x


def _hard_newton(rng):
    ftext, mk, gtext, g = rng.choice((_QUAD, _QUAD, _CUBE, _CUBE_X))
    for _ in attempts(300, "a Newton start whose answer is clear of a rounding tie"):
        c = rng.randint(3, 40) if ftext.startswith("x^2") else rng.randint(3, 60)
        f = mk(c)
        root = _root_estimate(f)
        x0 = max(1, round(root))
        if f(F(x0)) == 0:
            continue
        passes = rng.randint(1, 3)
        places = rng.choice((3, 4)) if passes > 1 else rng.choice((2, 3))
        key = _newton_run(f, g, x0, passes)
        if clear_of_boundary(key, places):
            break
    fx = ftext.format(c=c)
    code = _program(
        f"let f(x) = {fx}",
        f"let g(x) = {gtext}",
        _set("x", x0),
        _for("i", 1, passes, [_set("x", "x - f(x) ÷ g(x)")]),
        "print x",
    )
    x = F(x0)
    steps = [x]
    for _ in range(passes):
        x = x - f(x) / g(x)
        steps.append(x)
    wrong = [
        (_newton_run(f, g, x0, passes + 1), "vm-iterations-miscounted"),
        (_newton_run(f, g, x0, passes - 1), "vm-iterations-miscounted"),
        (_add_newton(f, g, x0, passes), "vm-newton-adds-correction"),
        (_slope_free(f, x0, passes), "vm-newton-forgets-slope"),
        (f(steps[-2]) / g(steps[-2]), "vm-newton-prints-correction"),
    ]
    wrong = [(v, t) for v, t in wrong if v is not None]
    working = [
        f"Each pass replaces x by x - f(x) ÷ g(x). Starting at {x0}: "
        + ", ".join(f"x = {float(v):.5f}" for v in steps[1:])
        + "."
    ]
    return single(
        _stem(rng, code),
        number_part(
            phrase(
                rng,
                ("What value is printed?", "What does the program print?", "Find the value that is printed."),
            ),
            key,
            places,
            2,
            working,
            wrong,
            hi=2 * max(steps),  # Newton's method from a sensible start stays within twice its largest iterate
        ),
    )


def _add_newton(f, g, x0, passes):
    x = F(x0)
    for _ in range(passes):
        x = x + f(x) / g(x)
    return x


def _slope_free(f, x0, passes):
    x = F(x0)
    for _ in range(passes):
        x = x - f(x)
        if not 0 < x < 2 * x0:  # only a slip that still looks like an iterate of a root search
            return None
    return x


def _bisect_run(f, a, b, passes, flip=False):
    a, b = F(a), F(b)
    m = (a + b) / 2
    for _ in range(passes):
        m = (a + b) / 2
        fm = f(m)
        if fm == 0:
            raise Redraw("a midpoint is exactly a root")
        if (fm < 0) != flip:
            a = m
        else:
            b = m
    return m, a, b


def _hard_bisection(rng):
    ftext, mk, _, _ = rng.choice((_QUAD, _QUAD, _CUBE, _CUBE_X))
    width = rng.choice((1, 2, 4))
    a = width * rng.randint(1, 3)
    b = a + width
    f_probe = mk(0)
    lo_c = f_probe(F(a)) - f_probe(F(0))
    hi_c = f_probe(F(b)) - f_probe(F(0))
    c = rng.randint(int(lo_c) + 1, int(hi_c) - 1)
    f = mk(c)
    check(f(F(a)) < 0 < f(F(b)), "the bracket must change sign")
    passes = rng.randint(2, 4)
    m, a_end, b_end = _bisect_run(f, a, b, passes)
    fx = ftext.format(c=c)
    code = _program(
        f"let f(x) = {fx}",
        _set("a", a),
        _set("b", b),
        _for(
            "i",
            1,
            passes,
            [_set("m", "(a + b) ÷ 2"), _if("f(m) < 0", [_set("a", "m")], [_set("b", "m")])],
        ),
        "print m",
    )
    wrong = [
        (_bisect_run(f, a, b, passes + 1)[0], "vm-iterations-miscounted"),
        (_bisect_run(f, a, b, passes - 1)[0], "vm-iterations-miscounted"),
        (_bisect_run(f, a, b, passes, flip=True)[0], "vm-bisect-wrong-half"),
        (a_end, "vm-bisect-reports-endpoint"),
        (b_end, "vm-bisect-reports-endpoint"),
    ]
    working = [
        f"f({a}) < 0 < f({b}). Each pass takes m = (a + b) ÷ 2 and keeps the half where f changes sign. "
        f"After {passes} passes m = {dec(m)}."
    ]
    return single(
        _stem(rng, code),
        number_part(
            rng.choice(("What value of m is printed?", "What does the program print?")),
            m,
            passes,
            2,
            working,
            wrong,
            lo=a,
            hi=b,  # every midpoint lies inside the starting bracket
        ),
    )


_AREA_FUNCS = (
    ("x^2", lambda x: x * x, 3),
    ("x^2 + 1", lambda x: x * x + 1, 3),
    ("x^3", lambda x: x**3, 3),
    ("1 ÷ x", lambda x: 1 / x, 3),
    ("8 - x^2", lambda x: 8 - x * x, 3),
)


def _hard_area(rng):
    ftext, f, _ = rng.choice(_AREA_FUNCS)
    method = rng.choice(("left", "left", "trapezium"))
    for _ in attempts(300, "bounds and strips giving a clean rounded sum"):
        a = rng.randint(1, 2)
        b = a + rng.randint(1, 3)
        if ftext.startswith("8"):
            a, b = rng.choice(((0, 2), (1, 2)))
        n = rng.choice((4, 5, 8))
        h = F(b - a, n)
        places = 2 if rng.random() < 0.6 else 3
        xs = [a + i * h for i in range(n + 1)]
        left = h * sum(f(x) for x in xs[:-1])
        right = h * sum(f(x) for x in xs[1:])
        trap = h * ((f(xs[0]) + f(xs[-1])) / 2 + sum(f(x) for x in xs[1:-1]))
        key = left if method == "left" else trap
        if clear_of_boundary(key, places):
            break
    head = [f"let f(x) = {ftext}", _set("a", a), _set("b", b), _set("n", n), _set("h", "(b - a) ÷ n")]
    if method == "left":
        body = [_set("s", 0), _for("i", 0, "n - 1", [_set("s", "s + f(a + i × h)")]), "print h × s"]
    else:
        body = [
            _set("s", "(f(a) + f(b)) ÷ 2"),
            _for("i", 1, "n - 1", [_set("s", "s + f(a + i × h)")]),
            "print h × s",
        ]
    code = _program(*head, *body)
    if method == "left":
        wrong = [
            (right, "vm-rect-right-for-left"),
            (trap, "vm-rect-trapezium-confused"),
            (left / h, "vm-sum-forgets-width"),
            (sum(f(xs[i]) for i in range(n - 1)) * h, "vm-iterations-miscounted"),
        ]
        working = [f"h = {dec(h)}. s adds f(a + i h) for i = 0 to {n - 1} (left ends), then h × s = {float(left):.5f}."]
    else:
        wrong = [
            (left, "vm-trap-left-rectangle-confused"),
            (right, "vm-trap-right-rectangle-confused"),
            (h * ((f(xs[0]) + f(xs[-1])) + sum(f(x) for x in xs[1:-1])), "vm-trap-ends-not-halved"),
            (trap / h, "vm-sum-forgets-width"),
        ]
        working = [
            f"h = {dec(h)}. s = (f(a) + f(b)) ÷ 2 plus the {n - 1} inner values, then h × s = {float(trap):.5f}."
        ]
    return single(
        _stem(rng, code),
        number_part(
            rng.choice(("What value is printed?", "What does the program print?")),
            key,
            places,
            2,
            working,
            wrong,
            lo=(b - a) * min(f(a), f(b)),  # f is monotone on [a, b], so the sum lies between the two rectangles
            hi=(b - a) * max(f(a), f(b)),
        ),
    )


_HARD = (_hard_newton, _hard_newton, _hard_bisection, _hard_bisection, _hard_area, _hard_area)

_TIERS = {
    "easy": lambda rng: draw(rng.choice(_EASY), rng),
    "medium": lambda rng: draw(rng.choice(_MEDIUM), rng),
    "hard": lambda rng: draw(rng.choice(_HARD), rng),
}


@generator(
    "vm_pseudocode_trace",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def vm_pseudocode_trace(rng, tier: str = "easy"):
    """Read short pseudocode and give the printed value; the tiers are described in the module docstring."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam chain ----------------------------------------------------------------------------------------------------


NEWTON_PASSES_ASKED = 2  # the exam chain's part (b) asks for x "after the loop has run a second time"
BISECTION_PASSES_ASKED = 5  # and the bisection chain's part (b) for b - a "after the loop has run five times"


def _chain_newton(rng):
    cubic = rng.random() < 0.4
    for _ in attempts(500, "a Newton program that stops after three to six passes, clear of rounding ties"):
        c = rng.randint(3, 60) if cubic else rng.randint(3, 80)
        f = (lambda v: v**3 - c) if cubic else (lambda v: v * v - c)
        g = (lambda v: 3 * v * v) if cubic else (lambda v: 2 * v)
        root = _root_estimate(f, F(0), F(10))
        x0 = math.ceil(root) + rng.choice((0, 0, 1))
        if f(F(x0)) == 0:
            continue
        tol = rng.choice((F(1, 10), F(1, 100), F(1, 1000), F(1, 10000)))
        x, n, xs = F(x0), 0, [F(x0)]
        d = F(1)
        for _i in attempts(60, "the stopping rule being met"):
            if not d > tol:
                break
            y = x - f(x) / g(x)
            d, x, n = x - y, y, n + 1
            xs.append(x)
        if 3 <= n <= 6 and all(clear_of_boundary(v, 3) for v in xs[1:3]) and clear_of_boundary(xs[-1], 4):
            break
    ftext = f"x^3 - {c}" if cubic else f"x^2 - {c}"
    gtext = "3 × x^2" if cubic else "2 × x"
    code = _program(
        f"let f(x) = {ftext}",
        f"let g(x) = {gtext}",
        _set("x", x0),
        _set("d", 1),
        _set("n", 0),
        _while(
            f"d > {dec(tol)}",
            [_set("y", "x - f(x) ÷ g(x)"), _set("d", "x - y"), _set("x", "y"), _set("n", "n + 1")],
        ),
        "print n",
        "print x",
    )
    stem = phrase(
        rng,
        (
            "The following pseudocode applies Newton's method to f and stops when the change is small: {code}.",
            "A program uses Newton's method and a stopping rule: {code}.",
            "Consider this pseudocode, which repeats a Newton step until the change in x is at most the tolerance: {code}.",
        ),
        code=code,
    )
    ft_a = "part_a - (part_a**3 - {c})/(3*part_a**2)" if cubic else "part_a - (part_a**2 - {c})/(2*part_a)"
    part_a = number_part(
        "What is the value of x after the loop has run once, correct to 3 decimal places?",
        xs[1],
        3,
        1,
        [f"x = {x0} - f({x0}) ÷ g({x0}) = {float(xs[1]):.5f}."],
        [
            (_add_newton(f, g, x0, 1), "vm-newton-adds-correction"),
            (_slope_free(f, x0, 1), "vm-newton-forgets-slope"),
            (F(x0), "vm-reads-initial-value"),
            (f(F(x0)) / g(F(x0)), "vm-newton-prints-correction"),
        ],
        lo=xs[-1] / 2,  # an iterate stays within a factor of two of the root it is closing in on
        hi=F(x0),  # the first step from the start lands between 0 and the start
    )
    part_b = number_part(
        "Use your answer to (a) to find the value of x after the loop has run a second time, correct to 3 decimal places.",
        xs[2],
        3,
        2,
        [f"Apply the same step to {float(xs[1]):.5f}: x = {float(xs[2]):.5f}."],
        [
            (xs[3] if len(xs) > 3 else None, "vm-iterations-miscounted"),
            (xs[1], "vm-iterations-miscounted"),
            (xs[1] + f(xs[1]) / g(xs[1]), "vm-newton-adds-correction"),
            (_slope_free(f, xs[1], 1), "vm-newton-forgets-slope"),
        ],
        lo=xs[-1] / 2,
        hi=xs[1],  # the second step lands between 0 and the first
        follow_through=ft_a.format(c=c),
    )
    part_c = fraction_part(
        "How many times does the loop run before the program stops?",
        n,
        2,
        [
            f"The changes d are {', '.join(f'{float(xs[k] - xs[k + 1]):.5f}' for k in range(n))}; "
            f"the first change of at most {dec(tol)} comes on pass {n}."
        ],
        [
            (n - 1, "vm-counts-wrong-passes"),
            (n + 1, "vm-counts-wrong-passes"),
            (n + 2, _NEAR),
            (n - 2, _NEAR),
            (n + 3, _NEAR),
        ],
        lo=NEWTON_PASSES_ASKED,  # part (b) has already run the loop a second time, so it ran at least twice
        nudge=True,
    )
    part_d = number_part(
        "What value of x is printed last, correct to 4 decimal places?",
        xs[-1],
        4,
        2,
        [f"After {n} passes x = {float(xs[-1]):.6f}."],
        [
            (xs[-2], "vm-stops-one-pass-early"),
            (xs[-2] - xs[-1], "vm-prints-wrong-variable"),
            (F(n), "vm-prints-wrong-variable"),
            (xs[min(n + 1, len(xs) - 1)] if n + 1 < len(xs) else _extra_step(f, g, xs[-1]), "vm-iterations-miscounted"),
        ],
        lo=xs[-1] / 2,
        # the loop ran at least twice and x only falls, so it is below the answer to (b), which the learner has to
        # 3 places (half a unit more is still consistent with it); plain near misses are then a last place apart
        hi=round_half_up(xs[NEWTON_PASSES_ASKED], 3) + F(1, 2000),
        step=F(1, 10000),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def _extra_step(f, g, x):
    return x - f(x) / g(x)


def _chain_bisection(rng):
    for _ in attempts(500, "a bisection program that stops after seven to ten passes"):
        width = rng.choice((2, 4))
        a0 = width * rng.randint(1, 3)
        b0 = a0 + width
        c = rng.randint(a0 * a0 + 1, b0 * b0 - 1)
        if math.isqrt(c) ** 2 == c:
            continue
        f = lambda v, c=c: v * v - c  # noqa: E731
        tol = rng.choice((F(1, 50), F(1, 100), F(1, 200), F(1, 500)))
        a, b, n = F(a0), F(b0), 0
        for _i in attempts(40, "the interval narrowing below the tolerance"):
            if not b - a > tol:
                break
            m = (a + b) / 2
            if f(m) == 0:
                break
            a, b = (m, b) if f(m) < 0 else (a, m)
            n += 1
        if f((a + b) / 2) == 0:
            continue
        if 7 <= n <= 10 and clear_of_boundary((a + b) / 2, 4):
            break
    code = _program(
        f"let f(x) = x^2 - {c}",
        _set("a", a0),
        _set("b", b0),
        _set("n", 0),
        _while(
            f"b - a > {dec(tol)}",
            [_set("m", "(a + b) ÷ 2"), _if("f(m) < 0", [_set("a", "m")], [_set("b", "m")]), _set("n", "n + 1")],
        ),
        "print n",
        "print (a + b) ÷ 2",
    )
    stem = phrase(
        rng,
        (
            "The following pseudocode uses bisection and stops when the interval is short enough: {code}.",
            "A program repeats the bisection step until b - a is at most the tolerance: {code}.",
            "Consider this pseudocode for locating a root of f by halving an interval: {code}.",
        ),
        code=code,
    )
    w2 = F(width, 4)
    w5 = F(width, 32)
    final = (a + b) / 2
    part_a = number_part(
        "What is the value of b - a after the loop has run twice, correct to 3 decimal places?",
        w2,
        3,
        1,
        [f"Each pass halves b - a, so after two passes it is {width} ÷ 4 = {dec(w2)}."],
        [
            (F(width, 2), "vm-width-halved-once"),
            (F(width, 8), "vm-width-halved-extra"),
            (F(width), "vm-reads-initial-value"),
            (F(width, 16), "vm-width-halved-extra"),
            (w2 + F(1, 2), "near-miss-value"),
            (w2 - F(1, 8), "near-miss-value"),
        ],
        hi=F(width),  # b - a only shrinks from its starting value
    )
    part_b = number_part(
        "Use your answer to (a) to find b - a after the loop has run five times, correct to 4 decimal places.",
        w5,
        4,
        2,
        [f"Three more halvings of {dec(w2)} give {dec(w2)} ÷ 8 = {dec(w5)}."],
        [
            (w2 / 4, "vm-iterations-miscounted"),
            (w2 / 16, "vm-iterations-miscounted"),
            (w2 / 2, "vm-iterations-miscounted"),
            (w2, "vm-iterations-miscounted"),
        ],
        hi=w2,  # and five passes leave no more than two did
        follow_through="part_a/8",
    )
    part_c = fraction_part(
        "How many times does the loop run before the program stops?",
        n,
        2,
        [f"b - a is {width} ÷ 2^k after k passes; the first k with {width} ÷ 2^k at most {dec(tol)} is k = {n}."],
        [
            (n - 1, "vm-counts-wrong-passes"),
            (n + 1, "vm-counts-wrong-passes"),
            (n + 2, "near-miss-value"),
            (n - 2, "near-miss-value"),
        ],
        lo=BISECTION_PASSES_ASKED,  # part (b) has already run the loop five times
        nudge=True,
    )
    part_d = number_part(
        "What value is printed last, correct to 4 decimal places?",
        final,
        4,
        2,
        [f"After {n} passes the interval is [{float(a):.5f}, {float(b):.5f}], whose midpoint is {float(final):.6f}."],
        [
            (a, "vm-bisect-reports-endpoint"),
            (b, "vm-bisect-reports-endpoint"),
            (F(n), "vm-prints-wrong-variable"),
            (_mid_after(f, a0, b0, n + 1), "vm-iterations-miscounted"),
        ],
        lo=F(a0),
        hi=F(b0),  # the printed midpoint lies inside the starting bracket
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def _mid_after(f, a0, b0, passes):
    a, b = F(a0), F(b0)
    for _ in range(passes):
        m = (a + b) / 2
        a, b = (m, b) if f(m) < 0 else (a, m)
    return (a + b) / 2


def _chain(rng):
    return (_chain_newton if rng.random() < 0.65 else _chain_bisection)(rng)


@generator("vm_pseudocode_trace_exam", level=17, presets={"exam": dict(tier="exam")})
@uses_rng
def vm_pseudocode_trace_exam(rng, tier: str = "exam"):
    """A loop with a stopping rule: one pass, the next pass following through, the pass count, then the printed value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
