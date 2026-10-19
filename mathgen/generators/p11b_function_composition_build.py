"""Generates `function_composition` questions in two modes, both built on g(x) = sqrt(x - a) and f(x) = 1/(x -
k).

Both skills draw g(x) = sqrt(x - a) and f(x) = 1/(x - k) from ONE shared
fixture: a single topic, ``function_composition``, differentiated only by a
``mode`` kwarg ("compose" / "when_defined") each skill's own ``sources``
supplies via ``params``. ``mathgen.core.make`` seeds its rng from
``f"{topic}:{difficulty or ''}:{seed}"`` -- params play no part in that
string -- so for a matching (difficulty, seed) pair the two modes see
bit-identical rng draws of (a, k, is_class_b). The fixture is built entirely
via unconditional rng calls before the function ever branches on ``mode``.

g's range is y >= 0. If k < 0, sqrt(x - a) can never equal k, so f(g(x)) is
defined on the whole of g's domain, x >= a (the "no exclusion" case). If
k >= 0, sqrt(x - a) = k has exactly one solution, x = a + k**2 (always
>= a, so genuinely inside g's domain), which must be excluded (the
"exclusion" case).

``mode= "when_defined"``'s hard tier gets a deterministic (not drawn) extra sentence that supplies the
pre-combined h(x) = f(g(x)) before asking where it is defined, without touching its answer/distractor logic at
all.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Symbol, sympify

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    paren_negative,
    pick_distractors,
    rational_text,
)

x = Symbol("x")


def _fmt_shift(base: str, shift: int) -> str:
    """``base - shift``, choosing the sign of ``shift`` so the ASCII form
    never renders ``- -n`` -- e.g. ``_fmt_shift("x", -9)`` -> ``"x + 9"``."""
    return f"{base} - {shift}" if shift >= 0 else f"{base} + {-shift}"


def _f_text(k: int) -> str:
    return f"1/({_fmt_shift('x', k)})"


def _g_text(a: int) -> str:
    return f"sqrt({_fmt_shift('x', a)})"


def _verify_compose_numeric(a: int, k: int, answer_text: str) -> None:
    """Independent check, a genuinely different route from construction:
    evaluate f(g(xv)) at 3 sample points with plain Python float arithmetic
    (never sympy, never re-deriving the same f-string), then parse
    ``answer_text`` itself with sympy and substitute the same xv, and confirm
    the two agree to within 1e-9. This verifies the STRING the learner is
    shown, not an internal object built some other way
    (p5c_inverse_function_build.py's ``_verify_answer_text`` discipline).
    """
    exclusion = a + k * k if k >= 0 else None
    xs: list[int] = []
    candidate = a + 1
    while len(xs) < 3:
        if exclusion is None or candidate != exclusion:
            xs.append(candidate)
        candidate += 1
    expr = sympify(answer_text.replace("^", "**"))
    for xv in xs:
        direct = 1.0 / (math.sqrt(xv - a) - k)
        via_string = float(expr.subs(x, xv))
        check(
            abs(direct - via_string) < 1e-9,
            f"f(g({xv})) = {direct} by direct float arithmetic but answer_text "
            f"{answer_text!r} evaluates to {via_string}",
        )


def _build_forward(a: int, k: int, phrasing: str) -> dict:
    """The original ``f(g(x))`` ask. ``phrasing="a"`` is the original
    wording; ``phrasing="b"`` is the same math, same answer/distractor
    construction, wording variety only."""
    f_text = _f_text(k)
    g_text = _g_text(a)
    answer_text = f"1/({_fmt_shift(g_text, k)})"

    _verify_compose_numeric(a, k, answer_text)

    # Misconceptions, all built from the same two sub-expressions (1/(...)
    # and sqrt(...)) as the answer, so none is identifiable by shape alone.
    reversed_order = f"sqrt({_fmt_shift(f_text, a)})"  # g(f(x)) instead of f(g(x))
    treats_as_addition = f"{f_text} + {g_text}"  # f(x) + g(x) instead of f(g(x))
    drops_inner = f"1/({_fmt_shift(_fmt_shift('x', a), k)})"  # f applied to x - a, omitting sqrt

    check_pairwise_distinct(
        sympify(answer_text.replace("^", "**")),
        sympify(reversed_order.replace("^", "**")),
        sympify(treats_as_addition.replace("^", "**")),
        sympify(drops_inner.replace("^", "**")),
        msg="a composition misconception collides with the answer or another distractor",
    )

    if phrasing == "a":
        text = f"Given f(x) = {f_text} and g(x) = {g_text}, find f(g(x))."
    else:
        text = f"Let f(x) = {f_text} and g(x) = {g_text}. Find f(g(x)) as a single expression."
    working = [
        f"g(x) = {g_text}, so f(g(x)) means substituting {g_text} for x in f.",
        f"f({g_text}) = {answer_text}",
    ]

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=pick_distractors(answer_text, [reversed_order, treats_as_addition, drops_inner]),
    )


def _verify_point_eval_numeric(a: int, k: int, m: int, c: int, ans) -> None:
    """Independent check for ``point_eval``: recompute 1/(sqrt(c-a) - k)
    directly with plain float arithmetic, calling ``math.sqrt`` on ``c - a``
    again rather than reusing ``m`` -- a genuinely separate route from the
    construction, which already assumed ``sqrt(c - a) == m``. ``m`` is
    accepted (matching the specified signature) but deliberately
    unused in the computation below, per that same "never reusing m" rule.
    """
    direct = 1.0 / (math.sqrt(c - a) - k)
    check(
        abs(direct - float(ans)) < 1e-9,
        f"f(g({c})) = {direct} by direct float arithmetic but the constructed answer {ans} does not match",
    )


def _build_point_eval(rng, a: int, k: int) -> dict:
    """Evaluate f(g(c)) at a concrete point c chosen so g(c) is a clean
    integer m (never an unresolved surd in a typed numeric answer)."""
    f_text = _f_text(k)
    g_text = _g_text(a)

    # Collision resample: m == k or m == -k makes a candidate below
    # undefined (division by zero); c == k makes drops_inner undefined too;
    # and drops_inner can algebraically coincide with sign_error for
    # specific (a, m, k) triples. Resample m (keep a, k) up to 30 times,
    # exactly like the recursive-definition generator's own resample
    # pattern (mathgen/generators/p11b_recursive_definition_build.py).
    for _attempt in range(30):
        m = rng.randint(1, 9)
        c = a + m * m
        if m == k or m == -k or c == k:
            continue
        ans = Fraction(1, m - k)
        forgot_f = Fraction(m, 1)  # reports g(c) itself, forgets to apply f
        sign_error = Fraction(1, m + k)  # sign error in the denominator
        drops_inner = Fraction(1, c - k)  # applies f directly to c instead of to g(c) = m
        if len({ans, forgot_f, sign_error, drops_inner}) == 4:
            break
    else:
        check(False, "could not resample m to avoid a point-eval distractor collision after 30 attempts")

    _verify_point_eval_numeric(a, k, m, c, ans)

    answer_text = rational_text(ans)
    forgot_f_text = rational_text(forgot_f)
    sign_error_text = rational_text(sign_error)
    drops_inner_text = rational_text(drops_inner)

    text = f"Given f(x) = {f_text} and g(x) = {g_text}, evaluate f(g({c}))."
    working = [
        f"g({c}) = sqrt({c} - {a}) = {m}",
        f"f({m}) = 1/({m} - {paren_negative(k)}) = {answer_text}",
    ]

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=pick_distractors(answer_text, [forgot_f_text, sign_error_text, drops_inner_text]),
    )


def _verify_reverse_numeric(a: int, k: int, answer_text: str) -> None:
    """Independent check for ``reverse``. g(f(x)) = sqrt(f(x) - a) is only
    real for x very close to k, so a uniformly random sample point would
    usually make ``math.sqrt`` raise. Use xv = k + 1/(a + j) for j in
    (1, 2, 3): this makes f(xv) = 1/(xv - k) = a + j exactly, so
    f(xv) - a = j > 0 for every j, keeping the sqrt real by construction.
    """
    expr = sympify(answer_text.replace("^", "**"))
    for j in (1, 2, 3):
        xv = k + 1.0 / (a + j)
        direct = math.sqrt(1.0 / (xv - k) - a)
        via_string = float(expr.subs(x, xv))
        check(
            abs(direct - via_string) < 1e-9,
            f"g(f({xv})) = {direct} by direct float arithmetic but answer_text "
            f"{answer_text!r} evaluates to {via_string}",
        )


def _build_reverse(a: int, k: int) -> dict:
    """Find g(f(x)) instead of f(g(x)) -- hard tier only. The answer here is
    literally the string the forward shape's own ``reversed_order``
    misconception already builds, now correct for this different question."""
    f_text = _f_text(k)
    g_text = _g_text(a)
    answer_text = f"sqrt({_fmt_shift(f_text, a)})"

    _verify_reverse_numeric(a, k, answer_text)

    # Misconceptions, mirrored from the forward shape's own.
    forward_instead = f"1/({_fmt_shift(g_text, k)})"  # computes f(g(x)) instead (the OLD answer)
    treats_as_addition = f"{f_text} + {g_text}"
    drops_inner = f"sqrt({_fmt_shift(_fmt_shift('x', k), a)})"  # g applied to x - k, omitting f's reciprocal

    check_pairwise_distinct(
        sympify(answer_text.replace("^", "**")),
        sympify(forward_instead.replace("^", "**")),
        sympify(treats_as_addition.replace("^", "**")),
        sympify(drops_inner.replace("^", "**")),
        msg="a reverse-composition misconception collides with the answer or another distractor",
    )

    text = f"Given f(x) = {f_text} and g(x) = {g_text}, find g(f(x))."
    working = [
        f"f(x) = {f_text}, so g(f(x)) means substituting {f_text} for x in g.",
        f"g({f_text}) = {answer_text}",
    ]

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=pick_distractors(answer_text, [forward_instead, treats_as_addition, drops_inner]),
    )


def _build_compose(rng, a: int, k: int, tier: str) -> dict:
    """Dispatcher: draw one prompt shape. ``forward_a``/``forward_b``/
    ``point_eval`` are available at every tier (>=3 shapes, satisfying
    test_prompt_variety); ``reverse`` is added to the pool at hard only, a
    shape that can never appear at a lower tier (deterministic by
    construction, satisfying test_tier_escalation)."""
    shapes = ["forward_a", "forward_b", "point_eval"]
    if tier == "hard":
        shapes = shapes + ["reverse"]
    shape = rng.choice(shapes)

    if shape == "forward_a":
        return _build_forward(a, k, phrasing="a")
    elif shape == "forward_b":
        return _build_forward(a, k, phrasing="b")
    elif shape == "point_eval":
        return _build_point_eval(rng, a, k)
    else:
        return _build_reverse(a, k)


def _build_when_defined(a: int, k: int, is_class_b: bool, tier: str = "medium") -> dict:
    f_text = _f_text(k)
    g_text = _g_text(a)

    if not is_class_b:
        # "No exclusion" case: g's range (y >= 0) can never equal a negative k.
        check(k < 0, "no-exclusion case (easy tier, or hard class A) requires k < 0")
        answer = f"Defined for all x >= {a}"
        candidates = [
            "Defined for all real x",
            f"Defined for all x <= {a}",
            # Misconception: applying the exclusion formula regardless of k's
            # sign, forgetting sqrt(...) can never equal a negative number.
            f"Defined for all x >= {a}, except x = {a + k * k}",
        ]
    else:
        # "Exclusion" case: sqrt(x - a) = k has exactly one solution.
        check(k >= 0, "exclusion case (medium tier, or hard class B) requires k >= 0")
        excluded = a + k * k
        # Independent check, a different route from "just read off which
        # branch of the fixture-draw `if` ran": confirm by direct
        # substitution that the excluded point genuinely is where
        # sqrt(x - a) = k, and that k >= 0 was actually true.
        check(
            abs(math.sqrt(excluded - a) - k) < 1e-9,
            f"excluded point x={excluded} does not satisfy sqrt(x - a) = k for a={a}, k={k}",
        )
        answer = f"Defined for all x >= {a}, except x = {excluded}"
        check(k != 0, "exclusion-case k must be nonzero so a - k*k differs from a + k*k")
        candidates = [
            "Defined for all real x",
            f"Defined for all x <= {a}",
            # Misconception: sign error solving sqrt(x - a) = k for x.
            f"Defined for all x >= {a}, except x = {a - k * k}",
        ]

    for candidate in candidates:
        check(candidate != answer, f"candidate distractor {candidate!r} equals the answer {answer!r}")

    if tier == "hard":
        # Hard tier supplies the pre-combined h(x) = f(g(x)) as a given, reusing the exact
        # composed-expression string _build_forward already constructs -- not re-derived a third
        # way. This is deterministic on every hard seed, not drawn.
        h_text = f"1/({_fmt_shift(g_text, k)})"
        text = (
            f"Given f(x) = {f_text} and g(x) = {g_text}, so h(x) = f(g(x)) = {h_text}, "
            "for what values of x is h(x) defined? Select the correct description."
        )
    else:
        text = (
            f"Given f(x) = {f_text} and g(x) = {g_text}, for what values of x is "
            "f(g(x)) defined? Select the correct description."
        )
    if not is_class_b:
        working = [
            f"f(x) is undefined only when its input is {k}. g(x) has range y >= 0 and {k} < 0, "
            f"so sqrt(x - {a}) can never equal {k}, "
            f"so f(g(x)) is defined on all of g's own domain, x >= {a}."
        ]
    else:
        working = [
            f"g(x) has range y >= 0. sqrt(x - {a}) = {k} has the one solution "
            f"x = {a} + {k}^2 = {a + k * k}, which must be excluded from x >= {a}."
        ]

    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=pick_distractors(answer, candidates),
    )


@generator(
    "function_composition",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def function_composition(rng, tier: str = "medium", mode: str = "compose"):
    """Shared fixture for ACMSM093 (find the composition) and ACMSM092
    (determine when the composition is defined): g(x) = sqrt(x - a),
    f(x) = 1/(x - k). ``a``, ``k`` and, at hard tier, ``is_class_b`` are all
    drawn via unconditional rng calls BEFORE branching on ``mode`` -- the
    branch only decides which text/answer/distractors dict to build from the
    already-fixed fixture, so the two skills' sources (which supply ``mode``
    via ``params``, never via ``difficulty``) see identical draws for a
    matching (difficulty, seed) pair. Every rng call the ``mode="compose"``
    shape draw and the ``mode="when_defined"`` hard-tier text need happens
    strictly inside ``_build_compose``/``_build_when_defined``, after this
    shared draw.
    """
    a = rng.randint(1, 12)
    if tier == "easy":
        k = rng.randint(-9, -1)
        is_class_b = False
    elif tier == "medium":
        k = rng.randint(1, 9)
        is_class_b = True
    elif tier == "hard":
        is_class_b = rng.choice([True, False])
        k = rng.randint(1, 9) if is_class_b else rng.randint(-9, -1)
    else:
        raise ValueError(f"tier must be 'easy', 'medium', or 'hard'; got {tier!r}")

    if mode == "compose":
        return _build_compose(rng, a, k, tier)
    elif mode == "when_defined":
        return _build_when_defined(a, k, is_class_b, tier)
    else:
        raise ValueError(f"mode must be 'compose' or 'when_defined'; got {mode!r}")
