"""Generates `uniform_convergence` items: decide whether a sequence of elementary functions converges uniformly on
a closed interval, and justify the bound or the failure.

The sequences are built in `_analysis_kit`, each with its exact supremum error s_n = sup |f_n - f| checked against a
grid of 801 points. Three kinds of question:

- a stated bound B_n that is valid (the bound is the supremum or a bound at least as large): prove the bound for
  every x and decide that the uniform error tends to 0;
- a stated bound that is too small (half the supremum): show it fails at a named point, find the true supremum, and
  still decide that the convergence is uniform;
- a pointwise limit with no bound: decide, and when the convergence is not uniform, exhibit points x_n at which
  the gap stays at least a fixed size.

The answer is marked against a rubric, so `answer` is a marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: sharp, simple bounds (x^k / n, cos(cx) / n, x^2 + x / n) and the classic x^n on [0, 1].
- medium: bounds that need an estimate or calculus (a shifted square, a root, x / (1 + n x^2)), and a bound that is
  too small for a simple sequence.
- hard: sequences whose error peaks at a moving point and does not shrink (n x / (1 + n^2 x^2), n x / (1 + n x)),
  and a too-small bound for a calculus sequence.
- exam: peaks of constant height (n x e^(-n x), x^n (1 - x^n)), an endpoint below 1, and x^n (1 - x), whose peak
  moves to 1 but shrinks to 0.
"""

from __future__ import annotations

from ..core import generator
from . import _analysis_kit as ak
from ._crit_kit import Fm, Prompt, finish, marking

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}

TAILS_U = (
    "Justify your decision with a bound or a point.",
    "Give the reason for each step.",
    "Show the calculation that decides it.",
)


def _head(p: Prompt, fam: ak.Family) -> None:
    iv = Fm(fam.interval_tex, fam.interval)
    p.t("For n = 1, 2, 3, ... let ").m(Fm("f_{n}(x)=", "f_n(x) = ") + fam.fn).t(" on the interval ").m(iv).t(
        ". The sequence converges pointwise to "
    )
    p.m(Fm("f(x)=", "f(x) = ") + fam.lim).t(". ")


def _where(fam: ak.Family) -> str:
    k = fam.key
    if k.startswith("rational_decay"):
        return "x = 1/sqrt(n)"
    if k.startswith("power_times_complement"):
        return "x = n/(n+1)"
    if k.startswith("cos_over_n") or k.startswith("root_abs"):
        return "x = 0"
    if k.startswith("power_below_one"):
        return "the right endpoint"
    if k.startswith("spike") or k.startswith("step"):
        return "x = 1/n"
    if k.startswith("power_difference") or k.startswith("power_unit"):
        return "x = 2^(-1/n)"
    return f"x = {fam.b:g}"


def _sc(rng, fam: ak.Family) -> ak.Family:
    """Multiply a zero-limit family by 2, 3 or 4 about half the time, for variety."""
    if fam.lim.plain == "0" and rng.random() < 0.55:
        return ak.scaled(fam, rng.choice((2, 3, 4)))
    return fam


def _valid_bound(rng, fam: ak.Family) -> dict:
    fam = _sc(rng, fam)
    p = Prompt()
    _head(p, fam)
    p.t("A student states that ").m(Fm("|f_{n}(x)-f(x)|\\le B_{n}", "|f_n(x) - f(x)| ≤ B_n")).t(
        " for every x in the interval, where "
    )
    p.m(Fm("B_{n}=", "B_n = ") + fam.bound_fm).t(". Prove that this bound is valid, and decide whether ").m(
        Fm("f_{n}")
    ).t(" converges to ").m(Fm("f"))
    p.t(" uniformly on the interval. ").t(rng.choice(TAILS_U))
    s = fam.sup_fm.plain
    ref = [
        f"The limit is f(x) = {fam.lim.plain}, so f_n(x) - f(x) can be written out and bounded for each x in the interval.",
        f"The largest value of |f_n(x) - f(x)| over the interval is {s}, reached at {_where(fam)}; so |f_n(x) - f(x)| ≤ {s}{'' if s == fam.bound_fm.plain else ' ≤ ' + fam.bound_fm.plain} for every x, and the stated bound holds.",
        f"B_n = {fam.bound_fm.plain} does not depend on x and tends to 0 as n tends to infinity.",
        "So the largest error over the interval tends to 0, and the convergence is uniform.",
    ]
    steps = [
        (
            "correctness",
            f"proves |f_n(x) - f(x)| ≤ {fam.bound_fm.plain} for every x in the interval, with the supremum {s} found or estimated correctly",
        ),
        (
            "logical_completeness",
            "shows the bound for all x at once (an estimate or the maximum), and then uses B_n → 0 to conclude uniform convergence",
        ),
        (
            "definitions_assumptions",
            "uses the definition: the largest error over the interval tends to 0 (or one N for all x)",
        ),
        ("communication", "states the verdict: the convergence is uniform"),
    ]
    traps = [
        ("tests_values_only", "does it check the bound at a few values of x and not for every x?"),
        ("pointwise_is_enough", "does it argue uniformity from pointwise convergence alone?"),
        ("bound_depends_on_x", "does the bound it proves still contain x?"),
    ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return finish(p, f"Uniform: the bound holds and the largest error {s} tends to 0", mark)


def _small_bound(rng, fam: ak.Family) -> dict:
    fam = _sc(rng, fam)
    s = fam.sup_fm
    half = Fm(f"\\frac{{1}}{{2}}\\left({s.tex}\\right)", f"(1/2)({s.plain})")
    p = Prompt()
    _head(p, fam)
    p.t("A student states that ").m(Fm("|f_{n}(x)-f(x)|\\le B_{n}", "|f_n(x) - f(x)| ≤ B_n")).t(
        " for every x in the interval, where "
    )
    p.m(Fm("B_{n}=", "B_n = ") + half).t(". Decide whether this bound is valid, and decide whether ").m(Fm("f_{n}")).t(
        " converges to "
    ).m(Fm("f"))
    p.t(" uniformly on the interval. ").t(rng.choice(TAILS_U))
    ref = [
        f"The bound is not valid. At {_where(fam)} the error is |f_n(x) - f(x)| = {s.plain}, which is twice B_n = {half.plain}, so the inequality fails at that point (for n = 1 this is at x = {fam.argmax(1):.3g}).",
        f"In fact the largest error over the interval is exactly {s.plain}, so |f_n(x) - f(x)| ≤ {s.plain} is the correct bound.",
        f"{s.plain} tends to 0 as n tends to infinity, so the largest error tends to 0 and the convergence is uniform even though the student's bound is wrong.",
    ]
    steps = [
        (
            "correctness",
            f"decides the stated bound is false and gives the point ({_where(fam)}) where the error is {s.plain}, which exceeds B_n",
        ),
        (
            "logical_completeness",
            "separates the two questions: the false bound, then the (true) uniform convergence from the correct supremum tending to 0",
        ),
        (
            "definitions_assumptions",
            "evaluates |f_n - f| at a point of the interval, and says one such point is enough to refute the bound",
        ),
        ("communication", "states both verdicts: bound not valid, convergence uniform"),
    ]
    traps = [
        (
            "wrong_bound_means_not_uniform",
            "does it say the convergence is not uniform because the student's bound is wrong?",
        ),
        ("accepts_the_bound", "does it accept the bound without checking the point where the error is largest?"),
        ("point_outside_interval", "is the refuting point inside the interval?"),
    ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return finish(
        p,
        f"The bound is not valid (it fails at {_where(fam)}); the convergence is uniform, the largest error being {s.plain}",
        mark,
    )


def _no_bound(rng, fam: ak.Family) -> dict:
    fam = _sc(rng, fam)
    p = Prompt()
    _head(p, fam)
    p.t("Decide whether ").m(Fm("f_{n}")).t(" converges to ").m(Fm("f")).t(
        " uniformly on the interval. If it does not, give points "
    )
    p.m(Fm("x_{n}")).t(" in the interval at which ").m(Fm("|f_{n}(x_{n})-f(x_{n})|", "|f_n(x_n) - f(x_n)|")).t(
        " stays at least a fixed positive size. "
    ).t(rng.choice(TAILS_U))
    w = fam.notes.get("witness", "")
    gap = fam.notes.get("gap")
    if not fam.uniform:
        gtxt = f"{gap:.3g}"
        ref = [
            f"The convergence is not uniform. Take {w}, which lies in the interval for every n.",
            f"At that point f_n(x_n) - f(x_n) has absolute value {gtxt} for every n (a constant that does not shrink), by direct substitution.",
            f"So the largest error over the interval is at least {gtxt} for every n; it does not tend to 0, and the convergence is not uniform.",
            "Pointwise convergence holds at each fixed x because the points x_n move as n grows; uniform convergence needs one bound for all x at once.",
        ]
        steps = [
            (
                "correctness",
                f"decides the convergence is not uniform and gives points such as {w} where the gap is {gtxt}",
            ),
            (
                "logical_completeness",
                "shows the gap at x_n stays at least a fixed size for every n (so the supremum does not tend to 0)",
            ),
            (
                "definitions_assumptions",
                "uses the definition: the largest error over the interval must tend to 0, or equivalently one N for all x",
            ),
            ("communication", "states the verdict and the witness points clearly"),
        ]
        traps = [
            ("pointwise_means_uniform", "does it argue from pointwise convergence that the convergence is uniform?"),
            (
                "fixed_x_test",
                "does it test a fixed x (where the sequence does converge to the limit) instead of points x_n that move with n?",
            ),
            ("gap_not_computed", f"does it actually compute the gap at its points x_n and show it is about {gtxt}?"),
        ]
        verdict = f"Not uniform: at {w} the error stays near {gtxt}"
    else:
        s = fam.sup_fm.plain
        ref = [
            f"The convergence is uniform. The largest value of |f_n(x) - f(x)| is {s}, reached at {_where(fam)}.",
            f"Since {s} tends to 0 as n tends to infinity, the largest error tends to 0.",
            "There are no points x_n that keep the gap at a fixed positive size, because every gap is at most the supremum, which tends to 0.",
        ]
        steps = [
            ("correctness", f"decides the convergence is uniform and finds the largest error {s}"),
            ("logical_completeness", "bounds |f_n - f| for all x (maximum or estimate) and shows the bound tends to 0"),
            ("definitions_assumptions", "uses the definition of uniform convergence"),
            ("communication", "says why no points x_n with a persistent gap exist"),
        ]
        traps = [
            ("looks_for_a_witness_and_finds_none", "does it conclude 'not uniform' because a guessed witness fails?"),
            ("pointwise_only", "does it show only pointwise convergence?"),
        ]
        verdict = f"Uniform: the largest error {s} tends to 0"
    mark = marking(reference=ref, steps=steps, traps=traps)
    return finish(p, verdict, mark)


def _easy(rng) -> dict:
    if rng.random() < 0.65:
        return _valid_bound(rng, rng.choice(ak.UNIFORM_SIMPLE)(rng))
    return _no_bound(rng, ak.power_unit())


def _medium(rng) -> dict:
    if rng.random() < 0.5:
        return _valid_bound(rng, rng.choice(ak.UNIFORM_CALCULUS)(rng))
    return _small_bound(rng, rng.choice(ak.UNIFORM_SIMPLE)(rng))


def _hard(rng) -> dict:
    r = rng.random()
    if r < 0.6:
        return _no_bound(rng, rng.choice([ak.spike_rational, ak.step_rational])())
    return _small_bound(rng, rng.choice(ak.UNIFORM_CALCULUS)(rng))


def _exam(rng) -> dict:
    r = rng.random()
    if r < 0.4:
        return _no_bound(rng, rng.choice([ak.spike_exponential, ak.power_difference])())
    if r < 0.6:
        return _no_bound(rng, rng.choice(ak.UNIFORM_HARD)(rng))
    if r < 0.8:
        return _valid_bound(rng, rng.choice(ak.UNIFORM_HARD)(rng))
    return _small_bound(rng, rng.choice(ak.UNIFORM_HARD)(rng))


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("uniform_convergence", level=17, presets=_PRESETS)
def uniform_convergence(rng, variant="easy"):
    """`ucv1-uniform-convergence-test`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
