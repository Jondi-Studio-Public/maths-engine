"""Generates `riemann_interchange` items: justify exchanging a limit and a Riemann integral for a uniformly
convergent sequence of elementary functions on a closed interval.

Every sequence is uniformly convergent (the sequences are built in `_analysis_kit`, each with its exact
supremum error checked against a grid and its integrals against Simpson's rule). The learner proves that the
limit of the integrals is the integral of the limit by bounding |integral of f_n - integral of f| by the interval
length times a bound that does not depend on x and tends to 0. The answer is marked against a rubric, so `answer`
is a marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: the uniform bound is supplied; prove the interchange and state the common value.
- medium: find a bound that does not depend on x (a direct estimate) and use it.
- hard: the bound needs calculus (the maximum of |f_n - f|), or an estimate that is not sharp but still tends to 0.
- exam: also evaluate the integral of f_n exactly and show its limit agrees; includes a right endpoint below 1,
  where the bound is attained at the end of the interval.
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

TAILS_I = (
    "Show exactly where uniform convergence is used.",
    "Make clear which step needs the bound to hold for every x at once.",
    "Give the reason for each step.",
)


def _fmt(x: float) -> str:
    return f"{x:g}"


def _intro(p: Prompt, fam: ak.Family) -> None:
    iv = Fm(fam.interval_tex, fam.interval)
    p.t("For n = 1, 2, 3, ... let ").m(Fm("f_{n}(x)=", "f_n(x) = ") + fam.fn).t(" on the interval ").m(iv).t(
        ", and let "
    )
    p.m(Fm("f(x)=", "f(x) = ") + fam.lim).t(". The sequence converges uniformly to ").m(Fm("f")).t(" on ").m(iv).t(". ")


def _bounds(fam: ak.Family) -> tuple[str, str]:
    if fam.key.startswith("power_below_one"):
        num, den = fam.notes["ratio"]
        return "0", f"({num}/{den})"
    return _fmt(fam.a), _fmt(fam.b)


def _bounds_tex(fam: ak.Family) -> tuple[str, str]:
    if fam.key.startswith("power_below_one"):
        num, den = fam.notes["ratio"]
        return "0", f"\\frac{{{num}}}{{{den}}}"
    return _fmt(fam.a), _fmt(fam.b)


def _statement(fam: ak.Family) -> Fm:
    a, b = _bounds(fam)
    at, bt = _bounds_tex(fam)
    return Fm(
        f"\\lim_{{n\\to\\infty}}\\int_{{{at}}}^{{{bt}}}f_{{n}}(x)\\,dx=\\int_{{{at}}}^{{{bt}}}f(x)\\,dx",
        f"lim (n→∞) ∫_{a}^{b} f_n(x) dx = ∫_{a}^{b} f(x) dx",
    )


def _length_text(fam: ak.Family) -> str:
    if fam.key.startswith("power_below_one"):
        return f"{fam.notes['ratio'][0]}/{fam.notes['ratio'][1]}"
    return _fmt(fam.b - fam.a)


def _item(
    rng,
    fam: ak.Family,
    *,
    give_bound: bool,
    need_calculus: bool,
    direct: bool,
) -> dict:
    a, b = _bounds(fam)
    length = fam.b - fam.a
    p = Prompt()
    _intro(p, fam)
    if give_bound:
        p.t("It is given that ").m(Fm("|f_{n}(x)-f(x)|\\le B_{n}", "|f_n(x) - f(x)| ≤ B_n")).t(
            " for every x in the interval, where "
        )
        p.m(Fm("B_{n}=", "B_n = ") + fam.bound_fm).t(" tends to 0. ")
        task = "Prove that "
    elif need_calculus:
        task = "By finding the largest value of the difference, or an estimate that does not depend on x, prove that "
    else:
        task = "By bounding the difference with a quantity that does not depend on x, prove that "
    p.t(task).m(_statement(fam)).t(", and state the common value. ")
    if direct:
        p.t("Also evaluate ").m(Fm("\\int f_{n}(x)\\,dx", "the integral of f_n")).t(
            " exactly and check directly that its limit is the same value. "
        )
    p.t(rng.choice(TAILS_I))
    s_fm = fam.sup_fm.plain
    sup_where = (
        f"The largest value of |f_n(x) - f(x)| is {s_fm}, reached at x = {_where(fam)}."
        if not give_bound
        else f"By the given bound, |f_n(x) - f(x)| ≤ {fam.bound_fm.plain} for every x."
    )
    ref = [
        f"The limit function is f(x) = {fam.lim.plain}, which is continuous on the interval, so every f_n and f can be integrated.",
        sup_where
        + (" The bound is the same for every x, which is what uniform convergence provides." if not give_bound else ""),
        f"For every n, |∫ f_n - ∫ f| = |∫ (f_n - f)| ≤ ∫ |f_n - f| ≤ ({_length_text(fam)}) × (the bound) because the interval has length {_fmt(length)}.",
        f"The bound tends to 0 as n tends to infinity, so |∫ f_n - ∫ f| tends to 0 and the limit of the integrals is ∫ f.",
        f"The common value is ∫ f = {fam.int_lim_fm.plain if fam.int_lim_fm else _fmt(_int_value(fam))}.",
    ]
    if direct and fam.int_fn_fm is not None:
        ref.append(
            f"Direct check: ∫ f_n = {fam.int_fn_fm.plain}, and its limit as n tends to infinity is {fam.int_lim_fm.plain}, agreeing with the value found."
        )
    steps = [
        (
            "correctness",
            f"identifies f and a bound for |f_n - f| that does not depend on x{', such as ' + s_fm if not give_bound else ''}",
        ),
        (
            "logical_completeness",
            "bounds the difference of the integrals by the length of the interval times the bound, and concludes the limit of the integrals is the integral of the limit",
        ),
        (
            "definitions_assumptions",
            "uses uniform convergence (one bound for all x at once) and notes f is integrable on the interval",
        ),
        ("communication", f"states the common value {fam.int_lim_fm.plain if fam.int_lim_fm else ''}".strip()),
    ]
    if direct and fam.int_fn_fm is not None:
        steps.append(
            (
                "correctness",
                f"evaluates the integral of f_n as {fam.int_fn_fm.plain} and finds its limit {fam.int_lim_fm.plain}",
            )
        )
    traps = [
        (
            "pointwise_only",
            "does it conclude the interchange from pointwise convergence alone, without a bound that holds for every x?",
        ),
        (
            "forgets_the_length",
            "does the estimate of the difference of the integrals include the length of the interval?",
        ),
        ("bound_depends_on_x", "does the bound it uses still contain x, so it is not a uniform bound?"),
    ]
    if fam.key.startswith("power_below_one"):
        traps.append(
            ("uses_one_as_the_bound", f"does it take sup x^n = 1, forgetting the interval ends at {a}, below 1?")
        )
    mark = marking(reference=ref, steps=steps, traps=traps)
    val = fam.int_lim_fm.plain if fam.int_lim_fm else _fmt(_int_value(fam))
    return finish(p, f"The limit of the integrals equals the integral of f, which is {val}", mark)


def _where(fam: ak.Family) -> str:
    x = fam.argmax(1) if fam.argmax else None
    key = fam.key
    if key.startswith("rational_decay"):
        return "1/sqrt(n)"
    if key.startswith("power_times_complement"):
        return "n/(n+1)"
    if key.startswith("cos_over_n") or key.startswith("root_abs"):
        return "0"
    if (
        key.startswith("power_below_one")
        or key.startswith("power_over_n")
        or key.startswith("square_plus")
        or key.startswith("shifted")
    ):
        return _fmt(fam.b) if not key.startswith("power_below_one") else "the right endpoint"
    return str(x)


def _int_value(fam: ak.Family) -> float:
    return fam.int_lim if fam.int_lim is not None else 0.0


def _shifted(rng, fam: ak.Family) -> ak.Family:
    """Add an offset to both f_n and f half the time, so the common value is not always 0."""
    if rng.random() < 0.4 or fam.int_fn is None:
        return fam
    if fam.lim.plain == "0":
        return ak.with_offset(fam, rng.choice((-2, -1, 1, 2, 3)), rng.choice((-3, -1, 0, 1, 2, 4)))
    return ak.with_offset(fam, 0, rng.choice((-3, -2, -1, 1, 2, 3)))


def _easy(rng) -> dict:
    fam = _shifted(rng, rng.choice(ak.UNIFORM_SIMPLE)(rng))
    return _item(rng, fam, give_bound=True, need_calculus=False, direct=False)


def _medium(rng) -> dict:
    pool = ak.UNIFORM_SIMPLE + [ak.UNIFORM_CALCULUS[0]]
    fam = _shifted(rng, rng.choice(pool)(rng))
    return _item(rng, fam, give_bound=False, need_calculus=False, direct=False)


def _hard(rng) -> dict:
    fam = _shifted(rng, rng.choice([ak.UNIFORM_CALCULUS[2], ak.UNIFORM_HARD[1], ak.UNIFORM_HARD[2]])(rng))
    return _item(rng, fam, give_bound=False, need_calculus=True, direct=False)


def _exam(rng) -> dict:
    pool = [
        ak.UNIFORM_HARD[0],
        ak.UNIFORM_HARD[0],
        ak.UNIFORM_CALCULUS[0],
        ak.UNIFORM_CALCULUS[2],
        ak.UNIFORM_HARD[1],
        ak.UNIFORM_HARD[2],
    ]
    fam = _shifted(rng, rng.choice(pool)(rng))
    return _item(rng, fam, give_bound=False, need_calculus=True, direct=True)


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("riemann_interchange", level=17, presets=_PRESETS)
def riemann_interchange(rng, variant="easy"):
    """`ucv1-riemann-integral-interchange`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
