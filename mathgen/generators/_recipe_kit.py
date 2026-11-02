"""Shared pieces for the exact-answer recipe subjects (`exd_recipe_*`).

Property names
--------------
A subject declares `Subject.properties` from the constants below, and a generic ask (`@ask(None, name, needs=(...))`)
is valid for any subject that carries every property in its `needs`. The names are provisional (a light rename is
expected), so every module imports them from here and no string literal repeats them.

* `PDF_DENSITY`: a continuous density on an interval. The subject supplies `density` (a SymPy expression in `X` and
  its unknown parameters), `lo`, `hi` (SymPy values or symbols), `rv` (the variable's name) and `threshold` (a point
  inside the support, for tail probabilities).
* `UNKNOWN_PARAMETERS`: the density has unknown constants `unknowns = (first, second)`, with `param_names`, their true
  values `truth`, the mean `given_mean` the question states, and the `valid(values)` test that says which solutions
  are admissible. The subject's `analysis` holds the solved relation and roots.
* `DISCRETE_DISTRIBUTION`: a distribution on whole numbers. The subject supplies `rv`, `support`, `pmf(j)` (a
  Fraction) and the integers the question asks about.

Subject-specific wrong answers come from hook methods on the subject (`relation_slips`, `point_slips`, ...) because
only the subject knows which slips its own working invites; the maths of every generic ask is derived from the
properties above and nothing else.
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..grading import GRADERS
from ._r3f_kit import ThinOptions
from types import SimpleNamespace

PDF_DENSITY = "bounded_support"
UNKNOWN_PARAMETERS = "unknown_constants_2"
DISCRETE_DISTRIBUTION = None  # no vocabulary property yet: the binomial asks are per-subject (exact_binomial)

X = sympy.Symbol("x")


def accepted(kind: str, prompt: str, answer, answer_text: str, reply: str, answer_form=None) -> bool:
    """Whether the real grader accepts `reply` for a part with this key."""
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=answer_form, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def formula(expr) -> str:
    """A SymPy expression as a learner would type it: one fraction, no stars, `^` for powers (`2(3 - d)/(3d^2)`)."""

    def text(e) -> str:
        return str(e).replace("**", "^").replace("*", "")

    numer, denom = sympy.fraction(sympy.together(expr))
    if denom == 1:
        return text(numer)
    top = f"({text(numer)})" if numer.is_Add else text(numer)
    bottom = f"({text(denom)})" if denom.is_Add or denom.is_Mul else text(denom)
    return f"{top}/{bottom}"


def expression_part(prompt, expr, text, marks, working, slips, *, answer_form=None, follow_through=None, keep=3):
    """A part answered by an exact expression. `text` writes `expr`; each slip is `(expression, misconception id,
    text)`. A slip is dropped when it repeats the key or an earlier slip, or when the real grader accepts it."""
    expr = sympy.sympify(expr)
    answer = str(expr)
    options: list[Distractor] = []
    seen = {text}
    for slip, tag, shown in slips:
        if shown in seen or sympy.simplify(sympy.sympify(slip) - expr) == 0:
            continue
        if accepted("expression", prompt, answer, text, shown, answer_form):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag))
        if len(options) == keep:
            break
    if len(options) < 2:
        raise ThinOptions(f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=answer,
        answer_text=text,
        answer_latex=sympy.latex(expr),
        marks=marks,
        working=working,
        distractors=options,
    )
    if answer_form:
        part["answer_form"] = answer_form
    if follow_through:
        part["follow_through"] = follow_through
    return part
