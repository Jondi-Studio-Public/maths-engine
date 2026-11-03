"""Recipe subject ``exponential_g``: g(x) = a^(2x) - b*a^x + c, set the way VCE Methods Exam 1 sets it (2023 Q4).

Maths and wrong-answer candidates live in ``mathgen/subjects/exponential_g.py``; this module is the composer glue
(``@subject`` and ``@ask``, see ``docs/design/recipes.md``). Logarithm answers are typed with natural logs
(``ln(5)/ln(2)``): the typed-answer parser reads ``log_b(x)`` only for b = 2 and 10, and the subject uses bases 2, 3 and 5. Every answer is exact, so the subject is the same
whether or not ``technology_active`` is set.

Asks: ``evaluate_at_log`` (g at x = log_a k, an integer), ``integer_solution`` (the whole-number root of g(x) = 0),
``log_solution`` (the other root, ``log_a(r)``), ``solve_shifted`` (the larger solution of g(x) = d, a new log).
"""

from __future__ import annotations

import math

from ..answers import Distractor
from ..core import check, latex_answer, phrase
from ..recipe_vocabulary import property_check
from ..recipes import Subject, ask, subject
from ..subjects import exponential_g as eg
from ._exd_x_w12c_kit import _accepts, int_part


_EVAL = (
    "Evaluate g(log_{a}({k})).",
    "Find the value of g(log_{a}({k})).",
    "Show that g(log_{a}({k})) is a whole number, and find it.",
)
_INT = (
    "Solve g(x) = 0 for the value of x that is a whole number.",
    "One solution of g(x) = 0 is a whole number. Find it.",
    "Find the whole-number value of x for which g(x) = 0.",
)
_LOG = (
    "Solve g(x) = 0 for the other value of x. Give it as an exact logarithm, typed with natural logs, like ln(5)/ln(2).",
    "Find the other solution of g(x) = 0 as an exact logarithm, typed with natural logs, like ln(5)/ln(2).",
    "The other solution of g(x) = 0 is not a whole number. Give it exactly, typed with natural logs, like ln(5)/ln(2).",
)
_SHIFT = (
    "Now solve g(x) = {d}. Give the larger solution as an exact logarithm, typed with natural logs, like ln(5)/ln(2).",
    "Find the larger solution of g(x) = {d} exactly, typed with natural logs, like ln(5)/ln(2).",
    "The equation g(x) = {d} has two solutions. Give the larger one as an exact logarithm, typed like ln(5)/ln(2).",
)


def _log_expr(base: int, arg: int) -> str:
    return f"log({arg})/log({base})"


def _log_part(prompt, base, arg, marks, working, cands):
    """An ``expression`` part whose key is log_base(arg); wrong texts the grader accepts are dropped."""
    text = f"log_{base}({arg})"
    key = _log_expr(base, arg)
    options, seen = [], {text}
    for wrong, tag in cands:
        if wrong in seen or _accepts("expression", text, key, wrong):
            continue
        seen.add(wrong)
        options.append(Distractor(wrong, tag, None))
    picked = options[:3]
    check(len(picked) >= 2, "fewer than two usable distractors")
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=key,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
    )


@subject("exponential_g")
class ExponentialG(Subject):
    families = frozenset({"exponential"})
    properties = frozenset({"quadratic_in_exponential", "has_log_root"})

    def __init__(self, s: eg.ExpG):
        self.s = s

    @classmethod
    def draw(cls, rng, technology_active):
        return cls(eg.draw(rng))

    def stem(self) -> str:
        s = self.s
        return f"Let g(x) = {s.a}^(2x) - {s.b}*{s.a}^x + {s.c}, for x in R."


@ask(None, "evaluate_at_log", needs=("quadratic_in_exponential",), default_marks=2)
def evaluate_at_log(subject, rng, ctx):
    s = subject.s
    k = eg.eval_k(rng, s)
    key = s.g_at_log(k)
    check(abs(key - (math.exp(2 * math.log(k)) - s.b * k + s.c)) < 1e-6, "g(log_a k) does not match")
    working = [
        f"{s.a}^(2 log_{s.a}({k})) = ({s.a}^(log_{s.a}({k})))^2 = {k}^2 = {k * k}.",
        f"g(log_{s.a}({k})) = {k * k} - {s.b}*{k} + {s.c} = {key}.",
    ]
    return int_part(
        phrase(rng, _EVAL, a=s.a, k=k),
        key,
        2,
        working,
        [(w, t) for w, t in eg.eval_wrong(s, k) if w != s.m],  # never show part (b)'s key
        lo=None,
        hi=None,
    )


@ask(None, "integer_solution", needs=("quadratic_in_exponential",), default_marks=1)
def integer_solution(subject, rng, ctx):
    s = subject.s
    working = [
        f"Let t = {s.a}^x: t^2 - {s.b}t + {s.c} = (t - {s.r1})(t - {s.r2}) = 0.",
        f"{s.a}^x = {s.r1} gives x = {s.m}.",
    ]
    wrong = [
        (s.r1, "exp-quadratic-root-given-as-x"),
        (s.r2, "exp-quadratic-root-given-as-x"),
        (s.r1 // s.a, "exp-root-divided-by-base"),
    ]
    wrong = [(w, t) for w, t in wrong if w != s.m and w > 0]
    return int_part(phrase(rng, _INT), s.m, 1, working, wrong, lo=0, hi=None)


@ask(None, "log_solution", needs=("quadratic_in_exponential",), default_marks=1)
def log_solution(subject, rng, ctx):
    s = subject.s
    working = [
        f"With t = {s.a}^x the other root is t = {s.r2}.",
        f"{s.a}^x = {s.r2} gives x = log_{s.a}({s.r2}).",
    ]
    return _log_part(
        phrase(rng, _LOG),
        s.a,
        s.r2,
        1,
        working,
        eg.log_root_wrong(s),
    )


@ask(None, "solve_shifted", needs=("quadratic_in_exponential",), default_marks=2, requires=("log_solution",))
def solve_shifted(subject, rng, ctx):
    s = subject.s
    p, q = rng.choice(eg.shifted_roots(s))
    d0 = s.c - p * q
    working = [
        f"g(x) = {d0} gives t^2 - {s.b}t + {s.c - d0} = (t - {p})(t - {q}) = 0 with t = {s.a}^x.",
        f"The larger root is t = {q}, so x = log_{s.a}({q}).",
    ]
    cands = [
        (f"log_{s.a}({p})", "exp-other-solution-given"),
        (str(q), "exp-quadratic-root-given-as-x"),
        (f"-log_{s.a}({q})", "exp-log-sign-dropped"),
        (f"log_{s.a}({p * q})", "exp-log-wrong-argument"),
        (f"log_{q}({s.a})", "log-base-and-argument-swapped"),
    ]
    seen = set()
    for part in ctx.earlier:
        seen.add(part["answer_text"])
        seen.update(d.text for d in part.get("distractors", ()))
    cands = [(w, t) for w, t in cands if w not in seen]
    return _log_part(phrase(rng, _SHIFT, a=s.a, d=d0), s.a, q, 2, working, cands)


@property_check("quadratic_in_exponential")
def _quadratic_in_exponential(subject) -> bool:
    """g(x) really is t^2 - b t + c in t = a^x: g at x = 0 and x = 1 agree with the quadratic."""
    if not isinstance(subject, ExponentialG):
        return True
    s = subject.s
    return all(abs(s.a ** (2 * x) - s.b * s.a**x + s.c - (s.a**x - s.r1) * (s.a**x - s.r2)) < 1e-9 for x in (0, 1, 2))


@property_check("has_log_root")
def _has_log_root(subject) -> bool:
    """One solution of g(x) = 0 is not a whole number."""
    if not isinstance(subject, ExponentialG):
        return True
    x = subject.s.log_root_value()
    return abs(x - round(x)) > 1e-9
