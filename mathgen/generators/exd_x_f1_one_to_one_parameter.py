"""Exam-tier source for ``senior-determine-if-a-function-is-one-to-one``.

The practice tier reads the turning point off a function written in vertex form. Here the quadratic
is in standard form ``a x^2 + p x + c``, and the question is posed about a RESTRICTED DOMAIN:

* ``parameter``: the domain is ``x >= k`` or ``x <= k`` and the learner must find the extreme value
  of the parameter ``p`` for which the restriction is one-to-one. The vertex ``-p / (2a)`` has to be
  compared with ``k``, and whether the answer is a "least" or a "greatest" value depends on both the
  sign of ``a`` and the direction of the domain.
* ``endpoint``: ``p`` is a number and the learner finds the largest (or smallest) ``k`` for which
  the function is one-to-one on ``(-inf, k]`` (or ``[k, inf)``); that is the vertex, found from the
  standard form.

Whenever the vertex is ``h = -p / (2a)``, ``f`` is one-to-one on a half-line with endpoint ``k``
exactly when ``k`` is on the correct side of ``h``. The threshold for ``p`` is ``-2 a k``. All
answers are integers.
"""

from __future__ import annotations

import random

from ..core import check, generator, join_terms, phrase, pick_distractors

_LATEX_REL = {"≥": r"\ge", "≤": r"\le"}

_PARAM_WORDINGS = (
    "The function f(x) = {poly} is defined for {domain}. Find the {extreme} value of p for which f is one-to-one.",
    "Let f(x) = {poly}, where {domain}. What is the {extreme} value of p that makes f one-to-one?",
    "For what {extreme} value of p is f(x) = {poly}, {domain}, a one-to-one function?",
)
_END_WORDINGS = (
    "The function f(x) = {poly} is defined for {domain}. Find the {extreme} value of k for which f is one-to-one.",
    "Let f(x) = {poly}, where {domain}. What is the {extreme} value of k for which f is one-to-one?",
)


def _latex_domain(domain: str) -> str:
    return "".join(_LATEX_REL.get(ch, ch) for ch in domain)


def _poly(a: int, p, c: int) -> str:
    """``a x^2 + p x + c`` as text; ``p`` may be the letter p."""
    terms = [(a, "x^2"), (p, "x") if isinstance(p, int) else (1, "px"), (c, "")]
    return join_terms(terms)


def _parameter(rng: random.Random) -> dict:
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    k = rng.choice([n for n in range(-5, 7) if n != 0])
    c = rng.choice([n for n in range(-7, 8) if n != 0])
    upward = rng.random() < 0.5  # domain x >= k
    # f(u) = f(v), u != v  <=>  u + v = -p / a. On x >= k that sum can be anything above 2k.
    # So f is one-to-one iff -p/a <= 2k, i.e. p >= -2ak when a > 0 and p <= -2ak when a < 0.
    # On x <= k the inequalities reverse.
    least = (a > 0) == upward
    answer = -2 * a * k
    wrong = [2 * a * k, -2 * k, -a * k, -k, k]
    domain = f"x ≥ {k}" if upward else f"x ≤ {k}"
    text = phrase(rng, _PARAM_WORDINGS, poly=_poly(a, "p", c), domain=domain, extreme="least" if least else "greatest")
    latex = rf"f(x) = {_poly(a, 'p', c)},\ {_latex_domain(domain)}"
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        wrong=wrong,
        working=[f"f is one-to-one on the half-line when the vertex x = -p/{2 * a} is not inside it: p = {answer}."],
    )


def _endpoint(rng: random.Random) -> dict:
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    h = rng.choice([n for n in range(-5, 7) if n != 0])
    p = -2 * a * h
    c = rng.choice([n for n in range(-7, 8) if n != 0])
    # One-to-one on (-inf, k] needs k <= h (greatest k = h); on [k, inf) needs k >= h (least k = h).
    upward = rng.random() < 0.5
    extreme = "least" if upward else "greatest"
    domain = "x ≥ k" if upward else "x ≤ k"
    text = phrase(rng, _END_WORDINGS, poly=_poly(a, p, c), domain=domain, extreme=extreme)
    latex = rf"f(x) = {_poly(a, p, c)},\ {_latex_domain(domain)}"
    wrong = [-h, 2 * h, p, c, -p]
    return dict(
        text=text,
        latex=latex,
        answer=h,
        wrong=wrong,
        working=[f"The vertex is at x = -({p})/(2 × {a}) = {h}, so the {extreme} value of k is {h}."],
    )


def _build(rng: random.Random) -> dict:
    for _ in range(100):
        d = _parameter(rng) if rng.random() < 0.6 else _endpoint(rng)
        wrong = []
        for w in d["wrong"]:
            if w != d["answer"] and w not in wrong:
                wrong.append(w)
        if len(wrong) < 3:
            continue
        wrong = wrong[:3]
        check(all(w != d["answer"] for w in wrong))
        answer_text = str(d["answer"])
        return dict(
            text=d["text"],
            latex=d["latex"],
            answer=d["answer"],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=d["working"],
            distractors=pick_distractors(answer_text, [str(w) for w in wrong]),
        )
    check(False, "could not build a one-to-one question")


@generator(
    "exd_x_f1_one_to_one_parameter",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_one_to_one_parameter(rng: random.Random, tier: str = "exam") -> dict:
    """Restricted-domain one-to-one condition on a quadratic in standard form, with a parameter."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
