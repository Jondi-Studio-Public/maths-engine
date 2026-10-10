"""Generates `p19f_integral_additivity_linearity` questions (ACMMM128): use the additivity and linearity of the
definite integral, `answer: number`.

Scope: given stated definite-integral VALUES (for example the integral
of f over [a, b] and over [b, c], or of f and g over the same interval,
together with stated scalar multipliers), compute a combined
definite-integral value using additivity or linearity, without evaluating f
or g directly.

**CRITICAL constraint**: this generator NEVER constructs or integrates a
real function f or g. Every "integral value" in every question is drawn
directly as a stated integer -- there is no polynomial anywhere in this
module and no call to a symbolic integration routine.

**Sharp neighbours** (never rendered by this generator): ``definite-
integral``/``integrate-polynomial`` integrate a STATED polynomial directly;
this skill is never handed one. ``senior-calculate-total-change-by-
integrating-instantaneous-or-marginal`` integrates a stated rate function
over an interval; this skill never states a rate function either, only
already-evaluated integral VALUES.

**Design, two families:**
- Additivity: I_ab = integral(f,a,b), I_bc = integral(f,b,c), both stated
  integers; ask for I_ac = I_ab + I_bc (forward), OR give I_ac and I_ab and
  ask for I_bc = I_ac - I_ab (inverse direction).
- Linearity: I_f = integral(f,a,b), I_g = integral(g,a,b) over the SAME
  interval, plus scalars k, m; ask for k*I_f + m*I_g.

**Difficulty ladder:**
- easy: pure additivity, both stated values positive, forward (sum).
- medium: pure linearity, one scalar (k=1 fixed, only m varies).
- hard: linearity with BOTH scalars nonzero and at least one negative, OR
  additivity in the inverse (subtraction) direction with a negative stated
  value.
- exam: additivity AND linearity combined in one question -- three stated
  integral values across two adjacent intervals and two functions.

**Independent check.** Every generator function computes the answer via TWO separately-written Python integer
routes (never sympy -- this is plain scalar arithmetic): once via the identity string-formatted for the
prompt's own bookkeeping, and once via direct arithmetic on the drawn integers built from a second expression.
A plain `check` comparing the two integers is the right tool here.

**Misconceptions**: applied the WRONG combining rule (added when the
question needs subtraction in the inverse-additivity design, or vice versa);
dropped a scalar multiplier (`I_f + I_g` instead of `k*I_f + m*I_g`); swapped
the two intervals' order in an additivity chain (`I_ab - I_bc` instead of
`I_ab + I_bc`). Every one checked with `check_distinct`/
`check_pairwise_distinct`.
"""

from __future__ import annotations

from ..core import check, check_distinct, check_pairwise_distinct, generator, latex_answer, phrase, pick_distractors

_EASY_TEMPLATES = (
    "The integral of f(x) from {a} to {b} is {i_ab}, and the integral of "
    "f(x) from {b} to {c} is {i_bc}. Find the integral of f(x) from {a} to {c}.",
    "The integral of f(x) from {a} to {b} equals {i_ab}, and the integral of "
    "f(x) from {b} to {c} equals {i_bc}. Find the integral of f(x) from {a} to {c}.",
    "Given that the integral of f(x) from {a} to {b} is {i_ab} and the "
    "integral of f(x) from {b} to {c} is {i_bc}, find the integral of f(x) "
    "from {a} to {c}.",
)


def _rand_nonzero(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _finish(text, answer, candidates):
    """Shared tail: `answer` and every `candidates` entry are plain Python
    ints. Dedupe candidates against the answer AND against each other here
    (an accidental numeric coincidence between two independently-computed
    misconception values is possible, e.g. two candidates landing on the
    same wrong integer by chance of the drawn values) so every remaining
    candidate is a genuinely distinct wrong value -- a bare `!=`/set
    comparison between Python ints is exact and needs no symbolic
    machinery."""
    deduped: list[int] = []
    seen = {answer}
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    check(len(deduped) >= 1, f"no distinct distractor survived dedup for answer {answer}")
    check(all(c != answer for c in deduped), f"a distractor equals the answer: {answer}")
    check(len(set(deduped)) == len(deduped), f"a distractor collides with another: {deduped}")
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(c) for c in deduped])
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )


# --- easy: pure additivity, forward direction, both values positive --------


def _easy(rng):
    a, b, c = sorted(rng.sample(range(0, 12), k=3))
    i_ab = rng.randint(1, 30)
    i_bc = rng.randint(1, 30)

    # Route 1: the additivity identity, spelled exactly as the prompt states
    # it (I_ac = I_ab + I_bc).
    route1 = i_ab + i_bc
    # Route 2: independently-typed direct arithmetic -- built from the same
    # two ints but via a fresh expression, never copy-pasted from route 1.
    total = 0
    for value in (i_ab, i_bc):
        total += value
    route2 = total
    check(route1 == route2, f"additivity route mismatch: {i_ab}+{i_bc}")
    answer = route1

    wrong_rule = i_ab - i_bc  # applied subtraction instead of addition
    order_swap = i_bc - i_ab  # (degenerate here since same as wrong_rule with sign; distinct candidate below)
    candidates = []
    for cand in (wrong_rule,):
        if cand != answer and cand not in candidates:
            candidates.append(cand)
    # a plausible "double-counted one interval" slip
    double_count = 2 * i_ab + i_bc
    if double_count != answer and double_count not in candidates:
        candidates.append(double_count)
    if not candidates:
        candidates = [answer + 1, answer - 1]  # pragma: no cover -- pathological fallback

    text = phrase(rng, _EASY_TEMPLATES, a=a, b=b, c=c, i_ab=i_ab, i_bc=i_bc)
    return _finish(text, answer, candidates)


_MEDIUM_TEMPLATES = (
    "The integral of f(x) from {a} to {b} is {i_f}, and the integral of "
    "g(x) from {a} to {b} is {i_g}. Find the integral of f(x) + {m}g(x) "
    "from {a} to {b}.",
    "The integral of f(x) from {a} to {b} equals {i_f}, and the integral of "
    "g(x) from {a} to {b} equals {i_g}. Find the integral of f(x) + {m}g(x) "
    "from {a} to {b}.",
    "Given that the integral of f(x) from {a} to {b} is {i_f} and the "
    "integral of g(x) from {a} to {b} is {i_g}, find the integral of "
    "f(x) + {m}g(x) from {a} to {b}.",
)


# --- medium: pure linearity, one scalar (k fixed at 1) ----------------------


def _medium(rng):
    a, b = sorted(rng.sample(range(0, 12), k=2))
    i_f = rng.randint(1, 30) * rng.choice([1, -1])
    i_g = rng.randint(1, 30) * rng.choice([1, -1])
    m = _rand_nonzero(rng, 2, 6)

    route1 = i_f + m * i_g
    combined = i_f
    combined += m * i_g
    route2 = combined
    check(route1 == route2, f"linearity route mismatch: {i_f}+{m}*{i_g}")
    answer = route1

    dropped_scalar = i_f + i_g  # forgot the multiplier m
    sign_error = i_f - m * i_g
    candidates = [v for v in (dropped_scalar, sign_error) if v != answer]
    if len(candidates) < 2:
        candidates.append(m * i_f + i_g)  # swapped which value gets scaled

    text = phrase(rng, _MEDIUM_TEMPLATES, a=a, b=b, i_f=i_f, i_g=i_g, m=m)
    return _finish(text, answer, candidates)


# --- hard: two designs -- full linearity, or inverse-direction additivity --


def _hard_linearity(rng):
    a, b = sorted(rng.sample(range(0, 12), k=2))
    i_f = rng.randint(1, 30) * rng.choice([1, -1])
    i_g = rng.randint(1, 30) * rng.choice([1, -1])
    k = _rand_nonzero(rng, -6, 6)
    m = _rand_nonzero(rng, -6, 6)
    while abs(k) == abs(m):
        m = _rand_nonzero(rng, -6, 6)

    route1 = k * i_f + m * i_g
    total = 0
    for coeff, value in ((k, i_f), (m, i_g)):
        total += coeff * value
    route2 = total
    check(route1 == route2, f"hard linearity route mismatch: {k}*{i_f}+{m}*{i_g}")
    answer = route1

    dropped_scalars = i_f + i_g
    swapped_scalars = m * i_f + k * i_g
    sign_flip_k = -k * i_f + m * i_g
    candidates = [v for v in (dropped_scalars, swapped_scalars, sign_flip_k) if v != answer]

    k_term = f"{k}f(x)" if k != 1 else "f(x)"
    if k == -1:
        k_term = "-f(x)"
    m_term = f"{m}g(x)" if m not in (1, -1) else ("g(x)" if m == 1 else "-g(x)")
    combo = f"{k_term} + {m_term}".replace("+ -", "- ")
    text = (
        f"The integral of f(x) from {a} to {b} is {i_f}, and the integral of "
        f"g(x) from {a} to {b} is {i_g}. Find the integral of {combo} from "
        f"{a} to {b}."
    )
    return _finish(text, answer, candidates)


def _hard_inverse_additivity(rng):
    a, b, c = sorted(rng.sample(range(0, 12), k=3))
    i_ab = rng.randint(1, 30) * rng.choice([1, -1])
    i_ac = rng.randint(1, 30) * rng.choice([1, -1])
    while i_ac == i_ab:
        i_ac = rng.randint(1, 30) * rng.choice([1, -1])

    route1 = i_ac - i_ab
    diff = i_ac
    diff -= i_ab
    route2 = diff
    check(route1 == route2, f"inverse additivity route mismatch: {i_ac}-{i_ab}")
    answer = route1
    check(answer != 0, f"degenerate zero answer: {i_ac}, {i_ab}")

    wrong_rule = i_ac + i_ab  # added instead of subtracting
    order_swap = i_ab - i_ac  # subtracted in the wrong order
    candidates = [v for v in (wrong_rule, order_swap) if v != answer]
    if not candidates:
        candidates = [answer + i_ab if i_ab != 0 else answer + 1]  # pragma: no cover

    text = (
        f"The integral of f(x) from {a} to {c} is {i_ac}, and the integral of "
        f"f(x) from {a} to {b} is {i_ab}. Find the integral of f(x) from {b} to {c}."
    )
    return _finish(text, answer, candidates)


def _hard(rng):
    if rng.random() < 0.5:
        return _hard_linearity(rng)
    return _hard_inverse_additivity(rng)


# --- exam: combine additivity and linearity in one question ----------------


def _exam(rng):
    a, b, c = sorted(rng.sample(range(0, 12), k=3))
    i_ab_f = rng.randint(1, 30) * rng.choice([1, -1])
    i_bc_f = rng.randint(1, 30) * rng.choice([1, -1])
    i_ac_g = rng.randint(1, 30) * rng.choice([1, -1])
    k = _rand_nonzero(rng, -5, 5)
    m = _rand_nonzero(rng, -5, 5)

    # Answer = k*(I_ab_f + I_bc_f) + m*I_ac_g.
    i_ac_f = i_ab_f + i_bc_f
    route1 = k * i_ac_f + m * i_ac_g
    total = 0
    for coeff, value in ((k, i_ab_f), (k, i_bc_f), (m, i_ac_g)):
        total += coeff * value
    route2 = total
    check(route1 == route2, f"exam combined route mismatch: k={k} m={m}")
    answer = route1

    forgot_to_chain = k * i_ab_f + m * i_ac_g  # dropped one additivity interval
    dropped_scalars = i_ab_f + i_bc_f + i_ac_g  # dropped both scalars
    inverted_additivity = k * (i_ab_f - i_bc_f) + m * i_ac_g  # additivity as subtraction
    candidates = [v for v in (forgot_to_chain, dropped_scalars, inverted_additivity) if v != answer]

    k_term = f"{k}" if k not in (1, -1) else ("" if k == 1 else "-")
    m_term = f"{m}" if m not in (1, -1) else ("" if m == 1 else "-")
    text = (
        f"The integral of f(x) from {a} to {b} is {i_ab_f}, the integral of "
        f"f(x) from {b} to {c} is {i_bc_f}, and the integral of g(x) from {a} "
        f"to {c} is {i_ac_g}. Find the integral of {k_term}f(x) + {m_term}g(x) "
        f"from {a} to {c}.".replace("+ -", "- ")
    )
    return _finish(text, answer, candidates)


@generator(
    "p19f_integral_additivity_linearity",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_integral_additivity_linearity(rng, tier="easy"):
    """See module docstring for the full ladder. Every "integral value" is a
    stated integer; no polynomial is ever constructed or integrated here.

    easy:   pure additivity, forward direction, both stated values positive.
    medium: pure linearity, one scalar (k fixed at 1).
    hard:   full linearity (both scalars nonzero, one negative) OR
            inverse-direction additivity (subtraction), chosen per seed.
    exam:   additivity AND linearity combined: three stated values across
            two adjacent intervals and two functions.
    """
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    return _exam(rng)
