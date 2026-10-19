"""Generates `r3k_markov` questions (easy, medium, hard) and the exam chain `exd_x_r3k_markov`.

``uc2-markov-n-step-three-state`` (``stochastic-processes``, answer ``fraction``): a Markov chain on three states given
by its transition matrix P (rows are the current state, columns the next).

* easy: one two-step transition probability P(X_2 = j | X_0 = i) = (P^2)_{ij}.
* medium: either P(X_n = j) for n = 2 or 3 from a stated initial distribution (two or three forward steps), or the
  probability of eventual absorption in state 1 from state 2 when states 1 and 3 are absorbing and state 2 can stay put.
* hard: either the expected number of steps to absorption from a transient state (two linear equations, state 3
  absorbing), or one component of the stationary distribution of an irreducible chain.
* exam (``exd_x_r3k_markov``): a chain with state 3 absorbing: (a) the matrix P^2 (a `matrix` part), (b) P(X_3 = 3 | X_0 =
  1), (c) the expected time to absorption from state 2, (d) the expected time to absorption from state 1 (follows
  through from (c)).

Every answer is worked out with exact fractions (matrix products, Gaussian elimination) and again with SymPy (matrix
powers, the fundamental matrix (I - Q)^-1, or the null space of P^T - I).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, phrase
from ._r3k_kit import clean_working, frac_distractors, frac_fields, fr, need, rat
from ._ucv1_linalg import fm, inverse, mat_str, mmul, mpow, rows_value
from ._ucv1_linalg import matrix_distractors as tagged_matrix_distractors


def _sym(p):
    return sympy.Matrix([[rat(v) for v in row] for row in p])


def _row(rng, d, zero_allowed=True):
    """A probability row (a, b, c)/d with at most one zero entry."""
    for _ in attempts(200, "a probability row"):
        a = rng.randint(0, d)
        b = rng.randint(0, d - a)
        c = d - a - b
        parts = [a, b, c]
        if parts.count(0) <= (1 if zero_allowed else 0):
            return [Fraction(v, d) for v in parts]
    raise AssertionError  # pragma: no cover


def _irreducible(p) -> bool:
    n = len(p)
    reach = [[p[i][j] > 0 or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    return all(all(r) for r in reach)


def _chain(rng, d):
    for _ in attempts(500, "an irreducible chain"):
        p = [_row(rng, d) for _ in range(3)]
        if _irreducible(p) and len({tuple(r) for r in p}) == 3:
            return p
    raise AssertionError  # pragma: no cover


def _absorbing_chain(rng, d):
    """States 1 and 2 transient, state 3 absorbing, with every state-1 and state-2 row non-trivial."""
    for _ in attempts(500, "an absorbing chain"):
        r1, r2 = _row(rng, d), _row(rng, d)
        p = [r1, r2, [Fraction(0), Fraction(0), Fraction(1)]]
        q = [[p[i][j] for j in range(2)] for i in range(2)]
        # absorption is certain exactly when I - Q is invertible
        if (
            sympy.Matrix([[rat(int(i == j)) - rat(q[i][j]) for j in range(2)] for i in range(2)]).det() != 0
            and r1[0] < 1
            and r2[1] < 1
        ):
            if r1[2] > 0 or r2[2] > 0:
                return p
    raise AssertionError  # pragma: no cover


def _two_site_scene(rng, kind):
    scenes = {
        "plain": (
            "A warehouse robot is at one of three charging stations, numbered 1, 2 and 3. Each minute it moves, or stays, "
            "according to the transition matrix P below, where row i gives the probabilities of the next station when the "
            "robot is at station i.",
            "A weather model classes each day as 1 (sunny), 2 (cloudy) or 3 (rainy). The day-to-day changes follow the "
            "transition matrix P below, where row i gives the probabilities for tomorrow when today is class i.",
            "A customer buys from one of three supermarkets, numbered 1, 2 and 3, each week. Switching follows the "
            "transition matrix P below, where row i gives the probabilities for next week when this week's shop is i.",
        ),
        "absorbing": (
            "A machine part is in one of three conditions each week: 1 (new), 2 (worn) or 3 (failed). A failed part stays "
            "failed. The weekly changes follow the transition matrix P below, where row i gives the probabilities for next "
            "week when the part is in condition i this week.",
            "A patient in a trial is, each month, in one of three states: 1 (symptomatic), 2 (improving) or 3 (discharged). "
            "A discharged patient stays discharged. Monthly changes follow the transition matrix P below, where row i "
            "gives the probabilities for next month when the patient is in state i this month.",
        ),
        "two_absorbing": (
            "A frog hops between three lily pads numbered 1, 2 and 3. Pads 1 and 3 have nets, so the frog stays once it "
            "lands there. From pad 2 it moves according to row 2 of the transition matrix P below, where row i gives the "
            "probabilities for the next pad when the frog is on pad i.",
            "A game piece moves on three squares numbered 1, 2 and 3. Squares 1 and 3 are traps from which it never "
            "leaves. The moves follow the transition matrix P below, where row i gives the probabilities for the next "
            "square when the piece is on square i.",
        ),
    }
    return rng.choice(scenes[kind])


# -- easy: a two-step transition probability -----------------------------------------------------------------------

_EASY = (
    "{scene} P = {matrix} Find the probability that the chain is in state {j} two steps from now, given that it is in "
    "state {i} now.",
    "{scene} P = {matrix} Given that the chain is in state {i} now, find P(X_2 = {j}), the probability that it is in state "
    "{j} after two steps.",
)


def _easy(rng):
    d = rng.choice((4, 5, 6, 8, 10))
    for _ in attempts(500, "a chain with a readable two-step probability"):
        p = _chain(rng, d)
        i, j = rng.randint(1, 3), rng.randint(1, 3)
        p2 = mmul(fm(p), fm(p))
        key = p2[i - 1][j - 1]
        if 0 < key < 1 and key.denominator <= 400:
            break
    check(
        key == sum(p[i - 1][k] * p[k][j - 1] for k in range(3)),
        "the product disagrees with the sum over the middle state",
    )
    sp = _sym(p)
    check(fr((sp**2)[i - 1, j - 1]) == key, "SymPy's matrix square disagrees")
    cands = [
        (p[i - 1][j - 1] ** 2, "r3k-markov-squared-the-entry"),
        (p[i - 1][j - 1], "r3k-markov-one-step-only"),
        (p2[j - 1][i - 1], "r3k-markov-transposed"),
        (sum(p[i - 1][k] + p[k][j - 1] for k in range(3)) / 6, "r3k-markov-added-not-multiplied"),
        (sum(p[i - 1][k] * p[j - 1][k] for k in range(3)), "r3k-markov-row-times-row"),
        (key - max(p[i - 1][k] * p[k][j - 1] for k in range(3)), "r3k-markov-missed-a-path"),
        (key / 2, "r3k-markov-averaged-the-paths"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = phrase(rng, _EASY, scene=_two_site_scene(rng, "plain"), matrix=mat_str(p), i=i, j=j)
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"P(X_2 = {j} | X_0 = {i}) = sum over the middle state k of P({i}→k) P(k→{j}).",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


# -- medium ------------------------------------------------------------------------------------------------------


def _medium_forward(rng):
    d = rng.choice((4, 5, 6, 8))
    p = _chain(rng, d)
    steps = rng.choice((2, 3))
    j = rng.randint(1, 3)
    for _ in attempts(100, "an initial distribution"):
        a = rng.randint(0, 6)
        b = rng.randint(0, 6 - a)
        c = 6 - a - b
        if (a, b, c).count(0) <= 1:
            break
    init = [Fraction(a, 6), Fraction(b, 6), Fraction(c, 6)]
    pn = mpow(fm(p), steps)
    key = sum(init[i] * pn[i][j - 1] for i in range(3))
    # second route: iterate the distribution one step at a time
    vec = list(init)
    for _ in range(steps):
        vec = [sum(vec[i] * p[i][k] for i in range(3)) for k in range(3)]
    check(vec[j - 1] == key, "stepping the distribution disagrees with the matrix power")
    check(fr((sympy.Matrix([[rat(v) for v in init]]) * _sym(p) ** steps)[0, j - 1]) == key, "SymPy disagrees")
    cands = [
        (sum(init[i] * mpow(fm(p), steps - 1)[i][j - 1] for i in range(3)), "r3k-markov-one-step-short"),
        (sum(init[i] * mpow(fm(p), steps + 1)[i][j - 1] for i in range(3)), "r3k-markov-one-step-too-many"),
        (sum(init[i] * p[i][j - 1] for i in range(3)) ** steps, "r3k-markov-powered-the-probability"),
        (sum(pn[i][j - 1] for i in range(3)) / 3, "r3k-markov-ignored-the-initial-distribution"),
        (sum(init[i] * pn[j - 1][i] for i in range(3)), "r3k-markov-transposed"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{_two_site_scene(rng, 'plain')} P = {mat_str(p)} The chain starts in state 1 with probability {init[0]}, in state 2 with "
        f"probability {init[1]} and in state 3 with probability {init[2]}. Find P(X_{steps} = {j})."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Multiply the row vector of initial probabilities by P, {steps} times.",
            f"P(X_{steps} = {j}) = {key}.",
        ],
        **frac_fields(key),
    )


def _medium_absorption(rng):
    d = rng.choice((4, 5, 6, 8, 10))
    for _ in attempts(500, "a middle row with both exits"):
        mid = _row(rng, d, zero_allowed=False)
        if mid[0] > 0 and mid[2] > 0 and mid[1] > 0:
            break
    p = [[Fraction(1), Fraction(0), Fraction(0)], mid, [Fraction(0), Fraction(0), Fraction(1)]]
    key = mid[0] / (mid[0] + mid[2])
    # second route: first-step analysis solved as a linear equation, and the fundamental matrix route
    h = sympy.Symbol("h")
    solved = sympy.solve(sympy.Eq(h, rat(mid[0]) + rat(mid[1]) * h), h)[0]
    check(fr(solved) == key, "first-step analysis disagrees with p21 / (p21 + p23)")
    long_run = mpow(fm(p), 60)
    check(abs(float(long_run[1][0]) - float(key)) < 1e-6, "the long-run row of P^n disagrees")
    cands = [
        (mid[0], "r3k-markov-one-step-only"),
        (mid[0] / (1 - mid[0]), "r3k-markov-absorption-wrong-denominator"),
        (mid[2] / (mid[0] + mid[2]), "r3k-markov-wrong-absorbing-state"),
        (mid[0] / mid[2], "r3k-markov-ratio-for-probability"),
        (mid[0] + mid[1] * mid[0], "r3k-markov-two-steps-only"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{_two_site_scene(rng, 'two_absorbing')} P = {mat_str(p)} The chain starts in state 2. Find the probability that it is "
        f"eventually absorbed in state 1."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Each step from state 2 stays put with probability {mid[1]}; whenever it leaves, it goes to state 1 with probability p21/(p21 + p23).",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


def _medium(rng):
    return (_medium_forward if rng.random() < 0.6 else _medium_absorption)(rng)


# -- hard ---------------------------------------------------------------------------------------------------------


def _times_to_absorption(p):
    """(m1, m2): expected steps to absorption from states 1 and 2, by Gauss-Jordan on (I - Q) m = 1."""
    q = [[p[0][0], p[0][1]], [p[1][0], p[1][1]]]
    a = [[Fraction(int(i == j)) - q[i][j] for j in range(2)] for i in range(2)]
    inv = inverse(a)
    return [inv[i][0] + inv[i][1] for i in range(2)]


def _hard_absorption_time(rng):
    d = rng.choice((4, 5, 6, 8, 10))
    p = _absorbing_chain(rng, d)
    m = _times_to_absorption(p)
    start = rng.choice((1, 2))
    key = m[start - 1]
    q = sympy.Matrix([[rat(p[0][0]), rat(p[0][1])], [rat(p[1][0]), rat(p[1][1])]])
    fundamental = (sympy.eye(2) - q).inv()
    sym_times = fundamental * sympy.Matrix([1, 1])
    check(fr(sym_times[start - 1]) == key, "the fundamental matrix disagrees with the linear equations")
    other = m[2 - start]
    pii = p[start - 1][start - 1]
    cands = [
        (1 / (1 - pii), "r3k-markov-ignored-the-other-transient-state"),
        (1 / p[start - 1][2] if p[start - 1][2] > 0 else None, "r3k-markov-direct-exit-only"),
        (other, "r3k-markov-wrong-start-state"),
        (m[0] + m[1], "r3k-markov-summed-both-starts"),
        (key - 1, "r3k-markov-forgot-the-step-count"),
        (key + 1, "r3k-markov-counted-states-not-steps"),
    ]
    dist = need(frac_distractors(key, cands, positive=True))
    text = (
        f"{_two_site_scene(rng, 'absorbing')} P = {mat_str(p)} The chain starts in state {start}. Find the "
        f"expected number of steps until it reaches state 3."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "Let m1 and m2 be the expected times from states 1 and 2: m_i = 1 + p_i1 m1 + p_i2 m2. Solve the pair.",
            f"The expected number of steps is {key}.",
        ],
        **frac_fields(key),
    )


def _stationary(p):
    """The stationary distribution by solving (P^T - I) pi = 0 with the normalisation row, in exact fractions."""
    a = [[p[j][i] - Fraction(int(i == j)) for j in range(3)] for i in range(3)]
    a[2] = [Fraction(1)] * 3
    inv = inverse(a)
    return [inv[i][2] for i in range(3)]


def _hard_stationary(rng):
    d = rng.choice((4, 5, 6, 8, 10))
    for _ in attempts(500, "a chain with a readable stationary distribution"):
        p = _chain(rng, d)
        pi = _stationary(p)
        if all(v > 0 for v in pi) and all(v.denominator <= 60 for v in pi) and len(set(pi)) > 1:
            break
    # second route: SymPy null space of P^T - I, normalised
    ns = (_sym(p).T - sympy.eye(3)).nullspace()
    check(len(ns) == 1, "stationary distribution is not unique")
    vec = ns[0] / sum(ns[0])
    check([fr(v) for v in vec] == pi, "SymPy's null space disagrees with the linear solve")
    check([sum(pi[i] * p[i][j] for i in range(3)) for j in range(3)] == pi, "pi P != pi")
    j = rng.randint(1, 3)
    key = pi[j - 1]
    # unnormalised solution with the last entry 1
    last = pi[2]
    unnorm = pi[j - 1] / last
    pt = (
        _stationary([[p[a][b] for a in range(3)] for b in range(3)])
        if _irreducible([[p[a][b] for a in range(3)] for b in range(3)])
        else None
    )
    cands = [
        (unnorm, "r3k-markov-stationary-unnormalised"),
        (pi[j - 1] / pi[0], "r3k-markov-stationary-unnormalised"),
        (pi[j - 1] / pi[1], "r3k-markov-stationary-unnormalised"),
        (pt[j - 1] if pt else None, "r3k-markov-stationary-columns-for-rows"),
        (Fraction(1, 3), "r3k-markov-stationary-assumed-uniform"),
        (sum(p[i][j - 1] for i in range(3)) / 3, "r3k-markov-stationary-column-average"),
        (p[0][j - 1], "r3k-markov-stationary-first-row-of-p"),
        (p[j - 1][j - 1], "r3k-markov-stationary-self-transition"),
        (pi[j % 3], "r3k-markov-stationary-wrong-component"),
        (pi[(j + 1) % 3], "r3k-markov-stationary-wrong-component"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{_two_site_scene(rng, 'plain')} P = {mat_str(p)} Find the long-run proportion of time that the chain spends in state {j} "
        f"(the component π_{j} of the stationary distribution π, which satisfies πP = π with the components summing to 1)."
    )
    return dict(
        text=text,
        distractors=dist,
        working=["Solve πP = π together with π_1 + π_2 + π_3 = 1.", f"π_{j} = {key}."],
        **frac_fields(key),
    )


def _hard(rng):
    return (_hard_absorption_time if rng.random() < 0.55 else _hard_stationary)(rng)


@generator(
    "r3k_markov",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_markov(rng, tier="easy"):
    """Three-state Markov chain questions; see the module docstring for the tiers."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


def _cd_distractors(p, m):
    """Distractors for parts (c) and (d); (d)'s never include the value that follow-through from one of (c)'s gives."""
    m1, m2 = m
    c_dist = need(
        frac_distractors(
            m2,
            [
                (1 / (1 - p[1][1]), "r3k-markov-ignored-the-other-transient-state"),
                (1 / p[1][2] if p[1][2] > 0 else None, "r3k-markov-direct-exit-only"),
                (m1, "r3k-markov-wrong-start-state"),
                (m2 - 1, "r3k-markov-forgot-the-step-count"),
                (m2 + 1, "r3k-markov-counted-states-not-steps"),
            ],
            positive=True,
        )
    )
    # A learner who is wrong in (c) but follows through correctly in (d) must never meet a (d) distractor: drop any
    # candidate that equals the follow-through value of one of (c)'s distractors.
    carried = {(1 + p[0][1] * _value(d.value)) / (1 - p[0][0]) for d in c_dist}
    d_cands = [
        (1 / (1 - p[0][0]), "r3k-markov-ignored-the-other-transient-state"),
        (1 / p[0][2] if p[0][2] > 0 else None, "r3k-markov-direct-exit-only"),
        (m2, "r3k-markov-wrong-start-state"),
        ((p[0][1] * m2) / (1 - p[0][0]), "r3k-markov-forgot-the-step-count"),
        (m1 + m2, "r3k-markov-summed-both-starts"),
        (m1 + 1, "r3k-markov-counted-states-not-steps"),
        (m1 - 1, "r3k-markov-forgot-the-step-count"),
    ]
    d_dist = need(frac_distractors(m1, [(v, t) for v, t in d_cands if v not in carried], positive=True))
    return c_dist, d_dist


@generator("exd_x_r3k_markov", level=19, presets={"exam": dict(tier="exam")})
@clean_working
def exd_x_r3k_markov(rng, tier="exam"):
    """P^2, a three-step probability, and the expected times to absorption from states 2 and 1 (the last from the first)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    d = rng.choice((4, 5, 6, 8))
    for _ in attempts(500, "an exam absorbing chain"):
        p = _absorbing_chain(rng, d)
        m = _times_to_absorption(p)
        if p[0][1] > 0 and m[0].denominator <= 40 and m[1].denominator <= 40:
            try:
                _cd_distractors(p, m)
            except VerificationError:
                continue
            break
    pf = fm(p)
    p2 = mmul(pf, pf)
    p3 = mmul(p2, pf)
    check(_sym(p) ** 2 == _sym(p2), "SymPy's P^2 disagrees")
    prob = p3[0][2]
    check(fr((_sym(p) ** 3)[0, 2]) == prob, "SymPy's P^3 entry disagrees")
    q = sympy.Matrix([[rat(p[0][0]), rat(p[0][1])], [rat(p[1][0]), rat(p[1][1])]])
    sym_times = (sympy.eye(2) - q).inv() * sympy.Matrix([1, 1])
    check([fr(v) for v in sym_times] == m, "the fundamental matrix disagrees")
    m1, m2 = m
    check(m1 == (1 + p[0][1] * m2) / (1 - p[0][0]), "m1 does not follow from m2")
    hadamard = [[v * v for v in row] for row in pf]
    a_dist = need(
        tagged_matrix_distractors(
            [
                (hadamard, "r3k-markov-squared-the-entry"),
                (mmul(pf, [list(r) for r in zip(*pf)]), "r3k-markov-row-times-row"),
                ([list(r) for r in zip(*p2)], "r3k-markov-transposed"),
                ([[2 * v for v in row] for row in pf], "r3k-markov-doubled-not-squared"),
            ],
            rows_value(p2),
        )
    )
    b_dist = need(
        frac_distractors(
            prob,
            [
                (p2[0][2], "r3k-markov-one-step-short"),
                (mpow(pf, 4)[0][2], "r3k-markov-one-step-too-many"),
                (p3[2][0], "r3k-markov-transposed"),
                (p[0][2] ** 3, "r3k-markov-squared-the-entry"),
            ],
            positive=True,
            at_most=Fraction(1),
        )
    )
    c_dist, d_dist = _cd_distractors(p, m)
    p2_text = mat_str(p2)
    scene = _two_site_scene(rng, "absorbing")
    ft_d = f"(1 + ({_ft(p[0][1])})*part_c)/(1 - ({_ft(p[0][0])}))"
    return dict(
        text=f"{scene} P = {mat_str(p)}",
        parts=[
            dict(
                prompt="Find the matrix P² of two-step transition probabilities.",
                marks=3,
                answer_kind="matrix",
                answer=rows_value(p2),
                answer_text=p2_text,
                answer_latex="plain:" + p2_text,
                working=["Multiply P by itself, row by column."],
                distractors=a_dist,
            ),
            dict(
                prompt="Given that the chain starts in state 1, find the probability that it is in state 3 after three steps.",
                marks=2,
                sensible_range=(0, 1),
                working=[f"Use the first row of P³: P(X_3 = 3 | X_0 = 1) = {prob}."],
                distractors=b_dist,
                **frac_fields(prob),
            ),
            dict(
                prompt="Find the expected number of steps until the chain reaches state 3, starting from state 2.",
                marks=3,
                sensible_range=(0, None),
                working=[
                    "Write m1 and m2 for the expected times from states 1 and 2 and solve m_i = 1 + p_i1 m1 + p_i2 m2.",
                    f"m2 = {m2}.",
                ],
                distractors=c_dist,
                **frac_fields(m2),
            ),
            dict(
                prompt="Hence find the expected number of steps until the chain reaches state 3, starting from state 1.",
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_d,
                working=[f"m1 = 1 + p11 m1 + p12 m2, so m1 = (1 + p12 m2)/(1 - p11) = {m1}."],
                distractors=d_dist,
                **frac_fields(m1),
            ),
        ],
    )


def _ft(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _value(stored) -> Fraction:
    """A distractor's stored exact value (an int or {"p", "q"}) as a Fraction."""
    return Fraction(stored["p"], stored["q"]) if isinstance(stored, dict) else Fraction(stored)
