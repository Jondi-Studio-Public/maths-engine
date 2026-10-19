"""Generates `r3k_geometric` questions (easy, medium, hard) and the exam chain `exd_x_r3k_geometric`.

``uc2-geometric-negative-binomial`` (``probability``, answer ``fraction``): the number of independent trials up to the
first success (geometric) or up to the r-th success (negative binomial).

* easy: P(X = k) or P(X > k) for the trial number X of the first success.
* medium: P(Y = n) where Y is the trial on which the r-th success happens (r = 2 or 3), where the last trial must be a
  success and the earlier r - 1 successes can fall anywhere before it.
* hard: one of four structural questions: P(X is even); the chance the player who goes first in a two-player race to a
  first success wins; Var(Y) for the r-th success; or a memoryless conditional probability.
* exam (``exd_x_r3k_geometric``): (a) P(X = k), (b) P(Y = n), (c) Var(X), (d) Var(Y) (follows through from (c)).

Every answer is worked out from a closed form and re-derived by a different route: a dynamic-programming recursion over
the trials (an exact binomial table), the probability generating function differentiated with SymPy, a long partial
sum, or first-step analysis solved as a linear equation.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._r3k_kit import clean_working, frac_distractors, frac_fields, fr, need, rat

# (setting sentence with {p}, one trial, trials, a success, a success with its article)
_SCENES = (
    (
        "A netballer takes shots at goal, and each shot scores with probability {p}, independently of the others.",
        "shot",
        "shots",
        "goal",
        "a goal",
    ),
    (
        "Each battery from a production line is tested, and fails the test with probability {p}, independently of the others.",
        "battery",
        "batteries",
        "failure",
        "a failure",
    ),
    (
        "A call centre rings households at random, and each call is answered with probability {p}, independently of the others.",
        "call",
        "calls",
        "answered call",
        "an answered call",
    ),
    (
        "A surf lifesaver throws a rescue line at a training target, and each throw lands on the target with probability {p}, independently of the others.",
        "throw",
        "throws",
        "hit",
        "a hit",
    ),
    (
        "A seed tray is checked one seed at a time, and each seed germinates with probability {p}, independently of the others.",
        "seed",
        "seeds",
        "germination",
        "a germination",
    ),
)


def _prob(rng, lo_den=3, hi_den=8):
    """A success probability a/b with a small denominator, between 1/6 and 3/5."""
    for _ in attempts(500, "a success probability"):
        b = rng.randint(lo_den, hi_den)
        a = rng.randint(1, b - 1)
        p = Fraction(a, b)
        if Fraction(1, 6) <= p <= Fraction(3, 5) and p != Fraction(1, 2):
            return p
    raise AssertionError  # pragma: no cover


def _ptxt(p: Fraction) -> str:
    return f"{p.numerator}/{p.denominator}"


def _scene(rng, p):
    setting, trial, trials, success, a_success = rng.choice(_SCENES)
    return setting.format(p=_ptxt(p)), trial, trials, success, a_success


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")


# -- the second route: exact recursions and generating functions ------------------------------------------------


def _binomial_table(p: Fraction, n: int) -> list[Fraction]:
    """P(S_n = j) for j = 0..n, by the recursion P(S_m = j) = q P(S_{m-1} = j) + p P(S_{m-1} = j-1)."""
    q = 1 - p
    row = [Fraction(1)]
    for _ in range(n):
        nxt = [Fraction(0)] * (len(row) + 1)
        for j, mass in enumerate(row):
            nxt[j] += q * mass
            nxt[j + 1] += p * mass
        row = nxt
    return row


def _prob_trial_of_rth_success(p: Fraction, r: int, n: int) -> Fraction:
    """P(r-th success on trial n) = P(exactly r-1 successes in the first n-1 trials) * p, from the recursion."""
    return _binomial_table(p, n - 1)[r - 1] * p if r - 1 <= n - 1 else Fraction(0)


def _pgf_variance(p: Fraction, r: int) -> Fraction:
    """Var of the trial of the r-th success from its probability generating function (p s / (1 - q s))^r."""
    s = sympy.Symbol("s")
    g = (rat(p) * s / (1 - rat(1 - p) * s)) ** r
    d1 = sympy.diff(g, s).subs(s, 1)
    d2 = sympy.diff(g, s, 2).subs(s, 1)
    return fr(sympy.nsimplify(d2 + d1 - d1**2))


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_EQ = (
    "{scene} Let X be the number of {trials} up to and including the first {success}. Find P(X = {k}).",
    "{scene} The {trials} are numbered 1, 2, 3, ... in order, and X is the number of the first {trial} that produces {a_success}. Find P(X = {k}).",
)
_EASY_GT = (
    "{scene} Let X be the number of {trials} up to and including the first {success}. Find P(X > {k}).",
    "{scene} The {trials} are numbered 1, 2, 3, ... in order, and X is the number of the first {trial} that produces {a_success}. Find the probability that more than {k} {trials} are needed.",
)


def _easy(rng):
    p = _prob(rng)
    q = 1 - p
    k = rng.randint(3, 6)
    scene, trial, trials, success, a_success = _scene(rng, p)
    if rng.random() < 0.5:
        key = q ** (k - 1) * p
        # second route: sum the first k masses of the recursion "failures then a success"
        table = [(_binomial_table(p, j - 1)[0]) * p for j in range(1, k + 1)]
        check(table[k - 1] == key, "the recursion disagrees with q^(k-1) p")
        cands = [
            (q**k * p, "r3k-geometric-exponent-off-by-one"),
            (k * p * q ** (k - 1), "r3k-geometric-binomial-for-first-success"),
            (q ** (k - 1), "r3k-geometric-forgot-the-success"),
            (p ** (k - 1) * q, "r3k-geometric-swapped-p-and-q"),
        ]
        text = phrase(rng, _EASY_EQ, scene=scene, trial=trial, trials=trials, success=success, a_success=a_success, k=k)
        working = [f"The first {k - 1} {trials} must all fail and the {_ordinal(k)} must succeed: q^({k - 1}) p."]
    else:
        key = q**k
        sf = _binomial_table(p, k)[0]
        check(sf == key, "the recursion disagrees with q^k")
        cands = [
            (1 - q**k, "r3k-geometric-tail-complement-slip"),
            (q ** (k - 1), "r3k-geometric-exponent-off-by-one"),
            (q ** (k - 1) * p, "r3k-geometric-tail-as-point-mass"),
            (p**k, "r3k-geometric-swapped-p-and-q"),
        ]
        text = phrase(rng, _EASY_GT, scene=scene, trial=trial, trials=trials, success=success, a_success=a_success, k=k)
        working = [f"More than {k} {trials} are needed exactly when the first {k} all fail: q^{k}."]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    return dict(text=text, distractors=dist, working=working + [f"The probability is {key}."], **frac_fields(key))


# -- medium -------------------------------------------------------------------------------------------------------

_MEDIUM = (
    "{scene} Find the probability that the {rth} {success} occurs on {trial} number {n}.",
    "{scene} Let Y be the number of {trials} up to and including the {rth} {success}. Find P(Y = {n}).",
)


def _medium(rng):
    p = _prob(rng)
    q = 1 - p
    r = rng.choice((2, 3))
    n = r + rng.randint(2, 4)
    scene, trial, trials, success, a_success = _scene(rng, p)
    key = Fraction(sympy.binomial(n - 1, r - 1)) * p**r * q ** (n - r)
    check(key == _prob_trial_of_rth_success(p, r, n), "the recursion disagrees with C(n-1, r-1) p^r q^(n-r)")
    cands = [
        (Fraction(sympy.binomial(n, r)) * p**r * q ** (n - r), "r3k-negbin-binomial-for-rth-success"),
        (p**r * q ** (n - r), "r3k-negbin-forgot-the-arrangements"),
        (Fraction(sympy.binomial(n, r - 1)) * p**r * q ** (n - r), "r3k-negbin-wrong-choose-argument"),
        (q ** (n - 1) * p, "r3k-negbin-geometric-for-rth-success"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    return dict(
        text=phrase(
            rng,
            _MEDIUM,
            scene=scene,
            trial=trial,
            trials=trials,
            success=success,
            a_success=a_success,
            rth=_ordinal(r),
            n=n,
        ),
        distractors=dist,
        working=[
            f"The {_ordinal(r)} {success} is on {trial} {n}, so {trial} {n} succeeds and exactly {r - 1} of the first "
            f"{n - 1} succeed: C({n - 1}, {r - 1}) p^{r} q^{n - r}.",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_even(rng):
    p = _prob(rng, 3, 9)
    q = 1 - p
    scene, trial, trials, success, a_success = _scene(rng, p)
    key = q / (1 + q)
    partial = sum(q ** (k - 1) * p for k in range(2, 400, 2))
    check(abs(float(key - partial)) < 1e-12, "the partial sum of the even masses disagrees with q/(1+q)")
    cands = [
        (1 / (1 + q), "r3k-geometric-even-for-odd"),
        (q**2 / (1 - q**2), "r3k-geometric-even-wrong-start"),
        (q / (1 - q**2), "r3k-geometric-even-forgot-p"),
        (q, "r3k-geometric-even-first-term-only"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{scene} Let X be the number of {trials} up to and including the first {success}. "
        f"Find the probability that X is even."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "P(X even) = qp + q^3 p + q^5 p + ... is a geometric series with first term qp and ratio q^2.",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


_PLAYERS = (("Mia", "Jack"), ("Priya", "Tom"), ("Chloe", "Liam"), ("Ava", "Noah"))


def _hard_race(rng):
    pa = _prob(rng, 3, 8)
    pb = _prob(rng, 3, 8)
    qa, qb = 1 - pa, 1 - pb
    check(qa * qb != 1, "no race")
    a, b = rng.choice(_PLAYERS)
    key = pa / (1 - qa * qb)
    w = sympy.Symbol("w")
    solved = sympy.solve(sympy.Eq(w, rat(pa) + rat(qa * qb) * w), w)[0]  # first-step analysis
    check(fr(solved) == key, "first-step analysis disagrees with p_A / (1 - q_A q_B)")
    b_wins = qa * pb / (1 - qa * qb)
    check(key + b_wins == 1, "the two win probabilities must sum to 1")
    cands = [
        (b_wins, "r3k-race-wrong-winner"),
        (pa / (pa + pb), "r3k-race-ratio-of-probabilities"),
        (pa / (1 - qa), "r3k-race-ignores-the-opponent"),
        (pa + qa * qb * pa, "r3k-race-two-rounds-only"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{a} and {b} take turns trying to hit a target, with {a} going first. {a} hits with probability "
        f"{_ptxt(pa)} on each attempt and {b} with probability {_ptxt(pb)}, all attempts being independent. "
        f"The first player to hit wins. Find the probability that {a} wins."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"{a} wins in round j if both players miss in every earlier round and {a} then hits: a geometric series with ratio q_A q_B.",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


def _hard_variance(rng):
    p = _prob(rng, 3, 8)
    q = 1 - p
    r = rng.choice((2, 3, 4))
    scene, trial, trials, success, a_success = _scene(rng, p)
    key = r * q / p**2
    check(_pgf_variance(p, r) == key, "the generating function disagrees with r q / p^2")
    cands = [
        (r * q / p, "r3k-negbin-variance-forgot-square"),
        (r / p**2, "r3k-negbin-variance-forgot-q"),
        (q / p**2, "r3k-negbin-variance-single-geometric"),
        (r * p / q**2, "r3k-negbin-variance-swapped-p-and-q"),
    ]
    dist = need(frac_distractors(key, cands, positive=True))
    text = f"{scene} Let Y be the number of {trials} up to and including the {_ordinal(r)} {success}. Find Var(Y)."
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"Y is the sum of {r} independent geometric waiting times, each with variance q/p^2.",
            f"Var(Y) = {key}.",
        ],
        **frac_fields(key),
    )


def _hard_memoryless(rng):
    p = _prob(rng, 3, 6)
    q = 1 - p
    m = rng.randint(2, 5)
    k = rng.randint(2, 4)
    scene, trial, trials, success, a_success = _scene(rng, p)
    key = 1 - q**k
    # second route: the conditional probability straight from its definition with the tail masses of the recursion
    num = _binomial_table(p, m)[0] - _binomial_table(p, m + k)[0]
    den = _binomial_table(p, m)[0]
    check(num / den == key, "the definition P(A and B)/P(B) disagrees with 1 - q^k")
    cands = [
        (1 - q ** (m + k), "r3k-geometric-ignores-the-memoryless-property"),
        (q**k, "r3k-geometric-tail-complement-slip"),
        (1 - q ** (k - 1), "r3k-geometric-exponent-off-by-one"),
        (p * q ** (k - 1), "r3k-geometric-point-mass-for-cumulative"),
    ]
    dist = need(frac_distractors(key, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{scene} Let X be the number of {trials} up to and including the first {success}. "
        f"Given that none of the first {m} {trials} produced {a_success}, find the probability that "
        f"the first {success} occurs within the next {k} {trials}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "P(X <= m + k | X > m) = (q^m - q^(m+k)) / q^m = 1 - q^k.",
            f"The probability is {key}.",
        ],
        **frac_fields(key),
    )


@generator(
    "r3k_geometric",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_geometric(rng, tier="easy"):
    """Geometric and negative binomial trials-to-success probabilities; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_even, _hard_race, _hard_variance, _hard_memoryless))(rng)


# -- exam ---------------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "{scene} Let X be the number of {trials} up to and including the first {success}, and let Y be the number of "
    "{trials} up to and including the {rth} {success}.",
    "{scene} The {trials} are numbered 1, 2, 3, ... in order. X is the number of the first {trial} that produces a "
    "{success}, and Y is the number of the {trial} that produces the {rth} {success}.",
)


@generator("exd_x_r3k_geometric", level=18, presets={"exam": dict(tier="exam")})
@clean_working
def exd_x_r3k_geometric(rng, tier="exam"):
    """P(X = k), P(Y = n), Var(X) and Var(Y), the last following through from Var(X)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = _prob(rng, 3, 8)
    q = 1 - p
    r = rng.choice((2, 3))
    k = rng.randint(3, 6)
    n = r + rng.randint(2, 4)
    scene, trial, trials, success, a_success = _scene(rng, p)
    pa = q ** (k - 1) * p
    check([_binomial_table(p, k - 1)[0] * p] == [pa], "the recursion disagrees with q^(k-1) p")
    pb = Fraction(sympy.binomial(n - 1, r - 1)) * p**r * q ** (n - r)
    check(pb == _prob_trial_of_rth_success(p, r, n), "the recursion disagrees with C(n-1, r-1) p^r q^(n-r)")
    vx = q / p**2
    check(_pgf_variance(p, 1) == vx, "the generating function disagrees with q / p^2")
    vy = r * vx
    check(_pgf_variance(p, r) == vy, "the generating function disagrees with r q / p^2")
    a_dist = need(
        frac_distractors(
            pa,
            [
                (q**k * p, "r3k-geometric-exponent-off-by-one"),
                (k * p * q ** (k - 1), "r3k-geometric-binomial-for-first-success"),
                (q ** (k - 1), "r3k-geometric-forgot-the-success"),
            ],
            positive=True,
            at_most=Fraction(1),
        )
    )
    b_dist = need(
        frac_distractors(
            pb,
            [
                (Fraction(sympy.binomial(n, r)) * p**r * q ** (n - r), "r3k-negbin-binomial-for-rth-success"),
                (p**r * q ** (n - r), "r3k-negbin-forgot-the-arrangements"),
                (q ** (n - 1) * p, "r3k-negbin-geometric-for-rth-success"),
            ],
            positive=True,
            at_most=Fraction(1),
        )
    )
    c_dist = need(
        frac_distractors(
            vx,
            [
                (1 / p**2, "r3k-negbin-variance-forgot-q"),
                (q / p, "r3k-negbin-variance-forgot-square"),
                (p / q**2, "r3k-negbin-variance-swapped-p-and-q"),
            ],
            positive=True,
        )
    )
    d_dist = need(
        frac_distractors(
            vy,
            [
                (r * q / p, "r3k-negbin-variance-forgot-square"),
                (r**2 * vx, "r3k-negbin-variance-scaled-like-a-multiple"),
                (vx, "r3k-negbin-variance-single-geometric"),
            ],
            positive=True,
        )
    )
    rth = _ordinal(r)
    return dict(
        text=phrase(
            rng, _EXAM_STEMS, scene=scene, trial=trial, trials=trials, success=success, a_success=a_success, rth=rth
        ),
        parts=[
            dict(
                prompt=f"Find P(X = {k}).",
                marks=2,
                sensible_range=(0, 1),
                working=[f"The first {k - 1} {trials} fail and the next succeeds: q^{k - 1} p = {pa}."],
                distractors=a_dist,
                **frac_fields(pa),
            ),
            dict(
                prompt=f"Find P(Y = {n}).",
                marks=3,
                sensible_range=(0, 1),
                working=[f"C({n - 1}, {r - 1}) p^{r} q^{n - r} = {pb}."],
                distractors=b_dist,
                **frac_fields(pb),
            ),
            dict(
                prompt="Find Var(X).",
                marks=2,
                sensible_range=(0, None),
                working=[f"Var(X) = q/p^2 = {vx}."],
                distractors=c_dist,
                **frac_fields(vx),
            ),
            dict(
                prompt="Hence find Var(Y).",
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{r}*part_c",
                working=[f"Y is a sum of {r} independent copies of X, so Var(Y) = {r} Var(X) = {vy}."],
                distractors=d_dist,
                **frac_fields(vy),
            ),
        ],
    )
