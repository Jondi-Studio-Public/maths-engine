"""Generates `r3i_series_sums` questions (easy, medium, hard) and the exam chain `r3i_exam_series_sums`.

Skill ``arithmetic-geometric-series-sums`` (``functions-sequences``, answer ``number``): sums of arithmetic and
geometric series. The stem never says which kind of sequence it is dealing with in the easy tier, and the formula is
never given.

* easy: the first three terms of a sequence are listed (arithmetic or geometric, not named) and the learner sums the
  first ``n`` terms.
* medium: one of the number of terms that gives a stated sum (a quadratic for an arithmetic series, a power for a
  geometric one), or a term found from the formula for the sum of the first ``n`` terms by taking
  ``S(n) - S(n - 1)``.
* hard: either an arithmetic series known only through two of its partial sums (a pair of equations), or a geometric
  series known through its sum to infinity and its second term, where the quadratic for the first term has a second
  root that does not converge and is rejected.
* exam (``r3i_exam_series_sums``): a savings plan (arithmetic) or a geometric series with a stated sum to infinity;
  four parts, the later ones following through from the earlier.

Every sum is re-derived by adding the terms one at a time, and every number of terms by searching for the first ``n``
whose running total reaches the stated sum.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3i_kit import frac, num_cands, num_fields, pick, retry_short, signed_text


def s_ap(a, d, n):
    return Fraction(n) * (2 * Fraction(a) + (n - 1) * Fraction(d)) / 2


def s_gp(a, r, n):
    a, r = Fraction(a), Fraction(r)
    return a * n if r == 1 else a * (r**n - 1) / (r - 1)


def _run_ap(a, d, n):
    total, term = Fraction(0), Fraction(a)
    for _ in range(n):
        total += term
        term += d
    return total


def _run_gp(a, r, n):
    total, term = Fraction(0), Fraction(a)
    for _ in range(n):
        total += term
        term *= r
    return total


def _terms(a, step, geometric, count=3) -> str:
    out, t = [], Fraction(a)
    for _ in range(count):
        out.append(str(int(t)))
        t = t * step if geometric else t + step
    return ", ".join(out) + ", ..."


# -- easy ---------------------------------------------------------------------------------------------------


def _easy(rng):
    if rng.random() < 0.5:
        a, d, n = rng.randint(2, 20), rng.choice([2, 3, 4, 5, 6, 7, -2, -3]), rng.randint(12, 30)
        if d < 0:
            a = abs(d) * n + rng.randint(5, 20)
        ans = s_ap(a, d, n)
        check(ans == _run_ap(a, d, n), "arithmetic sum differs from adding term by term")
        seq = _terms(a, d, False)
        cands = [
            (n * (2 * a + n * d) / 2, "r3i-ap-sum-n-instead-of-n-minus-1"),
            (n * (a + d * (n - 1)), "r3i-ap-sum-forgot-halving"),
            (a + (n - 1) * d, "r3i-sum-gave-last-term"),
            (s_gp(a, d, n) if abs(d) <= 7 and n <= 12 else None, "r3i-sum-wrong-series-formula"),
            (s_ap(a, d, n - 1), "r3i-sum-one-term-short"),
        ]
        working = [
            f"The terms differ by {d}, so the sequence is arithmetic with a = {a} and d = {d}.",
            f"S{n} = {n}/2 × (2 × {a} + {n - 1} × {d}) = {int(ans)}.",
        ]
    else:
        a, r, n = rng.randint(1, 5), rng.choice([2, 3, -2, 2, 3, 4]), rng.randint(5, 9)
        if abs(r) == 4:
            n = rng.randint(4, 6)
        ans = s_gp(a, r, n)
        check(ans == _run_gp(a, r, n), "geometric sum differs from adding term by term")
        seq = _terms(a, r, True)
        cands = [
            (a * (r ** (n - 1) - 1) / (r - 1), "r3i-gp-sum-power-off-by-one"),
            (a * r ** (n - 1), "r3i-sum-gave-last-term"),
            (s_ap(a, a * r - a, n), "r3i-sum-wrong-series-formula"),
            (a * (r**n + 1) / (r + 1) if r != -1 else None, "r3i-gp-sum-sign-slip"),
            (a * (r**n - 1), "r3i-gp-sum-forgot-divide-by-r-minus-1"),
        ]
        working = [
            f"Each term is {r} times the one before, so the sequence is geometric with a = {a} and r = {r}.",
            f"S{n} = {a}({r}^{n} - 1)/({r} - 1) = {int(ans)}.",
        ]
    fields = num_fields(ans)
    stem = phrase(
        rng,
        (
            "The first three terms of a sequence are {seq} Find the sum of the first {n} terms.",
            "A sequence begins {seq} Find the sum of its first {n} terms.",
            "The first three terms of a sequence are {seq} Find the total of the first {n} terms.",
        ),
        seq=seq,
        n=n,
    )
    return dict(text=stem, working=working, distractors=pick(fields, num_cands(cands, ans)), **fields)


# -- medium -------------------------------------------------------------------------------------------------


def _medium_count(rng):
    if rng.random() < 0.55:
        a, d, n = rng.randint(2, 15), rng.randint(2, 7), rng.randint(9, 26)
        total = s_ap(a, d, n)
        # second route: the first n whose running total reaches the stated sum
        run, term, found = Fraction(0), Fraction(a), 0
        for k in range(1, 200):
            run += term
            term += d
            if run == total:
                found = k
                break
        check(found == n, "the number of terms differs from searching the running total")
        cands = [
            (total / (a + (n - 1) * d) if total % (a + (n - 1) * d) == 0 else None, "r3i-terms-sum-over-last-term"),
            (Fraction(total, a) if Fraction(total, a).denominator == 1 else None, "r3i-terms-sum-over-first-term"),
            (n + 1, "r3i-terms-off-by-one"),
            (n - 1, "r3i-terms-off-by-one"),
            (Fraction(total - a, d) + 1, "r3i-terms-treated-sum-as-last-term"),
        ]
        seq = _terms(a, d, False)
        stem = f"How many terms of the arithmetic series {seq.replace(', ...', ' + ...').replace(', ', ' + ')} must be added to give a sum of {int(total)}?"
        working = [
            f"a = {a}, d = {d}. Set n/2 × (2 × {a} + (n - 1) × {d}) = {int(total)}.",
            f"{d}n²{signed_text(2 * a - d)}n - {2 * int(total)} = 0, whose positive solution is n = {n}.",
        ]
    else:
        a, r, n = rng.randint(1, 5), rng.choice([2, 3]), rng.randint(5, 10)
        total = s_gp(a, r, n)
        run, term, found = Fraction(0), Fraction(a), 0
        for k in range(1, 60):
            run += term
            term *= r
            if run == total:
                found = k
                break
        check(found == n, "the number of terms differs from searching the running total")
        last = a * r ** (n - 1)
        cands = [
            (n + 1, "r3i-terms-off-by-one"),
            (n - 1, "r3i-terms-off-by-one"),
            (n + 2, "r3i-terms-off-by-one"),
            (Fraction(total, a), "r3i-terms-sum-over-first-term"),
            (Fraction(last, a) if last % a == 0 else None, "r3i-terms-power-taken-as-count"),
        ]
        seq = _terms(a, r, True)
        stem = f"How many terms of the geometric series {seq.replace(', ...', ' + ...').replace(', ', ' + ')} must be added to give a sum of {int(total)}?"
        working = [
            f"a = {a}, r = {r}. Set {a}({r}^n - 1)/({r} - 1) = {int(total)}.",
            f"{r}^n = {int(total * (r - 1) / a + 1)}, so n = {n}.",
        ]
    ans = n
    fields = num_fields(ans)
    return dict(text=stem, working=working, distractors=pick(fields, num_cands(cands, ans)), **fields)


def _medium_difference(rng):
    p, q = rng.choice([2, 3, 4, 5, 6]), rng.choice([x for x in range(-8, 10) if x not in (0, 1, -1)])
    r = rng.choice([x for x in range(-6, 8) if x != 0])
    k = rng.randint(8, 20)

    def S(n):
        return p * n * n + q * n + r * 0  # no constant term: S(0) = 0

    check(S(0) == 0, "S(0) must be 0")
    ans = S(k) - S(k - 1)
    check(ans == 2 * p * k - p + q, "term differs from the closed form")
    # second route: the arithmetic series with this S(n) has a = S(1) and d = 2p
    check(S(1) + (k - 1) * 2 * p == ans, "term differs from a + (k - 1) d")
    expr = f"{p}n² {'+' if q > 0 else '-'} {abs(q)}n"
    cands = [
        (S(k), "r3i-term-gave-partial-sum"),
        (S(k) - S(k - 2), "r3i-term-wrong-previous-sum"),
        (S(k + 1) - S(k), "r3i-term-wrong-previous-sum"),
        (S(k) - S(1), "r3i-term-wrong-previous-sum"),
        (S(1) + (k - 1) * p, "r3i-term-common-difference-halved"),
    ]
    fields = num_fields(ans)
    return dict(
        text=f"The sum of the first n terms of a sequence is S(n) = {expr}. Find the {_ordinal(k)} term of the sequence.",
        working=[
            f"The k-th term is S(k) - S(k - 1). S({k}) = {S(k)} and S({k - 1}) = {S(k - 1)}.",
            f"T{k} = {S(k)} - {S(k - 1)} = {ans}.",
        ],
        distractors=pick(fields, num_cands(cands, ans)),
        **fields,
    )


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _medium(rng):
    return _medium_count(rng) if rng.random() < 0.6 else _medium_difference(rng)


# -- hard ---------------------------------------------------------------------------------------------------


def _hard_two_sums(rng):
    for _ in attempts(500, "an arithmetic series with distinct partial sums"):
        a, d = rng.randint(-6, 14), rng.choice([x for x in range(-5, 8) if x not in (0,)])
        m, n = sorted(rng.sample(range(4, 16), 2))
        k = rng.randint(16, 30)
        if m + 1 < n and a + (k - 1) * d != 0:
            break
    Sm, Sn = s_ap(a, d, m), s_ap(a, d, n)
    # solve the two equations m(2a + (m - 1)d)/2 = Sm and n(2a + (n - 1)d)/2 = Sn for d, then a
    d_found = (2 * Sn / n - 2 * Sm / m) / (n - m)
    a_found = (2 * Sm / m - (m - 1) * d_found) / 2
    check((a_found, d_found) == (a, d), "the two equations do not give back a and d")
    ans = a + (k - 1) * d
    check(_run_ap(a, d, k) - _run_ap(a, d, k - 1) == ans, "term differs from the running totals")
    cands = [
        (d, "r3i-gave-common-difference-not-term"),
        (a, "r3i-gave-first-term-not-term"),
        (a + k * d, "r3i-term-index-off-by-one"),
        (a + (k - 2) * d, "r3i-term-index-off-by-one"),
        (Fraction(Sm, m) + (k - 1) * d, "r3i-mean-taken-as-first-term"),
    ]
    fields = num_fields(ans)
    return dict(
        text=(
            f"The sum of the first {m} terms of an arithmetic series is {int(Sm)} and the sum of the first {n} terms is "
            f"{int(Sn)}. Find the {_ordinal(k)} term."
        ),
        working=[
            f"{m}/2 × (2a + {m - 1}d) = {int(Sm)} and {n}/2 × (2a + {n - 1}d) = {int(Sn)}.",
            f"Solving: a = {a}, d = {d}. T{k} = {a} + {k - 1} × {d} = {ans}.",
        ],
        distractors=pick(fields, num_cands(cands, ans)),
        **fields,
    )


def _gp_data(rng):
    """A geometric series with negative ratio -p/q, first term a, from which S_inf and T2 are stated."""
    p, q = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (5, 6)])
    m = rng.randint(1, 3)
    a = (p + q) * q * m
    r = Fraction(-p, q)
    return a, r


def _hard_infinite(rng):
    a, r = _gp_data(rng)
    s_inf = Fraction(a) / (1 - r)
    t2 = a * r
    check(s_inf.denominator == 1 and t2.denominator == 1, "stated values must be whole numbers")
    # a^2 - S a + S T2 = 0 has the roots a and S - a; keep the one with |T2 / a| < 1
    disc = s_inf * s_inf - 4 * s_inf * t2
    root = Fraction(sqrt_exact(disc))
    roots = [(s_inf + root) / 2, (s_inf - root) / 2]
    valid = [x for x in roots if x != 0 and abs(t2 / x) < 1]
    check(valid == [a], "exactly one root of the quadratic must give a convergent series")
    other = [x for x in roots if x != a][0]
    cands = [
        (other, "r3i-gp-nonconvergent-root-kept"),
        (s_inf - a, "r3i-gp-nonconvergent-root-kept"),
        (-t2, "r3i-gp-first-term-from-second-term-sign"),
        (Fraction(s_inf) * (1 + r), "r3i-gp-sum-to-infinity-formula-sign"),
        (s_inf, "r3i-gave-sum-not-first-term"),
    ]
    fields = num_fields(a)
    return dict(
        text=(
            f"A geometric series converges to a sum of {int(s_inf)}. Its second term is {int(t2)}. Find its first term."
        ),
        working=[
            f"a/(1 - r) = {int(s_inf)} and ar = {int(t2)}, so r = {int(t2)}/a and a²{signed_text(-int(s_inf))}a{signed_text(int(s_inf * t2))} = 0.",
            f"The roots are {int(roots[0])} and {int(roots[1])}; the series converges only when |r| < 1, which leaves a = {a}.",
        ],
        distractors=pick(fields, num_cands(cands, a)),
        **fields,
    )


def sqrt_exact(value) -> int:
    value = Fraction(value)
    check(value.denominator == 1 and value >= 0, "discriminant must be a non-negative whole number")
    root = int(round(float(value) ** 0.5))
    check(root * root == value, "discriminant must be a perfect square")
    return root


def _hard(rng):
    return _hard_two_sums(rng) if rng.random() < 0.5 else _hard_infinite(rng)


@generator(
    "r3i_series_sums",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_series_sums(rng, tier="easy"):
    """Sums of arithmetic and geometric series; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------


@retry_short
def _exam_savings(rng):
    for _ in attempts(500, "a savings plan with whole-number steps"):
        a, d = rng.choice([18, 27, 36, 45]), rng.choice([2, 4, 6, 8])
        k, K = rng.randint(14, 20), rng.randint(24, 32)
        n = rng.randint(34, 44)
        if k < K < n:
            break
    S10 = s_ap(a, d, 10)
    d_found = (2 * S10 - 20 * a) / 90
    check(d_found == d, "d differs from the stated total")
    term_k = a + (k - 1) * d
    total_K = s_ap(a, d, K)
    target = s_ap(a, d, n)
    run, term, found = Fraction(0), Fraction(a), 0
    for j in range(1, 200):
        run += term
        term += d
        if run == target:
            found = j
            break
    check(found == n, "the week differs from the running total")
    stem = phrase(
        rng,
        [
            "Priya starts a savings plan by putting aside ${a} in the first week and increases the amount by the "
            "same number of dollars every week. After 10 weeks she has put aside ${S10} in total.",
            "Priya is on a savings plan for a school trip. She sets aside ${a} in the first week and raises the "
            "amount by the same number of dollars each week, so that after 10 weeks the total she has put aside is "
            "${S10}.",
        ],
        a=a,
        S10=int(S10),
    )
    ft_term = f"{a}+{k - 1}*part_a"
    ft_total = f"{K}*(2*{a}+{K - 1}*part_a)/2"
    f_a, f_b, f_c, f_d = num_fields(d), num_fields(term_k), num_fields(total_K), num_fields(n)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the amount by which she increases her saving each week.",
                marks=2,
                sensible_range=(0, None),
                working=[f"10/2 × (2 × {a} + 9d) = {int(S10)}, so 9d = {int(2 * S10 / 10 - 2 * a)} and d = {d}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (Fraction(S10, 10) - a, "r3i-ap-difference-mean-minus-first"),
                            (9 * d, "r3i-ap-difference-not-divided-by-steps"),
                            ((Fraction(2 * S10, 10) - a) / 9, "r3i-ap-sum-first-term-not-doubled"),
                        ],
                        d,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt=f"Hence find the amount she puts aside in week {k}.",
                marks=1,
                follow_through=ft_term,
                sensible_range=(0, None),
                working=[f"{a} + {k - 1} × {d} = {int(term_k)}."],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (a + k * d, "r3i-term-index-off-by-one"),
                            (a + (k - 2) * d, "r3i-term-index-off-by-one"),
                            (a + (k - 1) * a, "r3i-term-difference-confused-with-first"),
                        ],
                        term_k,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt=f"Find the total amount she has put aside after {K} weeks.",
                marks=2,
                follow_through=ft_total,
                sensible_range=(0, None),
                working=[f"{K}/2 × (2 × {a} + {K - 1} × {d}) = {int(total_K)}."],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (K * (2 * a + K * d) / 2, "r3i-ap-sum-n-instead-of-n-minus-1"),
                            (K * (a + (K - 1) * d), "r3i-ap-sum-forgot-halving"),
                            (a + (K - 1) * d, "r3i-sum-gave-last-term"),
                        ],
                        total_K,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt=f"After how many weeks has she put aside exactly ${int(target)} in total?",
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"n/2 × (2 × {a} + (n - 1) × {d}) = {int(target)} gives {d}n²{signed_text(2 * a - d)}n - {2 * int(target)} = 0.",
                    f"The positive solution is n = {n}.",
                ],
                distractors=pick(
                    f_d,
                    num_cands(
                        [
                            (n + 1, "r3i-terms-off-by-one"),
                            (n - 1, "r3i-terms-off-by-one"),
                            (Fraction(target, a), "r3i-terms-sum-over-first-term"),
                            (Fraction(target - a, d) + 1, "r3i-terms-treated-sum-as-last-term"),
                        ],
                        n,
                    ),
                ),
                **f_d,
            ),
        ],
    )


@retry_short
def _exam_infinite(rng):
    for _ in attempts(500, "a ratio with whole-number terms"):
        p, q = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)])
        a = q**3 * rng.randint(1, 3)
        r = Fraction(p, q)
        s_inf = a / (1 - r)
        if s_inf.denominator == 1 and s_inf != a:
            break
    t4 = a * r**3
    s3 = s_gp(a, r, 3)
    tail = s_inf - s3
    check(t4.denominator == 1 and s3.denominator == 1, "terms must be whole numbers")
    check(r == 1 - Fraction(a) / s_inf, "r differs from 1 - a/S")
    # second route for the tail: the terms from the 4th onward form a geometric series with first term T4
    check(t4 / (1 - r) == tail, "the tail differs from the geometric sum starting at the 4th term")
    check(_run_gp(a, r, 3) == s3, "the sum of three terms differs from adding them")
    f_a, f_b, f_c, f_d = num_fields(r), num_fields(t4), num_fields(s3), num_fields(tail)
    return dict(
        text=phrase(
            rng,
            [
                "A geometric series has first term {a} and a sum to infinity of {s}.",
                "The first term of a geometric series is {a}, and the series converges to a sum of {s}.",
            ],
            a=a,
            s=int(s_inf),
        ),
        parts=[
            dict(
                prompt="Find the common ratio.",
                marks=2,
                sensible_range=(0, 1),
                working=[f"{a}/(1 - r) = {int(s_inf)}, so 1 - r = {rat(Fraction(a) / s_inf)} and r = {rat(r)}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (Fraction(a) / s_inf, "r3i-gp-ratio-gave-one-minus-r"),
                            (Fraction(a) / (s_inf + a), "r3i-gp-ratio-sum-formula-sign"),
                            (s_inf / a, "r3i-gp-ratio-inverted"),
                        ],
                        r,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Hence find the fourth term.",
                marks=1,
                follow_through=f"{a}*part_a**3",
                sensible_range=(0, None),
                working=[f"{a} × ({rat(r)})³ = {int(t4)}."],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (a * r**4, "r3i-term-index-off-by-one"),
                            (a * r**2, "r3i-term-index-off-by-one"),
                            (a * r * 3, "r3i-term-multiplied-not-powered"),
                        ],
                        t4,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="Find the sum of the first three terms.",
                marks=2,
                follow_through=f"{a}*(1-part_a**3)/(1-part_a)",
                sensible_range=(0, None),
                working=[f"{a}(1 - ({rat(r)})³)/(1 - {rat(r)}) = {int(s3)}."],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (s_gp(a, r, 4), "r3i-gp-sum-power-off-by-one"),
                            (s_gp(a, r, 2), "r3i-gp-sum-power-off-by-one"),
                            (a * (1 - r**3), "r3i-gp-sum-forgot-divide-by-r-minus-1"),
                        ],
                        s3,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Hence find the sum of all the terms from the fourth term onwards.",
                marks=2,
                follow_through=f"{int(s_inf)}-part_c",
                sensible_range=(0, None),
                working=[f"{int(s_inf)} - {int(s3)} = {int(tail)}."],
                distractors=pick(
                    f_d,
                    num_cands(
                        [
                            (s_inf - t4, "r3i-tail-subtracted-one-term"),
                            (s_inf - s_gp(a, r, 4), "r3i-tail-wrong-number-removed"),
                            (s3, "r3i-gave-sum-not-tail"),
                        ],
                        tail,
                    ),
                ),
                **f_d,
            ),
        ],
    )


def rat(value) -> str:
    from ..core import rational_text

    return rational_text(Fraction(value))


@generator("r3i_exam_series_sums", level=13, presets={"exam": dict(tier="exam")})
def r3i_exam_series_sums(rng, tier="exam"):
    """Exam chain: an arithmetic savings plan or a geometric series with a sum to infinity, four parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _exam_savings(rng) if rng.random() < 0.5 else _exam_infinite(rng)
