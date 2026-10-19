"""Probability scale and complement/union rules.

Two generators:
- Skill C (probability_scale_classify): Review the probability scale 0 <= P(A) <= 1.
  Given a finite equally-likely sample space and a rule (easy: explicit list,
  medium: simple rule, hard: compound rule), classify whether P(X) is impossible,
  certain, or between.

- Skill D (probability_complement_union): Review complement and union rules.
  Easy: complement rule P(not A) = 1 - P(A).
  Medium: mutually-exclusive union P(A union B) = P(A) + P(B).
  Hard: inclusion-exclusion P(A union B) = P(A) + P(B) - P(A intersect B).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text

# ============================================================================
# Skill C: Probability Scale Classification
# ============================================================================

OPTION_IMPOSSIBLE = "P(X) = 0 (X is impossible)"
OPTION_CERTAIN = "P(X) = 1 (X is certain)"
OPTION_BETWEEN = "0 < P(X) < 1 (X is possible but not certain)"
OPTIONS_SCALE = (OPTION_IMPOSSIBLE, OPTION_CERTAIN, OPTION_BETWEEN)


def classify_scale(n, matches):
    """Independent check: classify the probability scale outcome based on the
    number of matching outcomes. Never trusts the construction path; recomputes
    from the match count directly."""
    if len(matches) == 0:
        return OPTION_IMPOSSIBLE
    if len(matches) == n:
        return OPTION_CERTAIN
    return OPTION_BETWEEN


@generator(
    "probability_scale_classify",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def probability_scale_classify(rng, tier="medium"):
    """Classify a probability outcome (impossible, certain, or between) given
    a finite equally-likely sample space and a predicate rule."""

    n = rng.randint(4, 8)

    if tier == "easy":
        # Easy: explicit stated list, not a rule
        choice = rng.choice(["empty", "full", "partial"])

        if choice == "empty":
            # Event: "outcome is greater than n" -- vacuously impossible
            rule_text = f"the outcome is greater than {n}"
            matches = []
        elif choice == "full":
            # Event: "outcome is at most n" -- certain
            rule_text = f"the outcome is at most {n}"
            matches = list(range(1, n + 1))
        else:  # partial
            # Event: explicit listed set of 2 to n-1 members
            num_members = rng.randint(2, n - 1)
            matches = sorted(rng.sample(range(1, n + 1), num_members))
            matches_str = ", ".join(map(str, matches))
            rule_text = f"the outcome is in the set {{{matches_str}}}"

    elif tier == "medium":
        # Medium: genuine rule applied to S = {1, ..., n}
        rule_choice = rng.choice(["even", "odd", "multiple", "greater"])

        if rule_choice == "even":
            rule_text = "the outcome is even"
            matches = [v for v in range(1, n + 1) if v % 2 == 0]
        elif rule_choice == "odd":
            rule_text = "the outcome is odd"
            matches = [v for v in range(1, n + 1) if v % 2 == 1]
        elif rule_choice == "multiple":
            k = rng.randint(2, max(2, n // 2))
            rule_text = f"the outcome is a multiple of {k}"
            matches = [v for v in range(1, n + 1) if v % k == 0]
        else:  # greater
            k = rng.randint(0, n)
            rule_text = f"the outcome is greater than {k}"
            matches = [v for v in range(1, n + 1) if v > k]

    else:  # hard
        # Hard: compound rule (AND or OR of two simple rules). Resample the
        # rule pair (bounded) while the combined result is degenerate --
        # i.e. equal to either component's own match set alone. This kills
        # every case where the "compound" adds no genuine reasoning: two
        # identical rules (even+even), semantically-identical rules under
        # different names ("even" and "a multiple of 2" are the same set --
        # always true when n < 4 forces k=2 for "multiple"), and two
        # same-direction "greater than" conditions (always reducible to the
        # tighter bound alone). A combined set that differs from BOTH
        # components necessarily required evaluating both conditions.
        max_pair_attempts = 30
        for _pair_attempt in range(max_pair_attempts):
            rule_base = rng.choice(["even", "odd", "multiple", "greater"])
            rule_second = rng.choice(["even", "odd", "multiple", "greater"])
            combine = rng.choice(["AND", "OR"])

            if rule_base == "even":
                matches_base = [v for v in range(1, n + 1) if v % 2 == 0]
                text_base = "even"
            elif rule_base == "odd":
                matches_base = [v for v in range(1, n + 1) if v % 2 == 1]
                text_base = "odd"
            elif rule_base == "multiple":
                k1 = rng.randint(2, max(2, n // 2))
                matches_base = [v for v in range(1, n + 1) if v % k1 == 0]
                text_base = f"a multiple of {k1}"
            else:  # greater
                k1 = rng.randint(0, n)
                matches_base = [v for v in range(1, n + 1) if v > k1]
                text_base = f"greater than {k1}"

            if rule_second == "even":
                matches_second = [v for v in range(1, n + 1) if v % 2 == 0]
                text_second = "even"
            elif rule_second == "odd":
                matches_second = [v for v in range(1, n + 1) if v % 2 == 1]
                text_second = "odd"
            elif rule_second == "multiple":
                k2 = rng.randint(2, max(2, n // 2))
                matches_second = [v for v in range(1, n + 1) if v % k2 == 0]
                text_second = f"a multiple of {k2}"
            else:  # greater
                k2 = rng.randint(0, n)
                matches_second = [v for v in range(1, n + 1) if v > k2]
                text_second = f"greater than {k2}"

            if combine == "AND":
                matches = sorted(set(matches_base) & set(matches_second))
                rule_text = f"the outcome is {text_base} AND {text_second}"
            else:  # OR
                matches = sorted(set(matches_base) | set(matches_second))
                rule_text = f"the outcome is {text_base} OR {text_second}"

            if set(matches) != set(matches_base) and set(matches) != set(matches_second):
                break
        else:
            raise ValueError(
                f"could not find a non-degenerate compound rule pair for n={n} after {max_pair_attempts} attempts"
            )
        check(
            set(matches) != set(matches_base) and set(matches) != set(matches_second),
            "hard-tier compound rule must genuinely differ from both of its components",
        )

    # Independent check: verify classification
    label = classify_scale(n, matches)
    check(label in OPTIONS_SCALE, f"classification {label} not in {OPTIONS_SCALE}")

    # Compute the matching list for display
    non_matches = [v for v in range(1, n + 1) if v not in matches]
    check(len(matches) + len(non_matches) == n, "complement count mismatch")

    # Build question and answer
    s_str = "{1, ..., " + str(n) + "}"
    x_list_str = "{" + ", ".join(map(str, matches)) + "}" if matches else "{}"

    question_text = (
        f"Consider a sample space S = {s_str}. "
        f"Event X is defined by the rule: {rule_text}. "
        f"The outcomes matching this rule are: X = {x_list_str}. "
        f"What is true about P(X)?"
    )

    answer_text = label
    distractors = pick_distractors(label, [o for o in OPTIONS_SCALE if o != label], n=2)

    if not matches:
        working = [
            f"No outcome in S matches the rule, so X has 0 of the {n} outcomes.",
            f"P(X) = 0/{n} = 0, so X is impossible.",
        ]
    elif len(matches) == n:
        working = [
            f"Every outcome in S matches the rule, so X has all {n} of the {n} outcomes.",
            f"P(X) = {n}/{n} = 1, so X is certain.",
        ]
    else:
        working = [
            f"X has {len(matches)} of the {n} outcomes, so P(X) = {len(matches)}/{n}.",
            "This is greater than 0 and less than 1, so X is possible but not certain.",
        ]

    return dict(
        text=question_text,
        working=working,
        latex=r"\text{P(X)}",
        answer=label,
        answer_text=label,
        distractors=distractors,
    )


# ============================================================================
# Skill D: Complement and Union Rules
# ============================================================================

_DENOMS = (8, 10, 12)


def check_complement_rule(a, d):
    """Independent check for easy tier: verify P(A) + P(not A) = 1."""
    pa = Fraction(a, d)
    pna = Fraction(d - a, d)
    check(pa + pna == 1, "complement probabilities do not sum to 1")


def check_mutually_exclusive_union(a, b, d):
    """Independent check for medium tier: verify union via set union."""
    # Build explicit universe: a slots for A, b slots for B (disjoint)
    a_slots = set(range(a))
    b_slots = set(range(a, a + b))
    union_slots = a_slots | b_slots
    union_size = len(union_slots)
    expected = Fraction(a + b, d)
    actual = Fraction(union_size, d)
    check(actual == expected, f"union check failed: {actual} != {expected}")


def check_inclusion_exclusion_union(a, b, c, d):
    """Independent check for hard tier: verify union via set union with overlap."""
    # Build explicit universe: a slots for A, b slots for B with c overlapping
    a_slots = set(range(a))
    b_slots = set(range(a - c, a - c + b))  # last c of A overlap with first c of B
    union_slots = a_slots | b_slots
    union_size = len(union_slots)
    expected = Fraction(a + b - c, d)
    actual = Fraction(union_size, d)
    check(actual == expected, f"union check failed: {actual} != {expected}")


def _proper(options):
    """Keep only options that are possible probabilities (0 to 1)."""
    return [o for o in options if 0 <= Fraction(o) <= 1]


@generator(
    "probability_complement_union",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def probability_complement_union(rng, tier="medium"):
    """Calculate a probability using complement or union rules."""

    d = rng.choice(_DENOMS)

    if tier == "easy":
        # Easy: complement rule P(not A) = 1 - P(A)
        # Exclude 2*a == d (P(A) == 1/2, so P(not A) == P(A))
        a = rng.randint(1, d - 1)
        while 2 * a == d:
            a = rng.randint(1, d - 1)

        answer = Fraction(d - a, d)
        check_complement_rule(a, d)

        # Question and distractors
        question_text = f"Given P(A) = {a}/{d}, what is P(not A)?"
        answer_text = rational_text(answer)

        # Misconceptions:
        # 1. forgot-to-complement: reports P(A) itself
        distractor_forgot = rational_text(Fraction(a, d))
        # Proof: a/d == (d-a)/d iff 2a == d, which is excluded by the guard above
        check(Fraction(a, d) != answer, "forgot-to-complement distractor equals answer")

        # 2. off-by-one-numerator: reports (d-a+1)/d
        distractor_off_by_one = rational_text(Fraction(d - a + 1, d))
        # Proof: (d-a+1)/d - (d-a)/d = 1/d != 0 for any valid d
        check(Fraction(d - a + 1, d) != answer, "off-by-one distractor equals answer")
        check(abs(Fraction(d - a + 1, d) - answer) == Fraction(1, d), "off-by-one gap proof")

        # 3. off-by-one the other way: reports (d-a-1)/d, only when it is a real (non-zero) probability
        cands = [distractor_forgot, distractor_off_by_one]
        if d - a - 1 >= 1:
            cands.append(rational_text(Fraction(d - a - 1, d)))
        distractors = pick_distractors(answer_text, _proper(cands), n=2)

    elif tier == "medium":
        # Medium: mutually-exclusive union P(A union B) = P(A) + P(B)
        a = rng.randint(1, d - 2)
        b = rng.randint(1, d - 1 - a)
        check(b >= 1, "b must be at least 1")

        answer = Fraction(a + b, d)
        check_mutually_exclusive_union(a, b, d)

        question_text = (
            f"Events A and B are mutually exclusive. P(A) = {a}/{d} and P(B) = {b}/{d}. What is P(A union B)?"
        )
        answer_text = rational_text(answer)

        # Misconceptions:
        # 1. forgets-second-event: reports P(A) alone
        distractor_first = rational_text(Fraction(a, d))
        # Proof: a/d < (a+b)/d since b >= 1
        check(Fraction(a, d) < answer, "forgets-second distractor should be less")

        # 2. double-counts-first-event: reports 2*a + b
        distractor_double = rational_text(Fraction(2 * a + b, d))
        # Proof: (2a+b)/d > (a+b)/d since a >= 1
        check(Fraction(2 * a + b, d) > answer, "double-counts distractor should be more")

        # 3. multiplies instead of adding (the independent-events slip): reports P(A) x P(B)
        distractor_product = rational_text(Fraction(a * b, d * d))
        check(Fraction(a * b, d * d) != answer, "product distractor equals answer")
        distractors = pick_distractors(
            answer_text, _proper([distractor_first, distractor_double, distractor_product]), n=2
        )

    else:  # hard
        # Hard: inclusion-exclusion P(A union B) = P(A) + P(B) - P(A intersect B)
        # a, b in [2, d-2]; c in [max(1, a+b-(d-1)), min(a,b)-1] -- the upper
        # bound is min(a,b)-1, NOT min(a,b): c == min(a,b) means one event is
        # a subset of the other (A subset B when c==a, or B subset A when
        # c==b), and then the answer P(A union B) collapses to exactly the
        # larger marginal probability (or, when a==b==c, to all three given
        # values at once) -- a learner who wrongly reports "the larger
        # probability" or "P(A) == P(B) == P(A intersect B)" is graded
        # CORRECT on that whole seed class even though neither is the
        # inclusion-exclusion rule this tier is meant to test. Requiring
        # strict c < min(a,b) forces a + b - c to exceed both marginals and
        # P(A intersect B), so no such false accept is reachable.
        a = rng.randint(2, d - 2)
        b = rng.randint(2, d - 2)
        c_max = min(a, b) - 1
        c_min = max(1, a + b - (d - 1))

        # Resample a, b if c_min > c_max (up to 50 attempts)
        attempts = 0
        while c_min > c_max and attempts < 50:
            a = rng.randint(2, d - 2)
            b = rng.randint(2, d - 2)
            c_max = min(a, b)
            c_min = max(1, a + b - (d - 1))
            attempts += 1

        if c_min > c_max:
            raise ValueError(f"could not find valid a, b for hard tier after {attempts} attempts")

        c = rng.randint(c_min, c_max)
        check(c >= 1, "c must be at least 1 (genuine overlap)")
        check(a + b - c < d, "final answer must be < 1")

        answer = Fraction(a + b - c, d)
        check_inclusion_exclusion_union(a, b, c, d)

        question_text = (
            f"Events A and B overlap. "
            f"P(A) = {a}/{d}, P(B) = {b}/{d}, and P(A intersect B) = {c}/{d}. "
            f"What is P(A union B)?"
        )
        answer_text = rational_text(answer)

        # Misconceptions:
        # 1. forgot-overlap: reports P(A) + P(B) = (a+b)/d
        distractor_forgot = rational_text(Fraction(a + b, d))
        # Proof: (a+b)/d > (a+b-c)/d since c >= 1
        check(Fraction(a + b, d) > answer, "forgot-overlap distractor should be more")

        # 2. double-subtracts-overlap: reports (a+b-2*c)/d
        distractor_double_sub = rational_text(Fraction(a + b - 2 * c, d))
        # Proof: |(a+b-2c)/d - (a+b-c)/d| = |(-c)/d| = c/d != 0 since c >= 1
        check(Fraction(a + b - 2 * c, d) != answer, "double-subtracts distractor equals answer")
        check(abs(Fraction(a + b - 2 * c, d) - answer) == Fraction(c, d), "double-subtracts gap proof")

        # 3. reports the larger of the two marginals (strictly below the union because the overlap is a proper part)
        distractor_larger = rational_text(Fraction(max(a, b), d))
        check(Fraction(max(a, b), d) != answer, "larger-marginal distractor equals answer")
        distractors = pick_distractors(
            answer_text,
            _proper(
                [
                    distractor_forgot,
                    distractor_double_sub,
                    distractor_larger,
                    rational_text(Fraction(a + b - c - 1, d)),  # off by one
                    rational_text(Fraction(min(a, b), d)),  # reports the smaller marginal
                ]
            ),
            n=2,
        )

    if tier == "easy":
        raw = f"{d - a}/{d}"
        step = f"P(not A) = 1 - P(A) = 1 - {a}/{d} = {raw}"
    elif tier == "medium":
        raw = f"{a + b}/{d}"
        step = f"For mutually exclusive events, P(A union B) = P(A) + P(B) = {a}/{d} + {b}/{d} = {raw}"
    else:
        raw = f"{a + b - c}/{d}"
        step = f"P(A union B) = P(A) + P(B) - P(A intersect B) = {a}/{d} + {b}/{d} - {c}/{d} = {raw}"
    working = [step + ("." if raw == answer_text else f", which simplifies to {answer_text}.")]

    return dict(
        text=question_text,
        latex=r"\text{P}",
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        distractors=distractors,
        working=working,
    )
