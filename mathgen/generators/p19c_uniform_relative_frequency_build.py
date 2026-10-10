"""Generates `p19c_uniform_discrete_outcome` and `p19c_relative_frequency_estimate` questions on uniform discrete
outcomes and relative-frequency estimates.

Two unrelated, low-risk computation skills, built together in one module. They share no overlap risk
with each other.

- ``p19c_uniform_discrete_outcome``
  (``senior-recognise-uniform-discrete-random-variables-and-use-them``,
  ACMMM138) -- given a stated random phenomenon with n equally likely
  discrete outcomes, compute the probability of a specified outcome or
  event under the uniform discrete model. easy/medium ask for exactly ONE
  specified outcome (``1/n``); hard asks for a
  complement (``(n-1)/n``, either from a distractor-fact read or from an
  explicit list of degree values the learner must check are all equal) or a
  conditional over the remaining outcomes once one is ruled out
  (``1/(n-1)``) -- never a bare ``1/n``, the same shape as the easy tier;
  exam asks for the probability of landing on
  ANY ONE of a stated SET of m outcomes (``m/n`` in lowest terms,
  ``2 <= m < n``) -- a genuine reasoning step up, not just a harder read of
  the same single-outcome question. ``n`` is
  always DERIVED from a described context (a count of items, an arithmetic
  product of sub-divisions, or a choice between a uniform and a non-uniform
  phenomenon), never handed directly as a bare number to plug in. Never
  emits a probability-distribution TABLE with unequal fractions (that is the
  shipping ``senior-examine-simple-examples-of-non-uniform-discrete-random``
  row's own stimulus shape) and never asks "are these outcomes equally
  likely?" as its graded question (that is the shipping F-10
  ``f10-random-equal-likelihood-outcomes`` row's own graded question, an mcq
  classification that never computes a probability value).

- ``p19c_relative_frequency_estimate``
  (``senior-identify-relative-frequency-as-probability``, ACMEM152) -- given
  a stated count of times an outcome occurred in a stated number of trials,
  compute its relative frequency as an estimate of its probability. Always
  exactly ONE stated count out of exactly ONE stated total: never a
  frequency TABLE with several categories (that is the shipping
  ``senior-use-relative-frequencies-obtained-from-data-as-point`` /
  ``...-as-point-acmmm060`` / ``...-to-obtain`` rows' own stimulus shape,
  each strictly more complex). Graded as a decimal (``answer: number``) to a
  stated precision that the prompt itself states; the ``exam`` tier
  escalates the FORMAT convention (a percentage, one more decimal place)
  rather than the table complexity, which never changes.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    VerificationError,
    check_equivalent,
    generator,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_tolerance,
)

# --------------------------------------------------------------------------
# p19c_uniform_discrete_outcome (ACMMM138)
# --------------------------------------------------------------------------

# Numbered contexts: n can be any width, outcomes are "labelled 1 to n".
_NUMBERED_CONTEXTS = [
    (
        "A spinner is divided into {n} equal sectors, each labelled with a "
        "different whole number from 1 to {n}. The spinner is spun once.",
        "sector labelled {k}",
    ),
    (
        "A box contains {n} raffle tickets, each printed with a different "
        "whole number from 1 to {n}. One ticket is drawn at random.",
        "ticket printed with the number {k}",
    ),
    (
        "{n} identical-looking chairs around a table are each numbered "
        "differently from 1 to {n}. One chair is chosen at random.",
        "chair numbered {k}",
    ),
]

# Named contexts: n is capped to the pool length so every label is distinct.
_PRIZE_POOL = [
    "a toy car",
    "a gift voucher",
    "a chocolate bar",
    "a keyring",
    "a puzzle",
    "a sticker sheet",
    "a notebook",
    "a water bottle",
    "a badge",
    "a bookmark",
    "a pencil case",
    "a yo-yo",
]
_COLOUR_POOL = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
    "silver",
    "gold",
    "brown",
]


_DEGREES_MAX_ATTEMPTS = 200

# hard_read's complement_degrees sub-variant: divisors of 360 in [4, 12], so
# `360 // n` gives an EXACT whole-number degree value per sector -- a
# genuinely possible spinner, not just n equal-looking numbers that happen
# not to sum to a full turn.
_DEGREE_DIVISORS_OF_360 = [4, 5, 6, 8, 9, 10, 12]


def _n_from_range(rng, lo, hi):
    return rng.randint(lo, hi)


def _stratified_wrong_fractions(rng, above, below):
    """The hard tier's `complement_numbered`/`complement_degrees`/`conditional` sub-variants each built
    `wrong_candidates` in a FIXED order with a FIXED "N above the key, M below" split -- `pick_distractors`
    never shuffles, so it picked the SAME 3 survivors every draw, putting the key at a constant rank (rank 3
    of 4 on 200/200 seeds).

    A plain shuffle-then-pick-first-3 of one pooled list is not enough: with
    a fixed above/below split, the number of below-key survivors chosen is
    hypergeometric and stays concentrated (a symmetric 3-above/3-below pool
    still puts ~45% of draws on a single rank; only impractically large
    pools push that under 40%). Splitting into two STRATA and choosing how
    many come from each independently of their internal order breaks that:
    with `above`/`below` each >= 3 survivors, the number of below-key
    options offered is uniform over {0, 1, 2, 3}, so the key's rank is
    uniform over the reachable ranks by construction, not merely likely to
    vary.

    `above`/`below` are each lists of `Fraction`s guaranteed to sit
    strictly on their side of the key by construction; this function drops
    the constant `1/2` tell, drops anything outside (0, 1) or duplicated
    within its own stratum, draws how many of the final 3 come from
    `above` (clamped to what each stratum can supply), shuffles each
    stratum with `rng`, and returns the concatenation -- a plain list of
    `Fraction`s, not yet deduped against the key itself (the caller still
    runs the shared `!= prob` / `check_distinct` checks)."""

    def _clean(pool):
        seen: list[Fraction] = []
        out: list[Fraction] = []
        for f in pool:
            if f == Fraction(1, 2):
                continue
            if not (Fraction(0) < f < Fraction(1)):
                continue
            if f in seen:
                continue
            seen.append(f)
            out.append(f)
        return out

    above = _clean(above)
    below = _clean(below)
    lo = max(0, 3 - len(below))
    hi = min(3, len(above))
    check(lo <= hi, "not enough above/below survivors to offer 3 hard-tier distractors")
    a = rng.randint(lo, hi)
    rng.shuffle(above)
    rng.shuffle(below)
    return above[:a] + below[: 3 - a]


def _stratified_by_key(rng, key, candidates):
    """Split `candidates` by value into strictly above-key and strictly below-key strata (dropping any equal to
    the key) and hand them to `_stratified_wrong_fractions`, so the number of below-key options among the 3
    distractors is drawn independently of the candidates' order and the key's rank is spread over 1..4."""
    above = [f for f in candidates if f > key]
    below = [f for f in candidates if f < key]
    return _stratified_wrong_fractions(rng, above, below)


def _numbered_question(rng, n, note=""):
    intro, target_phrase = rng.choice(_NUMBERED_CONTEXTS)
    k = rng.randint(1, n)
    text = intro.format(n=n) + note + f" What is the probability that it is the {target_phrase.format(k=k)}?"
    return text


def _named_question(rng, n, pool_name, pool):
    items = rng.sample(pool, n)
    if pool_name == "prize":
        listing = ", ".join(items)
        target = rng.choice(items)
        text = (
            f"A game show wheel is divided into {n} equal sectors, each showing a different "
            f"one of {n} prizes: {listing}. The wheel is spun once. What is the probability "
            f"that it lands on the sector showing {target}?"
        )
    else:  # colour
        listing = ", ".join(items)
        target = rng.choice(items)
        text = (
            f"A bag holds one each of {n} differently coloured marbles: {listing}. One marble "
            f"is drawn at random. What is the probability that the marble drawn is {target}?"
        )
    return text


@generator(
    "p19c_uniform_discrete_outcome",
    level=12,
    presets={
        "easy": dict(variant="stated_count"),
        "medium": dict(variant="derived_count"),
        "hard": dict(variant="hard_read"),
        "exam": dict(variant="uniform_vs_nonuniform"),
    },
)
def p19c_uniform_discrete_outcome(rng, variant="stated_count"):
    """A stated random phenomenon has n equally likely discrete outcomes;
    compute a probability under the uniform discrete model -- a bare 1/n
    for easy/medium, a complement (n-1)/n or a conditional 1/(n-1) for hard, an m/n set-probability
    for exam. n is always derived from a described context, never handed as
    a bare number to plug in directly. Never emits a distribution table of
    unequal fractions and never grades an equally-likely-or-not
    classification -- every outcome here is equally likely by
    construction."""
    if variant == "stated_count":
        if rng.random() < 0.5:
            n = _n_from_range(rng, 5, 20)
            text = _numbered_question(rng, n)
        else:
            pool_name, pool = rng.choice([("prize", _PRIZE_POOL), ("colour", _COLOUR_POOL)])
            n = _n_from_range(rng, 4, len(pool))
            text = _named_question(rng, n, pool_name, pool)
        working = [f"There are {n} equally likely outcomes, so each has probability 1/{n}."]

    elif variant == "derived_count":
        # The prompt gives ONLY the a x b structure and never states the product n anywhere, so the
        # learner must multiply (the contract: "n is never handed directly"). Each sub-variant has
        # three phrasings over a wide (a, b) grid so the rendered text stays varied over 200 seeds.
        sub_variant = rng.choice(["wheel_split", "grid_seats", "tournament_teams", "shelf_books"])
        if sub_variant == "wheel_split":
            a = rng.randint(2, 8)
            b = rng.randint(2, 9)
            template = rng.choice(
                [
                    "A wheel is divided into {a} equal main sections, and each main section is "
                    "further divided into {b} equal smaller regions, each showing a different colour. "
                    "Every region is the same size. The wheel is spun once. What is the "
                    "probability that it lands on a particular one of the smaller regions?",
                    "A prize wheel has {a} equal main sections, each cut into {b} identical smaller "
                    "regions, and every region carries its own different prize. The wheel is "
                    "spun once. What is the probability that it stops on one particular region?",
                    "A spinner is split into {a} equal main sections. Each main section is then split "
                    "into {b} equal smaller regions, and every region has a different label. "
                    "The spinner is spun once. What is the probability of landing on a particular "
                    "labelled region?",
                ]
            )
            what = "regions"
        elif sub_variant == "grid_seats":
            a = rng.randint(2, 12)
            b = rng.randint(3, 12)
            template = rng.choice(
                [
                    "A hall has {a} rows of {b} identical seats, with one differently numbered raffle "
                    "stub taped under each seat, so every seat has its own stub. One stub is "
                    "chosen at random. What is the probability of choosing a particular stub?",
                    "A theatre has {a} rows with {b} seats in each row, and every seat has a "
                    "different ticket number stuck under it. One ticket number is drawn at random. "
                    "What is the probability that a particular seat's number is drawn?",
                    "A classroom is set out with {a} rows of {b} desks, and each desk holds "
                    "one differently named card. One card is picked at random. What is the "
                    "probability of picking a particular card?",
                ]
            )
            what = "stubs"
        elif sub_variant == "tournament_teams":
            a = rng.randint(2, 10)
            b = rng.randint(2, 10)
            template = rng.choice(
                [
                    "A tournament has {a} groups, and each group is split into {b} equally sized "
                    "teams, so every team is entered into a lucky draw. What is the "
                    "probability that a particular team is drawn?",
                    "A school carnival has {a} houses, each sending {b} equally skilled teams, and "
                    "all of the teams go into a random draw for first game. What is the "
                    "probability that a particular team is drawn first?",
                    "A league is made up of {a} divisions with {b} teams in each, and one team is "
                    "picked at random to host the final. Every team is equally likely to be "
                    "picked. What is the probability that a particular team is picked?",
                ]
            )
            what = "teams"
        else:
            a = rng.randint(2, 9)
            b = rng.randint(3, 12)
            template = rng.choice(
                [
                    "A library display has {a} shelves with {b} different books on each shelf, and "
                    "every book is equally likely to be chosen. One book is chosen at "
                    "random. What is the probability that a particular book is chosen?",
                    "A bookcase has {a} shelves, and each shelf holds {b} different novels, all "
                    "equally likely to be taken out. One novel is taken at random. What is "
                    "the probability that a particular novel is taken?",
                    "{a} boxes each hold {b} differently numbered tokens, and every token "
                    "is equally likely to be drawn. One token is drawn at random. What is the "
                    "probability of drawing a particular token?",
                ]
            )
            what = "books"
        n = a * b
        text = template.format(a=a, b=b)
        working = [f"n = {a} x {b} = {n} equally likely {what}, so probability = 1/{n}."]

    elif variant == "hard_read":
        # The three OLD sub-variants (distractor_fact/indirect_outcome/distractor_fact_named) all
        # graded a bare 1/n -- the exact same shape the easy tier grades, just with more prose
        # around it (200/200 seeds measured p == 1). Every sub-variant below now grades a genuinely
        # different fraction shape (a complement or a conditional), never a bare 1/n, while keeping
        # the shared "n equally likely outcomes, derived from context" spine and the read-complexity
        # flavour (a distractor fact, or an explicit degree list the learner must check rather than
        # being told "equal sectors" outright).
        sub_variant = rng.choice(["complement_numbered", "complement_degrees", "conditional"])
        if sub_variant == "complement_numbered":
            if rng.random() < 0.5:
                n = _n_from_range(rng, 5, 20)
                k = rng.randint(1, n)
                text = (
                    f"A spinner is divided into {n} equal-sized sectors, each labelled with a "
                    f"different whole number from 1 to {n}. Some sectors have larger numbers "
                    f"printed on them than others, but every sector is still exactly the same "
                    f"size. The spinner is spun once. What is the probability that it does NOT "
                    f"land on the sector labelled {k}?"
                )
                target_desc = f"the sector labelled {k}"
            else:
                n = _n_from_range(rng, 4, min(len(_PRIZE_POOL), 12))
                items = rng.sample(_PRIZE_POOL, n)
                target = rng.choice(items)
                listing = ", ".join(items)
                text = (
                    f"A game show wheel is divided into {n} equal-sized sectors, each showing a "
                    f"different one of {n} prizes: {listing}. Some of the sector colours look "
                    f"brighter than others, but every sector still takes up exactly the same "
                    f"share of the wheel. The wheel is spun once. What is the probability that "
                    f"it does NOT land on the sector showing {target}?"
                )
                target_desc = f"the sector showing {target}"
            working = [
                f"Every sector is still the same size despite the distracting detail, so all "
                f"{n} outcomes are equally likely: probability of {target_desc} is 1/{n}.",
                f"NOT landing on it is the complement: 1 - 1/{n} = {n - 1}/{n}.",
            ]
            prob = Fraction(n - 1, n)
            check_equivalent(
                Fraction(1) - Fraction(1, n),
                prob,
                "the complement of 1/n must equal (n-1)/n",
            )
            # See `_stratified_wrong_fractions` -- three ABOVE-key candidates (n/(n+1), (n+1)/(n+2),
            # (n+2)/(n+3), each provably > (n-1)/n by the same cross-multiplication argument, e.g.
            # for (n+1)/(n+2): (n+1)*n vs (n-1)*(n+2) = n^2 + n vs n^2 + n - 2) and four BELOW-key
            # candidates (1/n, (n-2)/n, 1/(n+1), (n-3)/n) stratified so the number of below-key
            # options in the final 3 is not fixed draw to draw.
            check(n >= 4, "complement_numbered needs n >= 4 for the above/below inequalities")
            above_pool = [Fraction(n, n + 1), Fraction(n + 1, n + 2), Fraction(n + 2, n + 3)]
            below_pool = [Fraction(1, n), Fraction(n - 2, n), Fraction(1, n + 1), Fraction(n - 3, n)]
            for f in above_pool:
                check(f > prob, "every above-key candidate must exceed the complement (n-1)/n")
            for f in below_pool:
                check(f < prob, "every below-key candidate must fall below the complement (n-1)/n")
            wrong_candidates = _stratified_wrong_fractions(rng, above_pool, below_pool)
        elif sub_variant == "complement_degrees":
            # n drawn only from divisors of 360 in [4, 12], so
            # `degree_value = 360 // n` is exact and every listed degree
            # value genuinely sums to a real, physically possible spinner
            # (360 degrees in total) -- not just "n equal-looking numbers".
            n = rng.choice(_DEGREE_DIVISORS_OF_360)
            degree_value = 360 // n
            check(n * degree_value == 360, "listed degree values must sum to exactly 360")
            k = rng.randint(1, n)
            degree_list = ", ".join(f"{degree_value} degrees" for _ in range(n))
            text = (
                f"A spinner has {n} sectors, each labelled with a different whole number from "
                f"1 to {n}. Listed in order around the spinner, their sizes are: {degree_list}. "
                f"The spinner is spun once. What is the probability that it does NOT land on "
                f"the sector labelled {k}?"
            )
            working = [
                f"Every one of the {n} listed sizes is exactly {degree_value} degrees, so all "
                f"{n} sectors are the same size and equally likely: probability of the sector "
                f"labelled {k} is 1/{n}.",
                f"NOT landing on it is the complement: 1 - 1/{n} = {n - 1}/{n}.",
            ]
            prob = Fraction(n - 1, n)
            check_equivalent(
                Fraction(1) - Fraction(1, n),
                prob,
                "the complement of 1/n must equal (n-1)/n",
            )
            # Same reasoning and pools as `complement_numbered` above. `n` here is always a divisor
            # of 360 in [4, 12], so n - 3 >= 1 always stays a valid probability below the key.
            above_pool = [Fraction(n, n + 1), Fraction(n + 1, n + 2), Fraction(n + 2, n + 3)]
            below_pool = [Fraction(1, n), Fraction(n - 2, n), Fraction(1, n + 1), Fraction(n - 3, n)]
            for f in above_pool:
                check(f > prob, "every above-key candidate must exceed the complement (n-1)/n")
            for f in below_pool:
                check(f < prob, "every below-key candidate must fall below the complement (n-1)/n")
            wrong_candidates = _stratified_wrong_fractions(rng, above_pool, below_pool)
        else:  # conditional
            n = _n_from_range(rng, 5, 20)
            check(n >= 3, "conditional hard-tier variant needs at least 3 outcomes")
            k = rng.randint(1, n)
            remaining_labels = [x for x in range(1, n + 1) if x != k]
            j = rng.choice(remaining_labels)
            text = (
                f"A spinner is divided into {n} equal sectors, each labelled with a different "
                f"whole number from 1 to {n}. The spinner is spun once, and it is known that it "
                f"did NOT land on the sector labelled {k}. Given this, what is the probability "
                f"that it landed on the sector labelled {j}?"
            )
            working = [
                f"Ruling out the sector labelled {k} leaves {n - 1} equally likely remaining "
                f"sectors out of the original {n}.",
                f"The sector labelled {j} is one of those {n - 1} remaining, equally likely "
                f"outcomes: probability = 1/{n - 1}.",
            ]
            prob = Fraction(1, n - 1)
            # Independent route: count the remaining outcomes directly (the
            # length of the filtered label list) rather than trusting
            # construction's own `n - 1` arithmetic.
            check_equivalent(
                Fraction(1, len(remaining_labels)),
                prob,
                "conditional probability must equal 1 over the remaining outcome count",
            )
            # The original three candidates were always "2 below the key, 1 above" (1/n and 1/(n+1)
            # below 1/(n-1); 2/(n-1) above), so `pick_distractors` offered the same fixed three
            # every draw, putting the key at a constant rank. Stratify into three ABOVE-key "counted
            # one/two/ three extra matching outcomes instead of none" candidates (2/(n-1), 3/(n-1),
            # 4/(n-1)) and three BELOW-key "off-by-one on the denominator" candidates (1/n, 1/(n+1),
            # 1/(n+2)) via `_stratified_wrong_fractions`, so the number of below-key options in the
            # final 3 is not fixed draw to draw. Some above-key candidates fall outside (0, 1) or
            # land exactly on the constant-1/2 tell for a handful of small n (e.g. 4/(n-1) == 1 when
            # n == 5) -- `_stratified_wrong_fractions` drops those rather than trusting every
            # candidate to survive.
            above_pool = [Fraction(2, n - 1), Fraction(3, n - 1), Fraction(4, n - 1)]
            below_pool = [Fraction(1, n), Fraction(1, n + 1), Fraction(1, n + 2)]
            for f in above_pool:
                check(f > prob, "every above-key candidate must exceed the conditional key 1/(n-1)")
            for f in below_pool:
                check(f < prob, "every below-key candidate must fall below the conditional key 1/(n-1)")
            wrong_candidates = _stratified_wrong_fractions(rng, above_pool, below_pool)
        hard_wrong_fractions = [f for f in wrong_candidates if f != prob and Fraction(0) < f < Fraction(1)]
        check_distinct(prob, *hard_wrong_fractions)

    elif variant == "uniform_vs_nonuniform":
        n = rng.randint(5, 18)
        n2 = rng.randint(5, 18)
        while n2 == n:
            n2 = rng.randint(5, 18)
        # Real step-up over the other tiers: the graded quantity is the probability of landing on
        # ANY ONE of a stated set of m outcomes (m >= 2, m < n), not a bare single-outcome 1/n --
        # while the underlying uniform-discrete recognition (derive n from the "equal sectors"
        # description) stays the shared spine.
        #
        # `exam_wrong_fractions` below spans both sides of m/n instead (some strictly greater, some
        # strictly less), filtered of the constant-1/2 tell and of any accidental collision with the
        # answer; redraw m (bounded loop) on the rare chance fewer than 2 survive or one collides
        # with the answer.
        for _m_attempt in range(50):
            m = rng.randint(2, n - 2)
            prob_candidate = Fraction(m, n)
            # A wider pool, split by value into strictly-above and strictly-below strata so the
            # key's rank is spread over 1..4 (before: rank 2 on 76.5% of exam draws).
            exam_pool = [
                Fraction(m - 1, n),  # missed one member of the stated set
                Fraction(m + 1, n),  # counted one member too many
                Fraction(m, n - 1),  # used n - 1 (off-by-one on the total)
                Fraction(m, n + 1),  # used n + 1
                Fraction(m, n2),  # used the OTHER spinner's own count
                Fraction(1, n),  # forgot to count the set at all
                Fraction(m + 1, n + 1),  # added one to both the set and the total
                Fraction(m + 2, n + 2),  # added two to both
                Fraction(m, n + 2),  # off-by-two on the total
            ]
            try:
                exam_wrong_fractions = _stratified_by_key(rng, prob_candidate, exam_pool)
            except VerificationError:
                continue
            exam_wrong_fractions = [
                f
                for f in exam_wrong_fractions
                if f != Fraction(1, 2) and f != prob_candidate and Fraction(0) < f < Fraction(1)
            ]
            if len(exam_wrong_fractions) == 3:
                break
        else:
            raise RuntimeError(f"could not draw enough exam-tier distractors for n={n} after 50 attempts")
        labels = sorted(rng.sample(range(1, n + 1), m))
        degrees = _uneven_degrees(rng, n2)
        check(sum(degrees) == 360, "exam-tier spinner degrees must sum to exactly 360")
        uniform_body = f"is divided into {n} equal sectors, each labelled with a different whole number from 1 to {n}"
        nonuniform_body = (
            f"is divided into {n2} sectors of different sizes "
            f"({', '.join(f'{d} degrees' for d in degrees)}), the sizes adding to 360 degrees "
            f"in total, each also labelled with a different whole number from 1 to {n2}"
        )
        bodies = [uniform_body, nonuniform_body]
        rng.shuffle(bodies)
        # The referent comes first: "one spinner" is introduced before "the other".
        parts = [f"one spinner {bodies[0]}", f"the other spinner {bodies[1]}"]
        # No literal `{`/`}` in the rendered prompt: `latex_prompt` wraps this whole
        # plain-text prompt in `\text{...}` and escapes any literal brace it finds to
        # `\{`/`\}`, which breaks the one-level `\text{[^{}]*}` nesting assumption relied
        # on by the mangled-prompt-latex detector.
        # A plain comma list reads just as naturally and avoids the nesting hazard.
        if len(labels) == 2:
            label_set_text = f"{labels[0]} or {labels[1]}"
        else:
            label_set_text = ", ".join(str(x) for x in labels[:-1]) + f", or {labels[-1]}"
        text = (
            f"Two spinners are being compared: {parts[0]}; and {parts[1]}. Only one of the two "
            f"gives every one of its labelled sectors an equal chance of being landed on. For "
            f"that spinner, what is the probability that it lands on a sector labelled "
            f"{label_set_text}?"
        )
        working = [
            f"The spinner with {n} EQUAL sectors is the uniform one; the other has unequal "
            f"sector sizes despite sharing the same style of numbered labels.",
            f"{m} of the {n} equally likely labels are in the stated set, so probability = {m}/{n}.",
        ]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if variant == "uniform_vs_nonuniform":
        m_effective = m
        prob = Fraction(m_effective, n)
        wrong_fractions = exam_wrong_fractions
        # Independent route: re-derive m/n from "n equally likely outcomes
        # each get an equal share of probability 1, and m of them are in the
        # stated event", i.e. n * prob == m, rather than trusting
        # construction's own Fraction(m, n) literal.
        check_equivalent(
            prob * n,
            m_effective,
            "m equally likely outcomes out of n must together get m/n of the total probability",
        )
    elif variant == "hard_read":
        # `prob`/`hard_wrong_fractions` were already computed, and already
        # independently re-derived via their own `check_equivalent` call, in
        # the sub-variant-specific branch above -- each sub-variant grades a
        # different fraction shape, so there is no single generic "n * prob"
        # identity to re-check here.
        wrong_fractions = hard_wrong_fractions
    else:
        m_effective = 1
        prob = Fraction(m_effective, n)
        # The old fixed trio {1/(n-1), 1/(n+1), one of 2/n} put the key at the median on 200/200
        # easy/medium draws. A wider pool of genuine misconceptions, stratified into above-key and
        # below-key, spreads the key over every rank.
        easy_pool = [
            Fraction(1, n - 1),  # undercounted the outcomes by one
            Fraction(2, n),  # "there must be two outcomes" doubling
            Fraction(3, n),  # counted three favourable outcomes
            Fraction(n - 1, n),  # gave the complement instead
            Fraction(1, n - 2),  # undercounted the outcomes by two
            Fraction(2, n - 1),  # two favourable out of one fewer
            Fraction(1, n + 1),  # overcounted the outcomes by one
            Fraction(1, n + 2),  # overcounted the outcomes by two
            Fraction(1, 2 * n),  # halved the share
        ]
        # `_stratified_by_key` drops the constant 1/2 tell (2/n at n == 4).
        wrong_fractions = _stratified_by_key(rng, prob, easy_pool)
        # Independent route: re-derive 1/n from "n equally likely outcomes
        # each get an equal share of probability 1", i.e. n * prob == 1,
        # rather than trusting construction's own Fraction(1, n) literal.
        check_equivalent(
            prob * n,
            m_effective,
            "1 equally likely outcome out of n must get 1/n of the total probability",
        )

    answer_text = rational_text(prob)
    candidates = [rational_text(f) for f in wrong_fractions]
    check_distinct(prob, *wrong_fractions)
    distractors = pick_distractors(answer_text, candidates, n=3)

    return dict(
        text=text,
        latex=text,
        answer={"p": prob.numerator, "q": prob.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _uneven_degrees(rng, count):
    """A composition of exactly 360 into `count` positive, visually unequal
    parts, for a genuinely non-uniform spinner.

    Gives every part a floor of 5 degrees FIRST (so the >= 5 floor is met by
    construction, never by rejection), then distributes the remaining budget
    ``360 - 5 * count`` across the `count` parts as a random composition
    (`count - 1` random cut points in the remaining budget, sorted, taking
    consecutive differences) -- the same "cut points -> differences" idea as
    a plain composition of 360, just applied to the leftover budget instead
    of the full 360 so the floor is unconditional. Retries a fresh draw only
    for the (much rarer) case that two adjacent parts land on the same size,
    so the returned parts ALWAYS sum to exactly 360 -- the prompt text
    states this."""
    # Sizes are now ALL distinct (the prompt says "sectors of different sizes"; two 23s on 119/200
    # draws contradicted it). Base sizes 5, 6..., 5 + count - 1 are distinct with floor 5; the
    # leftover budget is spread as a random composition whose parts are SORTED ascending before
    # being added to the ascending base, which keeps every size strictly increasing (so distinct)
    # and the sum exactly 360. The order around the spinner is then shuffled.
    base = [5 + i for i in range(count)]
    remaining = 360 - sum(base)
    if remaining < 0:
        raise RuntimeError(f"cannot fit {count} distinct sizes of at least 5 into 360")
    cuts = sorted(rng.randint(0, remaining) for _ in range(count - 1))
    extras = []
    prev = 0
    for c in cuts:
        extras.append(c - prev)
        prev = c
    extras.append(remaining - prev)
    extras.sort()
    parts = [b + e for b, e in zip(base, extras)]
    rng.shuffle(parts)
    return parts


# --------------------------------------------------------------------------
# p19c_relative_frequency_estimate (ACMEM152)
# --------------------------------------------------------------------------

_RF_CONTEXTS = [
    dict(
        subject="A basketball player",
        trial_noun="free throws",
        outcome_adj="successful",
        next_phrase="her next free throw",
        extra="who has been playing for the Eastside Hawks",
    ),
    dict(
        subject="A factory quality inspector",
        trial_noun="items inspected",
        outcome_adj="defective",
        next_phrase="the next item inspected",
        extra="who has worked at the plant for six years",
    ),
    dict(
        subject="A weather station",
        trial_noun="days recorded",
        outcome_adj="rainy",
        next_phrase="the next day recorded",
        extra="based in a coastal town",
    ),
    dict(
        subject="A call centre",
        trial_noun="calls answered",
        outcome_adj="resolved on the first attempt",
        next_phrase="the next call answered",
        extra="open every day of the week",
    ),
]


def _rf_verify(count: int, trials: int, prob: Fraction) -> None:
    """Independent route: recompute the estimate as 1 - (the complement
    count / trials), rather than construction's own direct count/trials
    division."""
    check(0 <= count <= trials, "count must not exceed the stated number of trials")
    complement = trials - count
    recomputed = 1 - Fraction(complement, trials)
    check(recomputed == prob, "complement-based recomputation disagrees with construction")


def _avoid_boundary(count: int, trials: int, dp: int) -> bool:
    """True if rounding count/trials to `dp` decimal places (or, for a
    percentage, 100*count/trials to `dp`) lands exactly on a rounding
    boundary (a tie), which the stated convention must never present."""
    scale = 10 ** (dp + 1)
    scaled = Fraction(count, trials) * scale
    return scaled.denominator == 1 and scaled % 10 == 5


def _avoid_boundary_pct(count: int, trials: int, dp: int) -> bool:
    scale = 10 ** (dp + 1)
    scaled = Fraction(count, trials) * 100 * scale
    return scaled.denominator == 1 and scaled % 10 == 5


_RF_MAX_ATTEMPTS = 50


@generator(
    "p19c_relative_frequency_estimate",
    level=12,
    presets={
        "easy": dict(variant="bare"),
        "medium": dict(variant="scenario"),
        "hard": dict(variant="derive"),
        "exam": dict(variant="percentage"),
    },
)
def p19c_relative_frequency_estimate(rng, variant="bare"):
    """Given a stated count of times an outcome occurred in a stated number
    of trials, compute its relative frequency as an estimate of its
    probability. Always exactly ONE stated count out of exactly ONE stated
    total number of trials -- never a frequency TABLE with several
    categories to distinguish (that is the shipping
    senior-use-relative-frequencies-... rows' own stimulus shape)."""
    survivors: list[str] = []
    for _attempt in range(_RF_MAX_ATTEMPTS):
        ctx = rng.choice(_RF_CONTEXTS)
        dp = 2

        if variant == "bare":
            trials = rng.randint(20, 80)
            count = rng.randint(1, trials - 1)
            while _avoid_boundary(count, trials, dp):
                count = rng.randint(1, trials - 1)
            bare_template = rng.choice(
                [
                    "Out of {trials} attempts, {count} were successful. Estimate the probability that "
                    "the next attempt is successful, as a decimal rounded to {dp} decimal places.",
                    "{count} out of {trials} attempts were successful. Estimate the probability that "
                    "the next attempt is successful, as a decimal rounded to {dp} decimal places.",
                    "In {trials} attempts, {count} succeeded. Based on this, estimate the probability "
                    "of success on the next attempt, as a decimal rounded to {dp} decimal places.",
                ]
            )
            text = bare_template.format(trials=trials, count=count, dp=dp)

        elif variant == "scenario":
            trials = rng.randint(20, 80)
            count = rng.randint(1, trials - 1)
            while _avoid_boundary(count, trials, dp):
                count = rng.randint(1, trials - 1)
            text = (
                f"{ctx['subject']}, {ctx['extra']}, recorded {trials} {ctx['trial_noun']}; {count} of "
                f"them were {ctx['outcome_adj']}. Estimate the probability that {ctx['next_phrase']} "
                f"is {ctx['outcome_adj']}, as a decimal rounded to {dp} decimal places."
            )

        elif variant == "derive":
            sub_variant = rng.choice(["success_miss", "percent_count"])
            if sub_variant == "success_miss":
                success = rng.randint(5, 40)
                miss = rng.randint(5, 40)
                trials = success + miss
                count = success
                while _avoid_boundary(count, trials, dp):
                    miss = rng.randint(5, 40)
                    trials = success + miss
                text = (
                    f"{ctx['subject']} recorded {ctx['trial_noun']} over a season: {success} were "
                    f"{ctx['outcome_adj']} and {miss} were not. Estimate the probability that "
                    f"{ctx['next_phrase']} is {ctx['outcome_adj']}, as a decimal rounded to {dp} "
                    f"decimal places."
                )
            else:
                # trials is a multiple of 100 so "pct% of them" converts to a
                # whole-number count EXACTLY -- never rounded -- so the stated
                # percentage and the graded probability always agree exactly.
                hundreds = rng.randint(1, 3)
                trials = 100 * hundreds
                pct = rng.randint(10, 90)
                count = pct * hundreds
                while _avoid_boundary(count, trials, dp):
                    pct = rng.randint(10, 90)
                    count = pct * hundreds
                check(count * 100 == pct * trials, "percentage-worded count must convert exactly, never rounded")
                text = (
                    f"{ctx['subject']} recorded {trials} {ctx['trial_noun']} in total; {pct}% of "
                    f"them were {ctx['outcome_adj']}. Estimate the probability that {ctx['next_phrase']} "
                    f"is {ctx['outcome_adj']}, as a decimal rounded to {dp} decimal places."
                )

        elif variant == "percentage":
            dp = 1
            trials = rng.randint(20, 200)
            count = rng.randint(1, trials - 1)
            while _avoid_boundary_pct(count, trials, dp):
                count = rng.randint(1, trials - 1)
            text = (
                f"{ctx['subject']} recorded {trials} {ctx['trial_noun']}; {count} of them were "
                f"{ctx['outcome_adj']}. Estimate the probability that {ctx['next_phrase']} is "
                f"{ctx['outcome_adj']}, expressed as a PERCENTAGE rounded to {dp} decimal place."
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")

        prob = Fraction(count, trials)
        _rf_verify(count, trials, prob)

        if variant == "percentage":
            value = round_half_up(prob * 100, dp)
        else:
            value = round_half_up(prob, dp)
        answer_text = f"{float(value):.{dp}f}"
        answer_value = float(value)

        # Misconceptions: forgets that the trial itself counts (off-by-one on
        # the numerator); swaps success and failure (reports the complement
        # relative frequency); uses the wrong denominator (favourable over
        # the complement rather than over the total, filtered below to stay
        # a genuine (0, range_max] relative frequency); treats the outcome
        # as equally likely regardless of the data. The former `wrong_raw`
        # (the bare count itself, formatted to dp places) is deliberately
        # NOT offered any more: for any `trials` in this range it is
        # essentially always > 1, an immediate "a relative frequency can
        # never exceed 1" tell.
        pct_scale = 100 if variant == "percentage" else 1
        wrong_off_by_one = float(round_half_up(Fraction(count - 1, trials) * pct_scale, dp))
        wrong_complement = float(round_half_up((1 - prob) * pct_scale, dp))
        wrong_denominator = (
            float(round_half_up(Fraction(count, trials - count) * pct_scale, dp)) if trials != count else None
        )
        wrong_half = float(round_half_up(Fraction(1, 2) * pct_scale, dp))

        raw_candidates = [wrong_off_by_one, wrong_complement]
        if wrong_denominator is not None:
            raw_candidates.append(wrong_denominator)
        raw_candidates.append(wrong_half)

        # Every candidate must stay a genuine (0, range_max] relative
        # frequency/percentage AND land at least `margin` away from the true
        # rounded answer in the same dp-place representation -- comfortably
        # clear of both the grader's own tolerance and simple rounding noise
        # -- or it is dropped from the pool rather than offered.
        range_max = 100.0 if variant == "percentage" else 1.0
        margin = float(10 * stated_precision_tolerance(dp))
        survivors = []
        for v in raw_candidates:
            if not (0 < v <= range_max):
                continue
            if abs(v - answer_value) < margin:
                continue
            formatted = f"{v:.{dp}f}"
            if formatted not in survivors:
                survivors.append(formatted)

        if len(survivors) >= 3:
            break
    # Retry loop keeps `text`/`count`/`trials`/`dp`/`answer_text`/
    # `answer_value`/`survivors` from its LAST attempt even on the rare draw
    # that never reaches 3 survivors within `_RF_MAX_ATTEMPTS`;
    # `pick_distractors` below is happy to return fewer than requested.

    distractors = pick_distractors(answer_text, survivors, n=3, tolerance=stated_precision_tolerance(dp))

    return dict(
        text=text,
        latex=text,
        answer=answer_value,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"probability = {count}/{trials} = {answer_text}" + ("%" if variant == "percentage" else "")],
        distractors=distractors,
    )
