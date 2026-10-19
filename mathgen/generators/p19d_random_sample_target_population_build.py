"""Random-sample validity and target population.

- ``p19d_random_sample_valid``
  (``senior-understand-the-concept-of-a-random-sample``, ACMMM171, domain
  senior-mathematics, answer: mcq)
  -- given a described method for selecting individuals from a stated
  population, judge whether that method constitutes a valid random sample
  (every member of the stated population has an equal, independent chance of
  selection). Boundary: this is NOT
  ``senior-simulate-repeated-random-sampling-from-a-variety-of``
  (ACMMM176/ACMSM139), which SIMULATES from an ALREADY-valid random sample
  and never judges whether a selection method itself is valid; and it is NOT
  ``senior-identify-the-target-population-to-be-surveyed`` (ACMEM132), which judges WHAT the population description
  should be, never whether a selection METHOD from an already-identified
  population is valid.

- ``p19d_target_population``
  (``senior-identify-the-target-population-to-be-surveyed``, ACMEM132,
  domain statistics-probability, answer: mcq)
  -- given a stated survey scenario and its research question, choose the
  correct description of the target population (neither narrower nor
  broader than what the research question actually asks about) from the
  offered options. Boundary: this is NOT
  ``senior-understand-the-concept-of-a-random-sample`` (ACMMM171), which judges whether a SELECTION METHOD from an
  already-identified population is valid, never what the population
  description should be.

Both skills are genuine scenario/classification mcq skills (never a computable
numeric answer), so each ships its own independent oracle in
``tests/test_p19d_random_sample_target_population_oracle.py`` that re-derives
the correct option from the rendered prompt text alone, using a generic
technique (keyword/definition matching for skill 1, bag-of-words Jaccard
similarity for skill 2) that never reads this module's own tables.
"""

from __future__ import annotations

import re

from ..core import check, generator, latex_answer

# ============================================================================
# Skill 1: p19d_random_sample_valid
# ============================================================================

DEFINITION_SENTENCE = (
    "A selection method produces a valid random sample only when every "
    "member of the stated population has an equal, independent chance of "
    "being selected."
)

# Each flaw label still contains the literal class-noun phrase the build test's absent-flaw check
# greps for ("convenience sample", "self-selection", "systematic selection", "excluded from ever
# being selected") -- those are legitimate, stable answer vocabulary, not the surface cue being
# removed; the cue being removed is scenario PROSE that echoes those same words (see the flaw
# templates below, none of which use "convenience", "volunt*", "systematic" or "exclu*").


def _label_valid(pop):
    # Uses only pop['unit'] (a generic noun -- "student") rather than the
    # full pop['population'] phrase, which embeds a proper noun ("Riverside
    # High School") that the scenario prose always states too: if the valid
    # option were the only one carrying that proper noun, its presence in
    # the stem would itself become a positional tell whenever the valid
    # option is offered -- exactly the class of surface cue being removed.
    return (
        f"Valid -- each {pop['unit']} would have matching odds of ending "
        f"up in the group, unaffected by anyone else's result"
    )


def _label_convenience(pop):
    return (
        f"Not valid -- convenience sample ({pop['units']} ended up in the "
        f"sample simply because they were nearby or on hand, not through "
        f"any randomising step)"
    )


def _label_self_select(pop):
    return f"Not valid -- self-selection bias ({pop['units']} each decided for themselves whether to take part)"


def _label_systematic(pop):
    return (
        f"Not valid -- systematic selection with no genuine random start "
        f"({pop['units']} were spaced at regular gaps, following whatever "
        f"arrangement they were already sitting in)"
    )


def _label_exclusion(pop):
    return (
        f"Not valid -- some of the {pop['units']} were excluded from ever "
        f"being selected, however the rest of the group came about"
    )


LABEL_FNS = {
    "convenience": _label_convenience,
    "self_select": _label_self_select,
    "systematic": _label_systematic,
    "exclusion": _label_exclusion,
}
FLAW_KEYS = list(LABEL_FNS)

# "age group" always fits any population and needs no tag.
POPULATIONS = [
    dict(
        population="the 240 students enrolled at Riverside High School",
        unit="student",
        units="students",
        size=240,
        tags=frozenset({"roll"}),
    ),
    dict(
        population="the 1,500 residents of the town of Mapleton",
        unit="resident",
        units="residents",
        size=1500,
        tags=frozenset({"geographic"}),
    ),
    dict(
        population="the 85 employees at the Northgate distribution centre",
        unit="employee",
        units="employees",
        size=85,
        tags=frozenset({"workplace"}),
    ),
    dict(
        population="the 620 members of the Fairview Community Gym",
        unit="member",
        units="members",
        size=620,
        tags=frozenset({"membership"}),
    ),
    dict(
        population="the 3,200 registered voters in the Kestrel electorate",
        unit="voter",
        units="voters",
        size=3200,
        tags=frozenset({"geographic"}),
    ),
    dict(
        population="the 410 patients on Dr Alvarez's clinic roster",
        unit="patient",
        units="patients",
        size=410,
        tags=frozenset({"roll"}),
    ),
]


# ---- valid methods ---------------------------------------------------------


def _valid_hat(pop, n, rng):
    return (
        f"Every {pop['unit']}'s name among {pop['population']} is written on an identical slip; "
        f"all the slips are placed in a barrel, mixed thoroughly, and {n} slips are drawn out "
        f"without looking."
    )


def _valid_rng_smallest(pop, n, rng):
    return (
        f"A computer's random number generator assigns every one of {pop['population']} a "
        f"unique random number, and the {n} {pop['units']} with the smallest assigned numbers "
        f"are selected."
    )


def _cap_first(s: str) -> str:
    """Capitalise only the first character, leaving the rest of the string's
    own casing (proper nouns like "Fairview Community Gym") untouched --
    ``str.capitalize()`` would lowercase them."""
    return s[:1].upper() + s[1:]


def _stratum_choices(pop) -> list:
    """Gate the offered STRATUM by the drawn population's tags, per the explicit compatibility table on
    ``POPULATIONS`` -- "age group" always fits; "suburb" only fits a geographic population; "membership tier"
    only fits a population that actually has members; "department" only fits a workplace."""
    tags = pop.get("tags", frozenset())
    choices = ["age group"]
    if "geographic" in tags:
        choices.append("suburb")
    if "membership" in tags:
        choices.append("membership tier")
    if "workplace" in tags:
        choices.append("department")
    return choices


def _valid_stratified(pop, n, rng):
    stratum = rng.choice(_stratum_choices(pop))
    return (
        f"{_cap_first(pop['population'])} are first divided into strata by {stratum}, in "
        f"proportion to each stratum's share of the population, and within every stratum "
        f"members are selected by a genuine random draw (names in a hat), giving {n} "
        f"{pop['units']} in total."
    )


_VALID_SHUFFLED_K_CHOICES = [5, 6, 7, 8, 9]


def _valid_shuffled_every_kth(pop, n, rng, k=None):
    if k is None:
        k = rng.choice(_VALID_SHUFFLED_K_CHOICES)
    return (
        f"The full list of {pop['population']} is randomly shuffled first, and then, starting "
        f"from that freshly shuffled order, every {k}th name is selected until {n} "
        f"{pop['units']} have been chosen."
    )


ALL_VALID_TEMPLATES = [_valid_hat, _valid_rng_smallest, _valid_stratified, _valid_shuffled_every_kth]


# ---- convenience ------------------------------------------------------------


def _conv_first_arrive(pop, n, rng):
    return f"Of {pop['population']}, the first {n} to arrive at the venue that morning are selected."


def _conv_same_floor(pop, n, rng):
    # "work on the same floor as the researcher" only makes sense for a workplace population --
    # gated out for every other population by ``_filter_context_compatible`` below (never called
    # directly against a non-workplace ``pop``; kept parameter-compatible with the other convenience
    # templates regardless).
    return (
        f"Of {pop['population']}, the {n} who happen to work on the same floor as the "
        f"researcher are selected, since gathering them takes hardly any extra effort."
    )


def _conv_closest(pop, n, rng):
    return f"Of {pop['population']}, the {n} living closest to the researcher's own house are selected."


def _filter_context_compatible(templates: list, pop: dict) -> list:
    """Remove any template from the offered list that does not fit the drawn population, per the explicit
    compatibility table on ``POPULATIONS``. Only ``_conv_same_floor`` (workplace-only) currently needs this;
    other convenience templates (``_conv_first_arrive``, ``_conv_closest``) are population-agnostic and always
    fit."""
    if "workplace" not in pop.get("tags", frozenset()):
        templates = [t for t in templates if t is not _conv_same_floor]
    return templates


# ---- self-selection ----------------------------------------------------------


def _self_online_poll(pop, n, rng):
    return (
        f"An online poll is posted where anyone from {pop['population']} who wants to can "
        f"click a link to respond; the first {n} responses are used."
    )


def _self_email_volunteer(pop, n, rng):
    return (
        f"A survey link is emailed to every one of {pop['population']}, and whoever chooses to "
        f"click through and complete it of their own accord is included, up to {n} responses."
    )


def _self_signup_sheet(pop, n, rng):
    return (
        f"A sign-up sheet is left for {pop['population']}, and any {pop['unit']} may add their "
        f"name to it if they choose; the first {n} names written down are used."
    )


# ---- systematic (no random start) --------------------------------------------
#
# Every template below is a genuine every-Nth-from-the-list construction with NO random start, and
# explicitly states the list was never shuffled/reordered/ randomised, mirroring the wording
# discipline of the one correct template that already existed (kept, unchanged in substance).

_SYS_UNSHUFFLED_K_CHOICES = [5, 6, 7, 8, 9]
_SYS_FIXED_ORDER_K_CHOICES = [4, 6, 7, 9]
_SYS_PRINTED_K_CHOICES = [5, 8, 10]


def _sys_unshuffled_every_kth(pop, n, rng, k=None):
    if k is None:
        k = rng.choice(_SYS_UNSHUFFLED_K_CHOICES)
    return (
        f"Starting from the very first name on the original, unshuffled register of "
        f"{pop['population']}, every {k}th name is selected until {n} {pop['units']} have "
        f"been chosen."
    )


def _sys_fixed_order_every_kth(pop, n, rng, k=None):
    if k is None:
        k = rng.choice(_SYS_FIXED_ORDER_K_CHOICES)
    return (
        f"{_cap_first(pop['population'])} are listed in the fixed order they appear on the "
        f"official roll, which is never reordered or randomised in any way; starting from the "
        f"very top of that roll, every {k}th person on it is selected until {n} {pop['units']} "
        f"have been chosen."
    )


def _sys_printed_every_kth(pop, n, rng, k=None):
    if k is None:
        k = rng.choice(_SYS_PRINTED_K_CHOICES)
    return (
        f"Working down the original printed list of {pop['population']} -- a list that is "
        f"never reshuffled or put into any random order -- every {k}th name on it is selected "
        f"until {n} {pop['units']} have been chosen."
    )


# Every k-based template's own draw domain, keyed by the function itself -- shared by
# ``_draw_k_and_n_for_template`` below so the domain is defined exactly once (as each function's own
# default) and never duplicated.
K_DOMAINS = {
    _sys_unshuffled_every_kth: _SYS_UNSHUFFLED_K_CHOICES,
    _sys_fixed_order_every_kth: _SYS_FIXED_ORDER_K_CHOICES,
    _sys_printed_every_kth: _SYS_PRINTED_K_CHOICES,
    _valid_shuffled_every_kth: _VALID_SHUFFLED_K_CHOICES,
}


def _draw_k_and_n_for_template(rng, pop, template):
    """Draw a (k, n) pair for a k-based template such that k * n never exceeds the population size. Draws n first
    (as before, from the usual 15-60 band) and keeps only the k choices from the template's own domain that
    fit; if none fit after a bounded number of attempts (the smallest population, 85, cannot support every k
    in every domain at n >= 15 -- e.g. k=9 needs n <= 9), falls back to the smallest k in the domain with n
    clamped down to whatever that k allows. For every domain this generator actually uses, that fallback n
    (population_size // min(domain)) still lands at or above the usual n >= 15 floor for the smallest
    population, 85 (e.g. min(domain)=5 -> n=17; min(domain)=4 -> n=21) -- checked rather
    than assumed."""
    domain = K_DOMAINS[template]
    pop_size = pop["size"]
    for _attempt in range(30):
        n = rng.randint(15, 60)
        valid_k = [k for k in domain if k * n <= pop_size]
        if valid_k:
            return rng.choice(valid_k), n
    k = min(domain)
    n = max(1, pop_size // k)
    return k, n


def _render_method(pop, n, rng, template):
    """Render a method template's text, redrawing a population-compatible
    (k, n) pair for k-based templates (the k * n <= population constraint) instead of using the
    passed-in ``n`` unconstrained. Returns ``(text, n_used)`` so callers that
    also need the actual sample size elsewhere in the same draw (the exam
    tier's Step 3, which restates "the n selected ...") stay consistent with
    whatever n the k-based template actually used."""
    if template in K_DOMAINS:
        k, n_used = _draw_k_and_n_for_template(rng, pop, template)
        return template(pop, n_used, rng, k=k), n_used
    return template(pop, n, rng), n


# ---- exclusion ----------------------------------------------------------------


def _excl_no_email_on_file(pop, n, rng):
    # Replaces the dropped _excl_skip_already_surveyed. This template is structurally, unambiguously
    # exclusion instead: anyone without an email address on file is never sent the survey at all,
    # however interested they might be.
    return (
        f"Of {pop['population']}, the survey's online form requires a working email address, "
        f"and any {pop['unit']} without one on file is never sent it at all, even though the "
        f"stated population includes everyone; {n} of the remaining {pop['units']} are then "
        f"chosen at random."
    )


def _excl_app_only(pop, n, rng):
    return (
        f"Of {pop['population']}, the survey is only sent to those who use the organisation's "
        f"mobile app, even though the stated population includes everyone; {n} app users are "
        f"then chosen at random from among them."
    )


def _excl_phone_only(pop, n, rng):
    return (
        f"Of {pop['population']}, only those with a phone number on file are eligible to be "
        f"randomly called, though the population includes everyone regardless of a listed "
        f"number; {n} of the eligible phone numbers are then dialled at random."
    )


FLAW_TEMPLATES = {
    "convenience": [_conv_first_arrive, _conv_same_floor, _conv_closest],
    "self_select": [_self_online_poll, _self_email_volunteer, _self_signup_sheet],
    "systematic": [_sys_unshuffled_every_kth, _sys_fixed_order_every_kth, _sys_printed_every_kth],
    "exclusion": [_excl_no_email_on_file, _excl_app_only, _excl_phone_only],
}

TIER_POOLS = {
    "easy": dict(
        valid=[_valid_hat, _valid_rng_smallest],
        convenience=[_conv_first_arrive, _conv_closest],
        self_select=[_self_online_poll, _self_signup_sheet],
        systematic=[_sys_unshuffled_every_kth, _sys_fixed_order_every_kth],
        exclusion=[_excl_app_only, _excl_phone_only],
    ),
    "medium": dict(
        valid=[_valid_stratified, _valid_hat],
        convenience=[_conv_same_floor, _conv_first_arrive],
        self_select=[_self_email_volunteer, _self_online_poll],
        systematic=[_sys_fixed_order_every_kth, _sys_printed_every_kth],
        exclusion=[_excl_no_email_on_file, _excl_app_only],
    ),
    "hard": dict(
        valid=ALL_VALID_TEMPLATES,
        convenience=FLAW_TEMPLATES["convenience"],
        self_select=FLAW_TEMPLATES["self_select"],
        systematic=FLAW_TEMPLATES["systematic"],
        exclusion=FLAW_TEMPLATES["exclusion"],
    ),
}

# Precedence-ordered internal classifier: self-selection and exclusion are checked first because
# their cue phrases are the most specific; systematic is checked before valid because a negated
# "reshuffled"/"unshuffled" phrase would otherwise satisfy a naive "shuffled" substring test meant
# for the valid class.
_SELF_SELECT_RE = re.compile(
    r"click(s|ed)?\s+(a\s+link|through)|wants?\s+to\b.*\bclick|chooses?\s+to\b|"
    r"if\s+they\s+choose|choose\s+to\s+(actually\s+)?(turn\s+up|attend|respond)|"
    r"own\s+accord|decide[sd]?\s+for\s+themselves",
    re.IGNORECASE,
)
_EXCLUSION_RE = re.compile(
    r"skipped\s+and\s+replaced|only\s+(sent|distributed|lists?)\s+to|"
    r"eligible\s+to\s+be\s+randomly\s+called|even\s+though\s+the\s+stated\s+population",
    re.IGNORECASE,
)
_SYSTEMATIC_EVERY_NTH_RE = re.compile(r"every\s+\d+\w{0,2}\s+(name|person|entry)", re.IGNORECASE)
_SYSTEMATIC_NO_RANDOM_RE = re.compile(
    r"unshuffled|never\s+reordered|never\s+reshuffled|never\s+randomised|"
    r"no\s+randomising|original(\s+printed)?\s+order|fixed\s+order",
    re.IGNORECASE,
)
_VALID_RE = re.compile(
    r"randomly\s+shuffled|random\s+number\s+generator\s+assigns|\bbarrel\b|\bstrata\b",
    re.IGNORECASE,
)
_CONVENIENCE_RE = re.compile(
    r"first\s+.*\barrive|\bclosest\b|same\s+floor|hardly\s+any\s+extra\s+effort|\bnearby\b",
    re.IGNORECASE,
)


def _classify_from_text(text: str) -> str:
    """Internal (in-generator) consistency check: an independently-coded keyword classifier over the method text,
    mirroring ``check_equivalent``'s role for a computational skill -- a second route that must agree with
    which template pool construction drew from."""
    if _SELF_SELECT_RE.search(text):
        return "self_select"
    if _EXCLUSION_RE.search(text):
        return "exclusion"
    if _SYSTEMATIC_EVERY_NTH_RE.search(text) and _SYSTEMATIC_NO_RANDOM_RE.search(text):
        return "systematic"
    if _VALID_RE.search(text):
        return "valid"
    if _CONVENIENCE_RE.search(text):
        return "convenience"
    return "unknown"


def _draw_scenario(rng, pools):
    target_is_valid = rng.random() < 0.35
    pop = rng.choice(POPULATIONS)
    n = rng.randint(15, 60)
    if target_is_valid:
        truth = "valid"
        template = rng.choice(pools["valid"])
    else:
        truth = rng.choice(FLAW_KEYS)
        candidates = pools[truth]
        if truth == "convenience":
            candidates = _filter_context_compatible(candidates, pop)
        template = rng.choice(candidates)
    method_text, _n_used = _render_method(pop, n, rng, template)
    check(
        _classify_from_text(method_text) == truth,
        f"p19d_random_sample_valid: text classifies as "
        f"{_classify_from_text(method_text)!r}, expected {truth!r}: {method_text!r}",
    )
    return truth, method_text, pop


def _build_options(rng, truth, pop, exclude_flaws=None):
    # ``exclude_flaws`` lets a specific call site (the systematic-frame exam draw) drop "exclusion"
    # from the offered pool entirely, removing the ambiguity by option construction rather than
    # further wording polish.
    exclude_flaws = exclude_flaws or set()
    pool = [f for f in FLAW_KEYS if f not in exclude_flaws]
    if truth == "valid":
        chosen_flaws = rng.sample(pool, 3)
        correct = _label_valid(pop)
    else:
        others = [f for f in pool if f != truth]
        chosen_flaws = [truth] + rng.sample(others, 2)
        correct = LABEL_FNS[truth](pop)
    options = [_label_valid(pop)] + [LABEL_FNS[f](pop) for f in chosen_flaws]
    check(len(set(options)) == 4, "p19d_random_sample_valid: options are not 4 distinct strings")
    rng.shuffle(options)
    distractors = [o for o in options if o != correct]
    check(len(distractors) == 3, "p19d_random_sample_valid: expected exactly 3 distractors")
    return correct, distractors


def _random_sample_question(rng, tier: str, method_text: str, correct: str, distractors: list[str]) -> dict:
    text = f"{DEFINITION_SENTENCE} {method_text} Classify whether this selection method produces a valid random sample."
    return dict(
        text=text,
        latex=text,
        answer=correct,
        answer_text=correct,
        answer_latex=latex_answer(correct, plain=True),
        distractors=distractors,
        working=[f"True classification: {correct}."],
    )


def _easy(rng):
    truth, method_text, pop = _draw_scenario(rng, TIER_POOLS["easy"])
    correct, distractors = _build_options(rng, truth, pop)
    return _random_sample_question(rng, "easy", method_text, correct, distractors)


def _medium(rng):
    truth, method_text, pop = _draw_scenario(rng, TIER_POOLS["medium"])
    correct, distractors = _build_options(rng, truth, pop)
    return _random_sample_question(rng, "medium", method_text, correct, distractors)


def _hard(rng):
    truth, method_text, pop = _draw_scenario(rng, TIER_POOLS["hard"])
    correct, distractors = _build_options(rng, truth, pop)
    return _random_sample_question(rng, "hard", method_text, correct, distractors)


def _exam_frame_step_clean(pop) -> str:
    return (
        f"Step 1: A sampling frame listing every one of {pop['population']} is compiled by "
        f"the organisation running the survey."
    )


def _exam_frame_step_biased(pop, rng, frame_size) -> str:
    # State an explicit ``frame_size`` here (a fraction of the population, drawn by the caller),
    # so Step 2 draws its sample from THIS number instead of from the full population.
    source = rng.choice(
        [
            "the organisation's mobile app",
            "the members-only email newsletter",
            "the loyalty-card database",
        ]
    )
    return (
        f"Step 1: A sampling frame is compiled, but it only lists the {pop['units']} who are "
        f"signed up to {source}, even though the stated population is all of "
        f"{pop['population']}; {frame_size} {pop['units']} are signed up."
    )


_BIASED_FRAME_DIVISORS = [3, 4, 5]


def _draw_biased_frame_size_and_n(rng, pop):
    """Every ``POPULATIONS`` entry's smallest size (85) still yields frame_size >= 17 for every divisor in
    ``_BIASED_FRAME_DIVISORS``, comfortably above the n >= 15 floor used everywhere else in this module."""
    divisor = rng.choice(_BIASED_FRAME_DIVISORS)
    frame_size = pop["size"] // divisor
    n = rng.randint(15, min(60, frame_size))
    return frame_size, n


def _exam_step2_frame_subgroup(pop, frame_size, n, rng) -> str:
    """Two clean, genuinely valid draw mechanisms, matching this module's own internal 'valid' vocabulary (barrel
    / random number generator) so ``_classify_from_text`` still reads Step 2 alone as a valid method."""
    if rng.random() < 0.5:
        return (
            f"The names of those {frame_size} {pop['units']} are each written on an "
            f"identical slip; the slips are placed in a barrel, mixed thoroughly, and {n} are "
            f"drawn out without looking."
        )
    return (
        f"A random number generator assigns every one of those {frame_size} {pop['units']} a "
        f"unique number, and the {n} with the smallest numbers are selected."
    )


def _exam_close_step_clean(pop, n) -> str:
    return f"Step 3: Every one of the {n} selected {pop['units']} is then contacted and takes part."


def _exam_close_step_self_select(pop, n) -> str:
    return (
        f"Step 3: Of the {n} {pop['units']} contacted, only those who choose to actually turn "
        f"up to the follow-up session are counted in the results."
    )


def _exam_close_step_convenience(pop, n) -> str:
    # Worded with "hardly any extra effort" (an existing marker in both this module's
    # ``_CONVENIENCE_RE`` and the oracle's ``_CONVENIENCE_PATTERNS`` -- not a lexicon change) rather
    # than "nearby"/"on hand", which also appear verbatim in ``_label_convenience`` and were found
    # to make this stem word a 100%-reliable key tell
    # (test_rsv_no_stem_word_is_a_reliable_key_tell).
    return (
        f"Step 3: Of the {n} selected {pop['units']}, only the ones the researcher can reach "
        f"again with hardly any extra effort are actually followed up and counted in the results."
    )


def _exam_frame_step_systematic(pop, rng, k=None) -> str:
    # A second flaw class (systematic, alongside exclusion's existing Step 1 variant) can now sit in
    # the sampling frame itself -- the frame is compiled by a non-random every-Nth pass over the
    # register, before Step 2 even draws from it. ``k`` is drawn by the caller via
    # ``_draw_k_and_n_for_template`` (reusing ``_sys_unshuffled_every_kth``'s own k domain) so the
    # frame it describes (population // k names) is never smaller than the sample size Step 2 goes
    # on to draw -- the k * n <= population invariant applies here too: the frame this Step 1 describes can hold
    # at most population // k names, so k times whatever Step 2 later samples from it must still not
    # exceed the stated population.
    if k is None:
        k = rng.choice(_SYS_UNSHUFFLED_K_CHOICES)
    return (
        f"Step 1: A sampling frame is compiled by working down the original, unshuffled "
        f"register of {pop['population']} and listing only every {k}th name on it, rather "
        f"than the full population."
    )


def _exam_step2_frame_draw(frame_size, n, rng) -> str:
    """``n`` is already guaranteed <= frame_size by the caller (``_draw_k_and_n_for_template``'s own k*n <=
    population invariant implies n <= population // k == frame_size, since both are integers), so no separate
    clamp is needed here. Two clean, genuinely valid draw mechanisms, matching this module's own internal
    'valid' vocabulary (barrel / random number generator)."""
    if rng.random() < 0.5:
        return (
            f"Every name on that list is written on an identical slip; the slips are placed "
            f"in a barrel, mixed thoroughly, and {n} are drawn out without looking."
        )
    return (
        f"A random number generator assigns every one of the {frame_size} names on that list "
        f"a unique number, and the {n} with the smallest numbers are selected."
    )


# Which flaw classes now have a non-inert Step 1 or Step 3 variant, and which step it sits in.
# Exclusion/self_select/systematic already had theirs; convenience is new -- the earlier round's
# oracle checked its "valid" pattern before its "convenience" pattern, so a convenience flaw sitting
# only in Step 3 while Step 2 reads as a clean valid method was indistinguishable from a genuinely
# valid draw to that (whole-text) oracle.
_EXAM_ALT_MODE = {
    "exclusion": "step1",
    "systematic": "step1",
    "self_select": "step3",
    "convenience": "step3",
}
# Probability of using the alt (non-Step-2) mode when the flaw class has one. All 4 flaw classes now
# have an alt mode, so the expected "Step 1 and Step 3 both clean" fraction is 0.35 (valid draws,
# always both clean) + 0.65 * (1 - prob) (flaw draws that land in step2 mode). At 0.85, that is 0.35
# + 0.65*0.15 = 0.4475 -- comfortably under the 50% ceiling.
_EXAM_ALT_MODE_PROB = 0.85

# The 3 valid templates with no k of their own -- used for Step 2 whenever
# Step 1 itself already carries a k-based (systematic) construction, so a
# single exam prompt never states two independent "every Kth ..." clauses
# (which would also make it ambiguous which one a human solver, or this
# module's own k*n invariant, is meant to read).
_VALID_NON_K_TEMPLATES = [_valid_hat, _valid_rng_smallest, _valid_stratified]


def _exam(rng):
    """Exam tier: a genuine three-step procedure where WHICH step carries the flaw (or whether none does) varies
    by draw, so a learner cannot skip Steps 1 and 3 as filler. Step 1 can be a biased sampling frame (the
    frame itself excludes part of the stated population, or is built by a non-random every-Nth pass, even if
    Step 2's drawing mechanism is a clean random draw); Step 3 can turn an otherwise-clean draw into a
    self-selection situation (only those who choose to attend are ultimately counted); or, as before, the flaw
    can sit in Step 2 itself with Steps 1 and 3 clean."""
    target_is_valid = rng.random() < 0.35
    pop = rng.choice(POPULATIONS)
    n = rng.randint(15, 60)
    exclude_flaws = None

    if target_is_valid:
        truth = "valid"
        template = rng.choice(ALL_VALID_TEMPLATES)
        step2_text, n_used = _render_method(pop, n, rng, template)
        check(
            _classify_from_text(step2_text) == "valid",
            f"p19d_random_sample_valid/exam: Step 2 text classifies as "
            f"{_classify_from_text(step2_text)!r}, expected 'valid': {step2_text!r}",
        )
        step1 = _exam_frame_step_clean(pop)
        step3 = _exam_close_step_clean(pop, n_used)
        flaw_location = "none -- every step is clean"
    else:
        truth = rng.choice(FLAW_KEYS)
        alt_mode = _EXAM_ALT_MODE.get(truth)
        mode = alt_mode if alt_mode and rng.random() < _EXAM_ALT_MODE_PROB else "step2"
        exclude_flaws = None

        if mode == "step1" and truth == "exclusion":
            frame_size, n_used = _draw_biased_frame_size_and_n(rng, pop)
            step1 = _exam_frame_step_biased(pop, rng, frame_size)
            step2_text = _exam_step2_frame_subgroup(pop, frame_size, n_used, rng)
            check(
                _classify_from_text(step2_text) == "valid",
                f"p19d_random_sample_valid/exam: Step 2 text (step1-flaw mode) classifies as "
                f"{_classify_from_text(step2_text)!r}, expected 'valid': {step2_text!r}",
            )
            step3 = _exam_close_step_clean(pop, n_used)
            flaw_location = "Step 1 (the sampling frame itself excludes part of the stated population)"
        elif mode == "step1" and truth == "systematic":
            # The k * n <= population invariant applies to this frame too: the frame Step 1 describes can hold at most
            # population // k names, so draw k and the eventual sample size n together
            # (population-compatible) before rendering either sentence.
            k_frame, n_used = _draw_k_and_n_for_template(rng, pop, _sys_unshuffled_every_kth)
            frame_size = pop["size"] // k_frame
            step1 = _exam_frame_step_systematic(pop, rng, k=k_frame)
            step2_text = _exam_step2_frame_draw(frame_size, n_used, rng)
            check(
                _classify_from_text(step2_text) == "valid",
                f"p19d_random_sample_valid/exam: Step 2 text (step1-systematic mode) classifies as "
                f"{_classify_from_text(step2_text)!r}, expected 'valid': {step2_text!r}",
            )
            step3 = _exam_close_step_clean(pop, n_used)
            flaw_location = "Step 1 (the sampling frame itself is compiled via a non-random every-Nth pass)"
            # from the frame, the OVERALL procedure genuinely reads as BOTH
            # "systematic" (the frame itself is a non-random subset) AND
            # "exclusion" (everyone outside the frame was never selectable)
            # -- not separable by wording. Remove the ambiguity by option
            # construction: never offer "exclusion" as a distractor here.
            exclude_flaws = {"exclusion"}
        elif mode == "step3" and truth == "self_select":
            step1 = _exam_frame_step_clean(pop)
            template = rng.choice(ALL_VALID_TEMPLATES)
            step2_text, n_used = _render_method(pop, n, rng, template)
            check(
                _classify_from_text(step2_text) == "valid",
                f"p19d_random_sample_valid/exam: Step 2 text (step3-flaw mode) classifies as "
                f"{_classify_from_text(step2_text)!r}, expected 'valid': {step2_text!r}",
            )
            step3 = _exam_close_step_self_select(pop, n_used)
            flaw_location = "Step 3 (only those who choose to attend are ultimately counted)"
        elif mode == "step3" and truth == "convenience":
            step1 = _exam_frame_step_clean(pop)
            template = rng.choice(ALL_VALID_TEMPLATES)
            step2_text, n_used = _render_method(pop, n, rng, template)
            check(
                _classify_from_text(step2_text) == "valid",
                f"p19d_random_sample_valid/exam: Step 2 text (step3-convenience mode) classifies as "
                f"{_classify_from_text(step2_text)!r}, expected 'valid': {step2_text!r}",
            )
            step3 = _exam_close_step_convenience(pop, n_used)
            check(
                _classify_from_text(step3) == "convenience",
                f"p19d_random_sample_valid/exam: Step 3 text (step3-convenience mode) classifies as "
                f"{_classify_from_text(step3)!r}, expected 'convenience': {step3!r}",
            )
            flaw_location = "Step 3 (only those conveniently reachable at follow-up are ultimately counted)"
        else:
            step1 = _exam_frame_step_clean(pop)
            candidates = FLAW_TEMPLATES[truth]
            if truth == "convenience":
                candidates = _filter_context_compatible(candidates, pop)
            template = rng.choice(candidates)
            step2_text, n_used = _render_method(pop, n, rng, template)
            check(
                _classify_from_text(step2_text) == truth,
                f"p19d_random_sample_valid/exam: Step 2 text classifies as "
                f"{_classify_from_text(step2_text)!r}, expected {truth!r}: {step2_text!r}",
            )
            step3 = _exam_close_step_clean(pop, n_used)
            flaw_location = "Step 2"

    correct, distractors = _build_options(rng, truth, pop, exclude_flaws=exclude_flaws)

    text = (
        f"{DEFINITION_SENTENCE} {step1} Step 2: {step2_text} {step3} Classify whether the "
        f"overall three-step procedure above produces a valid random sample."
    )
    return dict(
        text=text,
        latex=text,
        answer=correct,
        answer_text=correct,
        answer_latex=latex_answer(correct, plain=True),
        distractors=distractors,
        working=[f"True classification: {correct}.", f"Flaw location: {flaw_location}."],
    )


@generator(
    "p19d_random_sample_valid",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19d_random_sample_valid(rng, tier="easy"):
    """Judge whether a described method for selecting individuals from a
    stated population constitutes a valid random sample (ACMMM171). NOT the
    same operation as senior-simulate-repeated-random-sampling-from-a-variety-of
    (which simulates FROM an already-valid random sample) and NOT the same
    operation as senior-identify-the-target-population-to-be-surveyed (which
    judges the population description, never a selection method's
    validity)."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise ValueError(f"p19d_random_sample_valid: unknown tier {tier!r}")


# ============================================================================
# Skill 2: p19d_target_population
# ============================================================================

# The old CONTEXTS below had a single fixed
# ``attribute_full``/``qualifier_clause``/``qualifier_attribute`` per context (a pool of exactly
# len(CONTEXTS) prompts per tier); a single ``narrow``/``broad`` phrase that shared no vocabulary
# with the research question's own "{full}" wording (the key, "all {full}", was always the unique
# option echoing that wording, and easy/medium's key-unique-max-Jaccard was 200/200); a ``broad``
# field whose text already started with "all " for three contexts, doubled by ``_wrap(f"all
# {ctx['broad']}")`` into "all all..." (46/43/18 of 200 draws in easy/medium/hard); and a
# context-free ``SURVEYOR_WRONG`` distractor with zero content-word overlap with any scenario (a
# "never a defensible key" tell) that was also short and fixed, driving exam's
# key-is-longest-100%-of-draws tell.
#
# Fix: every context now offers 3 genuinely different ``attribute_variants``
# (varying WHICH attribute the research question asks about) and 3 genuinely
# different ``qualifier_variants`` (varying WHICH qualifying sub-group the
# question restricts to), randomly chosen per draw -- 12 contexts x 3
# variants = 36 combinations feeding easy/medium/exam's research-question
# text, comfortably clearing the >=20-distinct-prompts-at-60-seeds floor.
# ``narrow_clause``/``other_class`` replace the old ``narrow``/``broad``
# fields and are always phrased so the rendered distractor text contains
# ``ctx['full']`` itself (the same words the research question restates),
# removing the "key is the only option echoing the question" tell: every
# offered option -- correct and every distractor -- now shares that same
# core noun phrase, so raw word-overlap with the stem no longer singles out
# the key. There is no longer any "all {something}" construction applied to
# a phrase that might itself start with "all " (the old ``broad`` bug), and
# ``SURVEYOR_WRONG`` is gone entirely -- replaced by context-derived
# subgroup/superset/frame distractors of comparable, varying length (see
# ``_subgroup_option``/``_superset_option``/``_frame_surveyed_option``/
# ``_frame_reach_option`` below).
CONTEXTS = [
    dict(
        population_statement="Westfield College has 850 students enrolled.",
        full="students enrolled at Westfield College",
        attribute_variants=[
            "travel to school by bus",
            "bring a packed lunch rather than buying from the canteen",
            "own a mobile phone with a data plan",
        ],
        qualifier_variants=[
            ("who play a school sport", "intend to attend the interschool sports carnival"),
            ("who catch the school bus", "want an earlier bus service"),
            ("who are in the school choir", "want more rehearsal time before assembly"),
        ],
        narrow_clause="who are in Year 12",
        other_class="the staff who work at the college",
        frame_label="the college's canteen-app users",
        sample_n=60,
    ),
    dict(
        population_statement="The town of Mapleton has 2,000 residents.",
        full="residents of the town of Mapleton",
        attribute_variants=[
            "use public transport at least once a week",
            "recycle their household waste weekly",
            "shop at the local Mapleton markets",
        ],
        qualifier_variants=[
            ("who intend to vote in the upcoming council election", "support the proposed new highway"),
            ("who use the town's public library", "want longer opening hours"),
            ("who commute outside Mapleton for work", "want a direct bus service to the city"),
        ],
        narrow_clause="who are aged under 18",
        other_class="residents of the neighbouring village of Fernbrook",
        frame_label="people listed in the local phone directory",
        sample_n=150,
    ),
    dict(
        population_statement="The Northgate distribution centre has 85 employees.",
        full="employees at the Northgate distribution centre",
        attribute_variants=[
            "commute to work by car",
            "work more than one weekend shift a month",
            "have completed forklift safety training",
        ],
        qualifier_variants=[
            ("who work in the warehouse rather than the office", "want additional forklift safety training"),
            ("who work the night shift", "want a shift-swap system introduced"),
            ("who drive a company vehicle", "want a fuel card provided"),
        ],
        narrow_clause="who are on the night shift",
        other_class="drivers who deliver from the centre but are contracted externally",
        frame_label="staff with a company email address",
        sample_n=30,
    ),
    dict(
        population_statement="The Fairview Community Gym has 620 members.",
        full="members of the Fairview Community Gym",
        attribute_variants=[
            "use the gym at least twice a week",
            "have a personal training package",
            "use the gym's swimming pool",
        ],
        qualifier_variants=[
            ("who attend group fitness classes", "want more evening class times"),
            ("who use the gym's swimming pool", "want extended pool hours"),
            ("who joined within the last year", "want a new-member orientation session"),
        ],
        narrow_clause="who joined this year",
        other_class="residents of the Fairview suburb who are not members",
        frame_label="users of the gym's booking app",
        sample_n=80,
    ),
    dict(
        population_statement="Dr Alvarez's clinic roster has 410 patients.",
        full="patients on Dr Alvarez's clinic roster",
        attribute_variants=[
            "attend appointments in person rather than via telehealth",
            "have a chronic condition requiring regular review",
            "have already received their flu vaccination",
        ],
        qualifier_variants=[
            ("who have a scheduled flu vaccination this month", "have already received their flu vaccination"),
            ("who are aged over 65", "want a home-visit option"),
            ("who book appointments online", "want more available online timeslots"),
        ],
        narrow_clause="who have been on the roster for over 10 years",
        other_class="patients of other clinics in the medical district",
        frame_label="patients with a mobile number on file",
        sample_n=50,
    ),
    dict(
        population_statement="Ridgeview High School has 1,100 students enrolled.",
        full="students enrolled at Ridgeview High School",
        attribute_variants=[
            "bring a packed lunch rather than buying from the canteen",
            "catch the school bus",
            "own a mobile phone",
        ],
        qualifier_variants=[
            ("who catch the school bus", "want an earlier bus service"),
            ("who play a school sport", "want a new sports uniform"),
            ("who study a language subject", "want an extra language elective offered"),
        ],
        narrow_clause="who are in Year 7",
        other_class="families living in the Ridgeview school catchment area",
        frame_label="students who follow the school's social media account",
        sample_n=70,
    ),
    dict(
        population_statement="Harbourside Library has 3,400 registered members.",
        full="registered members of Harbourside Library",
        attribute_variants=[
            "visit the library at least once a month",
            "borrow audiobooks",
            "use the library's free wifi",
        ],
        qualifier_variants=[
            ("who borrow audiobooks", "would use a longer audiobook loan period"),
            ("who attend the library's weekend workshops", "want more workshop sessions"),
            ("who use the library's study rooms", "want longer study-room booking slots"),
        ],
        narrow_clause="who are aged under 12",
        other_class="residents who live within the library's local government area",
        frame_label="members with an email address on file",
        sample_n=90,
    ),
    dict(
        population_statement="Coastal Rowers Club has 240 members.",
        full="members of Coastal Rowers Club",
        attribute_variants=[
            "attend training at least twice a week",
            "row competitively in regattas",
            "own their own rowing boat",
        ],
        qualifier_variants=[
            ("who row in the masters division", "want more weekend training sessions"),
            ("who row competitively", "want additional coaching sessions"),
            ("who use the club's gym facilities", "want extended gym access hours"),
        ],
        narrow_clause="who are under 18",
        other_class="residents of the coastal suburb where the club is based",
        frame_label="members listed on the club's regatta roster",
        sample_n=40,
    ),
    dict(
        population_statement="Summit University's engineering faculty has 5,200 students.",
        full="engineering students at Summit University",
        attribute_variants=[
            "work a part-time job during semester",
            "commute more than an hour to campus",
            "use the campus library after hours",
        ],
        qualifier_variants=[
            ("who are enrolled in an industry placement unit", "found their industry placement useful"),
            ("who live in on-campus accommodation", "want extended accommodation contracts"),
            ("who are in their final year", "want more careers-fair sessions"),
        ],
        narrow_clause="who are in first year",
        other_class="students enrolled in other faculties at Summit University",
        frame_label="students subscribed to the faculty's email newsletter",
        sample_n=120,
    ),
    dict(
        population_statement="Brightside Cafe Chain has 18,000 loyalty-app members.",
        full="loyalty-app members of Brightside Cafe Chain",
        attribute_variants=[
            "visit a Brightside store at least once a week",
            "order a hot drink most visits",
            "use the loyalty app to pay in-store",
        ],
        qualifier_variants=[
            ("who order a hot drink most visits", "would try a new seasonal drink"),
            ("who visit more than one Brightside store", "want a cross-store rewards system"),
            ("who redeem rewards points regularly", "want more frequent reward offers"),
        ],
        narrow_clause="who joined in the last month",
        other_class="customers who have purchased in-store but never joined the loyalty app",
        frame_label="members who have opted in to push notifications",
        sample_n=200,
    ),
    dict(
        population_statement="Lakeside Council manages a region of 6,500 households.",
        full="households in the Lakeside Council region",
        attribute_variants=[
            "compost their food scraps at home",
            "use kerbside recycling collection",
            "own a rainwater tank",
        ],
        qualifier_variants=[
            ("who use kerbside recycling collection", "want a second recycling bin"),
            ("who have a home compost bin", "want a council-subsidised compost bin"),
            ("who live in an apartment rather than a house", "want a shared recycling facility"),
        ],
        narrow_clause="with children under 5",
        other_class="households in the wider metropolitan area beyond Lakeside Council",
        frame_label="households who receive the council's e-newsletter",
        sample_n=110,
    ),
    dict(
        population_statement="Peak Fitness Studio has 480 class-pass holders.",
        full="class-pass holders at Peak Fitness Studio",
        attribute_variants=[
            "attend at least one class a week",
            "have a multi-class weekly pass",
            "use the studio's booking app",
        ],
        qualifier_variants=[
            ("who attend morning classes", "want an earlier morning class time"),
            ("who attend evening classes", "want more evening class slots"),
            ("who have a multi-class weekly pass", "want a discounted annual pass"),
        ],
        narrow_clause="who joined this quarter",
        other_class="residents of the suburb where the studio is located",
        frame_label="holders who have the studio's booking app installed",
        sample_n=60,
    ),
]


def _wrap(phrase: str) -> str:
    return f"The target population is {phrase}."


def _as_verb_phrase(clause: str) -> str:
    """Strip a leading "who " so a relative clause ("who play a school
    sport") can be re-used as a plain verb phrase ("some also play a school
    sport")."""
    return clause[4:] if clause.startswith("who ") else clause


# ---- context-derived distractor shapes (replace the removed SURVEYOR_WRONG) ----
#
# Every shape below embeds ``ctx['full']`` -- the same core noun phrase the research question itself
# restates -- so no offered option (correct or distractor) is the unique one echoing the question's
# own wording.


def _subgroup_option(ctx) -> str:
    """A genuine subgroup of the stated population -- narrower than what the
    research question actually asks about."""
    return _wrap(f"{ctx['full']} {ctx['narrow_clause']}")


def _superset_option(ctx) -> str:
    """A genuine superset of the stated population -- broader than what the
    research question actually asks about."""
    return _wrap(f"{ctx['full']} and {ctx['other_class']}")


def _frame_surveyed_option(ctx, n) -> str:
    """The sample actually drawn, mistaken for the population it was drawn to represent. Deliberately does NOT
    repeat ``ctx['full']`` (unlike the other distractor shapes above): its own length is anchored to the
    (per-context) sample size ``n``'s digit count and the fixed instructional wording, which is sometimes
    shorter and sometimes longer than ``ctx['full']`` itself -- this is what supplies the exam/easy/medium
    length-tell fix, since a fixed-full-echoing distractor shape would always be longer than the bare "all
    {full}" key by a constant margin."""
    if n is not None:
        return _wrap(f"the {n} people who were actually surveyed")
    # The stem gives no sample size, so none may be invented; the wording varies by context instead, so this
    # distractor is not one fixed string shown in nearly every draw.
    wordings = (
        "the people who were actually surveyed",
        "only the people who answered the survey",
        "whoever the researcher happened to ask",
        "the people in the researcher's sample",
    )
    return _wrap(wordings[len(ctx["full"]) % len(wordings)])


def _consent_subgroup_option(ctx, clause) -> str:
    """A narrower subgroup of the QUALIFIED population the research question
    actually asks about -- reuses the question's own qualifier clause
    verbatim (so a raw stem/option word-overlap count cannot single out the
    key by being the only option that echoes the clause; a distractor this
    close to correct is exactly what forces a genuine reasoning check rather
    than a surface-word shortcut), then adds one further, genuinely
    narrower restriction that the question itself never states."""
    return _wrap(f"{ctx['full']} {clause}, but only those who returned a signed consent form")


def _frame_reach_option(ctx) -> str:
    """The sampling frame's own reach, mistaken for the stated population --
    used only in the exam tier, where the frame restriction is stated in the
    stem. Deliberately bare (does not repeat ``ctx['full']``), for the same
    length-variety reason as ``_frame_surveyed_option`` above: ``frame_label``
    is itself a genuinely per-context, scenario-derived phrase (not a fixed
    universal string like the removed SURVEYOR_WRONG) whose own length varies
    from context to context, sometimes shorter and sometimes longer than the
    key -- unlike a fixed-length or a full-echoing shape, which would recreate
    a length tell in the other direction."""
    return _wrap(ctx["frame_label"])


def _tp_easy(rng):
    ctx = rng.choice(CONTEXTS)
    attribute = rng.choice(ctx["attribute_variants"])
    text = f"{ctx['population_statement']} Research question: how many {ctx['full']} {attribute}?"
    correct = _wrap(f"all {ctx['full']}")
    options = [
        correct,
        _subgroup_option(ctx),
        _superset_option(ctx),
        _frame_surveyed_option(ctx, None),
    ]
    check(len(set(options)) == 4, "p19d_target_population/easy: options not distinct")
    rng.shuffle(options)
    distractors = [o for o in options if o != correct]
    return dict(text=text, correct=correct, distractors=distractors)


def _tp_medium(rng):
    ctx = rng.choice(CONTEXTS)
    n = ctx["sample_n"]
    attribute = rng.choice(ctx["attribute_variants"])
    text = (
        f"{ctx['population_statement']} A researcher surveys {n} {ctx['full']} to "
        f"investigate this. Research question: how many {ctx['full']} {attribute}?"
    )
    correct = _wrap(f"all {ctx['full']}")
    options = [
        correct,
        _subgroup_option(ctx),
        _superset_option(ctx),
        _frame_surveyed_option(ctx, n),
    ]
    check(len(set(options)) == 4, "p19d_target_population/medium: options not distinct")
    rng.shuffle(options)
    distractors = [o for o in options if o != correct]
    return dict(text=text, correct=correct, distractors=distractors)


def _tp_hard(rng):
    ctx = rng.choice(CONTEXTS)
    qualified_case = rng.random() < 0.5
    clause, qualifier_attribute = rng.choice(ctx["qualifier_variants"])
    if qualified_case:
        text = (
            f"{ctx['population_statement']} Research question: how many {ctx['full']} {clause} {qualifier_attribute}?"
        )
        correct = _wrap(f"{ctx['full']} {clause}")
        options = [
            correct,
            _wrap(f"all {ctx['full']}"),
            _consent_subgroup_option(ctx, clause),
            _superset_option(ctx),
        ]
    else:
        attribute = rng.choice(ctx["attribute_variants"])
        note = f"Of {ctx['full']}, some also {_as_verb_phrase(clause)}."
        text = f"{ctx['population_statement']} {note} Research question: how many {ctx['full']} {attribute}?"
        correct = _wrap(f"all {ctx['full']}")
        options = [
            correct,
            _wrap(f"{ctx['full']} {clause}"),
            _superset_option(ctx),
            _frame_surveyed_option(ctx, None),
        ]
    check(len(set(options)) == 4, "p19d_target_population/hard: options not distinct")
    rng.shuffle(options)
    distractors = [o for o in options if o != correct]
    return dict(text=text, correct=correct, distractors=distractors)


def _tp_exam(rng):
    ctx = rng.choice(CONTEXTS)
    clause, qualifier_attribute = rng.choice(ctx["qualifier_variants"])
    text = (
        f"{ctx['population_statement']} To reach them, the survey is only distributed to "
        f"{ctx['frame_label']}. Research question: how many {ctx['full']} "
        f"{clause} {qualifier_attribute}?"
    )
    correct = _wrap(f"{ctx['full']} {clause}")
    options = [
        correct,
        _wrap(f"all {ctx['full']}"),
        _consent_subgroup_option(ctx, clause),
        _frame_reach_option(ctx),
    ]
    check(len(set(options)) == 4, "p19d_target_population/exam: options not distinct")
    rng.shuffle(options)
    distractors = [o for o in options if o != correct]
    return dict(text=text, correct=correct, distractors=distractors)


def _target_population_question(built: dict) -> dict:
    text, correct, distractors = built["text"], built["correct"], built["distractors"]
    return dict(
        text=text,
        latex=text,
        answer=correct,
        answer_text=correct,
        answer_latex=latex_answer(correct, plain=True),
        distractors=distractors,
        working=[f"Correct target population: {correct}"],
    )


@generator(
    "p19d_target_population",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19d_target_population(rng, tier="easy"):
    """Given a stated survey scenario and its research question, choose the
    correct description of the target population (matching the research
    question exactly -- neither narrower nor broader) from the offered
    options (ACMEM132). NOT the same operation as
    senior-understand-the-concept-of-a-random-sample (which judges whether a
    SELECTION METHOD from an already-identified population is valid, never
    what the population description should be)."""
    if tier == "easy":
        built = _tp_easy(rng)
    elif tier == "medium":
        built = _tp_medium(rng)
    elif tier == "hard":
        built = _tp_hard(rng)
    elif tier == "exam":
        built = _tp_exam(rng)
    else:
        raise ValueError(f"p19d_target_population: unknown tier {tier!r}")
    return _target_population_question(built)
