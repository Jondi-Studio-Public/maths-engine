"""Random-sample validity (senior-understand-the-concept-of-a-random-sample, ACMMM171) and target population
(senior-identify-the-target-population-to-be-surveyed, ACMEM132)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

TOPIC_RSV = "p19d_random_sample_valid"
TOPIC_TP = "p19d_target_population"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


# VALID_LABEL is no longer a single fixed global string -- it is built per-scenario (a function of
# the drawn population), specifically so it does not lexically echo DEFINITION_SENTENCE's own
# wording ("equal, independent chance") and so the fourth option is never a memorisable fixed
# string. Every valid label still starts with "Valid --" and every flaw label starts with "Not valid
# --", so tests that need to ask "is this option the valid one" use that structural marker instead
# of a literal string equality against one fixed sentence.
def _is_valid_label(text: str) -> bool:
    return text.startswith("Valid --")


# ============================================================================
# p19d_random_sample_valid
# ============================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_exactly_four_options_always(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        options = [q.answer_text, *q.distractors]
        assert len(set(options)) == 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_valid_option_always_offered(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        assert any(_is_valid_label(o) for o in [q.answer_text, *q.distractors])


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_real_mcq_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, i).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        bogus_idx = len(options) + 5
        assert not grade_mcq(q, bogus_idx).correct
        assert not grade_mcq(q, "not-a-number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_definition_stated_explicitly(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        assert "equal, independent chance" in q.text


def test_rsv_exam_uses_numbered_steps_not_present_in_hard():
    for seed in SEEDS:
        q_hard = make(TOPIC_RSV, seed=seed, difficulty="hard")
        q_exam = make(TOPIC_RSV, seed=seed, difficulty="exam")
        assert "Step 1:" not in q_hard.text
        assert "Step 1:" in q_exam.text and "Step 2:" in q_exam.text and "Step 3:" in q_exam.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_valid_key_concentration_between_25_and_45_percent(difficulty):
    hits = sum(1 for s in range(200) if _is_valid_label(make(TOPIC_RSV, seed=s, difficulty=difficulty).answer_text))
    frac = hits / 200
    assert 0.25 <= frac <= 0.45, f"{difficulty}: valid concentration {frac:.3f} ({hits}/200)"


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_distractor_count_distribution_always_three(difficulty):
    counts = Counter(len(make(TOPIC_RSV, seed=s, difficulty=difficulty).distractors) for s in range(200))
    assert counts == Counter({3: 200}), f"{difficulty}: distractor-count distribution {counts}"


# Generic/definitional/instructional words that appear in EVERY draw's stem regardless of truth, or
# that are legitimate shared answer vocabulary across more than one option already, are not the tell
# being measured here and are stripped before counting -- mirrors the RED script's own stopword
# list.
_GENERIC_STOPWORDS_RSV = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "who",
    "is",
    "are",
    "to",
    "by",
    "for",
    "with",
    "that",
    "this",
    "and",
    "or",
    "from",
    "as",
    "it",
    "its",
    "their",
    "them",
    "were",
    "was",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "classify",
    "whether",
    "selection",
    "method",
    "produces",
    "valid",
    "random",
    "sample",
    "population",
    "member",
    "members",
    "stated",
    "not",
    "part",
    "being",
    "into",
    "some",
    "would",
    "each",
    "end",
    "up",
    "no",
    "one's",
    "anyone",
    "else's",
    # Generic outcome verbs a scenario sentence ends on regardless of its
    # class ("... until 40 residents have been chosen" / "are selected" /
    # "are used" / "dialled at random" / "eligible to be randomly called")
    # -- functionally synonyms of "selected" (already generic above), not a
    # genuine class-specific tell.
    "chosen",
    "picked",
    "included",
    "used",
    "dialled",
    "called",
    "drawn",
    "taken",
    "any",
    "they",
    "themselves",
}


def _content_words_rsv(text: str) -> set:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in _GENERIC_STOPWORDS_RSV and len(w) > 2}


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_no_stem_word_is_a_reliable_key_tell(difficulty):
    """No content word that appears in the stem should also appear in
    exactly one option -- with that option being the key -- in more than
    35% of the draws where the word is present at all (25% is chance for a
    1-in-4 mcq; 35% is the agreed ceiling)."""
    per_word_hits = Counter()
    per_word_total = Counter()
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        stem_words = _content_words_rsv(q.text)
        options = [q.answer_text, *q.distractors]
        for w in stem_words:
            opts_with_w = [o for o in options if w in _content_words_rsv(o)]
            if len(opts_with_w) == 1:
                per_word_total[w] += 1
                if opts_with_w[0] == q.answer_text:
                    per_word_hits[w] += 1
    offenders = {
        w: (per_word_hits[w], total)
        for w, total in per_word_total.items()
        if total >= 10 and per_word_hits[w] / total > 0.35
    }
    assert not offenders, f"{difficulty}: stem words that reliably tell the key: {offenders}"


_SYSTEMATIC_EVERY_NTH_CHECK = re.compile(
    r"every\s+\d+\w{0,2}\s+(name|person|entry).*"
    r"(unshuffled|never\s+reordered|never\s+reshuffled|never\s+randomised|no\s+randomising|"
    r"original(\s+printed)?\s+order|fixed\s+order)"
    r"|"
    r"(unshuffled|never\s+reordered|never\s+reshuffled|never\s+randomised|no\s+randomising|"
    r"original(\s+printed)?\s+order|fixed\s+order).*every\s+\d+\w{0,2}\s+(name|person|entry)",
    re.IGNORECASE | re.DOTALL,
)


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_systematic_keyed_prompts_are_genuine_every_nth(difficulty):
    """Every draw whose key is the systematic-selection label must contain a genuine
    every-Nth-from-the-existing-order construction with an explicit no-random-start statement."""
    systematic_keyed = 0
    matches = 0
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        if "systematic selection" in q.answer_text:
            systematic_keyed += 1
            if _SYSTEMATIC_EVERY_NTH_CHECK.search(q.text):
                matches += 1
    assert systematic_keyed >= 5, f"{difficulty}: too few systematic-keyed draws to test ({systematic_keyed})"
    assert matches == systematic_keyed, (
        f"{difficulty}: {systematic_keyed - matches}/{systematic_keyed} systematic-keyed draws "
        f"are not a genuine every-Nth-from-existing-order construction"
    )


def test_rsv_exam_flaw_location_varies_and_step2_can_be_clean_while_overall_invalid():
    """Confirm at least 2 distinct flaw-carrying locations occur across 60 seeds, and at least one draw exists
    where Step 2 alone reads as a clean/valid method but the overall key is non-valid (the flaw sits in Step 1
    or Step 3 instead)."""
    locations = set()
    step2_clean_but_overall_invalid = 0
    step2_clean_re = re.compile(
        r"randomly\s+shuffled|random\s+number\s+generator\s+assigns|\bbarrel\b|\bstrata\b",
        re.IGNORECASE,
    )
    for seed in range(60):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        m = re.search(r"Step 2: (.*?) Step 3:", q.text, re.DOTALL)
        assert m, f"no Step 2 sentence found in {q.text!r}"
        step2_text = m.group(1)
        overall_valid = _is_valid_label(q.answer_text)
        if step2_clean_re.search(step2_text):
            locations.add("step2_clean")
            if not overall_valid:
                step2_clean_but_overall_invalid += 1
        else:
            locations.add("step2_flawed")
    assert len(locations) >= 2, f"exam flaw-location shapes seen: {locations}"
    assert step2_clean_but_overall_invalid >= 1, (
        "expected at least one exam draw where Step 2 alone is clean but the overall "
        "classification is non-valid (flaw in Step 1 or Step 3)"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_option_position_not_identifiable_by_absent_flaw_words(difficulty):
    """Over many draws, no single flaw-word is absent so consistently that
    its absence alone reveals which position holds the valid option."""
    absent_flaw_when_valid = Counter()
    valid_draws = 0
    for s in range(300):
        q = make(TOPIC_RSV, seed=s, difficulty=difficulty)
        if not _is_valid_label(q.answer_text):
            continue
        valid_draws += 1
        offered = set(q.distractors)
        for flaw in [
            "convenience sample",
            "self-selection",
            "systematic selection",
            "excluded from ever being selected",
        ]:
            if not any(flaw in d for d in offered):
                absent_flaw_when_valid[flaw] += 1
    assert valid_draws >= 20
    for flaw, n in absent_flaw_when_valid.items():
        assert n / valid_draws < 0.9, f"{difficulty}: {flaw!r} absent on {n}/{valid_draws} valid draws"


def test_rsv_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-understand-the-concept-of-a-random-sample"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}


# ============================================================================
# p19d_target_population
# ============================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_real_mcq_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, i).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        bogus_idx = len(options) + 5
        assert not grade_mcq(q, bogus_idx).correct
        assert not grade_mcq(q, "not-a-number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_all_options_formatted_alike(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        for opt in [q.answer_text, *q.distractors]:
            assert opt.startswith("The target population is ") and opt.endswith(".")


def test_tp_exam_uses_a_two_clause_question_and_sampling_frame_not_present_in_hard():
    for seed in SEEDS:
        q_exam = make(TOPIC_TP, seed=seed, difficulty="exam")
        assert "only distributed to" in q_exam.text
        assert q_exam.text.count("Research question:") == 1


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_tp_easy_medium_question_is_direct_restatement_of_full_population(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        assert q.answer_text.startswith("The target population is all ")


def test_tp_hard_reaches_both_obvious_and_qualified_correct_answers():
    obvious = qualified = 0
    for seed in range(200):
        q = make(TOPIC_TP, seed=seed, difficulty="hard")
        if q.answer_text.startswith("The target population is all "):
            obvious += 1
        else:
            qualified += 1
    assert obvious >= 20 and qualified >= 20, f"obvious={obvious} qualified={qualified}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_key_option_position_concentration_under_30_percent(difficulty):
    positions = Counter()
    for s in range(200):
        q = make(TOPIC_TP, seed=s, difficulty=difficulty)
        options = mcq_options(q)
        positions[options.index(q.answer_text)] += 1
    for pos, n in positions.items():
        assert n / 200 < 0.30, f"{difficulty}: position {pos} concentration {n / 200:.3f}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_no_all_all_substring(difficulty):
    """Contexts whose superset/broader phrase itself started with "all " produced a literal "all all households"
    / "all all students" once wrapped by another "all " prefix."""
    for seed in range(200):
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        rendered = (q.text + " " + q.answer_text + " " + " ".join(q.distractors)).lower()
        assert "all all" not in rendered, f"{difficulty} seed={seed}: {rendered!r}"


# This stopword list and content-word intersection technique is deliberately duplicated from the RSV
# (skill 1) test file's own local metric rather than imported from the generator's own
# CONTEXTS/qualifier tables -- it must work purely off the rendered text.
_STOPWORDS_TP_TEST = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "who",
    "is",
    "are",
    "to",
    "by",
    "for",
    "with",
    "that",
    "this",
    "and",
    "or",
    "from",
    "as",
    "it",
    "its",
    "their",
    "them",
    "were",
    "was",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "how",
    "many",
    "some",
    "also",
    "all",
    "target",
    "population",
}


def _content_words_tp_test(text: str) -> set:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS_TP_TEST and len(w) > 2}


def _rq_sentence(text: str) -> str:
    m = re.search(r"research question:\s*(.+?)\?", text, re.IGNORECASE)
    assert m, f"no 'Research question:' sentence in {text!r}"
    return m.group(1)


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_key_not_reliably_the_unique_max_overlap_option(difficulty):
    """Raw intersection COUNT against the research question's own content
    words, not a Jaccard ratio normalised by each option's own size: a
    normalised ratio structurally always favours whichever option is the
    SMALLEST subset of the question's own wording (since correct, in the
    unqualified tiers, is exactly that subset, while every distractor adds
    further words the question never states, only ever enlarging its own
    denominator) -- so a Jaccard-style metric here would penalise every
    distractor for the very act of being a distractor, regardless of the
    generator's actual wording-echo tell. A raw count does not have that
    built-in bias: any option whose stated phrase contains at least the same
    stem content words as the key scores at least as high, tying rather than
    losing, which is what a genuine fix (every option sharing the key's own
    core vocabulary, per the generator's ``_subgroup_option``/
    ``_superset_option``/``_frame_*``/``_consent_subgroup_option``
    docstrings) should produce."""
    n = 200
    hits = 0
    for seed in range(n):
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        rq_tokens = _content_words_tp_test(_rq_sentence(q.text))
        options = [q.answer_text, *q.distractors]
        scores = [(o, len(_content_words_tp_test(o) & rq_tokens)) for o in options]
        max_score = max(sc for _, sc in scores)
        top = [o for o, sc in scores if sc == max_score]
        if len(top) == 1 and top[0] == q.answer_text:
            hits += 1
    frac = hits / n
    assert frac <= 0.35, f"{difficulty}: key is unique-max-overlap option in {hits}/{n} ({frac:.1%}) draws"


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_key_not_reliably_longest_or_shortest(difficulty):
    """In the exam tier the key was the longest option 200/200 draws (SURVEYOR_WRONG and the other distractors
    were systematically shorter)."""
    n = 200
    longest_hits = 0
    shortest_hits = 0
    for seed in range(n):
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        options = [q.answer_text, *q.distractors]
        lengths = [(o, len(o)) for o in options]
        max_len = max(l for _, l in lengths)
        min_len = min(l for _, l in lengths)
        longest = [o for o, l in lengths if l == max_len]
        shortest = [o for o, l in lengths if l == min_len]
        if len(longest) == 1 and longest[0] == q.answer_text:
            longest_hits += 1
        if len(shortest) == 1 and shortest[0] == q.answer_text:
            shortest_hits += 1
    assert longest_hits / n <= 0.35, f"{difficulty}: key is longest option in {longest_hits}/{n} draws"
    assert shortest_hits / n <= 0.35, f"{difficulty}: key is shortest option in {shortest_hits}/{n} draws"


def test_tp_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-identify-the-target-population-to-be-surveyed"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}


# Every check below reads only the rendered question text (never the generator's own module),
# matching this file's existing build-test discipline.
# ============================================================================

# ---- k * n must never exceed the stated population ------------

_POP_SIZE_RE = re.compile(
    r"the ([\d,]+) (?:students enrolled at Riverside High School|"
    r"residents of the town of Mapleton|"
    r"employees at the Northgate distribution centre|"
    r"members of the Fairview Community Gym|"
    r"registered voters in the Kestrel electorate|"
    r"patients on Dr Alvarez's clinic roster)",
    re.IGNORECASE,
)
_K_AND_N_RE = re.compile(
    # [^.]*? (not .*?, and not DOTALL) so this never crosses a sentence
    # boundary -- the exam tier can have TWO independent "every Kth ..."
    # constructions in the SAME rendered text (one in Step 1's sampling-frame
    # sentence, a different one in Step 2's own sentence), and a DOTALL/.*?
    # match would wrongly pair Step 1's k with Step 2's n (or vice versa).
    r"every\s+(\d+)\w{0,2}\s+(?:name|person)\b[^.]*?until\s+(\d+)\s+\S+\s+have\s+been\s+chosen",
    re.IGNORECASE,
)


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_k_times_n_never_exceeds_population(difficulty):
    """Every every-kth-style draw (systematic templates and the
    valid shuffled-every-kth template) must select no more names than the
    population physically holds -- k * n <= population, on every draw that
    uses such a template, 200 seeds."""
    violations = []
    checked = 0
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        m_kn = _K_AND_N_RE.search(q.text)
        if not m_kn:
            continue
        checked += 1
        k, n = int(m_kn.group(1)), int(m_kn.group(2))
        m_pop = _POP_SIZE_RE.search(q.text)
        assert m_pop, f"no population size found in {q.text!r}"
        pop_size = int(m_pop.group(1).replace(",", ""))
        if k * n > pop_size:
            violations.append((seed, k, n, pop_size))
    assert checked >= 5, f"{difficulty}: too few k-based draws to test ({checked})"
    assert not violations, (
        f"{difficulty}: {len(violations)}/{checked} k-based draws exceed the population "
        f"(seed, k, n, population): {violations[:10]}"
    )


# ---- A context clause must only be offered for a population it
# fits, per an EXPLICIT, TESTED compatibility table ---------------------

# The tested compatibility table: each population's tag(s), and which templates/strata require which
# tag.
_POP_TAGS_TABLE = {
    "Riverside High School": frozenset({"roll"}),
    "Mapleton": frozenset({"geographic"}),
    "Northgate distribution centre": frozenset({"workplace"}),
    "Fairview Community Gym": frozenset({"membership"}),
    "Kestrel electorate": frozenset({"geographic"}),
    "Dr Alvarez's clinic roster": frozenset({"roll"}),
}
_STRATUM_REQUIRED_TAG = {
    "age group": None,  # always fits
    "suburb": "geographic",
    "membership tier": "membership",
    "department": "workplace",
}


def _pop_key_from_text(text: str) -> str:
    for key in _POP_TAGS_TABLE:
        if key.lower() in text.lower():
            return key
    raise AssertionError(f"no known population found in {text!r}")


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_rsv_same_floor_clause_only_offered_for_workplace_population(difficulty):
    """'... work on the same floor as the researcher' only makes
    sense for a workplace population (Northgate employees)."""
    checked = 0
    violations = []
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        if "same floor as the researcher" not in q.text:
            continue
        checked += 1
        pop_key = _pop_key_from_text(q.text)
        if "workplace" not in _POP_TAGS_TABLE[pop_key]:
            violations.append((seed, pop_key))
    assert checked >= 3, f"{difficulty}: too few same-floor draws to test ({checked})"
    assert not violations, f"{difficulty}: same-floor clause drawn against non-workplace population: {violations}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_stratified_stratum_fits_drawn_population(difficulty):
    """_valid_stratified's stratum choice ('membership tier',
    'department', 'suburb', 'age group') must fit the drawn population's
    tags, per the explicit compatibility table above."""
    checked = 0
    violations = []
    stratum_re = re.compile(r"divided into strata by (age group|suburb|membership tier|department)")
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        m = stratum_re.search(q.text)
        if not m:
            continue
        checked += 1
        stratum = m.group(1)
        required_tag = _STRATUM_REQUIRED_TAG[stratum]
        if required_tag is None:
            continue
        pop_key = _pop_key_from_text(q.text)
        if required_tag not in _POP_TAGS_TABLE[pop_key]:
            violations.append((seed, stratum, pop_key))
    assert not violations, f"{difficulty}: stratum drawn that doesn't fit the population: {violations}"


# ---- Exam Step 1/Step 3 filler ------------------------------


def test_rsv_exam_step1_and_step3_both_clean_falls_under_70_percent():
    """Measure the fraction and assert it now falls under 70% -- a substantial, measured improvement."""
    both_clean = 0
    n = 200
    for seed in range(n):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        m1 = re.search(r"Step 1: (.*?) Step 2:", q.text, re.DOTALL)
        m3 = re.search(r"Step 3: (.*?) Classify whether", q.text, re.DOTALL)
        assert m1 and m3, f"missing Step 1 or Step 3 sentence in {q.text!r}"
        step1_clean = "A sampling frame listing every one of" in m1.group(1)
        step3_clean = "is then contacted and takes part" in m3.group(1)
        if step1_clean and step3_clean:
            both_clean += 1
    frac = both_clean / n
    assert frac < 0.70, f"exam: Step 1 & Step 3 both clean in {both_clean}/{n} ({frac:.1%}) draws"


def test_rsv_exam_step1_or_step3_carries_a_flaw_for_a_second_class_beyond_self_select():
    """Confirm at least one more flaw class now has a non-inert Step 1 (or Step 3) variant."""
    seen_non_inert_extra = False
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        m1 = re.search(r"Step 1: (.*?) Step 2:", q.text, re.DOTALL)
        assert m1
        step1_systematic = "unshuffled register" in m1.group(1) and "every" in m1.group(1)
        if step1_systematic:
            seen_non_inert_extra = True
            break
    assert seen_non_inert_extra, "no non-inert Step 1 variant found for a class beyond exclusion/self_select"


def test_rsv_exam_step1_systematic_frame_k_times_n_within_population():
    """The systematic Step 1 frame variant lists only every k-th name from the register (a frame of population //
    k names), so Step 2's eventual sample size n must not exceed that frame's own size -- i.e. k * n must
    still not exceed the stated population, exactly as the k * n constraint requires of every other k-based construction."""
    checked = 0
    violations = []
    frame_re = re.compile(r"listing only every (\d+)\w{0,2} name")
    n_re = re.compile(r"Step 3: (?:Every one of the|Of the) (\d+)")
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        m_k = frame_re.search(q.text)
        if not m_k:
            continue
        checked += 1
        k = int(m_k.group(1))
        m_n = n_re.search(q.text)
        assert m_n, f"no Step 3 sample size found in {q.text!r}"
        n_final = int(m_n.group(1))
        m_pop = _POP_SIZE_RE.search(q.text)
        assert m_pop, f"no population size found in {q.text!r}"
        pop_size = int(m_pop.group(1).replace(",", ""))
        if k * n_final > pop_size:
            violations.append((seed, k, n_final, pop_size))
    assert checked >= 3, f"too few systematic-frame exam draws to test ({checked})"
    assert not violations, f"systematic frame k*n exceeds population: {violations}"


# ---- Exclusion-key rework -----------------------------------


def test_rsv_skip_already_surveyed_template_no_longer_used():
    """Confirm this exact wording no longer appears in any exclusion-keyed draw."""
    for difficulty in TIERS:
        for seed in range(200):
            q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
            assert "skipped and replaced" not in q.text, (
                f"{difficulty} seed={seed}: debatable exclusion template still in use: {q.text!r}"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rsv_exclusion_pool_still_has_at_least_2_entries(difficulty):
    """If the debatable template was dropped without a replacement, the
    exclusion pool must still offer >= 2 distinct templates per tier so
    exclusion-keyed prompts retain some variety."""
    texts = set()
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        if "excluded from ever being selected" in q.answer_text:
            # Mask digits and population name so only the template SHAPE
            # differs across draws of the same exclusion template.
            shape = re.sub(r"[\d,]+", "N", q.text)
            for key in _POP_TAGS_TABLE:
                shape = shape.replace(key, "POP")
            texts.add(shape)
    assert len(texts) >= 2, f"{difficulty}: only {len(texts)} distinct exclusion-keyed template shapes: {texts}"


# Both frame-thinning exam Step-1 variants must have Step 2 draw from the FRAME they describe, not
# the full stated population; the exclusion ambiguity this created must be removed by option
# construction; and the Step 1/3 "both clean" filler share must fall below 50%.
# ============================================================================

_RSV_POP_PHRASES = [
    "240 students enrolled at Riverside High School",
    "1,500 residents of the town of Mapleton",
    "85 employees at the Northgate distribution centre",
    "620 members of the Fairview Community Gym",
    "3,200 registered voters in the Kestrel electorate",
    "410 patients on Dr Alvarez's clinic roster",
]
_RSV_POP_PHRASE_RE = re.compile("|".join(re.escape(p) for p in _RSV_POP_PHRASES))


def _exam_steps(text: str):
    m = re.search(r"Step 1: (.*?) Step 2: (.*?) Step 3: (.*?) Classify whether", text, re.DOTALL)
    assert m, f"could not split into three Step sentences: {text!r}"
    return m.group(1), m.group(2), m.group(3)


def test_rsv_exam_frame_step2_draws_from_frame_not_full_population():
    """RED proof: both frame-thinning Step-1 variants (systematic
    'listing only every kth name' and biased 'only lists the ... signed up
    to') described a frame SMALLER than the stated population, but Step 2's
    text still named the full population as what it draws from. Over 200
    seeds, this must no longer happen for either variant."""
    systematic_checked = systematic_bad = 0
    biased_checked = biased_bad = 0
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        step1, step2, _step3 = _exam_steps(q.text)
        if "listing only every" in step1 and "th name" in step1 and "rather than the full population" in step1:
            systematic_checked += 1
            if _RSV_POP_PHRASE_RE.search(step2):
                systematic_bad += 1
        if "only lists the" in step1 and "signed up to" in step1:
            biased_checked += 1
            if _RSV_POP_PHRASE_RE.search(step2):
                biased_bad += 1
    assert systematic_checked >= 5, f"too few systematic-frame draws to test ({systematic_checked})"
    assert biased_checked >= 5, f"too few biased-frame draws to test ({biased_checked})"
    assert systematic_bad == 0, (
        f"{systematic_bad}/{systematic_checked} systematic-frame draws still have Step 2 "
        f"naming the full population instead of the frame"
    )
    assert biased_bad == 0, (
        f"{biased_bad}/{biased_checked} biased-frame draws still have Step 2 naming the "
        f"full population instead of the frame"
    )


_FRAME_SIZE_RE = re.compile(r"the (\d+) names on that list|those (\d+) \w+")
_STEP3_N_RE = re.compile(r"Step 3: (?:Every one of the|Of the) (\d+)")


def test_rsv_exam_frame_step2_draw_never_exceeds_frame_size():
    """New swept invariant: for every frame-thinning draw, the n Step 2 actually draws is <= the frame size Step
    1 describes -- not just <= the stated population."""
    checked = 0
    violations = []
    frame_k_re = re.compile(r"listing only every (\d+)\w{0,2} name")
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        step1, step2, step3 = _exam_steps(q.text)
        m_pop = _POP_SIZE_RE.search(q.text)
        assert m_pop
        pop_size = int(m_pop.group(1).replace(",", ""))
        m_n = _STEP3_N_RE.search(f"Step 3: {step3}")
        assert m_n, f"no Step 3 sample size found in {step3!r}"
        n_final = int(m_n.group(1))
        m_k = frame_k_re.search(step1)
        if m_k:
            checked += 1
            frame_size = pop_size // int(m_k.group(1))
            if n_final > frame_size:
                violations.append((seed, "systematic", n_final, frame_size))
            continue
        m_fs = re.search(r"(\d+) [\w\-]+ are signed up", step1)
        if m_fs:
            checked += 1
            frame_size = int(m_fs.group(1))
            if n_final > frame_size:
                violations.append((seed, "biased", n_final, frame_size))
    assert checked >= 5, f"too few frame-thinning draws to test ({checked})"
    assert not violations, f"Step 2 draws more than the frame holds: {violations}"


def test_rsv_exam_systematic_frame_never_offers_exclusion_distractor():
    """The ambiguity half: once Step 2 draws honestly from the frame,
    the overall procedure genuinely reads as both 'systematic' and
    'exclusion'. Fixed by option construction: when truth is the
    systematic-frame variant, 'exclusion' must never be one of the offered
    distractors."""
    checked = 0
    violations = 0
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        step1, _step2, _step3 = _exam_steps(q.text)
        is_systematic_frame = (
            "listing only every" in step1 and "th name" in step1 and "rather than the full population" in step1
        )
        if not is_systematic_frame:
            continue
        if not q.answer_text.startswith("Not valid -- systematic selection"):
            continue
        checked += 1
        if any("excluded from ever being selected" in d for d in q.distractors):
            violations += 1
    assert checked >= 5, f"too few systematic-frame-keyed draws to test ({checked})"
    assert violations == 0, f"{violations}/{checked} systematic-frame draws still offer 'exclusion' as a distractor"


def test_rsv_exam_step1_and_step3_both_clean_falls_under_50_percent():
    """Push the Step-1/Step-3 flaw share further -- from a measured 65% both-clean down to under
    50%, by adding a fourth (convenience-at-Step-3) non-inert variant."""
    both_clean = 0
    n = 200
    for seed in range(n):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        step1, _step2, step3 = _exam_steps(q.text)
        step1_clean = "A sampling frame listing every one of" in step1
        step3_clean = "is then contacted and takes part" in step3
        if step1_clean and step3_clean:
            both_clean += 1
    frac = both_clean / n
    assert frac < 0.50, f"exam: Step 1 & Step 3 both clean in {both_clean}/{n} ({frac:.1%}) draws"


def test_rsv_exam_convenience_at_step3_variant_exists():
    """Confirm the new convenience-at-Step-3 variant actually
    occurs, with Step 2 reading as a clean valid method."""
    seen = False
    for seed in range(200):
        q = make(TOPIC_RSV, seed=seed, difficulty="exam")
        step1, step2, step3 = _exam_steps(q.text)
        if "hardly any extra effort" in step3 and q.answer_text.startswith("Not valid -- convenience"):
            seen = True
            break
    assert seen, "no convenience-at-Step-3 exam variant found over 200 seeds"
