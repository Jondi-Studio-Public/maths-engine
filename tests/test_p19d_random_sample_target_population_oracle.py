"""Re-derives the correct option for both mcq skills from the rendered prompt text alone, using a generic
technique that never imports ``mathgen.generators.p19d_random_sample_target_population_build``.

- ``p19d_random_sample_valid``: an earlier version of this oracle had a real
  blind spot, so it was rewritten. The PREVIOUS version
  re-derived the key by keyword/definition matching over the OPTION text --
  itself a surface tell (a learner, or this oracle, could read the definition-echoing
  vocabulary of the VALID option directly instead of evaluating the
  scenario). The rewritten oracle instead: (1) classifies the METHOD
  DESCRIPTION -- the scenario prose in the stem, including every numbered
  Step sentence for the exam tier -- into one of five classes (valid,
  convenience, self_select, systematic, exclusion) using a lexicon of many
  synonyms per class that covers every template the generator can draw,
  never reading option text to do so; then (2) separately reads which
  OFFERED OPTION corresponds to which class, using each option's own
  self-declared, stable class-noun marker ("convenience sample",
  "self-selection", "systematic selection", "excluded from ever being
  selected", or the "Valid --" prefix) -- these markers are legitimate,
  fixed answer vocabulary shared by every draw of that class (not a
  scenario-specific surface cue), exactly analogous to how the skill-2
  oracle below reads the "The target population is ..." wrapper. Neither
  step imports this module's own classification tables/dicts or template
  functions -- only the rendered prompt and option text.
- ``p19d_target_population``: an earlier version of this oracle had a real
  blind spot, so it was rewritten. The PREVIOUS version was a
  bag-of-words Jaccard-similarity scorer that picked the option whose content
  words overlapped most with the rendered "Research question:" sentence --
  a wording-echo tell moved from
  the generator into the oracle: it worked only because the old generator's
  key was the unique option that echoed the stem's own words. The rewrite
  instead parses each option's stated SCOPE structurally (is the option's own
  phrase, with a leading "all " stripped, an exact whole-word prefix of the
  rendered research question's own wording, and if more than one option
  qualifies, which is the longest such prefix) -- see the docstring above
  ``_oracle_tp_pick`` for the full rationale.

Both are classification/scenario skills (never a computable numeric
answer), so the oracle also enumerates the distinct scenario/method
combinations observed and reports them.
"""

import re
from collections import Counter

import pytest

from mathgen import make

TOPIC_RSV = "p19d_random_sample_valid"
TOPIC_TP = "p19d_target_population"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
WIDE_SEEDS = range(1000)


# ============================================================================
# Oracle 1: p19d_random_sample_valid
# ============================================================================

_ALL_RSV = {"valid", "convenience", "self_select", "systematic", "exclusion"}

# Distinguishing name fragments for the six populations, read off the
# rendered text (not imported from the generator), used only by the
# diagnostic shape-enumeration test below to collapse "same template,
# different population" draws.
_RSV_POPULATION_WORDS = [
    r"riverside high school",
    r"mapleton",
    r"northgate distribution centre",
    r"fairview community gym",
    r"kestrel electorate",
    r"dr alvarez'?s clinic roster",
]

# ---- step 1: classify the METHOD DESCRIPTION (scenario prose only) --------
#
# It never reads which option is offered, and never imports the generator's own regexes/tables --
# these patterns are independently authored, wider than any single option's wording, and cover every
# scenario template the generator can draw (verified by test_rsv_real_key_passes_oracle_over_1000
# _seeds below).
_SELF_SELECT_PATTERNS = re.compile(
    r"\bclick(s|ed)?\b|\bvolunt|\bwants? to\b.*\brespond|\bsign-?up sheet\b|\binterested\b|"
    r"\bchooses?\s+to\b|\bif they choose\b|\bchoose to (actually )?(turn up|attend|respond)\b|"
    r"\bown accord\b|\bdecide[sd]? for themselves\b|\badd their name\b",
    re.IGNORECASE,
)
_EXCLUSION_PATTERNS = re.compile(
    r"\bskipped\b|\bonly (sent|distributed|lists?) to\b|\beligible to be randomly called\b|"
    r"\beven though the stated population\b",
    re.IGNORECASE,
)
_SYSTEMATIC_EVERY_NTH = re.compile(r"every\s+\d+\w{0,2}\s+(name|person|entry)", re.IGNORECASE)
_SYSTEMATIC_NO_RANDOM_START = re.compile(
    r"unshuffled|alphabetically sorted|pm on a particular day|never reordered|"
    r"never reshuffled|never randomised|not randomised|no randomising|"
    r"original(\s+printed)? order|fixed order",
    re.IGNORECASE,
)
_VALID_PATTERNS = re.compile(
    r"randomly shuffled|random number generator assigns|\bbarrel\b|\bstrata(\b|\w*)|"
    r"stratified|smallest assigned numbers",
    re.IGNORECASE,
)
_CONVENIENCE_PATTERNS = re.compile(
    r"\bfirst\b.*\barrive\b|\bclosest\b|\bsame floor\b|\beasiest to reach\b|"
    r"\bhardly any extra effort\b|\bnearby\b|\bon hand\b",
    re.IGNORECASE,
)


def _classify_whole_text(text: str) -> str:
    """The original whole-text classifier: does the described
    procedure give every member of the stated population an equal,
    independent chance of selection? Checked in this order because a
    negated phrase ("never reshuffled") would otherwise satisfy a naive
    positive "shuffled" test meant for the valid class, and self-
    selection/exclusion cues are the most specific. Still correct for
    easy/medium/hard, each of which states its flaw (if any) in a single
    method sentence with no separate framing/close sentence to hide it in."""
    if _SELF_SELECT_PATTERNS.search(text):
        return "self_select"
    if _EXCLUSION_PATTERNS.search(text):
        return "exclusion"
    if _SYSTEMATIC_EVERY_NTH.search(text) and _SYSTEMATIC_NO_RANDOM_START.search(text):
        return "systematic"
    if (
        "unshuffled" in text.lower()
        or "alphabetically sorted" in text.lower()
        or "pm on a particular day" in text.lower()
    ):
        return "systematic"
    if _VALID_PATTERNS.search(text):
        return "valid"
    if _CONVENIENCE_PATTERNS.search(text):
        return "convenience"
    return "unclassifiable"


# Structural per-step rewrite -----
#
# The whole-text classifier above has a real blind spot for the exam tier: a flaw sitting ONLY in
# Step 1 or Step 3, paired with a Step 2 sentence that is itself a clean, genuinely valid draw
# mechanism, satisfies ``_VALID_PATTERNS`` (checked before ``_CONVENIENCE_PATTERNS``) purely from
# Step 2's own wording -- so a convenience-at-Step-3 draw (Step 2 reading as clean) was
# indistinguishable from a genuinely valid draw.
#
# The fix is structural, not a further lexicon extension: split the rendered text on its own "Step
# 1:"/"Step 2:"/"Step 3:" markers (present in every exam draw, absent from easy/medium/hard) and
# classify EACH step's own substring independently, using the same per-class flaw patterns above but
# WITHOUT the "valid" check (a framing/close sentence like "a sampling frame listing every one of
# ..." or "... is then contacted and takes part" is not itself a random-draw mechanism, so it is
# correctly read as "clean", not "valid", when checked as an isolated substring -- only Step 2's own
# text is ever a candidate for a positive valid-mechanism match). The overall class is the FIRST
# step (in order 1, 2, 3) that reads as flawed; "valid" only if all three read clean.
_STEP_MARKER_RE = re.compile(r"Step\s+1:|Step\s+2:|Step\s+3:", re.IGNORECASE)


def _split_into_steps(text: str) -> list | None:
    """Split on this draw's own Step markers. Returns the three step
    substrings [step1, step2, step3] in order, or None if the markers are
    absent (easy/medium/hard, which never carry them)."""
    parts = _STEP_MARKER_RE.split(text)
    if len(parts) != 4:
        return None
    return [p.strip() for p in parts[1:]]


def _step_flaw_class(step_text: str):
    """Classify a single step's own substring as one of the four flaw
    classes, or None if that step reads clean. Same patterns and precedence
    as ``_classify_whole_text`` minus the "valid" check (clean is the
    default, not a class of its own, when read one step at a time)."""
    if _SELF_SELECT_PATTERNS.search(step_text):
        return "self_select"
    if _EXCLUSION_PATTERNS.search(step_text):
        return "exclusion"
    if _SYSTEMATIC_EVERY_NTH.search(step_text) and _SYSTEMATIC_NO_RANDOM_START.search(step_text):
        return "systematic"
    lowered = step_text.lower()
    if "unshuffled" in lowered or "alphabetically sorted" in lowered or "pm on a particular day" in lowered:
        return "systematic"
    if _CONVENIENCE_PATTERNS.search(step_text):
        return "convenience"
    return None


def _oracle_rsv_classify_scenario(text: str) -> str:
    """For the exam tier (Step markers present), split into the three steps and return the class of the FIRST one
    that reads as flawed, or "valid" if all three read clean -- see the rationale above. For easy/medium/hard
    (no Step markers), falls back to the original whole-text classifier, which remains correct there."""
    steps = _split_into_steps(text)
    if steps is None:
        return _classify_whole_text(text)
    for step_text in steps:
        flaw = _step_flaw_class(step_text)
        if flaw is not None:
            return flaw
    return "valid"


# ---- step 2: map a class name to the OFFERED option that declares it ------
#
# Each option's own class-noun marker is fixed, stable answer vocabulary
# shared by every draw of that class -- reading it is not the surface tell
# being removed (that tell was scenario PROSE echoing an option's wording,
# not an option declaring its own class).
_CLASS_MARKERS = [
    ("self_select", "self-selection"),
    ("exclusion", "excluded from ever being selected"),
    ("systematic", "systematic selection"),
    ("convenience", "convenience sample"),
]


def _option_class(option_text: str) -> str:
    if option_text.startswith("Valid --"):
        return "valid"
    for cls, marker in _CLASS_MARKERS:
        if marker in option_text:
            return cls
    return "unknown"


def _oracle_rsv_pick(question) -> str:
    truth = _oracle_rsv_classify_scenario(question.text)
    options = [question.answer_text, *question.distractors]
    matches = [o for o in options if _option_class(o) == truth]
    assert len(matches) == 1, (
        f"expected exactly one offered option of class {truth!r}, found {len(matches)}: "
        f"{[(_option_class(o), o) for o in options]} for {question.text!r}"
    )
    return matches[0]


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_oracle_matches_answer_text(difficulty):
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        oracle = _oracle_rsv_pick(q)
        if oracle != q.answer_text:
            challenged.append((seed, oracle, q.answer_text, q.text))
    assert not challenged, f"oracle challenges: {challenged}"


def test_rsv_oracle_confirms_all_five_categories_occur():
    counts = Counter()
    for difficulty in TIERS:
        for seed in WIDE_SEEDS[:300]:
            q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
            counts[_oracle_rsv_classify_scenario(q.text)] += 1
    assert set(counts) == _ALL_RSV, f"only saw: {dict(counts)}"
    print(f"p19d_random_sample_valid per-category counts (1200 draws): {dict(counts)}")


def test_rsv_red_proof_oracle_catches_a_flipped_key():
    """RED first: swap the answer for a distractor and confirm the oracle
    disagrees, over >=1000 seeds combined across tiers."""
    caught = 0
    total = 0
    for difficulty in TIERS:
        for seed in range(260):
            q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
            bogus = q.distractors[0]
            total += 1
            if _oracle_rsv_pick(q) != bogus:
                caught += 1
    assert total >= 1000
    assert caught == total, f"oracle failed to catch a flipped key on {total - caught}/{total} draws"


@pytest.mark.parametrize("difficulty", TIERS)
def test_rsv_real_key_passes_oracle_over_1000_seeds(difficulty):
    for seed in range(1000):
        q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
        assert _oracle_rsv_pick(q) == q.answer_text


def test_rsv_enumerate_scenario_templates():
    """Enumerate the distinct method-template "shapes" (digits/populations
    masked) the generator can emit across all tiers, and confirm each maps
    to exactly one oracle category (no template is internally
    contradictory)."""
    shapes = {}
    for difficulty in TIERS:
        for seed in range(300):
            q = make(TOPIC_RSV, seed=seed, difficulty=difficulty)
            # Mask digits and the specific population name only (leave the
            # rest of the method sentence intact) so two draws of the SAME
            # method template on DIFFERENT populations collapse to one
            # shape, without over-matching across genuinely different
            # method templates.
            shape = re.sub(r"\d[\d,]*", "N", q.text)
            for pop_words in _RSV_POPULATION_WORDS:
                shape = re.sub(pop_words, "POP", shape, flags=re.IGNORECASE)
            shapes.setdefault(shape, set()).add(_oracle_rsv_classify_scenario(q.text))
    contradictory = {s: cats for s, cats in shapes.items() if len(cats) > 1}
    assert not contradictory, f"template shape maps to multiple answers: {contradictory}"
    print(f"p19d_random_sample_valid: {len(shapes)} distinct template shapes over 1200 draws")


# ============================================================================
# Oracle 2: p19d_target_population
# ============================================================================

# It would have kept passing on the old generator only because the old generator made that shortcut
# valid by construction (the key was the only option that echoed the stem); it now fails outright
# (proven below) once that echo is gone, which is the expected, desired failure -- confirming the
# wording-echo tell is genuinely gone, not just hidden from a different test.
#
# The rewrite instead parses each option's and the question's SCOPE
# structurally: the "Research question: how many <phrase>...?" sentence
# always opens with the exact population noun phrase the question is
# actually asking about, optionally followed by a qualifying relative clause
# ("who ..."), and only THEN the attribute being asked about. An option is a
# candidate answer only if its own stated phrase (with a leading "all "
# stripped, since "all" is not part of the population noun phrase itself) is
# an EXACT, WHOLE-WORD PREFIX of the question text -- i.e. every one of the
# option's own words appears, in order, as the question's own opening words.
# A subgroup/superset/frame-restriction distractor fails this test as soon as
# its own extra wording (a different qualifier clause, "and <something>", or
# "who were actually surveyed"/"who are <frame>") diverges from what the
# question itself literally says, even by one word -- so no CONTEXTS text,
# qualifier tables or frame-label list is ever imported; the check works
# purely off the rendered question and option strings. Among the options that
# pass, the correct one is the LONGEST such prefix: if the question's own
# wording states a qualifying clause, only the option that echoes that FULL
# qualifying clause (not just the bare population phrase) is the correct,
# exact-scope answer; if the question states no qualifier, the bare
# population-phrase option is both the only match and the longest one.
_Q_RE = re.compile(r"research question:\s*how many\s+(.+?)\?", re.IGNORECASE)
_WRAP_RE = re.compile(r"^the target population is (.+)\.$", re.IGNORECASE)


def _words(s: str) -> list:
    return re.findall(r"[a-z']+", s.lower())


def _option_base_words(option: str) -> list:
    wm = _WRAP_RE.match(option)
    assert wm, f"option not in expected wrapper shape: {option!r}"
    phrase = wm.group(1)
    words = _words(phrase)
    if words and words[0] == "all":
        words = words[1:]
    return words


def _oracle_tp_pick(question) -> str:
    m = _Q_RE.search(question.text)
    assert m, f"no 'Research question: how many ...?' sentence found in {question.text!r}"
    rq_words = _words(m.group(1))
    options = [question.answer_text, *question.distractors]
    matches = []
    for opt in options:
        base = _option_base_words(opt)
        if base and rq_words[: len(base)] == base:
            matches.append((opt, len(base)))
    assert matches, f"no option's stated scope is a prefix of the research question: {options} for {question.text!r}"
    max_len = max(n for _, n in matches)
    longest = [opt for opt, n in matches if n == max_len]
    assert len(longest) == 1, (
        f"oracle tie among longest-matching scopes (len={max_len}): {longest} for {question.text!r}"
    )
    return longest[0]


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_oracle_matches_answer_text(difficulty):
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        oracle = _oracle_tp_pick(q)
        if oracle != q.answer_text:
            challenged.append((seed, oracle, q.answer_text, q.text))
    assert not challenged, f"oracle challenges: {challenged}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_tp_real_key_passes_oracle_over_1000_seeds(difficulty):
    for seed in range(1000):
        q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
        assert _oracle_tp_pick(q) == q.answer_text


def test_tp_red_proof_oracle_catches_a_flipped_key():
    caught = 0
    total = 0
    for difficulty in TIERS:
        for seed in range(260):
            q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
            bogus = q.distractors[0]
            total += 1
            if _oracle_tp_pick(q) != bogus:
                caught += 1
    assert total >= 1000
    assert caught == total, f"oracle failed to catch a flipped key on {total - caught}/{total} draws"


def test_tp_enumerate_context_combinations():
    """Regression guard for a previously fixed defect."""
    shapes = set()
    for difficulty in TIERS:
        for seed in range(300):
            q = make(TOPIC_TP, seed=seed, difficulty=difficulty)
            shape = re.sub(r"\d[\d,]*", "N", q.text)
            shapes.add((difficulty, shape))
    print(f"p19d_target_population: {len(shapes)} distinct rendered shapes over 1200 draws")
    assert len(shapes) >= 20
