"""Independent oracle for the `p19b_response_variable` questions: it re-derives each answer from the rendered
question text.

Imports NOTHING from any p19b module and reads only what a learner is given (question.text, answer_text,
distractors, the grader's option list).

Route: from the rendered stimulus alone, (1) pull out the three recorded variables, (2) find
the one direction clause and read from its wording which variable is explanatory (set by the
researcher / the aim's given quantity / on the horizontal axis) and which is the response
(measured / predicted / on the vertical axis), (3) take the third variable as the extra, then
re-derive the key and check that exactly one of the four options is that pair and that every
distractor is one of the five coherent wrong assignments. It also enumerates every
(triple, direction-clause) combination seen over the sweep and asserts each is coherent.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p19b_response_variable"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

_STEM = "Which option correctly names the explanatory variable and the response variable for the association "
QUESTIONS = {
    "easy": _STEM + "between the two variables the study is about?",
    "medium": _STEM + "between the two variables the study is about?",
    "hard": _STEM + "that the line of best fit describes?",
    "exam": _STEM + "shown by the two axes of the scatterplot?",
}
# The stimulus must never label the third variable irrelevant, excluded or a mere label.
Z_LABEL_PHRASES = (
    "plays no part",
    "only a",
    "excludes",
    "covariate",
    "background",
    "grouping label",
    "not part",
    "does not involve",
    "leaves out",
    "only groups",
    "serves only",
    "excluded",
)
OPT_RE = re.compile(r"^Explanatory variable: (.+); response variable: (.+)$")

LIST_RES = (
    re.compile(r"^A study recorded ([^.,]+), ([^.,]+) and ([^.,]+) for each ([a-z ]+)\."),
    re.compile(r"^In a study, three quantities were recorded for each ([a-z ]+): ([^.,]+), ([^.,]+) and ([^.,]+)\."),
    re.compile(r"^The data set holds ([^.,]+), ([^.,]+) and ([^.,]+) for every ([a-z ]+) in the study\."),
)

# (clause pattern with {N} for a variable name, index (0 or 1) of the N that is EXPLANATORY).
# Reading each rule off the wording: the quantity the researcher sets / the influencing quantity in
# the aim / the given quantity of a line of best fit / the horizontal-axis quantity is explanatory;
# the measured / influenced / predicted / vertical one is the response.
RULES = (
    ("sets the level of {N} and measures {N}", 0),
    ("chooses the level of {N} for each [a-z ]+, then measures {N}", 0),
    ("varies {N} on purpose and observes {N}", 0),
    ("measures {N} after setting the level of {N}", 1),
    ("measures {N} for each [a-z ]+, having chosen the level of {N} in advance", 1),
    ("observes {N} while deliberately varying {N}", 1),
    ("researchers want to find out whether {N} is influenced by {N}", 1),
    ("to see whether {N} is affected by {N}", 1),
    ("the aim is to test whether {N} is influenced by {N}", 1),
    ("researchers describe how {N} changes according to {N}", 1),
    ("researchers want to find out whether {N} influences {N}", 0),
    ("to see whether {N} affects {N}", 0),
    ("the aim is to test whether {N} has an influence on {N}", 0),
    ("line of best fit is used to predict {N} from {N}", 1),
    ("line of best fit is fitted so that {N} can be estimated for a given value of {N}", 1),
    ("line of best fit gives {N} once the value of {N} is known", 1),
    ("line of best fit uses {N} to predict {N}", 0),
    ("line of best fit is fitted so that, given a value of {N}, {N} can be estimated", 0),
    ("line of best fit uses known values of {N} to estimate {N}", 0),
    ("puts {N} on the horizontal axis and {N} on the vertical axis", 0),
    ("puts {N} on the vertical axis and {N} on the horizontal axis", 1),
    ("has {N} along its horizontal axis and {N} along its vertical axis", 0),
    ("has {N} along its vertical axis and {N} along its horizontal axis", 1),
    ("The horizontal axis of the scatterplot shows {N}, and its vertical axis shows {N}", 0),
    ("The vertical axis of the scatterplot shows {N}, and its horizontal axis shows {N}", 1),
)


def _stimulus(text, tier):
    assert text.endswith(" " + QUESTIONS[tier]), text
    return text[: -len(" " + QUESTIONS[tier])]


def _three_names(stim):
    """The recorded-list sentence may come first, last or between other sentences."""
    for sentence in re.split(r"(?<=\.)\s+", stim):
        for i, rx in enumerate(LIST_RES):
            m = rx.match(sentence)
            if m:
                g = m.groups()
                names = g[:3] if i != 1 else g[1:]
                return list(names), i
    raise AssertionError(f"no recorded-list sentence in {stim!r}")


def derive(text, tier):
    """(E, R, Z, rule index, list-template index) read from the stimulus alone."""
    stim = _stimulus(text, tier)
    names, list_idx = _three_names(stim)
    assert len(set(names)) == 3, names
    alt = "(" + "|".join(re.escape(n) for n in names) + ")"
    matches = []
    for idx, (pattern, e_index) in enumerate(RULES):
        rx = re.compile(pattern.replace("{N}", alt))
        m = rx.search(stim)
        if m:
            matches.append((idx, m.groups(), e_index))
    assert len(matches) == 1, (matches, stim)
    idx, groups, e_index = matches[0]
    assert len(groups) == 2 and groups[0] != groups[1], (groups, stim)
    e, r = groups[e_index], groups[1 - e_index]
    (z,) = [n for n in names if n not in (e, r)]
    return e, r, z, idx, list_idx


def _questions():
    for tier in TIERS:
        for seed in range(N_SEEDS):
            yield tier, seed, make(TOPIC, seed=seed, difficulty=tier, tier=tier)


def test_key_rederived_from_stimulus_and_options_over_sweep():
    checked = 0
    for tier, seed, q in _questions():
        e, r, z, _idx, _lst = derive(q.text, tier)
        key = f"Explanatory variable: {e}; response variable: {r}"
        assert q.answer_text == key, (tier, seed, q.text, q.answer_text, key)
        options = [q.answer_text, *q.distractors]
        assert len(options) == 4 and len(set(options)) == 4, (tier, seed, options)
        assert sum(o == key for o in options) == 1
        assert sorted(mcq_options(q)) == sorted(options)
        allowed = {(r, e), (e, z), (z, r), (z, e), (r, z)}
        for d in q.distractors:
            m = OPT_RE.match(d)
            assert m and (m.group(1), m.group(2)) in allowed, (tier, seed, d)
        assert len({OPT_RE.match(d).groups() for d in q.distractors}) == 3
        stim = _stimulus(q.text, tier)
        lowered = stim.lower()
        assert not any(p in lowered for p in Z_LABEL_PHRASES), (tier, seed, stim)
        # the pair is identified by the aim / analysis sentence alone; the question names the same analysis
        if tier == "hard":
            assert "line of best fit" in q.text.rsplit("association", 1)[-1]
        if tier == "exam":
            assert q.text.rstrip().endswith("shown by the two axes of the scatterplot?")
        checked += 1
    assert checked == len(TIERS) * N_SEEDS


def test_no_giveaway_words_in_stimulus_and_question_present():
    for tier, seed, q in _questions():
        stim = _stimulus(q.text, tier).lower()
        for word in ("explanatory", "response", "dependent", "independent", "depend", "respond"):
            assert word not in stim, (tier, seed, word)
        assert q.text.count(QUESTIONS[tier]) == 1


def test_template_by_slot_matrix_is_coherent():
    """Every (variable triple, direction clause, list sentence) combination seen over the sweep
    is coherent: three distinct variables, the direction clause names exactly the pair (E, R),
    the extra never appears in it, the extra is stated to play no part, and the convention the
    tier relies on holds (exam: the horizontal-axis quantity is the key's explanatory one)."""
    seen = {}
    per_tier_triples = {t: set() for t in TIERS}
    per_tier_rules = {t: set() for t in TIERS}
    for tier, seed, q in _questions():
        e, r, z, idx, lst = derive(q.text, tier)
        seen.setdefault((e, r, z, idx, lst), []).append((tier, seed))
        per_tier_triples[tier].add((e, r, z))
        per_tier_rules[tier].add(idx)
    for (e, r, z, idx, lst), where in seen.items():
        assert len({e, r, z}) == 3
        pattern, e_index = RULES[idx]
        # the clause's own wording fixes which position is explanatory, re-checked directly
        if "horizontal axis" in pattern:
            first_horizontal = pattern.index("horizontal") < pattern.index("vertical")
            assert (e_index == 0) == first_horizontal
        for tier, seed in where:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            stim = _stimulus(q.text, tier)
            clause_sentence = [
                s for s in re.split(r"(?<=\.)\s+", stim) if re.search(RULES[idx][0].replace("{N}", ".+"), s)
            ]
            assert clause_sentence and z not in clause_sentence[0], (tier, seed, stim)
            assert e in clause_sentence[0] and r in clause_sentence[0]
    # the draw space is saturated: each tier shows many scenarios and every one of its rules
    for tier in TIERS:
        assert len(per_tier_triples[tier]) >= 24, (tier, len(per_tier_triples[tier]))
    assert len(per_tier_rules["easy"]) == 6 and len(per_tier_rules["medium"]) == 7
    assert len(per_tier_rules["hard"]) == 6 and len(per_tier_rules["exam"]) == 6
    # no tier reuses another tier's direction wording
    all_rules = [per_tier_rules[t] for t in TIERS]
    assert sum(len(s) for s in all_rules) == len(set().union(*all_rules))


def test_exam_convention_and_manipulation_cue_rederived():
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="exam", tier="exam")
        stim = _stimulus(q.text, "exam")
        e, r, z, _idx, _lst = derive(q.text, "exam")
        # convention re-derived independently of derive(): the name sitting in the clause part
        # that mentions "horizontal" is the explanatory one
        horiz_part = re.search(r"([^.]*horizontal[^.]*)\.", stim).group(1)
        vert_part = re.search(r"([^.]*vertical[^.]*)\.", stim).group(1)
        assert horiz_part == vert_part  # one sentence states both axes
        assert stim.find(e) >= 0 and horiz_part.find(e) >= 0 and horiz_part.find(r) >= 0

        def axis_of(name):
            m = re.search(
                re.escape(name) + r" (?:on the|along its) (horizontal|vertical) axis", horiz_part
            ) or re.search(r"(horizontal|vertical) axis(?: of the scatterplot)? shows " + re.escape(name), horiz_part)
            assert m, (name, horiz_part)
            return m.group(1)

        e_is_horizontal_side = axis_of(e) == "horizontal"
        r_is_vertical_side = axis_of(r) == "vertical"
        assert e_is_horizontal_side and r_is_vertical_side, horiz_part
        assert stim.find("horizontal") >= 0 and stim.find("vertical") >= 0
        assert not re.search(r"influenc|affect|cause|lead|predict(?!ions)|estimate", stim)
        assert "least-squares line" in stim and "predictions" in stim
        assert re.search(r"coloured|labelled", stim)
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        assert not re.search(r"influenc|affect|cause|lead to", _stimulus(q.text, "hard"))
