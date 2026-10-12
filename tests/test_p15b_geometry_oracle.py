"""Independent oracle for the `p15b_geometry` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text`/`question.answer_text`/`question.distractors`
(via `mathgen.grading.mcq_options` for the mcq skill) -- never anything from
`mathgen.generators.p15b_geometry_build` -- and re-derives each answer from
scratch: the induced map from the topological degree read out of the text
(single exponent, or the product of two composed exponents, or the
contractible-interval rule), and the Betti list from a fresh boundary-matrix
rank computation parsed out of the matrix tiers (and a small fixed lookup
table for the two prose tiers, keyed only by which FIXED complex the text
names/lists -- never by trusting the generator's own kind label).
"""

import re

import pytest
import sympy

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from tests.exd_x_r2h_support import legacy_exam_sources
from mathgen import make

FGM_ID = "ucv1-fundamental-group-map"
HOM_ID = "ucv1-simplex-homology"
SEEDS = range(1000)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return legacy_exam_sources(skill.id, {src.difficulty: src for src in skill.sources})


FGM_SOURCES = _skill_sources(FGM_ID)
HOM_SOURCES = _skill_sources(HOM_ID)


# ==========================================================================
# ucv1-fundamental-group-map
# ==========================================================================


def _asks_about_h(question_sentence: str) -> bool:
    """Whether the trailing question sentence asks about h = g o f (the COMPOSITE map) rather than bare f. This
    function is the fix: it reads which map name the question sentence itself uses, so the degree computed
    below is tied to what was asked, not to whatever exponents happen to appear in the text. Matches a
    standalone "h" token flanked by non-letters (not `\b`, which treats `_` as a word character and so misses
    "h_*" -- one of the actual shipped phrasings); a naive substring check also false-positives on ordinary
    English words that merely contain the letter h ("Which", "the"), defeating the check the other way."""
    return re.search(r"(?<![A-Za-z])h(?![A-Za-z])", question_sentence) is not None


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_the_induced_map(difficulty, seed):
    src = FGM_SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    if "interval" in q.text:
        derived = "the zero map"
    else:
        exponents = [int(x) for x in re.findall(r"z\^\{(-?\d+)\}", q.text)]
        assert exponents, q.text
        if len(exponents) == 2:
            # A composition is on the page (f AND g both defined) -- but the
            # question sentence itself decides whether the ASKED-for map is
            # the composite h = g o f (degree = a*b) or bare f alone
            # (degree = a, the first exponent only). Never assume it is
            # whichever quantity happens to make the generator's key line up.
            question_sentence = q.text.rsplit(". ", 1)[-1]
            if _asks_about_h(question_sentence):
                degree = exponents[0] * exponents[1]
            else:
                degree = exponents[0]
        else:
            degree = exponents[0]
        derived = "the zero map" if degree == 0 else f"multiplication by {degree}"

    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )
    options = mcq_options(q)
    assert len(options) == 4
    assert q.answer_text in options


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_exam_tier_question_names_h_not_bare_f(seed):
    """The exam tier's question sentence must name h/g o f, never ask about
    bare f alone while a composition is on the page -- re-derived from the
    rendered text's own trailing sentence, independent of the generator."""
    q = make("p15b_fundamental_group_map", seed=seed, difficulty="exam")
    question_sentence = q.text.rsplit(". ", 1)[-1]
    assert _asks_about_h(question_sentence), (seed, q.text)


# ==========================================================================
# ucv1-simplex-homology
# ==========================================================================

_FIXED_BETTI = {
    "filled_triangle": "1, 0, 0",
    "boundary_triangle": "1, 1",
    "single_edge": "1, 0",
}


def _count_edges_and_faces_from_simplex_list(text):
    edges_m = re.search(r"edges \[([^\]]*)\]", text)
    n_edges = len(edges_m.group(1).split(";")) if edges_m and edges_m.group(1).strip() else 0
    has_face = "one 2-dimensional face" in text
    return n_edges, has_face


def _kind_from_edges_faces(n_vertices, n_edges, has_face):
    if n_vertices == 2:
        return "single_edge"
    if has_face:
        return "filled_triangle"
    return "boundary_triangle"


def _oracle_from_matrix(text):
    """Rebuild d1 (and d2, if present) from the parsed matrix text and
    recompute the Betti numbers by rank, entirely independently of the
    generator's own kind label or cached Betti table."""
    d1_m = re.search(r"d1 \(rows ([^;]+); columns ([^)]+)\) = \[([^\]]*)\]", text)
    assert d1_m, text
    v_labels = [v.strip() for v in d1_m.group(1).split(",")]
    e_labels = [e.strip() for e in d1_m.group(2).split(",")]
    n0, n1 = len(v_labels), len(e_labels)
    rows = d1_m.group(3).split(";")
    d1 = sympy.zeros(n0, n1)
    for i, row in enumerate(rows):
        for j, val in enumerate(row.split(",")):
            d1[i, j] = int(val.strip())

    d2_m = re.search(r"d2 \(rows [^;]+; columns ([^)]+)\) = \[([^\]]*)\]", text)
    if d2_m:
        f_labels = [f.strip() for f in d2_m.group(1).split(",")]
        n2 = len(f_labels)
        rows2 = d2_m.group(2).split(";")
        d2 = sympy.zeros(n1, n2)
        for i, row in enumerate(rows2):
            for j, val in enumerate(row.split(",")):
                d2[i, j] = int(val.strip())
    else:
        n2 = 0
        d2 = sympy.zeros(n1, 0)

    r1 = d1.rank() if n1 else 0
    r2 = d2.rank() if n2 else 0
    b0 = n0 - r1
    b1 = (n1 - r1) - r2
    if n2:
        return f"{b0}, {b1}, {n2 - r2}"
    return f"{b0}, {b1}"


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_the_betti_list(difficulty, seed):
    src = HOM_SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    if "boundary matrices" in q.text:
        derived = _oracle_from_matrix(q.text)
    elif "X has vertices" in q.text:
        v_m = re.search(r"X has vertices \(([^)]*)\)", q.text)
        n_vertices = len(v_m.group(1).split(","))
        n_edges, has_face = _count_edges_and_faces_from_simplex_list(q.text)
        kind = _kind_from_edges_faces(n_vertices, n_edges, has_face)
        derived = _FIXED_BETTI[kind]
    else:
        if "filled triangle" in q.text:
            kind = "filled_triangle"
        elif "boundary of a triangle" in q.text:
            kind = "boundary_triangle"
        elif "single edge" in q.text:
            kind = "single_edge"
        else:
            raise AssertionError(q.text)
        derived = _FIXED_BETTI[kind]

    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds x 2 skills).")
