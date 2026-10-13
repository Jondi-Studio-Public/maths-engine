"""The shared skill contract: the checks every per-batch `_build` file repeats.

One function per family, each `check_<family>(skill_id)` raising AssertionError
with a located message. `tests/test_skill_contract_part_*.py` run them one pytest case per skill and family;
`scripts/skill_contract_baseline.py` runs them over the whole library to record the skills that fail
today (`tests/skill_contract_baseline.yaml`); `tests/test_skill_contract_mutants.py` shows that each
one fails on a deliberately broken skill. Read-only.

Families:

- shape         id/path/domain agree, prerequisites exist, the answer kind has a grader,
                no duplicate source
- determinism   (topic, seed, difficulty, params) -> byte-identical question
- distractors   never equal the answer, distinct, non-empty, not blank
- options       a number answer and its distractors share a shape; no text defects
 (build topics)
- variety       enough distinct prompts per source over the seed sweep
- answers       enough distinct answers per source over the seed sweep (a tier whose answer
                is nearly always the same can be passed by guessing it)
- option_count  a multiple-choice tier shows the same number of options every time, at least 4
- middle_answer when every option is a number, the answer is not almost always an inner one
- probability   a probability's wrong options are probabilities too (none above 1 or below 0)

The last four are the families a test runs over the whole library
(`tests/test_skill_option_quality.py`), against recorded backlogs that may only shrink.
- escalation    sources at different difficulties are different question pools
- grader        the real grader accepts the canonical answer and rejects every
                distractor and a synthetic wrong answer, plus figure plausibility
"""

from __future__ import annotations

if __name__ == "__main__":
    # The imports below need the repository root on sys.path, which running this file does not give.
    raise SystemExit("skill_contract.py is a library for tests and other scripts, not a command; import its checks")

import os
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from mathgen.grading.typed import GRADERS
from mathgen.skills import Skill, load_skills
from mathgen import make
from mathgen.core import REGISTRY
from mathgen.text_quality import find_text_defects
from scripts.regrade_sweep import _difficulties_for, _tuplify, sweep_skill

SEEDS = int(os.environ.get("MATHS_CONTRACT_SEEDS", "60"))
MIN_DISTINCT = 20
# Distinct answers a tier must give over the sweep. Low on purpose: an answer space can be small by design
# (number bonds to 5, classify an angle), and the point is to catch a tier a learner passes by typing the
# same answer. Measured 2026-10-10 over 2216 tiers: 297 give 5 or fewer; at 20 it would be 666.
MIN_DISTINCT_ANSWERS = 6
# A typed exam tier is held higher: it is the tier that stands for an exam question, and with a dozen answers a
# guess no longer passes it. A multiple-choice tier keeps `MIN_DISTINCT_ANSWERS`, because its options already
# make a guess a one-in-four.
MIN_DISTINCT_EXAM_ANSWERS = 12


@lru_cache(maxsize=1)
def library() -> tuple[dict[str, Skill], frozenset[str]]:
    """(live skills by id, every skill id). `load_skills` only: `validate` walks the whole tree (~55 s) and its
    checks are library-wide, owned."""
    _domains, skills = load_skills()
    live = {s.id: s for s in skills if s.status == "active" and s.sources and s.answer in GRADERS}
    return live, frozenset(s.id for s in skills)


def live_ids() -> list[str]:
    return sorted(library()[0])


def skill(skill_id: str) -> Skill:
    return library()[0][skill_id]


@lru_cache(maxsize=256)
def draws(skill_id: str):
    """[(source, difficulty, [Question per seed])] for every source and tier the
    sweep covers. Cached per skill so the families share one generation pass."""
    out = []
    for src in skill(skill_id).sources:
        params = _tuplify(src.params)
        for difficulty in _difficulties_for(src.topic, src.difficulty):
            out.append(
                (src, difficulty, [make(src.topic, seed=s, difficulty=difficulty, **params) for s in range(SEEDS)])
            )
    return out


def check_shape(skill_id: str) -> None:
    sk = skill(skill_id)
    assert sk.id == skill_id and sk.path is not None
    assert sk.path.stem == skill_id and sk.path.parent.name == sk.domain, "id/domain disagree with the path"
    assert not set(sk.prerequisites) - library()[1], "a prerequisite is not a skill"
    assert skill_id not in sk.prerequisites, "skill is its own prerequisite"
    assert sk.answer in GRADERS
    seen = set()
    for src in sk.sources:
        key = (src.topic, src.difficulty, repr(sorted(src.params.items())))
        assert key not in seen, f"duplicate source {key}"
        seen.add(key)


def check_determinism(skill_id: str) -> None:
    for src, difficulty, qs in draws(skill_id):
        params = _tuplify(src.params)
        for seed, q in enumerate(qs):
            again = make(src.topic, seed=seed, difficulty=difficulty, **params)
            assert q.to_json() == again.to_json(), f"{src.topic}/{difficulty} seed={seed} not deterministic"


def check_distractors(skill_id: str) -> None:
    for src, difficulty, qs in draws(skill_id):
        for seed, q in enumerate(qs):
            # A multi-part question carries its distractors on each part, not on the whole.
            for unit, where in _answer_units(q, f"{src.topic}/{difficulty} seed={seed}"):
                assert unit.answer_text not in unit.distractors, f"{where}: answer among distractors"
                assert len(unit.distractors) == len(set(unit.distractors)), f"{where}: duplicate distractors"
                assert unit.distractors, f"{where}: no distractors"
                assert all(str(d).strip() for d in unit.distractors), f"{where}: blank distractor"


def _answer_units(q, where: str):
    """(answer-bearing object, location) pairs: each part of a multi-part question, else the question."""
    parts = getattr(q, "parts", None)
    if parts:
        return [(p, f"{where} part ({p.label})") for p in parts]
    return [(q, where)]


def _places(text: str):
    head, dot, tail = text.partition(".")
    return len(tail) if dot and tail.isdigit() else None


def _build_topic(topic: str) -> bool:
    """The newer generators (`*_build.py`), the ones held to the learner-text bar in
    `text_quality.py`. Legacy topics keep their notation (python-style typed answers, `**` in the
    plain text; a front end renders `latex`)."""
    return REGISTRY[topic][1].__module__.rsplit(".", 1)[-1].endswith("_build")


def check_options(skill_id: str) -> None:
    number = skill(skill_id).answer == "number"
    for src, difficulty, qs in draws(skill_id):
        for seed, q in enumerate(qs):
            where = f"{src.topic}/{difficulty} seed={seed}"
            defects = find_text_defects(q, include_working=True) if _build_topic(src.topic) else []
            assert not defects, f"{where}: {defects}"
            if number:
                for unit, at in _answer_units(q, where):
                    for d in unit.distractors:
                        assert ("." in d) == ("." in unit.answer_text), (
                            f"{at}: answer {unit.answer_text!r} vs distractor {d!r}"
                        )
                        assert _places(d) == _places(unit.answer_text), (
                            f"{at}: answer {unit.answer_text!r} vs {d!r} differ in places"
                        )


POOL_FLOORS_PATH = Path(__file__).resolve().parents[1] / "tests" / "skill_contract_pool_floors.yaml"


@lru_cache(maxsize=1)
def pool_floors() -> dict[str, dict[str, int]]:
    """skill id -> {"topic/difficulty": distinct prompts over the sweep}: tiers whose
    question pool is finite and below 20 by design (number bonds to 5 has six).
    Recorded, not widened: widening changes what the easy tier asks, a scope call."""
    import yaml

    if not POOL_FLOORS_PATH.exists():
        return {}
    return yaml.safe_load(POOL_FLOORS_PATH.read_text(encoding="utf-8")) or {}


def check_variety(skill_id: str) -> None:
    """>= 20 distinct prompts per source and tier over the sweep; a tier listed in the
    pool-floors file must keep at least its recorded count, and must leave the file the
    day it reaches 20 (a stale entry is a failure, so the file only shrinks)."""
    floors = pool_floors().get(skill_id, {})
    need = min(MIN_DISTINCT, SEEDS)
    for src, difficulty, qs in draws(skill_id):
        key = f"{src.topic}/{difficulty}"
        distinct = len({q.text for q in qs})
        if key in floors:
            assert distinct < need, f"{key}: now {distinct} distinct prompts; remove it from {POOL_FLOORS_PATH.name}"
            assert distinct >= floors[key], f"{key}: pool shrank to {distinct}, recorded floor is {floors[key]}"
        else:
            assert distinct >= need, f"{key}: only {distinct} distinct prompts over {SEEDS} seeds"


ANSWER_FLOORS_PATH = Path(__file__).resolve().parents[1] / "tests" / "skill_contract_answer_floors.yaml"


@lru_cache(maxsize=1)
def answer_floors() -> dict[str, dict[str, int]]:
    """skill id -> {"topic/difficulty": distinct answers over the sweep}: the tiers below
    `MIN_DISTINCT_ANSWERS` when the check was added, each the backlog to widen or to rule
    small by design. Written by `scripts/build_contract_floors.py`."""
    import yaml

    if not ANSWER_FLOORS_PATH.exists():
        return {}
    return yaml.safe_load(ANSWER_FLOORS_PATH.read_text(encoding="utf-8")) or {}


def distinct_answers(skill_id: str) -> dict[str, int]:
    """{"topic/difficulty": distinct `answer_text` over the sweep} for every source and tier; two sources that
    differ only in params share a key, which holds the smaller count."""
    out: dict[str, int] = {}
    for src, difficulty, qs in draws(skill_id):
        key = f"{src.topic}/{difficulty}"
        distinct = len({q.answer_text for q in qs})
        out[key] = min(distinct, out.get(key, distinct))
    return out


def answer_minimum(skill_id: str, difficulty: str | None) -> int:
    """The distinct answers a tier must give: `MIN_DISTINCT_EXAM_ANSWERS` for a typed exam tier, else
    `MIN_DISTINCT_ANSWERS`, never more than the sweep has seeds."""
    exam = difficulty == "exam" and skill(skill_id).answer != "mcq"
    return min(MIN_DISTINCT_EXAM_ANSWERS if exam else MIN_DISTINCT_ANSWERS, SEEDS)


def check_answer_variety(skill_id: str) -> None:
    """Each source and tier gives at least `answer_minimum` distinct answers over the sweep. A tier listed in the
    answer-floors file is held at exactly its recorded count: fewer is a regression, and more must be recorded
    (or the entry removed once it reaches the minimum), so the file only ever moves towards empty."""
    floors = answer_floors().get(skill_id, {})
    counts = distinct_answers(skill_id)
    for key, distinct in counts.items():
        need = answer_minimum(skill_id, key.rsplit("/", 1)[1])
        if key in floors:
            assert distinct < need, f"{key}: now {distinct} distinct answers; remove it from {ANSWER_FLOORS_PATH.name}"
            assert distinct >= floors[key], f"{key}: answers shrank to {distinct}, recorded floor is {floors[key]}"
            assert distinct == floors[key], (
                f"{key}: answers rose to {distinct} from {floors[key]}; record it with build_contract_floors.py"
            )
        else:
            assert distinct >= need, f"{key}: only {distinct} distinct answers over {SEEDS} seeds (needs {need})"
    stale = set(floors) - set(counts)
    assert not stale, f"{sorted(stale)}: no longer a tier of this skill; remove from {ANSWER_FLOORS_PATH.name}"


OPTION_BASELINE_PATH = Path(__file__).resolve().parents[1] / "tests" / "skill_contract_option_baseline.yaml"
OPTION_FAMILIES = ("option_count", "answer_position", "filler_option", "probability")
MIN_OPTIONS = 4
# Chance puts the answer at an end of four sorted options half the time; a tier where it is there in fewer
# than one draw in five tells a learner to rule out the highest and the lowest.
MIN_EXTREME_SHARE = 0.2
# Chance puts the answer at any one rank of four a quarter of the time; a tier where one rank holds it in three
# draws of four (always the second smallest, say) tells a learner where to look.
MAX_SAME_RANK_SHARE = 0.75
# An option shown in this share of a tier's draws and never the answer is filler: it narrows four options to three.
FILLER_SHARE = 0.9
# A skill whose answers are all in [0, 1] is held to the probability check when its id or topic says it is one.
# By name, because nothing in a skill says its answer is a probability.
_PROBABILITY_RE = re.compile(r"probab|markov|binomial|poisson|relative.freq|determination|sample.space|_pdf_")
# A number, then an optional unit or percent sign: "1.200 m", "45%", "$12.50", "-3/4", "12,000 kg", "30°".
# The unit holds no digit but a trailing square or cube ("cm^2"), so "2x + 1" is not the number 2 in "x + 1".
_OPTION_RE = re.compile(r"^\s*(\$?)\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?|[-+]?\.\d+)\s*([^\d\s][^\d]*?(?:\^[23])?)?\s*$")


@lru_cache(maxsize=1)
def option_baseline() -> dict[str, dict[str, list[str]]]:
    """family -> skill id -> ["topic/difficulty", ...]: the tiers that failed `option_faults` when its checks
    were added. Written by `scripts/build_contract_floors.py`."""
    import yaml

    if not OPTION_BASELINE_PATH.exists():
        return {}
    return yaml.safe_load(OPTION_BASELINE_PATH.read_text(encoding="utf-8")) or {}


def _option_value(text) -> tuple[Fraction, str] | None:
    """(number, unit) for an option that is a number with an optional unit, currency or percent sign, else None.
    The unit is the text after the number ("m", "%", "cm^2"); a "$" prefix counts as the unit "$"."""
    m = _OPTION_RE.match(str(text).replace("\u2212", "-"))
    if not m:
        return None
    dollar, number, unit = m.groups()
    try:
        value = Fraction(number.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None
    return value, (dollar + (unit or "")).strip()


def option_values(texts) -> list[Fraction] | None:
    """The options' numbers, when every option is a number in one shared unit (so they compare), else None."""
    parsed = [_option_value(t) for t in texts]
    if None in parsed or len({unit for _, unit in parsed}) != 1:
        return None
    return [value for value, _ in parsed]


def option_faults(skill_id: str) -> dict[str, set[str]]:
    """family -> {"topic/difficulty"} for `OPTION_FAMILIES`. The first three hold only for multiple-choice skills,
    the only ones whose options a learner sees; `probability` holds for every skill, because a typed skill's
    distractors are the wrong answers its grader recognises.

    - option_count: the tier does not always show the same number of options, at least `MIN_OPTIONS`.
    - answer_position: in most draws every option is a number in one unit, and the answer sits at a predictable
      rank of the sorted options: at an end in under `MIN_EXTREME_SHARE` of draws, or at one rank in over
      `MAX_SAME_RANK_SHARE`.
    - filler_option: an option shown in over `FILLER_SHARE` of draws is never the answer.
    - probability: a probability offered a wrong option outside [0, 1].
    """
    sk = skill(skill_id)
    faults: dict[str, set[str]] = {family: set() for family in OPTION_FAMILIES}
    for src, difficulty, qs in draws(skill_id):
        key = f"{src.topic}/{difficulty}"
        if sk.answer == "mcq":
            counts = {1 + len(q.distractors) for q in qs}
            if len(counts) > 1 or min(counts) < MIN_OPTIONS:
                faults["option_count"].add(key)
            ranks = []
            for q in qs:
                values = option_values([q.answer_text, *q.distractors])
                if values is not None and len(values) >= 3:
                    ranks.append((sum(v < values[0] for v in values[1:]), len(values)))
            if ranks and len(ranks) >= len(qs) // 2:
                extreme = sum(r == 0 or r == n - 1 for r, n in ranks)
                same = max(ranks.count(r) for r in set(ranks))
                if extreme < MIN_EXTREME_SHARE * len(ranks) or same > MAX_SAME_RANK_SHARE * len(ranks):
                    faults["answer_position"].add(key)
            shown: dict[str, int] = {}
            for q in qs:
                for text in set(map(str, q.distractors)):
                    shown[text] = shown.get(text, 0) + 1
            answers = {str(q.answer_text) for q in qs}
            if any(n > FILLER_SHARE * len(qs) and text not in answers for text, n in shown.items()):
                faults["filler_option"].add(key)
        if _PROBABILITY_RE.search(f"{skill_id} {src.topic}"):
            answers_ = [_option_value(q.answer_text) for q in qs]
            if all(a is not None and a[1] == "" and 0 <= a[0] <= 1 for a in answers_):
                for q in qs:
                    if any((v := _option_value(d)) is not None and not 0 <= v[0] <= 1 for d in q.distractors):
                        faults["probability"].add(key)
                        break
    return faults


def check_options_quality(skill_id: str) -> None:
    """No tier has an option fault that is not in the option baseline, and every tier the baseline lists for
    this skill still has its fault (a fixed tier leaves the file, so it only shrinks)."""
    found = option_faults(skill_id)
    for family, tiers in found.items():
        recorded = set(option_baseline().get(family, {}).get(skill_id, []))
        assert not tiers - recorded, f"{family}: {sorted(tiers - recorded)}"
        assert not recorded - tiers, (
            f"{family}: {sorted(recorded - tiers)} fixed; remove from {OPTION_BASELINE_PATH.name}"
        )


def check_escalation(skill_id: str) -> None:
    """Each tier is its own pool: two tiers drawing the same questions is a label
    change, not an escalation."""
    pools: dict[str, set[str]] = {}
    for _src, difficulty, qs in draws(skill_id):
        pools.setdefault(difficulty or "", set()).update(q.text for q in qs)
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert pools[a] != pools[b], f"tiers {a!r} and {b!r} draw the same questions"


def check_grader(skill_id: str) -> None:
    result = sweep_skill(skill(skill_id), seeds=SEEDS)
    assert result.questions, "the sweep generated nothing"
    assert result.clean, {
        "false_rejects": result.false_rejects[:3],
        "false_accepts": result.false_accepts[:3],
        "exceptions": result.exceptions[:3],
        "figure_failures": result.figure_failures[:3],
    }


FAMILIES = {
    "shape": check_shape,
    "determinism": check_determinism,
    "distractors": check_distractors,
    "options": check_options,
    "variety": check_variety,
    "answers": check_answer_variety,
    "options_quality": check_options_quality,
    "escalation": check_escalation,
    "grader": check_grader,
}
