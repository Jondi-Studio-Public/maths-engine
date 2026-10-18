"""Per active, testable skill with >= 2 sources: does the hard (last) tier
add a prompt shape no lower tier has, or is it range-only -- same magnitude,
different number?

`classify_skill()` below is the one classification function. The tier-escalation tests import it
rather than copy it (`tests/test_pv1d_parabola_build.py` and `tests/test_pv1f_{ftc,cubic,trig,
function_graph}_build.py`, each asserting `verdict == "escalates" and adds == [False, True, True,
True]`), so a gate and this measurement cannot quietly disagree about what "escalates" means. Any
further gate should import it too.

Usage:
    python scripts/measure_tiers.py [--seeds N] [--ids ID [ID ...]]

Exit 0 always: this is a measurement tool, not a gate.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mathgen
from mathgen import skills as sm
from mathgen.text_quality import prompt_shape

# This script still runs (not one topic at a time), so its per-skill cost multiplies by the skill
# count in a way a single topic's fuzz test never pays.
DEFAULT_SEEDS = 60


def tuplify_params(params: dict) -> dict:
    """mathgen generators expect e.g. `kinds=(...)`; YAML/JSON only has lists."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def _testable(skill) -> bool:
    """Active, has sources, and not rubric-assessed (rubric skills have no deterministic grader)."""
    return skill.status == "active" and bool(skill.sources) and skill.answer != "rubric"


def source_shapes(topic: str, difficulty: str | None, params: dict, seeds: int) -> set[str]:
    """Distinct `prompt_shape` outputs for one skill source over `seeds`
    draws. A seed whose params raise (an invalid combination for this
    topic/params -- should not happen for a real skill source, but this is a
    measurement tool, not a gate) is skipped rather than aborting the whole
    skill."""
    shapes: set[str] = set()
    for seed in range(seeds):
        try:
            q = mathgen.make(topic, seed=seed, difficulty=difficulty, **tuplify_params(params))
        except Exception:
            continue
        shapes.add(prompt_shape(q))
    return shapes


def per_tier_shapes(skill, seeds: int = DEFAULT_SEEDS) -> list[set[str]]:
    """`[shape_set, ...]` one per source, in the skill's own YAML declaration
    order -- that order IS the tier ordering. A single-part `exam-single` source is a second exam source
    beside the skill's exam chain, not a further tier, so it is left out."""
    return [
        source_shapes(src.topic, src.difficulty, src.params, seeds) for src in skill.sources if src.key != "exam-single"
    ]


def classify_skill(skill, seeds: int = DEFAULT_SEEDS) -> tuple[str, list[bool]]:
    """`(verdict, adds_shape)` for one skill.

    `verdict` is one of:
      - "single-tier" -- fewer than 2 sources; nothing to compare (the
        note's "106 single-tier" bucket). `adds_shape` is `[]`.
      - "escalates" -- the last tier's shape set contains at least one shape
        absent from the union of every strictly-lower tier.
      - "range-only" -- the last tier adds nothing over that union: same
        reasoning shape, only a bigger number.

    `adds_shape[i]` is True when tier `i` (0 = easiest) adds a shape over the
    union of every tier before it; `adds_shape[0]` is always False (nothing
    is "below" the first tier to compare against).
    """
    tiers = per_tier_shapes(skill, seeds)
    if len(tiers) < 2:
        return "single-tier", []
    adds_shape: list[bool] = []
    union_lower: set[str] = set()
    for shapes in tiers:
        adds_shape.append(bool(shapes - union_lower) if union_lower else False)
        union_lower |= shapes
    verdict = "escalates" if adds_shape[-1] else "range-only"
    return verdict, adds_shape


def last_tier_adds_shape(skill, seeds: int = DEFAULT_SEEDS) -> bool:
    """True when `skill`'s hard (last) tier escalates -- the single boolean
    a tier-escalation gate needs per skill, without re-deriving the whole
    per-tier table itself."""
    verdict, _adds = classify_skill(skill, seeds)
    return verdict == "escalates"


def qualifying_skills(items) -> list:
    """Every active, testable skill (`_testable`) with >= 2
    sources -- a skill with fewer has nothing to compare and is counted in
    the single-tier bucket instead, never printed as range-only."""
    return [s for s in items if _testable(s) and len(s.sources) >= 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ids", nargs="+", default=None, help="restrict to these skill ids")
    args = parser.parse_args()

    _domains, items = sm.load_skills()
    testable = [s for s in items if _testable(s)]
    single_tier = [s for s in testable if len(s.sources) < 2]
    qualifying = qualifying_skills(items)
    if args.ids:
        wanted = set(args.ids)
        qualifying = [s for s in qualifying if s.id in wanted]

    counts: Counter[str] = Counter()
    range_only_ids: list[str] = []
    for skill in sorted(qualifying, key=lambda s: s.id):
        verdict, adds_shape = classify_skill(skill, args.seeds)
        counts[verdict] += 1
        if verdict == "range-only":
            range_only_ids.append(skill.id)
        per_tier = " ".join(f"{i}:{'+' if a else '='}" for i, a in enumerate(adds_shape))
        print(f"{skill.id}\t{skill.domain}\t{len(skill.sources)}\t{verdict}\t{per_tier}")

    if not args.ids:
        counts["single-tier"] += len(single_tier)

    print(
        f"# escalates={counts['escalates']} range-only={counts['range-only']} single-tier={counts['single-tier']}",
        file=sys.stderr,
    )
    if range_only_ids:
        print("# range-only: " + ", ".join(range_only_ids), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
