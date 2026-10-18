"""Regenerate `tests/generator_contract_baseline.yaml` from the current tree.

    python scripts/generator_contract_baseline.py --check [--jobs N] [--topics T ...]
    python scripts/generator_contract_baseline.py --write [--jobs N] [--topics T ...]

The baseline holds the shrink-only lists `tests/test_generator_contract.py` and the exam-part files read. It
goes stale whenever generators merge, so this script measures every list from the tree:

- `unknown_tier`: generators with a `tier` parameter that do not raise ValueError for `tier="not-a-tier"`, with
  what they do instead ("builds a question for an unknown tier" / "raises <Exc>, not ValueError");
- `unwired`: `exd_*`/`exam_*` generators with an exam preset that no skill uses as its exam source;
- `answer_space`: topic -> [distinct answers at least, commonest answer at most] over `ANSWER_SEEDS` exam seeds,
  for the topics below the default bar only, at 80% of the measured distinct count (rounded down) and 125% of
  the commonest count (rounded up). The pins are always set from the current measurement;
- `exam_params_deviations`: exam sources of `exd_*`/`exam_*` generators whose params are not `{"tier": "exam"}`
  (topic -> params), except the ones `expected_exam_params` derives from the skill id.

`--check` exits 1 and summarises the difference when the file is stale; `--write` rewrites it (the leading
comment block is kept as it is). The answer-space measurement builds 300 exam seeds for each of ~900 topics, about
30 CPU-minutes, so it runs in a process pool; `--topics` limits it to the named topics, merging into the existing
answer_space section (every other section is cheap and always measured in full).

`KNOWN_CRASH_SEEDS`, `TEXT_EXEMPT` and `SHAPES_PINNED` stay hand-maintained in
`tests/generator_contract_cases.py` because each entry needs a human reason. Both modes report, but never
write, what they observe against them: a crash seed that is not listed (or a listed one that no longer
crashes), and a text defect in the first `TEXT_SEEDS` questions of a topic that is not exempt.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import math
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen import make  # noqa: E402
from mathgen.core import PRESETS, REGISTRY  # noqa: E402
from mathgen.skills import load_skills  # noqa: E402
from mathgen.text_quality import find_text_defects  # noqa: E402
from tests.generator_contract_cases import (  # noqa: E402
    ANSWER_MAX_SHARE,
    ANSWER_MIN_DISTINCT,
    ANSWER_SEEDS,
    BASELINE_PATH,
    EXAM,
    EXAM_PREFIXES,
    EXAM_TOPICS,
    KNOWN_CRASH_SEEDS,
    TEXT_EXEMPT,
    TEXT_INCLUDES_WORKING,
    TEXT_SEEDS,
    build_exam_questions,
)

PIN_DISTINCT_FRACTION = 0.8
PIN_MOST_FACTOR = 1.25
SECTIONS = ("unknown_tier", "unwired", "answer_space", "exam_params_deviations")
# expected_exam_params in tests/test_generator_contract.py derives this one from the skill id.
DERIVED_PARAMS = {"exd_x_u5_partial_chain"}


def measure_topic(topic: str) -> dict:
    """The answer-space numbers, crash seeds and text defects of one exam topic. Runs in a pool worker."""
    questions, crashes = build_exam_questions(topic)
    answers = Counter(q.answer_text for q in questions)
    defects = []
    if topic not in TEXT_EXEMPT:
        for seed, q in enumerate(questions[:TEXT_SEEDS]):
            defects += [f"seed={seed}: {d}" for d in find_text_defects(q, include_working=TEXT_INCLUDES_WORKING)]
    return {
        "topic": topic,
        "distinct": len(answers),
        "most": max(answers.values()) if answers else 0,
        "crashes": crashes,
        "defects": defects[:3],
    }


def measure_answer_space(topics: list[str], jobs: int) -> list[dict]:
    if jobs <= 1:
        return [measure_topic(t) for t in topics]
    with multiprocessing.Pool(jobs) as pool:
        return list(pool.imap_unordered(measure_topic, topics, chunksize=1))


def pins(rows: list[dict]) -> dict[str, list[int]]:
    out = {}
    for r in rows:
        if r["distinct"] >= ANSWER_MIN_DISTINCT and r["most"] <= ANSWER_MAX_SHARE * ANSWER_SEEDS:
            continue
        out[r["topic"]] = [
            max(1, int(PIN_DISTINCT_FRACTION * r["distinct"] + 1e-9)),
            min(ANSWER_SEEDS, math.ceil(PIN_MOST_FACTOR * r["most"] - 1e-9)),
        ]
    return out


def unknown_tier() -> dict[str, str]:
    out = {}
    for topic, (_level, fn) in sorted(REGISTRY.items()):
        if "tier" not in inspect.signature(fn).parameters:
            continue
        try:
            make(topic, seed=0, tier="not-a-tier")
        except ValueError:
            continue
        except Exception as exc:
            out[topic] = f"raises {type(exc).__name__}, not ValueError"
        else:
            out[topic] = "builds a question for an unknown tier"
    return out


def library() -> tuple[set[str], set[str], list]:
    """(topics used as an exam source, topics used by any source, [(skill, exam source)])."""
    skills = load_skills()[1]
    exam = [(s, src) for s in skills for src in s.sources if src.difficulty == EXAM]
    return {src.topic for _s, src in exam}, {src.topic for s in skills for src in s.sources}, exam


def unwired(exam_wired: set[str], any_wired: set[str]) -> dict[str, str]:
    generators = sorted(t for t in REGISTRY if t.startswith(EXAM_PREFIXES) and EXAM in PRESETS.get(t, {}))
    return {
        t: "used by a skill, but not as its exam source" if t in any_wired else "no skill source uses it at any tier"
        for t in generators
        if t not in exam_wired
    }


def params_deviations(exam: list) -> dict[str, dict]:
    out = {}
    for _skill, src in exam:
        if src.topic.startswith(EXAM_PREFIXES) and src.topic not in DERIVED_PARAMS:
            params = dict(src.params or {})
            if params != {"tier": EXAM}:
                out[src.topic] = params
    return dict(sorted(out.items()))


def header(text: str) -> str:
    """The leading comment block of the existing file, up to its first blank line."""
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines)


def render(head: str, data: dict) -> str:
    parts = [head, ""] if head else []
    for name in SECTIONS:
        entries = data[name]
        parts.append(f"{name}:")
        for key, value in entries.items():
            if isinstance(value, str):
                shown = value
            else:
                shown = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip().removesuffix("\n...")
            parts.append(f"  {key}: {shown}")
        parts.append("")
    return "\n".join(parts)


def observations(rows: list[dict]) -> list[str]:
    notes = []
    for r in sorted(rows, key=lambda r: r["topic"]):
        topic = r["topic"]
        seen = {seed for seed, _msg in r["crashes"]}
        known = set(KNOWN_CRASH_SEEDS.get(topic, ()))
        for seed, msg in r["crashes"]:
            if seed not in known:
                notes.append(f"unlisted crash seed: {topic} seed={seed}: {msg[:100]} (add to KNOWN_CRASH_SEEDS)")
        for seed in sorted(known - seen):
            notes.append(f"stale KNOWN_CRASH_SEEDS entry: {topic} seed={seed} no longer raises")
        if r["defects"]:
            notes.append(f"text defect: {topic}: {'; '.join(r['defects'])} (fix it, or add to TEXT_EXEMPT with a why)")
    return notes


def summarise(old: dict, new: dict, topics: list[str] | None) -> list[str]:
    lines = []
    for name in SECTIONS:
        a, b = old.get(name) or {}, new[name]
        if name == "answer_space" and topics is not None:
            a = {k: v for k, v in a.items() if k in topics}
            b = {k: v for k, v in b.items() if k in topics}
        added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
        changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
        if not (added or removed or changed):
            continue
        lines.append(f"{name}: +{len(added)} -{len(removed)} ~{len(changed)}")
        for label, keys in (("add", added), ("remove", removed)):
            lines += [f"  {label} {k}: {b[k] if label == 'add' else a[k]}" for k in keys[:15]]
            if len(keys) > 15:
                lines.append(f"  ... and {len(keys) - 15} more to {label}")
        lines += [f"  change {k}: {a[k]} -> {b[k]}" for k in changed[:15]]
        if len(changed) > 15:
            lines.append(f"  ... and {len(changed) - 15} more changed")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if the baseline is stale")
    mode.add_argument("--write", action="store_true", help="rewrite the baseline from the tree")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (default: all cores)")
    ap.add_argument("--topics", nargs="+", metavar="TOPIC", help="measure answer_space for these exam topics only")
    args = ap.parse_args(argv)

    if args.topics:
        unknown = [t for t in args.topics if t not in EXAM_TOPICS]
        if unknown:
            ap.error(f"not exam topics: {unknown}")
    topics = sorted(args.topics) if args.topics else None

    text = BASELINE_PATH.read_text(encoding="utf-8") if BASELINE_PATH.exists() else ""
    old = yaml.safe_load(text) or {}
    exam_wired, any_wired, exam = library()
    rows = measure_answer_space(topics or EXAM_TOPICS, args.jobs)
    measured = pins(rows)
    answer_space = measured
    if topics is not None:  # keep the pins of the topics not measured
        answer_space = {k: v for k, v in (old.get("answer_space") or {}).items() if k not in topics} | measured
    new = {
        "unknown_tier": unknown_tier(),
        "unwired": unwired(exam_wired, any_wired),
        "answer_space": dict(sorted(answer_space.items())),
        "exam_params_deviations": params_deviations(exam),
    }
    rendered = render(header(text), new)
    notes = observations(rows)
    diff = summarise(old, new, topics)

    for note in notes:
        print(note)
    if args.write:
        BASELINE_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {BASELINE_PATH.relative_to(REPO_ROOT)}" + (" (changed)" if diff else " (unchanged)"))
        print("\n".join(diff))
        return 0
    if rendered != text and not diff:  # same data, different layout
        diff = ["".join(difflib.unified_diff(text.splitlines(True), rendered.splitlines(True), n=0))[:2000]]
    if diff:
        print(f"{BASELINE_PATH.name} is stale; run scripts/generator_contract_baseline.py --write")
        print("\n".join(diff))
    elif not notes:
        print("baseline is current")
    return 1 if diff or notes else 0


if __name__ == "__main__":
    raise SystemExit(main())
