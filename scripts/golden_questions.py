"""Record, and check, a golden corpus of question ids so an unintended change to a question is caught.

A question is named by (skill, source key, seed) and pinned by a hash of its content (see
`mathgen.engine.question_id`). This script walks every source of every active skill in the engine library for
seeds 0, 1 and 2 and writes one JSON line per question to `tests/golden/questions.txt` (`.txt` because CI's secret
scan refuses file types it does not read, `.jsonl` among them):

    {"answer": "6*sqrt(21)", "id": "simplify-surds/easy/0/3f9a0c1d2e4b", "text": "Simplify √756. Give ..."}

`answer` and `text` (the first 80 characters of the prompt) are there only so that a diff of an intentional
change is readable; the id's hash is what is compared. A generator that raises for a seed is recorded with
`raises-<ExceptionClass>` in place of the hash, so a change in that is caught too.

    python scripts/golden_questions.py --write   # regenerate the file after an intended change
    python scripts/golden_questions.py --check   # exit 1, listing the changed ids, if anything differs

Changing a generator, a source's params, difficulty or key changes the hashes of the questions it touches. A
pull request that does so regenerates the file with `--write` and says so in its description. The file is
generated: never edit it by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen import engine  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "questions.txt"
SEEDS = (0, 1, 2)
ANSWER_CHARS = 60
TEXT_CHARS = 80


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def golden_entry(skill, source, seed: int) -> dict:
    """The record for one (skill, source, seed): its id, a short answer and the start of the prompt."""
    key = engine._key_of(skill, source)
    try:
        question, qid = engine.make_with_id(skill, seed, source=source)
    except Exception as exc:  # noqa: BLE001 - a generator that raises is part of what is recorded
        return {"id": f"{skill.id}/{key}/{seed}/raises-{type(exc).__name__}"}
    return {"id": qid, "answer": _short(question.answer_text, ANSWER_CHARS), "text": _short(question.text, TEXT_CHARS)}


def skills_to_record(skills=None) -> list:
    """The active skills of the engine library (or of `skills`), in id order."""
    if skills is None:
        skills = engine.load_skills()[1]
    return sorted((s for s in skills if s.status == "active"), key=lambda s: s.id)


def stems_for(skill) -> list[str]:
    """`<skill>/<key>/<seed>` for every question recorded for `skill`, without generating anything."""
    return [f"{skill.id}/{engine._key_of(skill, src)}/{seed}" for src in skill.sources for seed in SEEDS]


def entries_for(skill) -> list[dict]:
    return [golden_entry(skill, src, seed) for src in skill.sources for seed in SEEDS]


def render(entries: list[dict]) -> str:
    lines = (json.dumps(e, sort_keys=True, ensure_ascii=False) for e in sorted(entries, key=lambda e: e["id"]))
    return "".join(line + "\n" for line in lines)


def current_entries() -> list[dict]:
    return [e for skill in skills_to_record() for e in entries_for(skill)]


def read_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def differences(recorded: list[dict], current: list[dict]) -> list[str]:
    """Readable lines naming each id that was added, removed or changed, comparing by (skill/key/seed) stem."""

    def stem(entry):
        return entry["id"].rsplit("/", 1)[0]

    old = {stem(e): e for e in recorded}
    new = {stem(e): e for e in current}
    out = []
    for s in sorted(old.keys() | new.keys()):
        a, b = old.get(s), new.get(s)
        if a == b:
            continue
        if a is None:
            out.append(f"added    {b['id']}")
        elif b is None:
            out.append(f"removed  {a['id']}")
        elif a["id"] != b["id"]:
            out.append(
                f"changed  {a['id']} -> {b['id'].rsplit('/', 1)[1]}\n           {a.get('text', '')!r}\n           {b.get('text', '')!r}"
            )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate tests/golden/questions.txt")
    mode.add_argument("--check", action="store_true", help="exit 1 if any question differs from the file")
    args = parser.parse_args(argv)

    current = current_entries()
    if args.write:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(render(current), encoding="utf-8", newline="\n")
        print(f"wrote {len(current)} questions to {GOLDEN_PATH.relative_to(REPO_ROOT)}")
        return 0
    diffs = differences(read_golden(), current)
    if diffs:
        print(f"{len(diffs)} question(s) differ from {GOLDEN_PATH.relative_to(REPO_ROOT)}:")
        print("\n".join(diffs))
        print("If this change is intended, run `python scripts/golden_questions.py --write` and say so in the PR.")
        return 1
    print(f"ok: {len(current)} questions match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
