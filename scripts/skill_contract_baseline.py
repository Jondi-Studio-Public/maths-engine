"""Record which live skills miss which contract family today.

    python scripts/skill_contract_baseline.py [--jobs N] [--ids ID ...] [--out PATH]
    python scripts/skill_contract_baseline.py --prune [--ids ID ...]
    python scripts/skill_contract_baseline.py --check-shrink-only REF

Runs every `scripts/skill_contract.py` family over the library (or the named ids)
in a process pool and writes `tests/skill_contract_baseline.yaml` as
`family: {skill-id: first failure message}`. Read-only apart from that file. The whole
library is ~16 minutes of CPU on a 4-core box.

`--prune` re-runs only the baselined (family, skill) entries (of `--ids` when given) and
drops those that now pass, so a skill-fix PR removes its own entries without a whole-library run.

`--check-shrink-only REF` is a guard for CI or review (it needs git history for REF):
exit 1 if the committed baseline lists any (family, skill) that the baseline at REF
did not. A skill that starts passing is already forced off the list by its strict
xfail, so together the list can only shrink.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "tests" / "skill_contract_baseline.yaml"


def run_one(skill_id: str) -> dict[str, str]:
    from scripts import skill_contract as sc  # heavy imports stay off the CI guard's path

    failures = {}
    for name, fn in sc.FAMILIES.items():
        try:
            fn(skill_id)
        except AssertionError as exc:
            failures[name] = " ".join(str(exc).split())[:300] or "assertion failed"
        except Exception as exc:  # noqa: BLE001 - a generator crash is a failure of the family
            failures[name] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:250]}"
    return failures


def grown(old: dict, new: dict) -> list[str]:
    """Entries in `new` that `old` lacks, as "family: skill"."""
    return sorted(
        f"{fam}: {sid}"
        for fam, skills in (new or {}).items()
        for sid in skills
        if sid not in ((old or {}).get(fam) or {})
    )


FLOORS = REPO_ROOT / "tests" / "skill_contract_pool_floors.yaml"


def weakened_floors(old: dict, new: dict) -> list[str]:
    """Pool-floor entries that are new in `new` or lower than in `old`."""
    out = []
    for sid, tiers in (new or {}).items():
        for key, n in tiers.items():
            was = ((old or {}).get(sid) or {}).get(key)
            if was is None:
                out.append(f"pool floor added: {sid} {key}={n}")
            elif n < was:
                out.append(f"pool floor lowered: {sid} {key} {was} -> {n}")
    return sorted(out)


def check_shrink_only(ref: str) -> int:
    import subprocess

    def at_ref(path: Path):
        show = subprocess.run(
            ["git", "show", f"{ref}:{path.relative_to(REPO_ROOT).as_posix()}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return (yaml.safe_load(show.stdout) or {}) if show.returncode == 0 else None

    old = at_ref(OUT) or {}
    new = yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    added = grown(old, new)
    floors_new = (yaml.safe_load(FLOORS.read_text(encoding="utf-8")) or {}) if FLOORS.exists() else {}
    floors_old = at_ref(FLOORS)  # None: the file does not exist at REF yet, so this change introduces it
    bad_floors = [] if floors_old is None else weakened_floors(floors_old, floors_new)
    if bad_floors:
        print(f"the pool floors may only shrink in scope or rise in value vs {ref}:")
        print("\n".join(f"  {b}" for b in bad_floors))
        return 1
    if added:
        print(
            f"the contract baseline may only shrink; {len(added)} entr{'y' if len(added) == 1 else 'ies'} added vs {ref}:"
        )
        print("\n".join(f"  {a}" for a in added))
        print("Fix the skill instead of baselining it (tests/skill_contract_baseline.yaml).")
        return 1
    print(f"contract baseline did not grow vs {ref}")
    return 0


def prune(args) -> int:
    """Drop baseline entries whose family now passes. Never adds an entry."""
    from scripts import skill_contract as sc

    raw = args.out.read_text(encoding="utf-8")
    head = "".join(line for line in raw.splitlines(True) if line.startswith("#"))
    baseline = yaml.safe_load(raw) or {}
    wanted = set(args.ids) if args.ids else {s for fam in baseline.values() for s in fam}
    todo = sorted(wanted & {s for fam in baseline.values() for s in fam})
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for sid, failures in zip(todo, pool.map(run_one, todo, chunksize=2)):
            results[sid] = failures
    dropped = []
    for fam, skills in baseline.items():
        for sid in list(skills):
            if sid in results and fam not in results[sid]:
                del skills[sid]
                dropped.append(f"{fam}: {sid}")
    baseline = {f: v for f, v in baseline.items() if v}
    args.out.write_text(
        head + yaml.safe_dump(baseline, sort_keys=True, allow_unicode=True, width=200), encoding="utf-8"
    )
    print(f"dropped {len(dropped)}:", *dropped, sep="\n  ")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-shrink-only", metavar="REF")
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args(argv)
    if args.check_shrink_only:
        return check_shrink_only(args.check_shrink_only)
    if args.prune:
        return prune(args)
    from scripts import skill_contract as sc

    ids = args.ids or sc.live_ids()
    baseline: dict[str, dict[str, str]] = {name: {} for name in sc.FAMILIES}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for sid, failures in zip(ids, pool.map(run_one, ids, chunksize=4)):
            for name, msg in failures.items():
                baseline[name][sid] = msg
    baseline = {n: dict(sorted(v.items())) for n, v in baseline.items() if v}
    args.out.write_text(
        "# Generated by scripts/skill_contract_baseline.py -- do not hand-edit.\n"
        "# family: {skill-id: first failure}. Strict xfail in tests/test_skill_contract.py.\n"
        + yaml.safe_dump(baseline, sort_keys=True, allow_unicode=True, width=200),
        encoding="utf-8",
    )
    print({n: len(v) for n, v in baseline.items()}, f"of {len(ids)} skills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
