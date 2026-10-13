"""Build the ready-made LMS question banks that go with an engine release.

For every active, gradeable skill that has sources, this writes one Moodle XML bank and one IMS QTI 2.1 package
(the format Canvas imports), then packs each format into one archive to attach to the GitHub release:

    dist/banks/moodle/<domain>/<skill-id>.xml
    dist/banks/qti/<domain>/<skill-id>.zip
    dist/banks/maths-engine-<version>-moodle-banks.zip
    dist/banks/maths-engine-<version>-qti-banks.zip

A bank holds N questions (default 20) per difficulty the skill offers, one section per difficulty (a Moodle
subcategory, a QTI assessment section). Questions come from a seed derived from the engine version, so a release's
banks are reproducible; `--seed` overrides it. Rubric skills are left out: the engine cannot mark them.

    py scripts/build_banks.py --jobs 8                      # Windows PowerShell
    python scripts/build_banks.py --skills simplify-surds   # a quick try

The output is the same whatever `--jobs` is. A skill whose bank fails to build is reported and skipped, and the
exit status is 1 once the rest is written. See docs/banks.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen import __version__, export, service  # noqa: E402

DEFAULT_PER_DIFFICULTY = 20
TIER_ORDER = ("easy", "medium", "hard", "exam")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
BANK_ROOT = "Maths Engine"

_LIB: service.Library | None = None


def default_seed(version: str = __version__) -> int:
    """The seed a release's banks use unless `--seed` says otherwise."""
    return service.derive_seed("maths-engine-banks", version)


def _library() -> service.Library:
    global _LIB
    if _LIB is None:
        _LIB = service.Library.load()
    return _LIB


def _section_plan(skill) -> list[tuple[str, str | int | None]]:
    """`(section title, source)` per section: one per difficulty in easy-to-exam order, then any unlabelled source
    by its key. A skill with no labels at all is one section of its first source."""
    plan: list[tuple[str, str | int | None]] = []
    labels = {s.difficulty for s in skill.sources if s.difficulty}
    for tier in [t for t in TIER_ORDER if t in labels] + sorted(labels - set(TIER_ORDER)):
        plan.append((tier.capitalize(), tier))
    for src in skill.sources:
        if not src.difficulty and src.effective_key and src.effective_key not in [p[1] for p in plan]:
            plan.append((str(src.effective_key), src.effective_key))
    return plan or [("Questions", None)]


def _path_part(text: str) -> str:
    return text.replace("/", "-").strip()


def build_one(skill_id: str, per_difficulty: int, seed: int) -> dict:
    """Build one skill's bank. Returns a dict with `moodle` (str), `qti` (bytes) and `meta`, or `error`."""
    try:
        lib = _library()
        skill = lib.skill(skill_id)
        plan = _section_plan(skill)
        spec = {
            "title": skill.name,
            "seed": service.derive_seed(seed, skill.id),
            "sections": [
                {"title": title, "items": [{"skill": skill.id, "source": source, "count": per_difficulty}]}
                for title, source in plan
            ],
        }
        paper = service.build_paper(lib, spec)
        paper["category"] = f"$course$/top/{BANK_ROOT}/{_path_part(skill.domain)}/{_path_part(skill.name)}"
        paper["section_categories"] = True
        moodle = export.paper_moodle_xml(paper)
        qti = export.paper_qti21_zip(paper)
        summary = service.skill_summary(skill)
        questions = sum(len(s["questions"]) for s in paper["versions"][0]["sections"])
        meta = {
            "id": skill.id,
            "name": skill.name,
            "domain": skill.domain,
            "years": summary["years"],
            "curriculum_codes": summary["curriculum_codes"],
            "difficulties": [title.lower() for title, _ in plan] if summary["difficulties"] else [],
            "questions": questions,
            "acara_attribution": service.needs_acara_attribution(skill),
        }
        return {"id": skill.id, "moodle": moodle, "qti": qti, "meta": meta}
    except Exception as exc:  # reported at the end, never silently dropped
        return {"id": skill_id, "error": f"{type(exc).__name__}: {exc}"}


def _init_worker() -> None:
    _library()


def select_skills(lib: service.Library, skills: list[str] | None, domain: str | None) -> tuple[list[str], list[str]]:
    """`(ids to build, notes)`. Unknown ids and domains raise `ValueError`."""
    notes: list[str] = []
    if domain and not any(s.domain == domain for s in lib.skills):
        raise ValueError(f"unknown domain {domain!r}")
    hits = {s.id for s in service.find_skills(lib, domain=domain, gradeable_only=True)}
    if skills:
        unknown = [i for i in skills if i not in lib.by_id]
        if unknown:
            raise ValueError("unknown skill id(s): " + ", ".join(unknown))
        for i in skills:
            if i not in hits:
                notes.append(f"left out {i}: not an active, gradeable skill with sources in this selection")
        chosen = {i for i in skills if i in hits}
    else:
        chosen = hits
    return sorted(chosen, key=lambda i: (lib.by_id[i].domain, i)), notes


def _readme(fmt: str, version: str, seed: int, per_difficulty: int, count: int, acara: bool) -> str:
    if fmt == "moodle":
        how = (
            "HOW TO IMPORT INTO MOODLE\n"
            "  1. Unzip this archive. Each skill is one file: moodle/<domain>/<skill-id>.xml.\n"
            "  2. In your course open Question bank > Import, choose the format 'Moodle XML',\n"
            "     and upload one or more of the .xml files.\n"
            "  3. Each bank arrives as a category tree: Maths Engine / <domain> / <skill name> / <difficulty>.\n"
            "  Multiple-choice and numeric questions are marked by Moodle. Questions whose answer is an\n"
            "  expression, pair or inequality are text answers matched exactly (ignoring case): read\n"
            "  those responses yourself or accept the feedback as a guide. The full answer and working\n"
            "  are in each question's general feedback.\n"
        )
        title = "Maths Engine question banks for Moodle (Moodle XML)"
    else:
        how = (
            "HOW TO IMPORT INTO CANVAS (new quizzes and classic quizzes) AND OTHER QTI 2.1 TOOLS\n"
            "  1. Unzip this archive. Each skill is one QTI package: qti/<domain>/<skill-id>.zip.\n"
            "     Do not unzip the inner packages: the LMS imports them as they are.\n"
            "  2. Canvas: Settings > Import Course Content > Content Type 'QTI .zip file', choose a package,\n"
            "     then Import. The questions appear in a quiz and in a question bank.\n"
            "  3. Other tools that read IMS QTI 2.1 (Moodle with the QTI plugin, Blackboard, Inspera) take\n"
            "     the same packages.\n"
            "  Each package has one section per difficulty. Multiple-choice and numeric questions are\n"
            "  marked by the LMS; expression-style answers are text answers matched exactly (ignoring\n"
            "  case), so read those responses yourself. The full answer and working are in the feedback.\n"
        )
        title = "Maths Engine question banks for Canvas and other QTI 2.1 tools"
    lines = [
        title,
        "=" * len(title),
        "",
        f"Engine version: {version}",
        f"Skills: {count}; questions per difficulty: {per_difficulty}; seed: {seed}",
        "Built from the maths-engine release of that version, so the same release gives the same banks.",
        "Source: https://github.com/Jondi-Studio-Public/maths-engine",
        "manifest.json lists every skill (id, name, domain, years, curriculum codes, difficulties, question",
        "count and file).",
        "Skills that need a person to mark them (proofs and explanations) are not included.",
        "",
        how,
        "LICENCE AND ATTRIBUTION",
        "  " + service.SKILLS_ATTRIBUTION,
    ]
    if acara:
        lines.append("  " + service.ACARA_ATTRIBUTION)
        lines.append("  (The senior-secondary skills in this archive carry the ACARA attribution above.)")
    return "\n".join(lines) + "\n"


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[name])


def _json_bytes(data) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _human(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB"


def build_banks(
    out: Path,
    *,
    skills: list[str] | None = None,
    domain: str | None = None,
    per_difficulty: int = DEFAULT_PER_DIFFICULTY,
    seed: int | None = None,
    jobs: int = 1,
    version: str = __version__,
) -> dict:
    """Write the banks and archives. Returns a summary dict with `failures` as `{skill id: error}`."""
    started = time.monotonic()
    seed = default_seed(version) if seed is None else seed
    lib = _library()
    ids, notes = select_skills(lib, skills, domain)
    if not ids:
        raise ValueError("no skills selected")

    results: dict[str, dict] = {}
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as pool:
            futures = [pool.submit(build_one, i, per_difficulty, seed) for i in ids]
            for future in as_completed(futures):
                res = future.result()
                results[res["id"]] = res
    else:
        for i in ids:
            results[i] = build_one(i, per_difficulty, seed)

    failures = {i: r["error"] for i, r in sorted(results.items()) if "error" in r}
    built = [results[i] for i in ids if "error" not in results[i]]

    for sub in ("moodle", "qti"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    safe_version = re.sub(r"[^A-Za-z0-9._+-]", "_", version)
    archives = {
        "moodle": out / f"maths-engine-{safe_version}-moodle-banks.zip",
        "qti": out / f"maths-engine-{safe_version}-qti-banks.zip",
    }
    for old in archives.values():
        old.unlink(missing_ok=True)

    payload: dict[str, dict[str, bytes]] = {"moodle": {}, "qti": {}}
    manifests: dict[str, list[dict]] = {"moodle": [], "qti": []}
    for res in built:
        meta = res["meta"]
        for fmt, ext, data in (
            ("moodle", "xml", res["moodle"].encode("utf-8")),
            ("qti", "zip", res["qti"]),
        ):
            if not SAFE_NAME.match(meta["id"]) or not SAFE_NAME.match(meta["domain"]):
                failures[meta["id"]] = f"unsafe file name for domain {meta['domain']!r} / id {meta['id']!r}"
                break
            rel = f"{fmt}/{meta['domain']}/{meta['id']}.{ext}"
            payload[fmt][rel] = data
            manifests[fmt].append({**meta, "file": rel})
    for fmt in manifests:
        manifests[fmt] = [m for m in manifests[fmt] if m["id"] not in failures]
        payload[fmt] = {k: v for k, v in payload[fmt].items() if any(m["file"] == k for m in manifests[fmt])}

    count = len(manifests["moodle"])
    acara = any(m["acara_attribution"] for m in manifests["moodle"])
    sizes = {}
    for fmt in ("moodle", "qti"):
        files = dict(payload[fmt])
        for rel, data in files.items():
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        entries = [{k: v for k, v in m.items() if k != "acara_attribution"} for m in manifests[fmt]]
        manifest = {
            "engine_version": version,
            "seed": seed,
            "per_difficulty": per_difficulty,
            "format": "Moodle XML" if fmt == "moodle" else "IMS QTI 2.1",
            "skills": entries,
        }
        files["manifest.json"] = _json_bytes(manifest)
        files["README.txt"] = _readme(fmt, version, seed, per_difficulty, count, acara).encode("utf-8")
        _write_zip(archives[fmt], files)
        sizes[fmt] = archives[fmt].stat().st_size

    return {
        "skills": count,
        "questions": sum(m["questions"] for m in manifests["moodle"]),
        "sizes": sizes,
        "archives": {k: str(v) for k, v in archives.items()},
        "failures": failures,
        "notes": notes,
        "seed": seed,
        "seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Moodle XML and QTI 2.1 question banks for every skill.")
    parser.add_argument("--out", default="dist/banks", help="output directory (default dist/banks)")
    parser.add_argument("--per-difficulty", type=int, default=DEFAULT_PER_DIFFICULTY, help="questions per difficulty")
    parser.add_argument("--seed", type=int, default=None, help="override the seed derived from the engine version")
    parser.add_argument("--skills", default=None, help="comma-separated skill ids to build")
    parser.add_argument("--domain", default=None, help="only skills in this domain")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes (output is identical for any value)")
    args = parser.parse_args(argv)

    if not 1 <= args.per_difficulty <= service.MAX_COUNT:
        parser.error(f"--per-difficulty must be from 1 to {service.MAX_COUNT}")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("--seed must be from 0 to 2**63 - 1")
    ids = [i.strip() for i in args.skills.split(",") if i.strip()] if args.skills else None
    if args.skills is not None and not ids:
        parser.error("--skills names no skill")

    try:
        result = build_banks(
            Path(args.out),
            skills=ids,
            domain=args.domain,
            per_difficulty=args.per_difficulty,
            seed=args.seed,
            jobs=args.jobs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for note in result["notes"]:
        print(note, file=sys.stderr)
    failures = result["failures"]
    print(
        f"{result['skills']} skills, {result['questions']} questions; "
        f"moodle {_human(result['sizes']['moodle'])}, qti {_human(result['sizes']['qti'])}; "
        f"{len(failures)} failed; {result['seconds']:.0f} s"
    )
    if failures:
        print("FAILED skills (not in the archives; do not release until fixed):", file=sys.stderr)
        for skill_id, error in failures.items():
            print(f"  {skill_id}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
