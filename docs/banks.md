# Ready-made question banks for Moodle and Canvas

Each engine release comes with two downloads on its GitHub release page, so a teacher can fill a question bank
without running anything:

| File | For | How |
|---|---|---|
| `maths-engine-<version>-moodle-banks.zip` | Moodle | Question bank, Import, format "Moodle XML" |
| `maths-engine-<version>-qti-banks.zip` | Canvas, and other tools that read IMS QTI 2.1 | Settings, Import Course Content, "QTI .zip file" |

Unzip the download and import the skills you want; each skill is one file (`moodle/<domain>/<skill-id>.xml`, or
`qti/<domain>/<skill-id>.zip`, which Canvas takes as it is, unopened). `README.txt` in each download repeats the
steps, and `manifest.json` lists every skill with its name, domain, year levels, ACARA codes, difficulties and
question count, so you can find the right file by year or code.

A bank holds 20 questions for each difficulty the skill offers, one section per difficulty (a Moodle subcategory, a
QTI assessment section), with the answer and working in the feedback. Skills that need a person to mark them
(proofs, explanations) are left out. Multiple-choice and numeric questions are marked by the LMS; answers that are
expressions, pairs or inequalities are text answers matched exactly, so read those responses yourself. The banks
are generated from the skill definitions (CC BY 4.0) and the senior-secondary skills carry the ACARA attribution;
both are printed in the README.

For other needs (a worksheet in versions, a bank for chosen skills, different numbers) use the HTTP API in
`docs/server.md`, or run the script below with your own filters.

## Building them for a release (maintainer)

From the repository root in Windows PowerShell, with the release tag checked out and the package installed in the
venv (`py -m pip install -e .`), so the banks carry the release's version:

```powershell
py scripts/build_banks.py --jobs 8
```

That writes `dist/banks/` (`dist/` is ignored by git): the per-skill files, and the two archives named with the
installed engine version. It takes a couple of minutes for the whole library (about 820 skills, 60,000 questions;
the archives are roughly 8 MB for Moodle and 55 MB for QTI) and prints one line:

```
821 skills, 59740 questions; moodle 7.7 MB, qti 53.6 MB; 0 failed; 106 s
```

A skill whose bank cannot be built is listed on stderr and left out, and the exit status is 1: fix the generator
(or hold the skill back) and build again before attaching anything. The questions come from a seed derived from the
version, so rebuilding the same release gives the same bytes, whatever `--jobs` is.

Options: `--out DIR`, `--per-difficulty N` (default 20, at most 100), `--seed N` (override the version-derived
seed), `--skills id1,id2`, `--domain NAME` (quick tries) and `--jobs N`.

Then attach the two archives to the release, with the web UI (Releases, Edit, drop the files in) or the GitHub CLI:

```powershell
gh release upload v0.1.0 dist/banks/maths-engine-0.1.0-moodle-banks.zip dist/banks/maths-engine-0.1.0-qti-banks.zip
```

Use `--clobber` to replace files already attached. Build from an installed package: a checkout that was never
installed reports its version as `0+unknown`.
