# Ready-made question banks for Moodle and Canvas

Each engine release comes with two downloads on its GitHub release page, so a teacher can fill a question bank
without running anything:

| File | For | How |
|---|---|---|
| `maths-engine-<version>-moodle-banks.zip` | Moodle | Question bank, Import, format "Moodle XML" |
| `maths-engine-<version>-qti-banks.zip` | Canvas, and other tools that read IMS QTI 2.1 (experimental) | Settings, Import Course Content, "QTI .zip file" |

The QTI packages are experimental: they are well-formed QTI 2.1 and the engine's tests check their marking, but no
import into Canvas or another LMS has been tried yet. Try one skill before importing many, and please report what
happens.

Unzip the download and import the skills you want; each skill is one file (`moodle/<domain>/<skill-id>.xml`, or
`qti/<domain>/<skill-id>.zip`, which Canvas takes as it is, unopened). `README.txt` in each download repeats the
steps, and `manifest.json` lists every skill with its name, domain, year levels, ACARA codes, difficulties and
question count, so you can find the right file by year or code.

A bank holds 20 questions for each difficulty the skill offers, one section per difficulty (a Moodle subcategory, a
QTI assessment section), with the answer and working in the feedback (Moodle's general feedback, QTI modal
feedback shown once a question is submitted). Skills that need a person to mark them
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
installed engine version. It takes a couple of minutes for the whole library (about 1,340 skills and 100,000
questions as of this writing; the archives are roughly 13 MB for Moodle and 85 MB for QTI) and prints one line like:

```
1078 skills, 79780 questions; moodle 13.4 MB, qti 85.0 MB; 0 failed; 152 s
```

A skill whose bank cannot be built is listed on stderr and left out, and the exit status is 1. That includes a
skill with a figure the SVG renderer cannot draw yet, and any file that does not parse as XML: fix the generator
(or hold the skill back) and build again before attaching anything. The questions come from a seed derived from the
version, so rebuilding the same release gives the same bytes, whatever `--jobs` is.

Options: `--out DIR`, `--per-difficulty N` (default 20, at most 100), `--seed N` (override the version-derived
seed), `--skills id1,id2`, `--domain NAME` (quick tries) and `--jobs N`.

Then attach the two archives to the release, with the web UI (Releases, Edit, drop the files in) or the GitHub CLI:

```powershell
gh release upload v0.2.0 dist/banks/maths-engine-0.2.0-moodle-banks.zip dist/banks/maths-engine-0.2.0-qti-banks.zip
```

Use `--clobber` to replace files already attached. Build from an installed package: a checkout that was never
installed reports its version as `0+unknown`.
