"""Every senior-* skill file carries the ACARA attribution header that NOTICE says it carries."""

import re

from mathgen import _paths

HEADER = (
    "# Name and description adapted from the Australian Curriculum: Senior Secondary Mathematics (Version 8.4),\n"
    "# (c) ACARA, https://v8.australiancurriculum.edu.au/, CC BY 4.0. Modified; see NOTICE.\n"
)

SENIOR_ID = re.compile(r"^id: senior-", re.MULTILINE)


def skill_files():
    root = _paths.checkout_skills_dir()
    return root, sorted(root.glob("*/*.yaml"))


def test_the_library_has_senior_skills_to_check():
    root, files = skill_files()
    senior = [p for p in files if p.name.startswith("senior-")]
    assert senior, f"no senior-* skill files found under {root}"


def test_every_senior_skill_file_starts_with_the_attribution_header():
    root, files = skill_files()
    missing = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if (path.name.startswith("senior-") or SENIOR_ID.search(text)) and not text.startswith(HEADER):
            missing.append(path.relative_to(root).as_posix())
    assert not missing, f"{len(missing)} senior-* skill file(s) lack the two-line ACARA header: {missing[:5]}"
