"""Fail when a commit in a pull request has no Developer Certificate of Origin sign-off.

Usage: python scripts/check_dco.py BASE_SHA HEAD_SHA [--pr-author LOGIN] [--repo PATH]

Checks every commit reachable from HEAD_SHA but not from BASE_SHA (BASE_SHA..HEAD_SHA).
Each one needs a line ``Signed-off-by: Name <email>`` in its message whose email is the
commit author's email (compared case-insensitively). ``git commit -s`` writes that line.

Merge commits are checked like any other. A merge can carry changes that are in neither
parent (a conflict resolution, or a commit built by hand with two parents and any tree),
and this script cannot tell a clean merge from one that does, so an unsigned merge is a
way round the check. A contributor fixes one with ``git merge --signoff`` (or
``git commit --amend --signoff`` on the merge), or by rebasing the branch onto main,
which replaces the merge with the branch's own commits.

One kind of commit is not checked: commits by Dependabot, but only when the pull request
itself was opened by Dependabot (``--pr-author dependabot[bot]``) and the commit's author
email is Dependabot's own. GitHub fixes the pull request author, so a contributor cannot
claim the exemption just by writing Dependabot's email into a commit. These pull requests
still need a code-owner review before they merge.

The script reads commits with ``git cat-file`` only, needs no network, no token and no
third-party app, and uses only the standard library. Exit status:

    0  every checked commit is signed off (or there was nothing to check)
    1  at least one commit is missing its sign-off
    2  bad usage, or git could not read the range; this is never reported as a pass
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# ``git commit -s`` and GitHub's web sign-off both write "Signed-off-by: Name <email>". The \r? lets
# a message with CRLF line endings (an editor on Windows, a pasted description) match as well.
SIGNOFF = re.compile(
    r"^[ \t]*Signed-off-by:[ \t]*(?P<name>[^<>\r\n]*?)[ \t]*<(?P<email>[^<>\s]+)>[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)

# pull request author -> the author email GitHub gives that bot's commits
EXEMPT_BOTS = {"dependabot[bot]": "49699333+dependabot[bot]@users.noreply.github.com"}

_SHA = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_AUTHOR = re.compile(r"^(?P<name>.*?) ?<(?P<email>[^<>]*)> \d+ [+-]\d{4}$")


class GitError(Exception):
    pass


def _git(repo: str, *args: str) -> bytes:
    try:
        done = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False)
    except OSError as error:
        raise GitError(f"cannot run git: {error}") from error
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()[:300]
        raise GitError(f"git {args[0]} failed (exit {done.returncode}): {detail}")
    return done.stdout


def parse_commit(raw: bytes) -> tuple[int, str, str, str]:
    """Return (parent count, author name, author email, message) from a raw commit object."""
    header, _, message = raw.decode("utf-8", "replace").partition("\n\n")
    parents = 0
    name = email = ""
    for line in header.split("\n"):
        if line.startswith("parent "):
            parents += 1
        elif line.startswith("author "):
            match = _AUTHOR.match(line[len("author ") :])
            if match:
                name, email = match.group("name"), match.group("email")
    return parents, name, email, message


def _clean(text: str, limit: int = 80) -> str:
    """Make attacker-controlled text safe to print: printable characters only, and short."""
    return "".join(ch if ch.isprintable() else "?" for ch in text)[:limit]


def _scan(base: str, head: str, pr_author: str, repo: str) -> tuple[int, list[tuple[bool, str]]]:
    """Return (commits examined, one (is a merge, problem line) pair per commit that lacks a sign-off)."""
    shas = _git(repo, "rev-list", "--reverse", f"{base}..{head}").decode().split()
    problems = []
    for sha in shas:
        parents, name, email, message = parse_commit(_git(repo, "cat-file", "commit", sha))
        is_merge = parents > 1
        if pr_author in EXEMPT_BOTS and email.lower() == EXEMPT_BOTS[pr_author].lower():
            continue
        signed = {m.group("email").lower() for m in SIGNOFF.finditer(message)}
        label = "merge commit  " if is_merge else ""
        if not email:
            problems.append((is_merge, f"{sha[:10]}  {label}commit has no readable author email"))
        elif email.lower() not in signed:
            subject = message.strip().split("\n", 1)[0]
            problems.append(
                (is_merge, f"{sha[:10]}  {label}{_clean(subject)}  (author: {_clean(name)} <{_clean(email)}>)")
            )
    return len(shas), problems


def check_range(base: str, head: str, pr_author: str = "", repo: str = ".") -> tuple[int, list[str]]:
    """Return (commits examined, one problem line per commit that lacks a sign-off)."""
    examined, problems = _scan(base, head, pr_author, repo)
    return examined, [line for _, line in problems]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Require a DCO Signed-off-by line on every commit of a pull request.")
    parser.add_argument("base", help="the pull request's base commit (full SHA)")
    parser.add_argument("head", help="the pull request's head commit (full SHA)")
    parser.add_argument("--pr-author", default="", help="login of the user who opened the pull request")
    parser.add_argument("--repo", default=".", help="path of the git checkout (default: the current directory)")
    args = parser.parse_args(argv)  # exits with status 2 on bad usage
    for label, value in (("base", args.base), ("head", args.head)):
        if not _SHA.fullmatch(value):
            parser.error(f"{label} must be a full 40 or 64 character hexadecimal commit SHA, got {_clean(value, 50)!r}")
    try:
        examined, problems = _scan(args.base, args.head, args.pr_author, args.repo)
    except GitError as error:
        print(f"check_dco.py: error: {error}", file=sys.stderr)
        return 2
    if not problems:
        print(f"DCO: {examined} commit(s) in range, all signed off or exempt.")
        return 0
    print(f"DCO: {len(problems)} of {examined} commit(s) have no Signed-off-by line matching the author's email:")
    for _, line in problems:
        print(f"  {line}")
    print()
    print("Every commit needs a line 'Signed-off-by: Your Name <you@example.com>' using the same email as the")
    print("commit's author. It certifies the Developer Certificate of Origin; see CONTRIBUTING.md. To fix a branch:")
    print("  git rebase --signoff origin/main && git push --force-with-lease")
    if any(is_merge for is_merge, _ in problems):
        print()
        print("A merge commit needs a sign-off too, because a merge can carry changes that no other commit shows.")
        print("The rebase above also removes the merge: it replays your own commits on top of main. To keep a merge")
        print("instead, sign it with 'git commit --amend --signoff' while it is checked out, or redo it with")
        print("'git merge --signoff origin/main'. GitHub's 'Update branch' button makes an unsigned merge,")
        print("so rebase instead of using it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
