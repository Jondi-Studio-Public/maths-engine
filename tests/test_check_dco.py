"""scripts/check_dco.py: a pull request fails unless every commit is signed off by its author."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_dco.py"

_spec = importlib.util.spec_from_file_location("check_dco", SCRIPT)
check_dco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_dco)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

ALICE = "alice@example.com"
BOT = "49699333+dependabot[bot]@users.noreply.github.com"


class Repo:
    """A throwaway git repository whose commits are made with a fixed, hook-free configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(path),
            "GIT_AUTHOR_NAME": "Default Author",
            "GIT_AUTHOR_EMAIL": "default@example.com",
            "GIT_COMMITTER_NAME": "Committer",
            "GIT_COMMITTER_EMAIL": "committer@example.com",
        }
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str, **env: str) -> str:
        done = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=self.path,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    def commit(self, subject: str, *paragraphs: str, author: str = ALICE, name: str = "Alice A") -> str:
        message = []
        for text in (subject, *paragraphs):
            message += ["-m", text]
        self.git(
            "commit",
            "--allow-empty",
            *message,
            GIT_AUTHOR_NAME=name,
            GIT_AUTHOR_EMAIL=author,
            GIT_COMMITTER_NAME="Committer",
            GIT_COMMITTER_EMAIL="committer@example.com",
        )
        return self.git("rev-parse", "HEAD")

    def commit_verbatim(self, message: bytes, author: str = ALICE, name: str = "Alice A") -> str:
        """Commit with exactly these message bytes (git would otherwise strip a trailing carriage return)."""
        message_file = self.path.parent / "message.txt"
        message_file.write_bytes(message)
        self.git(
            "commit",
            "--allow-empty",
            "--cleanup=verbatim",
            "-F",
            str(message_file),
            GIT_AUTHOR_NAME=name,
            GIT_AUTHOR_EMAIL=author,
        )
        return self.git("rev-parse", "HEAD")

    def check(self, base: str, head: str, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), base, head, "--repo", str(self.path), *extra],
            capture_output=True,
            text=True,
        )


@pytest.fixture
def repo(tmp_path):
    r = Repo(tmp_path)
    r.base = r.commit("base commit with no sign-off, already on main")
    return r


def test_signed_commit_passes(repo):
    head = repo.commit("Add a thing", f"Signed-off-by: Alice A <{ALICE}>")
    result = repo.check(repo.base, head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 commit(s)" in result.stdout


def test_unsigned_commit_fails_and_is_named(repo):
    head = repo.commit("Add a thing without sign-off")
    result = repo.check(repo.base, head)
    assert result.returncode == 1
    assert head[:10] in result.stdout and "Add a thing without sign-off" in result.stdout
    assert "git rebase --signoff" in result.stdout
    assert result.stderr == ""


def test_the_base_commit_itself_is_not_checked(repo):
    head = repo.commit("Add a thing", f"Signed-off-by: Alice A <{ALICE}>")
    assert repo.check(repo.base, head).returncode == 0
    assert repo.check(head, head).returncode == 0  # an empty range has nothing to check


def test_signoff_by_someone_else_fails(repo):
    head = repo.commit("Add a thing", "Signed-off-by: Mallory <mallory@example.com>")
    assert repo.check(repo.base, head).returncode == 1


def test_signoff_without_angle_brackets_fails(repo):
    head = repo.commit("Add a thing", f"Signed-off-by: Alice A {ALICE}")
    assert repo.check(repo.base, head).returncode == 1


def test_email_and_key_case_do_not_matter(repo):
    head = repo.commit("Add a thing", "signed-off-by: ALICE A <Alice@Example.COM>")
    assert repo.check(repo.base, head).returncode == 0


def test_signoff_in_a_message_with_crlf_line_endings_is_accepted(repo):
    message = f"Add a thing\r\n\r\nSome body.\r\n\r\nSigned-off-by: Alice A <{ALICE}>\r\n".encode()
    head = repo.commit_verbatim(message)
    assert b"\r\n" in subprocess.run(["git", "cat-file", "commit", head], cwd=repo.path, capture_output=True).stdout
    result = repo.check(repo.base, head)
    assert result.returncode == 0, result.stdout + result.stderr


def test_crlf_message_with_the_wrong_email_still_fails(repo):
    head = repo.commit_verbatim(b"Add a thing\r\n\r\nSigned-off-by: Mallory <mallory@example.com>\r\n")
    assert repo.check(repo.base, head).returncode == 1


def test_one_unsigned_commit_among_signed_ones_fails(repo):
    repo.commit("First", f"Signed-off-by: Alice A <{ALICE}>")
    middle = repo.commit("Second, unsigned")
    head = repo.commit("Third", f"Signed-off-by: Alice A <{ALICE}>")
    result = repo.check(repo.base, head)
    assert result.returncode == 1
    assert middle[:10] in result.stdout
    assert "1 of 3" in result.stdout


def test_commit_signed_by_its_author_not_its_committer(repo):
    """The committer here is someone else; only the author's sign-off counts."""
    head = repo.commit("Add a thing", "Signed-off-by: Committer <committer@example.com>")
    assert repo.check(repo.base, head).returncode == 1


ALICE_IDENTITY = {
    "GIT_AUTHOR_NAME": "Alice A",
    "GIT_AUTHOR_EMAIL": ALICE,
    "GIT_COMMITTER_NAME": "Alice A",
    "GIT_COMMITTER_EMAIL": ALICE,
}


def _diverged_topic(repo):
    """A signed commit on a topic branch and a signed commit on main, ready to be merged."""
    repo.git("checkout", "-q", "-b", "topic")
    topic = repo.commit("Topic work", f"Signed-off-by: Alice A <{ALICE}>")
    repo.git("checkout", "-q", "main")
    side = repo.commit("Main moved", f"Signed-off-by: Alice A <{ALICE}>")
    return topic, side


def test_a_merge_commit_signed_with_git_merge_signoff_passes(repo):
    """The fix the failure message suggests: ``git merge --signoff`` writes a sign-off the check accepts."""
    _diverged_topic(repo)
    repo.git("merge", "--no-ff", "--signoff", "-q", "-m", "Merge topic into main", "topic", **ALICE_IDENTITY)
    result = repo.check(repo.base, repo.git("rev-parse", "HEAD"))
    assert result.returncode == 0, result.stdout
    assert "3 commit(s)" in result.stdout  # topic, side and the merge itself were in the range


def test_an_unsigned_merge_commit_fails_and_says_how_to_fix_it(repo):
    topic, side = _diverged_topic(repo)
    repo.git("merge", "--no-ff", "-q", "-m", "Merge topic into main (unsigned)", "topic", **ALICE_IDENTITY)
    merged = repo.git("rev-parse", "HEAD")
    result = repo.check(repo.base, merged)
    assert result.returncode == 1, result.stdout
    assert "1 of 3 commit(s)" in result.stdout  # only the merge is named, not the signed commits it joins
    assert merged[:10] in result.stdout and "merge commit" in result.stdout
    assert topic[:10] not in result.stdout and side[:10] not in result.stdout
    assert "git merge --signoff" in result.stdout and "git rebase --signoff" in result.stdout
    assert result.stderr == ""


def test_the_merge_advice_is_only_printed_when_a_merge_is_at_fault(repo):
    head = repo.commit("Add a thing without sign-off")
    result = repo.check(repo.base, head)
    assert result.returncode == 1
    assert "git rebase --signoff" in result.stdout
    assert "merge" not in result.stdout.lower()


def test_amending_an_unsigned_merge_with_signoff_makes_it_pass(repo):
    _diverged_topic(repo)
    repo.git("merge", "--no-ff", "-q", "-m", "Merge topic into main (unsigned)", "topic", **ALICE_IDENTITY)
    assert repo.check(repo.base, repo.git("rev-parse", "HEAD")).returncode == 1
    repo.git("commit", "--amend", "--signoff", "--no-edit", "-q", **ALICE_IDENTITY)
    assert repo.check(repo.base, repo.git("rev-parse", "HEAD")).returncode == 0


def test_a_merge_that_edits_files_needs_a_signoff_like_any_other_commit(repo):
    """A conflict resolution can add content that neither parent has, so the merge itself must be signed."""
    repo.git("checkout", "-q", "-b", "topic")
    for branch, text in (("topic", "from topic\n"), ("main", "from main\n")):
        if branch == "main":
            repo.git("checkout", "-q", "main")
        (repo.path / "conflict.txt").write_text(text)
        repo.git("add", "conflict.txt")
        repo.commit(f"edit on {branch}", f"Signed-off-by: Alice A <{ALICE}>")
    subprocess.run(["git", "merge", "topic"], cwd=repo.path, env=repo.env, capture_output=True)  # conflicts
    (repo.path / "conflict.txt").write_text("resolved with new content\n")
    repo.git("add", "conflict.txt")
    repo.git("commit", "-q", "-m", "merge with an unsigned conflict resolution", **ALICE_IDENTITY)
    assert len(repo.git("rev-list", "--parents", "-n", "1", "HEAD").split()) == 3  # the commit has two parents
    assert repo.check(repo.base, repo.git("rev-parse", "HEAD")).returncode == 1
    repo.git("commit", "--amend", "--signoff", "--no-edit", "-q", **ALICE_IDENTITY)
    assert repo.check(repo.base, repo.git("rev-parse", "HEAD")).returncode == 0


def test_a_hand_built_merge_with_two_parents_and_any_tree_is_not_a_way_round(repo):
    """One ``git commit-tree`` call makes a 'merge' that carries any content; it must not pass unsigned."""
    topic, _ = _diverged_topic(repo)
    repo.git("checkout", "-q", "-b", "sneaky", repo.base)
    (repo.path / "payload.py").write_text("print('not reviewed as a change by anyone')\n")
    repo.git("add", "payload.py")
    tree = repo.git("write-tree")
    merge = repo.git("commit-tree", tree, "-p", topic, "-p", repo.base, "-m", "innocuous merge", **ALICE_IDENTITY)
    assert len(repo.git("rev-list", "--parents", "-n", "1", merge).split()) == 3
    result = repo.check(repo.base, merge)
    assert result.returncode == 1
    assert merge[:10] in result.stdout and "merge commit" in result.stdout
    signed = repo.git(
        "commit-tree", tree, "-p", topic, "-p", repo.base, "-m", "m", "-m", f"Signed-off-by: Alice A <{ALICE}>",
        **ALICE_IDENTITY,
    )  # fmt: skip
    assert repo.check(repo.base, signed).returncode == 0


def test_a_merge_commit_is_still_examined_when_it_is_the_only_commit_in_the_range(repo):
    """Two parents that are both already on main: the range holds just the merge, and it still needs a sign-off."""
    older = repo.commit("Second commit on main", f"Signed-off-by: Alice A <{ALICE}>")
    tree = repo.git("rev-parse", "HEAD^{tree}")
    merge = repo.git("commit-tree", tree, "-p", older, "-p", repo.base, "-m", "redundant merge", **ALICE_IDENTITY)
    result = repo.check(older, merge)
    assert result.returncode == 1, result.stdout
    assert "1 of 1 commit(s)" in result.stdout


def test_unsigned_commit_inside_a_merged_branch_still_fails(repo):
    repo.git("checkout", "-q", "-b", "topic")
    unsigned = repo.commit("Topic work, unsigned")
    repo.git("checkout", "-q", "main")
    repo.commit("Main moved", f"Signed-off-by: Alice A <{ALICE}>")
    repo.git("merge", "--no-ff", "-q", "-m", "Merge topic", "topic")
    result = repo.check(repo.base, repo.git("rev-parse", "HEAD"))
    assert result.returncode == 1
    assert unsigned[:10] in result.stdout


def test_dependabot_exemption_needs_the_bot_to_have_opened_the_pull_request(repo):
    head = repo.commit("Bump pytest", author=BOT, name="dependabot[bot]")
    assert repo.check(repo.base, head, "--pr-author", "dependabot[bot]").returncode == 0
    # A contributor who copies the bot's email into a commit does not get the exemption.
    assert repo.check(repo.base, head, "--pr-author", "mallory").returncode == 1
    assert repo.check(repo.base, head).returncode == 1


def test_dependabot_pull_request_does_not_exempt_a_human_commit_on_it(repo):
    bot = repo.commit("Bump pytest", author=BOT, name="dependabot[bot]")
    human = repo.commit("Sneak in a change")
    result = repo.check(repo.base, human, "--pr-author", "dependabot[bot]")
    assert result.returncode == 1
    assert human[:10] in result.stdout and bot[:10] not in result.stdout


@pytest.mark.parametrize(
    "base,head",
    [
        ("--all", "HEAD"),
        ("HEAD", "HEAD"),
        ("main", "main"),
        ("abc123", "def456"),
        ("0" * 40 + "\n", "1" * 40),
        ("g" * 40, "1" * 40),
        ("", ""),
    ],
)
def test_arguments_that_are_not_full_shas_are_refused(repo, base, head):
    result = repo.check(base, head)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_unknown_commit_is_an_error_not_a_pass(repo):
    result = repo.check("0" * 40, "1" * 40)
    assert result.returncode == 2
    assert "check_dco.py: error:" in result.stderr and "Traceback" not in result.stderr
    assert result.stdout == ""


def test_not_a_git_repository_is_an_error_not_a_pass(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "0" * 40, "1" * 40, "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path.parent)},
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_commit_text_cannot_inject_workflow_commands_or_control_characters(repo):
    head = repo.commit("::error::forged\x1b[31m annotation")
    result = repo.check(repo.base, head)
    assert result.returncode == 1
    assert all(not line.startswith("::") for line in result.stdout.splitlines())
    assert "\x1b" not in result.stdout


def test_parse_commit_reads_raw_commit_objects():
    raw = (
        b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n"
        b"parent " + b"a" * 40 + b"\n"
        b"parent " + b"b" * 40 + b"\n"
        b"author A. Person <a.person@example.com> 1700000000 +1000\n"
        b"committer C <c@example.com> 1700000000 +1000\n"
        b"\n"
        b"Subject\n\nBody\n\nSigned-off-by: A. Person <a.person@example.com>\n"
    )
    parents, name, email, message = check_dco.parse_commit(raw)
    assert (parents, name, email) == (2, "A. Person", "a.person@example.com")
    assert check_dco.SIGNOFF.search(message).group("email") == "a.person@example.com"
