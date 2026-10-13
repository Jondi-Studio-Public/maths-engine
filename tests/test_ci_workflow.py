"""The CI workflow fails closed.

The shell steps of .github/workflows/ci.yml are pulled out of the YAML and run for real against throwaway
git repositories and stub tools, so these tests exercise the text that GitHub will run, not a copy of it.
The rest checks the structure that the safety of a pull request run depends on: pinned actions, a read-only
token, no secrets, no dangerous triggers, and a ``ci-ok`` job that cannot be satisfied by a job that was
skipped, cancelled or never wired in.
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
SHARD_SCRIPT = ROOT / "scripts" / "ci_shard.py"

needs_shell = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None, reason="bash and git are needed to run workflow steps"
)


def load(path: Path = CI) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def step_of(job: str, *, step_id: str = "", name: str = "") -> dict:
    for step in load()["jobs"][job]["steps"]:
        if (step_id and step.get("id") == step_id) or (name and step.get("name") == name):
            return step
    raise AssertionError(f"no step id={step_id!r} name={name!r} in job {job!r}")


# ---------------------------------------------------------------------------
# Running a workflow step's shell the way GitHub does: bash -eo pipefail
# ---------------------------------------------------------------------------


def clean_env(home: Path, **extra: str) -> dict:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "author@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "author@example.com",
        **extra,
    }


def run_script(script: str, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def git(cwd: Path, *args: str, env: dict | None = None) -> str:
    done = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=env or clean_env(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


BASE_FILES = {
    "mathgen/core.py": "x = 1\n",
    "skills/a/b.yaml": "id: b\n",
    "tests/test_x.py": "def test_x():\n    pass\n",
    "README.md": "readme\n",
    "AGENTS.md": "agents\n",
    "CLAUDE.md": "@AGENTS.md\n",
    "CONTRIBUTING.md": "contributing\n",
    "SECURITY.md": "security\n",
    "docs/notes.md": "notes\n",
    "mathgen/README.md": "readme\n",
    "LICENSE": "mit\n",
    "LICENSE-CONTENT": "cc\n",
    "NOTICE": "notice\n",
    ".gitignore": "*.pyc\n",
}


def make_repo(path: Path, files: dict[str, str] | None = None) -> str:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    for name, text in (BASE_FILES if files is None else files).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return commit_all(path, "base")


def write(repo: Path, name: str, text: str) -> None:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


# ---------------------------------------------------------------------------
# changes: the docs-only filter
# ---------------------------------------------------------------------------


def _edit(*names: str):
    def edit(r):
        for name in names:
            write(r, name, "changed\n")

    return edit


def _add_new_md(r):
    write(r, "docs/new/page.md", "new\n")


def _edit_licence_files(r):
    for name in ("LICENSE", "LICENSE-CONTENT", "NOTICE"):
        write(r, name, "changed\n")


def _add_gitignore_line(r):
    write(r, ".gitignore", "*.pyc\nmathgen/core.py\n")


def _rename(src: str, dst: str):
    def edit(r):
        git(r, "mv", src, dst)

    return edit


def _add(name: str):
    def edit(r):
        write(r, name, "x = 1\n")

    return edit


def _delete(name: str):
    def edit(r):
        git(r, "rm", "-q", name)

    return edit


def _md_and_py(r):
    write(r, "docs/notes.md", "changed\n")
    write(r, "mathgen/core.py", "x = 2\n")


FILTER_CASES = {
    # Markdown below a directory is the only thing that counts as docs.
    "new-markdown-file": (_add_new_md, "false"),
    "edit-markdown-in-a-directory": (_edit("docs/notes.md"), "false"),
    "markdown-in-several-directories": (_edit("docs/notes.md", "mathgen/README.md", "docs/new/a.md"), "false"),
    "delete-markdown-in-a-directory": (_delete("docs/notes.md"), "false"),
    # Every file at the root is named or read by pyproject.toml, MANIFEST.in or a test (the packaging test
    # wants README.md and the licence files to exist, the workflow test reads AGENTS.md), so a change to it,
    # or its deletion, must run the tests.
    "readme-only": (_edit("README.md"), "true"),
    "agents-md": (_edit("AGENTS.md"), "true"),
    "claude-md": (_edit("CLAUDE.md"), "true"),
    "contributing-md": (_edit("CONTRIBUTING.md"), "true"),
    "security-md": (_edit("SECURITY.md"), "true"),
    "licence-and-notice": (_edit_licence_files, "true"),
    "delete-readme": (_delete("README.md"), "true"),
    "delete-agents-md": (_delete("AGENTS.md"), "true"),
    "delete-notice": (_delete("NOTICE"), "true"),
    "delete-licence": (_delete("LICENSE"), "true"),
    "delete-licence-content": (_delete("LICENSE-CONTENT"), "true"),
    "new-root-markdown-file": (_add("CHANGES.md"), "true"),
    "root-markdown-and-docs": (_edit("README.md", "docs/notes.md"), "true"),
    # ruff skips what .gitignore lists, so editing one changes what the lint job checks
    "gitignore-edit": (_add_gitignore_line, "true"),
    "nested-gitignore": (_add("mathgen/.gitignore"), "true"),
    "gitattributes": (_add(".gitattributes"), "true"),
    "markdown-and-code": (_md_and_py, "true"),
    "rename-code-to-md": (_rename("mathgen/core.py", "mathgen/core.md"), "true"),
    "rename-skill-to-md": (_rename("skills/a/b.yaml", "skills/a/b.md"), "true"),
    "rename-test-to-md": (_rename("tests/test_x.py", "tests/test_x.md"), "true"),
    "licence-prefixed-python-file": (_add("LICENSE_check.py"), "true"),
    "licence-prefixed-directory": (_add("LICENSE-x/helper.py"), "true"),
    "delete-a-test": (_delete("tests/test_x.py"), "true"),
    "upper-case-md-extension": (_add("docs/notes.MD"), "true"),
    "markdown-extension-lookalike": (_add("docs/notes.md.py"), "true"),
    "workflow-file": (_add(".github/workflows/other.yml"), "true"),
}


@pytest.fixture
def diff_step():
    return step_of("changes", step_id="diff")["run"]


def _run_filter(script: str, repo: Path, tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, str]:
    output = tmp_path / "github_output"
    output.write_text("")
    result = run_script(script, repo, clean_env(tmp_path, GITHUB_OUTPUT=str(output), **env))
    return result, output.read_text()


@needs_shell
@pytest.mark.parametrize("case", FILTER_CASES)
def test_docs_only_filter(case, diff_step, tmp_path):
    edit, expected = FILTER_CASES[case]
    repo = tmp_path / "repo"
    base = make_repo(repo)
    git(repo, "checkout", "-q", "-b", "pr")
    edit(repo)
    head = commit_all(repo, "change")
    result, output = _run_filter(diff_step, repo, tmp_path, EVENT="pull_request", BASE_SHA=base, HEAD_SHA=head)
    assert result.returncode == 0, result.stderr
    assert output.strip() == f"code={expected}"


@needs_shell
def test_docs_only_pull_request_is_still_docs_only_after_main_moves_on(diff_step, tmp_path):
    repo = tmp_path / "repo"
    old_base = make_repo(repo)
    git(repo, "checkout", "-q", "-b", "pr")
    write(repo, "docs/notes.md", "changed\n")
    head = commit_all(repo, "docs")
    git(repo, "checkout", "-q", "main")
    write(repo, "mathgen/core.py", "x = 2\n")
    new_base = commit_all(repo, "code change on main after the branch point")
    assert new_base != old_base
    _, output = _run_filter(diff_step, repo, tmp_path, EVENT="pull_request", BASE_SHA=new_base, HEAD_SHA=head)
    assert output.strip() == "code=false"


@needs_shell
def test_empty_diff_runs_everything(diff_step, tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    _, output = _run_filter(diff_step, repo, tmp_path, EVENT="pull_request", BASE_SHA=base, HEAD_SHA=base)
    assert output.strip() == "code=true"


@needs_shell
@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_events_other_than_pull_request_always_run_everything(event, diff_step, tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    _, output = _run_filter(diff_step, repo, tmp_path, EVENT=event, BASE_SHA="", HEAD_SHA="")
    assert output.strip() == "code=true"


@needs_shell
def test_unreadable_diff_fails_the_job_instead_of_reading_as_docs_only(diff_step, tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    head = git(repo, "rev-parse", "HEAD")
    result, output = _run_filter(diff_step, repo, tmp_path, EVENT="pull_request", BASE_SHA="0" * 40, HEAD_SHA=head)
    assert result.returncode != 0
    assert "code=" not in output


@needs_shell
def test_shallow_clone_missing_the_base_commit_fails_the_job(diff_step, tmp_path):
    origin = tmp_path / "origin"
    base = make_repo(origin)
    write(origin, "README.md", "changed\n")
    head = commit_all(origin, "docs")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{origin}", str(clone)],
        env=clean_env(tmp_path),
        check=True,
        capture_output=True,
    )
    result, output = _run_filter(diff_step, clone, tmp_path, EVENT="pull_request", BASE_SHA=base, HEAD_SHA=head)
    assert result.returncode != 0
    assert "code=" not in output


# ---------------------------------------------------------------------------
# test: the shard step must never run the whole suite or an empty list
# ---------------------------------------------------------------------------

SHARD_STEP = "Run this shard's test files"


@pytest.fixture
def shard_tree(tmp_path):
    """A small repository layout with the real ci_shard.py and a stub ``python`` that records pytest's arguments."""
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    shutil.copy(SHARD_SCRIPT, work / "scripts" / "ci_shard.py")
    (work / "tests").mkdir()
    for name, size in {"test_a.py": 400, "test_b.py": 300, "test_c d.py": 200, "test_e.py": 100}.items():
        (work / "tests" / name).write_text("#" * size)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pytest" ]; then\n'
        '  printf "%s\\n" "$@" > "$PYTEST_ARGS_FILE"\n'
        "  exit 0\n"
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    args_file = tmp_path / "pytest_args.txt"

    def run(shard: str, shards: str) -> subprocess.CompletedProcess:
        env = clean_env(
            tmp_path,
            SHARD=shard,
            SHARDS=shards,
            PYTEST_ARGS_FILE=str(args_file),
            REAL_PYTHON=sys.executable,
        )
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        return run_script(step_of("test", name=SHARD_STEP)["run"], work, env)

    run.work = work
    run.args_file = args_file
    return run


@needs_shell
def test_shard_step_runs_exactly_its_own_files(shard_tree):
    result = shard_tree("0", "2")
    assert result.returncode == 0, result.stderr
    args = shard_tree.args_file.read_text().splitlines()
    assert args[:2] == ["-m", "pytest"]
    assert args[-5:] == ["-q", "-n", "auto", "--dist", "worksteal"]
    files = args[2:-5]
    assert files == ["tests/test_a.py", "tests/test_e.py"]
    other = shard_tree.work / "scripts" / "ci_shard.py"
    # and the other shard gets the rest, including the file name with a space as one argument
    assert shard_tree("1", "2").returncode == 0
    assert shard_tree.args_file.read_text().splitlines()[2:-5] == ["tests/test_b.py", "tests/test_c d.py"]
    assert other.exists()


@needs_shell
@pytest.mark.parametrize(
    "shard,shards",
    [("2", "2"), ("9", "2"), ("x", "2"), ("", "2"), ("0", ""), ("0", "x"), ("-1", "2"), ("0", "99")],
    ids=[
        "index-equals-count",
        "index-too-big",
        "non-integer",
        "empty-index",
        "empty-count",
        "non-integer-count",
        "negative",
        "more-shards-than-files",
    ],
)
def test_shard_step_fails_and_never_starts_pytest_when_ci_shard_refuses(shard_tree, shard, shards):
    result = shard_tree(shard, shards)
    assert result.returncode != 0, result.stdout
    assert not shard_tree.args_file.exists(), "pytest was started although the shard list was bad"


@needs_shell
def test_shard_step_fails_on_an_empty_shard(shard_tree):
    for path in (shard_tree.work / "tests").iterdir():
        path.unlink()
    (shard_tree.work / "tests" / "test_big.py").write_text("#" * 50)
    (shard_tree.work / "tests" / "test_zero_a.py").write_text("")
    (shard_tree.work / "tests" / "test_zero_b.py").write_text("")
    # three files, three shards, but both empty files land on one shard, leaving shard 2 with nothing
    assert shard_tree("0", "3").returncode == 0
    shard_tree.args_file.unlink()
    result = shard_tree("2", "3")
    assert result.returncode != 0
    assert not shard_tree.args_file.exists()


@needs_shell
def test_shard_step_fails_when_ci_shard_is_missing_or_crashes(shard_tree):
    script = shard_tree.work / "scripts" / "ci_shard.py"
    script.unlink()
    assert shard_tree("0", "2").returncode != 0
    script.write_text("raise SystemExit(3)\n")
    assert shard_tree("0", "2").returncode == 3
    assert not shard_tree.args_file.exists()


@needs_shell
def test_shard_step_fails_when_ci_shard_succeeds_with_no_output(shard_tree):
    (shard_tree.work / "scripts" / "ci_shard.py").write_text("")
    result = shard_tree("0", "2")
    assert result.returncode == 1
    assert "has no test files" in result.stdout + result.stderr
    assert not shard_tree.args_file.exists(), "an empty list must not fall through to a whole-suite pytest run"


# ---------------------------------------------------------------------------
# ci-ok
# ---------------------------------------------------------------------------

CI_OK_STEP = "Require every job to have ended as expected"
PR_GREEN = dict(
    EVENT="pull_request", CODE="true", CHANGES="success", DCO="success", LINT="success",
    WORKFLOWS="success", SECRETS="success", TEST="success", SPEED="success",
)  # fmt: skip


def run_ci_ok(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess:
    env = clean_env(tmp_path, **{**PR_GREEN, **overrides})
    return run_script(step_of("ci-ok", name=CI_OK_STEP)["run"], tmp_path, env)


@needs_shell
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"CODE": "false", "TEST": "skipped", "SPEED": "skipped"},
        {"EVENT": "push", "DCO": "skipped"},
        {"EVENT": "schedule", "DCO": "skipped"},
        {"EVENT": "workflow_dispatch", "DCO": "skipped"},
    ],
    ids=["pull-request", "docs-only-pull-request", "push", "nightly", "manual"],
)
def test_ci_ok_accepts_exactly_the_expected_outcomes(tmp_path, overrides):
    result = run_ci_ok(tmp_path, **overrides)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_shell
@pytest.mark.parametrize("job", ["CHANGES", "LINT", "WORKFLOWS", "SECRETS"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", ""])
@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_ci_ok_rejects_any_always_run_job_that_did_not_succeed(tmp_path, job, outcome, event):
    extra = {"DCO": "skipped"} if event == "push" else {}
    assert run_ci_ok(tmp_path, EVENT=event, **{job: outcome}, **extra).returncode != 0


@needs_shell
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", ""])
def test_ci_ok_rejects_a_pull_request_whose_dco_job_did_not_succeed(tmp_path, outcome):
    assert run_ci_ok(tmp_path, DCO=outcome).returncode != 0


@needs_shell
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", ""])
def test_ci_ok_rejects_dco_running_on_anything_but_a_pull_request(tmp_path, outcome):
    assert run_ci_ok(tmp_path, EVENT="push", DCO=outcome).returncode != 0


@needs_shell
@pytest.mark.parametrize("job", ["TEST", "SPEED"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", ""])
def test_ci_ok_rejects_tests_that_were_needed_but_did_not_pass(tmp_path, job, outcome):
    assert run_ci_ok(tmp_path, **{job: outcome}).returncode != 0


@needs_shell
@pytest.mark.parametrize("job", ["TEST", "SPEED"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", ""])
def test_ci_ok_rejects_a_docs_only_verdict_unless_the_tests_were_skipped(tmp_path, job, outcome):
    skipped = {"TEST": "skipped", "SPEED": "skipped"}
    assert run_ci_ok(tmp_path, CODE="false", **{**skipped, job: outcome}).returncode != 0


@needs_shell
@pytest.mark.parametrize("code", ["", "maybe", "True", "1"])
def test_ci_ok_rejects_a_missing_or_unknown_changes_verdict(tmp_path, code):
    assert run_ci_ok(tmp_path, CODE=code).returncode != 0
    assert run_ci_ok(tmp_path, CODE=code, TEST="skipped", SPEED="skipped").returncode != 0


@needs_shell
def test_ci_ok_rejects_docs_only_on_an_event_that_cannot_be_docs_only(tmp_path):
    overrides = {"EVENT": "push", "DCO": "skipped", "CODE": "false", "TEST": "skipped", "SPEED": "skipped"}
    assert run_ci_ok(tmp_path, **overrides).returncode != 0


def test_ci_ok_depends_on_every_other_job_and_reads_each_result():
    doc = load()
    others = set(doc["jobs"]) - {"ci-ok"}
    job = doc["jobs"]["ci-ok"]
    assert set(job["needs"]) == others, "ci-ok must need every other job, or a new job could fail unnoticed"
    assert job["if"] == "always()"
    env_text = " ".join(str(v) for v in step_of("ci-ok", name=CI_OK_STEP)["env"].values())
    assert set(re.findall(r"needs\.([\w-]+)\.result", env_text)) == others


# ---------------------------------------------------------------------------
# secrets: the pull request must not configure its own scan
# ---------------------------------------------------------------------------

GITLEAKS_STEP = "Scan with a configuration the pull request cannot change"
# A stand-in for the gitleaks binary. It records each call and what the checkout looked like at that moment,
# and exits 1 when FAKE_FAIL_ON names the subcommand (git or dir) it was called with.
FAKE_GITLEAKS = """#!/usr/bin/env bash
mode="$1"
printf '%s\\n' "$*" >> "$RUNNER_TEMP/calls.txt"
config=""; ignore_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) config="$2"; shift ;;
    --gitleaks-ignore-path) ignore_dir="$2"; shift ;;
  esac
  shift
done
cp "$config" "$RUNNER_TEMP/config.seen"
if [ -f "$ignore_dir/.gitleaksignore" ]; then cp "$ignore_dir/.gitleaksignore" "$RUNNER_TEMP/ignore.seen"; fi
ls -A > "$RUNNER_TEMP/checkout.seen"
find . -name .gitattributes -not -path './.git/*' >> "$RUNNER_TEMP/attrs_in_checkout.seen"
git ls-files -- ':(glob)**/.gitattributes' >> "$RUNNER_TEMP/attrs_in_index.seen"
if [ "${FAKE_FAIL_ON:-}" = "$mode" ]; then exit 1; fi
"""
DEFAULT_CONFIG = "[extend]\nuseDefault = true\n"


def _scan(tmp_path: Path, repo: Path, **env: str) -> tuple[subprocess.CompletedProcess, Path]:
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    fake = runner_temp / "gitleaks"
    fake.write_text(FAKE_GITLEAKS)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    result = run_script(
        step_of("secrets", name=GITLEAKS_STEP)["run"], repo, clean_env(tmp_path, RUNNER_TEMP=str(runner_temp), **env)
    )
    return result, runner_temp


@needs_shell
def test_gitleaks_uses_the_base_commits_policy_files_and_deletes_the_pull_requests_copies(tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo, {**BASE_FILES, ".gitleaks.toml": "# base config\n", ".gitleaksignore": "base-fingerprint\n"})
    git(repo, "checkout", "-q", "-b", "pr")
    write(repo, ".gitleaks.toml", "[allowlist]\npaths = ['''.*''']\n")
    write(repo, ".gitleaksignore", "pr-fingerprint\n")
    commit_all(repo, "a pull request that silences the scan")
    result, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runner_temp / "config.seen").read_text() == "# base config\n"
    assert (runner_temp / "ignore.seen").read_text() == "base-fingerprint\n"
    checkout = (runner_temp / "checkout.seen").read_text().split()
    assert ".gitleaks.toml" not in checkout and ".gitleaksignore" not in checkout
    calls = (runner_temp / "calls.txt").read_text().splitlines()
    assert len(calls) == 2, "one pass over the history and one over the final tree"
    policy = runner_temp / "gitleaks-policy"
    for call, mode in zip(calls, ("git", "dir")):
        args = call.split()
        assert args[0] == mode and args[-1] == ".", call
        assert "--ignore-gitleaks-allow" in args and "--redact" in args, call
        assert f"--config {policy}/.gitleaks.toml" in call and f"--gitleaks-ignore-path {policy}" in call, call


@needs_shell
@pytest.mark.parametrize("failing", ["git", "dir"])
def test_gitleaks_runs_both_passes_and_fails_if_either_finds_something(tmp_path, failing):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    result, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base, FAKE_FAIL_ON=failing)
    assert result.returncode != 0
    assert [call.split()[0] for call in (runner_temp / "calls.txt").read_text().splitlines()] == ["git", "dir"]


@needs_shell
def test_gitleaks_passes_when_neither_pass_finds_anything(tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    result, _ = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_shell
def test_gitleaks_never_sees_a_gitattributes_file_in_the_checkout_or_the_index(tmp_path):
    """`* -diff` or `*.txt binary` makes `git log -p` print no patch, which hides a secret from the scan."""
    repo = tmp_path / "repo"
    base = make_repo(repo)
    git(repo, "checkout", "-q", "-b", "pr")
    write(repo, ".gitattributes", "* -diff\n")
    write(repo, "sub/deeper/.gitattributes", "*.txt binary\n")
    write(repo, "sub/deeper/secret.txt", "not a real secret\n")
    commit_all(repo, "a pull request that hides its diffs")
    assert git(repo, "ls-files", "--", ":(glob)**/.gitattributes").split() == [
        ".gitattributes",
        "sub/deeper/.gitattributes",
    ]
    result, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runner_temp / "attrs_in_checkout.seen").read_text() == ""
    assert (runner_temp / "attrs_in_index.seen").read_text() == ""
    assert (repo / "sub" / "deeper" / "secret.txt").exists(), "only the attributes files are removed"


@needs_shell
def test_gitleaks_falls_back_to_the_default_rules_when_the_base_has_no_policy_files(tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    git(repo, "checkout", "-q", "-b", "pr")
    write(repo, ".gitleaks.toml", "[allowlist]\npaths = ['''.*''']\n")
    write(repo, ".gitleaksignore", "pr-fingerprint\n")
    commit_all(repo, "a pull request that adds its own policy")
    result, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runner_temp / "config.seen").read_text() == DEFAULT_CONFIG
    assert not (runner_temp / "ignore.seen").exists()
    assert not {".gitleaks.toml", ".gitleaksignore"} & set((runner_temp / "checkout.seen").read_text().split())


@needs_shell
def test_gitleaks_on_a_push_uses_the_commits_own_policy_files(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo, {**BASE_FILES, ".gitleaks.toml": "# main config\n"})
    result, runner_temp = _scan(tmp_path, repo, EVENT="push", BASE_SHA="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runner_temp / "config.seen").read_text() == "# main config\n"


# ---------------------------------------------------------------------------
# Structure of every workflow
# ---------------------------------------------------------------------------

ALL_WORKFLOWS = sorted(p for p in WORKFLOWS.iterdir() if p.suffix in {".yml", ".yaml"})
ALLOWED_TRIGGERS = {"push", "pull_request", "schedule", "workflow_dispatch"}
#: The only runner a job may use: a version-pinned GitHub-hosted one. A fork's pull request runs the workflow as
#: its own branch wrote it, so a self-hosted runner would run a stranger's code on the maintainer's machine.
RUNS_ON = "ubuntu-24.04"


def triggers_of(doc: dict) -> set[str]:
    on = doc.get("on", doc.get(True))  # YAML 1.1 reads a bare "on" as the boolean True
    return set(on) if isinstance(on, (dict, list)) else {on}


def test_ci_workflow_exists():
    assert CI in ALL_WORKFLOWS


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_full_commit_sha_with_a_version_comment(path):
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)[ \t]*(#.*)?$", text, flags=re.M)
    assert len(found) == len(re.findall(r"^\s*(?:-\s+)?uses:", text, flags=re.M)), "a uses: line has an odd shape"
    for ref, comment in found:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}", ref), f"{ref} is not pinned to a commit SHA"
        assert re.match(r"#\s*v\d+(\.\d+)*\b", comment), f"{ref} needs a trailing '# vX.Y.Z' comment"


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_workflow_has_only_safe_triggers_a_read_only_token_and_no_secrets(path):
    doc = load(path)
    assert triggers_of(doc) <= ALLOWED_TRIGGERS, "pull_request_target, workflow_run, issue_comment etc. are not allowed"
    assert doc["permissions"] == {"contents": "read"}
    for job_id, job in doc["jobs"].items():
        assert "permissions" not in job, f"job {job_id} must not widen the token"
        assert "timeout-minutes" in job, f"job {job_id} needs a timeout"
        assert job["runs-on"] == RUNS_ON, f"job {job_id}: must run on the pinned hosted runner"
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w.-])secrets\.|secrets:\s*inherit", text), "no secrets in a public repository"
    code = re.sub(r"(?m)^\s*#.*$", "", text)
    assert "self-hosted" not in code, "a public repository never names a self-hosted runner"


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_no_run_step_interpolates_a_github_expression(path):
    """Values reach shell code through env: only, so a branch name or PR title cannot become code."""
    for job_id, job in load(path)["jobs"].items():
        for step in job["steps"]:
            assert "${{" not in step.get("run", ""), f"{job_id}: interpolates an expression into a run script"


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_every_checkout_drops_its_credentials(path):
    for job_id, job in load(path)["jobs"].items():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False, f"{job_id}: checkout keeps the token on disk"


def test_pull_request_runs_only_target_main_and_only_superseded_pr_runs_are_cancelled():
    doc = load()
    assert doc["on" if "on" in doc else True]["pull_request"] == {"branches": ["main"]}
    concurrency = doc["concurrency"]
    # The expressions cannot be evaluated here, so this pins their shape: pull requests share a group per
    # PR number and nothing else does (run_id is unique), and only pull_request runs are cancelled.
    assert "github.event.pull_request.number" in concurrency["group"]
    assert "github.run_id" in concurrency["group"]
    assert "github.ref" not in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_shard_count_is_the_matrix_length_everywhere_it_is_used():
    job = load()["jobs"]["test"]
    shards = job["strategy"]["matrix"]["shard"]
    assert shards == list(range(len(shards))) and len(shards) >= 1
    assert job["env"]["SHARD"] == "${{ matrix.shard }}"
    assert job["env"]["SHARDS"] == "${{ strategy.job-total }}"
    assert "strategy.job-total" in job["name"]
    assert job["strategy"]["fail-fast"] is False
    assert 'ci_shard.py "$SHARD" "$SHARDS"' in step_of("test", name=SHARD_STEP)["run"], (
        "ci_shard must get both numbers from env"
    )


def test_pytest_is_never_given_an_inline_command_substitution_for_its_file_list():
    for job_id, job in load()["jobs"].items():
        for step in job["steps"]:
            assert not re.search(r"pytest[^\n]*\$\(", step.get("run", "")), f"{job_id}: pytest $(...) hides failures"


def test_every_download_is_checksum_verified():
    for job_id, job in load()["jobs"].items():
        for step in job["steps"]:
            script = step.get("run", "")
            assert script.count("curl ") == script.count("sha256sum -c"), f"{job_id}: unverified download"
    for job_id in ("workflows", "secrets"):
        for key, value in load()["jobs"][job_id]["env"].items():
            if key.endswith("_SHA256"):
                assert re.fullmatch(r"[0-9a-f]{64}", str(value).split()[0]), key


def test_dco_job_runs_only_for_pull_requests_and_uses_the_pr_authors_login():
    job = load()["jobs"]["dco"]
    assert job["if"] == "github.event_name == 'pull_request'"
    run = step_of("dco", name="Require a Signed-off-by line on every commit")
    assert run["run"] == 'python3 scripts/check_dco.py "$BASE_SHA" "$HEAD_SHA" --pr-author "$PR_AUTHOR"'
    assert run["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    checkout = job["steps"][0]
    assert checkout["with"]["fetch-depth"] == 0, "the range check needs the history"


def test_actionlint_and_zizmor_run_in_ci():
    runs = " ".join(step.get("run", "") for step in load()["jobs"]["workflows"]["steps"])
    assert "actionlint" in runs and "-shellcheck" in runs and "zizmor" in runs


# ---------------------------------------------------------------------------
# Pinned dependencies, Dependabot, CODEOWNERS
# ---------------------------------------------------------------------------

CONSTRAINTS = ROOT / ".github" / "requirements-ci.txt"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _constraint_pins() -> dict[str, str]:
    pins = {}
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.!+_-]*)", line)
            assert match, f"not an exact pin: {line!r}"
            assert _canonical(match.group(1)) not in pins, f"pinned twice: {line!r}"
            pins[_canonical(match.group(1))] = match.group(2)
    return pins


def test_pip_constraint_points_at_the_constraints_file():
    """PIP_CONSTRAINT pins what pip installs. From pip 26.2 only PIP_BUILD_CONSTRAINT pins the build backend."""
    env = load()["env"]
    assert env["PIP_CONSTRAINT"] == "${{ github.workspace }}/.github/requirements-ci.txt"
    assert env["PIP_BUILD_CONSTRAINT"] == env["PIP_CONSTRAINT"]
    assert CONSTRAINTS.is_file()


def test_constraints_pin_every_direct_dependency_the_build_backend_and_the_lint_tools():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wanted = list(project["project"]["dependencies"])
    wanted += [req for extra in project["project"].get("optional-dependencies", {}).values() for req in extra]
    wanted += project["build-system"]["requires"]
    names = {_canonical(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req).group(0)) for req in wanted}
    names |= {"ruff", "zizmor"}  # installed by name in the lint and workflows jobs
    pins = _constraint_pins()
    assert not names - set(pins), f"unpinned in {CONSTRAINTS.name}: {sorted(names - set(pins))}"


def test_every_package_the_workflow_installs_by_name_is_pinned():
    pins = _constraint_pins()
    for job_id, job in load()["jobs"].items():
        for step in job["steps"]:
            for match in re.finditer(r"^\s*pip install ([A-Za-z][\w.-]*)", step.get("run", ""), flags=re.M):
                assert _canonical(match.group(1)) in pins, f"{job_id}: pip install {match.group(1)} is not pinned"


def test_dependabot_updates_actions_and_the_constraints_file_weekly_in_groups():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert config["version"] == 2
    by_ecosystem = {entry["package-ecosystem"]: entry for entry in config["updates"]}
    assert set(by_ecosystem) == {"github-actions", "pip"}
    for entry in by_ecosystem.values():
        assert entry["schedule"]["interval"] == "weekly"
        assert entry["groups"], "updates should arrive grouped"
    pip_dir = by_ecosystem["pip"]["directory"]
    assert (ROOT / pip_dir.lstrip("/") / CONSTRAINTS.name).is_file(), "dependabot would not find the constraints file"


CODEOWNER = "@jondihaynes"
PROTECTED = [
    ".github/", "scripts/", "tests/", "pyproject.toml", "AGENTS.md", "SECURITY.md", "NOTICE", "LICENSE*",
]  # fmt: skip


def _codeowner_rules() -> list[tuple[str, list[str]]]:
    rules = []
    for line in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            rules.append((fields[0], fields[1:]))
    return rules


def test_codeowners_starts_with_a_catch_all_so_no_file_can_be_left_out():
    """A PR can neuter CI through conftest.py, pytest.ini, a ruff config, .gitignore... no list is complete."""
    rules = _codeowner_rules()
    assert rules[0] == ("*", [CODEOWNER]), "the first rule must be '* @owner'"
    # The last matching rule wins, so a later rule naming anyone else would take a path out of review.
    assert all(owners == [CODEOWNER] for _, owners in rules), "every rule must name only the maintainer"
    assert [pattern for pattern, _ in rules].count("*") == 1, "a second '*' rule would override the first"


def test_codeowners_also_lists_the_files_that_decide_what_green_means():
    rules = {pattern.lstrip("/"): owners for pattern, owners in _codeowner_rules()}
    for path in PROTECTED:
        assert rules.get(path) == [CODEOWNER], f"{path} must be owned by {CODEOWNER}"
    for path in (".gitattributes", ".gitignore", "conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "ruff.toml"):
        assert rules.get(path) == [CODEOWNER], f"{path} changes how tests and lint run or what the scan sees"


def test_ci_yml_and_agents_md_do_not_present_the_codeowners_list_as_complete():
    header = CI.read_text(encoding="utf-8").split("\non:", 1)[0]
    assert "catch-all" in header and "cannot be complete" in header
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = agents.split("## What CODEOWNERS protects", 1)[1].split("\n## ", 1)[0]
    assert "* @jondihaynes" in section and "No list of paths can be complete" in section


# ---------------------------------------------------------------------------
# AGENTS.md: what an agent does with a pull request branch
# ---------------------------------------------------------------------------


def test_agents_md_says_a_pull_request_branch_and_its_agent_files_are_untrusted():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = text.split("## A pull request branch is untrusted", 1)[1].split("\n## ", 1)[0]
    for needle in ("`AGENTS.md`", "`CLAUDE.md`", "`.claude/`", "`.mcp.json`", "untrusted", "`main`", "`gh pr diff`"):
        assert needle in section, needle
    assert "Never review it by checking it out and running its code" in section


def test_agents_md_requires_signoff_and_covers_unsigned_merges():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    commits = text.split("## Commits", 1)[1].split("\n## ", 1)[0]
    assert "`Signed-off-by:`" in commits
    assert "merge commit" in commits and "check_dco.py" in commits


# ---------------------------------------------------------------------------
# The docs-only filter is sound only while nothing reads the files it skips
# ---------------------------------------------------------------------------

# A quoted string in a test, script or module that looks like the name of a document.
_DOC_NAME = re.compile(
    r"""["']([\w./*\\-]*(?:\.md|README|LICENSE|NOTICE)[\w./*\\-]*)["']""",
)
# The CI tests name a lot of documents as test data, so only what they read from the repository counts.
_ROOT_READ = re.compile(
    r"""\bROOT\s*/\s*["']([\w./*\\-]*(?:\.md|README|LICENSE|NOTICE)[\w./*\\-]*)["']""",
)
# A document-like word inside pyproject.toml (strings only) or MANIFEST.in.
_CONFIG_NAME = re.compile(r"""(?<![\w./*-])((?:[\w.-]+/)*[\w.-]*(?:\.md|README|LICENSE|NOTICE)[\w.-]*)(?![\w/*-])""")
_CI_TESTS = {"tests/test_ci_workflow.py", "tests/test_ci_shard.py", "tests/test_check_dco.py"}
_ALLOWED_DOC_MENTIONS = {("tests/test_render_skill_questions.py", "render.md")}  # a file the test writes


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _named_documents() -> list[tuple[str, str]]:
    """(where, name) for each document-like file that a test, script, module, pyproject.toml or MANIFEST.in names."""
    found = []
    for folder in ("tests", "scripts", "mathgen"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            pattern = _ROOT_READ if rel in _CI_TESTS else _DOC_NAME
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for match in pattern.finditer(line):
                    if (rel, match.group(1)) not in _ALLOWED_DOC_MENTIONS:
                        found.append((f"{rel}:{number}", match.group(1)))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for text in _walk_strings(pyproject):
        found += [("pyproject.toml", match.group(1)) for match in _CONFIG_NAME.finditer(text)]
    manifest = ROOT / "MANIFEST.in"
    if manifest.is_file():
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            found += [(f"MANIFEST.in:{number}", m.group(1)) for m in _CONFIG_NAME.finditer(line.split("#", 1)[0])]
    return found


def _classify(tmp_path: Path, paths) -> dict[str, str]:
    """What the real ``changes`` step says ("code=true" or "code=false") about a pull request touching only each path."""
    step = step_of("changes", step_id="diff")["run"]
    repo = tmp_path / "probe"
    base = make_repo(repo)
    verdicts = {}
    for path in paths:
        git(repo, "checkout", "-q", "-B", "probe", base)
        write(repo, path, "probe\n")
        head = commit_all(repo, f"touch {path}")
        result, output = _run_filter(step, repo, tmp_path, EVENT="pull_request", BASE_SHA=base, HEAD_SHA=head)
        assert result.returncode == 0, result.stderr
        verdicts[path] = output.strip()
    return verdicts


@needs_shell
def test_a_document_that_a_test_script_or_config_names_is_never_docs_only(tmp_path):
    """If one were, a pull request that changed or deleted only that file would skip the tests that need it."""
    named = _named_documents()
    where = {}
    for place, name in named:
        where.setdefault(name.removeprefix("./"), place)
    assert "AGENTS.md" in where, "the scan found no read of AGENTS.md, so it is broken"
    globs = sorted(f"{place}: {name}" for name, place in where.items() if "*" in name)
    assert not globs, "A glob over documents cannot be classified; name the files:\n" + "\n".join(globs)
    verdicts = _classify(tmp_path, sorted(name for name in where if ".." not in name))
    docs_only = [f"{where[name]}: {name}" for name, verdict in verdicts.items() if verdict != "code=true"]
    assert not docs_only, (
        "These are named or read by a test, script, pyproject.toml or MANIFEST.in, but the 'changes' job in ci.yml "
        "treats them as docs-only, so a pull request that changes only that file would run no tests. Take them out "
        "of the docs-only pattern, or stop naming them:\n" + "\n".join(docs_only)
    )


@needs_shell
def test_markdown_under_plugins_is_code_for_the_docs_only_filter(tmp_path):
    """The plugins' skills are instructions agents follow, and tests/public/test_plugins.py reads every one of them."""
    verdicts = _classify(tmp_path, ["plugins/maths-engine-tutor/skills/maths-tutoring/SKILL.md", "docs/server.md"])
    assert verdicts == {
        "plugins/maths-engine-tutor/skills/maths-tutoring/SKILL.md": "code=true",
        "docs/server.md": "code=false",
    }


def _root_files() -> list[str]:
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        names = done.stdout.decode("utf-8", "replace").split("\0")
    except (OSError, subprocess.CalledProcessError):
        names = [p.name for p in ROOT.iterdir() if p.is_file()]
    return sorted({name for name in names if name and "/" not in name})


@needs_shell
def test_every_file_at_the_repository_root_is_code_for_the_docs_only_filter(tmp_path):
    names = _root_files()
    assert {"AGENTS.md", "README.md", "pyproject.toml"} <= set(names)
    verdicts = _classify(tmp_path, names)
    assert {name: v for name, v in verdicts.items() if v != "code=true"} == {}


# ---------------------------------------------------------------------------
# secrets: the file types and paths gitleaks skips on its own, and the policy commit
# ---------------------------------------------------------------------------

FILE_TYPES_STEP = "Refuse files the secret scan would skip"
ALLOWED_TREE = {
    **BASE_FILES,
    "pyproject.toml": "x = 1\n",
    "MANIFEST.in": "graft skills\n",
    "setup.cfg": "",
    "notes.txt": "n\n",
    ".github/CODEOWNERS": "* @owner\n",
    ".github/workflows/ci.yml": "name: ci\n",
    ".github/requirements-ci.txt": "pytest==1\n",
    ".gitattributes": "",
    ".gitleaksignore": "",
    ".gitleaks.toml": "",
    "tests/data/table.yaml": "a: 1\n",
    "mathgen/library/helpers.py": "x = 1\n",  # a directory called library, not lib
    "tests/envelope/test_a.py": "x = 1\n",  # a directory called envelope, not env
    "docs/vendor_notes.md": "x\n",  # vendor only as part of a file name
    # the Agent Plugins manifests, at their own paths
    "plugins/p/plugin.json": "{}\n",
    "plugins/p/mcp.json": "{}\n",
    "plugins/p/.claude-plugin/plugin.json": "{}\n",
    ".claude-plugin/marketplace.json": "{}\n",
}


def _refuse_files(tmp_path: Path, files: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    repo = cwd
    if repo is None:
        repo = tmp_path / "repo"
        make_repo(repo, files)
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir(exist_ok=True)
    env = clean_env(tmp_path, RUNNER_TEMP=str(runner_temp))
    return run_script(step_of("secrets", name=FILE_TYPES_STEP)["run"], repo, env)


@needs_shell
def test_the_text_types_this_repository_holds_pass_the_file_type_check(tmp_path):
    result = _refuse_files(tmp_path, ALLOWED_TREE)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_shell
def test_every_tracked_file_of_this_repository_passes_the_file_type_check(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = _refuse_files(tmp_path, {}, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


# Each of these hides a secret from gitleaks 8.30 with its default rules, which was shown by scanning
# a token placed in a file of that name; archives are only opened with --max-archive-depth.
REFUSED_PATHS = [
    "logo.png",
    "logo.svg",
    "scan.jpg",
    "photo.gif",
    "font.woff",
    "doc.pdf",
    "report.docx",
    "sheet.xlsx",
    "data.bin",
    "tool.exe",
    "debug.pdb",
    "model.gltf",
    "data.zip",
    "data.tar.gz",
    "data.gz",
    "data.tar",
    "data.7z",
    "go.sum",
    "go.mod",
    "package-lock.json",
    "Pipfile.lock",
    "poetry.lock",
    "data.json",
    "plugin.json",  # the manifest names pass only at the plugins' own paths
    "mcp.json",
    "marketplace.json",
    "docs/mcp.json",
    "plugins/plugin.json",
    "plugins/p/marketplace.json",
    "plugins/p/data.json",
    "plugins/p/skills/s/plugin.json",  # a case pattern's * would match the extra folders
    "plugins/p/q/mcp.json",
    "plugins/p/x/.claude-plugin/plugin.json",
    "page.html",
    "run.sh",
    "Makefile",
    "gradlew",
    ".env",
    ".gitmodules",
    "UPPER.PY",
    "notes.rst",
    "node_modules/pkg/x.py",
    "a/bower_components/x.py",
    "vendor/bundle/x.py",
    "vendor/github.com/x/y.py",
    "env/lib/x.py",
    "tests/venv/x.py",
    "virtualenv/x.py",
    "lib/python3.12/x.py",
    "lib64/x.py",
    "mathgen/lib/x.py",
    "pkg-1.0.dist-info/METADATA.txt",
    "a_gitleaks.toml.py",
    "d/gitleaks.toml/x.py",
    "sub/.gitleaks.toml",
]


@needs_shell
@pytest.mark.parametrize("path", REFUSED_PATHS)
def test_files_the_secret_scan_skips_by_name_or_path_are_refused(tmp_path, path):
    result = _refuse_files(tmp_path, {**BASE_FILES, path: "content\n"})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::" in result.stdout and path.rsplit("/", 1)[-1] in result.stdout


@needs_shell
def test_every_refused_file_is_reported_and_the_good_ones_are_not(tmp_path):
    result = _refuse_files(tmp_path, {**ALLOWED_TREE, "a.png": "x", "b/c.zip": "x"})
    assert result.returncode == 1
    errors = [line for line in result.stdout.splitlines() if line.startswith("::error::")]
    assert len(errors) == 2 and "a.png" in errors[0] and "c.zip" in errors[1]


@needs_shell
def test_a_submodule_is_refused(tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{base},deps.py")
    result = _refuse_files(tmp_path, {}, cwd=repo)
    assert result.returncode == 1
    assert "submodule" in result.stdout


@needs_shell
def test_an_empty_file_list_fails_instead_of_passing(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo, {})
    result = _refuse_files(tmp_path, {}, cwd=repo)
    assert result.returncode == 1


@needs_shell
def test_a_refused_path_cannot_forge_a_workflow_command(tmp_path):
    name = "x\n::notice::forged.png"
    result = _refuse_files(tmp_path, {**BASE_FILES, name: "x"})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert lines and all(line.startswith("::error::") for line in lines if line.startswith("::")), lines
    assert not any(line.startswith("::notice::") for line in lines)


@needs_shell
@pytest.mark.parametrize("base", ["0" * 40, "f" * 40, "", "not-a-sha"], ids=["zeros", "unknown", "empty", "junk"])
def test_gitleaks_fails_when_the_policy_commit_is_missing(tmp_path, base):
    repo = tmp_path / "repo"
    make_repo(repo, {**BASE_FILES, ".gitleaks.toml": "# base config\n"})
    result, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    assert result.returncode != 0, "a missing base commit must not fall back to the default rules"
    assert "missing" in result.stdout + result.stderr
    assert not (runner_temp / "calls.txt").exists(), "gitleaks ran although its policy commit could not be read"


@needs_shell
def test_gitleaks_opens_archives_in_the_final_tree_pass_only(tmp_path):
    repo = tmp_path / "repo"
    base = make_repo(repo)
    _, runner_temp = _scan(tmp_path, repo, EVENT="pull_request", BASE_SHA=base)
    history, tree = (call.split() for call in (runner_temp / "calls.txt").read_text().splitlines())
    assert history[0] == "git" and tree[0] == "dir"
    assert "--max-archive-depth" not in history
    assert tree[tree.index("--max-archive-depth") + 1] == "3"
