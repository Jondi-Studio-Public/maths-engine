# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue or pull
request. Reports go through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**.

If that button is not there (private reporting is a repository setting, and it
can be off), open a minimal public issue titled "Security contact request".
Put no details in it: no description of the problem, no affected file, no
payload. The maintainer will answer in the issue with a private way to send the
report.

Include what you found, how to reproduce it, and the version or commit. You
can expect an acknowledgement within 7 days. This is a small project run by
one person, so fixes are made on a best-effort basis, and you will be told
when one is released.

## Scope

In scope:

- Code execution, unsafe deserialisation or path traversal when loading skill
  YAML or running the CLI (`python -m mathgen`).
- Code execution through answer text: an answer string, typed or spoken, that
  makes a grader run code, read or write files, or reach the network, whether
  it arrives through the Python API (`mathgen.engine.grade`), `python -m
  mathgen --answer`, or any other entry point.
- Denial of service from a crafted answer string, for example a grader that
  hangs or exhausts memory on adversarial input.
- Secrets or personal data accidentally committed to this repository.
- Problems in the CI workflows, such as a token with more access than it needs.

Out of scope:

- A generated question with a wrong answer, or a skill with a wrong
  description. Please open a normal issue for those.
- Vulnerabilities in dependencies (SymPy, NumPy, PyYAML). Report those
  upstream.
- Anything about a website or app built on top of this package. This
  repository contains no website, server or user data.

Only the latest release and the `main` branch receive fixes.
