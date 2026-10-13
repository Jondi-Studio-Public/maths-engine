# maths-engine-tutor

Maths practice with an AI agent. The agent finds a skill that fits what you are learning, sets one question at a
time, and has the engine mark each answer. It coaches the method after a wrong attempt and never gives the answer:
the server in this plugin has no way to return one.

## Skills

- `maths-tutoring`: the practice loop. One question at a time, marking through the engine, feedback from the
  misconception it names, stepping down to a prerequisite after repeated misses and up after a streak.
- `maths-skill-finder`: finding a skill by topic, school year, Australian Curriculum code, domain or answer kind.

## Tools (server role `learner`)

`engine_info`, `find_skills`, `get_skill`, `make_questions`, `get_question`, `grade_answer`, `build_paper` (JSON and
printable HTML worksheets, without answers). See `docs/mcp.md`.

Proofs and explanations cannot be marked by the engine; the agent says they need a person.

## Install

Needs [uv](https://docs.astral.sh/uv/) on your `PATH`. See [../README.md](../README.md) for Claude Code and other
clients.

## Licences

Code MIT, skills CC BY 4.0, senior content adapted from ACARA under CC BY 4.0.
