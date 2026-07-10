# Contributing to MLLoop

Thanks for your interest in MLLoop! This project is young and moving fast — issues,
design feedback, and pull requests are all welcome.

## Development setup

```bash
git clone <repo-url> mlloop && cd mlloop
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Requires Python ≥ 3.11. The test suite is self-contained (synthetic datasets, temp
directories) and should pass in under a minute.

## Project layout

```
src/mlloop/
  ledger.py       # SQLite ledger + JSONL event log
  service.py      # ALL workflow gates live here — the heart of the project
  artifacts.py    # artifact contract validation, dataset fingerprinting
  diagnostics.py  # post-run diagnostics battery
  forensics.py    # data forensics battery + verdict synthesis
  metrics.py      # metric registry, bootstrap noise floor
  report.py       # self-contained HTML reports
  server.py       # MCP tool wrappers (thin — no logic here)
  cli.py          # mlloop serve / status / init / report
tests/            # pytest suite; conftest.py has the shared fixtures
DESIGN.md         # the design document — read this first
```

## Ground rules

- **Gates live in `service.py`, not in prompts.** The core thesis of MLLoop is that
  workflow rules are enforced at the tool layer. If your change relies on the agent
  "behaving", it belongs in documentation, not code.
- **Refusal messages teach.** Every `GateError` message must tell the agent what to do
  instead. Write them as instructions, not complaints.
- **Diagnostics consume artifacts, never training code.** New diagnostics must work
  from `predictions.parquet` + `meta.json` + the goal dataset alone.
- **Every gate gets a test.** New rules in `service.py` need a test in
  `tests/test_gates.py` (or a new file) proving both the refusal and the happy path.
- **Keep the server thin.** `server.py` maps tools to service calls and formats
  errors — nothing else.

## Pull requests

1. Fork, branch from `main`, keep changes focused.
2. `pytest` must pass; add tests for new behavior.
3. Update `CHANGELOG.md` under *Unreleased*.
4. If you change the workflow, tools, or artifact contract, update `DESIGN.md` and
   `README.md` accordingly.

## Reporting issues

Include: what you asked the agent to do, the tool call and response (from
`.mlloop/events.jsonl`), and your Python/OS versions. For data-dependent bugs a
minimal synthetic dataset that reproduces the issue is gold.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
