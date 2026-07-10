# MLLoop

**A scientific-method harness for AI-driven machine learning.**

Coding agents (Claude Code, opencode, ...) can already write training code and run ten
variants overnight. What they don't do by themselves is science: diagnose *why* a model
underperforms, form falsifiable hypotheses, run discriminating experiments, and — when
the data itself is the problem — produce evidence strong enough to convince stakeholders.

MLLoop is an MCP server that sits between the agent and your training code and enforces
that loop at the tool layer, not via prompts:

- **Experiment ledger** — every run, hypothesis, and decision recorded in SQLite plus an
  append-only JSONL event log, all under `.mlloop/` in your project.
- **Hypothesis gate** — `run_start` refuses any experiment that doesn't test a registered,
  falsifiable hypothesis. No hypothesis, no run.
- **Artifact contract** — each run writes standardized `predictions.parquet` + `meta.json`;
  diagnostics never read your training code, so any framework works.
- **Diagnostics battery** — after every run: error slices, bootstrap noise floor ("what
  delta counts as evidence"), confusion/residuals, calibration, overfit gap. Diagnosing
  the previous run is itself a gate: no diagnosis, no next experiment.
- **Data Verdict Report** — when runs stagnate, `forensics_run` interrogates the dataset
  with independent probes (shuffled-label signal check, confident-learning label-noise
  estimation, conflicting-duplicate bound, learning curve, per-feature signal) and
  `report_generate` renders a stakeholder-readable HTML verdict: is the ceiling set by
  the data or by the modeling? *Demo: inject 20% label noise into a clean dataset — the
  report catches it, quantifies it, and lists the suspect rows.*
- **Dashboard** *(Phase 2)* — iteration tree, hypothesis board, and metric trajectory for
  the morning-after review of an overnight autonomous session.

Status: **Phase 1** — ledger, gates, diagnostics, forensics, and reports all working.
Full design: [DESIGN.md](DESIGN.md).
Agent setup (Claude Code / opencode / Codex): [docs/integrations.md](docs/integrations.md).

## Quickstart

```bash
pip install -e .
cd your-ml-project
mlloop init --agent claude    # or opencode / codex / all — writes the MCP config
```

Then tell your agent to train a model. The enforced workflow:

| Step | Tool | Gate |
|---|---|---|
| 1 | `goal_define` | Locks dataset, target column, primary metric. Required first. |
| 2 | `run_start(kind='baseline')` | First run must be a simple baseline. |
| 3 | `diagnose_run` | Every finished run must be diagnosed before the next experiment. |
| 4 | `hypothesis_register` | Falsifiable claim about what limits performance, from the diagnosis. |
| 5 | `run_start(hypothesis_id=...)` | Refused without a registered hypothesis. |
| 6 | `run_finish` | Validates the artifact contract before accepting results. |
| 7 | `hypothesis_resolve` / `decision_record` | Evidence-backed resolution, recorded decisions. |
| 8 | `forensics_run` → `report_generate` | When stagnating: interrogate the data, render the verdict. |

`status` shows the current state and allowed actions at any time; `ledger_query` restores
full context after an agent restart or context compaction.

## Contributing

Issues, design feedback, and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Please note the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[Apache-2.0](LICENSE)
