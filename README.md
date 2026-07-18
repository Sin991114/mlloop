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
  delta counts as evidence"), confusion/residuals, calibration, the operating curve
  (overkill vs catch rate, with degenerate-prediction detection), a SHAP explanation of
  missed positives (are they feature-limited or learnable?), and the overfit gap.
  Diagnosing the previous run is itself a gate: no diagnosis, no next experiment.
- **Data Verdict Report** — when runs stagnate, `forensics_run` interrogates the dataset
  with independent probes (shuffled-label signal check, confident-learning label-noise
  estimation, conflicting-duplicate bound, learning curve, per-feature signal) and
  `report_generate` renders a stakeholder-readable HTML verdict: is the ceiling set by
  the data or by the modeling? *Demo: inject 20% label noise into a clean dataset — the
  report catches it, quantifies it, and lists the suspect rows.*
- **Domain context** — `context_register` records what columns MEAN in domain terms
  (learned from dataset docs, domain MCP servers/skills, or the user); error slices and
  reports become domain-readable, and every report ships a data dictionary.
- **FE-opportunity probe** — `fe_probe` prices feature engineering before you spend runs
  on it: screens arithmetic combinations and stacked-model features for incremental
  signal with a paired, multiple-testing-adjusted significance bar. The probe generates
  hypotheses; the ledger tests them.
- **Ensemble probe & paired comparisons** — `ensemble_probe` prices combining finished
  runs with zero training (from their stored predictions); `compare_runs` and
  `run_finish` resolve small-but-real deltas with paired bootstrap significance on
  shared rows, far sharper than the single-run noise floor.
- **Exploration discipline** — stopping requires evidence (target met, high-confidence
  data-limited verdict, or budget exhaustion); until then `status` keeps the pressure on
  and stagnation suggests concrete pivots. Budgets cover both run count and wall-clock
  training time, and HPO sweeps are first-class runs.
- **Custom metrics** — a domain metric (AMS, weighted cost, ...) plugs in as a python
  file defining `metric(predictions) -> float` (`goal_define(metric_script=...)` or
  `metric_register`); the noise floor is then computed in the metric's real units.
  `goal_define` also refuses task-mismatched metrics and flags accuracy-on-imbalance
  with an advisory.
- **Dashboard** — "The Lab Ledger": lineage tree with hypothesis-labeled edges, metric
  journey with target line and noise-floor band, a narrated overnight log, evidence
  rail, and per-run dossiers — built for the morning-after review of an overnight
  autonomous session. The MCP server auto-opens it in your browser on the first tool
  call (`MLLOOP_NO_DASHBOARD=1` to disable); `mlloop dashboard` serves it manually.

Status: **Phase 2** — ledger, gates, diagnostics, forensics, reports, and dashboard.
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
