# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.0.3] - 2026-07-11

### Added
- Custom metric scripts: `goal_define(metric_script=...)` or the new
  `metric_register` tool attach a python file defining
  `metric(predictions: DataFrame) -> float`; the noise floor then bootstraps the
  REAL primary metric instead of falling back to accuracy. Field-fixed on the
  Higgs workspace: the AMS noise floor is now ±0.19 AMS (was an accuracy-unit
  ±0.003) — which also shows the plateau runs were statistically inevitable.
- Metric-choice guardrails at `goal_define`: a known metric that mismatches the
  task type is refused (e.g. rmse on classification); accuracy on imbalanced
  classification returns a `metric_advisory` (majority-class baseline already
  scores that high) without blocking.
- The MCP server now auto-serves the dashboard in-process and opens the user's
  browser on the first tool call; every tool response carries `dashboard_url`.
  Disable with `MLLOOP_NO_DASHBOARD=1`; port via `MLLOOP_DASHBOARD_PORT`
  (default 8137, scanning upward if taken).
- Dashboard redesigned as **The Lab Ledger**: morning-review reading order
  (state strip → stat tiles → lineage tree/journey chart → narrated overnight
  log → evidence rail with hypothesis claims, verdict, FE probes, and the data
  dictionary), run dossier drawer with deliverables and captioned figures,
  editorial paper/ink aesthetic with light + dark themes. Lineage is the
  default view; `/api/state` now includes context, fe_probes, events, and
  per-run noise floors.

### Fixed
- `report_generated` event payload key renamed to `report_kind` (collided with
  the event-kind key and garbled the event log).

## [0.0.2] - 2026-07-11

### Added
- `operating_curve` diagnostic: overkill (share of good flagged) vs catch rate
  (recall) with named operating points, plus degenerate-prediction detection — a
  constant score now reports "AUC is meaningless for this run" instead of hiding
  behind the number.
- `missed_positives` diagnostic: an out-of-fold reference model splits uncaught
  positives into feature-limited rows (they look like negatives to any model) vs
  learnable misses, with a SHAP beeswarm of what pulls them toward the negative
  class. Requires `shap` (now a dependency).
- `diagnose_run(refresh=True)` recomputes a stored diagnosis with the current
  battery.
- `context_register`: a domain-semantics ledger — record what each feature IS in
  domain terms (source and details included). Registered meanings annotate error
  slices in diagnostics, render as a data dictionary in both reports, and `status`
  nudges the agent to acquire domain context when coverage is zero.
- `fe_probe`: prices the feature-engineering opportunity before spending runs —
  screens difference/ratio/product combinations of the top numeric features
  (ranked by model permutation importance, so interaction-only features are not
  missed) plus stacked-model features (isolation-forest score, out-of-fold kNN),
  using paired fold-wise gains with a 3x-SEM multiple-testing bar. Verdicts:
  `fe_worth_testing` (with named candidates) or `fe_unlikely_to_help`.
- Reproducibility deliverables in the artifact contract: every run must now ship
  `train.*` (self-contained seeded training script), `infer.*` (scores unseen data
  from the shipped model), and `model.*` (the serialized model). `run_finish`
  records the sha256 of both scripts in the ledger and auto-generates
  `predictions.csv` from the parquet. `meta.json` gains a recommended
  `feature_importance` key.

### Fixed
- Forensics-kind runs are excluded from best-run selection: their metrics come
  from deliberately different (possibly contaminated) evaluation protocols.

## [0.0.1] - 2026-07-11

### Added — Phase 0: ledger & gates
- SQLite experiment ledger with append-only JSONL event mirror under `.mlloop/`.
- MCP server (stdio) with core tools: `goal_define`, `status`, `run_start`,
  `run_finish`, `run_abandon`, `hypothesis_register`, `hypothesis_resolve`,
  `decision_record`, `ledger_query`.
- Workflow gates enforced server-side: goal required before any run; first run must
  be a baseline; **every experiment must reference a registered falsifiable
  hypothesis**; one run open at a time; run budget (`policy.max_runs`); hypothesis
  resolution requires finished evidence runs that actually tested it.
- Artifact contract: `predictions.parquet` + `meta.json` validated at `run_finish`;
  invalid artifacts keep the run open with actionable errors.
- Dataset fingerprinting (rows, column schema, sha256) locked at `goal_define`.
- CLI: `mlloop serve | status | init`.

### Added — Phase 1: diagnostics & forensics
- `diagnose_run` battery: error slicing, bootstrap metric noise floor ("minimum
  believable improvement"), confusion analysis / residual analysis, calibration (ECE),
  class balance, overfit gap.
- Diagnosis gate: a finished run must be diagnosed before the next experiment starts
  (state `DIAGNOSE_PENDING`; disable with `policy.enforce_diagnosis = false`).
- `forensics_run` battery: shuffled-label signal check, label-noise estimation
  (Cleanlab confident learning) with suspect-row list, conflicting-duplicate
  irreducible-error bound, learning-curve plateau detection, per-feature mutual
  information with leakage warning, simple-model reference band — synthesized into a
  verdict: `no_signal` / `data_limited` / `more_data_needed` / `model_limited`.
- `report_generate`: self-contained HTML **Data Verdict Report** and experiment
  report (inline SVG charts, zero external assets).
- Stagnation tracking in `status` (`forensics_recommended` after
  `policy.forensics_after` consecutive non-improving runs, default 3).
- Agent integrations: `mlloop init --agent claude|opencode|codex|all` writes/prints
  MCP config for Claude Code, opencode, and Codex CLI (`docs/integrations.md`).
- CLI: `mlloop report --kind verdict|experiment`.

### Added — Phase 2: dashboard
- `mlloop dashboard`: local read-only web UI over the ledger (FastAPI + a single
  self-contained HTML page, no build chain, auto-refreshing every 5 s).
- Views: iteration tree (nodes colored by improvement vs parent beyond/below the
  metric direction, edges labeled with driving hypotheses, click-through to run
  detail), metric timeline, hypothesis board (kanban by status), decision log, and
  forensics list with verdict chips.
- Run detail drawer: metrics, model meta, diagnosis conclusions, and all diagnostic
  charts inline.
- `/verdict/{id}` renders the full Data Verdict Report directly from the ledger.
- Runs now carry a `diagnosed` flag in `ledger_query` output.
