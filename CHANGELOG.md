# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
