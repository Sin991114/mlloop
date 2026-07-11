# MLLoop — Design Document v0.1

> Project name **MLLoop** (named 2026-07-10), PyPI package `mlloop` (verified available).
>
> One-line positioning: **a harness that forces AI coding agents to practice the
> scientific method while training ML models** — experiment ledger + diagnostics
> toolbox + data forensics report + iteration visualization, delivered as an MCP
> server that plugs into Claude Code, opencode, Codex, or any MCP-capable agent.

---

## 1. Problem statement

Data scientists can already have a coding agent run ten training variants overnight,
but two gaps remain:

**Gap A: mechanical iteration, no diagnosis.**
Agents execute a pre-written list of plans in order. They do not study *why* the model
underperformed after iteration 1, form hypotheses, design discriminating experiments,
and plan iteration 2 from the conclusions. A plan list is open-loop; the scientific
method is closed-loop.

**Gap B: when the data is guilty, there is no evidence.**
The worst case is wrong labels, or features that simply do not encode the target. The
model stalls, and the data scientist has no standardized, persuasive body of evidence
to tell the boss or the client "the problem is the data, not the modeling." This is a
political problem, not just a technical one.

**Key insight: neither gap should be solved by hoping the LLM behaves — both should be
enforced by a harness.** LLMs are perfectly capable of error analysis and hypothesis
generation; what they lack is a workflow structure that forces them to do it, plus
ready-made diagnostics that make "doing the diagnosis" cheaper than skipping it.

## 2. Ecosystem positioning

| Existing tool | What it does | What it lacks |
|---|---|---|
| AIDE, MLE-agent, other agentic ML | enumerate plans → execute → pick best | no diagnose→hypothesize→verify loop; opaque process |
| W&B / MLflow | experiment tracking, metric logging | records *what ran*, never *why it ran or whether the hypothesis held* |
| Cleanlab | label-noise detection | a point tool — no report, no place in an iteration loop |
| AutoML (AutoGluon etc.) | automated tuning and model selection | fully opaque; with bad data it just hands you a bad score |

**MLLoop's layer: the scientific-method layer between the agent and the training
code.** The agent still writes code and runs training (what it is good at); MLLoop:

1. forces every experiment to hang off a falsifiable hypothesis (a forcing function);
2. provides standardized diagnostics so one tool call yields error slices / learning
   curves / label-noise estimates;
3. records the whole iteration as a structured ledger, rendered as a human-readable
   experiment tree;
4. when the model stalls, produces a stakeholder-facing **Data Verdict Report** with
   one command.

## 3. Core concepts

| Concept | Description |
|---|---|
| **Goal Spec** | task definition: task type, dataset fingerprint, evaluation metric, target value, budget (time / run count), constraints (metric immutable, data immutable, etc.) |
| **Run** | one train+evaluate cycle. Must produce standardized artifacts (see §6) |
| **Hypothesis** | falsifiable: statement + rationale + prediction ("if H holds, experiment X should observe Y") + status (open / testing / confirmed / refuted / inconclusive) |
| **Diagnosis** | structured result of running the diagnostics battery on a Run |
| **Decision Record** | end-of-iteration decision: which evidence, which path, what was abandoned |
| **Iteration Tree** | Runs connected parent→child into a tree/DAG, edges annotated with the driving hypothesis |
| **Data Verdict Report** | output of the forensics battery, an evidence report written for non-technical readers |

## 4. Core loop state machine

```
GOAL_INTAKE ──► BASELINE ──► DIAGNOSE ──► HYPOTHESIZE ──► PLAN ──► EXECUTE
                                 ▲                                    │
                                 │            EVALUATE ◄──────────────┘
                                 │                │
                                 └── continue ◄── DECIDE ──► done ──► FINAL_REPORT
                                                    │
                                                    └──► data_suspect ──► FORENSICS ──► VERDICT_REPORT
```

Rules per state (enforced by the MCP server at the tool layer, not requested via
prompts):

- **GOAL_INTAKE**: `goal_define` is required before any Run. The metric and dataset
  fingerprint are locked here; changing them later requires an explicit
  `decision_record` (prevents the agent from quietly swapping metrics to "improve" the
  score).
- **BASELINE**: the first Run must be a simple baseline (majority class / linear model /
  shallow tree) — the anchor for every later comparison.
- **DIAGNOSE**: after a Run finishes, `diagnose_run` must be called before the next Run
  can start. Results are written to the ledger.
- **HYPOTHESIZE**: every new Run (baseline excepted) must reference an open Hypothesis.
  No hypothesis → `run_start` refuses. This is the single most important gate in the
  system.
- **DECIDE**: resolving a hypothesis (confirmed/refuted/inconclusive) requires evidence
  (references to specific Runs and Diagnoses). Whether an improvement is significant is
  judged against the noise floor (§7) — noise must not be reported as progress.
- **FORENSICS escalation** (any trigger recommends, repeated triggers force):
  - N consecutive iterations (default 3) with no significant improvement;
  - diagnostics estimate label noise above threshold;
  - the best model is barely better than the shuffled-label baseline.

## 5. MCP server interface

Server name: `mlloop`. Tools grouped by responsibility:

**Ledger (write)**
| Tool | Responsibility |
|---|---|
| `goal_define` | define/lock the Goal Spec |
| `run_start` | declare a Run (must reference a hypothesis_id, baseline excepted); returns run_id and the artifact output path |
| `run_finish` | submit Run results; validates artifact-contract completeness |
| `hypothesis_register` | register a hypothesis (statement, rationale, falsifiable prediction, test design) |
| `hypothesis_resolve` | close a hypothesis, evidence required |
| `decision_record` | record an iteration decision |

**Diagnostics (compute)**
| Tool | Responsibility |
|---|---|
| `diagnose_run` | run the diagnostics battery (§7) on a Run; returns structured results + chart files |
| `forensics_run` | run the data forensics battery (§8) |

**Query (read)**
| Tool | Responsibility |
|---|---|
| `ledger_query` | query the ledger: iteration history, hypothesis board, metric trajectory (how the agent recovers context after compaction or a new session) |
| `status` | current state-machine position, budget remaining, allowed next actions |

**Reports**
| Tool | Responsibility |
|---|---|
| `report_generate` | generate the final experiment report or the Data Verdict Report (self-contained shareable HTML) |

A companion prompt template (docs/integrations.md) teaches the agent the workflow —
but correctness never depends on the agent behaving: the gates live server-side.

## 6. Artifact contract

Diagnostics never read the user's training code; they consume standardized artifacts
only. `run_start` returns a directory path; `run_finish` validates that these files
exist with the correct schema:

```
runs/<run_id>/
  predictions.parquet    # required: row_id (0-based row index into the goal dataset),
                         #           y_true, y_pred, (classification) proba_*
  cv_predictions.parquet # recommended: out-of-fold predictions, for label-noise and
                         #              learning-curve diagnostics
  meta.json              # required: model description, hyperparams, feature list,
                         #           training time, random seed
  model.pkl              # optional
```

The dataset itself is registered at `goal_define` (path + fingerprint: row count,
column schema, content hash); the forensics battery reads the raw data directly.

**This contract is the foundation of the whole diagnostics layer**: as long as the
agent can write predictions in this format, MLLoop does not care whether it used
sklearn, XGBoost, or anything else.

## 7. Diagnostics battery (tabular v1)

`diagnose_run` runs everything by default; each item yields a structured conclusion
plus a chart (stored with the run, reused by reports and the dashboard):

1. **Error slicing** — bucket by each feature, find the worst-performing segments,
   report the top-k worst slices;
2. **Learning curve** — fit on training-set subsamples and extrapolate: direct evidence
   for "would more data help";
3. **Confusion analysis** — classification: confusion matrix + most-confused pairs;
   regression: residuals vs prediction/features;
4. **Calibration** — reliability curve + ECE;
5. **Overfit gap** — train/validation gap (requires the agent to report
   `train_<metric>` in `run_finish`);
6. **Noise floor** — bootstrap resampling of the prediction rows yields the metric's
   variance and a "minimum believable improvement"; all later run comparisons are
   judged against it (multi-seed reruns refine it);
7. **Feature-importance stability** — permutation importance variance across folds
   *(deferred: requires model access)*;
8. **Leakage heuristics** — suspiciously high single-feature AUC, duplicate rows across
   splits (single-feature check implemented in the forensics battery).

## 8. Data forensics battery (Data Verdict Report) — the killer feature

Answers one question: **"is the ceiling set by the data, or by the modeling?"** Each
line of evidence is independent and cross-checkable:

| Probe | Question answered | Evidence produced |
|---|---|---|
| **Label-noise estimate** (Cleanlab confident learning) | how many labels are probably wrong? | estimated noise rate + list of most-suspect rows (show them to people) |
| **Shuffled-label baseline** | is there any real signal at all? | real model vs model trained on permuted labels; gap ≈ 0 means no signal |
| **Conflicting-sample rate** | how large is the irreducible error? | fraction of identical/near-identical feature rows with different labels — a hard lower bound on error |
| **Learning-curve extrapolation** | not enough data, or ceiling reached? | plateau detection + trend |
| **Feature signal** | do the features discriminate at all? | per-feature mutual information / single-feature AUC + leakage warning |
| **Simple-model reference band** | how much headroom does fancy modeling have? | quick kNN / linear / GBM cross-validated scores vs current best |

Output is a **self-contained HTML report** for non-technical readers: each line of
evidence gets a plain-language conclusion + confidence + chart, synthesized into a
verdict (e.g. "an estimated 18% of labels are wrong (high confidence); the current
metric is at 94% of the ceiling implied by the conflict rate; switching models is
expected to gain <1 point; prioritize re-annotating class X"). **This report is the
ammunition the data scientist takes into the meeting.**

## 9. Dashboard (local web, Phase 2)

`mlloop dashboard` starts a local server, read-only over the ledger (SQLite). Core
views:

1. **Iteration Tree** — the main view. Nodes = Runs (color = significant improvement /
   flat / regression vs parent), edges = driving hypotheses. Click for details. This is
   the "first thing you look at the morning after";
2. **Hypothesis Board** — kanban: open / testing / confirmed / refuted, each card
   linking to its evidence;
3. **Metrics Timeline** — primary-metric trajectory across iterations with the
   noise-floor band;
4. **Run Detail** — all diagnostics charts for a single run;
5. **Verdict Viewer** — rendered forensics report + export.

Morning-review experience target: *"Overnight: 12 experiments across 4 hypotheses. H1
confirmed (class imbalance dominates; reweighting +2.1 AUC, beyond the noise band). H2
refuted. H3 inconclusive — insufficient data. Two consecutive non-improving iterations
auto-triggered forensics: estimated label noise 12% — review the suspect-label list
first thing today."*

## 10. Overnight autonomy policy

The user pre-authorizes via the `policy` field of `goal_define`; the server enforces:

- **Budget**: max runs / max wall-clock / per-run timeout;
- **Escalation rules**: when the forensics battery fires automatically (default: 3
  consecutive non-improving runs);
- **Red lines (never without human approval)**: changing the evaluation metric,
  deleting data rows, editing labels, using the test set for any training decision;
- **Stop conditions**: target reached / budget exhausted / forensics verdict
  "data-limited" with high confidence → stop and leave a message instead of burning
  budget pointlessly.

## 11. Technology choices

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python ≥ 3.11 | where the ML ecosystem lives |
| MCP | official `mcp` Python SDK (stdio server) | supported by Claude Code, opencode, Codex |
| Ledger | SQLite (source of truth) + JSONL event log (auditable/replayable) | zero deployment, single copyable file |
| Diagnostics deps | scikit-learn, cleanlab, pandas, pyarrow | mature open source |
| Charts | matplotlib → inline SVG | zero-JS, truly self-contained reports (see changelog) |
| Dashboard | FastAPI + lightweight frontend (htmx/vanilla first) | no frontend build chain for the MVP |
| Reports | self-contained HTML (inline SVG) | one file you can send to the boss |
| License | Apache-2.0 | open source + patent-clause friendly |

## 12. Milestones

- **Phase 0 — ledger & gates** *(done)*: SQLite schema; `goal_define` / `run_start` /
  `run_finish` / `hypothesis_*` / `decision_record` / `ledger_query` / `status`;
  artifact-contract validation; state-machine gates. Acceptance: a full
  baseline→diagnose→hypothesize→experiment→resolve cycle in Claude Code with a complete
  ledger.
- **Phase 1 — diagnostics & forensics** *(done)*: the full §7 battery, the §8 battery,
  HTML reports. Acceptance (**flagship demo**): take a clean public tabular dataset,
  inject 20% label noise; the verdict report must catch it and estimate a rate close to
  20% — this demo goes at the top of the README.
- **Phase 2 — dashboard** *(done)*: the five views, read-only over the ledger.
- **Phase 3 — polish & ecosystem**: full overnight policy, opencode/Codex adapter
  verification, PyPI release.

## 13. Open questions

1. ~~Project name and PyPI package~~ → **decided: MLLoop / `mlloop`** (PyPI verified
   available 2026-07-10);
2. Hypothesis-gate strictness: hard-refuse vs warn-and-allow when `run_start` has no
   hypothesis (current: hard refusal; escape hatch via policy);
3. Some forensics probes train models themselves (e.g. shuffled-label) — runtime
   control on large datasets (current: row subsampling, default cap 20k);
4. Multi-objective goals (v1: one primary metric + monitor metrics only);
5. The repository directory is still named `RogueGames`; rename after adoption.

---

## Changelog

**2026-07-10 — Phase 0 + Phase 1 implemented.** Deviations from the original design:

- **Charts**: originally vega-lite specs (§7/§11); Phase 1 ships matplotlib-generated
  inline SVG instead (zero-JS, truly self-contained reports). Vega-lite will be
  re-evaluated for the Phase 2 dashboard.
- **Noise floor**: §7's multi-seed rerun variance is implemented as bootstrap
  resampling in Phase 1 (`diagnose_run` cannot retrain models itself); multi-seed
  reruns remain an agent-side complement.
- **Diagnosis gate live**: a finished run must be diagnosed before the next experiment
  starts (state `DIAGNOSE_PENDING`); disable with policy
  `{"enforce_diagnosis": false}`.
- **Forensics trigger**: the consecutive-non-improving-run threshold is policy
  `forensics_after` (default 3); `status` surfaces `forensics_recommended` (advisory in
  Phase 1, not forced).
- **Overfit gap**: requires the agent to report `train_<primary_metric>` alongside the
  validation metric in `run_finish`.
