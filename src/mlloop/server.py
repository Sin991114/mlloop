"""MLLoop MCP server — thin tool wrappers over LedgerService.

All workflow gates live in service.py; this module only maps MCP tools to
service calls and turns GateError refusals into structured JSON the agent
can act on instead of a raw exception.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .service import GateError, LedgerService

INSTRUCTIONS = """\
MLLoop is a scientific-method harness for ML experimentation. Enforced workflow:
1. goal_define — locks dataset, target column and primary metric. Required before anything else.
2. run_start(kind='baseline') — the first run must be a simple baseline.
3. diagnose_run — every finished run must be diagnosed before the next experiment;
   study the failure modes (error slices, noise floor, calibration, confusion).
4. hypothesis_register — a falsifiable explanation of what limits performance,
   derived from the diagnosis.
5. run_start(hypothesis_id=...) — experiments without a registered hypothesis are refused.
6. Each run must write predictions.parquet + meta.json into its artifact_dir before run_finish.
7. hypothesis_resolve with evidence runs; decision_record each iteration's conclusion.
8. When runs stagnate (status reports forensics_recommended) or you suspect the data itself:
   forensics_run interrogates the dataset (label noise, signal check, conflict rate,
   learning curve), then report_generate(kind='verdict') produces the stakeholder report.
Call status anytime for the current state and allowed actions; ledger_query to recover
full context after a restart or context compaction.
"""


def resolve_workspace(workspace: str | None = None) -> Path:
    return Path(workspace or os.environ.get("MLLOOP_WORKSPACE") or Path.cwd()).resolve()


def create_server(workspace: str | None = None) -> FastMCP:
    service = LedgerService(resolve_workspace(workspace))
    mcp = FastMCP("mlloop", instructions=INSTRUCTIONS)

    def call(fn, **kwargs) -> str:
        try:
            result = fn(**kwargs)
        except GateError as exc:
            result = {"ok": False, "refused": True, "error": str(exc)}
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @mcp.tool()
    def goal_define(
        task_type: str,
        dataset_path: str,
        target_column: str,
        primary_metric: str,
        metric_direction: str,
        target_value: float | None = None,
        monitor_metrics: list[str] | None = None,
        constraints: dict | None = None,
        policy: dict | None = None,
    ) -> str:
        """Define and LOCK the project goal. Must be called before any run.

        task_type is 'classification' or 'regression'. dataset_path points to a
        csv/tsv/parquet file, fingerprinted at definition time. metric_direction is
        'maximize' or 'minimize'. policy sets the run budget, e.g. {"max_runs": 30}.
        The primary metric and dataset cannot be changed afterwards.
        """
        return call(
            service.goal_define,
            task_type=task_type,
            dataset_path=dataset_path,
            target_column=target_column,
            primary_metric=primary_metric,
            metric_direction=metric_direction,
            target_value=target_value,
            monitor_metrics=monitor_metrics,
            constraints=constraints,
            policy=policy,
        )

    @mcp.tool()
    def status() -> str:
        """Current workflow state, allowed next actions, budget, and best run so far.

        Call this whenever unsure what to do next, or after a restart.
        """
        return call(service.status)

    @mcp.tool()
    def hypothesis_register(
        statement: str, rationale: str, prediction: str, test_plan: str
    ) -> str:
        """Register a falsifiable hypothesis about WHY performance is limited.

        statement: the claim (e.g. 'minority-class label noise dominates the error').
        rationale: why the evidence so far supports suspecting this.
        prediction: what a discriminating experiment should observe if the claim is true —
        must be falsifiable, ideally with a magnitude.
        test_plan: the cheapest experiment that can falsify the claim.
        Every experiment run must reference a registered hypothesis.
        """
        return call(
            service.hypothesis_register,
            statement=statement,
            rationale=rationale,
            prediction=prediction,
            test_plan=test_plan,
        )

    @mcp.tool()
    def hypothesis_resolve(
        hypothesis_id: str,
        resolution: str,
        evidence_run_ids: list[str],
        narrative: str,
    ) -> str:
        """Resolve a hypothesis with evidence: 'confirmed', 'refuted', or 'inconclusive'.

        evidence_run_ids must contain at least one finished run that was started with
        this hypothesis_id. narrative states the reasoning in one or two sentences.
        """
        return call(
            service.hypothesis_resolve,
            hypothesis_id=hypothesis_id,
            resolution=resolution,
            evidence_run_ids=evidence_run_ids,
            narrative=narrative,
        )

    @mcp.tool()
    def run_start(
        intent: str,
        kind: str = "experiment",
        hypothesis_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        """Open a new run. Returns run_id, the artifact_dir to write into, and the contract.

        kind: 'baseline' (first run, no hypothesis), 'experiment' (requires
        hypothesis_id of an open hypothesis), or 'forensics'. intent: one sentence on
        what this run changes and why. parent_run_id defaults to the last finished run.
        Only one run may be open at a time.
        """
        return call(
            service.run_start,
            intent=intent,
            kind=kind,
            hypothesis_id=hypothesis_id,
            parent_run_id=parent_run_id,
        )

    @mcp.tool()
    def run_finish(run_id: str, metrics: dict, notes: str | None = None) -> str:
        """Close a run: validates the artifact contract, records metrics, compares to parent/best.

        metrics must include the locked primary metric. If the artifacts are invalid the
        run stays open and the response lists what to fix. Deltas returned are raw —
        treat small ones as noise until Phase 1 adds significance testing.
        """
        return call(service.run_finish, run_id=run_id, metrics=metrics, notes=notes)

    @mcp.tool()
    def run_abandon(run_id: str, reason: str) -> str:
        """Abandon a run that failed or is no longer worth finishing. reason is recorded."""
        return call(service.run_abandon, run_id=run_id, reason=reason)

    @mcp.tool()
    def decision_record(
        summary: str, evidence: dict | None = None, next_action: str | None = None
    ) -> str:
        """Record an iteration decision: what was decided, on what evidence, what happens next.

        Use after resolving hypotheses, when changing direction, or when stopping.
        """
        return call(
            service.decision_record, summary=summary, evidence=evidence, next_action=next_action
        )

    @mcp.tool()
    def diagnose_run(run_id: str, refresh: bool = False) -> str:
        """Run the diagnostics battery on a finished run: error slices, metric noise floor,
        confusion/residuals, calibration, operating curve (overkill vs catch rate, with
        degenerate-prediction detection), missed-positive SHAP explanation
        (feature-limited vs learnable misses), class balance, overfit gap.

        Required after every finished run before the next experiment can start. The noise
        floor tells you the minimum delta that counts as evidence rather than noise.
        Hypotheses should be derived from the dominant failure mode found here.
        refresh=True recomputes a stored diagnosis with the current battery.
        """
        return call(service.diagnose_run, run_id=run_id, refresh=refresh)

    @mcp.tool()
    def forensics_run(quick: bool = False) -> str:
        """Interrogate the DATASET itself: is the ceiling set by the data or the modeling?

        Runs independent probes — shuffled-label signal check, label-noise estimation
        (confident learning), conflicting-duplicate rate, learning curve, per-feature
        signal, quick reference models — and synthesizes a verdict. Use when runs
        stagnate or you suspect labels/features. quick=True trades precision for speed.
        Trains small internal reference models; does not touch the run ledger.
        """
        return call(service.forensics_run, quick=quick)

    @mcp.tool()
    def report_generate(kind: str = "verdict", output_path: str | None = None) -> str:
        """Generate a self-contained HTML report.

        kind='verdict': the stakeholder-facing Data Verdict Report from the latest
        forensics results — the evidence that the data (or the modeling) is the problem.
        kind='experiment': the full iteration story — runs, hypotheses, decisions,
        metric trajectory.
        """
        return call(service.report_generate, kind=kind, output_path=output_path)

    @mcp.tool()
    def ledger_query(view: str = "summary", run_id: str | None = None, limit: int = 20) -> str:
        """Query the experiment ledger. Views: 'summary' (default), 'runs', 'run' (needs
        run_id), 'hypotheses', 'decisions', 'events', 'diagnosis' (needs run_id),
        'forensics'. Use to recover context after a restart or to study history before
        forming a new hypothesis."""
        return call(service.ledger_query, view=view, run_id=run_id, limit=limit)

    return mcp
