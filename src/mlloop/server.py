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
4. Understand the data, not just the metrics: consult dataset docs, domain MCP servers,
   skills, or the user about what the columns MEAN, and record it with context_register —
   error slices and reports become domain-readable. Before investing runs in feature
   engineering, run fe_probe to price the opportunity.
5. hypothesis_register — a falsifiable explanation of what limits performance,
   derived from the diagnosis.
6. run_start(hypothesis_id=...) — experiments without a registered hypothesis are refused.
7. Each run must write the full artifact contract into its artifact_dir before run_finish
   (train.*, infer.*, model.*, predictions.parquet, meta.json).
8. hypothesis_resolve with evidence runs; decision_record each iteration's conclusion.
9. When runs stagnate (status reports forensics_recommended) or you suspect the data itself:
   forensics_run interrogates the dataset (label noise, signal check, conflict rate,
   learning curve), then report_generate(kind='verdict') produces the stakeholder report.
STOPPING REQUIRES EVIDENCE: keep hypothesizing until the target is met, a
high-confidence data-limited verdict lands, or the budget runs out — then record the
stop with decision_record citing that evidence. When runs stagnate, PIVOT before
giving up: ensemble_probe (price combining finished runs — zero training),
compare_runs (paired significance on shared rows), fe_probe, a genuinely different
model family, or forensics_run. "I ran out of ideas" is not a stop condition.
Call status anytime for the current state and allowed actions; ledger_query to recover
full context after a restart or context compaction.
"""


def resolve_workspace(workspace: str | None = None) -> Path:
    return Path(workspace or os.environ.get("MLLOOP_WORKSPACE") or Path.cwd()).resolve()


def create_server(workspace: str | None = None) -> FastMCP:
    ws = resolve_workspace(workspace)
    service = LedgerService(ws)
    mcp = FastMCP("mlloop", instructions=INSTRUCTIONS)
    dashboard = {"attempted": False, "url": None}

    def ensure_dashboard() -> None:
        """On first tool use, serve the dashboard in-process and pop the user's browser.

        Disable with MLLOOP_NO_DASHBOARD=1 (e.g. headless/CI). Port from
        MLLOOP_DASHBOARD_PORT (default 8137), scanning upward if taken.
        """
        if dashboard["attempted"] or os.environ.get("MLLOOP_NO_DASHBOARD"):
            dashboard["attempted"] = True
            return
        dashboard["attempted"] = True
        import socket
        import threading
        import webbrowser

        base_port = int(os.environ.get("MLLOOP_DASHBOARD_PORT", "8137"))
        chosen = None
        for port in range(base_port, base_port + 20):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                    chosen = port
                    break
                except OSError:
                    continue
        if chosen is None:
            return

        def serve() -> None:
            import uvicorn

            from .dashboard import create_app

            config = uvicorn.Config(
                create_app(str(ws)), host="127.0.0.1", port=chosen, log_level="error"
            )
            # uvicorn skips signal handlers off the main thread; safe as a daemon.
            uvicorn.Server(config).run()

        threading.Thread(target=serve, daemon=True, name="mlloop-dashboard").start()
        dashboard["url"] = f"http://127.0.0.1:{chosen}"
        try:
            webbrowser.open(dashboard["url"])
        except Exception:
            pass

    def call(fn, **kwargs) -> str:
        ensure_dashboard()
        try:
            result = fn(**kwargs)
        except GateError as exc:
            result = {"ok": False, "refused": True, "error": str(exc)}
        if isinstance(result, dict) and dashboard["url"]:
            result.setdefault("dashboard_url", dashboard["url"])
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
        metric_script: str | None = None,
    ) -> str:
        """Define and LOCK the project goal. Must be called before any run.

        task_type is 'classification' or 'regression'. dataset_path points to a
        csv/tsv/parquet file, fingerprinted at definition time. metric_direction is
        'maximize' or 'minimize'. policy sets the run budget, e.g. {"max_runs": 30}.
        The primary metric and dataset cannot be changed afterwards. For a custom
        metric (not in the built-in registry) pass metric_script: a python file
        defining `metric(predictions: DataFrame) -> float` so the noise floor is
        computed in the metric's real units. The response may include a
        metric_advisory when the metric choice looks like a known trap.
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
            metric_script=metric_script,
        )

    @mcp.tool()
    def metric_register(script_path: str) -> str:
        """Attach a custom metric script to the locked goal (the metric NAME never changes;
        this only teaches MLLoop how to COMPUTE it).

        The script defines `metric(predictions: DataFrame) -> float`; predictions is the
        run's predictions frame (row_id, y_true, y_pred, proba_*, plus any extra columns
        your training code shipped, e.g. per-row weights). After registering, re-run
        diagnose_run(..., refresh=True) so noise floors are recomputed in real units.
        """
        return call(service.metric_register, script_path=script_path)

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
    def context_register(
        feature: str, meaning: str, source: str | None = None, details: dict | None = None
    ) -> str:
        """Record domain semantics for a feature: what the column IS in domain terms.

        Use whenever you learn what a column means — from dataset docs, domain MCP
        servers, skills, or the user (e.g. 'die_x: horizontal die position on the
        wafer; edge dies have higher defect rates'). Registered semantics annotate
        error slices in diagnostics and appear as a data dictionary in reports.
        source: where the knowledge came from. details: units, generation process,
        known effects. Engineered-feature names are allowed.
        """
        return call(
            service.context_register, feature=feature, meaning=meaning, source=source, details=details
        )

    @mcp.tool()
    def compare_runs(run_a: str, run_b: str) -> str:
        """Paired comparison of two finished runs on their shared held-out rows.

        Shared-row pairing cancels the common variance, so far smaller deltas are
        resolvable than the single-run noise floor suggests. Use before resolving a
        hypothesis when the raw delta looks small — 'indistinguishable' and 'small but
        real' are different verdicts.
        """
        return call(service.compare_runs, run_a=run_a, run_b=run_b)

    @mcp.tool()
    def ensemble_probe(run_ids: list[str] | None = None) -> str:
        """Price ensembling with ZERO training: average the stored held-out predictions of
        finished runs on their shared rows and score against the best single run — paired,
        on identical rows.

        Defaults to every finished baseline/experiment run; pass run_ids to choose members.
        Verdict ensemble_worth_testing -> register an ensembling hypothesis and confirm with
        a real run (seed bagging, diverse families, stacking). ensemble_unlikely_to_help
        usually means the members are too correlated — diversify model families first.
        """
        return call(service.ensemble_probe, run_ids=run_ids)

    @mcp.tool()
    def fe_probe(top_k: int = 5, quick: bool = False) -> str:
        """Price the feature-engineering opportunity BEFORE spending runs on it.

        Screens arithmetic combinations (difference/ratio/product of the top numeric
        features) and stacked-model features (isolation-forest score, out-of-fold kNN
        prediction) for incremental cross-validated signal, with a multiple-testing
        adjusted significance bar. Verdict fe_worth_testing -> register a hypothesis
        for the named candidate and confirm with a run; fe_unlikely_to_help -> spend
        the budget elsewhere. The probe generates hypotheses, never results.
        """
        return call(service.fe_probe, top_k=top_k, quick=quick)

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
