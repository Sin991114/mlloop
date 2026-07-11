"""Read-only local dashboard over the experiment ledger (DESIGN.md §9).

Serves a single self-contained HTML page plus a small JSON API. Never writes
to the ledger — the agent owns all writes through the MCP tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .service import GateError, LedgerService

SAFE_SVG = re.compile(r"^[A-Za-z0-9_\-]+\.svg$")
RUN_ID = re.compile(r"^R\d+$")
FORENSICS_ID = re.compile(r"^F\d+$")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(workspace: str | None = None) -> FastAPI:
    from .server import resolve_workspace

    service = LedgerService(resolve_workspace(workspace))
    app = FastAPI(title="MLLoop Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state() -> dict:
        status = service.status()
        payload = {
            "workspace": str(service.ledger.workspace),
            "status": status,
            "runs": [],
            "hypotheses": [],
            "decisions": [],
            "forensics": [],
            "context": [],
            "fe_probes": [],
            "events": [],
        }
        if status["state"] == "NEED_GOAL":
            return payload
        payload["runs"] = service.ledger_query(view="runs")["runs"]
        payload["hypotheses"] = service.ledger_query(view="hypotheses")["hypotheses"]
        payload["decisions"] = service.ledger_query(view="decisions")["decisions"]
        payload["forensics"] = service.ledger_query(view="forensics")["forensics"]
        payload["context"] = service.ledger_query(view="context")["feature_context"]
        payload["fe_probes"] = service.ledger_query(view="fe_probes")["fe_probes"]
        payload["events"] = service.ledger_query(view="events", limit=80)["events"]
        # Per-run noise floor (minimum believable improvement) from its diagnosis.
        with service.ledger.connect() as con:
            rows = con.execute("SELECT run_id, results FROM diagnoses").fetchall()
        floors = {}
        for row in rows:
            details = json.loads(row["results"])["items"].get("noise_floor", {}).get("details", {})
            if details.get("min_significant_delta") is not None:
                floors[row["run_id"]] = details["min_significant_delta"]
        for run in payload["runs"]:
            run["noise_floor"] = floors.get(run["id"])
        return payload

    @app.get("/api/run/{run_id}")
    def run_detail(run_id: str) -> dict:
        if not RUN_ID.match(run_id):
            raise HTTPException(status_code=400, detail="invalid run id")
        try:
            run = service.ledger_query(view="run", run_id=run_id)["run"]
        except GateError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        try:
            diagnosis = service.ledger_query(view="diagnosis", run_id=run_id)["results"]
        except GateError:
            diagnosis = None
        charts = []
        if diagnosis:
            for item in diagnosis["items"].values():
                if item.get("chart"):
                    charts.append(f"/charts/runs/{run_id}/{item['chart']}")
        return {"run": run, "diagnosis": diagnosis, "charts": charts}

    @app.get("/api/forensics/{forensics_id}")
    def forensics_detail(forensics_id: str) -> dict:
        if not FORENSICS_ID.match(forensics_id):
            raise HTTPException(status_code=400, detail="invalid forensics id")
        with service.ledger.connect() as con:
            row = con.execute("SELECT * FROM forensics WHERE id = ?", (forensics_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown forensics id")
        results = json.loads(row["results"])
        charts = [
            f"/charts/forensics/{forensics_id}/{item['chart']}"
            for item in results["items"].values()
            if item.get("chart")
        ]
        return {"results": results, "verdict": json.loads(row["verdict"]), "charts": charts}

    def _serve_svg(base: Path, name: str) -> FileResponse:
        if not SAFE_SVG.match(name):
            raise HTTPException(status_code=400, detail="invalid chart name")
        path = base / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="chart not found")
        return FileResponse(path, media_type="image/svg+xml")

    @app.get("/charts/runs/{run_id}/{name}")
    def run_chart(run_id: str, name: str) -> FileResponse:
        if not RUN_ID.match(run_id):
            raise HTTPException(status_code=400, detail="invalid run id")
        return _serve_svg(service.ledger.runs_dir / run_id / "diagnostics", name)

    @app.get("/charts/forensics/{forensics_id}/{name}")
    def forensics_chart(forensics_id: str, name: str) -> FileResponse:
        if not FORENSICS_ID.match(forensics_id):
            raise HTTPException(status_code=400, detail="invalid forensics id")
        return _serve_svg(service.ledger.root / "forensics" / forensics_id, name)

    @app.get("/verdict/{forensics_id}", response_class=HTMLResponse)
    def verdict_page(forensics_id: str) -> str:
        from .report import verdict_report_html

        if not FORENSICS_ID.match(forensics_id):
            raise HTTPException(status_code=400, detail="invalid forensics id")
        with service.ledger.connect() as con:
            goal = con.execute("SELECT * FROM goal WHERE id = 1").fetchone()
            row = con.execute("SELECT * FROM forensics WHERE id = ?", (forensics_id,)).fetchone()
            context = service._context_rows(con)
        if goal is None or row is None:
            raise HTTPException(status_code=404, detail="unknown forensics id")
        return verdict_report_html(
            service._goal_summary(goal),
            json.loads(row["results"]),
            json.loads(row["verdict"]),
            service.ledger.root / "forensics" / forensics_id,
            context=context,
        )

    return app
