"""MLLoop command-line interface: serve (MCP over stdio), status, init, report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _merge_json_config(path: Path, make_entry) -> str:
    """Insert the mlloop entry into a JSON config file, preserving existing content."""
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return f"SKIPPED {path.name}: existing file is not valid JSON — add the snippet manually."
    else:
        config = {}
    changed = make_entry(config)
    if not changed:
        return f"{path.name}: mlloop entry already present."
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return f"Wrote {path}"


def _claude_entry(config: dict) -> bool:
    servers = config.setdefault("mcpServers", {})
    if "mlloop" in servers:
        return False
    servers["mlloop"] = {"command": "mlloop", "args": ["serve"]}
    return True


def _opencode_entry(config: dict) -> bool:
    config.setdefault("$schema", "https://opencode.ai/config.json")
    servers = config.setdefault("mcp", {})
    if "mlloop" in servers:
        return False
    servers["mlloop"] = {"type": "local", "command": ["mlloop", "serve"], "enabled": True}
    return True


def _codex_snippet(workspace: Path) -> str:
    return (
        "# Codex CLI: add to ~/.codex/config.toml (config is global, so pin the workspace):\n"
        "[mcp_servers.mlloop]\n"
        'command = "mlloop"\n'
        f'args = ["serve", "--workspace", {json.dumps(str(workspace))}]\n'
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mlloop",
        description="MLLoop — a scientific-method harness for AI-driven ML training.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the MCP server over stdio")
    status = subparsers.add_parser("status", help="print the current workflow state as JSON")
    init = subparsers.add_parser("init", help="create .mlloop/ and configure agent MCP entries")
    init.add_argument(
        "--agent",
        choices=["claude", "opencode", "codex", "all", "none"],
        default="none",
        help="write the MCP config for this agent (codex prints a snippet; config is global)",
    )
    report = subparsers.add_parser("report", help="generate an HTML report")
    report.add_argument("--kind", choices=["verdict", "experiment"], default="verdict")
    report.add_argument("--output", default=None, help="output path for the HTML file")
    dashboard = subparsers.add_parser("dashboard", help="serve the local read-only dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8137)
    for sub in (serve, status, init, report, dashboard):
        sub.add_argument(
            "--workspace",
            default=None,
            help="project directory (default: $MLLOOP_WORKSPACE or the current directory)",
        )
    args = parser.parse_args(argv)

    from .server import create_server, resolve_workspace

    workspace = resolve_workspace(args.workspace)

    if args.command == "serve":
        create_server(args.workspace).run()
    elif args.command == "dashboard":
        import uvicorn

        from .dashboard import create_app

        print(f"MLLoop dashboard: http://{args.host}:{args.port}  (workspace: {workspace})")
        uvicorn.run(create_app(str(workspace)), host=args.host, port=args.port, log_level="warning")
    elif args.command == "status":
        from .service import LedgerService

        print(json.dumps(LedgerService(workspace).status(), indent=2, ensure_ascii=False))
    elif args.command == "report":
        from .service import GateError, LedgerService

        try:
            result = LedgerService(workspace).report_generate(kind=args.kind, output_path=args.output)
            print(result["path"])
        except GateError as exc:
            raise SystemExit(f"refused: {exc}")
    elif args.command == "init":
        from .service import LedgerService

        LedgerService(workspace)  # creates .mlloop/
        print(f"Initialized {workspace / '.mlloop'}")
        if args.agent in ("claude", "all"):
            print(_merge_json_config(workspace / ".mcp.json", _claude_entry))
        if args.agent in ("opencode", "all"):
            print(_merge_json_config(workspace / "opencode.json", _opencode_entry))
        if args.agent in ("codex", "all"):
            print(_codex_snippet(workspace))
        if args.agent == "none":
            print("\nClaude Code (.mcp.json in the project):")
            print(json.dumps({"mcpServers": {"mlloop": {"command": "mlloop", "args": ["serve"]}}}, indent=2))
            print("\nopencode (opencode.json in the project):")
            print(
                json.dumps(
                    {"mcp": {"mlloop": {"type": "local", "command": ["mlloop", "serve"], "enabled": True}}},
                    indent=2,
                )
            )
            print()
            print(_codex_snippet(workspace))
            print("Or rerun: mlloop init --agent claude|opencode|codex|all")


if __name__ == "__main__":
    main()
