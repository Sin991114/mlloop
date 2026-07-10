# Connecting MLLoop to your coding agent

MLLoop is a standard MCP server over stdio, so any MCP-capable agent can drive it.
Install first (`pip install mlloop` once released, or `pip install -e .` from source),
then configure your agent. The fastest path:

```bash
cd your-ml-project
mlloop init --agent claude     # or opencode / codex / all
```

`mlloop init` creates `.mlloop/` (the ledger) and writes/merges the agent config.
All state lives in `.mlloop/` inside your project — add it to `.gitignore` or commit
it if you want the experiment history versioned.

## Claude Code

`mlloop init --agent claude` writes `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "mlloop": { "command": "mlloop", "args": ["serve"] }
  }
}
```

Or add it with the CLI: `claude mcp add mlloop -- mlloop serve`

## opencode

`mlloop init --agent opencode` writes `opencode.json` in the project root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mlloop": { "type": "local", "command": ["mlloop", "serve"], "enabled": true }
  }
}
```

## Codex CLI

Codex config is global (`~/.codex/config.toml`), so pin the workspace explicitly.
`mlloop init --agent codex` prints the exact snippet with your absolute path:

```toml
[mcp_servers.mlloop]
command = "mlloop"
args = ["serve", "--workspace", "C:/path/to/your-ml-project"]
```

> Windows note: if `mlloop` is not on the PATH the agent uses, point `command` at the
> venv executable, e.g. `C:/path/to/.venv/Scripts/mlloop.exe`.

## Teaching the agent the workflow

The server announces the enforced workflow via MCP instructions, and every refused
call explains what to do instead — most agents self-correct from that alone. For best
results, add this to your project's `CLAUDE.md` / `AGENTS.md`:

```markdown
## ML training workflow
This project uses the mlloop MCP server for all model training work. Rules:
- Call mlloop `status` first; it tells you the current state and allowed actions.
- Never train without an open run: `run_start` -> train -> write artifacts -> `run_finish`.
- After every run, `diagnose_run` and study the failure modes before proposing changes.
- Every experiment must test a registered falsifiable hypothesis (`hypothesis_register`).
- If runs stagnate or you suspect the data/labels, run `forensics_run` and generate
  the verdict report instead of blindly trying more models.
- Improvements smaller than the noise floor (from `diagnose_run`) are noise, not progress.
```

## Overnight autonomous sessions

Set the budget in `goal_define` (`policy={"max_runs": 40}`) and let the agent run.
The gates keep it honest while you sleep; in the morning check:

- `mlloop status` — where things stand,
- `mlloop report --kind experiment` — the full iteration story,
- `mlloop report --kind verdict` — if forensics ran, the data-quality evidence.
