# Security Policy

## Supported versions

MLLoop is pre-1.0; only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them privately via
GitHub's "Report a vulnerability" (Security Advisories) on the repository, and we
will respond as quickly as we can.

## Scope notes

- The MCP server runs locally over stdio and executes no user code; it reads the
  dataset registered at `goal_define` and the artifact files written by your agent.
- Generated HTML reports embed data from your dataset (feature names, suspect rows).
  Treat them with the same sensitivity as the data itself before sharing.
- `.mlloop/` contains your full experiment history, including dataset paths and
  metadata — review before committing it to a public repository.
