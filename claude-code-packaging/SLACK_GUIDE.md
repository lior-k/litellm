# Run Claude Code through the Alice WonderFence guardrail

Routes Claude Code → local LiteLLM proxy + WonderFence → Bedrock Opus 4.7 *or* Anthropic API direct.

## Prereqs

- macOS, Claude Code CLI (`claude` on PATH)
- WonderFence API key + App ID (UUID)
- Bedrock: AWS SSO to an AF acct w/ Opus 4.7 — or — Anthropic: an `sk-ant-...` key. we recommend using an Anthropic key since we beleive vanguard is using anthropic directly, not through bedrock. to get an Anthropic key turn to aviju@alice.io

## Install (one time)

```bash
git clone -b feat-claude-code-packaging git@github.com:lior-k/litellm.git
cd litellm/claude-code-packaging
./install.sh
```

Installer asks Bedrock vs Anthropic, prompts for keys, writes `~/.alice-litellm/.env`, and symlinks `alice-claude` onto PATH.

## Run (two terminals)

```bash
# Terminal A — proxy (foreground)
login_aws_dev          # Bedrock only, if AWS token expired
./start.sh             # or: ./start.sh --bedrock | --anthropic

# Terminal B — Claude Code via guardrail
alice-claude           # same flags as `claude`
```

`alice-claude` health-checks the proxy first; if it's not up it tells you to start it.

## Update later

```bash
git pull && ./install.sh   # idempotent, keeps your .env
```

## Gotchas

- `proxy not reachable` → start.sh isn't running in another terminal.
- `AWS credentials expired` → run `login_aws_dev`.
- `Content blocked by Alice WonderFence` → flagged by policy; correlation ID in the error looks it up in the Alice dashboard.
