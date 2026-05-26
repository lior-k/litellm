# alice-claude — Claude Code through Alice WonderFence guardrail

This directory packages a local [LiteLLM](https://github.com/BerriAI/litellm)
proxy + the Alice **WonderFence** content-safety guardrail and a thin wrapper
(`alice-claude`) that routes [Claude Code](https://docs.anthropic.com/claude/docs/claude-code)
through them to **AWS Bedrock Claude Opus 4.7**.

```
┌──────────────┐  http://127.0.0.1:4000  ┌────────────────────────┐  AWS  ┌──────────────────────┐
│ Claude Code  │ ─────────────────────▶ │ LiteLLM + WonderFence  │ ────▶ │ Bedrock Opus 4.7     │
│ (alice-claude)│                       │  (this directory)      │       │ us-west-2            │
└──────────────┘                        └────────────────────────┘       └──────────────────────┘
```

## Prerequisites

- macOS (Apple Silicon or Intel)
- Claude Code CLI installed (`claude` on PATH)
- AWS SSO access to the AF dev/sandbox account (you can run `login_dev`)
- WonderFence API key + App ID (UUID)

## Install

```bash
git clone -b feat-claude-code-packaging git@github.com:lior-k/litellm.git
cd litellm/claude-code-packaging
./install.sh
```

`install.sh` will:
1. Install [`uv`](https://github.com/astral-sh/uv) if missing.
2. Create an isolated venv at `~/.alice-litellm/venv`.
3. `uv pip install` LiteLLM + WonderFence SDK + boto3.
4. Prompt for `WONDERFENCE_API_KEY` and `WONDERFENCE_APP_ID`, written to
   `~/.alice-litellm/.env` (mode 600).
5. Verify AWS creds via `boto3 sts.get_caller_identity()` — if expired,
   tell you to run `login_dev`.
6. Symlink `alice-claude` onto PATH (`/usr/local/bin`, `/opt/homebrew/bin`,
   or `~/.local/bin` — whichever is writable).

## Run

**Two terminals.**

```bash
# Terminal A — LiteLLM proxy (foreground)
login_dev                # if AWS token expired
cd path/to/litellm/claude-code-packaging
./start.sh               # binds 127.0.0.1:4000

# Terminal B — Claude Code via guardrail
alice-claude             # exact same flags as `claude`
```

`alice-claude` pings the proxy's `/health/liveliness` before exec'ing
`claude`; if the proxy isn't up it tells you to start it.

## What WonderFence does to your traffic

| Hook | When | Action |
|---|---|---|
| `pre_call` | Before the prompt reaches Bedrock | BLOCK → 400 to Claude Code; MASK → prompt rewritten |
| `post_call` | After Bedrock responds, before reply reaches Claude Code | BLOCK → 400; MASK → response rewritten |

Blocked requests return:

```json
{
  "error": {
    "message": "Content blocked by Alice WonderFence safety policy",
    "type": "wonderfence_content_policy_violation",
    "wonderfence_correlation_id": "...",
    "wonderfence_detections": "jailbreak(score=0.95)[1 span(s)]"
  }
}
```

## Configuration

| File | What |
|---|---|
| `litellm-config.yaml` | LiteLLM model list + guardrail wiring (renamed from `config.yaml` because litellm's root `.gitignore` excludes that name) |
| `~/.alice-litellm/.env` | `WONDERFENCE_API_KEY`, `WONDERFENCE_APP_ID`, `AWS_REGION` |
| `~/.alice-litellm/messages/` | Per-request dumps (debug; auto-created, gitignored) |
| `wonderfence_guardrail.py` | The guardrail implementation |

To use a different Bedrock model, edit `litellm-config.yaml`. To suppress request
dumps, set `WONDERFENCE_MESSAGES_DIR=/dev/null` in the env file (or
delete the dir periodically).

## Troubleshooting

**`alice-claude: LiteLLM proxy not reachable`** — `start.sh` is not running
in another terminal. Open one and run it.

**`AWS credentials not available or expired`** — run `login_dev` (pritunl
VPN + AWS SSO refresh; alias from `af-config`).

**`Content blocked by Alice WonderFence safety policy`** — WonderFence
flagged the request. Correlation ID in the error body looks up the
evaluation in Alice's dashboard.

**Bedrock model not accessible** — your AWS role must have
`bedrock:InvokeModel` on `us.anthropic.claude-opus-4-7` in `us-west-2`.
Check with: `aws bedrock list-foundation-models --region us-west-2`.

## Uninstall

```bash
./uninstall.sh           # removes ~/.alice-litellm and the symlink
```

Does not touch `~/.aws`, the cloned fork, or your Claude Code install.

## Update

```bash
git pull
./install.sh             # idempotent — re-runs deps install, preserves .env
```
