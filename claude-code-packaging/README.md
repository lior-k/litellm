# alice-claude — Claude Code through Alice WonderFence guardrail

This directory packages a local [LiteLLM](https://github.com/BerriAI/litellm)
proxy + the Alice **WonderFence** content-safety guardrail and a thin wrapper
(`alice-claude`) that routes [Claude Code](https://docs.anthropic.com/claude/docs/claude-code)
through them to either **AWS Bedrock Claude Opus 4.7** or the **Anthropic
API direct** (latest Opus / Sonnet / Haiku).

```
┌──────────────┐  http://127.0.0.1:4000  ┌────────────────────────┐         ┌──────────────────────┐
│ Claude Code  │ ─────────────────────▶ │ LiteLLM + WonderFence  │ ──┬───▶  │ Bedrock Opus 4.7     │
│(alice-claude)│                        │ (this directory)        │   │     │ us-west-2            │
└──────────────┘                        └────────────────────────┘   │     └──────────────────────┘
                                                                      └──▶  ┌──────────────────────┐
                                                                            │ Anthropic API direct │
                                                                            │ (latest Opus/Sonnet) │
                                                                            └──────────────────────┘
```

Provider is chosen at install (or per-run via `./start.sh --bedrock | --anthropic`).

## Prerequisites

- macOS (Apple Silicon or Intel)
- Claude Code CLI installed (`claude` on PATH)
- WonderFence API key + App ID (UUID) — required for both providers
- **Bedrock:** AWS SSO access to an AF account with Bedrock Opus 4.7 (you can run `login_dev`)
- **Anthropic:** a real Anthropic API key (`sk-ant-...` from console.anthropic.com)

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
4. **Ask which provider** (Bedrock or Anthropic — no default, you must choose).
5. Prompt for `WONDERFENCE_API_KEY` + `WONDERFENCE_APP_ID`. Then, depending
   on the provider, either `ANTHROPIC_API_KEY` or just `AWS_REGION` (default
   `us-west-2`). All written to `~/.alice-litellm/.env` (mode 600).
6. Provider preflight — for Bedrock, runs `boto3 sts.get_caller_identity()`
   and tells you to `login_dev` if expired. For Anthropic, skipped.
7. Symlink `alice-claude` onto PATH (`/usr/local/bin`, `/opt/homebrew/bin`,
   or `~/.local/bin` — whichever is writable).

## Run

**Two terminals.**

```bash
# Terminal A — LiteLLM proxy (foreground)
login_dev                       # only needed for Bedrock, if AWS token expired
cd path/to/litellm/claude-code-packaging
./start.sh                      # uses PROVIDER from .env
# Or override per-run:
./start.sh --bedrock
./start.sh --anthropic

# Terminal B — Claude Code via guardrail
alice-claude                    # exact same flags as `claude`
```

`alice-claude` pings the proxy's `/health/liveliness` before exec'ing
`claude`; if the proxy isn't up it tells you to start it.

## Switching providers

The `.env` `PROVIDER=` line is the persistent default. To switch for a
single proxy run, pass `--bedrock` or `--anthropic` to `start.sh`. To
switch persistently, edit `~/.alice-litellm/.env` (or just re-run
`./install.sh` and pick the other one).

## What WonderFence does to your traffic

| Hook | When | Action |
|---|---|---|
| `pre_call` | Before the prompt reaches Bedrock | BLOCK → 400 to Claude Code; MASK → prompt rewritten |
| `post_call` | After Bedrock responds, before reply reaches Claude Code | BLOCK → 400; MASK → response rewritten |

### Rolling 10K buffer

On both sides, WonderFence sees a **rolling 10 000-byte window** of recent text.

- **Request:** all `role: user` messages are concatenated reverse-chrono, then
  trimmed to the last 10 KB (UTF-8-safe). That single buffer goes to
  `evaluate_prompt` — no longer just the last 100 chars of the latest message.
- **Streaming response:** chunks accumulate into a rolling 10 KB buffer.
  `evaluate_response` fires at most once per chunk, and only after at least
  `WONDERFENCE_EVAL_BYTES_INCREMENT` bytes (default 200) have arrived since
  the previous eval. A mid-stream BLOCK closes the stream with a
  wire-format error frame (Anthropic SSE or OpenAI `content_filter` chunk).
  Mid-stream MASK is unenforceable for already-released text and logs a
  WARN.

Tune with `WONDERFENCE_BUFFER_BYTES` and `WONDERFENCE_EVAL_BYTES_INCREMENT`
in `~/.alice-litellm/.env`.

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
| `litellm-config-bedrock.yaml` | LiteLLM config — all Claude Code model names route to Bedrock Opus 4.7 |
| `litellm-config-anthropic.yaml` | LiteLLM config — Opus/Sonnet/Haiku families route to their latest Anthropic-direct counterparts |
| `~/.alice-litellm/.env` | `WONDERFENCE_API_KEY`, `WONDERFENCE_APP_ID`, `AWS_REGION` |
| `~/.alice-litellm/messages/` | Per-request dumps (debug; auto-created, gitignored) |
| `wonderfence_guardrail.py` | The guardrail implementation |

To use a different model, edit the relevant `litellm-config-*.yaml`. To suppress request
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
