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

- macOS
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
login_aws_dev                   # only needed for Bedrock, if AWS token expired
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

### What gets sent to WonderFence

The unit of text sent for evaluation is **10 000 bytes** (UTF-8-safe).

Text larger than 10 KB is split into **10 KB sections with a 100-byte overlap**
and every section is evaluated in parallel, so a detection past the first 10 KB
isn't missed. BLOCK on any section → 400 for the whole request/response; MASK
across sections is reassembled best-effort.

- **Request:** only the **latest** user message goes to `evaluate_prompt`
  (not the whole conversation), but **all** of it — nothing is trimmed. A large
  `tool_result` therefore costs several parallel Alice calls before the LLM
  request starts.
- **Non-streaming response:** the full response, sectioned as above.
- **Streaming response:** chunks are **held, not forwarded**, until the eval
  covering them returns ALLOW. `evaluate_response` fires at most once per chunk,
  and only after at least `WONDERFENCE_EVAL_BYTES_INCREMENT` bytes (default 200)
  have arrived since the previous eval — so text reaches the client in ~200-byte
  bursts. A BLOCK discards the held batch (those bytes never reach the user) and
  closes the stream with a wire-format error frame (Anthropic SSE or OpenAI
  `content_filter` chunk). MASK is unenforceable on a held batch: it logs a WARN
  and the batch is released as-is.

`WONDERFENCE_STREAM_EVAL_MODE` picks what each streaming eval carries:

| Mode | Sent per eval | Trade-off |
|---|---|---|
| `deltas` (default) | the current + previous ~200-byte window (~400 B) | ~25× cheaper payload; misses a violation that only reads as one across a span longer than two windows |
| `rolling` | the rolling 10 KB buffer | widest detection context; ~10 KB per call even though only ~200 bytes are new |
| `accumulate_all` | the whole response, once | BLOCK **and** MASK enforceable, but no live tokens — the client waits for the full response |

Tune the sizes with `WONDERFENCE_BUFFER_BYTES`, `WONDERFENCE_EVAL_BYTES_INCREMENT`,
and `WONDERFENCE_RESPONSE_SECTION_OVERLAP_BYTES` in `~/.alice-litellm/.env`.

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

To use a different model, edit the relevant `litellm-config-*.yaml`.

### Per-request message dumps

Controlled by the guardrail's **`debug`** flag in `litellm-config-*.yaml` — the
single switch for all guardrail debug output. `debug: true` writes every dump
**and** prints the guardrail's DEBUG log lines; `debug: false` does neither.
It is independent of `LITELLM_LOG`, which only governs LiteLLM's own logging.

With `debug: true`, every request/response is dumped (for debug / replay) to:

```
~/.alice-litellm/messages/
```

Filenames are `<timestamp>[_<session_id>]_<hook_name>.{json,txt,jsonl}` —
one set per hook (`pre_call`, `during_call`, `post_call`, `post_call_stream`,
`post_call_stream_final`, etc.) plus per-chunk logs for streamed responses
(`*_chunks_text.jsonl`, `*_chunks_raw.jsonl`, `*_latest.txt`).

Override the location with `WONDERFENCE_MESSAGES_DIR=/some/path` in
`~/.alice-litellm/.env`. The directory is auto-created and gitignored — clear it
periodically if it grows. To turn dumps off, set `debug: false`.

> Note: an inline `LITELLM_LOG=DEBUG ./start.sh` has no effect — `start.sh`
> sources `~/.alice-litellm/.env` with `set -a`, so the file's value wins. Edit
> the file to change LiteLLM's log level. The guardrail's own output doesn't
> depend on it either way.

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
