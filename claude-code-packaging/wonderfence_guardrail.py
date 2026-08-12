"""Alice WonderFence Guardrail Integration for LiteLLM.

This module provides a custom guardrail implementation that integrates Alice WonderFence
with LiteLLM to evaluate user prompts and LLM responses for content safety.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import time
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral, ModelResponse
from wonderfence_sdk.client import WonderFenceV2Client
from wonderfence_sdk.models import Actions, AnalysisContext, EvaluateMessageResponse

logger = verbose_proxy_logger.getChild("wonderfence")

MESSAGES_DIR = Path(
    os.environ.get("WONDERFENCE_MESSAGES_DIR")
    or (Path.home() / ".alice-litellm" / "messages")
)

BUFFER_MAX_BYTES = int(os.environ.get("WONDERFENCE_BUFFER_BYTES", "10000"))
EVAL_BYTES_INCREMENT = int(os.environ.get("WONDERFENCE_EVAL_BYTES_INCREMENT", "200"))
RESPONSE_SECTION_OVERLAP_BYTES = int(
    os.environ.get("WONDERFENCE_RESPONSE_SECTION_OVERLAP_BYTES", "100")
)
RESPONSE_SECTION_MAX_CONCURRENCY = int(
    os.environ.get("WONDERFENCE_RESPONSE_SECTION_CONCURRENCY", "10")
)

# `x-claude-code-session-id`, `x-litellm-session-id`, `x-<vendor>-session-id`.
_SESSION_ID_HEADER_RE = re.compile(r"^x-.+-session-id$")

# Monotonic per-process counter so dump filenames sort by write order even when
# timestamps collide at ms resolution (concurrent requests interleave).
_dump_seq = itertools.count(1)


def _seq_ts() -> str:
    """Filename prefix `<date>-<time>-<ms>_<seq>` for dump files."""
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
    return f"{ts}_{next(_dump_seq):06d}"


def _configure_logging(debug: bool) -> None:
    """Make the guardrail's `debug` flag the single switch for its debug output.

    litellm's handler sits on the parent (verbose_proxy_logger) at LITELLM_LOG's
    level, so raising only our logger's level emits records the parent handler
    then drops — DEBUG lines never appear (while bare print()s do, which is the
    confusing part). Our own handler + propagate=False decouples us from
    LITELLM_LOG in both directions: `debug: true` shows our DEBUG lines even at
    LITELLM_LOG=INFO, `debug: false` hides them even at LITELLM_LOG=DEBUG.
    """
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not debug:
        return
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s:%(levelname)s: "
                "%(filename)s:%(lineno)d - %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.propagate = False


def _dumps_enabled() -> bool:
    """Per-request dumps are debug-only — they mirror full request/response
    bodies to disk, useful for replay and dead weight (plus a data-retention
    liability) otherwise. Driven by the same `debug` flag via _configure_logging.
    """
    return logger.isEnabledFor(logging.DEBUG)


class StreamEvalMode(str, Enum):
    """How a streamed response is evaluated (WONDERFENCE_STREAM_EVAL_MODE).

    DELTAS (default): gated on the current + previous EVAL_BYTES_INCREMENT
    window only (~2 increments per call) — cheap payload, but a violation that
    only reads as one across a longer span is missed.
    ROLLING: gated on a rolling BUFFER_MAX_BYTES window, re-sent every
    EVAL_BYTES_INCREMENT bytes — widest detection context, ~10KB per call.
    ACCUMULATE_ALL: buffer the whole response, evaluate once, then re-stream.
    """

    ROLLING = "rolling"
    DELTAS = "deltas"
    ACCUMULATE_ALL = "accumulate_all"


def _split_overlapping_utf8(text: str, max_bytes: int, overlap_bytes: int) -> list[str]:
    """Split `text` into <= `max_bytes` UTF-8 sections that overlap by
    `overlap_bytes`, so a detection straddling a section boundary isn't missed.

    Sections step by `max_bytes - overlap_bytes`. Byte-window cuts use
    errors='ignore' so a split codepoint at either edge is dropped rather than
    corrupting the section. Returns `[text]` when it already fits.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]
    step = max(1, max_bytes - overlap_bytes)
    sections: list[str] = []
    start = 0
    n = len(encoded)
    while start < n:
        sections.append(
            encoded[start : start + max_bytes].decode("utf-8", errors="ignore")
        )
        if start + max_bytes >= n:
            break
        start += step
    return sections


class _RollingBuffer:
    """Per-request rolling byte buffer for streaming response eval.

    `append_text` keeps only the last `max_bytes` and tracks bytes added since
    the last `mark_evaluated()`. `needs_eval()` flips true once that delta
    crosses `eval_increment` — one eval per chunk regardless of chunk size.

    Plain class (not @dataclass): this module is loaded dynamically by LiteLLM
    via spec.loader.exec_module() and never registered in sys.modules, so
    Python 3.13's dataclass machinery crashes resolving string annotations
    (from __future__ import annotations) via sys.modules.get(__module__).
    """

    def __init__(self, max_bytes: int, eval_increment: int) -> None:
        self.max_bytes = max_bytes
        self.eval_increment = eval_increment
        self._buf: bytes = b""
        self._since_eval: int = 0

    def append_text(self, text: str) -> None:
        chunk = text.encode("utf-8")
        self._buf = (self._buf + chunk)[-self.max_bytes :]
        self._since_eval += len(chunk)

    def needs_eval(self) -> bool:
        return self._since_eval >= self.eval_increment

    def mark_evaluated(self) -> None:
        self._since_eval = 0

    def pending_bytes(self) -> int:
        return self._since_eval

    def text(self) -> str:
        return self._buf.decode("utf-8", errors="ignore")


class _DeltaWindows:
    """Two-window buffer for streaming eval: the eval text is the previous
    window plus the current one (~2 × `eval_increment`) instead of the whole
    rolling window, so each call carries ~400 bytes instead of ~10K.

    Exposes the same 5 methods as `_RollingBuffer`, so the gated loop is
    identical for both. Every adjacent window pair is evaluated exactly once,
    so a detection straddling a window boundary is still covered — but a
    violation that only reads as one across a longer span is not.

    Windows close only on chunk boundaries and whole chunk text is appended, so
    neither window ever holds a split codepoint.
    """

    def __init__(self, eval_increment: int) -> None:
        self.eval_increment = eval_increment
        self._prev: bytes = b""
        self._cur: bytes = b""

    def append_text(self, text: str) -> None:
        self._cur += text.encode("utf-8")

    def needs_eval(self) -> bool:
        return len(self._cur) >= self.eval_increment

    def mark_evaluated(self) -> None:
        self._prev, self._cur = self._cur, b""

    def pending_bytes(self) -> int:
        return len(self._cur)

    def text(self) -> str:
        return (self._prev + self._cur).decode("utf-8", errors="ignore")


def _safe_serialize(obj: Any, seen: set[int] | None = None) -> Any:
    """Walk obj, breaking cycles and converting unknowns via str()."""
    if seen is None:
        seen = set()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    oid = id(obj)
    if oid in seen:
        return f"<cycle to {type(obj).__name__}>"
    seen.add(oid)
    try:
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v, seen) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_safe_serialize(v, seen) for v in obj]
        if hasattr(obj, "model_dump"):
            try:
                return _safe_serialize(obj.model_dump(), seen)
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return _safe_serialize(vars(obj), seen)
        return str(obj)
    finally:
        seen.discard(oid)


def _session_id_from_headers(headers: Any) -> str | None:
    """Session id from an `x-<vendor>-session-id` request header.

    Claude Code sends `x-claude-code-session-id: <uuid>`. Newer LiteLLM maps
    such headers onto `litellm_session_id` itself, but the pinned runtime
    (1.81.0) has no such mapping, so read the header directly.
    """
    if not isinstance(headers, dict):
        return None
    normalized = {k.lower(): v for k, v in headers.items() if isinstance(k, str)}
    explicit = normalized.get("x-litellm-session-id")
    if explicit:
        return str(explicit)
    for key, value in normalized.items():
        if _SESSION_ID_HEADER_RE.match(key) and isinstance(value, str) and value:
            return value
    return None


def _normalize_user_id(value: Any) -> str | None:
    """Reduce a user-id candidate to a usable identity string.

    Claude Code sets Anthropic's `metadata.user_id` to a 150-char JSON blob
    (`{"device_id": ..., "account_uuid": ..., "session_id": ...}`), which the
    100-char cap would truncate into invalid JSON. Pull the real identity out:
    `account_uuid` when signed in, else the stable per-machine `device_id`.
    Plain strings pass through; anything unusable yields None so the caller
    falls through to the next candidate.
    """
    if not isinstance(value, str) or not value:
        return None
    if not value.startswith("{"):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if not isinstance(parsed, dict):
        return value
    picked = parsed.get("account_uuid") or parsed.get("device_id")
    return str(picked) if picked else None


def _extract_session_id(data: Any) -> str | None:
    """Try to extract session_id from request data, then from request headers."""
    if not isinstance(data, dict):
        return None
    sid = data.get("litellm_session_id") or data.get("session_id")
    if sid:
        return str(sid)
    metadata = data.get("metadata") or data.get("litellm_metadata") or {}
    if isinstance(metadata, dict):
        sid = metadata.get("litellm_session_id") or metadata.get("session_id")
        if sid:
            return str(sid)
    for source in (metadata, data.get("proxy_server_request")):
        if isinstance(source, dict):
            sid = _session_id_from_headers(source.get("headers"))
            if sid:
                return sid
    return None


def _dump_request(hook_name: str, data: Any) -> None:
    """Save full request data to messages/<ts>_<seq>[_<session_id>]_<hook_name>.json."""
    if not _dumps_enabled():
        return
    try:
        MESSAGES_DIR.mkdir(exist_ok=True)
        ts = _seq_ts()
        sid = _extract_session_id(data) or (
            _extract_session_id(data.get("data")) if isinstance(data, dict) else None
        )
        sid_part = f"_{sid}" if sid else ""
        dest = MESSAGES_DIR / f"{ts}{sid_part}_{hook_name}.json"
        safe = _safe_serialize(data)
        dest.write_text(json.dumps(safe, indent=2, default=str))
    except Exception as e:
        logger.debug("Failed to dump request for %s: %s", hook_name, e, exc_info=True)


def _dump_response_text(response: Any, tag: str = "response_text") -> None:
    """Save plain-text LLM response content to messages/<ts>_<seq>_<tag>.txt."""
    if not _dumps_enabled():
        return
    try:
        text = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content or ""
        if not text:
            return
        MESSAGES_DIR.mkdir(exist_ok=True)
        ts = _seq_ts()
        dest = MESSAGES_DIR / f"{ts}_{tag}.txt"
        dest.write_text(text)
    except Exception as e:
        logger.debug("Failed to dump %s: %s", tag, e, exc_info=True)


def _dump_eval_text(text: str, tag: str, sid: str | None = None) -> None:
    """Save the EXACT text sent to WonderFence to messages/<ts>_<seq>[_<sid>]_<tag>.txt.

    `tag` is `<event>_eval_prompt` or `<event>_eval_response`. Unlike the
    request/response dumps, this is post-flattening — the exact text handed to
    the eval layer, before it is split into sections.
    """
    if not _dumps_enabled():
        return
    try:
        if not text:
            return
        MESSAGES_DIR.mkdir(exist_ok=True)
        sid_part = f"_{sid}" if sid else ""
        dest = MESSAGES_DIR / f"{_seq_ts()}{sid_part}_{tag}.txt"
        dest.write_text(text)
    except Exception as e:
        logger.debug("Failed to dump %s: %s", tag, e, exc_info=True)


def _append_eval_buffer(
    path: Path, index: int, buffer_text: str, final: bool = False
) -> None:
    """Append one entry per eval call: the buffer sent to WonderFence.

    One `eval-NN` block per evaluate_response call, not per Claude chunk — the
    buffer is the ~10K rolling window (rolling mode) or the current + previous
    increment window (deltas mode), so each entry is written on its own lines.
    """
    if not _dumps_enabled():
        return
    try:
        tag = f"eval-{index:02d}{' (final tail)' if final else ''}"
        with path.open("a") as f:
            f.write(
                f"===== {tag} — {len(buffer_text)} chars "
                f"({len(buffer_text.encode('utf-8'))} bytes) =====\n"
                f"{buffer_text}\n\n"
            )
    except Exception as e:
        logger.debug("Failed to append eval buffer %d: %s", index, e, exc_info=True)


def _make_chunk_log_paths(sid: str | None) -> tuple[Path, Path, Path, Path]:
    """Return (text_path, raw_path, response_full_path, eval_buffers_path) for the
    current streaming request. Filenames mirror _dump_request datetime-first scheme.

    Each file takes its own sequence number, assigned in the order the files are
    listed here: raw wire events first, then extracted per-chunk text, then the
    assembled response, then the eval inputs.

    Paths are still returned when dumps are off — the writers themselves no-op.
    """
    if _dumps_enabled():
        MESSAGES_DIR.mkdir(exist_ok=True)
    sid_part = f"_{sid}" if sid else ""
    raw = MESSAGES_DIR / f"{_seq_ts()}{sid_part}_post_call_stream_chunks_raw.jsonl"
    text = MESSAGES_DIR / f"{_seq_ts()}{sid_part}_post_call_stream_chunks_text.jsonl"
    # Every chunk's text, appended as it arrives — logged before the eval runs, so
    # this holds the COMPLETE response Claude produced, including any chunks the
    # gate later withheld from the user.
    response_full = (
        MESSAGES_DIR / f"{_seq_ts()}{sid_part}_post_call_stream_response_full.txt"
    )
    # One entry per eval call: the rolling buffer sent to Alice (eval-01, eval-02…).
    eval_buffers = (
        MESSAGES_DIR / f"{_seq_ts()}{sid_part}_post_call_stream_eval_chunks.txt"
    )
    return text, raw, response_full, eval_buffers


def _serialize_chunk_raw(chunk: Any) -> str:
    """Best-effort string form of chunk for raw log."""
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk).decode("utf-8", errors="replace")
    for attr in ("model_dump_json", "json"):
        fn = getattr(chunk, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return repr(chunk)


def _log_chunk(
    idx: int, chunk: Any, text_path: Path, raw_path: Path, response_full_path: Path
) -> None:
    """Append one line to each per-request log + append running text + print to stdout.
    Best effort."""
    if not _dumps_enabled():
        return
    try:
        text = WonderFenceGuardrail._extract_chunk_text(chunk)
        raw = _serialize_chunk_raw(chunk)
        with text_path.open("a") as f:
            f.write(json.dumps({"i": idx, "text": text}) + "\n")
        with raw_path.open("a") as f:
            f.write(json.dumps({"i": idx, "raw": raw}) + "\n")
        if text:
            with response_full_path.open("a") as f:
                f.write(text)
        print(f"[wonderfence] chunk[{idx}]: {text!r}", flush=True)
    except Exception as e:
        print(f"[wonderfence] chunk log failed (idx={idx}): {e!r}", flush=True)


class SafetyCheckUnavailable(Exception):
    """Raised when WonderFence safety check is unavailable."""


class WonderFenceGuardrail(CustomGuardrail):
    """Custom guardrail that uses Alice WonderFence to evaluate prompts and responses.

    This guardrail operates in two modes:
    - pre_call: Evaluates user prompts before sending to the LLM
    - post_call: Evaluates LLM responses before returning to the user

    Both ``api_key`` and ``app_id`` can be overridden per request via metadata,
    per API key, or per team, allowing multiple applications/tenants to share a
    single LiteLLM proxy with different WonderFence credentials and policies.

    Resolution order for api_key:
    1. Request metadata: ``metadata.wonderfence_api_key``
    2. API key metadata: ``user_api_key_metadata.wonderfence_api_key``
    3. Team metadata: ``user_api_key_team_metadata.wonderfence_api_key``
    4. Default: configured ``api_key`` or ``WONDERFENCE_API_KEY`` env var

    Resolution order for app_id:
    1. Request metadata: ``metadata.wonderfence_app_id``
    2. API key metadata: ``user_api_key_metadata.wonderfence_app_id``
    3. Team metadata: ``user_api_key_team_metadata.wonderfence_app_id``
    4. Default: configured ``app_id`` or ``WONDERFENCE_APP_ID`` env var

    Args:
        guardrail_name: Unique identifier for this guardrail instance
        api_key: Default WonderFence API key (defaults to WONDERFENCE_API_KEY env var).
            Can be overridden per request via metadata.
        app_id: Default application UUID whose policies to apply (defaults to
            WONDERFENCE_APP_ID env var). Can be overridden per request via metadata.
        block_message: Message to display when content is blocked
        platform: Cloud platform identifier (e.g., 'aws', 'azure', 'databricks')
        fail_open: When True, allow requests to proceed if WonderFence is unreachable.
            When False (default), block requests when WonderFence is unavailable.
        debug: The single debug switch, independent of LITELLM_LOG. When True,
            the guardrail's DEBUG log lines are printed AND every per-request
            dump under MESSAGES_DIR is written. When False (default), neither.
        max_cached_clients: Max SDK clients cached per guardrail instance, keyed by
            API key (defaults to WONDERFENCE_MAX_CACHED_CLIENTS env var, then 10).
        connection_pool_limit: Max connections in each SDK client's HTTP pool
            (defaults to WONDERFENCE_CONNECTION_POOL_LIMIT env var, then SDK default).
        stream_eval_mode: How a streamed response is evaluated — see
            StreamEvalMode (defaults to WONDERFENCE_STREAM_EVAL_MODE env var,
            then "deltas"). An unknown value raises ValueError.
        **kwargs: Additional arguments passed to CustomGuardrail
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str | None = None,
        app_id: str | None = None,
        block_message: str = "Content violates our policies and has been blocked",
        platform: str | None = None,
        fail_open: bool = False,
        debug: bool = False,
        max_cached_clients: int | None = None,
        connection_pool_limit: int | None = None,
        stream_eval_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("WONDERFENCE_API_KEY")
        self.app_id = app_id or os.environ.get("WONDERFENCE_APP_ID")

        self.guardrail_name = guardrail_name
        self.block_message = block_message
        self.platform = platform
        self.fail_open = fail_open

        # Streaming response mode — see StreamEvalMode. Both gated modes hold
        # chunks until the eval covering them returns ALLOW, so a BLOCK is
        # caught before the offending bytes reach the user (MASK still can't be
        # applied); they differ only in how much context each call carries.
        # An unknown value raises ValueError here, at init.
        self.stream_eval_mode = StreamEvalMode(
            stream_eval_mode
            or os.environ.get("WONDERFENCE_STREAM_EVAL_MODE")
            or StreamEvalMode.DELTAS.value
        )

        _configure_logging(debug)
        self._client_cache: OrderedDict[str, WonderFenceV2Client] = OrderedDict()
        self._client_cache_maxsize = max_cached_clients or int(
            os.environ.get("WONDERFENCE_MAX_CACHED_CLIENTS", "10")
        )
        self._connection_pool_limit: int | None = connection_pool_limit or (
            int(os.environ.get("WONDERFENCE_CONNECTION_POOL_LIMIT"))
            if os.environ.get("WONDERFENCE_CONNECTION_POOL_LIMIT")
            else None
        )

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.during_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        logger.info("WonderFence initialized: app_id=%s", self.app_id)

    async def _get_client(self, api_key: str) -> WonderFenceV2Client:
        """Return a cached WonderFenceV2Client for the given api_key (LRU)."""
        if api_key in self._client_cache:
            self._client_cache.move_to_end(api_key)
            return self._client_cache[api_key]
        client = WonderFenceV2Client(
            api_key=api_key,
            connection_pool_limit=self._connection_pool_limit,
        )
        self._client_cache[api_key] = client
        if len(self._client_cache) > self._client_cache_maxsize:
            _, evicted_client = self._client_cache.popitem(last=False)
            try:
                await evicted_client.close()
            except Exception:
                logger.warning(
                    "Failed to close evicted WonderFence client", exc_info=True
                )
        return client

    def _resolve_app_id(
        self, data: dict[str, Any], user_api_key_dict: UserAPIKeyAuth
    ) -> str:
        """Resolve app_id from request metadata, key metadata, team metadata, or default.

        Priority:
        1. Request metadata (``metadata.wonderfence_app_id``)
        2. API key metadata (``user_api_key_metadata.wonderfence_app_id``)
        3. Team metadata (``user_api_key_team_metadata.wonderfence_app_id``)
        4. Default app_id from config / env var
        """
        # 1. Request-level metadata
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        req_app_id = metadata.get("wonderfence_app_id")
        if req_app_id:
            return req_app_id

        # 2. API key metadata (set via /key/generate)
        key_metadata = metadata.get("user_api_key_metadata") or {}
        if isinstance(key_metadata, dict) and key_metadata.get("wonderfence_app_id"):
            return key_metadata["wonderfence_app_id"]

        # 3. Team metadata (set via /team/new or /team/update)
        team_metadata = metadata.get("user_api_key_team_metadata") or {}
        if isinstance(team_metadata, dict) and team_metadata.get("wonderfence_app_id"):
            return team_metadata["wonderfence_app_id"]

        # 4. Default
        if not self.app_id:
            raise ValueError(
                "No wonderfence_app_id found in request metadata, key metadata, "
                "team metadata, or default config (WONDERFENCE_APP_ID)"
            )
        return self.app_id

    def _resolve_api_key(
        self, data: dict[str, Any], user_api_key_dict: UserAPIKeyAuth
    ) -> str:
        """Resolve api_key from request metadata, key metadata, team metadata, or default.

        Priority:
        1. Request metadata (``metadata.wonderfence_api_key``)
        2. API key metadata (``user_api_key_metadata.wonderfence_api_key``)
        3. Team metadata (``user_api_key_team_metadata.wonderfence_api_key``)
        4. Default api_key from config / env var
        """
        # 1. Request-level metadata
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        req_api_key = metadata.get("wonderfence_api_key")
        if req_api_key:
            return req_api_key

        # 2. API key metadata (set via /key/generate)
        key_metadata = metadata.get("user_api_key_metadata") or {}
        if isinstance(key_metadata, dict) and key_metadata.get("wonderfence_api_key"):
            return key_metadata["wonderfence_api_key"]

        # 3. Team metadata (set via /team/new or /team/update)
        team_metadata = metadata.get("user_api_key_team_metadata") or {}
        if isinstance(team_metadata, dict) and team_metadata.get("wonderfence_api_key"):
            return team_metadata["wonderfence_api_key"]

        # 4. Default
        if not self.api_key:
            raise ValueError(
                "No wonderfence_api_key found in request metadata, key metadata, "
                "team metadata, or default config (WONDERFENCE_API_KEY)"
            )
        return self.api_key

    def _build_analysis_context(self, request_data: dict[str, Any]) -> AnalysisContext:
        """Build WonderFence AnalysisContext from request data.

        Extracts metadata including user_id, session_id, model name, and provider
        from various possible locations in the request data.

        Args:
            request_data: Request data dictionary containing model, metadata, etc.

        Returns:
            AnalysisContext instance for WonderFence SDK
        """
        metadata = (
            request_data.get("metadata") or request_data.get("litellm_metadata") or {}
        )
        model_str = request_data.get("model", "")

        # Extract provider and clean model name
        provider = None
        model_name = model_str
        if model_str:
            try:
                model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
            except Exception:
                # Fallback: simple prefix extraction
                if "/" in model_str:
                    provider, model_name = model_str.split("/", 1)

        # User ID from API key metadata (prefixed by framework). End-user IDs
        # win; `user_api_key_user_id` (the owner of the virtual key, set via
        # /key/generate --user_id) is the last resort — coarser than an end
        # user, but the only identity available when the caller doesn't pass one.
        user_id = None
        for candidate in (
            metadata.get("user_api_key_end_user_id"),
            metadata.get("end_user_id"),
            metadata.get("user_id"),
            metadata.get("user_api_key_user_id"),
        ):
            user_id = _normalize_user_id(candidate)
            if user_id:
                break
        if user_id and len(user_id) > 100:
            user_id = user_id[:100]

        # Session ID: `litellm_session_id` in extra_body, else an
        # `x-<vendor>-session-id` header (Claude Code sends one per session).
        session_id = _extract_session_id(request_data)

        return AnalysisContext(
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider=provider,
            platform=self.platform,
        )

    @staticmethod
    def _is_harness_block(text: str) -> bool:
        """True if a text block is Claude Code scaffolding, not user content.

        Claude Code packs CLAUDE.md, tool/skill listings, IDE context and local
        slash-command output into the user message as tagged text blocks. They
        dwarf the real prompt (~50KB vs ~100B), so evaluating them wastes the
        10KB budget and lets an injection in an early block fall outside the
        tail-trim.

        Matching is whole-block only (starts with the open tag AND ends with the
        matching close tag): a `<system-reminder>` string appearing *inside*
        fetched content must never be able to hide that content from the eval.
        """
        stripped = text.strip()
        for tag in ("system-reminder", "local-command-caveat", "local-command-stdout"):
            if stripped.startswith(f"<{tag}>") and stripped.endswith(f"</{tag}>"):
                return True
        # The /slash-command trio arrives as one block of adjacent elements.
        if stripped.startswith("<command-name>") and stripped.endswith(
            ("</command-args>", "</command-message>", "</command-name>")
        ):
            return True
        return False

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Flatten a single user-role message to plain text (string, text blocks,
        tool_result blocks), dropping Claude Code harness scaffolding blocks.

        tool_result blocks are always kept — they carry fetched pages, file
        contents and command output, which is where injected content arrives.
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "") or ""
                        if WonderFenceGuardrail._is_harness_block(text):
                            continue
                        parts.append(text)
                    elif block.get("type") == "tool_result":
                        tr = block.get("content", "")
                        if isinstance(tr, str):
                            parts.append(f"[tool_result] {tr}")
                        elif isinstance(tr, list):
                            for sub in tr:
                                if isinstance(sub, dict) and sub.get("type") == "text":
                                    parts.append(
                                        f"[tool_result] {sub.get('text', '') or ''}"
                                    )
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content) if content else ""

    def _collect_user_text_full(self, data: dict[str, Any]) -> str:
        """Return the latest user message text, flattened but NOT trimmed.

        Only the most recent user-role message is considered — not the
        concatenated history. Handles string content, text blocks, and
        tool_result blocks. If the newest user message has no text, scans older
        ones until it finds text (e.g. an image-only turn is skipped).
        """
        messages = data.get("messages", []) or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            text = self._message_text(message)
            if not text:
                continue
            return text
        return ""

    def _extract_response_content(self, response: ModelResponse) -> str:
        """Extract content from the first choice in the response.

        Args:
            response: LLM response object

        Returns:
            Content from first choice, or empty string if not found
        """
        if not (hasattr(response, "choices") and response.choices):
            return ""

        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content or ""

        return ""

    def _format_detections(self, result: EvaluateMessageResponse) -> str:
        """Format detection details from a WonderFence evaluation result for logging."""
        detections = getattr(result, "detections", None)
        if not detections:
            return "none"
        parts = []
        for d in detections:
            entry = f"{d.type}(score={d.score:.2f})"
            if d.spans:
                entry += f"[{len(d.spans)} span(s)]"
            parts.append(entry)
        return ", ".join(parts)

    def _handle_evaluation_result(
        self, result: EvaluateMessageResponse, evaluation_type: str, content: str
    ) -> str:
        """Process WonderFence evaluation result and return modified content if needed.

        Args:
            result: WonderFence EvaluateMessageResponse containing action, detections, etc.
            evaluation_type: Type of evaluation ('prompt' or 'response')
            content: Original content being evaluated

        Returns:
            Modified content (if MASK action) or original content

        Raises:
            HTTPException: If action is BLOCK (status 400)
        """
        action = result.action.value
        action_text = result.action_text
        correlation_id = getattr(result, "correlation_id", None)
        detection_summary = self._format_detections(result)

        if action == Actions.BLOCK.value:
            logger.warning(
                "WonderFence BLOCKED %s: wonderfence_correlation_id=%s evaluation_type=[%s], detections=[%s] content=%s...",
                evaluation_type,
                correlation_id,
                evaluation_type,
                detection_summary,
                content[:100],
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": self.block_message,
                    "type": "wonderfence_content_policy_violation",
                    "wonderfence_correlation_id": correlation_id,
                    "wonderfence_detections": detection_summary,
                },
            )

        elif action == Actions.MASK.value:
            if action_text:
                logger.warning(
                    "WonderFence MASKED %s: wonderfence_correlation_id=%s detections=[%s]",
                    evaluation_type,
                    correlation_id,
                    detection_summary,
                )
                return action_text
            logger.warning(
                "WonderFence returned MASK action without action_text for %s: "
                "wonderfence_correlation_id=%s detections=[%s]",
                evaluation_type,
                correlation_id,
                detection_summary,
            )

        if action == Actions.DETECT.value:
            logger.warning(
                "WonderFence DETECTED violation in %s: wonderfence_correlation_id=%s detections=[%s] content=%s...",
                evaluation_type,
                correlation_id,
                detection_summary,
                content[:100],
            )
        else:
            logger.debug(
                "WonderFence ALLOWED %s: wonderfence_correlation_id=%s detections=[%s]",
                evaluation_type,
                correlation_id,
                detection_summary,
            )

        return content

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle errors during evaluation.

        Args:
            error: Exception that occurred
            context: Context string ('pre_call' or 'post_call')

        Raises:
            HTTPException: If error is a guardrail intervention (re-raised)
            SafetyCheckUnavailable: For all other errors (unless fail_open is True)
        """
        # Re-raise guardrail interventions
        if isinstance(error, HTTPException):
            raise

        if self.fail_open:
            logger.critical(
                "WonderFence unreachable (fail-open): proceeding without guardrail. "
                "context=%s error=%s",
                context,
                str(error),
                exc_info=True,
            )
            return

        logger.error("WonderFence %s error: %s", context, str(error), exc_info=True)
        raise SafetyCheckUnavailable(
            f"Alice WonderFence guardrails unavailable: [{context}] {error}"
        ) from error

    async def _evaluate_prompt(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        hook_name: str,
    ) -> dict[str, Any]:
        """Evaluate user prompt against WonderFence policies.

        Shared logic for both pre_call and during_call hooks.

        Args:
            data: Request data containing messages
            user_api_key_dict: User API key metadata
            hook_name: Hook identifier for logging ('pre_call' or 'during_call')

        Returns:
            Modified request data (with potentially masked content)

        Raises:
            HTTPException: If content is blocked (status 400)
            SafetyCheckUnavailable: If evaluation fails
        """
        try:
            full_text = self._collect_user_text_full(data)
            if not full_text:
                logger.debug(
                    "WonderFence %s: no user message found, skipping", hook_name
                )
                return data

            # No tail-trim: the whole message is evaluated, split into
            # BUFFER_MAX_BYTES sections by the common layer when oversized.
            modified_content = await self._evaluate_text(
                full_text, data, user_api_key_dict, hook_name, kind="prompt"
            )

            if modified_content != full_text:
                for message in reversed(data.get("messages", [])):
                    if message.get("role") != "user":
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        message["content"] = modified_content
                    elif isinstance(content, list):
                        # Replace text in first text block only, preserve tool_result/etc.
                        replaced = False
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                if not replaced:
                                    block["text"] = modified_content
                                    replaced = True
                                else:
                                    block["text"] = ""
                        if not replaced:
                            content.append({"type": "text", "text": modified_content})
                    else:
                        message["content"] = modified_content
                    break

            return data

        except Exception as e:
            self._handle_error(e, hook_name)
            return data

    @staticmethod
    def _sanitize_messages(data: dict[str, Any]) -> dict[str, Any]:
        """Remove empty text content blocks that Bedrock Converse API rejects.

        Also drops thinking blocks (LiteLLM mistranslates them to empty text on
        Bedrock) and whitespace-only text blocks. Sanitizes nested tool_result
        content arrays.
        """

        def _is_empty_text_block(b: Any) -> bool:
            if not isinstance(b, dict):
                return False
            if b.get("type") != "text":
                return False
            txt = b.get("text") or ""
            return not txt.strip()

        def _sanitize_block_list(blocks: list) -> list:
            out = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "thinking":
                    continue
                if isinstance(b, dict) and b.get("type") == "tool_reference":
                    name = b.get("tool_name") or b.get("name") or "unknown"
                    b = {"type": "text", "text": f"Tool reference: {name}"}
                if _is_empty_text_block(b):
                    continue
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, list):
                        new_inner = _sanitize_block_list(inner)
                        if not new_inner:
                            new_inner = [{"type": "text", "text": " "}]
                        b = {**b, "content": new_inner}
                    elif isinstance(inner, str) and not inner.strip():
                        b = {**b, "content": " "}
                out.append(b)
            return out

        for message in data.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                if not content.strip():
                    message["content"] = " "
                continue
            if not isinstance(content, list):
                continue
            filtered = _sanitize_block_list(content)
            if not filtered:
                filtered = [{"type": "text", "text": " "}]
            message["content"] = filtered
        return data

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> dict[str, Any]:
        """Evaluate user prompt before sending to LLM."""
        logger.debug(
            "async_pre_call_hook called: user_api_key_dict=%s cache=%s data=%s call_type=%s",
            user_api_key_dict,
            cache,
            data,
            call_type,
        )
        _dump_request("pre_call_request", data)
        data = self._sanitize_messages(data)
        _dump_request("pre_call_sanitized", data)
        return await self._evaluate_prompt(data, user_api_key_dict, "pre_call")

    async def async_moderation_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> dict[str, Any]:
        """Evaluate user prompt in parallel with the LLM call (during_call hook)."""
        logger.debug(
            "async_moderation_hook called: data=%s user_api_key_dict=%s call_type=%s",
            data,
            user_api_key_dict,
            call_type,
        )
        _dump_request("during_call_request", data)
        return await self._evaluate_prompt(data, user_api_key_dict, "during_call")

    async def async_post_call_success_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict[str, Any],
        response: ModelResponse,
    ) -> ModelResponse:
        """Evaluate LLM response before returning to user.

        Args:
            user_api_key_dict: User API key metadata
            data: Original request data
            response: LLM response object

        Returns:
            Modified response (with potentially masked content)

        Raises:
            HTTPException: If content is blocked (status 400)
            SafetyCheckUnavailable: If evaluation fails
        """
        logger.debug(
            "async_post_call_success_hook called: user_api_key_dict=%s data=%s response=%s",
            user_api_key_dict,
            data,
            response,
        )
        _dump_request("post_call_request", {"data": data, "response": str(response)})
        try:
            logger.debug("WonderFence post_call started")
            response_content = self._extract_response_content(response)
            if not response_content:
                logger.debug(
                    "WonderFence post_call: no response content found, skipping"
                )
                return response

            modified_content = await self._evaluate_text(
                response_content, data, user_api_key_dict, "post_call", kind="response"
            )

            if modified_content != response_content:
                response.choices[0].message.content = modified_content
                _dump_response_text(response, "post_call_response_masked")

            _dump_request("post_call_final", {"data": data, "response": str(response)})
            return response

        except Exception as e:
            self._handle_error(e, "post_call")
            return response

    async def _evaluate_text(
        self,
        text: str,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        hook_name: str,
        kind: str,
        chunk_count: int | None = None,
        dump_input: bool = True,
    ) -> str:
        """Send text to WonderFence and return possibly-modified content.

        Common layer for both directions: `kind` is "prompt" or "response" and
        selects the SDK call. Text larger than BUFFER_MAX_BYTES is split into
        overlapping sections (overlap = RESPONSE_SECTION_OVERLAP_BYTES) and
        evaluated concurrently (bounded by RESPONSE_SECTION_MAX_CONCURRENCY),
        so detections past the first 10K aren't missed. A BLOCK on any section
        raises (blocks the whole request/response). MASK across multiple
        sections is reassembled best-effort.

        `dump_input=False` suppresses the eval-text dump for the gated stream
        evals, which fire every ~200 bytes — dumping each would mean dozens of
        files per request, and their input is already accumulated in
        *_post_call_stream_eval_chunks.txt.
        """
        api_key = self._resolve_api_key(data, user_api_key_dict)
        app_id = self._resolve_app_id(data, user_api_key_dict)
        client = await self._get_client(api_key)
        context = self._build_analysis_context(data)

        if dump_input:
            _dump_eval_text(text, f"{hook_name}_eval_{kind}", _extract_session_id(data))

        sections = _split_overlapping_utf8(
            text, BUFFER_MAX_BYTES, RESPONSE_SECTION_OVERLAP_BYTES
        )
        sent_chars = sum(len(s) for s in sections)
        sent_bytes = sum(len(s.encode("utf-8")) for s in sections)
        chunk_note = (
            f" from {chunk_count} stream chunks" if chunk_count is not None else ""
        )
        print(
            f"[wonderfence] {hook_name} → evaluate_{kind}: {sent_chars} chars "
            f"({sent_bytes} bytes) sent to Alice in {len(sections)} Alice call(s){chunk_note} "
            f"[content={len(text)} chars]",
            flush=True,
        )
        logger.debug(
            "WonderFence %s sending to evaluate_%s "
            "(app_id=%s, context=%s, len=%d, sections=%d)",
            hook_name,
            kind,
            app_id,
            context,
            len(text),
            len(sections),
        )

        async def _evaluate_section(section: str) -> EvaluateMessageResponse:
            if kind == "prompt":
                return await client.evaluate_prompt(
                    app_id=app_id, prompt=section, context=context, custom_fields=None
                )
            return await client.evaluate_response(
                app_id=app_id, response=section, context=context, custom_fields=None
            )

        if len(sections) == 1:
            result = await _evaluate_section(sections[0])
            return self._handle_evaluation_result(result, kind, sections[0])

        sem = asyncio.Semaphore(RESPONSE_SECTION_MAX_CONCURRENCY)

        async def _eval_section(i: int, section: str) -> str:
            async with sem:
                logger.debug(
                    "WonderFence %s evaluate_%s section %d/%d (len=%d)",
                    hook_name,
                    kind,
                    i + 1,
                    len(sections),
                    len(section),
                )
                result = await _evaluate_section(section)
                # Raises HTTPException on BLOCK — gather propagates it, blocking
                # the whole request/response.
                return self._handle_evaluation_result(result, kind, section)

        # gather preserves order, so handled_parts aligns with sections.
        handled_parts = await asyncio.gather(
            *(_eval_section(i, s) for i, s in enumerate(sections))
        )

        if all(h == s for h, s in zip(handled_parts, sections)):
            return text

        logger.warning(
            "WonderFence %s: MASK across %d sections — reassembling best-effort "
            "(overlap=%d bytes)",
            hook_name,
            len(sections),
            RESPONSE_SECTION_OVERLAP_BYTES,
        )
        reassembled = handled_parts[0]
        for part in handled_parts[1:]:
            trimmed = part.encode("utf-8")[RESPONSE_SECTION_OVERLAP_BYTES:]
            reassembled += trimmed.decode("utf-8", errors="ignore")
        return reassembled

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        """Best-effort text extraction for per-chunk WonderFence eval."""
        if isinstance(chunk, (bytes, bytearray)):
            return WonderFenceGuardrail._extract_text_from_sse_bytes([bytes(chunk)])
        try:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content
        except (AttributeError, IndexError, TypeError):
            pass
        return ""

    @staticmethod
    def _track_open_block(chunk: Any, state: dict[str, int | None]) -> None:
        """Update client-visible content-block state as a chunk is RELEASED.

        `state` holds "open" (index of the block currently open on the client,
        or None) and "max" (highest content_block index started so far, -1 if
        none). Used by the gated path to aim the BLOCK frame at the right index
        instead of a hardcoded 0. No-op for non-bytes chunks.
        """
        if not isinstance(chunk, (bytes, bytearray)):
            return
        raw = bytes(chunk).decode("utf-8", errors="replace")
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(d, dict):
                continue
            t = d.get("type")
            if t == "content_block_start":
                idx = d.get("index", 0)
                state["open"] = idx
                if idx > (state["max"] if state["max"] is not None else -1):
                    state["max"] = idx
            elif t == "content_block_stop":
                if state["open"] == d.get("index", 0):
                    state["open"] = None

    def _build_block_frames(
        self,
        is_bytes_format: bool,
        released_any: bool,
        open_index: int | None,
        next_index: int,
    ) -> list[Any]:
        """Wire-format BLOCK frame(s) for mid-stream injection.

        - bytes (Anthropic SSE), released_any=True: close the client's open
          content block (open_index, if any) and open a fresh text block at
          next_index for the block_message (content_block_stop →
          content_block_start(text) → content_block_delta → content_block_stop →
          message_delta → message_stop). Required because with extended thinking
          the open block is a later index (e.g. 1), so a bare delta at index 0
          targets the already-stopped thinking block and the client rejects it.
        - bytes (Anthropic SSE), released_any=False (nothing streamed yet — the
          held batch still contains message_start): a complete minimal valid
          message (message_start → content_block_start → content_block_delta →
          content_block_stop → message_delta → message_stop) so the client
          isn't handed a bare delta with no message envelope.
        - ModelResponseStream (OpenAI-style): one chunk with delta.content set
          to block_message and finish_reason="content_filter" — self-contained
          either way.
        All bytes variants are joined into a single byte chunk.
        """
        if is_bytes_format:
            stop_event = {"type": "message_stop"}
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "stop_sequence", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
            if released_any:
                idx = next_index
                delta_event = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": self.block_message},
                }
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                }
                block_stop = {"type": "content_block_stop", "index": idx}
                sse = ""
                if open_index is not None:
                    close_open = {"type": "content_block_stop", "index": open_index}
                    sse += (
                        f"event: content_block_stop\ndata: {json.dumps(close_open)}\n\n"
                    )
                sse += (
                    f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                    f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
                    f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"
                )
                return [sse.encode("utf-8")]

            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": self.block_message},
            }
            start_event = {
                "type": "message_start",
                "message": {
                    "id": "msg_wonderfence_block",
                    "type": "message",
                    "role": "assistant",
                    "model": "",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            block_stop = {"type": "content_block_stop", "index": 0}
            sse = (
                f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
                f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
                f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"
            )
            return [sse.encode("utf-8")]

        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=self.block_message),
                    finish_reason="content_filter",
                )
            ],
        )
        return [chunk]

    @staticmethod
    def _extract_text_from_sse_bytes(chunks: list[bytes]) -> str:
        """Extract assistant text from Anthropic SSE byte chunks (/v1/messages)."""
        texts: list[str] = []
        try:
            raw = b"".join(chunks).decode("utf-8", errors="replace")
        except Exception:
            return ""
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    texts.append(delta.get("text", ""))
        return "".join(texts)

    async def async_post_call_streaming_iterator_hook(  # This is the litellm hook for streaming responses, which is called after the LLM has started streaming its response.
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Evaluate the streamed response, withholding chunks until they clear.

        Both gated modes hold chunks, eval every ~200 bytes, release the held
        batch on ALLOW, and discard it + inject a BLOCK frame otherwise. They
        differ only in what each eval carries:
        - stream_eval_mode=deltas (default): the current + previous ~200-byte
          window only — ~400 bytes per call, narrower detection context.
        - stream_eval_mode=rolling: a rolling 10K window — widest detection
          context, ~10K per call.

        stream_eval_mode=accumulate_all: buffer everything, eval once, re-yield.
        - ModelResponseStream chunks (/v1/chat/completions): assemble via
          stream_chunk_builder, re-stream via MockResponseIterator (BLOCK + MASK).
        - Raw SSE bytes (/v1/messages Anthropic): parse SSE, re-yield original
          bytes (BLOCK only — can't reliably rebuild SSE for MASK).
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        sid = _extract_session_id(request_data)
        text_path, raw_path, response_full_path, eval_buffers_path = (
            _make_chunk_log_paths(sid)
        )
        eval_idx = 0

        if self.stream_eval_mode is not StreamEvalMode.ACCUMULATE_ALL:
            logger.debug(
                "WonderFence post_call_stream: gated mode=%s "
                "(max_bytes=%d, increment=%d)",
                self.stream_eval_mode.value,
                BUFFER_MAX_BYTES,
                EVAL_BYTES_INCREMENT,
            )
            rb: _RollingBuffer | _DeltaWindows = (
                _DeltaWindows(EVAL_BYTES_INCREMENT)
                if self.stream_eval_mode is StreamEvalMode.DELTAS
                else _RollingBuffer(BUFFER_MAX_BYTES, EVAL_BYTES_INCREMENT)
            )
            pending: list[Any] = []  # buffered, not yet released, in order
            released_any = False
            chunk_count = 0
            is_bytes_format = None
            # Client-visible content-block state, updated only on release, so a
            # BLOCK frame targets the block the client actually has open.
            blockstate: dict[str, int | None] = {"open": None, "max": -1}

            async for chunk in response:
                _log_chunk(chunk_count, chunk, text_path, raw_path, response_full_path)
                chunk_count += 1
                if is_bytes_format is None:
                    is_bytes_format = isinstance(chunk, (bytes, bytearray))
                text = self._extract_chunk_text(chunk)
                if text:
                    rb.append_text(text)
                pending.append(chunk)

                if rb.needs_eval():
                    eval_idx += 1
                    _append_eval_buffer(eval_buffers_path, eval_idx, rb.text())
                    try:
                        modified = await self._evaluate_text(
                            rb.text(),
                            request_data,
                            user_api_key_dict,
                            "post_call_stream_gated",
                            kind="response",
                            chunk_count=chunk_count,
                            dump_input=False,
                        )
                        if modified != rb.text():
                            logger.warning(
                                "WonderFence post_call_stream_gated: MASK requested "
                                "mid-stream — unenforceable on gated batch, releasing "
                                "chunks as-is"
                            )
                        for c in pending:  # ALLOW → release the batch, in order
                            self._track_open_block(c, blockstate)
                            yield c
                        released_any = True
                        pending.clear()
                        rb.mark_evaluated()
                    except HTTPException as e:  # BLOCK → discard held batch
                        blocked_text = "".join(
                            self._extract_chunk_text(c) for c in pending
                        )
                        first_idx = chunk_count - len(pending)
                        print(
                            f"[wonderfence] BLOCKED & withheld chunks "
                            f"{first_idx}..{chunk_count - 1} ({len(pending)} chunks, "
                            f"{len(blocked_text)} chars): {blocked_text!r}",
                            flush=True,
                        )
                        logger.warning(
                            "WonderFence post_call_stream_gated: BLOCK, discarding "
                            "%d held chunks (%d..%d): %r detail=%s",
                            len(pending),
                            first_idx,
                            chunk_count - 1,
                            blocked_text,
                            e.detail,
                        )
                        max_started = blockstate["max"]
                        for frame in self._build_block_frames(
                            bool(is_bytes_format),
                            released_any,
                            open_index=blockstate["open"],
                            next_index=(max_started if max_started is not None else -1)
                            + 1,
                        ):
                            yield frame
                        return
                    except Exception as e:  # noqa: BLE001 — eval error → release
                        logger.error(
                            "WonderFence post_call_stream_gated eval failed "
                            "(releasing): %s",
                            e,
                        )
                        for c in pending:
                            self._track_open_block(c, blockstate)
                            yield c
                        released_any = True
                        pending.clear()
                        rb.mark_evaluated()

            # Stream ended with a sub-increment tail still held → now enforceable.
            if pending and rb.pending_bytes() > 0:
                eval_idx += 1
                _append_eval_buffer(eval_buffers_path, eval_idx, rb.text(), final=True)
                try:
                    await self._evaluate_text(
                        rb.text(),
                        request_data,
                        user_api_key_dict,
                        "post_call_stream_gated_final",
                        kind="response",
                        chunk_count=chunk_count,
                        dump_input=False,
                    )
                    for c in pending:
                        self._track_open_block(c, blockstate)
                        yield c
                except HTTPException as e:
                    blocked_text = "".join(self._extract_chunk_text(c) for c in pending)
                    first_idx = chunk_count - len(pending)
                    print(
                        f"[wonderfence] BLOCKED & withheld final chunks "
                        f"{first_idx}..{chunk_count - 1} ({len(pending)} chunks, "
                        f"{len(blocked_text)} chars): {blocked_text!r}",
                        flush=True,
                    )
                    logger.warning(
                        "WonderFence post_call_stream_gated_final: BLOCK, withholding "
                        "chunks %d..%d: %r detail=%s",
                        first_idx,
                        chunk_count - 1,
                        blocked_text,
                        e.detail,
                    )
                    max_started = blockstate["max"]
                    for frame in self._build_block_frames(
                        bool(is_bytes_format),
                        released_any,
                        open_index=blockstate["open"],
                        next_index=(max_started if max_started is not None else -1) + 1,
                    ):
                        yield frame
                    return
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "WonderFence post_call_stream_gated_final eval failed "
                        "(releasing): %s",
                        e,
                    )
                    for c in pending:
                        yield c
            elif pending:  # textless tail (e.g. trailing SSE stop events) → release
                for c in pending:
                    yield c

            logger.debug(
                "WonderFence post_call_stream (gated): chunks=%d is_bytes=%s",
                chunk_count,
                is_bytes_format,
            )
            return

        all_chunks: list[Any] = []
        async for chunk in response:  # blocks till we have the full response
            all_chunks.append(chunk)
            _log_chunk(
                len(all_chunks) - 1, chunk, text_path, raw_path, response_full_path
            )

        is_bytes = bool(all_chunks) and all(
            isinstance(c, (bytes, bytearray)) for c in all_chunks
        )
        total_len = sum(
            len(c) if isinstance(c, (bytes, bytearray, str)) else len(str(c))
            for c in all_chunks
        )
        logger.debug(
            "async_post_call_streaming_iterator_hook called: chunks=%d total_len=%d is_bytes=%s",
            len(all_chunks),
            total_len,
            is_bytes,
        )
        if is_bytes:
            chunks_dump: Any = b"".join(all_chunks).decode("utf-8", errors="replace")
        else:
            chunks_dump = all_chunks
        _dump_request(
            "post_call_stream",
            {
                "data": request_data,
                "chunk_count": len(all_chunks),
                "is_bytes": is_bytes,
                "chunks": chunks_dump,
            },
        )

        if not all_chunks:
            return

        if is_bytes:
            try:
                response_content = self._extract_text_from_sse_bytes(all_chunks)
                if not response_content:
                    logger.debug("WonderFence post_call_stream: no SSE text, skipping")
                else:
                    modified = await self._evaluate_text(
                        response_content,
                        request_data,
                        user_api_key_dict,
                        "post_call_stream",
                        kind="response",
                    )
                    if modified != response_content:
                        logger.warning(
                            "WonderFence post_call_stream: MASK requested on raw SSE — "
                            "cannot rebuild stream, passing original through"
                        )
            except Exception as e:
                self._handle_error(e, "post_call_stream")
            _dump_request(
                "post_call_stream_final",
                {
                    "data": request_data,
                    "chunk_count": len(all_chunks),
                    "is_bytes": True,
                    "chunks": b"".join(all_chunks).decode("utf-8", errors="replace"),
                },
            )
            for chunk in all_chunks:
                yield chunk
            return

        try:
            assembled = stream_chunk_builder(chunks=all_chunks)
        except Exception as e:
            logger.debug(
                "WonderFence post_call_stream: stream_chunk_builder failed (%s), passing through",
                e,
            )
            for chunk in all_chunks:
                yield chunk
            return

        if not isinstance(assembled, ModelResponse):
            logger.debug(
                "WonderFence post_call_stream: assembled is %s, passing through",
                type(assembled).__name__,
            )
            for chunk in all_chunks:
                yield chunk
            return

        try:
            response_content = self._extract_response_content(assembled)
            if not response_content:
                logger.debug("WonderFence post_call_stream: no content, skipping")
                for chunk in all_chunks:
                    yield chunk
                return

            modified_content = await self._evaluate_text(
                response_content,
                request_data,
                user_api_key_dict,
                "post_call_stream",
                kind="response",
            )

            final_content = modified_content
            _dump_request(
                "post_call_stream_final",
                {
                    "data": request_data,
                    "is_bytes": False,
                    "final_content": final_content,
                },
            )

            if final_content == response_content:
                for chunk in all_chunks:
                    yield chunk
                return

            assembled.choices[0].message.content = final_content
            mock = MockResponseIterator(model_response=assembled)
            async for chunk in mock:
                yield chunk

        except Exception as e:
            self._handle_error(e, "post_call_stream")
            for chunk in all_chunks:
                yield chunk
