"""Alice WonderFence Guardrail Integration for LiteLLM.

This module provides a custom guardrail implementation that integrates Alice WonderFence
with LiteLLM to evaluate user prompts and LLM responses for content safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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


def _tail_bytes_utf8(text: str, max_bytes: int) -> str:
    """Suffix of `text` whose UTF-8 encoding fits in `max_bytes`, no split codepoint."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode("utf-8", errors="ignore")


@dataclass
class _RollingBuffer:
    """Per-request rolling byte buffer for streaming response eval.

    `append_text` keeps only the last `max_bytes` and tracks bytes added since
    the last `mark_evaluated()`. `needs_eval()` flips true once that delta
    crosses `eval_increment` — one eval per chunk regardless of chunk size.
    """
    max_bytes: int
    eval_increment: int
    _buf: bytes = field(default=b"")
    _since_eval: int = 0

    def append_text(self, text: str) -> None:
        chunk = text.encode("utf-8")
        self._buf = (self._buf + chunk)[-self.max_bytes:]
        self._since_eval += len(chunk)

    def needs_eval(self) -> bool:
        return self._since_eval >= self.eval_increment

    def mark_evaluated(self) -> None:
        self._since_eval = 0

    def pending_bytes(self) -> int:
        return self._since_eval

    def text(self) -> str:
        return self._buf.decode("utf-8", errors="ignore")


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


def _extract_session_id(data: Any) -> str | None:
    """Try to extract session_id from request data."""
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
    return None


def _dump_request(hook_name: str, data: Any) -> None:
    """Save full request data to messages/<ts>[_<session_id>]_<hook_name>.json."""
    try:
        MESSAGES_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
        sid = _extract_session_id(data) or (
            _extract_session_id(data.get("data")) if isinstance(data, dict) else None
        )
        sid_part = f"_{sid}" if sid else ""
        dest = MESSAGES_DIR / f"{ts}{sid_part}_{hook_name}.json"
        safe = _safe_serialize(data)
        dest.write_text(json.dumps(safe, indent=2, default=str))
        print(f"[wonderfence] dumped {hook_name} -> {dest}", flush=True)
    except Exception as e:
        print(f"[wonderfence] DUMP FAILED {hook_name}: {e!r}", flush=True)
        logger.error("Failed to dump request for %s: %s", hook_name, e, exc_info=True)


def _dump_response_text(response: Any, tag: str = "response_text") -> None:
    """Save plain-text LLM response content to messages/<ts>_<tag>.txt."""
    try:
        text = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content or ""
        if not text:
            return
        MESSAGES_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
        dest = MESSAGES_DIR / f"{ts}_{tag}.txt"
        dest.write_text(text)
        print(f"[wonderfence] dumped {tag} -> {dest}", flush=True)
    except Exception as e:
        print(f"[wonderfence] DUMP {tag} FAILED: {e!r}", flush=True)


def _dump_latest_user_message(data: Any, tag: str = "user_request") -> None:
    """Save plain-text latest user message to messages/<ts>[_<sid>]_<tag>.txt."""
    try:
        if not isinstance(data, dict):
            return
        text = ""
        for message in reversed(data.get("messages", []) or []):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            tr = block.get("content", "")
                            if isinstance(tr, str):
                                parts.append(f"[tool_result] {tr}")
                            elif isinstance(tr, list):
                                for sub in tr:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        parts.append(f"[tool_result] {sub.get('text', '')}")
                    elif isinstance(block, str):
                        parts.append(block)
                text = "\n".join(parts)
            else:
                text = str(content)
            break
        if not text:
            return
        MESSAGES_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
        sid = _extract_session_id(data)
        sid_part = f"_{sid}" if sid else ""
        dest = MESSAGES_DIR / f"{ts}{sid_part}_{tag}.txt"
        dest.write_text(text)
        print(f"[wonderfence] dumped {tag} -> {dest}", flush=True)
    except Exception as e:
        print(f"[wonderfence] DUMP {tag} FAILED: {e!r}", flush=True)


def _make_chunk_log_paths(sid: str | None) -> tuple[Path, Path, Path]:
    """Return (text_path, raw_path, latest_path) for current streaming request.
    Filenames mirror _dump_request datetime-first scheme."""
    MESSAGES_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
    sid_part = f"_{sid}" if sid else ""
    text = MESSAGES_DIR / f"{ts}{sid_part}_chunks_text.jsonl"
    raw = MESSAGES_DIR / f"{ts}{sid_part}_chunks_raw.jsonl"
    latest = MESSAGES_DIR / f"{ts}{sid_part}_latest.txt"
    return text, raw, latest


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
    idx: int, chunk: Any, text_path: Path, raw_path: Path, latest_path: Path
) -> None:
    """Append one line to each per-request log + append running text + print to stdout.
    Best effort."""
    try:
        text = WonderFenceGuardrail._extract_chunk_text(chunk)
        raw = _serialize_chunk_raw(chunk)
        with text_path.open("a") as f:
            f.write(json.dumps({"i": idx, "text": text}) + "\n")
        with raw_path.open("a") as f:
            f.write(json.dumps({"i": idx, "raw": raw}) + "\n")
        if text:
            with latest_path.open("a") as f:
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
        debug: When True, set guardrail logger to DEBUG level. Defaults to False.
        max_cached_clients: Max SDK clients cached per guardrail instance, keyed by
            API key (defaults to WONDERFENCE_MAX_CACHED_CLIENTS env var, then 10).
        connection_pool_limit: Max connections in each SDK client's HTTP pool
            (defaults to WONDERFENCE_CONNECTION_POOL_LIMIT env var, then SDK default).
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
        stream_buffer_replacements: bool | None = None,
        stream_buffer_eval: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("WONDERFENCE_API_KEY")
        self.app_id = app_id or os.environ.get("WONDERFENCE_APP_ID")

        self.guardrail_name = guardrail_name
        self.block_message = block_message
        self.platform = platform
        self.fail_open = fail_open

        def _env_bool(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

        self.stream_buffer_replacements = (
            stream_buffer_replacements
            if stream_buffer_replacements is not None
            else _env_bool("WONDERFENCE_STREAM_BUFFER_REPLACEMENTS")
        )
        self.stream_buffer_eval = (
            stream_buffer_eval
            if stream_buffer_eval is not None
            else _env_bool("WONDERFENCE_STREAM_BUFFER_EVAL")
        )

        if debug:
            logger.setLevel(logging.DEBUG)
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
                logger.warning("Failed to close evicted WonderFence client", exc_info=True)
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
            request_data.get("metadata")
            or request_data.get("litellm_metadata")
            or {}
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

        # User ID from API key metadata (prefixed by framework)
        user_id = (
            metadata.get("user_api_key_end_user_id")
            or metadata.get("end_user_id")
            or metadata.get("user_id")
        )
        if user_id and len(user_id) > 100:
            user_id = user_id[:100]

        # Session ID: Users pass `litellm_session_id` in request extra_body
        session_id = (
            request_data.get("litellm_session_id")
            or metadata.get("litellm_session_id")
            or metadata.get("session_id")
        )

        return AnalysisContext(
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider=provider,
            platform=self.platform,
        )

    def _extract_user_message(self, data: dict[str, Any]) -> str:
        """Extract the latest user message from request data.

        Args:
            data: Request data containing messages array

        Returns:
            Content of the latest user message, or empty string if not found
        """
        messages = data.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    return "\n".join(parts)
                return str(content)
        return ""

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Flatten a single user-role message to plain text (string, text blocks,
        tool_result blocks). Mirrors _dump_latest_user_message coverage."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", "") or "")
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

    def _collect_user_buffer(self, data: dict[str, Any]) -> str:
        """Walk messages reverse-chrono, concat user-role text, return the
        UTF-8-safe suffix that fits in BUFFER_MAX_BYTES.

        Walks newest→oldest so the most recent user content is preserved when
        the buffer is full. The returned string is in chronological order
        (oldest kept message first, newest last).
        """
        messages = data.get("messages", []) or []
        collected: list[str] = []
        size = 0
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            text = self._message_text(message)
            if not text:
                continue
            collected.append(text)
            size += len(text.encode("utf-8"))
            if size >= BUFFER_MAX_BYTES:
                break
        if not collected:
            return ""
        joined = "\n".join(reversed(collected))
        return _tail_bytes_utf8(joined, BUFFER_MAX_BYTES)

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
                evaluation_type, correlation_id, evaluation_type, detection_summary, content[:100],
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
                    evaluation_type, correlation_id, detection_summary,
                )
                return action_text
            logger.warning(
                "WonderFence returned MASK action without action_text for %s: "
                "wonderfence_correlation_id=%s detections=[%s]",
                evaluation_type, correlation_id, detection_summary,
            )

        if action == Actions.DETECT.value:
            logger.warning(
                "WonderFence DETECTED violation in %s: wonderfence_correlation_id=%s detections=[%s] content=%s...",
                evaluation_type, correlation_id, detection_summary, content[:100],
            )
        else:
            logger.debug(
                "WonderFence ALLOWED %s: wonderfence_correlation_id=%s detections=[%s]",
                evaluation_type, correlation_id, detection_summary,
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
            prompt_buffer = self._collect_user_buffer(data)
            if not prompt_buffer:
                logger.debug("WonderFence %s: no user message found, skipping", hook_name)
                return data

            api_key = self._resolve_api_key(data, user_api_key_dict)
            app_id = self._resolve_app_id(data, user_api_key_dict)
            client = await self._get_client(api_key)
            context = self._build_analysis_context(data)

            logger.debug(
                "WonderFence %s sending to evaluate_prompt (app_id=%s, context=%s, len=%d): %s",
                hook_name,
                app_id,
                context,
                len(prompt_buffer),
                prompt_buffer,
            )

            result = await client.evaluate_prompt(
                app_id=app_id, prompt=prompt_buffer, context=context, custom_fields=None
            )

            modified_content = self._handle_evaluation_result(
                result, "prompt", prompt_buffer
            )

            if modified_content != prompt_buffer:
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
            user_api_key_dict, cache, data, call_type,
        )
        _dump_request("pre_call", data)
        _dump_latest_user_message(data, "user_request_before")
        data = self._apply_pre_call_replacements_to_data(data)
        _dump_latest_user_message(data, "user_request_after")
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
            data, user_api_key_dict, call_type,
        )
        _dump_request("during_call", data)
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
            user_api_key_dict, data, response,
        )
        _dump_request("post_call", {"data": data, "response": str(response)})
        _dump_response_text(response, "response_text_before")
        try:
            logger.debug("WonderFence post_call started")
            response_content = self._extract_response_content(response)
            if not response_content:
                logger.debug(
                    "WonderFence post_call: no response content found, skipping"
                )
                return response

            modified_content = await self._evaluate_response_text(
                response_content, data, user_api_key_dict, "post_call"
            )

            if modified_content != response_content:
                response.choices[0].message.content = modified_content

            current = self._extract_response_content(response)
            replaced = self._apply_post_call_replacements(current)
            if replaced != current:
                response.choices[0].message.content = replaced
                logger.debug("WonderFence post_call: applied post-call string replacements")

            _dump_response_text(response, "response_text_after")
            _dump_request("post_call_final", {"data": data, "response": str(response)})
            return response

        except Exception as e:
            self._handle_error(e, "post_call")
            return response

    async def _evaluate_response_text(
        self,
        response_content: str,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        hook_name: str,
    ) -> str:
        """Send response text to WonderFence and return possibly-modified content."""
        api_key = self._resolve_api_key(data, user_api_key_dict)
        app_id = self._resolve_app_id(data, user_api_key_dict)
        client = await self._get_client(api_key)
        context = self._build_analysis_context(data)

        logger.debug(
            "WonderFence %s sending to evaluate_response (app_id=%s, context=%s, len=%d): %s",
            hook_name, app_id, context, len(response_content), response_content,
        )
        response_content = response_content[:10240]
        result = await client.evaluate_response(
            app_id=app_id, response=response_content, context=context, custom_fields=None
        )
        return self._handle_evaluation_result(result, "response", response_content)

    @staticmethod
    def _apply_post_call_replacements(text: str) -> str:
        return text

    @staticmethod
    def _apply_pre_call_replacements(text: str) -> str:
        return text

    @staticmethod
    def _apply_pre_call_replacements_to_data(data: dict[str, Any]) -> dict[str, Any]:
        """Rewrite ALL messages in-place using pre_call replacements (all roles)."""
        repl = WonderFenceGuardrail._apply_pre_call_replacements
        for message in data.get("messages", []) or []:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = repl(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = repl(block.get("text", ""))
        return data

    @staticmethod
    def _replacement_overlap() -> int:
        """Max chars to hold back between chunks to catch cross-boundary patterns."""
        return max(len("Lior"), len("King")) - 1  # 3

    @staticmethod
    def _set_chunk_text(chunk: Any, text: str) -> None:
        """Set delta.content on a ModelResponseStream chunk (in-place)."""
        try:
            chunk.choices[0].delta.content = text
        except (AttributeError, IndexError, TypeError):
            pass

    @staticmethod
    def _apply_replacements_to_chunk(chunk: Any) -> Any:
        """Per-chunk replacement (no cross-boundary awareness). Used for bytes
        chunks and the buffered path."""
        if isinstance(chunk, (bytes, bytearray)):
            return bytes(chunk).replace(b"Lior", b"Tali").replace(b"King", b"Queen")
        try:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                delta.content = WonderFenceGuardrail._apply_post_call_replacements(content)
        except (AttributeError, IndexError, TypeError):
            pass
        return chunk

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

    def _build_block_frames(self, is_bytes_format: bool) -> list[Any]:
        """Wire-format BLOCK frame(s) for mid-stream injection.

        - bytes (Anthropic SSE): one content_block_delta with block_message
          + one message_stop, joined as a single byte chunk.
        - ModelResponseStream (OpenAI-style): one chunk with delta.content
          set to block_message and finish_reason="content_filter".
        """
        if is_bytes_format:
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": self.block_message},
            }
            stop_event = {"type": "message_stop"}
            sse = (
                f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
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

    async def _fire_and_forget_eval(
        self,
        text: str,
        request_data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        hook_name: str,
    ) -> None:
        """Run WonderFence eval without enforcing BLOCK/MASK (already streamed)."""
        try:
            await self._evaluate_response_text(text, request_data, user_api_key_dict, hook_name)
        except HTTPException as e:
            logger.warning(
                "WonderFence %s (fire-and-forget) would BLOCK but stream already released: %s",
                hook_name, e.detail,
            )
        except Exception as e:
            logger.error("WonderFence %s (fire-and-forget) failed: %s", hook_name, e)

    @staticmethod
    def _apply_replacements_to_sse_bytes(chunks: list[bytes]) -> list[bytes]:
        """Rewrite Anthropic SSE so post-call replacements survive cross-delta splits.

        Text deltas are buffered per content_block index, joined, run through
        `_apply_post_call_replacements`, then collapsed into a single text_delta
        per block (emitted at the position of that block's first text_delta).
        Subsequent text_deltas for the same block are dropped. Non-text events
        pass through unchanged.
        """
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        events = raw.split("\n\n")
        per_block_text: dict[int, str] = {}
        per_block_first_idx: dict[int, int] = {}
        parsed: list[tuple[str, Any]] = []  # (kind, payload)

        for i, ev in enumerate(events):
            if not ev.strip():
                parsed.append(("raw", ev))
                continue
            data_line = None
            for line in ev.split("\n"):
                if line.startswith("data: "):
                    data_line = line[6:]
                    break
            if data_line is None:
                parsed.append(("raw", ev))
                continue
            try:
                data = json.loads(data_line)
            except (json.JSONDecodeError, ValueError):
                parsed.append(("raw", ev))
                continue
            if (
                isinstance(data, dict)
                and data.get("type") == "content_block_delta"
                and isinstance(data.get("delta"), dict)
                and data["delta"].get("type") == "text_delta"
            ):
                idx = data.get("index", 0)
                per_block_text.setdefault(idx, "")
                per_block_text[idx] += data["delta"].get("text", "") or ""
                if idx not in per_block_first_idx:
                    per_block_first_idx[idx] = i
                parsed.append(("text_delta", (idx, data)))
            else:
                parsed.append(("raw", ev))

        replaced_text = {
            idx: WonderFenceGuardrail._apply_post_call_replacements(t)
            for idx, t in per_block_text.items()
        }

        out_events: list[str] = []
        for i, (kind, payload) in enumerate(parsed):
            if kind == "raw":
                out_events.append(payload)
                continue
            idx, data = payload
            if per_block_first_idx.get(idx) == i:
                data["delta"]["text"] = replaced_text.get(idx, "")
                out_events.append(
                    f"event: content_block_delta\ndata: {json.dumps(data)}"
                )
            # subsequent text_deltas for same block are dropped

        rebuilt = "\n\n".join(out_events)
        return [rebuilt.encode("utf-8")]

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

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Buffer streaming chunks, evaluate full response, then re-yield.

        Two paths:
        - ModelResponseStream chunks (/v1/chat/completions): assemble via
          stream_chunk_builder, run eval, re-stream via MockResponseIterator
          (supports BLOCK + MASK).
        - Raw SSE bytes (/v1/messages Anthropic): parse SSE, run eval, re-yield
          original bytes (supports BLOCK only — can't reliably rebuild SSE for MASK).
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        sid = _extract_session_id(request_data)
        text_path, raw_path, latest_path = _make_chunk_log_paths(sid)
        print(
            f"[wonderfence] chunk logs: text={text_path.name} raw={raw_path.name} "
            f"latest={latest_path.name}",
            flush=True,
        )

        if not self.stream_buffer_replacements:
            logger.debug(
                "WonderFence post_call_stream: streaming rolling-buffer mode "
                "(max_bytes=%d, increment=%d)",
                BUFFER_MAX_BYTES, EVAL_BYTES_INCREMENT,
            )
            rb = _RollingBuffer(BUFFER_MAX_BYTES, EVAL_BYTES_INCREMENT)
            chunk_count = 0
            is_bytes_format: bool | None = None

            async for chunk in response:
                _log_chunk(chunk_count, chunk, text_path, raw_path, latest_path)
                chunk_count += 1

                if is_bytes_format is None:
                    is_bytes_format = isinstance(chunk, (bytes, bytearray))

                text = self._extract_chunk_text(chunk)
                if text:
                    rb.append_text(text)

                if rb.needs_eval():
                    try:
                        modified = await self._evaluate_response_text(
                            rb.text(),
                            request_data,
                            user_api_key_dict,
                            "post_call_stream_rolling",
                        )
                        if modified != rb.text():
                            logger.warning(
                                "WonderFence post_call_stream_rolling: MASK requested "
                                "mid-stream — already-released bytes unenforceable, "
                                "passing chunk through"
                            )
                        rb.mark_evaluated()
                    except HTTPException as e:
                        logger.warning(
                            "WonderFence post_call_stream_rolling: BLOCK mid-stream, "
                            "injecting error frame. detail=%s",
                            e.detail,
                        )
                        for frame in self._build_block_frames(bool(is_bytes_format)):
                            yield frame
                        return
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "WonderFence post_call_stream_rolling eval failed (continuing): %s",
                            e,
                        )
                        rb.mark_evaluated()

                yield chunk

            if rb.pending_bytes() > 0:
                try:
                    await self._evaluate_response_text(
                        rb.text(),
                        request_data,
                        user_api_key_dict,
                        "post_call_stream_rolling_final",
                    )
                except HTTPException as e:
                    logger.warning(
                        "WonderFence post_call_stream_rolling_final: would BLOCK but "
                        "stream already released. detail=%s",
                        e.detail,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "WonderFence post_call_stream_rolling_final eval failed: %s", e
                    )

            print(
                f"[wonderfence] post_call_stream (rolling): chunks={chunk_count} "
                f"is_bytes={is_bytes_format}",
                flush=True,
            )
            return

        all_chunks: list[Any] = []
        async for chunk in response:    # blocks till we have the full response
            all_chunks.append(chunk)
            _log_chunk(len(all_chunks) - 1, chunk, text_path, raw_path, latest_path)

        is_bytes = bool(all_chunks) and all(isinstance(c, (bytes, bytearray)) for c in all_chunks)
        total_len = sum(
            len(c) if isinstance(c, (bytes, bytearray, str)) else len(str(c))
            for c in all_chunks
        )
        print(
            f"[wonderfence] post_call_stream: chunks={len(all_chunks)} "
            f"total_len={total_len} is_bytes={is_bytes}",
            flush=True,
        )
        logger.debug(
            "async_post_call_streaming_iterator_hook called: chunks=%d total_len=%d is_bytes=%s",
            len(all_chunks), total_len, is_bytes,
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
                    modified = await self._evaluate_response_text(
                        response_content, request_data, user_api_key_dict, "post_call_stream"
                    )
                    if modified != response_content:
                        logger.warning(
                            "WonderFence post_call_stream: MASK requested on raw SSE — "
                            "cannot rebuild stream, passing original through"
                        )
            except Exception as e:
                self._handle_error(e, "post_call_stream")
            logger.debug("WonderFence post_call_stream: applied post-call string replacements (SSE)")
            final_chunks = self._apply_replacements_to_sse_bytes(all_chunks)
            _dump_request(
                "post_call_stream_final",
                {
                    "data": request_data,
                    "chunk_count": len(final_chunks),
                    "is_bytes": True,
                    "chunks": b"".join(final_chunks).decode("utf-8", errors="replace"),
                },
            )
            for chunk in final_chunks:
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

            modified_content = await self._evaluate_response_text(
                response_content, request_data, user_api_key_dict, "post_call_stream"
            )

            final_content = self._apply_post_call_replacements(modified_content)
            if final_content != modified_content:
                logger.debug("WonderFence post_call_stream: applied post-call string replacements")

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
