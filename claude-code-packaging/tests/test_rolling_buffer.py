"""Unit tests for the rolling 10KB buffer (request + streaming response)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


# --------------------------- _RollingBuffer ---------------------------


def test_rolling_buffer_under_limit():
    from wonderfence_guardrail import _RollingBuffer

    rb = _RollingBuffer(max_bytes=10_000, eval_increment=200)
    rb.append_text("a" * 5_000)
    assert rb.text() == "a" * 5_000


def test_rolling_buffer_overflow_keeps_suffix():
    from wonderfence_guardrail import _RollingBuffer

    rb = _RollingBuffer(max_bytes=10_000, eval_increment=200)
    rb.append_text("X" * 5_000 + "Y" * 10_000)
    text = rb.text()
    assert len(text.encode("utf-8")) == 10_000
    assert text == "Y" * 10_000


def test_rolling_buffer_needs_eval_threshold():
    from wonderfence_guardrail import _RollingBuffer

    rb = _RollingBuffer(max_bytes=10_000, eval_increment=200)
    rb.append_text("a" * 199)
    assert not rb.needs_eval()
    rb.append_text("b")
    assert rb.needs_eval()
    rb.mark_evaluated()
    assert not rb.needs_eval()


def test_rolling_buffer_one_eval_per_large_chunk():
    """A single 1000-byte chunk increments `_since_eval` once → one eval."""
    from wonderfence_guardrail import _RollingBuffer

    rb = _RollingBuffer(max_bytes=10_000, eval_increment=200)
    rb.append_text("z" * 1000)
    assert rb.needs_eval()
    rb.mark_evaluated()
    assert not rb.needs_eval()


def test_rolling_buffer_utf8_boundary():
    """Trim must not split a multi-byte codepoint."""
    from wonderfence_guardrail import _RollingBuffer

    # Each emoji is 4 bytes in UTF-8 — fill past the boundary with them.
    emoji = "🐉"  # 4 bytes
    rb = _RollingBuffer(max_bytes=10_000, eval_increment=200)
    rb.append_text(emoji * 3000)  # 12_000 bytes
    text = rb.text()
    # Must decode cleanly: no partial codepoint preserved.
    text.encode("utf-8").decode("utf-8")
    # Total size ≤ 10_000.
    assert len(text.encode("utf-8")) <= 10_000
    # All preserved chars are full emoji.
    assert set(text) == {emoji}


# --------------------------- _tail_bytes_utf8 ---------------------------


def test_tail_bytes_utf8_under_limit():
    from wonderfence_guardrail import _tail_bytes_utf8

    assert _tail_bytes_utf8("hello", 100) == "hello"


def test_tail_bytes_utf8_trims_to_suffix():
    from wonderfence_guardrail import _tail_bytes_utf8

    text = "A" * 5 + "B" * 100
    out = _tail_bytes_utf8(text, 50)
    assert len(out.encode("utf-8")) == 50
    assert out == "B" * 50


def test_tail_bytes_utf8_no_partial_codepoint():
    from wonderfence_guardrail import _tail_bytes_utf8

    emoji = "🐉"  # 4 bytes
    text = emoji * 100
    out = _tail_bytes_utf8(text, 10)
    # 10 bytes = 2 full emoji + 2 garbage bytes; errors=ignore drops them.
    assert all(ch == emoji for ch in out)
    assert len(out.encode("utf-8")) <= 10


# --------------------------- _collect_user_buffer (request side) ---------


def _allow_result():
    return type(
        "R",
        (),
        {
            "action": type("A", (), {"value": "ALLOW"})(),
            "action_text": None,
            "detections": [],
            "correlation_id": "c-1",
        },
    )()


@pytest.mark.asyncio
async def test_evaluate_prompt_sends_latest_message_only(guardrail, mock_client):
    """Only the latest user message goes to WonderFence, not prior history."""
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())

    messages = [
        {"role": "user", "content": "OLD-" + ("a" * 2_000)},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "LATEST-" + ("b" * 2_000)},
    ]
    data = {"model": "gpt-4", "messages": messages}

    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    assert mock_client.evaluate_prompt.await_count == 1
    sent = mock_client.evaluate_prompt.await_args.kwargs["prompt"]
    assert sent.startswith("LATEST-")
    assert "OLD-" not in sent
    assert "a" not in sent


@pytest.mark.asyncio
async def test_evaluate_prompt_tail_caps_large_latest_message(guardrail, mock_client):
    """A latest message > 10K is tail-trimmed to 10000 bytes UTF-8."""
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())

    messages = [{"role": "user", "content": "x" * 5_000 + "y" * 10_000}]
    data = {"model": "gpt-4", "messages": messages}

    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    sent = mock_client.evaluate_prompt.await_args.kwargs["prompt"]
    assert len(sent.encode("utf-8")) == 10_000
    assert sent == "y" * 10_000


@pytest.mark.asyncio
async def test_collect_user_buffer_latest_only_with_blocks(guardrail):
    """Latest user message only; tool_result/text blocks flattened, prior msgs ignored."""
    data = {
        "messages": [
            {"role": "user", "content": "first"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "block-text"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "tr-text"}],
                    },
                ],
            },
        ],
    }
    buf = guardrail._collect_user_buffer(data)
    assert "first" not in buf
    assert "block-text" in buf
    assert "[tool_result] tr-text" in buf


# --------------------- _split_overlapping_utf8 (response sections) ---------


def test_split_single_section_when_under_limit():
    from wonderfence_guardrail import _split_overlapping_utf8

    assert _split_overlapping_utf8("hello", 10_000, 100) == ["hello"]


def test_split_sections_with_overlap():
    from wonderfence_guardrail import _split_overlapping_utf8

    text = "".join(chr(ord("a") + (i % 26)) for i in range(25_000))  # 25000 bytes ascii
    sections = _split_overlapping_utf8(text, 10_000, 100)
    # step = 9900 → starts 0, 9900, 19800 → 3 sections
    assert len(sections) == 3
    assert len(sections[0].encode("utf-8")) == 10_000
    assert len(sections[1].encode("utf-8")) == 10_000
    assert len(sections[2].encode("utf-8")) == 25_000 - 19_800
    # Overlap: last 100 bytes of section 0 == first 100 bytes of section 1.
    assert sections[0][-100:] == sections[1][:100]


def test_split_no_partial_codepoint():
    from wonderfence_guardrail import _split_overlapping_utf8

    text = "🐉" * 3_000  # 12_000 bytes
    sections = _split_overlapping_utf8(text, 10_000, 100)
    for sec in sections:
        sec.encode("utf-8").decode("utf-8")  # must not raise


@pytest.mark.asyncio
async def test_evaluate_response_sections_each(guardrail, mock_client):
    """A 25K response → evaluate_response called once per section (3)."""
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())

    big = "z" * 25_000
    out = await guardrail._evaluate_response_text(
        big, {"model": "gpt-4"}, user_api_key_dict=None, hook_name="post_call"
    )
    assert mock_client.evaluate_response.await_count == 3
    assert out == big  # nothing masked → original returned unchanged


@pytest.mark.asyncio
async def test_evaluate_response_block_in_later_section(guardrail, mock_client):
    """BLOCK on the 2nd section raises HTTPException for the whole response."""
    n = {"i": 0}

    async def _block_2nd(**kwargs):
        n["i"] += 1
        if n["i"] == 2:
            raise HTTPException(status_code=400, detail={"error": "blocked"})
        return _allow_result()

    mock_client.evaluate_response = AsyncMock(side_effect=_block_2nd)

    with pytest.raises(HTTPException):
        await guardrail._evaluate_response_text(
            "z" * 25_000, {"model": "gpt-4"}, user_api_key_dict=None, hook_name="post_call"
        )
    assert n["i"] == 2  # stopped at the blocking section


# --------------------------- Streaming response ---------------------------


def _modelresponse_stream_chunk(text: str):
    """Build a minimal ModelResponseStream chunk with `text` in delta.content."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text))],
    )


async def _agen(items):
    for it in items:
        yield it


@pytest.mark.asyncio
async def test_streaming_happy_path_eval_count(guardrail, mock_client):
    """60 chunks × 50 bytes = 3000 bytes / 200 increment ≈ 15 evals."""
    result = type(
        "R",
        (),
        {
            "action": type("A", (), {"value": "ALLOW"})(),
            "action_text": None,
            "detections": [],
            "correlation_id": "c-1",
        },
    )()
    mock_client.evaluate_response = AsyncMock(return_value=result)

    chunks = [_modelresponse_stream_chunk("x" * 50) for _ in range(60)]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    assert len(yielded) == 60
    # 60 * 50 = 3000 bytes; 3000 / 200 = 15 → exactly 15 evals (no final
    # safety-net call because pending_bytes is 0 after the last eval).
    assert mock_client.evaluate_response.await_count == 15


@pytest.mark.asyncio
async def test_streaming_block_midstream(guardrail, mock_client):
    """Raise HTTPException on 5th eval → chunks 1-4 yielded, then block frame."""
    call_count = {"n": 0}

    async def _raise_on_5th(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 5:
            raise HTTPException(
                status_code=400,
                detail={"error": "blocked", "wonderfence_correlation_id": "c-x"},
            )
        return type(
            "R",
            (),
            {
                "action": type("A", (), {"value": "ALLOW"})(),
                "action_text": None,
                "detections": [],
                "correlation_id": "c-1",
            },
        )()

    mock_client.evaluate_response = AsyncMock(side_effect=_raise_on_5th)

    # Each chunk is 200 bytes → 1 eval per chunk.
    chunks = [_modelresponse_stream_chunk("y" * 200) for _ in range(10)]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    # Yields: chunks 1..5 (chunk 5 triggers the 5th eval which raises BEFORE
    # the chunk is yielded) + 1 block frame.
    # Iterator order: extract_text → append → needs_eval? → eval (raise) → return
    # So chunk 5 itself is NOT yielded.
    assert len(yielded) == 5  # 4 raw chunks + 1 block frame
    block_frame = yielded[-1]
    # block frame is a ModelResponseStream with finish_reason=content_filter
    assert block_frame.choices[0].finish_reason == "content_filter"
    assert (
        guardrail.block_message
        in block_frame.choices[0].delta.content
    )


@pytest.mark.asyncio
async def test_streaming_block_midstream_bytes_sse(guardrail, mock_client):
    """Anthropic SSE bytes path: BLOCK → SSE block frame in raw bytes."""

    async def _raise(**kwargs):
        raise HTTPException(
            status_code=400,
            detail={"error": "blocked", "wonderfence_correlation_id": "c-x"},
        )

    mock_client.evaluate_response = AsyncMock(side_effect=_raise)

    sse_chunk = (
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"text_delta","text":"' + (b"a" * 250) + b'"}}\n\n'
    )

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen([sse_chunk]),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    # Eval raised on chunk #1, so 0 raw chunks + 1 block frame (bytes).
    assert len(yielded) == 1
    frame = yielded[0]
    assert isinstance(frame, (bytes, bytearray))
    text = bytes(frame).decode("utf-8")
    assert "content_block_delta" in text
    assert "message_stop" in text
    assert guardrail.block_message in text
