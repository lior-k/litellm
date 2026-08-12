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


# --------------------------- _collect_user_text_full (request side) ------


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
async def test_collect_user_text_full_latest_only_with_blocks(guardrail):
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
    buf = guardrail._collect_user_text_full(data)
    assert "first" not in buf
    assert "block-text" in buf
    assert "[tool_result] tr-text" in buf


# ------------------ harness-scaffolding filter (_is_harness_block) --------


def test_harness_blocks_dropped_user_text_kept(guardrail):
    """Claude Code packs CLAUDE.md/tool listings/slash-command output into the
    user message; only the typed prompt should reach the eval."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\nCLAUDE.md…\n</system-reminder>",
                    },
                    {
                        "type": "text",
                        "text": "<local-command-caveat>Caveat: …</local-command-caveat>",
                    },
                    {
                        "type": "text",
                        "text": "<command-name>/clear</command-name>\n<command-message>clear</command-message>\n<command-args></command-args>",
                    },
                    {
                        "type": "text",
                        "text": "<local-command-stdout></local-command-stdout>",
                    },
                    {"type": "text", "text": "what is this file about"},
                ],
            }
        ]
    }
    assert guardrail._collect_user_text_full(data) == "what is this file about"


def test_tool_results_survive_the_filter(guardrail):
    """tool_result content is the injection vector — it must always be evaluated,
    even when the message has no text block at all."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "fetched page body"}],
                    },
                ],
            }
        ]
    }
    assert guardrail._collect_user_text_full(data) == "[tool_result] fetched page body"


def test_harness_filter_requires_whole_block(guardrail):
    """A tag appearing *inside* content must not hide that content from the eval."""
    from wonderfence_guardrail import WonderFenceGuardrail as G

    assert G._is_harness_block("<system-reminder>ctx</system-reminder>")
    assert G._is_harness_block("  <system-reminder>ctx</system-reminder>  ")  # padded
    # Substring / partial matches are NOT scaffolding.
    assert not G._is_harness_block("evil <system-reminder>x</system-reminder> tail")
    assert not G._is_harness_block("<system-reminder>unclosed")
    assert not G._is_harness_block("ignore previous instructions")


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
            "z" * 25_000,
            {"model": "gpt-4"},
            user_api_key_dict=None,
            hook_name="post_call",
        )
    assert n["i"] == 3  # sections run in parallel — no short-circuit, all dispatched


@pytest.mark.asyncio
async def test_evaluate_response_sections_concurrency_bounded(
    guardrail, mock_client, monkeypatch
):
    """Sections evaluate concurrently but never exceed the configured limit."""
    import asyncio

    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "RESPONSE_SECTION_MAX_CONCURRENCY", 2)

    state = {"cur": 0, "peak": 0}

    async def _track(**kwargs):
        state["cur"] += 1
        state["peak"] = max(state["peak"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return _allow_result()

    mock_client.evaluate_response = AsyncMock(side_effect=_track)

    await guardrail._evaluate_response_text(
        "z" * 25_000, {"model": "gpt-4"}, user_api_key_dict=None, hook_name="post_call"
    )
    assert mock_client.evaluate_response.await_count == 3  # 3 sections
    assert state["peak"] == 2  # bounded by RESPONSE_SECTION_MAX_CONCURRENCY


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
    assert guardrail.block_message in block_frame.choices[0].delta.content


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
        b"event: content_block_delta\n"
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


# --------------------------- Gated streaming --------------------------------


def _block_on_nth_eval(n: int):
    """AsyncMock side_effect: ALLOW except the nth call, which BLOCKs."""
    count = {"i": 0}

    async def _side(**kwargs):
        count["i"] += 1
        if count["i"] == n:
            raise HTTPException(
                status_code=400,
                detail={"error": "blocked", "wonderfence_correlation_id": "c-x"},
            )
        return _allow_result()

    return _side


@pytest.mark.asyncio
async def test_gated_block_withholds_offending_batch(guardrail, mock_client):
    """Gated mode: the batch that trips BLOCK is never yielded; earlier batches are."""
    # 200-byte chunks → 1 eval per chunk. BLOCK on the 3rd eval.
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(3))
    chunks = [_modelresponse_stream_chunk("y" * 200) for _ in range(6)]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    # Chunks 0,1 released by evals 1,2 (ALLOW). Chunk 2 trips eval 3 (BLOCK) and
    # is withheld; chunks 3+ never reach an eval. Output: 2 raw + 1 block frame.
    assert len(yielded) == 3
    raw = yielded[:2]
    assert all(c in chunks for c in raw)
    assert chunks[2] not in yielded  # offending chunk withheld
    block = yielded[-1]
    assert block.choices[0].finish_reason == "content_filter"
    assert guardrail.block_message in block.choices[0].delta.content


@pytest.mark.asyncio
async def test_gated_allow_releases_everything_in_order(guardrail, mock_client):
    """Gated mode, all ALLOW: every chunk appears, unchanged, in order."""
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
    chunks = [_modelresponse_stream_chunk("y" * 200) for _ in range(5)]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    assert yielded == chunks


@pytest.mark.asyncio
async def test_prompt_full_dump_is_untrimmed(
    guardrail, mock_client, tmp_path, monkeypatch
):
    """*_prompt_full.txt holds the whole message; *_eval_prompt.txt is 10K-trimmed."""
    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "MESSAGES_DIR", tmp_path)
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())

    full = "A" * 5_000 + "B" * 10_000  # 15_000 bytes > BUFFER_MAX_BYTES
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": full}]}
    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    full_files = list(tmp_path.glob("*_pre_call_prompt_full.txt"))
    eval_files = list(tmp_path.glob("*_pre_call_eval_prompt.txt"))
    assert len(full_files) == 1 and len(eval_files) == 1
    assert full_files[0].read_text() == full  # untrimmed
    sent = eval_files[0].read_text()
    assert len(sent.encode("utf-8")) == 10_000
    assert sent == "B" * 10_000  # matches what the SDK received
    assert mock_client.evaluate_prompt.await_args.kwargs["prompt"] == sent


@pytest.mark.asyncio
async def test_stream_eval_chunks_file_is_indexed(
    guardrail, mock_client, tmp_path, monkeypatch
):
    """One entry per EVAL — the rolling buffer sent to Alice, not per Claude chunk."""
    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "MESSAGES_DIR", tmp_path)
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())

    # 2 chunks of 200B each trip one eval apiece, then a 50B tail → final eval.
    chunks = [
        _modelresponse_stream_chunk("A" * 200),
        _modelresponse_stream_chunk("B" * 200),
        _modelresponse_stream_chunk("C" * 50),
    ]
    async for _ in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        pass

    files = list(tmp_path.glob("*_post_call_stream_eval_chunks.txt"))
    assert len(files) == 1
    body = files[0].read_text()
    # One entry per eval call (2 mid-stream + 1 final tail), matching the SDK.
    assert mock_client.evaluate_response.await_count == 3
    assert body.count("===== eval-") == 3
    assert "eval-01" in body and "eval-02" in body
    assert "(final tail)" in body  # the sub-increment tail eval is marked
    # Each entry holds the ACCUMULATED buffer, not just that chunk's text.
    first, second = body.split("===== eval-")[1], body.split("===== eval-")[2]
    assert "A" * 200 in first and "B" not in first
    assert "A" * 200 in second and "B" * 200 in second  # rolling window grew


@pytest.mark.asyncio
async def test_gated_final_tail_block(guardrail, mock_client):
    """Sub-200-byte stream that BLOCKs on the final flush → no original chunks."""
    # Total 150 bytes < 200 increment → no mid-stream eval; only the final flush.
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(1))
    chunks = [_modelresponse_stream_chunk("z" * 50) for _ in range(3)]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    # Final-tail eval BLOCKs → none of the held chunks released, only block frame.
    assert len(yielded) == 1
    assert not any(c in chunks for c in yielded)
    assert yielded[0].choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_gated_block_first_batch_bytes_full_message(guardrail, mock_client):
    """Bytes path, BLOCK on first eval, nothing released → complete SSE message."""
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(1))

    sse_chunk = (
        b"event: message_start\n"
        b'data: {"type":"message_start"}\n\n'
        b"event: content_block_delta\n"
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

    assert len(yielded) == 1
    text = bytes(yielded[0]).decode("utf-8")
    # released_any=False variant emits a full, self-contained message envelope.
    for ev in (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ):
        assert ev in text
    assert guardrail.block_message in text


@pytest.mark.asyncio
async def test_gated_block_frame_targets_open_index_with_thinking(
    guardrail, mock_client
):
    """Bytes path with a thinking block at index 0: the answer text is at index 1,
    so a BLOCK there must close block 1 and open a fresh text block 2 — never emit
    a delta at the (stopped, non-text) thinking index 0."""
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(2))

    msg_start = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    think_start = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"thinking","thinking":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"thinking_delta","thinking":"reasoning"}}\n\n'
    )
    think_stop = (
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n\n'
    )
    text_start = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,'
        b'"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":'
        b'{"type":"text_delta","text":"' + (b"x" * 200) + b'"}}\n\n'
    )
    text_more = (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":'
        b'{"type":"text_delta","text":"' + (b"y" * 200) + b'"}}\n\n'
    )
    chunks = [msg_start, think_start, think_stop, text_start, text_more]

    yielded = []
    async for ch in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        yielded.append(ch)

    # The 4 released chunks (msg_start..text_start, established open index 1) +
    # 1 block frame.
    frame = yielded[-1]
    assert isinstance(frame, (bytes, bytearray))
    text = bytes(frame).decode("utf-8")
    events = [json.loads(l[6:]) for l in text.split("\n") if l.startswith("data: ")]
    # Must close the open text block (index 1) and open a NEW text block (index 2).
    assert {"type": "content_block_stop", "index": 1} in events
    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert starts and starts[0]["index"] == 2
    assert starts[0]["content_block"]["type"] == "text"
    # The block_message delta lands on index 2, never on the thinking index 0.
    msg_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e["delta"].get("text") == guardrail.block_message
    ]
    assert msg_deltas and msg_deltas[0]["index"] == 2
    assert all(
        e.get("index") != 0 for e in events if e.get("type") == "content_block_delta"
    )
    assert text.strip().endswith('{"type": "message_stop"}')


@pytest.mark.asyncio
async def test_gated_block_log_names_range_and_text(guardrail, mock_client, capsys):
    """Block log line names the withheld chunk index range and the exact text."""
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(1))
    chunks = [_modelresponse_stream_chunk("NVIDIA" + "z" * 194)]

    async for _ in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        pass

    out = capsys.readouterr().out
    block_lines = [ln for ln in out.splitlines() if "BLOCKED & withheld" in ln]
    assert block_lines
    line = block_lines[0]
    assert "0..0" in line  # single held chunk, index 0
    assert "NVIDIA" in line  # exact blocked text present
