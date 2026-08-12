"""Unit tests for the WonderFence eval buffers (request + streaming response)."""

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


# --------------------------- _DeltaWindows ---------------------------


def test_delta_windows_pairs_current_and_previous():
    from wonderfence_guardrail import _DeltaWindows

    dw = _DeltaWindows(eval_increment=200)
    dw.append_text("A" * 200)
    assert dw.needs_eval()
    assert dw.text() == "A" * 200  # first window has no predecessor
    dw.mark_evaluated()

    dw.append_text("B" * 200)
    assert dw.text() == "A" * 200 + "B" * 200  # previous + current
    dw.mark_evaluated()

    dw.append_text("C" * 200)
    assert dw.text() == "B" * 200 + "C" * 200  # "A" has rotated out


def test_delta_windows_pending_bytes_resets_on_rotation():
    from wonderfence_guardrail import _DeltaWindows

    dw = _DeltaWindows(eval_increment=200)
    dw.append_text("z" * 150)
    assert not dw.needs_eval()
    assert dw.pending_bytes() == 150
    dw.append_text("z" * 50)
    assert dw.needs_eval()
    dw.mark_evaluated()
    assert dw.pending_bytes() == 0
    assert not dw.needs_eval()


def test_delta_windows_single_oversized_chunk():
    """One 25K chunk → one oversized text(), which the common layer then splits."""
    from wonderfence_guardrail import _DeltaWindows

    dw = _DeltaWindows(eval_increment=200)
    dw.append_text("z" * 25_000)
    assert dw.needs_eval()
    assert len(dw.text().encode("utf-8")) == 25_000


def test_delta_windows_utf8_roundtrip():
    """Windows close on chunk boundaries, so no codepoint is ever split."""
    from wonderfence_guardrail import _DeltaWindows

    emoji = "🐉"  # 4 bytes
    dw = _DeltaWindows(eval_increment=200)
    dw.append_text(emoji * 51)  # 204 bytes
    assert dw.needs_eval()
    assert dw.text() == emoji * 51
    dw.mark_evaluated()
    dw.append_text(emoji * 51)
    assert dw.text() == emoji * 102


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
async def test_evaluate_prompt_sections_large_latest_message(guardrail, mock_client):
    """A 15K latest message is split into sections covering ALL of it — the head
    is evaluated too, not tail-trimmed away."""
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())

    full = "x" * 5_000 + "y" * 10_000  # 15_000 bytes
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": full}]}

    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    # step = 10_000 - 100 → starts 0, 9_900 → 2 sections.
    assert mock_client.evaluate_prompt.await_count == 2
    sent = [c.kwargs["prompt"] for c in mock_client.evaluate_prompt.await_args_list]
    assert sent[0] == full[:10_000]  # first section starts at the head
    assert sent[0].startswith("x" * 5_000)
    assert sent[1] == full[9_900:]
    # Union of the sections covers every byte of the message.
    assert sent[0] + sent[1][100:] == full


@pytest.mark.asyncio
async def test_evaluate_prompt_oversized_sections_each(guardrail, mock_client):
    """A 25K latest message → 3 evaluate_prompt calls covering the whole thing."""
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())

    full = "".join(chr(ord("a") + (i % 26)) for i in range(25_000))
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": full}]}

    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    assert mock_client.evaluate_prompt.await_count == 3
    sent = [c.kwargs["prompt"] for c in mock_client.evaluate_prompt.await_args_list]
    assert sent[0] + sent[1][100:] + sent[2][100:] == full


@pytest.mark.asyncio
async def test_evaluate_prompt_block_in_later_section(guardrail, mock_client):
    """BLOCK on the 2nd prompt section raises for the whole request."""
    n = {"i": 0}

    async def _block_2nd(**kwargs):
        n["i"] += 1
        if n["i"] == 2:
            raise HTTPException(status_code=400, detail={"error": "blocked"})
        return _allow_result()

    mock_client.evaluate_prompt = AsyncMock(side_effect=_block_2nd)
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "z" * 25_000}]}

    with pytest.raises(HTTPException):
        await guardrail._evaluate_prompt(
            data, user_api_key_dict=None, hook_name="pre_call"
        )
    assert n["i"] == 3  # sections run in parallel — all dispatched, no short-circuit


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
    out = await guardrail._evaluate_text(
        big,
        {"model": "gpt-4"},
        user_api_key_dict=None,
        hook_name="post_call",
        kind="response",
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
        await guardrail._evaluate_text(
            "z" * 25_000,
            {"model": "gpt-4"},
            user_api_key_dict=None,
            hook_name="post_call",
            kind="response",
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

    await guardrail._evaluate_text(
        "z" * 25_000,
        {"model": "gpt-4"},
        user_api_key_dict=None,
        hook_name="post_call",
        kind="response",
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
async def test_gated_block_withholds_offending_batch(make_guardrail, mock_client):
    """Rolling mode: the batch that trips BLOCK is never yielded; earlier ones are."""
    # 200-byte chunks → 1 eval per chunk. BLOCK on the 3rd eval.
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(3))
    guardrail = make_guardrail(stream_eval_mode="rolling", debug=True)
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
async def test_prompt_eval_dump_is_untrimmed(
    make_guardrail, mock_client, tmp_path, monkeypatch
):
    """*_eval_prompt.txt holds the whole message (pre-split); no *_prompt_full.txt."""
    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "MESSAGES_DIR", tmp_path)
    mock_client.evaluate_prompt = AsyncMock(return_value=_allow_result())
    guardrail = make_guardrail(debug=True)

    full = "A" * 5_000 + "B" * 10_000  # 15_000 bytes > BUFFER_MAX_BYTES
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": full}]}
    await guardrail._evaluate_prompt(data, user_api_key_dict=None, hook_name="pre_call")

    assert list(tmp_path.glob("*_prompt_full.txt")) == []  # merged into eval_prompt
    eval_files = list(tmp_path.glob("*_pre_call_eval_prompt.txt"))
    assert len(eval_files) == 1
    assert eval_files[0].read_text() == full


@pytest.mark.asyncio
async def test_stream_eval_chunks_file_is_indexed(
    make_guardrail, mock_client, tmp_path, monkeypatch
):
    """One entry per EVAL — the rolling buffer sent to Alice, not per Claude chunk."""
    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "MESSAGES_DIR", tmp_path)
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
    guardrail = make_guardrail(stream_eval_mode="rolling", debug=True)

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
    # dump_input=False on the gated sites: no per-eval *_eval_response.txt files.
    assert list(tmp_path.glob("*_eval_response.txt")) == []
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


# --------------------------- Deltas streaming mode --------------------------


async def _run_stream(g, chunks):
    """Drive the streaming hook to completion, returning the yielded items."""
    out = []
    async for ch in g.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=_agen(chunks),
        request_data={"model": "gpt-4", "messages": []},
    ):
        out.append(ch)
    return out


def _sent_responses(mock_client):
    return [c.kwargs["response"] for c in mock_client.evaluate_response.await_args_list]


@pytest.mark.asyncio
async def test_deltas_sends_two_windows_not_whole_text(make_guardrail, mock_client):
    """Each eval carries the previous + current window (~2 increments), never the
    whole accumulated response."""
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
    g = make_guardrail(stream_eval_mode="deltas")
    chunks = [_modelresponse_stream_chunk(c * 200) for c in "ABCDE"]

    yielded = await _run_stream(g, chunks)

    assert yielded == chunks
    a, b, c, d, e = ("A" * 200, "B" * 200, "C" * 200, "D" * 200, "E" * 200)
    # First window has no predecessor; every later eval is exactly prev+cur.
    assert _sent_responses(mock_client) == [a, a + b, b + c, c + d, d + e]
    # Never the 1000-byte accumulated text.
    assert all(len(s) <= 400 for s in _sent_responses(mock_client))


@pytest.mark.asyncio
async def test_deltas_block_withholds_offending_batch(make_guardrail, mock_client):
    """Deltas mode: the batch that trips BLOCK is never yielded; earlier ones are."""
    mock_client.evaluate_response = AsyncMock(side_effect=_block_on_nth_eval(3))
    g = make_guardrail(stream_eval_mode="deltas")
    chunks = [_modelresponse_stream_chunk("y" * 200) for _ in range(6)]

    yielded = await _run_stream(g, chunks)

    assert len(yielded) == 3  # chunks 0,1 released + 1 block frame
    assert chunks[2] not in yielded
    assert yielded[-1].choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_deltas_final_tail_evaluates_prev_plus_cur(make_guardrail, mock_client):
    """Stream ends mid-window → the existing final-tail branch evals prev+cur."""
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
    g = make_guardrail(stream_eval_mode="deltas")
    chunks = [
        _modelresponse_stream_chunk("A" * 200),
        _modelresponse_stream_chunk("B" * 50),
    ]

    yielded = await _run_stream(g, chunks)

    assert yielded == chunks
    assert _sent_responses(mock_client) == ["A" * 200, "A" * 200 + "B" * 50]


@pytest.mark.asyncio
async def test_deltas_boundary_end_releases_textless_trailer(
    make_guardrail, mock_client
):
    """Stream ends on a window boundary with textless trailer chunks → no extra
    eval, trailers still released."""
    mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
    g = make_guardrail(stream_eval_mode="deltas")
    chunks = [
        _modelresponse_stream_chunk("A" * 200),
        _modelresponse_stream_chunk(""),  # e.g. a trailing stop event
    ]

    yielded = await _run_stream(g, chunks)

    assert yielded == chunks
    assert _sent_responses(mock_client) == ["A" * 200]  # no final-tail eval


# --------------------------- Stream eval mode config ------------------------


@pytest.mark.asyncio
async def test_env_mode_selects_eval_path(make_guardrail, mock_client, monkeypatch):
    """Each of the three env values drives a different eval payload for the same
    stream: rolling accumulates, deltas pairs two windows, accumulate_all evals
    the whole response once."""
    a, b, c = "A" * 200, "B" * 200, "C" * 200

    async def _run(mode):
        monkeypatch.setenv("WONDERFENCE_STREAM_EVAL_MODE", mode)
        mock_client.evaluate_response = AsyncMock(return_value=_allow_result())
        g = make_guardrail()
        await _run_stream(g, [_modelresponse_stream_chunk(t) for t in (a, b, c)])
        return _sent_responses(mock_client)

    assert await _run("rolling") == [a, a + b, a + b + c]
    assert await _run("deltas") == [a, a + b, b + c]
    assert await _run("accumulate_all") == [a + b + c]


def test_stream_eval_mode_default_is_deltas(make_guardrail, monkeypatch):
    from wonderfence_guardrail import StreamEvalMode

    monkeypatch.delenv("WONDERFENCE_STREAM_EVAL_MODE", raising=False)
    assert make_guardrail().stream_eval_mode is StreamEvalMode.DELTAS


def test_stream_eval_mode_unknown_value_raises_at_init(make_guardrail, monkeypatch):
    with pytest.raises(ValueError):
        make_guardrail(stream_eval_mode="bogus")

    monkeypatch.setenv("WONDERFENCE_STREAM_EVAL_MODE", "bogus")
    with pytest.raises(ValueError):
        make_guardrail()


# --------------------------- Request dumps ----------------------------------


def test_dumps_are_gated_on_the_debug_flag(make_guardrail, tmp_path, monkeypatch):
    """Every dump writer no-ops with debug=False and writes with debug=True."""
    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "MESSAGES_DIR", tmp_path)

    def _write_all():
        wf._dump_request("pre_call_request", {"messages": []})
        wf._dump_eval_text("prompt text", "pre_call_eval_prompt")
        wf._append_eval_buffer(tmp_path / "eval_chunks.txt", 1, "buffered")
        wf._log_chunk(
            0,
            _modelresponse_stream_chunk("chunk text"),
            tmp_path / "chunks_text.jsonl",
            tmp_path / "chunks_raw.jsonl",
            tmp_path / "response_full.txt",
        )

    make_guardrail(debug=False)
    _write_all()
    assert list(tmp_path.iterdir()) == []  # nothing written

    make_guardrail(debug=True)
    _write_all()
    written = {p.name for p in tmp_path.iterdir()}
    assert "eval_chunks.txt" in written
    assert {"chunks_text.jsonl", "chunks_raw.jsonl", "response_full.txt"} <= written
    assert any(n.endswith("_pre_call_request.json") for n in written)
    assert any(n.endswith("_pre_call_eval_prompt.txt") for n in written)


def test_debug_flag_decouples_from_litellm_log(make_guardrail):
    """`debug` is the only switch: it emits our DEBUG lines via our own handler
    instead of litellm's, which sits on the parent at LITELLM_LOG's level."""
    import logging

    from wonderfence_guardrail import logger

    make_guardrail(debug=True)
    assert logger.isEnabledFor(logging.DEBUG)
    assert logger.handlers  # own handler — LITELLM_LOG=INFO can't drop the records
    assert logger.propagate is False

    make_guardrail(debug=False)
    # Pinned to INFO, so LITELLM_LOG=DEBUG on the parent can't turn us back on.
    assert not logger.isEnabledFor(logging.DEBUG)


# --------------------------- AnalysisContext --------------------------------


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"user_api_key_user_id": "key-owner"}, "key-owner"),  # last-resort fallback
        (
            {"user_api_key_end_user_id": "end-user", "user_api_key_user_id": "owner"},
            "end-user",  # end-user still wins over the key owner
        ),
        ({}, None),
    ],
)
def test_context_user_id_falls_back_to_key_owner(
    guardrail, monkeypatch, metadata, expected
):
    from unittest.mock import Mock

    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "AnalysisContext", Mock())
    guardrail._build_analysis_context({"model": "gpt-4", "metadata": metadata})
    assert wf.AnalysisContext.call_args.kwargs["user_id"] == expected


# --------------------------- session id from headers ------------------------


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"litellm_session_id": "explicit"}, "explicit"),  # body wins
        (
            {
                "litellm_metadata": {
                    "headers": {"X-Claude-Code-Session-Id": "cc-sess-1"}
                }
            },
            "cc-sess-1",  # Claude Code's per-session header, case-insensitive
        ),
        (
            {
                "proxy_server_request": {
                    "headers": {"x-claude-code-session-id": "psr-1"}
                }
            },
            "psr-1",  # fallback location
        ),
        (
            {
                "litellm_metadata": {
                    "headers": {
                        "x-claude-code-session-id": "vendor",
                        "x-litellm-session-id": "litellm",
                    }
                }
            },
            "litellm",  # explicit litellm header outranks the generic one
        ),
        ({"litellm_metadata": {"headers": {"user-agent": "curl"}}}, None),
        ({}, None),
    ],
)
def test_extract_session_id_from_headers(data, expected):
    from wonderfence_guardrail import _extract_session_id

    assert _extract_session_id(data) == expected


def test_context_session_id_uses_header(guardrail, monkeypatch):
    """The SDK context picks up the header-derived session id too."""
    from unittest.mock import Mock

    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "AnalysisContext", Mock())
    guardrail._build_analysis_context(
        {
            "model": "gpt-4",
            "litellm_metadata": {"headers": {"x-claude-code-session-id": "sess-42"}},
        }
    )
    assert wf.AnalysisContext.call_args.kwargs["session_id"] == "sess-42"


# --------------------------- user id normalization --------------------------

_CC_BLOB = json.dumps(
    {"device_id": "d" * 64, "account_uuid": "", "session_id": "sess-1"}
)
_CC_BLOB_SIGNED_IN = json.dumps(
    {"device_id": "d" * 64, "account_uuid": "acct-9", "session_id": "sess-1"}
)


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"user_id": _CC_BLOB}, "d" * 64),  # not signed in → device_id
        ({"user_id": _CC_BLOB_SIGNED_IN}, "acct-9"),  # signed in → account_uuid
        ({"user_id": "plain-id"}, "plain-id"),  # plain strings pass through
        ({"user_id": "{not json"}, "{not json"),  # unparseable → as-is
        # Blob with no usable identity → fall through to the next candidate.
        (
            {"user_id": json.dumps({"session_id": "s"}), "user_api_key_user_id": "own"},
            "own",
        ),
        ({"user_id": _CC_BLOB, "user_api_key_end_user_id": "end"}, "end"),  # order kept
        ({"user_id": {"device_id": "x"}}, None),  # non-string → unusable
    ],
)
def test_user_id_normalization(guardrail, monkeypatch, metadata, expected):
    from unittest.mock import Mock

    import wonderfence_guardrail as wf

    monkeypatch.setattr(wf, "AnalysisContext", Mock())
    guardrail._build_analysis_context({"model": "gpt-4", "metadata": metadata})
    assert wf.AnalysisContext.call_args.kwargs["user_id"] == expected
