"""Tests for benchmark miss/hit classification."""

import httpx
import pytest

import benchmark


@pytest.mark.asyncio
async def test_failed_miss_skips_corresponding_hit():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="upstream unavailable")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        classification, miss_latency, hit_latency = await benchmark.run_iteration(
            client,
            "http://gateway.test/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "prompt"}]},
            {},
        )

    assert classification == "failed"
    assert calls == 1
    assert miss_latency is None
    assert hit_latency is None


@pytest.mark.asyncio
async def test_confirmed_l1_hit_forms_complete_pair():
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "same prompt"}],
    }
    observed_payloads = []

    async def handler(request):
        observed_payloads.append(request.read())
        if len(observed_payloads) == 1:
            return httpx.Response(
                200,
                headers={"X-Prospect-Cache": "miss"},
                json={"choices": [{"message": {"content": "upstream"}}]},
            )
        return httpx.Response(
            200,
            headers={"X-Prospect-Cache": "l1"},
            json={"choices": [{"message": {"content": "cached"}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        classification, miss_latency, hit_latency = await benchmark.run_iteration(
            client,
            "http://gateway.test/v1/chat/completions",
            payload,
            {},
        )

    assert classification == "pair"
    assert miss_latency is not None
    assert hit_latency is not None
    assert observed_payloads[0] == observed_payloads[1]


@pytest.mark.asyncio
async def test_repeated_request_without_l1_header_is_unconfirmed():
    call_count = 0

    async def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                headers={"X-Prospect-Cache": "miss"},
                json={"ok": True},
            )
        return httpx.Response(
            200,
            headers={"X-Prospect-Cache": "miss"},
            json={"ok": True},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        classification, miss_latency, hit_latency = await benchmark.run_iteration(
            client,
            "http://gateway.test/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "prompt"}]},
            {},
        )

    assert classification == "unconfirmed"
    assert miss_latency is not None
    assert hit_latency is None
