"""Tests for exact-match cache behavior through the FastAPI gateway."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prospect_ai.core.normalizer import payload_to_cache_key
from prospect_ai.domain.types import CacheEntry
from prospect_ai.infrastructure.server import OpenAIGateway
from prospect_ai.infrastructure.storage.memory import MemoryCacheBackend


def _request(prompt: str) -> dict:
    return {
        "model": "gemini-test",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.mark.asyncio
async def test_identical_requests_call_upstream_once_and_store_l1():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )

    assert gateway.storage is storage
    assert gateway.proxy.cache_backend is storage

    upstream = AsyncMock(
        return_value=(
            200,
            {
                "choices": [{"message": {"content": "Gemini response"}}],
                "usage": {"completion_tokens": 7},
            },
        )
    )

    payload = _request("What is the weather?")
    transport = httpx.ASGITransport(app=gateway.app)

    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post("/v1/chat/completions", json=payload)
            second = await client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Prospect-Cache"] == "miss"
    assert second.headers["X-Prospect-Cache"] == "l1"
    assert first.json()["choices"][0]["message"]["content"] == "Gemini response"
    assert second.json()["choices"][0]["message"]["content"] == "Gemini response"
    assert upstream.await_count == 1

    cache_key = payload_to_cache_key(payload)
    cached = await storage.get_by_hash(cache_key)
    assert cached is not None
    assert cached.response_text == "Gemini response"

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_different_requests_do_not_share_cached_response():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )

    upstream = AsyncMock(
        side_effect=[
            (
                200,
                {
                    "choices": [{"message": {"content": "Response A"}}],
                    "usage": {"completion_tokens": 3},
                },
            ),
            (
                200,
                {
                    "choices": [{"message": {"content": "Response B"}}],
                    "usage": {"completion_tokens": 4},
                },
            ),
        ]
    )

    transport = httpx.ASGITransport(app=gateway.app)
    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions", json=_request("Question A")
            )
            second = await client.post(
                "/v1/chat/completions", json=_request("Question B")
            )

    assert first.json()["choices"][0]["message"]["content"] == "Response A"
    assert second.json()["choices"][0]["message"]["content"] == "Response B"
    assert upstream.await_count == 2
    assert (await storage.stats())["entries"] == 2

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_endpoint_returns_503_after_upstream_503_and_never_caches():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    gateway.proxy.max_retries = 1

    upstream = AsyncMock(
        return_value=(
            503,
            {"error": "Gemini temporarily unavailable"},
        )
    )

    transport = httpx.ASGITransport(
        app=gateway.app,
        raise_app_exceptions=False,
    )
    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_request("Unavailable question"),
            )

    assert response.status_code == 503
    assert "X-Prospect-Cache" not in response.headers
    assert upstream.await_count == 1
    assert (await storage.stats())["entries"] == 0

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_endpoint_returns_504_for_upstream_timeout():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )

    transport = httpx.ASGITransport(
        app=gateway.app,
        raise_app_exceptions=False,
    )
    with patch.object(
        gateway.proxy,
        "_send_request",
        AsyncMock(side_effect=httpx.ReadTimeout("timed out")),
    ):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_request("Timeout question"),
            )

    assert response.status_code == 504
    assert (await storage.stats())["entries"] == 0

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_endpoint_returns_502_for_connection_failure():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )

    transport = httpx.ASGITransport(
        app=gateway.app,
        raise_app_exceptions=False,
    )
    with patch.object(
        gateway.proxy,
        "_send_request",
        AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    ):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_request("Connection question"),
            )

    assert response.status_code == 502
    assert (await storage.stats())["entries"] == 0

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_ten_identical_requests_share_one_upstream_call():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    upstream_started = asyncio.Event()
    release_upstream = asyncio.Event()

    async def send_upstream(_url, _headers, _body):
        upstream_started.set()
        await release_upstream.wait()
        return (
            200,
            {
                "choices": [{"message": {"content": "Shared Gemini response"}}],
                "usage": {"completion_tokens": 5},
            },
        )

    upstream = AsyncMock(side_effect=send_upstream)
    payload = _request("One simultaneous question")
    transport = httpx.ASGITransport(app=gateway.app)

    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            requests = [
                asyncio.create_task(
                    client.post("/v1/chat/completions", json=payload)
                )
                for _ in range(10)
            ]
            await asyncio.wait_for(upstream_started.wait(), timeout=1)
            release_upstream.set()
            responses = await asyncio.gather(*requests)

    assert upstream.await_count == 1
    assert all(response.status_code == 200 for response in responses)
    assert all(
        response.json()["choices"][0]["message"]["content"]
        == "Shared Gemini response"
        for response in responses
    )
    assert gateway.proxy._inflight == {}

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_different_cache_keys_run_upstream_concurrently():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    active_calls = 0
    maximum_active_calls = 0

    async def send_upstream(_url, _headers, body):
        nonlocal active_calls, maximum_active_calls
        active_calls += 1
        maximum_active_calls = max(maximum_active_calls, active_calls)
        await asyncio.sleep(0.05)
        active_calls -= 1
        prompt = body["messages"][0]["content"]
        return (
            200,
            {
                "choices": [{"message": {"content": f"Response for {prompt}"}}],
                "usage": {"completion_tokens": 3},
            },
        )

    upstream = AsyncMock(side_effect=send_upstream)
    transport = httpx.ASGITransport(app=gateway.app)

    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            responses = await asyncio.gather(*[
                client.post(
                    "/v1/chat/completions",
                    json=_request(f"Independent question {index}"),
                )
                for index in range(5)
            ])

    assert upstream.await_count == 5
    assert maximum_active_calls == 5
    assert all(response.status_code == 200 for response in responses)
    assert gateway.proxy._inflight == {}

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_waiters_receive_shared_failure_and_failure_is_not_cached():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    gateway.proxy.max_retries = 1
    upstream_started = asyncio.Event()
    release_upstream = asyncio.Event()

    async def send_upstream(_url, _headers, _body):
        upstream_started.set()
        await release_upstream.wait()
        return 503, {"error": "Gemini unavailable"}

    upstream = AsyncMock(side_effect=send_upstream)
    payload = _request("Shared failure question")
    transport = httpx.ASGITransport(
        app=gateway.app,
        raise_app_exceptions=False,
    )

    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            requests = [
                asyncio.create_task(
                    client.post("/v1/chat/completions", json=payload)
                )
                for _ in range(10)
            ]
            await asyncio.wait_for(upstream_started.wait(), timeout=1)
            release_upstream.set()
            responses = await asyncio.gather(*requests)

    assert upstream.await_count == 1
    assert all(response.status_code == 503 for response in responses)
    assert (await storage.stats())["entries"] == 0
    assert gateway.proxy._inflight == {}

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_failed_shared_request_can_be_retried_by_subsequent_request():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    gateway.proxy.max_retries = 1
    call_count = 0

    async def send_upstream(_url, _headers, _body):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 503, {"error": "Gemini unavailable"}
        return (
            200,
            {
                "choices": [{"message": {"content": "Retry succeeded"}}],
                "usage": {"completion_tokens": 2},
            },
        )

    upstream = AsyncMock(side_effect=send_upstream)
    payload = _request("Retry this question")
    transport = httpx.ASGITransport(
        app=gateway.app,
        raise_app_exceptions=False,
    )

    with patch.object(gateway.proxy, "_send_request", upstream):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            failed = await client.post("/v1/chat/completions", json=payload)
            retried = await client.post("/v1/chat/completions", json=payload)

    assert failed.status_code == 503
    assert retried.status_code == 200
    assert retried.json()["choices"][0]["message"]["content"] == "Retry succeeded"
    assert upstream.await_count == 2
    assert gateway.proxy._inflight == {}

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_cancelling_waiter_does_not_cancel_shared_operation():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
    )
    upstream_started = asyncio.Event()
    release_upstream = asyncio.Event()

    async def send_upstream(_url, _headers, _body):
        upstream_started.set()
        await release_upstream.wait()
        return (
            200,
            {
                "choices": [{"message": {"content": "Still shared"}}],
                "usage": {"completion_tokens": 2},
            },
        )

    upstream = AsyncMock(side_effect=send_upstream)
    payload = _request("Cancellation question")
    cache_key = payload_to_cache_key(payload)

    with patch.object(gateway.proxy, "_send_request", upstream):
        cancelled_waiter = asyncio.create_task(
            gateway.proxy.call_upstream(payload, cache_key)
        )
        await asyncio.wait_for(upstream_started.wait(), timeout=1)
        surviving_waiter = asyncio.create_task(
            gateway.proxy.call_upstream(payload, cache_key)
        )

        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter

        release_upstream.set()
        result = await asyncio.wait_for(surviving_waiter, timeout=1)

    assert result[0] == "Still shared"
    assert upstream.await_count == 1
    assert gateway.proxy._inflight == {}

    await gateway.proxy.close()


@pytest.mark.asyncio
async def test_l2_hit_exposes_cache_status_header():
    storage = MemoryCacheBackend()
    gateway = OpenAIGateway(
        storage_backend=storage,
        upstream_base_url="https://example.test/v1",
        upstream_api_key="test-key",
        similarity_threshold=0.92,
    )
    await storage.store(
        CacheEntry(
            request_hash="preexisting-l2-entry",
            embedding_vector=[1.0, 0.0],
            response_text="Semantic response",
            finish_reason="stop",
            tokens_used=4,
            cached_at=time.time(),
        )
    )

    class EmbeddingStub:
        def embed(self, _text):
            return [1.0, 0.0]

    transport = httpx.ASGITransport(app=gateway.app)
    with patch(
        "prospect_ai.infrastructure.server.get_embedder",
        return_value=EmbeddingStub(),
    ):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_request("Semantically matching question"),
            )

    assert response.status_code == 200
    assert response.headers["X-Prospect-Cache"] == "l2"
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "Semantic response"
    )

    await gateway.proxy.close()
