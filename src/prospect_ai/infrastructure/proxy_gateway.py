"""Async HTTP proxy gateway for upstream LLM relay."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx

from prospect_ai.core.embedder import get_embedder
from prospect_ai.domain.types import CacheEntry
from prospect_ai.infrastructure.storage.base import CacheStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class ProxyError:
    """Proxy error result."""

    status_code: int
    detail: str


class ProxyGateway:
    """Proxy for upstream LLM provider relay."""

    def __init__(
        self,
        upstream_base_url: str = "https://api.openai.com/v1",
        upstream_api_key: str = "",
        cache_backend: CacheStorageBackend | None = None,
        similarity_threshold: float = 0.92,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ):
        """Initialize proxy."""
        self.upstream_base_url = upstream_base_url
        self.upstream_api_key = upstream_api_key
        self.cache_backend = cache_backend
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client = httpx.AsyncClient(timeout=timeout_seconds)
        self._inflight_lock = asyncio.Lock()
        self._inflight: dict[
            str,
            asyncio.Task[tuple[str, dict[str, int]] | ProxyError],
        ] = {}

    async def call_upstream(
        self,
        request_body: dict[str, Any],
        cache_key: str,
    ) -> tuple[str, dict[str, int]] | ProxyError:
        """Coalesce identical non-streaming requests by normalized cache key."""
        async with self._inflight_lock:
            task = self._inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._execute_upstream(request_body, cache_key)
                )
                self._inflight[cache_key] = task

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise

    async def _execute_upstream(
        self,
        request_body: dict[str, Any],
        cache_key: str,
    ) -> tuple[str, dict[str, int]] | ProxyError:
        """Execute one upstream operation for an inflight cache-key group."""
        try:
            return await self._execute_upstream_with_retries(request_body, cache_key)
        finally:
            async with self._inflight_lock:
                current = self._inflight.get(cache_key)
                if current is asyncio.current_task():
                    del self._inflight[cache_key]

    async def _execute_upstream_with_retries(
        self,
        request_body: dict[str, Any],
        cache_key: str,
    ) -> tuple[str, dict[str, int]] | ProxyError:
        """Call upstream with retry logic for a coalesced request."""
        url = f"{self.upstream_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.upstream_api_key}",
            "Content-Type": "application/json",
        }
        request_body = {k: v for k, v in request_body.items() if k != "stream"}
        request_body["stream"] = False

        last_status = 502
        for attempt in range(self.max_retries):
            try:
                status, response_data = await self._send_request(
                    url, headers, request_body
                )
            except httpx.TimeoutException:
                logger.error("Upstream request timed out")
                return ProxyError(504, "Upstream request timed out")
            except httpx.RequestError as exc:
                logger.error("Upstream connection failed: %s", exc)
                return ProxyError(502, "Failed to connect to upstream")

            if status == 200:
                content_obj = response_data.get("choices", [{}])[0]
                content = content_obj.get("message", {}).get("content", "")
                usage = response_data.get("usage", {})
                request_text = json.dumps(request_body.get("messages", []))
                await self._store_cache(cache_key, content, usage, request_text)
                return content, usage

            if status == 429:
                last_status = 429
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue

            if status >= 500:
                last_status = status
                logger.warning(f"Upstream error {status}. Retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

            logger.error(f"Upstream error {status}: {response_data}")
            return ProxyError(status, str(response_data.get("error", response_data)))

        logger.error(f"Max retries ({self.max_retries}) exhausted")
        if last_status == 429:
            return ProxyError(429, "Upstream rate limit exceeded after retries")
        if last_status == 503:
            return ProxyError(503, "Upstream service unavailable after retries")
        return ProxyError(502, "Upstream returned server errors after retries")

    async def stream_upstream(
        self,
        request_body: dict[str, Any],
        cache_key: str,
    ) -> AsyncGenerator[str, None]:
        """Stream response from upstream with SSE passthrough.

        Streaming is intentionally not coalesced: each client owns an independent
        async response generator, and sharing one generator across clients would
        allow each stream consumer to interfere with the other's delivery.
        """
        url = f"{self.upstream_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.upstream_api_key}",
            "Content-Type": "application/json",
        }
        request_body = {k: v for k, v in request_body.items() if k != "stream"}
        request_body["stream"] = True

        full_response = ""
        async with self.client.stream(
            "POST", url, json=request_body, headers=headers
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"Upstream error {response.status_code}: {error_text}")
                yield f"data: {{'error': 'Upstream error {response.status_code}'}}"
                return

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    data_str = line[6:].strip()

                    if data_str == "[DONE]":
                        if self.cache_backend and full_response:
                            request_text = json.dumps(request_body.get("messages", []))
                            await self._store_cache(
                                cache_key, full_response, {}, request_text
                            )
                        yield "data: [DONE]"
                        break

                    parse_ok, chunk = self._parse_sse_chunk(data_str)
                    if parse_ok:
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            full_response += content
                        yield f"data: {data_str}"
                    else:
                        logger.warning(f"Failed to parse SSE chunk: {data_str}")
                        yield f"data: {data_str}"

    async def _send_request(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        """Send HTTP request. Returns (status_code, response_dict)."""
        try:
            result = await self.client.post(url, json=body, headers=headers)
            if result.status_code == 200:
                return result.status_code, result.json()
            return result.status_code, {"error": result.text}
        except httpx.TimeoutException:
            raise
        except httpx.RequestError:
            raise

    def _parse_sse_chunk(self, data_str: str) -> tuple[bool, dict[str, Any]]:
        """Parse SSE JSON chunk. Returns (success, parsed_dict)."""
        parsed = json.loads(data_str)
        return True, parsed

    async def _store_cache(
        self,
        cache_key: str,
        response_text: str,
        usage: dict[str, int],
        request_text: str = "",
    ) -> None:
        """Store an upstream response in the cache before returning it."""
        if not self.cache_backend:
            return

        embedder = get_embedder()
        embedding = embedder.embed(request_text) if request_text else []

        entry = CacheEntry(
            request_hash=cache_key,
            embedding_vector=embedding,
            response_text=response_text,
            finish_reason="stop",
            tokens_used=usage.get("completion_tokens", 0),
            cached_at=time.time(),
        )
        await self.cache_backend.store(entry)
        logger.info("Cached response: %s...", cache_key[:8])

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
