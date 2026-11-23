
import argparse
import asyncio
import math
import statistics
import time

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Prospect AI L1 cache miss and hit latency."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier for repeatable prompts (for example, run-01).",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    return args


def health_url(completions_url: str) -> str:
    """Derive /health from the configured completions endpoint."""
    marker = "/v1/chat/completions"
    if marker in completions_url:
        return completions_url.rsplit(marker, 1)[0] + "/health"
    return completions_url.rstrip("/") + "/health"


def p95(values: list[float]) -> float:
    """Calculate nearest-rank p95 latency."""
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def print_metrics(label: str, latencies: list[float], success_count: int) -> dict | None:
    print(f"\n{label}")
    print(f"  Successful samples: {success_count}")
    if not latencies:
        print("  Mean: unavailable")
        print("  Median: unavailable")
        print("  Minimum: unavailable")
        print("  Maximum: unavailable")
        print("  p95: unavailable")
        return None

    metrics = {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "minimum": min(latencies),
        "maximum": max(latencies),
        "p95": p95(latencies),
    }
    print(f"  Mean: {metrics['mean']:.6f} seconds")
    print(f"  Median: {metrics['median']:.6f} seconds")
    print(f"  Minimum: {metrics['minimum']:.6f} seconds")
    print(f"  Maximum: {metrics['maximum']:.6f} seconds")
    print(f"  p95: {metrics['p95']:.6f} seconds")
    return metrics


async def check_health(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url)
        return response.status_code == 200
    except httpx.TimeoutException:
        return False
    except httpx.RequestError:
        return False


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    errors: dict[str, int],
) -> tuple[float | None, str | None]:
    """Send one request and return elapsed seconds and cache status."""
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - start
        errors["timeout"] = errors.get("timeout", 0) + 1
        print(f"  Request timeout after {elapsed:.6f} seconds")
        return None, None
    except httpx.RequestError as exc:
        elapsed = time.perf_counter() - start
        errors["network_error"] = errors.get("network_error", 0) + 1
        print(f"  Network error after {elapsed:.6f} seconds: {exc.__class__.__name__}")
        return None, None

    elapsed = time.perf_counter() - start
    if response.status_code != 200:
        key = f"http_{response.status_code}"
        errors[key] = errors.get(key, 0) + 1
        print(f"  HTTP error {response.status_code} after {elapsed:.6f} seconds")
        return None, None
    return elapsed, response.headers.get("X-Prospect-Cache")


async def run_iteration(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    errors: dict[str, int],
) -> tuple[str, float | None, float | None]:
    """Run one miss/hit pair and classify it without reusing payload objects."""
    miss_latency, miss_cache_status = await send_request(
        client, url, payload, errors
    )
    if miss_latency is None:
        print("  Cache miss: no successful sample")
        return "failed", None, None

    miss_status = (miss_cache_status or "").lower()
    if miss_status != "miss":
        print(
            "  Cache miss: unconfirmed "
            f"(X-Prospect-Cache={miss_cache_status or 'missing'})"
        )
        return "unconfirmed_miss", None, None

    print(f"  Cache miss: {miss_latency:.6f} seconds")

    # This is deliberately the exact same payload object and JSON representation.
    hit_latency, hit_cache_status = await send_request(
        client, url, payload, errors
    )
    if hit_latency is None:
        print("  Cache hit: no successful sample")
        return "miss_only", miss_latency, None

    if hit_cache_status and hit_cache_status.lower() == "l1":
        print(f"  Cache hit: {hit_latency:.6f} seconds")
        return "pair", miss_latency, hit_latency

    print(
        "  Cache hit: unconfirmed "
        f"(X-Prospect-Cache={hit_cache_status or 'missing'})"
    )
    return "unconfirmed", miss_latency, None


async def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"benchmark-{time.time_ns()}"
    miss_latencies: list[float] = []
    hit_latencies: list[float] = []
    pair_miss_latencies: list[float] = []
    pair_hit_latencies: list[float] = []
    successful_miss_count = 0
    successful_pairs = 0
    failed_iterations = 0
    miss_only_successes = 0
    unconfirmed_hits = 0
    unconfirmed_misses = 0
    errors: dict[str, int] = {}

    print(f"Run ID: {run_id}")
    print(f"Iterations: {args.iterations}")
    print(f"Model: {args.model}")
    print(f"Gateway URL: {args.url}")
    print(f"Timeout: {args.timeout:.1f} seconds")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        gateway_health_url = health_url(args.url)
        if not await check_health(client, gateway_health_url):
            print(
                "\nGateway health check failed.\n"
                f"Expected a 200 response from {gateway_health_url}.\n"
                "Start the gateway first, for example:\n"
                "  uvicorn run:app --host 127.0.0.1 --port 8000"
            )
            return

        print(f"Gateway healthy: {gateway_health_url}")

        for iteration in range(1, args.iterations + 1):
            prompt = (
                f"benchmark-{run_id}-iteration-{iteration}: "
                "Explain exactly one useful fact about database indexes."
            )
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": False,
            }

            print(f"\nIteration {iteration}/{args.iterations}")
            classification, miss_latency, hit_latency = await run_iteration(
                client, args.url, payload, errors
            )
            if miss_latency is not None:
                successful_miss_count += 1
                miss_latencies.append(miss_latency)

            if classification == "failed":
                failed_iterations += 1
            elif classification == "unconfirmed_miss":
                unconfirmed_misses += 1
            elif classification == "miss_only":
                miss_only_successes += 1
            elif classification == "unconfirmed":
                unconfirmed_hits += 1
            elif classification == "pair":
                successful_pairs += 1
                if hit_latency is not None:
                    hit_latencies.append(hit_latency)
                    pair_miss_latencies.append(miss_latency)
                    pair_hit_latencies.append(hit_latency)

    print_metrics(
        "Cache miss statistics", miss_latencies, len(miss_latencies)
    )
    print_metrics(
        "Cache hit statistics", hit_latencies, len(hit_latencies)
    )

    print("\nIteration classifications")
    print(f"  Successful miss-hit pairs: {successful_pairs}")
    print(f"  Failed iterations (miss did not succeed): {failed_iterations}")
    print(f"  Unconfirmed misses: {unconfirmed_misses}")
    print(f"  Miss-only successes: {miss_only_successes}")
    print(f"  Unconfirmed hits: {unconfirmed_hits}")

    if pair_miss_latencies and pair_hit_latencies:
        miss_mean = statistics.mean(pair_miss_latencies)
        hit_mean = statistics.mean(pair_hit_latencies)
        absolute_reduction = miss_mean - hit_mean
        percentage_reduction = (
            (absolute_reduction / miss_mean * 100) if miss_mean else 0.0
        )
        print("\nCache-hit latency improvement")
        print(
            f"  Mean absolute reduction: {absolute_reduction:.6f} seconds"
        )
        print(
            f"  Mean percentage reduction: {percentage_reduction:.2f}%"
        )

    if errors:
        print("\nRequest errors")
        for error_type, count in sorted(errors.items()):
            print(f"  {error_type}: {count}")
    else:
        print("\nRequest errors: none")

    print(
        "\nReminder: inspect the Prospect AI server logs for "
        "'Cached response' and 'L1 cache hit' entries to verify upstream "
        "Gemini storage and hit activity."
    )
    print(
        "The benchmark timings alone do not prove an exact upstream Gemini call count."
    )


if __name__ == "__main__":
    asyncio.run(main())
