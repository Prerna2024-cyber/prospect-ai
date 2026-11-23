
import asyncio
import time

import httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"

PAYLOAD = {
    "model": "gemini-3.5-flash-lite",
    "messages": [
        {
            "role": "user",
            "content": "Explain how a distributed message queue works in two sentences.",
        }
    ],
}


async def send_request(client, request_number):
    start = time.perf_counter()

    response = await client.post(URL, json=PAYLOAD)

    latency = time.perf_counter() - start

    return {
        "request": request_number,
        "status": response.status_code,
        "cache": response.headers.get("X-Prospect-Cache", "none"),
        "latency": round(latency, 3),
    }


async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            send_request(client, i)
            for i in range(1, 11)
        ]

        results = await asyncio.gather(*tasks)

    print("\nConcurrency Test Results")
    print("-" * 45)

    for result in results:
        print(result)

    print("\nSummary")
    print("Total requests:", len(results))
    print("Successful:", sum(r["status"] == 200 for r in results))
    print("Failed:", sum(r["status"] != 200 for r in results))


if __name__ == "__main__":
    asyncio.run(main())
