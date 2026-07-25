import asyncio
import httpx
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class RequestResult:
    request_id: int
    latency: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str = ""


async def single_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    request_id: int,
    prompt: str,
    max_tokens: int,
    api_url: str
) -> RequestResult:
    async with semaphore:
        start = time.time()
        try:
            response = await client.post(
                f"{api_url}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens},
                timeout=120.0
            )
            latency = time.time() - start

            if response.status_code == 200:
                data = response.json()
                return RequestResult(
                    request_id=request_id,
                    latency=latency,
                    prompt_tokens=data["prompt_tokens"],
                    completion_tokens=data["completion_tokens"],
                    success=True
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    latency=latency,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            latency = time.time() - start
            return RequestResult(
                request_id=request_id,
                latency=latency,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(e)
            )


def calculate_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[index]


async def run_benchmark(
    api_url: str,
    num_users: int,
    num_requests: int,
    prompt: str,
    max_tokens: int
) -> dict:
    semaphore = asyncio.Semaphore(num_users)

    async with httpx.AsyncClient() as client:
        tasks = [
            single_request(client, semaphore, i, prompt, max_tokens, api_url)
            for i in range(num_requests)
        ]
        results: List[RequestResult] = await asyncio.gather(*tasks)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    latencies = [r.latency for r in successful]
    total_tokens = sum(r.completion_tokens for r in successful)
    total_time = sum(r.latency for r in successful)

    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_users": num_users,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "error_rate": len(failed) / num_requests if num_requests > 0 else 0,
        "latency_p50": calculate_percentile(latencies, 50),
        "latency_p95": calculate_percentile(latencies, 95),
        "latency_avg": sum(latencies) / len(latencies) if latencies else 0,
        "tokens_per_second": tokens_per_second,
        "requests": [asdict(r) for r in results]
    }

    return summary


def save_results(results: dict, engine_type: str) -> str:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = data_dir / f"results_{engine_type}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    return str(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Tester")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--engine", default="hf")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Explique machine learning em uma frase.")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        api_url=args.api_url,
        num_users=args.users,
        num_requests=args.requests,
        prompt=args.prompt,
        max_tokens=args.max_tokens
    ))

    filename = save_results(results, args.engine)
    print(f"Resultados salvos em: {filename}")
    print(f"Requisições: {results['successful_requests']}/{results['num_requests']}")
    print(f"P50: {results['latency_p50']:.2f}s | P95: {results['latency_p95']:.2f}s")
    print(f"Tokens/s: {results['tokens_per_second']:.2f}")
