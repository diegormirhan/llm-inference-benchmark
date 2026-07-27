# LLM Inference Benchmark: AMD RX 9060XT (16GB) with ROCm

Comparative benchmark of four LLM inference engines on consumer AMD hardware. The goal is to measure throughput, latency, and VRAM behavior under concurrent load, with live telemetry from the GPU.

This project is built specifically for Linux systems running AMD GPUs via ROCm. It has not been tested on Windows or NVIDIA hardware.

## Hardware Requirements

- **GPU**: AMD Radeon RX 9060XT (16GB VRAM)
- **OS**: Linux (tested on native installation, not WSL)
- **Driver**: ROCm 6.x with `amdsmi` available
- **RAM**: 32GB+ recommended (for model loading and KV cache)

## Models

All models are from the Qwen 2.5 family to keep tokenizer consistency across engines:

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Base | `Qwen/Qwen2.5-3B-Instruct` | ~6.5GB (fp16) | HF baseline, vLLM, Speculative target |
| AWQ | `Qwen/Qwen2.5-3B-Instruct-AWQ` | ~2GB (4-bit) | Quantized inference |
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | ~1.2GB | Speculative decoding draft model |

## Engines

Four inference backends are implemented, each exposing the same API contract:

1. **HuggingFace Baseline**: Transformers with no optimizations. Serves as the reference point for measuring the gains from vLLM's features.

2. **vLLM**: PagedAttention for memory efficiency, prefix caching to reuse computation across similar prompts, and chunked prefill to avoid blocking on long inputs.

3. **AWQ (vLLM)**: Same vLLM engine but loading the pre-quantized 4-bit AWQ model. Trades inference speed for ~70% lower VRAM usage.

4. **Speculative Decoding (vLLM)**: Uses the 3B model as target and the 0.5B model as draft. The draft proposes tokens quickly, and the target verifies them in parallel. Effective when the draft's predictions are frequently correct.

## Architecture

```
streamlit run dashboard/app.py
├── Streamlit process (:8501)
│   └── Manages subprocess lifecycle
│
└── python main.py --engine <hf|vllm|awq|speculative> --api-only
    └── Litestar API (:8000)
        └── Loads one engine on GPU (owns VRAM)
            └── POST /generate → engine.generate() → GPU
```

The dashboard is the parent process. It starts the API as a subprocess, monitors its health via `/health`, and triggers load tests by spawning `load_tester.tester` as another subprocess. The tester uses `httpx.AsyncClient` with `asyncio.gather` to simulate concurrent users sending requests to the API.

All GPU memory is owned by the API process. The dashboard and tester run on CPU only.

## Project Structure

```
llm-inference-benchmark/
├── main.py                   # Orchestrator: starts API, optionally Streamlit
├── engines/
│   ├── base_engine.py        # Abstract base class
│   ├── hf_engine.py          # Motor 1: HuggingFace transformers
│   ├── vllm_engine.py        # Motor 2: vLLM with PagedAttention
│   ├── awq_engine.py         # Motor 3: vLLM + AWQ 4-bit quantization
│   ├── speculative_engine.py # Motor 4: vLLM + speculative decoding
│   └── api.py                # Litestar API with graceful shutdown
├── load_tester/
│   └── tester.py             # Concurrent load testing via httpx + asyncio
├── telemetry/
│   └── monitor.py            # GPU metrics via amdsmi (VRAM, power, clocks)
├── dashboard/
│   └── app.py                # Streamlit UI with live consoles and charts
└── data/                     # Results (JSON) and telemetry (CSV) per run
```

## How to Run

### Prerequisites

Install dependencies with ROCm support:

```bash
# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Install vLLM with ROCm
pip install vllm

# Other dependencies
pip install -r requirements.txt
```

Set your HuggingFace token if using gated models:

```bash
export HF_TOKEN=your_token_here
```

### Dashboard Mode (Recommended)

The dashboard manages the API lifecycle and provides a visual interface:

```bash
streamlit run dashboard/app.py
```

In the browser:
1. Select the engine in the sidebar
2. Click "Start API" and wait for the status to turn green
3. Configure benchmark parameters (users, requests, max tokens, prompt size)
4. Click "Run benchmark"
5. Watch the live consoles and telemetry in the "Consoles" tab
6. View aggregated results and charts in the "Results" tab

To switch engines: click "Stop API", select a different engine, then "Start API" again.

### CLI Mode

For headless or automated runs:

```bash
# Terminal 1: Start the API
python main.py --engine vllm --api-only

# Terminal 2: Run the benchmark
python -m load_tester.tester \
    --engine vllm \
    --users 10 \
    --requests 50 \
    --max-tokens 200 \
    --prompt-size long
```

Results are saved to `data/results_<engine>_<timestamp>.json` and telemetry to `data/telemetry_<engine>_<timestamp>.csv`.

## API Contract

All engines expose the same endpoints:

```
POST /generate
Request:  { "prompt": str, "max_tokens": int }
Response: {
    "generated_text": str,
    "engine_used": str,
    "prompt_tokens": int,
    "completion_tokens": int,
    "elapsed_seconds": float
}

GET /health
Response: {
    "status": "ok",
    "active_engine": str,
    "model": str
}
```

## Results

### Throughput and Latency

<!-- PLACEHOLDER: Insert screenshot from dashboard Results tab showing throughput and latency charts -->

The bar charts compare tokens/second and P50/P95 latency across all four engines under identical load conditions.

### GPU Telemetry

<!-- PLACEHOLDER: Insert screenshot showing VRAM usage, GPU utilization, power draw, and clock speeds over time -->

Time-series charts show how each engine uses GPU resources during the benchmark. The VRAM plot highlights the point where HuggingFace baseline runs out of memory under concurrency (visible as error spikes), while vLLM variants remain stable.

### Key Findings

- **AWQ** uses ~70% less VRAM than the fp16 baseline but is ~24% slower due to dequantization overhead.
- **vLLM** maintains consistent P50/P95 latency even under high concurrency, thanks to PagedAttention.
- **HuggingFace baseline** degrades rapidly with concurrent requests and hits OOM errors.
- **Speculative decoding** shows speedup when the draft model's predictions are frequently accepted by the target.
- **Warmup** is critical: the first request after startup is slow due to Triton kernel compilation. Subsequent requests are fast.

## Graceful Shutdown

The API implements signal handlers (SIGTERM, SIGINT) and atexit hooks to release GPU memory cleanly:

1. Signal received → `engine.shutdown()` is called
2. Engine drops model references, runs `gc.collect()`, calls `torch.cuda.empty_cache()`
3. Process exits via `os._exit(0)`

The dashboard sends SIGTERM to the entire process group (API + workers), ensuring no orphaned processes hold VRAM.

## License

MIT
