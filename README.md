# LLM Inference Benchmark: AMD GPU with ROCm

Comparative benchmark of four LLM inference engines on consumer AMD hardware. The goal is to measure throughput, latency, and VRAM behavior under concurrent load, with live telemetry from the GPU.

This project is built specifically for Linux systems running AMD GPUs via ROCm. It has not been tested on Windows or NVIDIA hardware.

## Hardware Requirements

- **GPU**: AMD GPU with at least 8GB VRAM
- **OS**: Linux (tested on native installation, not WSL)
- **Driver**: ROCm 6.x with `amdsmi` available
- **RAM**: 24GB+ recommended (for model loading and KV cache)

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

### Directory Tree

```
llm-inference-benchmark/
├── main.py                          # Entry point: orchestrates API and dashboard processes
├── requirements.txt                 # Python dependencies (litestar, vllm, streamlit, amdsmi)
├── .env                             # Environment variables (HF_TOKEN, ENGINE_TYPE, etc.)
├── .gitignore                       # Git ignore rules
│
├── engines/                         # Inference engine implementations
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings (model IDs, engine type, env vars)
│   ├── base_engine.py               # Abstract base class with load_model(), generate(), shutdown()
│   ├── hf_engine.py                 # HuggingFace transformers (baseline, no optimizations)
│   ├── vllm_engine.py               # vLLM with PagedAttention + prefix caching + chunked prefill
│   ├── awq_engine.py                # vLLM + AWQ 4-bit quantization (pre-quantized model)
│   ├── speculative_engine.py        # vLLM + speculative decoding (target 3B + draft 0.5B)
│   └── api.py                       # Litestar API server with graceful shutdown handlers
│
├── load_tester/                     # Concurrent load testing
│   ├── __init__.py
│   └── tester.py                    # httpx + asyncio: simulates N concurrent users
│
├── telemetry/                       # GPU monitoring
│   ├── __init__.py
│   └── monitor.py                   # amdsmi integration: VRAM, GPU%, power, clocks (500ms interval)
│
├── dashboard/                       # Streamlit web interface
│   ├── __init__.py
│   └── app.py                       # Parent process manager, live consoles, results visualization
│
└── data/                            # Runtime outputs (gitignored)
    ├── results_<engine>_<timestamp>.json    # Benchmark metrics (throughput, latency, error rate)
    ├── telemetry_<engine>_<timestamp>.csv   # GPU telemetry time series
    └── logs/
        ├── api.log                  # API server output (tailable in dashboard)
        └── tester.log               # Load tester output (tailable in dashboard)
```

### Component Responsibilities

**main.py** — Orchestrator
- Parses CLI arguments (`--engine`, `--api-only`)
- Starts the Litestar API as a subprocess with ROCm environment variables
- Optionally starts Streamlit dashboard (unless `--api-only` is set)
- Handles SIGTERM/SIGINT to gracefully terminate child processes
- Sets `PYTHONPATH` and `FLASH_ATTENTION_TRITON_AMD_ENABLE` for ROCm compatibility

**engines/config.py** — Configuration
- Loads settings from `.env` using `pydantic-settings`
- Defines model IDs (base, AWQ, target, draft)
- Validates engine type (`hf`, `vllm`, `awq`, `speculative`)
- Centralizes all configuration to avoid scattered `os.getenv()` calls

**engines/base_engine.py** — Abstract Interface
- Defines the contract all engines must implement:
  - `load_model()`: loads model weights into VRAM
  - `generate(prompt, max_tokens)`: performs inference, returns metrics
  - `warmup()`: optional pre-compilation of Triton kernels
  - `shutdown()`: releases VRAM (drops references, `gc.collect()`, `torch.cuda.empty_cache()`)
- All engines inherit from this base class

**engines/hf_engine.py** — HuggingFace Baseline
- Uses `transformers.AutoModelForCausalLM` with `torch.float16`
- No optimizations (baseline for comparison)
- Degrades under concurrency (no PagedAttention)
- Hits OOM errors with high concurrent load

**engines/vllm_engine.py** — vLLM Optimized
- Uses `vllm.AsyncLLMEngine` with:
  - `enable_prefix_caching=True`: reuses KV cache for common prompt prefixes
  - `enable_chunked_prefill=True`: splits long prompts to avoid blocking
  - `gpu_memory_utilization=0.85`: reserves 85% of VRAM for KV cache
- Maintains stable latency under high concurrency
- Spawns worker subprocesses for distributed execution

**engines/awq_engine.py** — AWQ Quantized
- Same vLLM engine but with `quantization="awq_marlin"`
- Loads pre-quantized 4-bit model from HuggingFace Hub
- Uses ~70% less VRAM than fp16 baseline
- ~24% slower due to dequantization overhead

**engines/speculative_engine.py** — Speculative Decoding
- Uses vLLM's `speculative_config` with:
  - Target model: `Qwen/Qwen2.5-3B-Instruct` (3B parameters)
  - Draft model: `Qwen/Qwen2.5-0.5B-Instruct` (0.5B parameters)
  - `num_speculative_tokens=5`: draft proposes 5 tokens, target verifies in parallel
- Effective when draft predictions are frequently correct
- Total VRAM: ~7.7GB (target + draft + KV cache)

**engines/api.py** — Litestar API Server
- Exposes `POST /generate` and `GET /health` endpoints
- Loads one engine at startup based on `ENGINE_TYPE` env var
- Calls `engine.warmup()` before accepting requests
- Implements graceful shutdown:
  - Signal handlers (SIGTERM, SIGINT) call `engine.shutdown()`
  - `atexit` hook ensures cleanup on normal exit
  - `os._exit(0)` bypasses uvicorn's redundant shutdown

**load_tester/tester.py** — Concurrent Load Generator
- Simulates N concurrent users via `httpx.AsyncClient` + `asyncio.gather`
- Sends POST requests to `/generate` endpoint
- Measures per-request latency (client-side)
- Collects metrics:
  - Throughput (tokens/second)
  - Latency percentiles (P50, P95, average)
  - Error rate (failed requests / total)
- Starts/stops `GPUMonitor` automatically during the test
- Saves results to `data/results_<engine>_<timestamp>.json`

**telemetry/monitor.py** — GPU Telemetry Collector
- Uses `amdsmi` to sample GPU metrics every 500ms
- Collects:
  - `vram_used_mb`: VRAM usage in megabytes
  - `gpu_percent`: GPU utilization percentage
  - `power_w`: socket power draw in watts
  - `gfx_clock_mhz`: graphics core clock speed
  - `mem_clock_mhz`: memory clock speed
- Runs in a background thread during load tests
- Saves time series to `data/telemetry_<engine>_<timestamp>.csv`
- Handles metric collection failures gracefully (logs warning, continues)

**dashboard/app.py** — Streamlit Web Interface
- Acts as the parent process (manages API and tester subprocesses)
- Sidebar controls:
  - Engine selection (dropdown)
  - Start/Stop API buttons
  - Benchmark parameters (users, requests, max_tokens, prompt_size)
- Main area with two tabs:
  - **Consoles**: live tail of `api.log` and `tester.log` with auto-scroll
  - **Results**: aggregated metrics and charts from `data/` directory
- Health monitoring: polls `/health` endpoint every 2s to detect API status
- Process management:
  - Starts API with `start_new_session=True` (isolated process group)
  - Sends SIGTERM to entire group on stop (kills API + vLLM workers)
  - Falls back to SIGKILL after timeout
- Results visualization:
  - KPI cards (tokens/second, P95 latency per engine)
  - Bar charts (throughput comparison, latency comparison)
  - Time series (VRAM, GPU%, power, clock speeds)
  - Run history table (all benchmark runs in `data/`)

### Data Flow

```
User → Streamlit Dashboard (:8501)
         ↓ (HTTP)
         ↓ "Start API"
         ↓
         ├─→ main.py --engine vllm --api-only
         │     ↓
         │     └─→ engines.api (Litestar :8000)
         │           ↓
         │           └─→ VLLMEngine.load_model() → GPU VRAM
         │
         ↓ (HTTP)
         ↓ "Run Benchmark"
         ↓
         └─→ load_tester.tester (subprocess)
               ↓
               ├─→ telemetry.monitor.start() → data/telemetry_*.csv
               │
               ├─→ httpx.AsyncClient × N users
               │     ↓ (HTTP POST /generate)
               │     ↓
               │     └─→ engines.api → VLLMEngine.generate() → GPU
               │
               └─→ Save metrics → data/results_*.json
```

### Environment Variables

Defined in `.env` (loaded by `engines/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace API token for model downloads |
| `MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Base model for HF and vLLM engines |
| `AWQ_MODEL` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | Pre-quantized AWQ model |
| `TARGET_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Target model for speculative decoding |
| `DRAFT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Draft model for speculative decoding |
| `ENGINE_TYPE` | `hf` | Active engine: `hf`, `vllm`, `awq`, or `speculative` |
| `MAX_CONCURRENT_REQUESTS` | `100` | Max concurrent requests per API instance |

### Key Design Decisions

1. **Single engine per process**: The API loads one engine at a time. Switching engines requires restarting the API. This avoids VRAM contention and simplifies resource management.

2. **Dashboard as parent process**: Streamlit owns the subprocess lifecycle. This allows clean shutdown (SIGTERM to process group) and prevents orphaned GPU processes.

3. **No streaming or TTFT**: This version measures end-to-end latency only. Streaming and time-to-first-token are not implemented (future work).

4. **ROCm-specific optimizations**: The project uses ROCm-native PyTorch and vLLM builds. CUDA graphs and Flash Attention are enabled where supported.

5. **Graceful shutdown**: All engines implement `shutdown()` to release VRAM explicitly. Signal handlers ensure cleanup even on abrupt termination.

## How to Run

### Prerequisites

Install dependencies with ROCm support:

```bash
# Install AMD drivers
sudo amdgpu-install

# Clone the repository
git clone https://github.com/diegormirhan/llm-inference-benchmark.git
cd llm-inference-benchmark

# Set up your Python virtual environment.
python3.14 -m venv .venv
source .venv/bin/activate

# Follow AMD guide to install Rocm according to your system requirements
https://rocm.docs.amd.com/en/latest/install/rocm.html

# Follow AMD guide to install Pytorch and Vllm packages according to your system requirements
https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html

# Other dependencies
pip install -r requirements.txt
```

Create your .env file and set the tokens:

```bash
HF_TOKEN=your-hf-token

MODEL_ID=Qwen/Qwen2.5-3B-Instruct
AWQ_MODEL = Qwen/Qwen2.5-3B-Instruct-AWQ
TARGET_MODEL = Qwen/Qwen2.5-3B-Instruct
DRAFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct

ENGINE_TYPE=hf

MAX_CONCURRENT_REQUESTS=100
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

## License

MIT
