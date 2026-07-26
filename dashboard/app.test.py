"""
LLM Inference Benchmark Dashboard (test version).

Streamlit is the parent process: it starts/stops `main.py --api-only` (API server
owning the GPU) and runs `load_tester.tester` as a background subprocess, tailing
both logs live. Results and telemetry charts are read from data/.
"""

import atexit
import json
import os
import re
import signal
import subprocess
import sys
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
API_LOG = LOG_DIR / "api.log"
TESTER_LOG = LOG_DIR / "tester.log"
API_URL = "http://localhost:8000"

ENGINE_LABELS = {
    "hf": "HuggingFace (baseline)",
    "vllm": "vLLM (PagedAttention)",
    "awq": "vLLM + AWQ 4-bit",
    "speculative": "vLLM + Speculative (0.5B draft)",
}
ENGINE_SHORT = {
    "hf": "HF baseline",
    "vllm": "vLLM",
    "awq": "AWQ 4-bit",
    "speculative": "Speculative",
}

# Telemetry metric -> (csv column, human label)
TELEMETRY_METRICS = {
    "VRAM usage (MB)": "vram_used_mb",
    "GPU utilization (%)": "gpu_percent",
    "Power draw (W)": "power_w",
    "GFX clock (MHz)": "gfx_clock_mhz",
}


# ---------------------------------------------------------------------------
# Process manager (one per server, survives reruns via cache_resource)
# ---------------------------------------------------------------------------
class ProcessManager:
    """Owns the API and tester subprocesses plus their log file handles."""

    def __init__(self) -> None:
        self.api_process: subprocess.Popen | None = None
        self.api_engine: str | None = None
        self.api_cmd = ""
        self.tester_process: subprocess.Popen | None = None
        self.tester_cmd = ""
        self._log_handles = []
        atexit.register(self.shutdown)

    def _open_log(self, path: Path):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", buffering=1, errors="replace")  # line buffered
        self._log_handles.append(handle)
        return handle

    # --- API lifecycle ---
    def start_api(self, engine: str) -> None:
        self.stop_api()
        log = self._open_log(API_LOG)
        cmd = [sys.executable, "main.py", "--engine", engine, "--api-only"]
        self.api_cmd = "python " + " ".join(cmd[1:])
        # New session: lets us signal the whole process group on stop
        self.api_process = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.api_engine = engine

    def stop_api(self) -> None:
        proc = self.api_process
        if proc and proc.poll() is None:
            # SIGINT triggers main.py's KeyboardInterrupt cleanup (kills the API child)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        self.api_process = None

    def is_api_running(self) -> bool:
        return self.api_process is not None and self.api_process.poll() is None

    # --- Tester lifecycle ---
    def start_tester(self, engine: str, users: int, requests: int,
                     max_tokens: int, prompt_size: str) -> None:
        log = self._open_log(TESTER_LOG)
        cmd = [
            sys.executable, "-m", "load_tester.tester",
            "--engine", engine,
            "--users", str(users),
            "--requests", str(requests),
            "--max-tokens", str(max_tokens),
            "--prompt-size", prompt_size,
        ]
        self.tester_cmd = "python " + " ".join(cmd[1:])
        self.tester_process = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def is_tester_running(self) -> bool:
        return self.tester_process is not None and self.tester_process.poll() is None

    def shutdown(self) -> None:
        self.stop_api()
        if self.is_tester_running():
            os.killpg(os.getpgid(self.tester_process.pid), signal.SIGTERM)
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass


@st.cache_resource
def get_process_manager() -> ProcessManager:
    return ProcessManager()


pm = get_process_manager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_api_health() -> dict | None:
    """Returns the /health payload, or None if the API is unreachable."""
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=1) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        pass
    return None


def tail_file(path: Path, max_lines: int = 300) -> str:
    if not path.exists():
        return "(no output yet)"
    with open(path, "r", errors="replace") as f:
        return "".join(deque(f, maxlen=max_lines)) or "(no output yet)"


def load_runs() -> list[dict]:
    """All results_*.json parsed, newest first. Engine/time come from the filename."""
    runs = []
    for path in DATA_DIR.glob("results_*.json"):
        m = re.match(r"results_(\w+)_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})\.json", path.name)
        if not m:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        data["_engine"] = m.group(1)
        data["_dt"] = datetime.strptime(m.group(2), "%d-%m-%Y_%H-%M-%S")
        data["_file"] = path.name
        runs.append(data)
    return sorted(runs, key=lambda r: r["_dt"], reverse=True)


def latest_per_engine(runs: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for run in runs:  # already newest-first
        latest.setdefault(run["_engine"], run)
    return latest


def load_telemetry(latest: dict[str, dict]) -> pd.DataFrame:
    """Concatenates the telemetry CSV of each engine's latest run, time-normalized."""
    frames = []
    for engine, run in latest.items():
        tel_file = run.get("telemetry_file", "")
        path = PROJECT_ROOT / tel_file if tel_file else None
        if not path or not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            continue
        df["elapsed_s"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().round(1)
        df["engine"] = ENGINE_SHORT.get(engine, engine)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def telemetry_chart(tel: pd.DataFrame, column: str) -> pd.DataFrame:
    df = tel[["elapsed_s", "engine", column]].dropna()
    return df.pivot_table(index="elapsed_s", columns="engine", values=column)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Inference Benchmark",
    page_icon=":material/speed:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: engine control + benchmark configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Engine control")
    st.selectbox(
        "Inference engine",
        options=list(ENGINE_LABELS.keys()),
        format_func=lambda e: ENGINE_LABELS[e],
        key="engine_select",
    )

    @st.fragment(run_every=2)
    def api_status_fragment() -> None:
        health = get_api_health()
        managed = pm.is_api_running()
        external = health is not None and not managed

        if health and managed:
            st.markdown(f":green[●] **Ready** — `{health.get('active_engine', '?')}`")
        elif managed:
            st.markdown(":orange[●] **Loading model on GPU...**")
        elif external:
            st.markdown(":blue[●] **Running (external process)**")
        else:
            st.markdown(":gray[●] **Stopped**")

        if pm.api_engine and pm.api_engine != st.session_state.engine_select and managed:
            st.caption(f"Running `{pm.api_engine}` — stop it to switch engines.")
        if external:
            st.caption("An API started outside this dashboard is on :8000. Stop it to manage engines here.")

        col_start, col_stop = st.columns(2)
        with col_start:
            st.button(
                "Start API", type="primary", width="stretch",
                disabled=managed or external,
                on_click=pm.start_api,
                args=(st.session_state.engine_select,),
            )
        with col_stop:
            st.button(
                "Stop API", width="stretch",
                disabled=not managed,
                on_click=pm.stop_api,
            )

    api_status_fragment()

    st.divider()
    st.header("Benchmark configuration")
    st.slider("Concurrent users", min_value=1, max_value=50, value=5, key="users")
    st.slider("Total requests", min_value=1, max_value=200, value=10, key="requests")
    st.slider("Max tokens", min_value=1, max_value=2048, value=150, key="max_tokens")
    st.segmented_control(
        "Prompt size", options=["short", "long"], default="short", key="prompt_size",
        help="'long' stresses vLLM's chunked prefill.",
    )

    def _run_clicked() -> None:
        health = get_api_health()
        # Register results under the engine the API actually loaded
        engine = health.get("active_engine") if health else st.session_state.engine_select
        pm.start_tester(
            engine=engine,
            users=st.session_state.users,
            requests=st.session_state.requests,
            max_tokens=st.session_state.max_tokens,
            prompt_size=st.session_state.prompt_size,
        )

    @st.fragment(run_every=2)
    def run_fragment() -> None:
        ready = get_api_health() is not None
        st.button(
            "Run benchmark", type="primary", width="stretch",
            disabled=not ready or pm.is_tester_running(),
            on_click=_run_clicked,
        )
        if pm.is_tester_running():
            st.info("Benchmark running... watch the Consoles tab.", icon=":material/hourglass_top:")

    run_fragment()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("LLM Inference Benchmark")
st.caption("AMD RX 9060XT (16GB) · ROCm · HF vs vLLM vs AWQ vs Speculative Decoding")

tab_consoles, tab_results = st.tabs(["Consoles", "Results"])

with tab_consoles:

    @st.fragment(run_every=2)
    def consoles_fragment() -> None:
        # Detect tester completion to refresh results and notify
        was_running = st.session_state.get("_tester_was_running", False)
        is_running = pm.is_tester_running()
        if was_running and not is_running:
            st.session_state["_tester_was_running"] = False
            st.toast("Benchmark finished — results updated.", icon=":material/check_circle:")
            st.rerun(scope="app")
        st.session_state["_tester_was_running"] = is_running

        col_api, col_tester = st.columns(2)
        with col_api:
            st.subheader("API server")
            st.code(pm.api_cmd or f"python main.py --engine {st.session_state.engine_select} --api-only",
                    language="bash")
            with st.container(height=500):
                st.code(tail_file(API_LOG), language="log")
        with col_tester:
            st.subheader("Load tester")
            default_cmd = (
                f"python -m load_tester.tester --engine {st.session_state.engine_select} "
                f"--users {st.session_state.users} --requests {st.session_state.requests} "
                f"--max-tokens {st.session_state.max_tokens} --prompt-size {st.session_state.prompt_size}"
            )
            st.code(pm.tester_cmd or default_cmd, language="bash")
            with st.container(height=500):
                st.code(tail_file(TESTER_LOG), language="log")

    consoles_fragment()

with tab_results:
    runs = load_runs()
    latest = latest_per_engine(runs)

    if not latest:
        st.info("No benchmark results in data/ yet. Start the API and run a benchmark from the sidebar.",
                icon=":material/info:")
    else:
        summary_rows = []
        for engine, run in latest.items():
            summary_rows.append({
                "Engine": ENGINE_SHORT.get(engine, engine),
                "Timestamp": run["timestamp"],
                "Users": run["num_users"],
                "Requests": run["num_requests"],
                "Max tokens": run["max_tokens"],
                "Tokens/s": run["tokens_per_second"],
                "P50 (s)": run["latency_p50"],
                "P95 (s)": run["latency_p95"],
                "Avg (s)": run["latency_avg"],
                "Error rate": run["error_rate"],
            })
        summary_df = pd.DataFrame(summary_rows)

        # KPI row: throughput per engine
        with st.container(horizontal=True):
            for engine, run in latest.items():
                st.metric(
                    ENGINE_SHORT.get(engine, engine),
                    f"{run['tokens_per_second']:.1f} tok/s",
                    f"P95 {run['latency_p95']:.2f}s",
                    border=True,
                )

        # OOM signal: HF baseline is expected to degrade under concurrency
        for engine, run in latest.items():
            if run["error_rate"] > 0:
                st.warning(
                    f"**{ENGINE_SHORT.get(engine, engine)}**: {run['error_rate']:.0%} of requests failed "
                    f"({run['failed_requests']}/{run['num_requests']}) — signature of VRAM exhaustion under concurrency.",
                    icon=":material/warning:",
                )

        # Throughput + latency comparison
        col_tput, col_lat = st.columns(2)
        with col_tput:
            with st.container(border=True):
                st.markdown("**Throughput per engine**")
                st.bar_chart(summary_df.set_index("Engine")[["Tokens/s"]], y_label="Tokens/s")
        with col_lat:
            with st.container(border=True):
                st.markdown("**Latency per engine**")
                st.bar_chart(summary_df.set_index("Engine")[["P50 (s)", "P95 (s)"]], y_label="Seconds")

        # Telemetry time series (latest run per engine, time-normalized)
        tel = load_telemetry(latest)
        if not tel.empty:
            available = [
                (label, col) for label, col in TELEMETRY_METRICS.items()
                if col in tel.columns and not tel[col].dropna().empty
            ]
            if available:
                st.subheader("GPU telemetry (latest run per engine)")
                cols = st.columns(2)
                for i, (label, column) in enumerate(available):
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{label}**")
                            st.line_chart(
                                telemetry_chart(tel, column),
                                x_label="Elapsed (s)", y_label=label,
                            )

        with st.expander("Run history (all runs in data/)"):
            history_rows = [{
                "Engine": ENGINE_SHORT.get(r["_engine"], r["_engine"]),
                "Timestamp": r["timestamp"],
                "Users": r["num_users"],
                "Requests": r["num_requests"],
                "Max tokens": r["max_tokens"],
                "Tokens/s": r["tokens_per_second"],
                "P50 (s)": r["latency_p50"],
                "P95 (s)": r["latency_p95"],
                "Error rate": r["error_rate"],
                "File": r["_file"],
            } for r in runs]
            st.dataframe(
                pd.DataFrame(history_rows),
                hide_index=True,
                column_config={
                    "Tokens/s": st.column_config.NumberColumn(format="%.2f"),
                    "P50 (s)": st.column_config.NumberColumn(format="%.2f"),
                    "P95 (s)": st.column_config.NumberColumn(format="%.2f"),
                    "Error rate": st.column_config.NumberColumn(format="percent"),
                },
            )
