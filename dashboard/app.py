import atexit, json, os, re, signal, subprocess, urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
API_LOG = LOG_DIR / "api.log"
TESTER_LOG = LOG_DIR / "tester.log"
API_URL = "https://localhost:8000"

ENGINE_LABELS = {
    "hf": "HuggingFace (baseline)",
    "vllm": "vLLM (PagedAttention)",
    "awq": "vLLM + AWQ 4-bit",
    "speculative": "vLLM + Speculative (0.5B draft)"
}

ENGINE_SHORT = {
    "hf": "HF baseline",
    "vllm": "vLLM",
    "awq": "AWQ 4-bit",
    "speculative": "Speculative"
}

TELEMETRY_METRICS = {
    "VRAM usage (MB)": "vram_used_mb",
    "GPU utilization (%)": "gpu_percent",
    "Power draw (W)": "power_w",
    "GFX clock (MHz)": "gfx_clock_mhz"
}