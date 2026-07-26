# Contexto do Projeto: LLM Inference Benchmark

## Objetivo
Benchmark comparativo de 4 motores de inferência LLM em AMD RX 9060XT (16GB VRAM) com ROCm.

## Motores
1. **HF Baseline**: HuggingFace transformers (baseline sem otimizações)
2. **vLLM**: PagedAttention + Prefix Caching + Chunked Prefill
3. **AWQ**: vLLM com quantização 4-bit (modelo pré-quantizado)
4. **Speculative**: vLLM com target 3B + draft 0.5B

## Modelo
- Base: `Qwen/Qwen2.5-3B-Instruct` (~6.5GB fp16)
- AWQ: `Qwen/Qwen2.5-3B-Instruct-AWQ` (~2GB 4-bit)
- Draft: `Qwen/Qwen2.5-0.5B-Instruct` (~1.2GB)

## Arquitetura
```
python main.py --engine <hf|vllm|awq|speculative>
├── API Litestar (:8000) → carrega 1 motor na GPU
└── Streamlit (:8501) → cliente HTTP, sem GPU
    └── Botão "Run Benchmark" → load_tester/tester.py
        └── httpx.AsyncClient + asyncio.gather
            └── POST /generate → API → GPU
```

## Estrutura de Arquivos
```
engines/
├── base_engine.py      # ABC com load_model(), generate(), warmup()
├── hf_engine.py        # Motor 1: retorna dict com métricas
├── vllm_engine.py      # Motor 2: TextPrompt + warmup no startup
├── awq_engine.py       # Motor 3: quantization="awq"
├── speculative_engine.py # Motor 4: speculative_config
└── api.py              # Litestar + Settings + on_startup com warmup

load_tester/
└── tester.py           # run_benchmark() + GPUMonitor integrado
                        # Parâmetros: users, requests, max_tokens, prompt-size

telemetry/
└── monitor.py          # GPUMonitor: amdsmi a cada 500ms, salva CSV
                        # Métricas: VRAM, % GPU, power_w, gfx_clock_mhz, mem_clock_mhz

dashboard/
└── app.test.py         # Streamlit (Dia 5 - versão de teste)
                        # Processo PAI: gerencia main.py --api-only via subprocess

data/
├── results_<engine>_<timestamp>.json  # Métricas de inferência
├── telemetry_<engine>_<timestamp>.csv # VRAM, % GPU, power, clocks
└── logs/               # api.log e tester.log (tail ao vivo no dashboard)
```

## Contrato da API
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
```

## Correções Aplicadas
1. **API consolidada**: Settings extraído, engine_type removido do payload
2. **Retorno padronizado**: Todas engines retornam dict com métricas
3. **Warmup**: Método `warmup()` chamado em `on_startup` (não em generate)
4. **TextPrompt**: vLLM engines usam `TextPrompt` em vez de string raw
5. **Timestamp BR**: `%d/%m/%Y %H:%M:%S` em results, `%d-%m-%Y_%H-%M-%S` em filenames
6. **Monitor integrado**: tester inicia/para GPUMonitor automaticamente
7. **Variáveis ROCm**: PYTHONPATH e FLASH_ATTENTION injetados no main.py

## Status dos Dias
- ✅ Dia 1: Consolidação e correções
- ✅ Dia 2: AWQ e Speculative engines
- ✅ Dia 3: Load Tester com prompt curto/longo
- ✅ Dia 4: Telemetria AMD integrada
- ⏳ Dia 5: Dashboard Streamlit (`app.test.py` criado, validar manualmente → virar `app.py`)
- ⏳ Dia 6: README.md e LinkedIn

## Dashboard (Dia 5) — Implementado em app.test.py
1. **Streamlit é o processo pai**: gerencia `python main.py --engine X --api-only` via
   subprocess (botões Start/Stop API na sidebar). `main.py` ganhou a flag `--api-only`.
2. **2 consoles ao vivo lado a lado** (aba Consoles): tail de `data/logs/api.log` e
   `data/logs/tester.log` via `st.fragment(run_every=2)`, com o comando exibido acima.
3. **Controles na sidebar**: engine (selectbox), users, requests, max_tokens (sliders),
   prompt size (segmented_control short/long). Botão Run desabilitado até /health ok.
4. **Benchmark em background**: tester.py via subprocess; ao terminar, toast + rerun
   automático atualizam os gráficos.
5. **Gráficos (aba Results)**: último run de cada engine — KPIs tok/s, bar charts de
   throughput e latência P50/P95, telemetria (VRAM, % GPU, power, clock GFX) ao longo
   do tempo normalizado. Warning de error_rate > 0 (evidencia OOM do HF).
6. **Telemetria ampliada**: monitor.py agora coleta `power_w` (socket_power),
   `gfx_clock_mhz` e `mem_clock_mhz` (falha isolada por métrica, sem derrubar amostra).
7. **ProcessManager** via `@st.cache_resource`: start_new_session + SIGINT (aciona o
   finally do main.py que mata a API filha), fallback killpg. atexit cleanup.

## Como Rodar (novo fluxo Dia 5)
```bash
# Um terminal só:
streamlit run dashboard/app.test.py

# Na UI: escolher engine → Start API (aguarda "Ready") → configurar → Run benchmark
# Trocar de engine: Stop API → selecionar → Start API (Streamlit não reinicia)
```

## Como Rodar (modo CLI clássico)
```bash
# Terminal 1: API + Streamlit (fluxo antigo, main.py orquestra tudo)
python main.py --engine vllm

# Terminal 2: Teste manual
python -m load_tester.tester --engine vllm --users 5 --requests 10 --max-tokens 150 --prompt-size long

# Resultados
cat data/results_vllm_*.json
cat data/telemetry_vllm_*.csv
```

## Observações Importantes
- AWQ é ~24% mais lento que vLLM (overhead de desquantização), mas usa 70% menos VRAM
- HF baseline degrada rapidamente com concorrência (sem PagedAttention)
- vLLM mantém P50/P95 próximos mesmo sob carga
- Warmup pré-compila kernels Triton (evita spikes na primeira requisição)
- UTF-8 no JSON aparece como `\u00e1` mas é decodificado automaticamente pelo Streamlit
