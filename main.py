import argparse, os, signal, sys, time, subprocess, urllib.request, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger("Orchestrator")

def wait_for_api(url: str, timeout: int = 700) -> bool:
    # Aguarda a API ficar saudável na GPU antes de abrir o Streamlit. 
    start_time = time.time()
    logger.info(f"Aguardando inicialização da API em {url}...")
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health") as response:
                if response.status == 200:
                    logger.info("API inicializada e saudável na GPU!")
                    return True
        except Exception:
            time.sleep(2)
    logger.error("Timeout: A API demorou demais para inicializar.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Orquestrador do Benchmark de LLM")
    parser.add_argument(
        "--engine",
        type=str,
        default="hf",
        choices=["hf", "vllm", "awq", "speculative"],
        help="Motor de inferência a ser testado ('hf', 'vllm', 'awq', 'speculative')"
    )
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="Sobe apenas a API, sem abrir o Streamlit (usado quando o dashboard gerencia o processo)"
    )
    args = parser.parse_args()

    # Define a variável de ambiente para a API saber qual engine carregar
    env = os.environ.copy()
    env["ENGINE_TYPE"] = args.engine

    venv_path = sys.prefix
    env["PYTHONPATH"] = f"{venv_path}/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi"
    env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    python_executable = sys.executable

    logger.info(f"Iniciando API Litestar com motor: '{args.engine}'...")

    api_process = subprocess.Popen(
        [python_executable, "-m", "engines.api"],
        env=env
    )
    streamlit_process = None

    # Handler de SIGTERM: cobre o caso `kill <main.py_pid>` (ou killpg vindo
    # do dashboard) e garante que engines.api também seja encerrado,
    # evitando que vire processo órfão segurando :8000 e a VRAM.
    # Timeouts curtos porque engine.shutdown() do vLLM pode levar 10s+;
    # o fallback é SIGKILL e o OS reclaims a VRAM.
    def _on_sigterm(_signum, _frame):
        logger.info("SIGTERM recebido, encerrando processos filhos...")
        if api_process.poll() is None:
            try:
                api_process.send_signal(signal.SIGTERM)
                api_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                api_process.kill()
                try:
                    api_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        if streamlit_process is not None and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait()
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        if args.engine_only:
            # Modo dashboard: Streamlit é o processo pai, main.py só mantém a API da engine viva
            logger.info("Modo --engine-only: API rodando. Aguardando encerramento externo...")
            api_process.wait()
        # Aguarda a API estar online
        elif wait_for_api("http://localhost:8000"):
            logger.info("Iniciando Dashboard do Streamlit...")
            streamlit_process = subprocess.Popen(
                [python_executable, "-m", "streamlit", "run", "dashboard/app.py"]
            )
            streamlit_process.wait()
        else:
            logger.error("Falha ao iniciar a API. Encerrando orquestrador.")
    except KeyboardInterrupt:
        logger.info("\nEncerrando API e Streamlit...")
    finally:
        if api_process.poll() is None:
            try:
                api_process.send_signal(signal.SIGTERM)
                api_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("API não respondeu ao SIGTERM em 2s, enviando SIGKILL")
                api_process.kill()
                try:
                    api_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        if streamlit_process is not None:
            streamlit_process.terminate()
            streamlit_process.wait()

        logger.info("Todos os processos foram encerrados com sucesso.")

if __name__ == "__main__":
    main()
