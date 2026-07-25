import argparse, os, sys, time, subprocess, urllib.request, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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
    args = parser.parse_args()

    # Define a variável de ambiente para a API saber qual engine carregar
    env = os.environ.copy()
    env["ENGINE_TYPE"] = args.engine

    python_executable = sys.executable

    logger.info(f"Iniciando API Litestar com motor: '{args.engine}'...")

    api_process = subprocess.Popen(
        [python_executable, "-m", "engines.api"],
        env=env
    )
    streamlit_process = None

    try:
        # Aguarda a API estar online
        if wait_for_api("http://localhost:8000"):
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
        api_process.terminate()
        api_process.wait()
        if streamlit_process is not None:
            streamlit_process.terminate()
            streamlit_process.wait()

        logger.info("Todos os processos foram encerrados com sucesso.")

if __name__ == "__main__":
    main()
