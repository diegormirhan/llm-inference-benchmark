import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tvm_ffi")

import atexit
import logging
import os
import signal
import uvicorn
from typing import Dict

from litestar import Litestar, post, get
from litestar.datastructures import State
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from engines.hf_engine import HuggingFaceEngine
from engines.vllm_engine import VLLMEngine
from engines.awq_engine import AWQEngine
from engines.speculative_engine import SpeculativeEngine

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger("LitestarServer")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    hf_token: str = Field(default="", alias="HF_TOKEN")

    model_id: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        alias="MODEL_ID",
        description="ID do modelo padrão no Hugging Face"
    )

    awq_model: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct-AWQ",
        alias="AWQ_MODEL",
        description="ID do modelo AWQ no Hugging Face"
    )

    target_model: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        alias="TARGET_MODEL",
        description="ID do modelo target no Hugging Face"
    )
    draft_model: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        alias="DRAFT_MODEL",
        description="ID do modelo draft no Hugging Face"
    )
    engine_type: str = Field(
        default="hf",
        alias="ENGINE_TYPE",
        description="Motor ativo para inferência: 'hf', 'vllm', 'awq', 'speculative'"
    )
    max_concurrent_requests: int = Field(
        default=100,
        alias="MAX_CONCURRENT_REQUESTS"
    )

settings = Settings()

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt de texto para o modelo")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Número máximo de tokens a gerar")

class GenerationResponse(BaseModel):
    generated_text: str
    engine_used: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float

@get("/health")
async def health_check() -> Dict[str, str]:
    # Endpoint de checagem de saúde do servidor.
    return {
        "status": "ok",
        "active_engine": settings.engine_type,
        "model": settings.model_id
    }

@post("/generate", status_code=200)
async def generate_text(data: GenerationRequest, state: State) -> GenerationResponse:
    # Endpoint para geração de texto utilizando o motor ativo
    engine = state.active_engine
    logger.info(f"Recebida requisição de geração usando motor ativo: {settings.engine_type}")
    result = await engine.generate(prompt=data.prompt, max_tokens=data.max_tokens)
    
    return GenerationResponse(
        generated_text=result["text"],
        engine_used=settings.engine_type,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        elapsed_seconds=result["elapsed_seconds"]
    )

async def on_startup(app: Litestar) -> None:
    engine_type = settings.engine_type.lower()
    logger.info(f"Inicializando APENAS o motor '{engine_type}' com o modelo: {settings.model_id}...")

    if engine_type == "hf":
        app.state.active_engine = HuggingFaceEngine(model_id=settings.model_id)
    elif engine_type == "vllm":
        app.state.active_engine = VLLMEngine(model_id=settings.model_id)
    elif engine_type == "awq":
        app.state.active_engine = AWQEngine(awq_model=settings.awq_model)
    elif engine_type == "speculative":
        app.state.active_engine = SpeculativeEngine(target_model=settings.target_model, draft_model=settings.draft_model)
    else:
        raise ValueError(f"Motor '{engine_type}' não suportado. Escolha 'hf', 'vllm', 'awq', 'speculative'.")

    await app.state.active_engine.warmup()
    logger.info(f"Motor '{engine_type}' inicializado e pronto na GPU!")

    def _release() -> None:
        eng = getattr(app.state, "active_engine", None)
        if eng is None:
            return
        try:
            eng.shutdown()
        except Exception as exc:
            logger.error(f"Falha no shutdown do engine: {exc}")
        app.state.active_engine = None

    def _signal_handler(signum: int, _frame) -> None:
        logger.info(f"Sinal {signum} recebido, liberando engine antes de sair...")
        _release()
        # os._exit evita que o uvicorn execute um shutdown redundante
        # (que poderia disparar _release novamente via atexit).
        os._exit(0)

    atexit.register(_release)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

app = Litestar(
    route_handlers=[health_check, generate_text],
    on_startup=[on_startup],
)

if __name__ == "__main__":
    # access_log=False: skips the per-request uvicorn access line.
    # /health is polled every 2s by the dashboard and would flood the log;
    # /generate already emits its own logger.info at the app level.
    uvicorn.run("engines.api:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
