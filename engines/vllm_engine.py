import uuid, time, gc
import torch
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from engines.base_engine import BaseEngine
from vllm.inputs import TextPrompt

class VLLMEngine(BaseEngine):
    # Motor 2: Vllm otimizado
    def load_model(self) -> None:
        self.logger.info("Iniciando o vLLM (alocando PagedAttention)...")

        if self.model_id is None:
            raise ValueError("model_id is required to run the model")

        # Carrega o tokenizer para aplicar o chat template corretamente
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # Configs for model
        engine_args = AsyncEngineArgs(
            model = self.model_id,
            trust_remote_code= True,
            enable_prefix_caching= True,
            enable_chunked_prefill= True,
            enforce_eager= False,
            gpu_memory_utilization = 0.85
        )

        # O AsyncLLMEngine é feito para lidar com milhares de requisições simultâneas
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.info(f"vLLM Engine carregado com sucesso para {self.model_id}")

    async def warmup(self) -> None:
        self.logger.info("Executando warmup para pré-compilar kernels...")
        warmup_prompt = TextPrompt(prompt="Warmup")
        warmup_params = SamplingParams(max_tokens=10, temperature=0.7)
        generator = self.engine.generate(warmup_prompt, warmup_params, "warmup")
        async for _ in generator:
            pass
        self.logger.info("Warmup concluído")

    async def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        start = time.time()
        # No vLLM, configuramos como o texto será gerado via SamplingParams
        sampling_params = SamplingParams(
            max_tokens = max_tokens,
            temperature = 0.7
        )

        # Formata o prompt usando o chat template do modelo (ex: ChatML para TinyLlama)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = TextPrompt(prompt=formatted_prompt)


        # id único para cada requisição para não misturar as respostas
        request_id = str(uuid.uuid4())
        # geração assíncrona
        generator = self.engine.generate(inputs, sampling_params, request_id)

        # o vLLM nos devolve o texto sendo gerado token por token (stream)
        final_output = None
        async for request_output in generator:
            final_output = request_output

        assert final_output is not None, "No output Generated"

        elapsed = time.time() - start
        prompt_tokens = len(final_output.prompt_token_ids) # type: ignore
        completion_tokens = len(final_output.outputs[0].token_ids)

        return {
            "text": final_output.outputs[0].text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_seconds": elapsed
        }

    def shutdown(self) -> None:
        # Dropa a referência do AsyncLLMEngine (libera KV cache + PagedAttention),
        # força GC e devolve a memória cacheada ao ROCm/HIP allocator.
        self.logger.info("Desligando vLLM engine e liberando VRAM...")
        self.engine = None
        self.tokenizer = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            self.logger.warning(f"empty_cache falhou (ignorado): {exc}")
        self.logger.info("VRAM liberada")