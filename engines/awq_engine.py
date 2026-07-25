import uuid, time
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from engines.base_engine import BaseEngine

class AWQEngine(BaseEngine):
    # Motor 3: vLLM com modelo AWQ 4-bits pré-quantizado.
    def load_model(self) -> None:
        self.logger.info(f"Iniciando o vLLM com o modelo Quantizado AWQ 4-bits...")

        if self.awq_model is None:
            raise ValueError("awq_model is required to run the model")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.awq_model, trust_remote_code=True
        )

        # carrega o tokenizer 
        engine_args = AsyncEngineArgs(
            model = self.awq_model,
            trust_remote_code = True,
            quantization = "awq", # ativa o suporte AWQ do vLLM (4-bit quantization)
            enable_prefix_caching = True, # cacheia o prefixo comum entre requisições. se vários usuários enviam prompts similares, o vllm reutiliza a computação de prefixo
            enable_chunked_prefill= True, # divide prompts longos em chunks para nao travar a geração.
            enforce_eager= False, # permite uso do cuda graphs
            gpu_memory_utilization=0.85
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.info(f"vLLM AWQ Engine carregado para {self.awq_model}")

    # cria o motor assíncrono com as configurações acima
    async def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        start = time.time()
        sampling_params = SamplingParams(max_tokens = max_tokens, temperature=0.7)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # aplica o chat template do Qwen
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(formatted_prompt, sampling_params, request_id)

        # gera um ID único para a requisição e inicia a geração assíncrona
        final_output = None
        async for request_output in generator:
            final_output = request_output

        # garantir que final_output não retorna None
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

