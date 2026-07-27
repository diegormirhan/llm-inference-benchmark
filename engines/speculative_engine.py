import uuid, time, gc
import torch
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from engines.base_engine import BaseEngine

class SpeculativeEngine(BaseEngine):
    # Motor 4: vLLM com Speculative Decoding (target 3B + draft 0.5B).
    def load_model(self) -> None:
        self.logger.info("Iniciando vLLM com Speculative Decoding...")

        if self.target_model is None or self.draft_model is None:
            raise ValueError("target_model and draft_model is required to run the model")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.target_model, trust_remote_code = True
        )
        
        engine_args = AsyncEngineArgs(
            model = self.target_model,
            trust_remote_code = True,
            enable_prefix_caching = True,
            enable_chunked_prefill= True,
            gpu_memory_utilization= 0.85,
            speculative_config={
                "model": self.draft_model,
                "num_speculative_tokens": 5,
                "method": "draft_model",
            }
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.info(
            f"vLLM Speculative Engine carregado: target={self.target_model}, draft={self.draft_model}"
        )

    async def warmup(self) -> None:
        self.logger.info("Executando warmup para pré-compilar kernels...")
        from vllm.inputs import TextPrompt
        warmup_params = SamplingParams(max_tokens=10, temperature=0.7)
        warmup_prompt = TextPrompt(prompt="Warmup")
        generator = self.engine.generate(warmup_prompt, warmup_params, "warmup")
        async for _ in generator:
            pass
        self.logger.info("Warmup concluído")

    async def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        from vllm.inputs import TextPrompt
        start = time.time()
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = TextPrompt(prompt=formatted_prompt)

        request_id = str(uuid.uuid4())
        generator = self.engine.generate(inputs, sampling_params, request_id)

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
        # Mesma lógica do VLLMEngine. O draft 0.5B + target 3B + KV cache
        # ocupam ~8GB; sem isto, parte da VRAM fica órfã após SIGTERM.
        self.logger.info("Desligando vLLM Speculative engine e liberando VRAM...")
        self.engine = None
        self.tokenizer = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            self.logger.warning(f"empty_cache falhou (ignorado): {exc}")
        self.logger.info("VRAM liberada")