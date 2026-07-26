import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from engines.base_engine import BaseEngine

class HuggingFaceEngine(BaseEngine):
    # Motor 1: Baseline
    def load_model(self) -> None:
        self.logger.info("Iniciando o download via transformers...")

        # O ROCm da AMD usa o backend "cuda" do PyTorch de forma transparente
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_id is None:
            raise ValueError("model_id is required to run the model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Carregamos em float16 para não explodir a VRAM logo de cara
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        self.logger.info(f"Modelo {self.model_id} carregando com sucesso no {self.device}")

    async def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        start = time.time()
        self.logger.info(f"Gerando inferência para prompt de tamanho {len(prompt)} caracteres")

        # Formata o prompt usando o chat template do modelo
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenização (preparando o texto para a mtemática da GPU)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids.shape[-1]

        # Geração de texto
        with torch.no_grad():
            outputs = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens = max_tokens,
                do_sample = True,
                temperature = 0.7,
                pad_token_id = self.tokenizer.eos_token_id
            )

        elapsed = time.time() - start    
        completion_tokens = outputs.shape[-1] - prompt_tokens
        # Decodificação (transformando números de volta em texto)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_seconds": elapsed
        }