from abc import ABC, abstractmethod
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class BaseEngine(ABC):
    # abstract base class representing as Inference Engine. Enforces a standard contract for all engines (HF, vLLM, AWQ).
    def __init__(self, model_id: Optional[str] = None, target_model: Optional[str] = None, draft_model: Optional[str] = None, awq_model: Optional[str] = None):
        self.model_id = model_id
        self.target_model = target_model
        self.draft_model = draft_model
        self.awq_model = awq_model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_id}")
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        # Abstract method to load the model into VRAM. Must be implemented by subclasses.
        pass

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        # Abstract method to perform inference. Returns the generated text.
        pass

    async def warmup(self) -> None:
        # Optional warmup method. Override in subclasses if needed.
        pass
