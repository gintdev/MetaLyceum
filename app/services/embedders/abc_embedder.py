from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

class ABCEmbedder(ABC):

    @abstractmethod
    async def get_emnedding(self, text:str) -> np.ndarray:
        pass

    @abstractmethod
    async def store_embedding():
