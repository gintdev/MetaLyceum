from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class ABCEmbedder(ABC):
    """Абстрактный базовый класс для embedder-ов"""

    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Получить embedding для текста
        
        Args:
            text: Текст для эмбеддирования
            
        Returns:
            numpy array с embedding-ом
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Получить embeddings для списка текстов
        
        Args:
            texts: Список текстов для эмбеддирования
            
        Returns:
            Список numpy arrays с embedding-ами
        """
        pass
