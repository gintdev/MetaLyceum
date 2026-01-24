import os
import logging
from typing import Any, Dict, List, Optional
import backoff
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from app.config import settings
from app.services.embedders.abc_embedder import ABCEmbedder

logger = logging.getLogger(__name__)


def binary_quantize(embeddings: np.ndarray) -> np.ndarray:
    """Бинарная квантизация embedding-а"""
    return (embeddings >= np.mean(embeddings)).astype(np.float32)


class OpenAIEmbedder(ABCEmbedder):
    """Embedder с использованием OpenAI API"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        quantization: str = "float",
        api_key: Optional[str] = settings.OPENAI_API_KEY,
        api_base: Optional[str] = None,
    ):
        """
        Инициализация OpenAI embedder
        
        Args:
            model_name: Имя модели OpenAI (по умолчанию text-embedding-3-small)
            dimensions: Размерность embedding-а
            quantization: Тип квантизации ('float' или 'binary')
            api_key: OpenAI API ключ
            api_base: Base URL для OpenAI API
        """
        self.model = model_name
        self.dimensions = dimensions
        self.quantization = quantization
        self.openai_client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/")
        )

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=3)
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Получить embedding для одного текста
        
        Args:
            text: Текст для эмбеддирования
            
        Returns:
            numpy array с embedding-ом
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            
            embedding = np.array(response.data[0].embedding)
            
            if self.quantization == "binary":
                embedding = binary_quantize(embedding)
            
            logger.debug(f"Successfully generated embedding for text: {text[:50]}...")
            return embedding
            
        except RateLimitError:
            logger.warning("Rate limit exceeded for OpenAI API")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Получить embeddings для списка текстов
        
        Args:
            texts: Список текстов для эмбеддирования
            
        Returns:
            Список numpy arrays с embedding-ами
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions
            )
            
            embeddings = []
            for item in response.data:
                embedding = np.array(item.embedding)
                
                if self.quantization == "binary":
                    embedding = binary_quantize(embedding)
                
                embeddings.append(embedding)
            
            logger.debug(f"Successfully generated embeddings for {len(texts)} texts")
            return embeddings
            
        except RateLimitError:
            logger.warning("Rate limit exceeded for OpenAI API")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")