import os
import logging
from typing import Any, Dict, List, Optional
import backoff
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError

logger = logging.getLogger(__name__)
from abc_embedder import ABCEmbedder

def binary_quantize(embeddings: np.ndarray) -> np.ndarray:
    return (embeddings >= np.mean(embeddings)).astype(np.float32)

class OpenAIEmbedder(ABCEmbedder):
    def __init__(
        self,
        model_name: str,
        dimensions: int,
        quantization: str,
        api_base: Optional[str] = None,
    ):
        self.model = model_name
        self.dimensions = dimensions
        self.quantization = quantization
        self.openai_client = AsyncOpenAI(
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/")
        )

@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=3)
async def get_embedding(self, text: str) -> np.ndarray:
    try:
        response = await self.openai_client.with_options(
            timeout=30.0
        ).embeddings.create(model=self.model, input=text, encoding_format="float")

        embedding = np.array(response.data[0].embedding)
        if self.quantization == "binary":
            return binary_quantize(embedding)
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