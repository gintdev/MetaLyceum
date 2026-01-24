"""Embedders для генерирования embeddings текста"""

from app.services.embedders.abc_embedder import ABCEmbedder
from app.services.embedders.openai_embedder import OpenAIEmbedder

__all__ = [
    "ABCEmbedder",
    "OpenAIEmbedder",
]
