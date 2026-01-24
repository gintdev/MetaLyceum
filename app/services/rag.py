import logging
from typing import List, Dict, Optional
import numpy as np

from app.services.embedders.openai_embedder import OpenAIEmbedder
from app.services.pdf_processor import PDFProcessor
from app.services.text_chunker import TextChunker
from app.services.vector_store import QdrantVectorStore
from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG (Retrieval Augmented Generation) сервис
    Интегрирует: embedder, PDF processor, text chunker, vector store
    """
    
    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        text_chunker: Optional[TextChunker] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ):
        """
        Инициализация RAG сервиса
        
        Args:
            embedder: OpenAI embedder инстанс
            pdf_processor: PDF processor инстанс
            text_chunker: Text chunker инстанс
            vector_store: Qdrant vector store инстанс
        """
        self.embedder = embedder or OpenAIEmbedder(
            model_name=settings.OPENAI_MODEL,
            dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS,
            quantization=settings.OPENAI_QUANTIZATION,
        )
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.text_chunker = text_chunker or TextChunker()
        self.vector_store = vector_store or QdrantVectorStore()
    
    async def initialize(self):
        """Инициализировать RAG сервис (создать коллекцию Qdrant)"""
        try:
            await self.vector_store.initialize_collection()
            logger.info("✓ RAG сервис инициализирован")
        except Exception as e:
            logger.error(f"✗ Ошибка при инициализации RAG сервиса: {str(e)}")
            raise
    
    async def process_article(
        self,
        article_id: int,
        filename: str,
    ) -> Dict:
        """
        Полная обработка статьи: скачивание -> парсинг -> chunking -> embedding -> сохранение
        
        Args:
            article_id: ID статьи в БД
            filename: Имя файла на Яндекс Диске
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            logger.info(f"🚀 Начало обработки статьи {article_id}: {filename}")
            
            # 1. Скачать PDF и извлечь текст
            logger.info("1️⃣ Скачивание и извлечение текста из PDF...")
            text = await self.pdf_processor.process_pdf(filename)
            logger.info(f"   ✓ Текст извлечен: {len(text)} символов")
            
            # 2. Разбить текст на чанки
            logger.info("2️⃣ Разбиение текста на чанки...")
            chunks = self.text_chunker.chunk_text_with_metadata(
                text=text,
                article_id=article_id,
                filename=filename
            )
            logger.info(f"   ✓ Создано {len(chunks)} чанков")
            
            # 3. Получить embeddings для каждого чанка
            logger.info("3️⃣ Генерирование embeddings...")
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = await self.embedder.embed_texts(chunk_texts)
            logger.info(f"   ✓ Embeddings созданы: {len(embeddings)} векторов")
            
            # 4. Сохранить в Qdrant
            logger.info("4️⃣ Сохранение в векторной БД...")
            point_ids = await self.vector_store.add_vectors(embeddings, chunks)
            logger.info(f"   ✓ Сохранено {len(point_ids)} точек в Qdrant")
            
            logger.info(f"✅ Статья {article_id} успешно обработана")
            
            return {
                "article_id": article_id,
                "filename": filename,
                "text_length": len(text),
                "chunks_count": len(chunks),
                "embeddings_count": len(embeddings),
                "stored_points": len(point_ids),
                "point_ids": point_ids,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"✗ Ошибка при обработке статьи {article_id}: {str(e)}")
            raise
    
    async def reprocess_article(
        self,
        article_id: int,
        filename: str,
    ) -> Dict:
        """
        Переобработка статьи: удалить старые данные и обработать заново
        
        Args:
            article_id: ID статьи в БД
            filename: Имя файла на Яндекс Диске
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            logger.info(f"🔄 Переобработка статьи {article_id}...")
            
            # Удалить старые векторы
            await self.vector_store.delete_by_article_id(article_id)
            
            # Обработать заново
            result = await self.process_article(article_id, filename)
            logger.info(f"✅ Статья {article_id} переобработана")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Ошибка при переобработке статьи {article_id}: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск по запросу: embedding -> поиск в Qdrant
        
        Args:
            query: Текстовый запрос
            limit: Количество результатов
            
        Returns:
            Список найденных документов
        """
        try:
            logger.info(f"🔍 Поиск по запросу: {query[:50]}...")
            
            # 1. Получить embedding для запроса
            query_embedding = await self.embedder.get_embedding(query)
            
            # 2. Найти похожие документы
            results = await self.vector_store.search(query_embedding, limit)
            
            logger.info(f"✓ Найдено {len(results)} релевантных документов")
            return results
            
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске: {str(e)}")
            raise
    
    async def search_in_article(
        self,
        query: str,
        article_id: int,
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск по запросу в конкретной статье
        
        Args:
            query: Текстовый запрос
            article_id: ID статьи для фильтрации
            limit: Количество результатов
            
        Returns:
            Список найденных документов в статье
        """
        try:
            logger.info(f"🔍 Поиск по запросу в статье {article_id}: {query[:50]}...")
            
            # 1. Получить embedding для запроса
            query_embedding = await self.embedder.get_embedding(query)
            
            # 2. Найти похожие документы в статье
            results = await self.vector_store.search_by_article_id(
                query_embedding,
                article_id,
                limit
            )
            
            logger.info(f"✓ Найдено {len(results)} документов в статье {article_id}")
            return results
            
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске в статье {article_id}: {str(e)}")
            raise
    
    async def get_vector_store_info(self) -> Dict:
        """Получить информацию о векторной БД"""
        try:
            info = await self.vector_store.get_collection_info()
            return info
        except Exception as e:
            logger.error(f"✗ Ошибка при получении информации: {str(e)}")
            raise


# Глобальный инстанс RAG сервиса
_rag_service: Optional[RAGService] = None


async def get_rag_service() -> RAGService:
    """Получить инстанс RAGService (singleton pattern)"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        await _rag_service.initialize()
    return _rag_service
