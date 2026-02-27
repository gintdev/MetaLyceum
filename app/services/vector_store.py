import logging
from typing import List, Dict, Optional
import uuid
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
)
from app.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Сервис для работы с векторной БД Qdrant"""
    
    def __init__(
        self,
        url: str = settings.QDRANT_URL,
        port: int = settings.QDRANT_PORT,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        vector_size: int = settings.OPENAI_EMBEDDING_DIMENSIONS,
    ):
        """
        Инициализация Qdrant клиента
        
        Args:
            url: URL Qdrant сервера
            port: Порт Qdrant сервера
            collection_name: Имя коллекции
            vector_size: Размерность векторов embedding-ов
        """
        self.client = AsyncQdrantClient(
            url=url,
            port=port,
            prefer_grpc=False,
            check_compatibility=False  # Отключить проверку совместимости версий
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
    
    async def initialize_collection(self):
        """Инициализировать коллекцию если её нет"""
        try:
            # Проверить существует ли коллекция
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Создание коллекции: {self.collection_name}")
                
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE  # Косинусное расстояние для embeddings
                    )
                )
                
                logger.info(f"✓ Коллекция создана успешно: {self.collection_name}")
            else:
                logger.info(f"Коллекция уже существует: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"✗ Ошибка при инициализации коллекции: {str(e)}")
            raise
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        chunks: List[Dict],
    ) -> List[str]:
        """
        Добавить векторы в Qdrant
        
        Args:
            vectors: Список embedding-ов (numpy arrays)
            chunks: Список словарей с метаданными чанков
            
        Returns:
            Список ID добавленных точек
        """
        try:
            points = []
            point_ids = []
            
            for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
                # Генерировать уникальный ID
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Конвертировать numpy array в список для Qdrant
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                # Создать point с метаданными
                point = PointStruct(
                    id=point_id,
                    vector=vector_list,
                    payload={
                        "article_id": chunk.get("article_id"),
                        "filename": chunk.get("filename"),
                        "chunk_index": chunk.get("chunk_index"),
                        "chunk_text": chunk.get("chunk_text"),
                        "chunk_size": chunk.get("chunk_size"),
                    }
                )
                points.append(point)
            
            # Добавить points в Qdrant
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✓ {len(points)} векторов добавлено в коллекцию {self.collection_name}")
            return point_ids
            
        except Exception as e:
            logger.error(f"✗ Ошибка при добавлении векторов: {str(e)}")
            raise
    
    async def search(
        self,
        query_vector: np.ndarray,
        limit: int = settings.SEARCH_LIMIT,
        filters: Optional[Filter] = None,
    ) -> List[Dict]:
        """
        Поиск похожих векторов
        
        Args:
            query_vector: Query embedding вектор
            limit: Количество результатов
            filters: Фильтры для поиска (опционально)
            
        Returns:
            Список найденных документов с метаданными и score
        """
        try:
            # Конвертировать numpy array в список
            query_vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
            # Выполнить поиск используя query_points
            # Возвращает QueryResponse с атрибутом .points
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector_list,
                limit=limit,
                query_filter=filters,
                with_payload=True
            )
            
            # Получить список точек из response
            results = response.points if hasattr(response, 'points') else response
            
            # Преобразовать результаты
            documents = []
            for result in results:
                doc = {
                    "id": result.id,
                    "score": result.score,
                    **result.payload
                }
                documents.append(doc)
            
            logger.debug(f"✓ Найдено {len(documents)} документов")
            return documents
            
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске: {str(e)}")
            raise
    
    async def search_by_article_id(
        self,
        query_vector: np.ndarray,
        article_id: int,
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск похожих векторов с фильтром по article_id
        
        Args:
            query_vector: Query embedding вектор
            article_id: ID статьи для фильтрации
            limit: Количество результатов
            
        Returns:
            Список найденных документов
        """
        # Создать фильтр по article_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="article_id",
                    match=MatchValue(value=article_id)
                )
            ]
        )
        
        return await self.search(query_vector, limit, filter_condition)

    async def search_by_article_ids(
        self,
        query_vector: np.ndarray,
        article_ids: List[int],
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск похожих векторов с фильтром по списку article_id

        Args:
            query_vector: Query embedding вектор
            article_ids: Список ID статей для фильтрации
            limit: Количество результатов

        Returns:
            Список найденных документов
        """
        if not article_ids:
            return []

        match_condition = (
            MatchValue(value=article_ids[0])
            if len(article_ids) == 1
            else MatchAny(any=article_ids)
        )

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="article_id",
                    match=match_condition
                )
            ]
        )

        return await self.search(query_vector, limit, filter_condition)
    
    async def delete_by_article_id(self, article_id: int) -> int:
        """
        Удалить все векторы для статьи
        
        Args:
            article_id: ID статьи
            
        Returns:
            Количество удаленных точек
        """
        try:
            # Создать фильтр
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="article_id",
                        match=MatchValue(value=article_id)
                    )
                ]
            )
            
            # Удалить точки
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"✓ Векторы удалены для article_id={article_id}")
            return article_id
            
        except Exception as e:
            logger.error(f"✗ Ошибка при удалении векторов: {str(e)}")
            raise
    
    async def get_collection_info(self) -> Dict:
        """Получить информацию о коллекции"""
        try:
            info = await self.client.get_collection(self.collection_name)
            
            # В Qdrant: 
            # - points_count = количество точек (документов/чанков)
            # - vectors_count = это параметр конфигурации размерности
            # Для статистики нам нужно количество реальных точек
            points_count = getattr(info, 'points_count', 0)
            
            return {
                "name": self.collection_name,
                "documents_count": points_count,  # Количество документов (чанков)
                "points_count": points_count,     # Синоним для совместимости
            }
        except Exception as e:
            logger.error(f"✗ Ошибка при получении информации о коллекции: {str(e)}")
            # Вернуть пустую статистику вместо ошибки
            return {
                "name": self.collection_name,
                "documents_count": 0,
                "points_count": 0,
            }


# Глобальный инстанс vector store
_vector_store: Optional[QdrantVectorStore] = None


async def get_vector_store() -> QdrantVectorStore:
    """Получить инстанс QdrantVectorStore (singleton pattern)"""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
        await _vector_store.initialize_collection()
    return _vector_store
