from pydantic import BaseModel, Field, AliasChoices
from typing import Optional, List
from datetime import datetime


class ArticleCreate(BaseModel):
    """Схема для создания новой статьи"""
    title: str = Field(..., min_length=1, max_length=1024, description="Название статьи")
    source: str = Field(..., min_length=1, max_length=256, description="Источник статьи")
    abstract: Optional[str] = Field(None, max_length=5000, description="Аннотация статьи")
    authors: Optional[List[str]] = Field(None, description="Список авторов")
    keywords: Optional[List[str]] = Field(None, description="Ключевые слова")
    published_year: Optional[int] = Field(None, ge=1900, le=2100, description="Год публикации")
    file_name: Optional[str] = Field(None, max_length=512, description="Имя файла")
    download_url: Optional[str] = Field(None, max_length=2048, description="URL для скачивания")
    parsed_at: Optional[datetime] = Field(None, description="Время парсинга")
    process: bool = Field(True, description="Обработать для RAG сразу же (по умолчанию True)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Критика чистого разума",
                "source": "Akademie-Ausgabe",
                "abstract": "Фундаментальный труд Канта о природе познания",
                "authors": ["Иммануил Кант"],
                "keywords": ["эпистемология", "критика", "разум", "априори"],
                "published_year": 1781,
                "file_name": "kritika_chystogo_razuma.pdf",
                "download_url": "https://example.com/files/kritika.pdf",
                "parsed_at": datetime.now(),
                "process": True
            }
        }


class ArticleUpdate(BaseModel):
    """Схема для обновления статьи"""
    title: Optional[str] = Field(None, min_length=1, max_length=1024)
    source: Optional[str] = Field(None, min_length=1, max_length=256)
    abstract: Optional[str] = Field(None, max_length=5000)
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    published_year: Optional[int] = Field(None, ge=1900, le=2100)
    file_name: Optional[str] = Field(None, max_length=512)
    download_url: Optional[str] = Field(None, max_length=2048)
    parsed_at: Optional[datetime] = Field(None, max_length=64)


class ArticleResponse(BaseModel):
    """Схема для ответа при получении статьи"""
    id: int
    title: str
    source: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    published_year: Optional[int] = None
    views_count: int = 0
    downloads_count: int = 0
    file_name: Optional[str] = None
    download_url: Optional[str] = None
    parsed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ArticleListResponse(BaseModel):
    """Схема для списка статей"""
    items: List[ArticleResponse]
    total: int
    page: int
    page_size: int


class ArticleCreateResponse(BaseModel):
    """Ответ при создании статьи с опциональной информацией об обработке"""
    article: ArticleResponse
    processing: bool = Field(description="Статья обрабатывается в фоне")
    message: Optional[str] = None


# ===== RAG Schemas =====

class ProcessArticleRequest(BaseModel):
    """Запрос на обработку статьи для RAG"""
    article_id: int = Field(..., description="ID статьи для обработки")
    filename: str = Field(..., description="Имя файла на Яндекс Диске")

    class Config:
        json_schema_extra = {
            "example": {
                "article_id": 1,
                "filename": "kritika_chystogo_razuma.pdf"
            }
        }


class ProcessArticleResponse(BaseModel):
    """Ответ на обработку статьи"""
    article_id: int
    filename: str
    text_length: int
    chunks_count: int
    embeddings_count: int
    stored_points: int
    point_ids: List[str]
    status: str


class SearchRequest(BaseModel):
    """Запрос на поиск в RAG"""
    query: str = Field(..., min_length=1, max_length=2048, description="Поисковый запрос")
    article_id: Optional[int] = Field(None, description="ID статьи для фильтрации (опционально)")
    limit: int = Field(5, ge=1, le=100, description="Количество результатов")
    query_type: int = Field(0, ge=0, le=2, description="Тип ответа: 0 - базовый, 1 - эссе, 2 - рекомендация литературы")
    keyword: Optional[str] = Field(None, min_length=1, max_length=256, description="Ключевое слово для фильтрации статей")
    keywords: Optional[List[str]] = Field(None, description="Список ключевых слов для фильтрации статей")
    source: Optional[str] = Field(None, min_length=1, max_length=256, description="Источник статьи для фильтрации")
    sources: Optional[List[str]] = Field(None, description="Список источников для фильтрации статей")
    year_from: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="Нижняя граница года публикации",
        validation_alias=AliasChoices("year_from", "publication_year_from", "publication_date_from")
    )
    year_to: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="Верхняя граница года публикации",
        validation_alias=AliasChoices("year_to", "publication_year_to", "publication_date_to")
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Что такое трансцендентальная логика?",
                "article_id": None,
                "limit": 5,
                "query_type": 0,
                "keywords": ["эпистемология", "разум"],
                "sources": ["Stanford Encyclopedia of Philosophy", "Internet Encyclopedia of Philosophy"],
                "year_from": 1990,
                "year_to": 2025
            }
        }


class ChunkResult(BaseModel):
    """Результат поиска - найденный чанк"""
    id: str
    score: float
    article_id: int
    filename: str
    chunk_index: int
    chunk_text: str
    chunk_size: int


class SearchResponse(BaseModel):
    """Ответ на запрос поиска"""
    query: str
    results: List[ChunkResult]
    count: int
    article_id: Optional[int] = None


class VectorStoreInfo(BaseModel):
    """Информация о векторной БД"""
    name: str
    documents_count: int
    points_count: int