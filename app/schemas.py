from pydantic import BaseModel, Field
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
                "parsed_at": datetime.now()
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
