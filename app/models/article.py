from sqlalchemy import Column, String, DateTime, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from app.models.base import Base

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(1024), nullable=False)
    source = Column(String(256), nullable=False)
    abstract = Column(Text, nullable=True)
    authors = Column(ARRAY(String), nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    published_year = Column(Integer, nullable=True)
    views_count = Column(Integer, default=0)
    downloads_count = Column(Integer, default=0)
    file_name = Column(String(512), nullable=True)
    download_url = Column(String(2048), nullable=True)
    parsed_at = Column(DateTime(timezone=True),nullable=True)
    
    def __repr__(self):
        return f"<Article(article_id='{self.id}', title='{self.title}')>"