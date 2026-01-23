from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DB_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DB_PORT: int
    DB_HOST: str
    
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 3600  # 1 hour
    DB_POOL_TIMEOUT: int = 30
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct DATABASE_URL from individual PostgreSQL parameters"""
        return f"postgresql+asyncpg://{self.DB_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"

    # RAG / Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PORT: Optional[int] = 6333

    # Yandex.Disk
    YADISK_TOKEN: Optional[str] = None
    YADISK_FOLDER: str = "MetaLyceum"

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

settings = Settings()