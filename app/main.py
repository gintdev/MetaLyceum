from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.db import init_db
from app.routes import articles, auth, chats
from app.services.rag import get_rag_service

# Инициализация базы данных и RAG сервиса при запуске приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    print("✓ База данных инициализирована")
    
    # Инициализировать RAG сервис и Qdrant коллекцию
    try:
        rag_service = await get_rag_service()
        info = await rag_service.get_vector_store_info()
        print(f"✓ RAG сервис инициализирован")
        print(f"  - Коллекция: {info['name']}")
        print(f"  - Чанков: {info['points_count']}")
    except Exception as e:
        print(f"⚠ Ошибка инициализации RAG сервиса: {str(e)}")
    
    yield
    # Shutdown
    print("✓ Приложение остановлено")


# Создание FastAPI приложения
app = FastAPI(
    title="MetaLyceum API",
    description="API для управления философскими статьями и RAG системы",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production следует ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(articles.router)
app.include_router(auth.router)
app.include_router(chats.router)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Проверка здоровья приложения"""
    try:
        # Проверить подключение к RAG сервису
        rag = await get_rag_service()
        info = await rag.get_vector_store_info()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "vector_db": {
                "collection": info['name'],
                "chunks": info['points_count'],
                "documents_count": info['documents_count']
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Root endpoint
@app.get("/", tags=["root"])
async def read_root():
    """Главная страница API"""
    return {
        "message": "Добро пожаловать в MetaLyceum API",
        "description": "Система RAG для интерактивного обучения философии",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "articles": "/articles/",
            "rag_search": "/articles/search",
            "auth": "/auth/",
            "chats": "/chats/",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
       "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )