from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.db import init_db
from app.routes import articles

# Инициализация базы данных при запуске приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    print("✓ База данных инициализирована")
    yield
    # Shutdown
    print("✓ Приложение остановлено")


# Создание FastAPI приложения
app = FastAPI(
    title="MetaLyceum API",
    description="API для управления философскими статьями",
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


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Проверка здоровья приложения"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/", tags=["root"])
async def read_root():
    """Главная страница API"""
    return {
        "message": "Добро пожаловать в MetaLyceum API",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
       "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )