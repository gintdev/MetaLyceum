from sqlalchemy.exc import IntegrityError
from typing import List, Annotated
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.article import Article
from app.schemas import (
    ArticleCreate, ArticleUpdate, ArticleResponse, ArticleListResponse, ArticleCreateResponse,
    ProcessArticleRequest, ProcessArticleResponse,
    SearchRequest, SearchResponse, ChunkResult, VectorStoreInfo
)
from app.services.rag import get_rag_service

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post(
    "/",
    response_model=ArticleCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новую статью",
    description="Создает новую статью с метаданными по философии и опционально обрабатывает её для RAG"
)
async def create_article(
    article: ArticleCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db)
) -> ArticleCreateResponse:
    """
    Создает новую статью в базе данных и опционально запускает обработку для RAG.
    
    - **title**: название статьи (обязательно)
    - **source**: источник статьи (обязательно)
    - **abstract**: аннотация статьи
    - **authors**: список авторов
    - **keywords**: ключевые слова
    - **published_year**: год публикации
    - **file_name**: имя файла на Яндекс Диске
    - **download_url**: URL для скачивания
    - **parsed_at**: время парсинга
    - **process**: обработать для RAG сразу же (по умолчанию True)
    """
    try:
        # Сохранить метаданные статьи
        article_data = article.model_dump(exclude={'process'})
        db_article = Article(**article_data)
        session.add(db_article)
        await session.commit()
        await session.refresh(db_article)
        
        article_response = ArticleResponse.model_validate(db_article)
        processing = False
        message = None
        
        # Если требуется обработка и есть имя файла
        if article.process and article.file_name:
            processing = True
            message = f"Статья сохранена и запущена фоновая обработка для RAG"
            
            # Запустить обработку в фоне
            async def process_in_background():
                try:
                    rag_service = await get_rag_service()
                    await rag_service.process_article(db_article.id, article.file_name)
                except Exception as e:
                    print(f"❌ Ошибка при фоновой обработке статьи {db_article.id}: {str(e)}")
            
            # Добавить фоновую задачу
            background_tasks.add_task(process_in_background)
        elif article.process and not article.file_name:
            message = "Статья сохранена. Обработка для RAG требует file_name"
        else:
            message = "Статья сохранена (обработка для RAG отключена)"
        
        return ArticleCreateResponse(
            article=article_response,
            processing=processing,
            message=message
        )
        
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ошибка при сохранении статьи. Проверьте введенные данные."
        )
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.args[0] if e.args else "Внутренняя ошибка сервера"
        )


@router.get(
    "/",
    response_model=ArticleListResponse,
    summary="Получить список статей",
    description="Получает список статей с поддержкой пагинации"
)
async def get_articles(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 10,
    session: AsyncSession = Depends(get_db)
) -> ArticleListResponse:
    """
    Получает список статей с поддержкой пагинации.
    
    - **page**: номер страницы (по умолчанию 1)
    - **page_size**: количество элементов на странице (по умолчанию 10, максимум 100)
    """
    # Получить общее количество статей
    total_result = await session.execute(select(func.count(Article.id)))
    total = total_result.scalar()
    
    # Получить статьи для текущей страницы
    offset = (page - 1) * page_size
    result = await session.execute(
        select(Article)
        .offset(offset)
        .limit(page_size)
    )
    articles = result.scalars().all()
    
    return ArticleListResponse(
        items=[ArticleResponse.model_validate(article) for article in articles],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Получить статью по ID",
    description="Получает детальную информацию о статье по ID"
)
async def get_article(
    article_id: int,
    session: AsyncSession = Depends(get_db)
) -> ArticleResponse:
    """
    Получает информацию о конкретной статье по ID.
    
    - **article_id**: ID статьи
    """
    result = await session.execute(select(Article).where(Article.id == article_id))
    article = result.scalar_one_or_none()
    
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Статья с ID {article_id} не найдена"
        )
    
    return ArticleResponse.model_validate(article)


@router.put(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Обновить статью",
    description="Обновляет информацию о статье"
)
async def update_article(
    article_id: int,
    article_update: ArticleUpdate,
    session: AsyncSession = Depends(get_db)
) -> ArticleResponse:
    """
    Обновляет информацию о статье.
    
    - **article_id**: ID статьи
    - **article_update**: данные для обновления (все поля опциональны)
    """
    result = await session.execute(select(Article).where(Article.id == article_id))
    article = result.scalar_one_or_none()
    
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Статья с ID {article_id} не найдена"
        )
    
    update_data = article_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(article, field, value)
    
    try:
        session.add(article)
        await session.commit()
        await session.refresh(article)
        return ArticleResponse.model_validate(article)
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обновлении статьи"
        )


@router.delete(
    "/{article_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить статью",
    description="Удаляет статью из базы данных"
)
async def delete_article(
    article_id: int,
    session: AsyncSession = Depends(get_db)
):
    """
    Удаляет статью из базы данных.
    
    - **article_id**: ID статьи
    """
    result = await session.execute(select(Article).where(Article.id == article_id))
    article = result.scalar_one_or_none()
    
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Статья с ID {article_id} не найдена"
        )
    
    try:
        await session.delete(article)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при удалении статьи"
        )


# RAG endpoints

@router.post(
    "/{article_id}/process",
    response_model=ProcessArticleResponse,
    status_code=status.HTTP_200_OK,
    summary="Обработать статью для RAG",
    description="Скачивает PDF, парсит текст, создает embedding и сохраняет в Qdrant"
)
async def process_article_for_rag(
    article_id: int,
    request: ProcessArticleRequest,
    session: AsyncSession = Depends(get_db)
) -> ProcessArticleResponse:
    """
    Обработка статьи для RAG системы:
    1. Скачивание PDF с Яндекс Диска
    2. Извлечение текста из PDF
    3. Разбиение текста на чанки
    4. Генерирование embedding-ов с OpenAI
    5. Сохранение в векторную БД Qdrant
    
    - **article_id**: ID статьи в PostgreSQL БД
    - **filename**: Имя файла на Яндекс Диске
    """
    try:
        # Проверить что статья существует
        result = await session.execute(select(Article).where(Article.id == article_id))
        article = result.scalar_one_or_none()
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Статья с ID {article_id} не найдена"
            )
        
        # Получить RAG сервис и обработать статью
        rag_service = await get_rag_service()
        result = await rag_service.process_article(article_id, request.filename)
        
        return ProcessArticleResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке статьи: {str(e)}"
        )


@router.post(
    "/{article_id}/reprocess",
    response_model=ProcessArticleResponse,
    status_code=status.HTTP_200_OK,
    summary="Переобработать статью",
    description="Удаляет старые данные и переобрабатывает статью"
)
async def reprocess_article_for_rag(
    article_id: int,
    request: ProcessArticleRequest,
    session: AsyncSession = Depends(get_db)
) -> ProcessArticleResponse:
    """
    Переобработка статьи (удаление старых векторов + новая обработка).
    Используйте когда нужно обновить embeddings статьи.
    
    - **article_id**: ID статьи в PostgreSQL БД
    - **filename**: Имя файла на Яндекс Диске
    """
    try:
        # Проверить что статья существует
        result = await session.execute(select(Article).where(Article.id == article_id))
        article = result.scalar_one_or_none()
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Статья с ID {article_id} не найдена"
            )
        
        # Получить RAG сервис и переобработать статью
        rag_service = await get_rag_service()
        result = await rag_service.reprocess_article(article_id, request.filename)
        
        return ProcessArticleResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при переобработке статьи: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Поиск по всем статьям",
    description="Выполняет RAG поиск по всем обработанным статьям"
)
async def search_all_articles(request: SearchRequest) -> SearchResponse:
    """
    RAG поиск по всему корпусу статей.
    
    - **query**: Текстовый запрос на русском языке
    - **limit**: Максимальное количество результатов (по умолчанию 5)
    """
    try:
        rag_service = await get_rag_service()
        results = await rag_service.search(request.query, request.limit)
        
        # Преобразовать результаты в ChunkResult
        chunk_results = [ChunkResult(**result) for result in results]
        
        return SearchResponse(
            query=request.query,
            results=chunk_results,
            count=len(chunk_results),
            article_id=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при поиске: {str(e)}"
        )


@router.post(
    "/{article_id}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Поиск в конкретной статье",
    description="Выполняет RAG поиск только в пределах одной статьи"
)
async def search_in_article(
    article_id: int,
    request: SearchRequest,
    session: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """
    RAG поиск в конкретной статье.
    
    - **article_id**: ID статьи для поиска
    - **query**: Текстовый запрос на русском языке
    - **limit**: Максимальное количество результатов (по умолчанию 5)
    """
    try:
        # Проверить что статья существует
        result = await session.execute(select(Article).where(Article.id == article_id))
        article = result.scalar_one_or_none()
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Статья с ID {article_id} не найдена"
            )
        
        # Выполнить поиск в статье
        rag_service = await get_rag_service()
        results = await rag_service.search_in_article(request.query, article_id, request.limit)
        
        # Преобразовать результаты в ChunkResult
        chunk_results = [ChunkResult(**result) for result in results]
        
        return SearchResponse(
            query=request.query,
            results=chunk_results,
            count=len(chunk_results),
            article_id=article_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при поиске в статье: {str(e)}"
        )


@router.get(
    "/rag/stats",
    response_model=VectorStoreInfo,
    status_code=status.HTTP_200_OK,
    summary="Статистика векторной БД",
    description="Получить информацию о Qdrant коллекции"
)
async def get_vector_store_stats() -> VectorStoreInfo:
    """
    Получить информацию о векторной БД Qdrant (количество документов, векторов и т.д.)
    """
    try:
        rag_service = await get_rag_service()
        info = await rag_service.get_vector_store_info()
        return VectorStoreInfo(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики: {str(e)}"
        )


@router.post(
    "/ask",
    summary="Ответить развернуто на вопрос",
    description="Найти релевантные чанки и сгенерировать ответ через OpenAI"
)
async def ask_question(request: SearchRequest):
    """
    Полный RAG цикл: поиск + генерация ответа
    
    - Ищет релевантные чанки в векторной БД
    - Добавляет их в контекст
    - Отправляет вопрос с контекстом в OpenAI
    - Возвращает развёрнутый ответ с источниками
    """
    try:
        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query=request.query,
            limit=request.limit or 5,
            query_type=request.query_type
        )
        
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "chunks_used": result["chunks_count"],
            "status": result["status"]
        }
        
    except Exception as e:
        print(f"DEBUG endpoint: Exception = {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}"
        )
    
@router.post(
    "/essay",
    summary = "Написать эссе",
    description = "Найти релевантные чанки-источники и сгенерировать ответ через OpenAI"
)
async def write_essay(request: SearchRequest):
    try:
        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query = request.query,
            limit = request.limit or 10,
            query_type = 1
        )
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")

        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "chunks_used": result["chunks_count"],
            "status": result["status"]
        }
    except Exception as e:
        print(f"DEBUG endpoint: Exception = {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}"
        )

@router.post(
    "/literature",
    summary = "Рекомендация материалов",
    description = "Найти релевантные чанки-источники и сгенерировать ответ через OpenAI"
)
async def recomend_literature(request: SearchRequest):
    try:
        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query = request.query,
            limit = request.limit or 5,
            query_type = 2
        )
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")

        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "chunks_used": result["chunks_count"],
            "status": result["status"]
        }
    except Exception as e:
        print(f"DEBUG endpoint: Exception = {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}"
        )