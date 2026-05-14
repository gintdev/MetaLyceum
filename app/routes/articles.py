from sqlalchemy.exc import IntegrityError
from typing import List, Annotated
import asyncio
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import select, func, Text, cast, or_
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.article import Article
from app.schemas import (
    ArticleCreate, ArticleUpdate, ArticleResponse, ArticleListResponse, ArticleCreateResponse,
    ProcessArticleRequest, ProcessArticleResponse,
    SearchRequest, SearchResponse, ChunkResult, VectorStoreInfo
)
from app.services.rag import get_rag_service
from app.services.pdf_processor import get_pdf_processor

router = APIRouter(prefix="/articles", tags=["articles"])


def _safe_pdf_download_name(
    requested_filename: str,
    title: str | None,
    authors: list[str] | None,
    published_year: int | None,
) -> str:
    """format: Authors - title, year.pdf."""
    base_stem = Path(requested_filename).name
    if base_stem.lower().endswith(".pdf"):
        base_stem = base_stem[:-4]

    title_part = (title or "").strip() or base_stem or "untitled"
    authors_part = ", ".join([a.strip() for a in (authors or []) if a and a.strip()]) or "Unknown authors"
    year_part = str(published_year) if published_year else "n.d."

    raw_name = f"{authors_part} - {title_part}, {year_part}.pdf"
    # Strip control and path separator chars to avoid malformed response headers.
    safe_name = re.sub(r"[\r\n\t\x00-\x1f\x7f/\\]+", " ", raw_name)
    safe_name = re.sub(r"\s+", " ", safe_name).strip(" .")
    return safe_name or "download.pdf"


def _build_pdf_citation_display_name(
    requested_filename: str,
    title: str | None,
    authors: list[str] | None,
    published_year: int | None,
    ref_index: int | None,
) -> str:
    
    base_stem = Path(requested_filename).name
    if base_stem.lower().endswith(".pdf"):
        base_stem = base_stem[:-4]

    title_part = (title or "").strip() or base_stem or "untitled"
    first_author = next((a.strip() for a in (authors or []) if a and a.strip()), "Unknown author")
    year_part = str(published_year) if published_year else "n.d."
    ref_prefix = f"[{ref_index}] " if isinstance(ref_index, int) and ref_index > 0 else ""

    raw_name = f"{ref_prefix}{first_author} {title_part} {year_part}.pdf"
    safe_name = re.sub(r"[\r\n\t\x00-\x1f\x7f/\\]+", " ", raw_name)
    safe_name = re.sub(r"\s+", " ", safe_name).strip(" .")
    return safe_name or "download.pdf"


async def _find_article_for_file(
    session: AsyncSession,
    filename: str,
    article_id: int | None = None,
) -> Article | None:
    if article_id is not None:
        article_by_id = await session.execute(select(Article).where(Article.id == article_id))
        found = article_by_id.scalar_one_or_none()
        if found:
            return found

    normalized_filename = filename.replace("\\", "/")
    requested_basename = Path(normalized_filename).name

    article_result = await session.execute(
        select(Article).where(
            or_(
                Article.file_name == filename,
                Article.file_name == normalized_filename,
                Article.file_name == requested_basename,
                Article.file_name.like(f"%/{requested_basename}"),
                Article.file_name.ilike(f"%{requested_basename}"),
            )
        )
    )
    return article_result.scalars().first()


async def _enrich_pdf_files_with_display_names(
    session: AsyncSession,
    pdf_files: list[dict],
) -> list[dict]:
    enriched_items: list[dict] = []
    for file_item in pdf_files:
        filename = file_item.get("filename")
        if not filename:
            continue

        article = await _find_article_for_file(
            session=session,
            filename=filename,
            article_id=file_item.get("article_id"),
        )
        display_name = _build_pdf_citation_display_name(
            requested_filename=filename,
            title=article.title if article else None,
            authors=article.authors if article else None,
            published_year=article.published_year if article else None,
            ref_index=file_item.get("ref_index") if isinstance(file_item.get("ref_index"), int) else None,
        )

        enriched_items.append({
            **file_item,
            "display_name": display_name,
        })

    return enriched_items


def _enrich_sources_with_display_names(
    sources: list[dict],
    pdf_files: list[dict],
) -> list[dict]:
    display_name_by_key: dict[tuple[object, object, object], str] = {}
    for file_item in pdf_files:
        key = (file_item.get("article_id"), file_item.get("filename"), file_item.get("ref_index"))
        display_name = file_item.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            display_name_by_key[key] = display_name.strip()

    enriched_sources: list[dict] = []
    for source in sources:
        if not isinstance(source, dict):
            continue

        key = (source.get("article_id"), source.get("filename"), source.get("ref_index"))
        display_name = display_name_by_key.get(key)
        if not display_name:
            enriched_sources.append(source)
            continue

        enriched_sources.append({
            **source,
            "display_name": display_name,
        })

    return enriched_sources


def _build_literature_answer_from_sources(
    pdf_files: list[dict],
    sources: list[dict],
) -> str:
    if not pdf_files:
        return "К сожалению, в базе знаний не найдено релевантных материалов для рекомендации."

    snippets_by_ref: dict[int, str] = {}
    for source in sources:
        ref_index = source.get("ref_index")
        if not isinstance(ref_index, int) or ref_index in snippets_by_ref:
            continue

        text = (source.get("text") or "").strip()
        if not text or text == "N/A":
            continue

        clean_text = re.sub(r"\s+", " ", text)
        snippets_by_ref[ref_index] = clean_text[:220]

    sorted_files = sorted(
        pdf_files,
        key=lambda item: (
            item.get("ref_index") if isinstance(item.get("ref_index"), int) else 10**9,
            item.get("filename") or "",
        ),
    )

    lines = ["Рекомендованные источники из базы знаний:"]
    for idx, file_item in enumerate(sorted_files, 1):
        ref_index = file_item.get("ref_index") if isinstance(file_item.get("ref_index"), int) else idx
        source_name = (file_item.get("display_name") or file_item.get("filename") or f"Источник {ref_index}").strip()
        snippet = snippets_by_ref.get(ref_index)

        if snippet:
            lines.append(f"{ref_index}. {source_name} - Релевантно по найденному фрагменту: {snippet}")
        else:
            lines.append(f"{ref_index}. {source_name} - Релевантно вашему запросу по результатам поиска в базе.")

    return "\n\n".join(lines)


async def _get_filtered_article_ids(
    request: SearchRequest,
    session: AsyncSession,
) -> List[int] | None:
    keywords_filter = []
    if request.keyword and request.keyword.strip():
        keywords_filter.append(request.keyword.strip())
    if request.keywords:
        keywords_filter.extend([kw.strip() for kw in request.keywords if kw and kw.strip()])

    sources_filter = []
    if request.sources:
        sources_filter.extend([src.strip() for src in request.sources if src and src.strip()])

    source_filter = request.source.strip() if request.source and request.source.strip() else None
    has_metadata_filters = bool(source_filter or sources_filter or keywords_filter or request.year_from is not None or request.year_to is not None)

    if request.year_from is not None and request.year_to is not None and request.year_from > request.year_to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нижняя граница года не может быть больше верхней"
        )

    if not has_metadata_filters:
        return None

    stmt = select(Article.id)

    if sources_filter:
        stmt = stmt.where(Article.source.in_(sources_filter))
    elif source_filter:
        stmt = stmt.where(Article.source.ilike(f"%{source_filter}%"))

    if keywords_filter:
        stmt = stmt.where(
            Article.keywords.overlap(
                cast(keywords_filter, ARRAY(Text))
            )
        )

    if request.year_from is not None:
        stmt = stmt.where(Article.published_year >= request.year_from)

    if request.year_to is not None:
        stmt = stmt.where(Article.published_year <= request.year_to)

    result = await session.execute(stmt)
    return list(result.scalars().all())


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
    "/keywords",
    response_model=List[str],
    summary="Получить список ключевых слов",
    description="Возвращает уникальные ключевые слова из БД с опциональной фильтрацией по префиксу"
)
async def get_keywords(
    query: str | None = Query(None, min_length=1, max_length=256),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_db)
) -> List[str]:
    keywords_subquery = (
        select(func.unnest(Article.keywords).label("keyword"))
        .where(Article.keywords.is_not(None))
        .subquery()
    )

    stmt = select(keywords_subquery.c.keyword).distinct().where(
        keywords_subquery.c.keyword.is_not(None)
    )

    if query:
        stmt = stmt.where(keywords_subquery.c.keyword.ilike(f"{query.strip()}%"))

    stmt = stmt.order_by(keywords_subquery.c.keyword).limit(limit)

    result = await session.execute(stmt)
    return [keyword for keyword in result.scalars().all() if keyword]


@router.get(
    "/sources",
    response_model=List[str],
    summary="Получить список источников",
    description="Возвращает уникальные источники из БД"
)
async def get_sources(
    query: str | None = Query(None, min_length=1, max_length=256),
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    session: AsyncSession = Depends(get_db)
) -> List[str]:
    stmt = select(Article.source).distinct().where(Article.source.is_not(None))

    if query:
        stmt = stmt.where(Article.source.ilike(f"{query.strip()}%"))

    stmt = stmt.order_by(Article.source).limit(limit)

    result = await session.execute(stmt)
    return [source for source in result.scalars().all() if source]


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
async def search_all_articles(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db)
) -> SearchResponse:
    try:
        filtered_article_ids = await _get_filtered_article_ids(request, session)
        if filtered_article_ids is not None and not filtered_article_ids:
            return SearchResponse(
                query=request.query,
                results=[],
                count=0
            )

        rag_service = await get_rag_service()
        results = await rag_service.search(
            request.query,
            request.limit,
            article_ids=filtered_article_ids
        )
        
        # Преобразовать результаты в ChunkResult
        chunk_results = [ChunkResult(**result) for result in results]
        
        return SearchResponse(
            query=request.query,
            results=chunk_results,
            count=len(chunk_results)
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
    try:
        filtered_article_ids = await _get_filtered_article_ids(request, session)
        if filtered_article_ids is not None and article_id not in filtered_article_ids:
            return SearchResponse(
                query=request.query,
                results=[],
                count=0,
                article_id=article_id
            )

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
    try:
        rag_service = await get_rag_service()
        info = await rag_service.get_vector_store_info()
        return VectorStoreInfo(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики: {str(e)}"
        )


@router.get(
    "/files/source-pdf",
    summary="Скачать PDF источника",
    description="Скачивает оригинальный PDF по filename с Яндекс Диска и отдает пользователю"
)
async def download_source_pdf(
    background_tasks: BackgroundTasks,
    filename: str = Query(..., min_length=1, max_length=1024, description="Имя файла на Яндекс Диске"),
    session: AsyncSession = Depends(get_db),
):
    try:
        pdf_processor = get_pdf_processor()
        local_path = await pdf_processor.download_pdf(filename)

        background_tasks.add_task(pdf_processor.cleanup_temp_file, local_path)

        article = await _find_article_for_file(session=session, filename=filename)
        download_name = _safe_pdf_download_name(
            requested_filename=filename,
            title=article.title if article else None,
            authors=article.authors if article else None,
            published_year=article.published_year if article else None,
        )

        return FileResponse(
            path=local_path,
            media_type="application/pdf",
            filename=download_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при скачивании PDF: {str(e)}"
        )

# Ответ на вопрос
@router.post(
    "/ask",
    summary="Ответить развернуто на вопрос",
    description="Найти релевантные чанки и сгенерировать ответ через OpenAI"
)
async def ask_question(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db)
):
    try:
        filtered_article_ids = await _get_filtered_article_ids(request, session)
        if filtered_article_ids is not None and not filtered_article_ids:
            return {
                "query": request.query,
                "answer": "К сожалению, по заданным фильтрам статьи не найдены.",
                "sources": [],
                "pdf_files": [],
                "chunks_used": 0,
                "status": "no_results"
            }

        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query=request.query,
            limit=request.limit or 5,
            query_type=request.query_type,
            article_ids=filtered_article_ids,
        )
        pdf_files = await _enrich_pdf_files_with_display_names(
            session=session,
            pdf_files=result.get("pdf_files", []),
        )
        sources = _enrich_sources_with_display_names(
            sources=result.get("sources", []),
            pdf_files=pdf_files,
        )
        
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": sources,
            "pdf_files": pdf_files,
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

# Написание эссе 
@router.post(
    "/essay",
    summary = "Написать эссе",
    description = "Найти релевантные чанки-источники и сгенерировать ответ через OpenAI"
)
async def write_essay(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db)
):
    try:
        filtered_article_ids = await _get_filtered_article_ids(request, session)
        if filtered_article_ids is not None and not filtered_article_ids:
            return {
                "query": request.query,
                "answer": "К сожалению, по заданным фильтрам статьи не найдены.",
                "sources": [],
                "pdf_files": [],
                "chunks_used": 0,
                "status": "no_results"
            }

        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query = request.query,
            limit = request.limit or 10,
            query_type = 1,
            article_ids=filtered_article_ids,
        )
        pdf_files = await _enrich_pdf_files_with_display_names(
            session=session,
            pdf_files=result.get("pdf_files", []),
        )
        sources = _enrich_sources_with_display_names(
            sources=result.get("sources", []),
            pdf_files=pdf_files,
        )
        answer = _build_literature_answer_from_sources(pdf_files=pdf_files, sources=sources)
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")

        return {
            "query": result["query"],
            "answer": answer,
            "sources": sources,
            "pdf_files": pdf_files,
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

# Рекомендация литературы
@router.post(
    "/literature",
    summary = "Рекомендация материалов",
    description = "Найти релевантные чанки-источники и сгенерировать ответ через OpenAI"
)
async def recomend_literature(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db)
):
    try:
        filtered_article_ids = await _get_filtered_article_ids(request, session)
        if filtered_article_ids is not None and not filtered_article_ids:
            return {
                "query": request.query,
                "answer": "К сожалению, по заданным фильтрам статьи не найдены.",
                "sources": [],
                "pdf_files": [],
                "chunks_used": 0,
                "status": "no_results"
            }

        rag_service = await get_rag_service()
        result = await rag_service.generate_answer(
            query = request.query,
            limit = request.limit or 5,
            query_type = 2,
            article_ids=filtered_article_ids,
        )
        pdf_files = await _enrich_pdf_files_with_display_names(
            session=session,
            pdf_files=result.get("pdf_files", []),
        )
        sources = _enrich_sources_with_display_names(
            sources=result.get("sources", []),
            pdf_files=pdf_files,
        )
        print(f"DEBUG endpoint: result keys = {result.keys()}")
        print(f"DEBUG endpoint: sources = {result.get('sources')}")

        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": sources,
            "pdf_files": pdf_files,
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