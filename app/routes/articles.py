from sqlalchemy.exc import IntegrityError
from typing import List, Annotated
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.article import Article
from app.schemas import ArticleCreate, ArticleUpdate, ArticleResponse, ArticleListResponse

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post(
    "/",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новую статью",
    description="Создает новую статью с метаданными по философии"
)
async def create_article(
    article: ArticleCreate,
    session: AsyncSession = Depends(get_db)
) -> ArticleResponse:
    """
    Создает новую статью в базе данных.
    
    - **title**: название статьи (обязательно)
    - **source**: источник статьи (обязательно)
    - **abstract**: аннотация статьи
    - **authors**: список авторов
    - **keywords**: ключевые слова
    - **published_year**: год публикации
    - **file_name**: имя файла
    - **download_url**: URL для скачивания
    - **parsed_at**: время парсинга
    """
    try:
        db_article = Article(**article.model_dump())
        session.add(db_article)
        await session.commit()
        await session.refresh(db_article)
        return ArticleResponse.model_validate(db_article)
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