from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.chat import Chat
from app.models.message import Message
from app.models.user import User
from app.routes.articles import _enrich_pdf_files_with_display_names, _enrich_sources_with_display_names, _get_filtered_article_ids
from app.schemas import (
    ChatAskRequest,
    ChatAskResponse,
    ChatCreateRequest,
    ChatListItem,
    ChatResponse,
    ChatUpdateRequest,
    MessageListResponse,
    MessageResponse,
)
from app.services.auth import get_current_user
from app.services.rag import get_rag_service


router = APIRouter(prefix="/chats", tags=["chats"])


def _sanitize_text(value: str) -> str:
    # PostgreSQL text/jsonb cannot contain NUL byte.
    return value.replace("\x00", "")


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    return value


async def _get_owned_chat_or_404(chat_id: int, user_id: int, session: AsyncSession) -> Chat:
    result = await session.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.user_id == user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден")
    return chat


@router.get(
    "/",
    response_model=list[ChatListItem],
    summary="Список чатов пользователя",
)
async def list_chats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[ChatListItem]:
    result = await session.execute(
        select(Chat)
        .where(Chat.user_id == current_user.id)
        .order_by(desc(Chat.updated_at), desc(Chat.id))
    )
    chats = result.scalars().all()

    chat_ids = [chat.id for chat in chats]
    previews_by_chat_id: dict[int, str] = {}
    if chat_ids:
        message_stmt: Select[tuple[Message]] = (
            select(Message)
            .where(Message.chat_id.in_(chat_ids))
            .order_by(desc(Message.created_at), desc(Message.id))
        )
        message_result = await session.execute(message_stmt)
        for message in message_result.scalars().all():
            if message.chat_id in previews_by_chat_id:
                continue
            previews_by_chat_id[message.chat_id] = message.content[:140]

    return [
        ChatListItem(
            id=chat.id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            last_message_preview=previews_by_chat_id.get(chat.id),
        )
        for chat in chats
    ]


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать чат",
)
async def create_chat(
    payload: ChatCreateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ChatResponse:
    title = (payload.title or "New chat").strip()
    chat = Chat(user_id=current_user.id, title=title or "New chat")
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return ChatResponse.model_validate(chat)


@router.patch(
    "/{chat_id}",
    response_model=ChatResponse,
    summary="Переименовать чат",
)
async def rename_chat(
    chat_id: int,
    payload: ChatUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ChatResponse:
    chat = await _get_owned_chat_or_404(chat_id, current_user.id, session)
    chat.title = payload.title.strip()
    chat.updated_at = datetime.now(timezone.utc)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return ChatResponse.model_validate(chat)


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить чат",
)
async def delete_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> None:
    chat = await _get_owned_chat_or_404(chat_id, current_user.id, session)
    await session.delete(chat)
    await session.commit()


@router.get(
    "/{chat_id}/messages",
    response_model=MessageListResponse,
    summary="Получить сообщения чата",
)
async def get_chat_messages(
    chat_id: int,
    limit: int = Query(200, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> MessageListResponse:
    await _get_owned_chat_or_404(chat_id, current_user.id, session)

    result = await session.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return MessageListResponse(items=[MessageResponse.model_validate(message) for message in messages])


@router.post(
    "/{chat_id}/ask",
    response_model=ChatAskResponse,
    summary="Задать вопрос в рамках чата",
)
async def ask_in_chat(
    chat_id: int,
    request: ChatAskRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ChatAskResponse:
    chat = await _get_owned_chat_or_404(chat_id, current_user.id, session)
    safe_query = _sanitize_text(request.query)

    user_message = Message(
        chat_id=chat.id,
        role="user",
        content=safe_query,
        sources=[],
        query_type=request.query_type,
    )
    session.add(user_message)

    if chat.title == "New chat":
        chat.title = safe_query.strip()[:60] or "New chat"

    filtered_article_ids = await _get_filtered_article_ids(request, session)
    if filtered_article_ids is not None and not filtered_article_ids:
        answer = "К сожалению, по заданным фильтрам статьи не найдены."
        assistant_message = Message(
            chat_id=chat.id,
            role="assistant",
            content=answer,
            sources=[],
            query_type=request.query_type,
        )
        session.add(assistant_message)
        chat.updated_at = datetime.now(timezone.utc)
        session.add(chat)
        await session.commit()

        return ChatAskResponse(
            chat_id=chat.id,
            query=safe_query,
            answer=answer,
            sources=[],
            pdf_files=[],
            chunks_used=0,
            status="no_results",
        )

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
    sources = _sanitize_json_value(
        _enrich_sources_with_display_names(
            sources=result.get("sources", []),
            pdf_files=pdf_files,
        )
    )
    answer_text = _sanitize_text(result.get("answer", "Пустой ответ от сервера"))
    query_text = _sanitize_text(result.get("query", safe_query))

    assistant_message = Message(
        chat_id=chat.id,
        role="assistant",
        content=answer_text,
        sources=sources,
        query_type=request.query_type,
    )
    session.add(assistant_message)

    chat.updated_at = datetime.now(timezone.utc)
    session.add(chat)

    await session.commit()

    return ChatAskResponse(
        chat_id=chat.id,
        query=query_text,
        answer=answer_text,
        sources=sources,
        pdf_files=pdf_files,
        chunks_used=result.get("chunks_count", 0),
        status=result.get("status", "ok"),
    )
