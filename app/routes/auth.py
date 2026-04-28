from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.user import User
from app.schemas import AuthResponse, GoogleAuthRequest, UserResponse
from app.services.auth import create_access_token, get_current_user, verify_google_id_token


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/google",
    response_model=AuthResponse,
    summary="Вход через Google",
)
async def google_auth(
    payload: GoogleAuthRequest,
    session: AsyncSession = Depends(get_db),
) -> AuthResponse:
    google_user = verify_google_id_token(payload.credential)
    google_sub = google_user["sub"]
    email = google_user["email"].lower()
    display_name = google_user.get("name")
    avatar_url = google_user.get("picture")

    user_result = await session.execute(select(User).where(User.google_sub == google_sub))
    user = user_result.scalar_one_or_none()

    if user is None:
        # Fallback: if same email exists from previous auth setup, attach Google account to it.
        email_result = await session.execute(select(User).where(User.email == email))
        user = email_result.scalar_one_or_none()

    if user is None:
        user = User(
            google_sub=google_sub,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    else:
        user.google_sub = google_sub
        user.email = email
        user.display_name = display_name
        user.avatar_url = avatar_url
        session.add(user)
        await session.commit()
        await session.refresh(user)

    if not user.google_sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не удалось привязать Google аккаунт",
        )

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Текущий пользователь",
)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)
