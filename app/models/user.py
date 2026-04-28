from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from app.models.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    google_sub = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    display_name = Column(String(120), nullable=True)
    avatar_url = Column(String(2048), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
