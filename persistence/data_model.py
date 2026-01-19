import random
import string
import logging
from datetime import datetime
from typing import ClassVar, Optional

from sqlalchemy import Integer, String, Float, Boolean, DateTime, ForeignKey, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deepfake.persistence.base import ModelBase, SessionType

logger = logging.getLogger(__name__)


def generate_unique_id(session: SessionType, model, length: int = 10) -> str:
    """Generate a unique ID for a given SQLAlchemy model."""
    while True:
        candidate = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        exists = session.execute(
            select(model).where(model.id == candidate)
        ).scalar_one_or_none()
        if not exists:
            return candidate


class User(ModelBase):
    __tablename__ = "users"
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)  # Keep as int
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)

    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="user", cascade="all, delete-orphan"
    )
    images: Mapped[list["Image"]] = relationship(
        "Image", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Video(ModelBase):
    __tablename__ = "videos"
    session: ClassVar[SessionType]
    use_db: bool = True

    id: Mapped[str] = mapped_column(String(5), primary_key=True, unique=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    duration: Mapped[Optional[float]] = mapped_column(Float)
    file_path: Mapped[Optional[str]] = mapped_column(String)
    uploadedDate: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    video_filename: Mapped[Optional[str]] = mapped_column(String)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="videos")

    result: Mapped[Optional["Result"]] = relationship(
        "Result", back_populates="video", uselist=False, cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = generate_unique_id(self.session, Video)
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<Video(id={self.id}, title={self.title})>"

    @staticmethod
    def commit():
        Video.session.commit()

    @staticmethod
    def rollback():
        Video.session.rollback()


class Image(ModelBase):
    __tablename__ = "images"
    session: ClassVar[SessionType]
    use_db: bool = True

    id: Mapped[str] = mapped_column(String(5), primary_key=True, unique=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    file_path: Mapped[Optional[str]] = mapped_column(String)
    uploadedDate: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    image_filename: Mapped[Optional[str]] = mapped_column(String)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="images")

    result: Mapped[Optional["Result"]] = relationship(
        "Result", back_populates="image", uselist=False, cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = generate_unique_id(self.session, Image)
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<Image(id={self.id}, title={self.title})>"

    @staticmethod
    def commit():
        Image.session.commit()

    @staticmethod
    def rollback():
        Image.session.rollback()


class Result(ModelBase):
    __tablename__ = "results"
    session: ClassVar[SessionType]

    id: Mapped[str] = mapped_column(String(5), primary_key=True, unique=True)

    video_id: Mapped[Optional[str]] = mapped_column(ForeignKey("videos.id"), unique=True)
    image_id: Mapped[Optional[str]] = mapped_column(ForeignKey("images.id"), unique=True)

    analysis_model: Mapped[str] = mapped_column(String, nullable=False)
    detection_score: Mapped[float] = mapped_column(Float, nullable=False)
    deepfake_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    real_score: Mapped[Optional[float]] = mapped_column(Float)
    fake_score: Mapped[Optional[float]] = mapped_column(Float)

    video: Mapped[Optional["Video"]] = relationship("Video", back_populates="result")
    image: Mapped[Optional["Image"]] = relationship("Image", back_populates="result")

    def __init__(self, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = generate_unique_id(self.session, Result)
        super().__init__(**kwargs)

    def __repr__(self):
        target = "Video" if self.video_id else "Image"
        return (
            f"<Result(id={self.id}, target={target}, "
            f"model={self.analysis_model}, score={self.detection_score}, "
            f"detected={self.deepfake_detected})>"
        )
