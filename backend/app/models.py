import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    google_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    picture_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    google_access_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    user_models: Mapped[List["UserModel"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # upload, camera, google_photos
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="documents")
    pages: Mapped[List["Page"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Page.page_number"
    )


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id"), nullable=False, index=True
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    rotation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    crop_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    crop_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    crop_w: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    crop_h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_warped: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )  # 0/1 flag: whether perspective warp has been applied
    processing_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="idle", server_default="idle"
    )  # idle | processing | done | error
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="pages")
    ocr_results: Mapped[List["OcrResult"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    visual_elements: Mapped[List["VisualElement"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )


class VisualElement(Base):
    __tablename__ = "visual_elements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(Integer, ForeignKey("pages.id"), nullable=False, index=True)
    element_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    geometry: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    page: Mapped["Page"] = relationship(back_populates="visual_elements")


class OcrResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(Integer, ForeignKey("pages.id"), nullable=False, index=True)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    translation_applied: Mapped[bool] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    language: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    language_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    script: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    language_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ink_layer: Mapped[str] = mapped_column(
        String(50), nullable=False, default="MAIN_INK", server_default="MAIN_INK"
    )
    ink_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    page: Mapped["Page"] = relationship(back_populates="ocr_results")
    corrections: Mapped[List["Correction"]] = relationship(
        back_populates="ocr_result", cascade="all, delete-orphan"
    )


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ocr_result_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ocr_results.id"), nullable=False, index=True
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_text: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_bbox_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    corrected_bbox_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    corrected_bbox_w: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    corrected_bbox_h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    ocr_result: Mapped["OcrResult"] = relationship(back_populates="corrections")


class UserModel(Base):
    __tablename__ = "user_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    lora_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    num_corrections_trained: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="user_models")
