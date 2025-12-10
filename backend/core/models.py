from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel
from datetime import datetime
import uuid

class Project(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    images: List["Image"] = Relationship(back_populates="project")
    videos: List["Video"] = Relationship(back_populates="project")

class Image(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    filename: str
    project_id: str = Field(foreign_key="project.id")
    status: str = "pending" # pending, processing, completed, error
    width: int = 0
    height: int = 0
    
    project: Optional[Project] = Relationship(back_populates="images")
    annotations: List["Annotation"] = Relationship(back_populates="image")

class Annotation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    image_id: str = Field(foreign_key="image.id")
    label: str
    confidence: float
    bbox_json: str # Stored as JSON string "[x, y, w, h]"
    segmentation_json: str # Stored as RLE or polygon JSON string
    
    image: Optional[Image] = Relationship(back_populates="annotations")

class Video(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    filename: str
    project_id: str = Field(foreign_key="project.id")
    status: str = "pending" # pending, processing, completed, error
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    
    project: Optional[Project] = Relationship(back_populates="videos")
    annotations: List["VideoAnnotation"] = Relationship(back_populates="video")

class VideoAnnotation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    video_id: str = Field(foreign_key="video.id")
    label: str
    frame_number: int
    object_id: int # ID of the object being tracked
    bbox_json: str # [x, y, w, h]
    segmentation_json: str
    
    video: Optional[Video] = Relationship(back_populates="annotations")

class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = None
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(SQLModel):
    email: str
    password: str
    full_name: Optional[str] = None

class UserRead(SQLModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
