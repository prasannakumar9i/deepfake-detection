from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr


# Pydantic Schemas

class Ping(BaseModel):
    status: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr

    model_config = {
        "from_attributes": True
    }

class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str
    password: str


class StatusMsg(BaseModel):
    status: str

class ResultSchema(BaseModel):
    analysis_model: str
    detection_score: float
    deepfake_detected: bool
    confidence: float

    model_config = {
        "from_attributes": True
    }
    
class DeepfakeResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    duration: Optional[float]
    file_path: Optional[str]
    uploadedDate: Optional[datetime]
    video_filename: Optional[str]
    result: Optional[ResultSchema]

    model_config = {
        "from_attributes": True
    }