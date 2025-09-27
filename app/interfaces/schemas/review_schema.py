from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ReviewCreateSchema(BaseModel):
    user_id: str = Field(..., min_length=1)
    restaurante_id: str = Field(..., min_length=1)
    fecha: datetime = Field(... , min_length=1)
    rating: float = Field(..., ge=1.0, le=5.0)
    comentario: Optional[str] = Field(None, max_length=1000)
    es_favorito: bool = False

class ReviewResponseSchema(BaseModel):
    id: str
    user_id: str
    restaurante_id: str
    rating: float
    comentario: Optional[str] = None
    es_favorito: bool = False
    year: int
    month: int
    day: int
    weekday: int
    fecha: datetime