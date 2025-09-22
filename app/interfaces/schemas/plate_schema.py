from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PlateCreateSchema(BaseModel):
    nombre_plato: str = Field(... , min_length=2, max_length=200)
    categoria: str = Field(... , min_length=1)
    precio: float = Field(... , gt=0)
    popularidad: Optional[int] = Field(0, ge=0)

class PlateResponseSchema(BaseModel):
    id: Optional[str] = None
    nombre_plato: str
    categoria: str
    precio: float
    popularidad: int = 0
    predicted_rating: Optional[float] = None
    