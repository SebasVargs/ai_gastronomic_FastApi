from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserCreateSchema(BaseModel):
    nombre: str = Field(... , min_length=2, max_length=100)
    edad: int = Field(... , ge=1, le=120)
    origen: str = Field(... , min_length=2, max_length=100)
    preferencias: List[str] = Field(... , min_length=1)

class UserResponseSchema(BaseModel):
    id: str
    nombre: str
    edad: int
    origen: str
    preferencias: List[str]
    fecha_registro: Optional[datetime] = None
    activo: bool = True

class UserUpdateSchema(BaseModel):
    nombre: Optional[str] = Field(None, min_length=2, max_length=100)
    edad: Optional[int] = Field(None, ge=1, le=120)
    origen: Optional[str] = Field(None, min_length=2, max_length=100)
    preferencias: Optional[List[str]] = None