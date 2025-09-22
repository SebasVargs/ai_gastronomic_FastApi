from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class User(BaseModel):
    id: Optional[str] = None
    _id: Optional[str] = None
    nombre: str
    edad: int
    origen: str
    preferencias: List[str]
    fecha_registro: Optional[datetime] = None
    activo: bool = True

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }