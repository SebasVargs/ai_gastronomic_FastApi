from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class Plate(BaseModel):
    id: Optional[str] = None
    _id: Optional[str] = None
    nombre_plato: str
    categoria: str
    precio: float
    popularidad: Optional[float] = 0.0
    rating: Optional[float] = None
    categoria_plato: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }