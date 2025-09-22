from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class Review(BaseModel):
    id: Optional[str] = None
    _id: Optional[str] = None
    id_usuario: str
    id_restaurante: str
    fecha: datetime
    rating: float
    comentario: Optional[str] = None
    es_favorito: bool = False

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }