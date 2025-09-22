from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime

class HorarioRestaurante(BaseModel):
    lunes_apertura_min: Optional[int] = None
    lunes_cierre_min: Optional[int] = None
    martes_apertura_min: Optional[int] = None
    martes_cierre_min: Optional[int] = None
    miércoles_apertura_min: Optional[int] = None
    miércoles_cierre_min: Optional[int] = None
    jueves_apertura_min: Optional[int] = None
    jueves_cierre_min: Optional[int] = None
    viernes_apertura_min: Optional[int] = None
    viernes_cierre_min: Optional[int] = None
    sábado_apertura_min: Optional[int] = None
    sábado_cierre_min: Optional[int] = None
    domingo_apertura_min: Optional[int] = None
    domingo_cierre_min: Optional[int] = None

class Restaurant(BaseModel):
    id: Optional[str] = None
    _id: Optional[str] = None
    nombre: str
    latitud: float
    longitud: float
    categoria: str
    horarios: HorarioRestaurante
    telefono: int
    direccion: str

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }