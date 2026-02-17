from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class HorarioSchema(BaseModel):
    lunes_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    lunes_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    martes_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    martes_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    miércoles_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    miércoles_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    jueves_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    jueves_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    viernes_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    viernes_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    sábado_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    sábado_cierre_min: Optional[int] = Field(None, ge=0, le=1440)
    domingo_apertura_min: Optional[int] = Field(None, ge=0, le=1440)
    domingo_cierre_min: Optional[int] = Field(None, ge=0, le=1440)

class RestaurantCreateSchema(BaseModel):
    id_restaurante: str = Field(... , min_length=1)
    nombre: str = Field(... , min_length=2, max_length=200)
    latitud: float = Field(... , ge=-90, le=90)
    longitud: float = Field(... , ge=-180, le=180)
    categoria: str = Field(... , min_length=1)
    direccion: str = Field(... , min_length=1)
    telefono: str = Field(... , max_length=20)
    horarios: HorarioSchema

class RestaurantResponseSchema(BaseModel):
    id: Optional[str] = None
    id_restaurante: str
    nombre: str
    latitud: float
    longitud: float
    categoria: str
    horarios: HorarioSchema
    telefono: str
    direccion: str
    predicted_rating: Optional[float] = None
    cluster: Optional[int] = None
    distancia_km: Optional[float] = None